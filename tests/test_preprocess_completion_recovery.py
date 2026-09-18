"""Synthetic recordings exercise publication and interrupted-build recovery."""
import h5py
import numpy as np
import pyarrow.parquet as pq
import pytest

from twm.react_preprocess import config, pipeline, tactile
from twm.react_preprocess.complete import is_complete


@pytest.fixture
def recording(tmp_path, monkeypatch):
    source = tmp_path / "source" / "2026-09-18" / "episode_001.h5"
    source.parent.mkdir(parents=True)
    with h5py.File(source, "w") as f:
        f.create_group("metadata")
        f["timestamps"] = np.arange(4, dtype=np.float64) / 30
        # Two cameras and no wrists are valid; optional views must not be
        # inferred from a fixed seven-stream rig.
        for key in ("realsense/cam0/color", "realsense/cam2/color",
                    "gelsight/left/frames", "gelsight/right/frames"):
            f[key] = np.zeros((4, 2, 3, 3), dtype=np.uint8)
        f["realsense/cam0/depth"] = np.ones((4, 2, 3), dtype=np.uint16)

    class ByteWriter:
        # Isolate ffmpeg only; alignment, tactile metrics, metadata, and
        # publication all use the real pipeline on this small H5.
        def __init__(self, path):
            path.parent.mkdir(parents=True, exist_ok=True)
            self.output = path.open("wb")

        def __enter__(self):
            return self

        def write(self, block):
            self.output.write(block.tobytes())

        def __exit__(self, *args):
            self.output.close()

    root = tmp_path / "release"
    monkeypatch.setattr(config, "STAGE_ROOT", root)
    monkeypatch.setattr(pipeline, "rgb_writer", ByteWriter)
    monkeypatch.setattr(pipeline, "depth_writer", ByteWriter)
    monkeypatch.setattr(tactile, "rgb_writer", ByteWriter)
    return source, root


def _complete(root, source, **kwargs):
    return is_complete(root, "pushT", source.parent.name, source.stem,
                       source_h5=source, **kwargs)


@pytest.mark.parametrize("force_rebuild", [False, True])
def test_sidecar_failure_never_publishes_completion_and_retry_finishes(
        recording, monkeypatch, force_rebuild):
    source, root = recording
    if force_rebuild:
        assert pipeline.build_episode(source, "pushT").status == "OK"
    write_sidecar = pipeline._write_detect_sidecar

    def fail_sidecar(path, *args, **kwargs):
        path.write_bytes(b"interrupted")
        raise OSError("sidecar disk failure")

    monkeypatch.setattr(pipeline, "_write_detect_sidecar", fail_sidecar)
    with pytest.raises(OSError, match="sidecar disk failure"):
        pipeline.build_episode(source, "pushT", force=force_rebuild)
    parquet = root / "pushT/meta" / source.parent.name / f"{source.stem}.parquet"
    assert not parquet.exists(), "a failed sidecar must not leave a completion parquet"

    monkeypatch.setattr(pipeline, "_write_detect_sidecar", write_sidecar)
    assert pipeline.build_episode(source, "pushT").status == "OK"
    assert pq.read_table(parquet).num_rows == 4
    assert _complete(root, source)
    assert pipeline.build_episode(source, "pushT").status == "skipped"


@pytest.mark.parametrize("wrist_slots", [(), ("cam0",), ("cam0", "cam1")])
def test_completion_uses_only_cameras_in_the_source(recording, wrist_slots):
    source, root = recording
    with h5py.File(source, "a") as f:
        for slot in wrist_slots:
            f[f"arducam/{slot}/frames"] = np.zeros((4, 2, 3, 3), np.uint8)
    assert pipeline.build_episode(source, "pushT").status == "OK"
    assert _complete(root, source)
    assert not is_complete(root, "pushT", source.parent.name, source.stem)
    assert pipeline.build_episode(source, "pushT").status == "skipped"


@pytest.mark.parametrize("artifact", ["video", "sidecar", "empty_video"])
def test_missing_required_artifact_is_rebuilt_without_force(recording, artifact):
    source, root = recording
    assert pipeline.build_episode(source, "pushT").status == "OK"
    video_dir, meta_dir = config.stage_dirs("pushT", source.parent.name, source.stem)
    path = (meta_dir / f"{source.stem}._detect.pt" if artifact == "sidecar"
            else video_dir / "view_right.mp4")
    if artifact == "empty_video":
        path.write_bytes(b"")
    else:
        path.unlink()
    assert pipeline.build_episode(source, "pushT").status == "OK"
    assert path.stat().st_size > 0


def test_requesting_depth_retries_a_build_without_depth(recording):
    source, root = recording
    assert pipeline.build_episode(source, "pushT").status == "OK"
    assert pipeline.build_episode(source, "pushT", with_depth=True).status == "OK"
    assert _complete(root, source, with_depth=True)
    assert pipeline.build_episode(source, "pushT", with_depth=True).status == "skipped"


def test_meta_only_completion_does_not_suppress_later_video_build(recording):
    source, root = recording
    assert pipeline.build_episode(source, "pushT", encode_video=False).status == "OK"
    assert pipeline.build_episode(source, "pushT", encode_video=False).status == "skipped"
    assert pipeline.build_episode(source, "pushT").status == "OK"
    assert _complete(root, source)


def test_failed_parquet_write_is_not_published(recording, monkeypatch):
    source, root = recording

    def fail_table(table, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"partial parquet")
        raise OSError("parquet disk failure")

    monkeypatch.setattr(pipeline.meta_mod, "write_table", fail_table)
    with pytest.raises(OSError, match="parquet disk failure"):
        pipeline.build_episode(source, "pushT")
    _, meta_dir = config.stage_dirs("pushT", source.parent.name, source.stem)
    assert not (meta_dir / f"{source.stem}.parquet").exists()
