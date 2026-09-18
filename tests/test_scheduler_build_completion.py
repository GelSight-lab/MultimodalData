"""Scheduling uses the builder's artifact contract, including optional cameras."""
import os

import h5py
import pytest

from twm import pipeline_stages as PS


@pytest.fixture
def episode(tmp_path, monkeypatch):
    data, release = tmp_path / "data", tmp_path / "release"
    monkeypatch.setattr(PS, "DATA_ROOT", data)
    monkeypatch.setattr(PS, "RELEASE", release)
    source = data / "rope/2026-09-18/episode_001.h5"
    source.parent.mkdir(parents=True)
    with h5py.File(source, "w") as f:
        # A valid optional-camera rig, with source depth that the scheduler
        # does not request. Completion only needs the declared RGB streams.
        for key in ("realsense/cam0/color", "realsense/cam0/depth",
                    "gelsight/left/frames", "gelsight/right/frames"):
            f[key] = [0]
    meta = release / "rope/meta/2026-09-18"
    meta.mkdir(parents=True)
    videos = release / "rope/videos/2026-09-18/episode_001"
    videos.mkdir(parents=True)
    for name in ("view_right", "tactile_left", "tactile_right"):
        (videos / f"{name}.mp4").write_bytes(b"encoded stream")
    (meta / "episode_001._detect.pt").write_bytes(b"detection sidecar")
    parquet = meta / "episode_001.parquet"
    parquet.write_bytes(b"completed parquet")
    return source, parquet, videos


@pytest.mark.parametrize("incomplete", ["parquet_only", "meta_only", "missing_sidecar",
                                       "stale_parquet", "empty_video"])
def test_scheduler_retries_incomplete_builds(episode, incomplete):
    source, parquet, videos = episode
    sidecar = parquet.with_suffix("._detect.pt")
    if incomplete in ("parquet_only", "meta_only"):
        for video in videos.iterdir():
            video.unlink()
    if incomplete in ("parquet_only", "missing_sidecar"):
        sidecar.unlink()
    if incomplete == "stale_parquet":
        old = parquet.stat().st_mtime - 3600
        os.utime(parquet, (old, old))
    if incomplete == "empty_video":
        (videos / "view_right.mp4").write_bytes(b"")
    coverage = PS.coverage("build", "rope")
    assert (coverage.done, coverage.total) == (0, 1)
    assert coverage.missing == ["2026-09-18/episode_001"]
    commands = PS._build(task="rope")
    assert len(commands) == 1
    assert commands[0][-2:] == ["--episodes", "episode_001"]
    assert not PS.BY_NAME["build"].produced("rope")


def test_scheduler_accepts_source_optional_cameras_without_requesting_depth(episode):
    coverage = PS.coverage("build", "rope")
    assert coverage.complete and coverage.done == 1
    assert PS._build(task="rope") == []
    assert PS.BY_NAME["build"].produced("rope")
    command = PS._build(task="rope", date="2026-09-18")[0]
    assert "--with-depth" not in command and "--meta-only" not in command


def test_unknown_source_does_not_relax_build_prerequisites(episode):
    source, _, _ = episode
    source.unlink()
    assert not PS.BY_NAME["build"].produced("rope")


def test_unreadable_source_is_queued_for_build_diagnosis(episode):
    source, _, _ = episode
    source.write_bytes(b"interrupted HDF5 header")
    assert PS.coverage("build", "rope").missing == ["2026-09-18/episode_001"]
    assert PS._build(task="rope")
