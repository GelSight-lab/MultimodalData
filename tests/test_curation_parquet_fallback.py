"""Curation must index the episodes whose source recordings are gone.

`build_task` discovers by parquet and REFUSES when a parquet has no
`_detect.pt` sidecar, because rebuilding the indices from the rest would drop
those episodes from `episodes.jsonl`, `segments.json` and `bad_frames.json` —
which is exactly what it did once, rewriting 35 rows to 3.

The refusal is right and stays. What was missing is the other half: 32 of the
43 published motherboard episodes are pre-2026-07 recordings whose H5 was
deleted, so their sidecar can never be rebuilt, and the refusal made the whole
task uncurateable. Every array the sidecar carries is also a column of the
published parquet, so the report can be derived from the release itself.

Derived, and SAID to be derived: `_contact_meta["source"]` records which of the
two it came from, so nothing downstream mistakes a derivation for a build.
"""
import numpy as np
import pytest

from twm.react_preprocess import curation


def _parquet(path, n=40, *, with_object=True):
    import pyarrow as pa
    import pyarrow.parquet as pq
    path.parent.mkdir(parents=True, exist_ok=True)
    pose = lambda o: [[o, 0.1, 0.2, 0.0, 0.0, 0.0, 1.0]] * n
    cols = {
        "frame_idx": np.arange(n, dtype=np.int32),
        "timestamp": 100.0 + np.arange(n) / 30.0,
        "sensor_left_pose": pose(0.3),
        "sensor_right_pose": pose(0.6),
        "tactile_left_intensity": np.linspace(0, 2, n, dtype=np.float32),
        "tactile_right_intensity": np.linspace(2, 0, n, dtype=np.float32),
        "source_h5_frame": (np.arange(n) + 7).astype(np.int32),
    }
    if with_object:
        cols["object_pose"] = pose(0.45)
    pq.write_table(pa.table(cols), str(path))
    return path


def test_a_published_episode_without_its_sidecar_is_still_indexed(tmp_path):
    p = _parquet(tmp_path / "motherboard/meta/2026-05-10/episode_000.parquet")
    report, cm = curation.episode_report(p)
    assert report["n_frames"] == 40
    assert cm["trim_offset"] == 7, "the trim comes from source_h5_frame"
    assert cm["source"] == "parquet"


def test_a_derived_report_says_so(tmp_path):
    """A derivation must not be mistaken for a build artefact."""
    p = _parquet(tmp_path / "motherboard/meta/2026-05-10/episode_000.parquet")
    _, cm = curation.episode_report(p)
    assert cm["source"] == "parquet"


def test_the_detectors_all_run_on_a_derived_report(tmp_path):
    p = _parquet(tmp_path / "motherboard/meta/2026-05-10/episode_000.parquet")
    report, _ = curation.episode_report(p)
    for key in curation.BAD_KEYS:
        assert key in report, f"{key} missing — a detector was skipped"


def test_a_task_whose_sidecars_are_gone_curates_instead_of_refusing(tmp_path):
    root = tmp_path / "motherboard"
    for i in range(3):
        _parquet(root / "meta/2026-05-10" / f"episode_{i:03d}.parquet")
    stats = curation.build_task("motherboard", tmp_path, write=False)
    assert stats["episodes"] == 3


def test_a_parquet_too_damaged_to_derive_from_is_still_refused_by_name(tmp_path):
    """The refusal exists so nothing vanishes quietly. A parquet missing the
    columns a report needs cannot be indexed, and has to say which one."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    root = tmp_path / "motherboard"
    d = root / "meta/2026-05-10"
    d.mkdir(parents=True)
    pq.write_table(pa.table({"frame_idx": np.arange(4, dtype=np.int32)}),
                   str(d / "episode_000.parquet"))
    with pytest.raises(Exception) as exc:
        curation.build_task("motherboard", tmp_path, write=False)
    assert "episode_000" in str(exc.value)
