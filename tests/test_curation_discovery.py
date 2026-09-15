"""Rebuilding a task's indices must not silently shrink them.

`build_task` discovered episodes through `meta/**/*._detect.pt` sidecars,
which only freshly built episodes have. Run against the real motherboard tree
(35 parquet, 3 sidecars) it rewrote `episodes.jsonl` from 35 rows to 3 and
truncated `segments.json` and `bad_frames.json` the same way. Nothing failed;
the loss surfaced two steps later when the force export refused an episode it
could no longer find.
"""
import json

import pytest
import torch

from twm.react_preprocess import curation


def _real_parquet(path, n=20):
    """A real parquet: curation now DERIVES a report from it when the sidecar
    is gone, so a one-byte stand-in no longer stands in."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    pose = [[0.3, 0.1, 0.2, 0.0, 0.0, 0.0, 1.0]] * n
    pq.write_table(pa.table({
        "frame_idx": np.arange(n, dtype=np.int32),
        "timestamp": 100.0 + np.arange(n) / 30.0,
        "sensor_left_pose": pose,
        "sensor_right_pose": pose,
        "tactile_left_intensity": np.linspace(0, 2, n, dtype=np.float32),
        "tactile_right_intensity": np.linspace(2, 0, n, dtype=np.float32),
        "source_h5_frame": np.arange(n, dtype=np.int32),
    }), str(path))


def _tree(tmp_path, dates, sidecars_for=None):
    root = tmp_path / "motherboard"
    for date, eps in dates.items():
        d = root / "meta" / date
        d.mkdir(parents=True)
        for ep in eps:
            _real_parquet(d / f"{ep}.parquet")
            if sidecars_for is None or (date, ep) in sidecars_for:
                torch.save({"timestamps": torch.zeros(20),
                            "sensor_left_pose": torch.zeros(20, 7),
                            "sensor_right_pose": torch.zeros(20, 7),
                            "tactile_left_intensity": torch.zeros(20),
                            "tactile_right_intensity": torch.zeros(20),
                            "_contact_meta": {"trim_offset": 0,
                                              "active_sensors": ["left", "right"]}},
                           d / f"{ep}._detect.pt")
    return root


# The refusal these two used to assert has been replaced, not dropped. It
# existed so that an episode without a sidecar could never vanish from the
# indices; but the pre-2026-07 episodes' source H5 is deleted, so their sidecar
# can NEVER be rebuilt and the refusal made the whole task uncurateable
# (32 of 43 motherboard episodes, blocking every publish). `episode_report`
# now derives the same arrays from the published parquet. What must still hold
# is the invariant the refusal was protecting: nothing is silently dropped.

def test_a_parquet_without_its_sidecar_is_indexed_not_dropped(tmp_path):
    root = _tree(tmp_path, {"2026-05-10": ["episode_000", "episode_001"],
                            "2026-09-09": ["episode_000"]},
                 sidecars_for={("2026-09-09", "episode_000")})
    stats = curation.build_task("motherboard", root.parent, write=False)
    assert stats["episodes"] == 3, "an episode without a sidecar went missing"


def test_a_whole_task_of_sidecar_less_episodes_still_indexes_every_one(tmp_path):
    root = _tree(tmp_path, {"2026-05-10": [f"episode_{i:03d}" for i in range(12)]},
                 sidecars_for=set())
    stats = curation.build_task("motherboard", root.parent, write=False)
    assert stats["episodes"] == 12


def test_an_existing_up_axis_survives_a_rebuild(tmp_path, monkeypatch):
    """The published rows carry up_axis 'z'; a rebuild that drops it makes
    calib_epoch read every Z-up world offset as Y-up."""
    root = _tree(tmp_path, {"2026-05-10": ["episode_000"]})
    (root / "episodes.jsonl").write_text(json.dumps(
        {"episode": "2026-05-10/episode_000", "date": "2026-05-10",
         "n_frames": 10, "up_axis": "z"}) + "\n")
    monkeypatch.setattr(curation, "episode_report", lambda det, video_dir=None: (
        {"n_frames": 10, "duration_s": 0.33, "total_bad_frames": 0,
         "intensity_spikes": [], "ot_freezes": [], "pose_teleports": [],
         "cam_corruption": [], "tactile_corruption": []}, {}))
    monkeypatch.setattr(curation, "_bad_intervals", lambda report: [])
    stats = curation.build_task("motherboard", root.parent, write=True)
    assert stats["episodes"] == 1
    row = json.loads((root / "episodes.jsonl").read_text().splitlines()[0])
    assert row["up_axis"] == "z"
