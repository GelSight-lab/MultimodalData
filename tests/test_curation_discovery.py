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


def _tree(tmp_path, dates, sidecars_for=None):
    root = tmp_path / "motherboard"
    for date, eps in dates.items():
        d = root / "meta" / date
        d.mkdir(parents=True)
        for ep in eps:
            (d / f"{ep}.parquet").write_bytes(b"x")
            if sidecars_for is None or (date, ep) in sidecars_for:
                torch.save({"timestamps": torch.zeros(10)}, d / f"{ep}._detect.pt")
    return root


def test_a_parquet_without_its_sidecar_is_refused_by_name(tmp_path):
    root = _tree(tmp_path, {"2026-05-10": ["episode_000", "episode_001"],
                            "2026-09-09": ["episode_000"]},
                 sidecars_for={("2026-09-09", "episode_000")})
    with pytest.raises(FileNotFoundError) as exc:
        curation.build_task("motherboard", root.parent, write=False)
    msg = str(exc.value)
    assert "2026-05-10/episode_000" in msg and "2026-05-10/episode_001" in msg
    assert "2026-09-09" not in msg          # the one that is covered is not named


def test_the_refusal_says_how_many_are_missing(tmp_path):
    root = _tree(tmp_path, {"2026-05-10": [f"episode_{i:03d}" for i in range(12)]},
                 sidecars_for=set())
    with pytest.raises(FileNotFoundError, match="12 of 12"):
        curation.build_task("motherboard", root.parent, write=False)


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
