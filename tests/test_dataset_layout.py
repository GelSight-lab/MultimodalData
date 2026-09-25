"""A prepared session folder must match the published dataset layout.

Every check here is one that failed silently at least once while preparing
the 2026-09-09 session: a folder that uploaded cleanly and was still wrong.
"""
import json

import numpy as np
import pandas as pd
import pytest

from twm.dataset_layout import check_layout

DATE = "2026-09-09"
EPS = ("episode_000", "episode_001")
VIDEOS = ("view_left.mp4", "view_middle.mp4", "view_right.mp4",
          "tactile_left.mp4", "tactile_right.mp4")
DEPTHS = ("depth_left.mkv", "depth_middle.mkv", "depth_right.mkv")
CAM_FILES = ("T_mocap_to_cam_left", "T_mocap_to_cam_middle", "T_mocap_to_cam_right")
GEL_FILES = ("T_gel_to_rigid_left.json", "T_gel_to_rigid_right.json")
FORCE_COLS = ("force_left_normal_n", "force_left_penetration_mm",
              "force_left_target_pose", "force_left_source_frame",
              "force_right_normal_n", "force_right_penetration_mm",
              "force_right_target_pose", "force_right_source_frame")


def make_folder(tmp_path, rows=100, **skip):
    root = tmp_path / "validation"
    (root / "meta" / DATE).mkdir(parents=True)
    for ep in EPS:
        cols = {"frame_idx": np.arange(rows), "timestamp": np.arange(rows, dtype=float)}
        if not skip.get("no_force"):
            for c in FORCE_COLS:
                cols[c] = np.zeros(rows)
        pd.DataFrame(cols).to_parquet(root / "meta" / DATE / f"{ep}.parquet")
        vdir = root / "videos" / DATE / ep; vdir.mkdir(parents=True)
        for v in VIDEOS:
            if skip.get("missing_video") == (ep, v):
                continue
            (vdir / v).write_bytes(b"x")
        ddir = root / "depth" / DATE / ep; ddir.mkdir(parents=True)
        for d in DEPTHS:
            (ddir / d).write_bytes(b"x")
        if not skip.get("no_preview"):
            pdir = root / "previews" / DATE; pdir.mkdir(parents=True, exist_ok=True)
            (pdir / f"{ep}.mp4").write_bytes(b"x")

    cal = root / "calibration"; cal.mkdir()
    created = skip.get("calib_created", "2026-09-09T07:41:39")
    for c in CAM_FILES:
        (cal / f"{c}.json").write_text(json.dumps(
            {"created_at": created, "camera_serial": "X", "rmse_px": 0.5}))
        (cal / f"{c}.npy").write_bytes(b"x")
    for g in GEL_FILES:
        (cal / g).write_text(json.dumps({"gel_center_in_rigid_mm": [0, 0, 0]}))
    if not skip.get("no_calibration_json"):
        (cal / "calibration.json").write_text(json.dumps({
            "task": "validation", "calibration_id": "sept-09",
            "created": skip.get("calib_json_created", "2026-09-09"),
            "applies_to_dates": skip.get("applies_to", [DATE])}))

    eps = list(EPS)[:-1] if skip.get("episodes_short") else list(EPS)
    (root / "episodes.jsonl").write_text("\n".join(json.dumps(
        {"episode": f"{DATE}/{e}", "date": DATE,
         "n_frames": rows + (7 if skip.get("wrong_n_frames") else 0)}) for e in eps))
    (root / "splits.json").write_text(json.dumps(
        {"episodes": {f"{DATE}/{e}": {} for e in eps}}))
    for name in ("segments.json", "bad_frames.json"):
        (root / name).write_text(json.dumps({"episodes": {}}))
    return root


def problems(root):
    return [p.message for p in check_layout(root, DATE).problems]


def test_a_complete_folder_passes(tmp_path):
    r = check_layout(make_folder(tmp_path), DATE)
    assert r.ok, r.problems
    assert r.episodes == list(EPS)


def test_a_missing_video_is_caught(tmp_path):
    root = make_folder(tmp_path, missing_video=("episode_001", "view_middle.mp4"))
    assert any("view_middle.mp4" in m for m in problems(root))


def test_missing_previews_are_caught(tmp_path):
    """The first upload of this session shipped no previews at all."""
    assert any("preview" in m.lower() for m in problems(make_folder(tmp_path, no_preview=True)))


def test_parquet_without_force_columns_is_caught(tmp_path):
    assert any("force_left_normal_n" in m for m in problems(make_folder(tmp_path, no_force=True)))


def test_an_episode_absent_from_episodes_jsonl_is_caught(tmp_path):
    """curate rebuilt episodes.jsonl from _detect.pt sidecars and dropped 29
    of 32 rows; nothing downstream noticed until the force export refused."""
    assert any("episodes.jsonl" in m and "episode_001" in m
               for m in problems(make_folder(tmp_path, episodes_short=True)))


def test_a_row_count_that_disagrees_with_the_parquet_is_caught(tmp_path):
    assert any("n_frames" in m for m in problems(make_folder(tmp_path, wrong_n_frames=True)))


def test_a_missing_calibration_json_is_caught(tmp_path):
    assert any("calibration.json" in m
               for m in problems(make_folder(tmp_path, no_calibration_json=True)))


def test_a_calibration_json_that_names_another_epoch_is_caught(tmp_path):
    """Shipping the June files under a September label, or the reverse, is
    the failure this whole session turned on."""
    root = make_folder(tmp_path, calib_json_created="2026-06-26")
    assert any("2026-06-26" in m and "2026-09-09" in m for m in problems(root))


def test_a_calibration_that_does_not_claim_this_session_is_caught(tmp_path):
    root = make_folder(tmp_path, applies_to=["2026-05-10"])
    assert any("applies_to_dates" in m for m in problems(root))


def test_the_report_renders_as_text(tmp_path):
    text = check_layout(make_folder(tmp_path, no_preview=True), DATE).table()
    assert "preview" in text.lower() and "FAILED" in text


def test_a_corrupt_calibration_json_is_reported_as_corrupt_not_missing(tmp_path):
    """A checker that swallows a parse error tells you to add a file that is
    already there."""
    root = make_folder(tmp_path)
    (root / "calibration" / "calibration.json").write_text("{not json")
    msgs = problems(root)
    assert any("unparseable" in m for m in msgs)
    assert not any("missing calibration/calibration.json" in m for m in msgs)


def test_an_episode_missing_from_splits_is_a_failure_not_a_warning(tmp_path):
    """A loader treats an unlisted episode as training data and reports
    nothing, so this cannot be advisory."""
    root = make_folder(tmp_path)
    (root / "splits.json").write_text(json.dumps(
        {"episodes": {f"{DATE}/{EPS[0]}": {}}}))
    msgs = problems(root)
    assert any(EPS[1] in m and "splits.json" in m for m in msgs)


def test_one_wrist_video_without_the_other_is_a_failure(tmp_path):
    """Half a pair means the build broke, not that the rig lacked cameras."""
    root = make_folder(tmp_path)
    (root / "videos" / DATE / EPS[0] / "wrist_left.mp4").write_bytes(b"x")
    assert any("wrist_right.mp4" in m for m in problems(root))


def test_no_wrist_videos_at_all_is_only_a_warning(tmp_path):
    r = check_layout(make_folder(tmp_path), DATE)
    assert r.ok
    assert any("wrist" in w for w in r.warnings)
