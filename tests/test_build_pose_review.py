from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import cv2
import pytest
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "twm" / "scripts"))
import build_pose_review as BPR  # noqa: E402


def poses(yaw_deg, x_mm):
    out = np.zeros((len(yaw_deg), 7), float)
    out[:, 0] = np.asarray(x_mm) / 1000.0
    out[:, 3:] = R.from_euler("z", yaw_deg, degrees=True).as_quat()
    return out


def write_parquet(path: Path, left, right, source_start=100):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({
        "source_h5_frame": np.arange(source_start, source_start + len(left)),
        "sensor_left_pose": pa.array(left.tolist()),
        "sensor_right_pose": pa.array(right.tolist()),
    }), path)


def test_scan_preserves_source_frames_and_hand(tmp_path):
    p = tmp_path / "meta" / "2026-09-11" / "episode_004.parquet"
    good = poses([0, 1, 2, 3, 4], [0, 1, 2, 3, 4])
    jump = poses([0, 1, 101, 2, 3], [0, 1, 31, 2, 3])
    write_parquet(p, good, jump, source_start=8548)

    events = BPR.scan_parquet(p)

    assert len(events) == 1
    assert events[0]["side"] == "right"
    assert events[0]["source_start"] == 8550
    assert events[0]["source_end"] == 8550
    assert events[0]["event_id"] == (
        "2026-09-11__episode_004__right__8550-8550__returning_excursion")


def test_scan_tree_loads_per_side_known_gaps(tmp_path):
    root = tmp_path / "release" / "motherboard"
    p = root / "meta" / "2026-05-11" / "episode_013.parquet"
    good = poses([0] * 30, np.arange(30))
    write_parquet(p, good, good, source_start=5000)
    (root / "pose_gaps.json").write_text(json.dumps({
        "2026-05-11/episode_013": {"left": [[5, 20]]}
    }))

    events = BPR.scan_tree(root)

    assert len(events) == 1
    assert events[0]["kind"] == "long_gap"
    assert events[0]["source_start"] == 5005
    assert events[0]["source_end"] == 5024


def test_nearby_unresolved_events_merge_for_one_review_clip():
    base = {
        "date": "d", "episode": "e", "side": "left",
        "repairable": False, "source_start": 100, "source_end": 100,
        "kind": "persistent_branch", "event_id": "a", "evidence": {},
        "chart": {"transition_source_frames": [99, 100],
                  "rotation_deg": [2, 90], "translation_mm": [1, 30]},
    }
    second = {**base, "source_start": 108, "source_end": 109,
              "event_id": "b", "kind": "plausible_motion",
              "chart": {"transition_source_frames": [108, 109],
                        "rotation_deg": [40, 2], "translation_mm": [3, 1]}}

    merged = BPR.merge_review_events([base, second], max_gap=15)

    assert len(merged) == 1
    assert merged[0]["source_start"] == 100
    assert merged[0]["source_end"] == 109
    assert merged[0]["component_ids"] == ["a", "b"]
    assert merged[0]["chart"]["transition_source_frames"] == [99, 100, 108, 109]


def test_report_writes_all_json_and_unresolved_csv(tmp_path):
    repairable = {
        "event_id": "repair", "date": "d", "episode": "e", "side": "left",
        "kind": "returning_excursion", "repairable": True,
        "source_start": 1, "source_end": 1, "confidence": .9, "evidence": {},
    }
    review = {**repairable, "event_id": "review", "repairable": False,
              "kind": "persistent_branch", "source_start": 9, "source_end": 9}

    paths = BPR.write_metadata([repairable, review], tmp_path)

    payload = json.loads(paths["json"].read_text())
    assert [x["event_id"] for x in payload["events"]] == ["repair", "review"]
    with paths["csv"].open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert [x["event_id"] for x in rows] == ["review"]


def _write_video(path: Path, n=10, size=(64, 48)):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                             30.0, size)
    assert writer.isOpened()
    for i in range(n):
        frame = np.full((size[1], size[0], 3), i * 10, np.uint8)
        writer.write(frame)
    writer.release()


def test_render_uses_requested_context_and_tolerates_missing_wrist(tmp_path, monkeypatch):
    root = tmp_path / "release" / "motherboard"
    date, episode = "2026-09-11", "episode_004"
    good = poses([0] * 10, np.arange(10))
    write_parquet(root / "meta" / date / f"{episode}.parquet", good, good,
                  source_start=100)
    video_dir = root / "videos" / date / episode
    for name in ("view_left", "view_middle", "view_right",
                 "tactile_left", "tactile_right"):
        _write_video(video_dir / f"{name}.mp4")
    event = {
        "event_id": "review", "date": date, "episode": episode,
        "side": "left", "kind": "persistent_branch", "confidence": .8,
        "repairable": False, "row_start": 4, "row_end": 4,
        "source_start": 104, "source_end": 104,
    }
    out = tmp_path / "review.mp4"
    overlay_calls = []
    monkeypatch.setattr(BPR, "_load_review_calibration",
                        lambda task, date: (["cam"], "left-gel", "right-gel"))
    monkeypatch.setattr(BPR, "draw_projection_overlay",
                        lambda *a, **k: overlay_calls.append((a, k)))

    result = BPR.render_event_clip(event, root, out, context_frames=2)

    assert result["frames"] == 5
    assert len(overlay_calls) == 5
    cap = cv2.VideoCapture(str(out))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 5
    ok, frame = cap.read()
    cap.release()
    assert ok and frame.shape[1] == 1280


def test_html_links_every_clip_and_exports_persistent_decisions(tmp_path):
    events = [
        {"event_id": "a", "date": "d", "episode": "ep", "side": "left",
         "kind": "persistent_branch", "confidence": .8, "repairable": False,
         "source_start": 10, "source_end": 10, "clip": "clips/a.mp4",
         "evidence": {"jump_rotation_deg": 100},
         "chart": {"transition_source_frames": [9, 10],
                   "rotation_deg": [2, 100], "translation_mm": [1, 30]}},
        {"event_id": "b", "date": "d", "episode": "ep", "side": "right",
         "kind": "long_gap", "confidence": 1.0, "repairable": False,
         "source_start": 20, "source_end": 40, "clip": "clips/b.mp4",
         "evidence": {"duration_frames": 21}, "chart": {}},
    ]

    path = BPR.write_html(events, tmp_path / "index.html")
    html = path.read_text()

    assert "clips/a.mp4" in html and "clips/b.mp4" in html
    assert 'data-decision="keep_raw"' in html
    assert 'data-decision="accept_repair"' in html
    assert 'data-decision="invalidate"' in html
    assert 'data-decision="unsure"' in html
    assert "localStorage" in html
    assert "Export decisions.json" in html
    assert "rotation_deg" in html and "translation_mm" in html


def test_review_selects_all_medium_low_and_deterministic_high_sample():
    events = [
        {"event_id": "high-a", "confidence": "HIGH"},
        {"event_id": "high-b", "confidence": "HIGH"},
        {"event_id": "medium", "confidence": "MEDIUM"},
        {"event_id": "low", "confidence": "LOW"},
    ]

    no_high = BPR.select_review_events(events, high_sample_rate=0.0, seed=4)
    all_events = BPR.select_review_events(events, high_sample_rate=1.0, seed=4)
    first = BPR.select_review_events(events, high_sample_rate=0.5, seed=9)
    second = BPR.select_review_events(events, high_sample_rate=0.5, seed=9)

    assert {event["event_id"] for event in no_high} == {"medium", "low"}
    assert {event["event_id"] for event in all_events} == {
        "high-a", "high-b", "medium", "low"}
    assert first == second


def test_render_candidate_draws_raw_and_repaired_pose_overlays(tmp_path, monkeypatch):
    root = tmp_path / "release" / "motherboard"
    date, episode = "2026-09-11", "episode_004"
    raw = poses([0] * 10, np.arange(10))
    candidate = raw.copy(); candidate[4, 0] += 0.02
    write_parquet(root / "meta" / date / f"{episode}.parquet", raw, raw,
                  source_start=100)
    candidate_path = tmp_path / "candidate.parquet"
    write_parquet(candidate_path, candidate, raw, source_start=100)
    video_dir = root / "videos" / date / episode
    for name in ("view_left", "view_middle", "view_right",
                 "tactile_left", "tactile_right"):
        _write_video(video_dir / f"{name}.mp4")
    event = {
        "event_id": "review", "date": date, "episode": episode,
        "side": "left", "kind": "branch_discontinuity", "confidence": "MEDIUM",
        "row_start": 4, "row_end": 4, "source_start": 104, "source_end": 104,
    }
    overlay_calls = []
    monkeypatch.setattr(BPR, "_load_review_calibration",
                        lambda task, date: (["cam"], "left-gel", "right-gel"))
    monkeypatch.setattr(BPR, "draw_projection_overlay",
                        lambda panel, opt, *a, **k: overlay_calls.append(opt))

    info = BPR.render_event_clip(
        event, root, tmp_path / "candidate.mp4", context_frames=1,
        candidate_parquet=candidate_path)

    assert info["frames"] == 3
    assert len(overlay_calls) == 6
    raw_x = overlay_calls[2]["sensor_left"][1][0]
    candidate_x = overlay_calls[3]["sensor_left"][1][0]
    assert candidate_x - raw_x == pytest.approx(0.02)


def test_candidate_event_loader_adds_motion_and_residual_charts(tmp_path):
    root = tmp_path / "release" / "motherboard"
    candidate_root = tmp_path / "candidate" / "motherboard"
    date, episode = "2026-09-11", "episode_004"
    raw = poses(np.arange(20), np.arange(20))
    candidate = raw.copy(); candidate[8, 0] += 0.02
    write_parquet(root / "meta" / date / f"{episode}.parquet", raw, raw,
                  source_start=100)
    candidate_path = candidate_root / "meta" / date / f"{episode}.parquet"
    write_parquet(candidate_path, candidate, raw, source_start=100)
    table = pq.read_table(candidate_path)
    table = table.append_column(
        "pose_left_repaired", pa.array([i == 8 for i in range(20)]))
    pq.write_table(table, candidate_path)
    event_path = candidate_root / "repair_events" / date / f"{episode}.json"
    event_path.parent.mkdir(parents=True)
    event_path.write_text(json.dumps({
        "manifest_digest": "a" * 64, "task": "motherboard", "date": date,
        "episode": episode, "events": [{
            "event_id": "left:8-8:branch", "side": "left", "start": 8,
            "end": 8, "kind": "branch", "confidence": "MEDIUM",
            "method": "robust_se3_hermite", "replaced_frames": [8],
            "evidence": {},
        }],
    }))

    events = BPR.load_candidate_events(root, candidate_root, chart_context=3)

    assert len(events) == 1
    chart = events[0]["chart"]
    assert set(chart) >= {
        "rotation_raw_deg", "rotation_candidate_deg",
        "translation_raw_mm", "translation_candidate_mm",
        "linear_velocity_candidate_mm_s", "linear_acceleration_candidate_mm_s2",
        "pose_translation_residual_mm", "pose_rotation_residual_deg",
        "anchor_rows", "retained_rows", "replaced_rows",
    }


def test_build_review_renders_only_unresolved_events(monkeypatch, tmp_path):
    repairable = {
        "event_id": "repair", "date": "d", "episode": "e", "side": "left",
        "kind": "returning_excursion", "repairable": True, "confidence": .9,
        "source_start": 1, "source_end": 1, "row_start": 1, "row_end": 1,
        "evidence": {}, "chart": {},
    }
    unresolved = {**repairable, "event_id": "review", "repairable": False,
                  "kind": "persistent_branch", "source_start": 9,
                  "source_end": 9, "row_start": 9, "row_end": 9}
    monkeypatch.setattr(BPR, "scan_tree", lambda root: [repairable, unresolved])
    rendered = []

    def fake_render(event, root, output, context_frames=60):
        rendered.append(event["event_id"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"clip")
        return {"frames": 1}

    monkeypatch.setattr(BPR, "render_event_clip", fake_render)

    result = BPR.build_review(tmp_path / "root", tmp_path / "out", render=True)

    assert rendered == ["review"]
    assert result["review_events"][0]["clip"].startswith("clips/")
    assert (tmp_path / "out" / "index.html").exists()
