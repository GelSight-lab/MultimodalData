"""The layout gate must compare the VIDEO to the PARQUET, and must fail.

Two holes let defective folders through the publish gate this session:

* The gate checked that `videos/<date>/<ep>/*.mp4` **exist**; it never decoded
  one. A frame-count mismatch between a video and the parquet beside it — the
  defect the cut path already guards (`segment.py`) — was invisible on the
  whole-episode path.
* Missing LeRobot index columns were a `warning`, so 84 parquets shipped
  without `task`/`task_index`/`episode`/`episode_index`/`frame_index` and the
  gate said pass. A warning nobody fails on is a comment.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from twm.dataset_layout import (CAM_CALIB, DEPTH_FILES,  # noqa: E402
                                FORCE_COLUMNS, GEL_CALIB, VIDEO_FILES,
                                check_layout)

DATE = "2026-09-11"
EP = "episode_000"


def _video(path: Path, n: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = np.zeros((n, 16, 16, 3), np.uint8)
    for i in range(n):
        frames[i] = i * 8
    p = subprocess.Popen(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "16x16", "-r", "30",
         "-i", "-", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "0",
         "-pix_fmt", "yuv444p", "-an", str(path)], stdin=subprocess.PIPE)
    p.communicate(frames.tobytes())
    assert p.returncode == 0


def _folder(root: Path, rows: int, video_frames: int, *, index_cols=True) -> Path:
    meta = root / "meta" / DATE
    meta.mkdir(parents=True, exist_ok=True)
    cols = {"frame_idx": np.arange(rows, dtype=np.int32)}
    for c in FORCE_COLUMNS:
        cols[c] = np.zeros(rows, np.float32)
    if index_cols:
        cols |= {"task": pa.array(["rope"] * rows),
                 "task_index": pa.array(np.full(rows, 2, np.int64)),
                 "episode": pa.array([f"{DATE}/{EP}"] * rows),
                 "episode_index": pa.array(np.zeros(rows, np.int64)),
                 "frame_index": pa.array(np.arange(rows, dtype=np.int64))}
    pq.write_table(pa.table(cols), str(meta / f"{EP}.parquet"))
    for f in VIDEO_FILES:
        _video(root / "videos" / DATE / EP / f, video_frames)
    for f in DEPTH_FILES:
        d = root / "depth" / DATE / EP / f
        d.parent.mkdir(parents=True, exist_ok=True)
        d.write_bytes(b"x")
    (root / "previews" / DATE).mkdir(parents=True, exist_ok=True)
    (root / "previews" / DATE / f"{EP}.mp4").write_bytes(b"x")
    (root / "episodes.jsonl").write_text(json.dumps(
        {"episode": f"{DATE}/{EP}", "date": DATE, "n_frames": rows}) + "\n")
    for n in ("segments.json", "bad_frames.json"):
        (root / n).write_text("{}")
    (root / "splits.json").write_text(json.dumps(
        {"episodes": {f"{DATE}/{EP}": {"test": []}}, "guard_frames": 16,
         "max_train_window": 16}))
    cal = root / "calibration"
    cal.mkdir(parents=True, exist_ok=True)
    for c in CAM_CALIB:
        (cal / f"{c}.json").write_text(json.dumps({"created_at": "2026-09-09"}))
        (cal / f"{c}.npy").write_bytes(b"x")
    for g in GEL_CALIB:
        (cal / g).write_text("{}")
    (cal / "calibration.json").write_text(json.dumps(
        {"created": "2026-09-09", "applies_to_dates": [DATE]}))
    return root


def _problems(rep) -> str:
    return " | ".join(f"{p.check}: {p.message}" for p in rep.problems)


def test_matching_counts_pass(tmp_path):
    rep = check_layout(_folder(tmp_path, 12, 12), DATE, count_frames=True)
    assert rep.ok, _problems(rep)


def test_a_video_shorter_than_the_parquet_fails(tmp_path):
    rep = check_layout(_folder(tmp_path, 12, 9), DATE, count_frames=True)
    assert not rep.ok
    assert any("frames but" in p.message for p in rep.problems), _problems(rep)


def test_a_video_longer_than_the_parquet_fails(tmp_path):
    rep = check_layout(_folder(tmp_path, 9, 12), DATE, count_frames=True)
    assert not rep.ok, _problems(rep)


def test_counting_is_off_by_default(tmp_path):
    """`check_remote` writes empty placeholders — it has no video bytes."""
    rep = check_layout(_folder(tmp_path, 12, 9), DATE)
    assert rep.ok, _problems(rep)


def test_missing_index_columns_warn_but_do_not_block(tmp_path):
    """They are not LeRobot's set and nothing reads them.

    The standard is `episode_index, frame_index, timestamp, index, task_index`
    (checked against lerobot/pusht and lerobot/aloha_sim_insertion_human, both
    v3.0). Ours swaps in two invented strings and omits two real ones. And
    `ReactVideoDataset` locates data by path, never by these columns. Reported
    so the drift is visible; not a gate, because a gate that blocks new data
    over metadata with no consumer costs more than it protects.
    """
    rep = check_layout(_folder(tmp_path, 12, 12, index_cols=False), DATE)
    assert rep.ok, _problems(rep)
    assert any("no task" in w for w in rep.warnings), rep.warnings
