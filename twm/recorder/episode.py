"""Where episodes live on disk, how they are numbered, and how they are logged."""
from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

from twm.recorder.schema import create_episode_file

VALID_ENDINGS = ("operator", "quit", "watchdog")
LOG_COLUMNS = ("saved_at", "task", "date", "episode", "frames", "duration_s",
               "size_mb", "optitrack", "path", "notes")


@dataclass(frozen=True)
class EpisodeSummary:
    episode_num: int
    path: Path
    task: str
    frame_count: int
    fps: int
    valid: bool
    ended_by: str           # operator | quit | watchdog | overload | writer_fault
                            # | disk_low | capture_stall | sensor_error
    reason: str
    max_tick_gap_s: float
    gap_count: int
    queue_peak_fraction: float
    writer_mean_mb_s: float
    has_optitrack: bool
    sensor_restarts: Dict[str, int] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps else 0.0

    def attrs(self) -> Dict[str, Any]:
        """metadata attributes written at finalize (schema.write_episode_attrs)."""
        return {
            "frame_count": self.frame_count,
            "duration_s": self.duration_s,
            "valid": bool(self.valid),
            "invalid_reason": "" if self.valid else f"{self.ended_by}: {self.reason}",
            "ended_by": self.ended_by,
            "max_tick_gap_s": self.max_tick_gap_s,
            "gap_count": self.gap_count,
            "queue_peak_fraction": self.queue_peak_fraction,
            "writer_mean_mb_s": self.writer_mean_mb_s,
            "ended_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "sensor_restarts": json.dumps(dict(self.sensor_restarts), sort_keys=True),
        }

    def notes(self) -> str:
        if not self.valid:
            return f"INVALID: {self.ended_by}: {self.reason}"
        if self.ended_by != "operator":
            return f"auto-ended: {self.ended_by}: {self.reason}".rstrip(": ")
        return ""

    def describe(self) -> str:
        text = (f"Episode {self.episode_num:03d} saved — {self.frame_count} frames, "
                f"{self.duration_s:.1f}s")
        if not self.valid:
            text += f" — INVALID ({self.ended_by}: {self.reason})"
        elif self.ended_by != "operator":
            text += f" ({self.notes()})"
        restarted = {n: c for n, c in self.sensor_restarts.items() if c}
        if restarted:
            text += " — sensor restarts: " + ", ".join(f"{n}={c}" for n, c in sorted(restarted.items()))
        return text


def next_episode_number(date_dir) -> int:
    date_dir = Path(date_dir)
    if not date_dir.is_dir():
        return 0
    nums = []
    for p in date_dir.iterdir():
        name = p.name
        if name.startswith("episode_") and name.endswith(".h5"):
            try:
                nums.append(int(name[len("episode_"):-len(".h5")]))
            except ValueError:
                continue
    return max(nums) + 1 if nums else 0


def write_log_row(log_path, row: Dict[str, Any]) -> None:
    """Append one CSV row, writing the header if the file is new."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    new = not log_path.exists()
    with open(log_path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(LOG_COLUMNS))
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in LOG_COLUMNS})


class EpisodeStore:
    def __init__(self, data_dir, task: str, date: Optional[str] = None):
        self.data_dir = Path(data_dir)
        self.task = task
        self.date = date or time.strftime("%Y-%m-%d")

    @property
    def date_dir(self) -> Path:
        return self.data_dir / self.task / self.date

    @property
    def log_path(self) -> Path:
        return self.data_dir / "dataset_log.csv"

    def next_episode_number(self) -> int:
        return next_episode_number(self.date_dir)

    def create(self, episode_num: int, realsense_serials: Sequence[str],
               gelsight_serials: Sequence[str], fps: int,
               arducam_config=None, n_realsense=None,
               depth_aligned=True, arducam_encoding="bgr8") -> Tuple[Any, Path]:
        f, path = create_episode_file(str(self.date_dir), episode_num,
                                      list(realsense_serials), list(gelsight_serials),
                                      fps, task_name=self.task,
                                      arducam_config=arducam_config,
                                      n_realsense=n_realsense,
                                      depth_aligned=depth_aligned,
                                      arducam_encoding=arducam_encoding)
        return f, Path(path)

    def log(self, s: EpisodeSummary) -> Path:
        size_mb = round(os.path.getsize(s.path) / 1e6, 1) if Path(s.path).is_file() else 0
        write_log_row(self.log_path, {
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "task": s.task,
            "date": self.date,
            "episode": f"ep_{s.episode_num:03d}",
            "frames": s.frame_count,
            "duration_s": round(s.duration_s, 2),
            "size_mb": size_mb,
            "optitrack": "yes" if s.has_optitrack else "no",
            "path": os.path.relpath(s.path, self.data_dir),
            "notes": s.notes(),
        })
        return self.log_path
