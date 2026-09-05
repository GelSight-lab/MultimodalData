#!/usr/bin/env python3
"""TWM data collection — compatibility facade.

The recorder lives in `twm.recorder` (see its package docstring). This
module keeps the names older scripts import and forwards `main()`.

    python -m twm.data_collection --task <task_name>
    python -m twm.recorder --help
"""
from __future__ import annotations

import os
import time
from typing import Optional

from twm.recorder import config as _config
from twm.recorder.episode import next_episode_number, write_log_row  # noqa: F401
from twm.recorder.frames import full_rig_tick_nbytes
from twm.recorder.schema import (append_optitrack, append_ticks,  # noqa: F401
                                 create_episode_file, tick_from_legacy)
from twm.recorder.writer import EpisodeWriter, WriterOverloaded  # noqa: F401
from twm.viz import (CAM_CALIB_NAME, TRACKER_COLORS,  # noqa: F401
                     build_preview_panel as make_preview,
                     draw_projection_overlay, load_calibrations,
                     make_optitrack_panel)

REALSENSE_SERIALS = list(_config.REALSENSE_SERIALS)
GELSIGHT_SERIALS = dict(_config.GELSIGHT_SERIALS)
DATA_DIR = str(_config.DATA_DIR)
FPS = _config.FPS

flush_optitrack_to_hdf5 = append_optitrack


def append_camera_frames_batch(f, batch):
    """Legacy batch of (color, depth, gs, timestamp[, gs_ts[, arducam[, arducam_ts]]])."""
    append_ticks(f, [tick_from_legacy(item) for item in batch])


def append_camera_frame(f, color_frames, depth_frames, gs_frames, timestamp):
    append_camera_frames_batch(f, [(color_frames, depth_frames, gs_frames, timestamp)])


class HDF5Writer:
    """Legacy adapter over EpisodeWriter. Never drops: `enqueue` raises
    WriterOverloaded when `maxsize` ticks are already buffered."""

    FLUSH_INTERVAL_S = 10.0

    def __init__(self, maxsize: int = 90, batch_size: int = 10):
        self._writer = EpisodeWriter(
            capacity_bytes=maxsize * full_rig_tick_nbytes(),
            batch_size=batch_size, flush_interval_s=self.FLUSH_INTERVAL_S)

    def enqueue(self, f, color_frames, depth_frames, gs_frames, timestamp,
                gs_timestamps=None, arducam_frames=None, arducam_timestamps=None):
        self._writer.submit(f, tick_from_legacy(
            (color_frames, depth_frames, gs_frames, timestamp, gs_timestamps,
             arducam_frames, arducam_timestamps)))

    def flush(self):
        self._writer.drain()

    def stop(self):
        self._writer.stop()

    def stats(self):
        return self._writer.stats()

    @property
    def queue_size(self) -> int:
        return self._writer.stats().queue_items

    @property
    def dropped_frames(self) -> int:
        return 0


def log_episode(data_dir, task_name, episode_num, h5_path, frame_count, fps,
                has_optitrack=True, notes=""):
    """Append one row to <data_dir>/dataset_log.csv (legacy signature)."""
    write_log_row(os.path.join(data_dir, "dataset_log.csv"), {
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "task": task_name,
        "date": time.strftime("%Y-%m-%d"),
        "episode": f"ep_{episode_num:03d}",
        "frames": frame_count,
        "duration_s": round(frame_count / fps, 2) if fps else 0,
        "size_mb": round(os.path.getsize(h5_path) / 1e6, 1) if os.path.isfile(h5_path) else 0,
        "optitrack": "yes" if has_optitrack else "no",
        "path": os.path.relpath(h5_path, data_dir),
        "notes": notes,
    })


def main(argv: Optional[list] = None) -> int:
    from twm.recorder.app import main as _main
    return _main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
