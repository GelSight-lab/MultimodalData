"""Recorder configuration: one frozen dataclass tree, built from the CLI."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

REALSENSE_SERIALS: Tuple[str, ...] = (
    "143322063538",
    "104122062574",
    "217222066989",
)
GELSIGHT_SERIALS: Dict[str, str] = {
    # The units on the rig as of 2026-09-06 (confirmed by the operator).
    # 28YGZL6K / 2BGLKZNT were the 2026-08-07 replacements; not attached now.
    "left": "2DUPB53G",
    "right": "2BKRDTAD",
}
DATA_DIR = Path("/media/yxma/Disk1/twm/data")
FPS = 30
OT_TRACKERS: Tuple[str, ...] = ("motherboard", "sensor_left", "sensor_right")
ACTIVE_SENSOR_CHOICES: Dict[str, Tuple[str, ...]] = {
    "both": ("sensor_left", "sensor_right"),
    "left": ("sensor_left",),
    "right": ("sensor_right",),
}


@dataclass(frozen=True)
class WriterConfig:
    """Bounds and thresholds for EpisodeWriter (see writer.py)."""
    queue_seconds: float = 3.0         # capacity, in seconds of ticks
    batch_size: int = 10               # ticks per resize+write
    flush_interval_s: float = 10.0     # H5Fflush cadence (crash safety)
    warn_fraction: float = 0.25        # preview turns yellow above this
    overload_fraction: float = 0.5     # fail-fast if above this for ...
    overload_sustained_s: float = 3.0  # ... this long
    max_tick_gap_s: float = 0.5        # fail-fast if two ticks are further apart


@dataclass(frozen=True)
class DiskConfig:
    min_free_gb: float = 50.0          # refuse to start / auto-end below this
    bandwidth_test_s: float = 2.0      # startup self-test duration; 0 disables
    min_bandwidth_margin: float = 1.5  # writer must sustain fps × margin


@dataclass(frozen=True)
class RecorderConfig:
    task: str
    data_dir: Path = DATA_DIR
    fps: int = FPS
    realsense_serials: Tuple[str, ...] = REALSENSE_SERIALS
    gelsight_serials: Dict[str, str] = field(
        default_factory=lambda: dict(GELSIGHT_SERIALS))
    active_sensors: Tuple[str, ...] = ACTIVE_SENSOR_CHOICES["both"]
    use_arducam: bool = True
    arducam_config_path: Optional[Path] = None
    show_projection: bool = True
    startup_timeout_s: float = 15.0
    settle_s: float = 1.0
    warmup_drop_frames: int = 10
    ot_preflight_max_age_s: float = 2.0
    ot_watchdog_timeout_s: float = 10.0
    preview_fps: int = 15
    writer: WriterConfig = WriterConfig()
    disk: DiskConfig = DiskConfig()

    @property
    def tick_dt(self) -> float:
        return 1.0 / self.fps


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="TWM multimodal data collection")
    p.add_argument("--task", required=True,
                   help="Task name; top-level folder under the data dir.")
    p.add_argument("--active_sensors", default="both",
                   choices=sorted(ACTIVE_SENSOR_CHOICES),
                   help="Which GelSight rigid bodies the OptiTrack watchdog "
                        "monitors.")
    p.add_argument("--no_projection", action="store_true",
                   help="Disable the GelSight→camera projection overlay.")
    p.add_argument("--no_arducam", action="store_true",
                   help="Run without the two sensor-mounted Arducams.")
    p.add_argument("--arducam_config", default=None,
                   help="Arducam JSON config (default twm/config/arducam.json).")
    p.add_argument("--data_dir", default=None,
                   help=f"Episode root (default {DATA_DIR}).")
    p.add_argument("--queue_seconds", type=float, default=None,
                   help="Writer queue capacity in seconds of ticks.")
    p.add_argument("--min_free_gb", type=float, default=None,
                   help="Refuse to record below this much free disk.")
    p.add_argument("--no_bandwidth_test", action="store_true",
                   help="Skip the startup write-throughput self-test.")
    return p


def parse_args(argv: Optional[Sequence[str]] = None) -> RecorderConfig:
    a = build_parser().parse_args(argv)
    writer = WriterConfig()
    if a.queue_seconds is not None:
        writer = WriterConfig(**{**writer.__dict__, "queue_seconds": a.queue_seconds})
    disk = DiskConfig()
    disk_kw = dict(disk.__dict__)
    if a.min_free_gb is not None:
        disk_kw["min_free_gb"] = a.min_free_gb
    if a.no_bandwidth_test:
        disk_kw["bandwidth_test_s"] = 0.0
    disk = DiskConfig(**disk_kw)
    return RecorderConfig(
        task=a.task,
        data_dir=Path(a.data_dir) if a.data_dir else DATA_DIR,
        active_sensors=ACTIVE_SENSOR_CHOICES[a.active_sensors],
        use_arducam=not a.no_arducam,
        arducam_config_path=Path(a.arducam_config) if a.arducam_config else None,
        show_projection=not a.no_projection,
        writer=writer,
        disk=disk,
    )
