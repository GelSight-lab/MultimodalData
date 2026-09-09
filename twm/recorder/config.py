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
# Where each RealSense stands, in REALSENSE_SERIALS order (cam0, cam1, cam2).
# The single source for "which camera is this": logs, the preview labels and
# the calibration file names all derive from it.
REALSENSE_POSITIONS: Tuple[str, ...] = ("right", "left", "middle")


def realsense_position(serial: str) -> str:
    """'right' / 'left' / 'middle' for a rig camera, 'unknown' otherwise."""
    try:
        return REALSENSE_POSITIONS[REALSENSE_SERIALS.index(serial)]
    except ValueError:
        return "unknown"


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
    use_optitrack: bool = True
    use_arducam: bool = True
    arducam_config_path: Optional[Path] = None
    show_projection: bool = True
    align_depth: bool = True   # False: store depth in the depth camera's own frame
    arducam_encoding: str = "mjpeg"   # the camera's own JPEG; "bgr8" decodes at record time
    startup_timeout_s: float = 15.0
    settle_s: float = 1.0
    warmup_drop_frames: int = 10
    ot_preflight_max_age_s: float = 2.0
    ot_watchdog_timeout_s: float = 10.0
    gui_fps: int = 15         # window refresh (render + imshow); imshow alone costs ~10 ms a frame
    preview_fps: int = 15     # base panel rebuild rate (5 ms each; the CPU is the rig's scarcest
                              # resource); the overlay is redrawn at the GUI's 30 Hz regardless
    writer: WriterConfig = WriterConfig()
    disk: DiskConfig = DiskConfig()

    def __post_init__(self) -> None:
        # A caller may construct RecorderConfig directly (tests, or future
        # callers that don't go through config_from_namespace) with
        # use_optitrack=False but a stale/non-empty active_sensors; force
        # the invariant here too so the OptiTrack watchdog and preflight
        # never see active bodies to check when OptiTrack isn't running.
        if not self.use_optitrack and self.active_sensors:
            object.__setattr__(self, "active_sensors", ())

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
    p.add_argument("--arducam_raw", action="store_true",
                   help="Decode the wrist cameras at record time and store raw BGR "
                        "(the pre-2026-09 format). The default stores the camera's "
                        "own MJPEG, which is ~28 MB/s and 0.3 of a core cheaper.")
    p.add_argument("--raw_depth", action="store_true",
                   help="Record depth in the depth camera's own frame instead of "
                        "aligning it to color on the fly (saves ~0.25 CPU core per "
                        "camera). Align it later with `python -m twm.realsense_align "
                        "apply`.")
    p.add_argument("--realsense_serials", default=None,
                   help="Comma-separated RealSense serials to record (default: the rig's three).")
    p.add_argument("--no_optitrack", action="store_true",
                   help="Record without OptiTrack (no ROS needed); pose datasets stay empty.")
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
    p.add_argument("--bandwidth_margin", type=float, default=1.5,
                   help="Writer must sustain fps × this margin (default 1.5; "
                        "lower it for disks that can't clear the default headroom).")
    return p


def config_from_namespace(a: argparse.Namespace) -> RecorderConfig:
    writer = WriterConfig()
    if a.queue_seconds is not None:
        writer = WriterConfig(**{**writer.__dict__, "queue_seconds": a.queue_seconds})
    disk = DiskConfig()
    disk_kw = dict(disk.__dict__)
    if a.min_free_gb is not None:
        disk_kw["min_free_gb"] = a.min_free_gb
    if a.no_bandwidth_test:
        disk_kw["bandwidth_test_s"] = 0.0
    disk_kw["min_bandwidth_margin"] = a.bandwidth_margin
    disk = DiskConfig(**disk_kw)
    return RecorderConfig(
        task=a.task,
        data_dir=Path(a.data_dir) if a.data_dir else DATA_DIR,
        realsense_serials=tuple(s.strip() for s in a.realsense_serials.split(",")
                                if s.strip()) if a.realsense_serials else REALSENSE_SERIALS,
        active_sensors=() if a.no_optitrack else ACTIVE_SENSOR_CHOICES[a.active_sensors],
        use_optitrack=not a.no_optitrack,
        use_arducam=not a.no_arducam,
        arducam_config_path=Path(a.arducam_config) if a.arducam_config else None,
        show_projection=not a.no_projection,
        align_depth=not a.raw_depth,
        arducam_encoding="bgr8" if a.arducam_raw else "mjpeg",
        writer=writer,
        disk=disk,
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> RecorderConfig:
    return config_from_namespace(build_parser().parse_args(argv))
