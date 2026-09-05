"""Checks that must pass before the recorder starts and before each episode.

Each check is a pure function returning a CheckResult so the app can print
every failure at once instead of the first one it trips over.
"""
from __future__ import annotations

import logging
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from twm.recorder.config import RecorderConfig, WriterConfig
from twm.recorder.frames import full_rig_tick_nbytes, synthetic_tick
from twm.recorder.schema import append_ticks, create_episode_file
from twm.recorder.writer import (EpisodeWriter, WriterOverloaded, WriterStats,
                                 queue_capacity_bytes)

log = logging.getLogger("twm.recorder")


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


def failures(results: Sequence[CheckResult]) -> List[CheckResult]:
    return [r for r in results if not r.ok]


def format_report(results: Sequence[CheckResult]) -> str:
    return "\n".join(f"  {'ok  ' if r.ok else 'FAIL'} {r.name}: {r.detail}" for r in results)


def _existing_parent(path: Path) -> Path:
    path = Path(path)
    while not path.exists() and path.parent != path:
        path = path.parent
    return path


def check_disk_free(path, min_free_gb: float,
                    disk_usage: Callable[[str], Any] = shutil.disk_usage) -> CheckResult:
    free_gb = disk_usage(str(_existing_parent(Path(path)))).free / 1e9
    return CheckResult("disk_free", free_gb >= min_free_gb,
                       f"{free_gb:.1f} GB free (need {min_free_gb:g} GB)")


def stale_trackers(poses: Mapping[str, Any], active: Sequence[str],
                   max_age_s: float, now: float) -> List[str]:
    """Watchdog view: bodies whose last sample is older than max_age_s.
    A body with no sample at all is not stale (it may simply never have
    been in the volume)."""
    out = []
    for name in active:
        p = poses.get(name)
        age = now - p[0] if p is not None else 0.0
        if age > max_age_s:
            out.append(f"{name} silent {age:.1f}s")
    return out


def check_optitrack_fresh(poses: Mapping[str, Any], active: Sequence[str],
                          max_age_s: float, now: float) -> CheckResult:
    problems = []
    for name in active:
        p = poses.get(name)
        if p is None:
            problems.append(f"{name}: no data yet")
        elif now - p[0] > max_age_s:
            problems.append(f"{name}: last sample {now - p[0]:.1f}s old")
    return CheckResult("optitrack_fresh", not problems,
                       "; ".join(problems) if problems else
                       f"all of {', '.join(active)} fresh (< {max_age_s:g}s)")


def check_writer_idle(stats: WriterStats) -> CheckResult:
    if stats.fault:
        return CheckResult("writer_idle", False, f"writer fault: {stats.fault}")
    return CheckResult("writer_idle", stats.queue_items == 0,
                       f"{stats.queue_items} ticks still queued")


def check_write_bandwidth(directory, fps: int, seconds: float, margin: float,
                          writer_config: WriterConfig, n_arducam: int = 0,
                          clock: Callable[[], float] = time.monotonic,
                          sleep: Callable[[float], None] = time.sleep) -> CheckResult:
    """Push synthetic full-rig ticks through the real writer into a temp
    file in `directory` for `seconds`; require fps × margin ticks/s.

    The synthetic file uses the legacy schema; the requirement is scaled by
    the extra bytes Arducams add so the number still means "the real rig".
    """
    if seconds <= 0:
        return CheckResult("write_bandwidth", True, "skipped")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    scale = full_rig_tick_nbytes(3, 2, n_arducam) / full_rig_tick_nbytes(3, 2, 0)
    required = fps * margin * scale
    ticks = [synthetic_tick(0.0, seed=k) for k in range(2)]
    with tempfile.TemporaryDirectory(dir=str(directory), prefix=".twm_bandwidth_") as tmp:
        f, path = create_episode_file(tmp, 0, [], [], fps, task_name="bandwidth_test")
        writer = EpisodeWriter(
            capacity_bytes=queue_capacity_bytes(1.0, fps, ticks[0].nbytes()),
            batch_size=writer_config.batch_size, flush_interval_s=float("inf"),
            sink=append_ticks)
        n = 0
        t0 = clock()
        try:
            while clock() - t0 < seconds:
                try:
                    writer.submit(f, ticks[n % 2])
                    n += 1
                except WriterOverloaded:
                    sleep(0.002)
            writer.drain()
            elapsed = clock() - t0
            f.flush()
            file_mb = Path(path).stat().st_size / 1e6
        finally:
            writer.stop()
            f.close()
    rate = n / elapsed if elapsed else 0.0
    return CheckResult(
        "write_bandwidth", rate >= required,
        f"{rate:.1f} ticks/s sustained, need {required:.1f} "
        f"({fps} fps × {margin:g} margin{f' × {scale:.2f} arducam' if n_arducam else ''}); "
        f"{file_mb / elapsed:.0f} MB/s to disk after compression")


def run_startup_preflight(config: RecorderConfig, n_arducam: int = 0,
                          disk_usage: Callable[[str], Any] = shutil.disk_usage
                          ) -> List[CheckResult]:
    return [
        check_disk_free(config.data_dir, config.disk.min_free_gb, disk_usage),
        check_write_bandwidth(config.data_dir, config.fps, config.disk.bandwidth_test_s,
                              config.disk.min_bandwidth_margin, config.writer,
                              n_arducam=n_arducam),
    ]


def run_episode_preflight(config: RecorderConfig, poses: Mapping[str, Any],
                          stats: WriterStats, now: float,
                          disk_usage: Callable[[str], Any] = shutil.disk_usage
                          ) -> List[CheckResult]:
    return [
        check_disk_free(config.data_dir, config.disk.min_free_gb, disk_usage),
        check_optitrack_fresh(poses, config.active_sensors,
                              config.ot_preflight_max_age_s, now),
        check_writer_idle(stats),
    ]
