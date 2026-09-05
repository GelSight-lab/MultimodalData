"""One status line for the preview and the log, derived from WriterStats."""
from __future__ import annotations

from typing import Optional, Tuple

from twm.recorder.writer import WriterStats


def health_level(stats: WriterStats, warn_fraction: float,
                 min_free_gb: Optional[float]) -> str:
    if stats.fault:
        return "fail"
    if (min_free_gb is not None and stats.disk_free_gb is not None
            and stats.disk_free_gb < min_free_gb):
        return "fail"
    if stats.fraction > warn_fraction or stats.overloaded_since is not None:
        return "warn"
    return "ok"


def minutes_remaining(stats: WriterStats) -> Optional[float]:
    if stats.disk_free_gb is None or stats.file_mb_s <= 0:
        return None
    return stats.disk_free_gb * 1e3 / stats.file_mb_s / 60.0


def health_line(stats: WriterStats, warn_fraction: float,
                min_free_gb: Optional[float]) -> Tuple[str, str]:
    level = health_level(stats, warn_fraction, min_free_gb)
    parts = [f"writer {stats.fraction:.0%}", f"{stats.mean_mb_s:.0f} MB/s"]
    if stats.disk_free_gb is not None:
        mins = minutes_remaining(stats)
        parts.append(f"disk {stats.disk_free_gb:.0f} GB"
                     + (f" (~{mins:.0f} min)" if mins is not None else ""))
    if stats.fault:
        parts.append(f"FAULT: {stats.fault}")
    else:
        parts.append({"ok": "OK", "warn": "WARN", "fail": "FAIL"}[level])
    return " | ".join(parts), level
