"""One status line for the preview and the log, derived from WriterStats."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

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


STALE_SENSOR_S = 1.0


def health_line(stats: WriterStats, warn_fraction: float,
                min_free_gb: Optional[float],
                sensors: Optional[Dict[str, Dict[str, Any]]] = None) -> Tuple[str, str]:
    """One status string and its level. `sensors` is the rig's sensor_status();
    a stream whose capture clock is older than STALE_SENSOR_S is named and
    bumps the level to warn (its restart is handled by the rig's supervisor)."""
    level = health_level(stats, warn_fraction, min_free_gb)
    parts = [f"writer {stats.fraction:.0%}", f"{stats.mean_mb_s:.0f} MB/s"]
    stale = []
    for name, st in (sensors or {}).items():
        age = st.get("stale_s")
        if age is not None and age > STALE_SENSOR_S:
            extra = ", ".join(x for x in ("restarting" if st.get("restarting") else "",
                                          f"{st['restarts']} restarts" if st.get("restarts") else "") if x)
            stale.append(f"{name} STALE {age:.1f}s" + (f" ({extra})" if extra else ""))
    if stale and level == "ok":
        level = "warn"
    if stats.disk_free_gb is not None:
        mins = minutes_remaining(stats)
        parts.append(f"disk {stats.disk_free_gb:.0f} GB"
                     + (f" (~{mins:.0f} min)" if mins is not None else ""))
    parts.extend(stale)
    if stats.fault:
        parts.append(f"FAULT: {stats.fault}")
    else:
        parts.append({"ok": "OK", "warn": "WARN", "fail": "FAIL"}[level])
    return " | ".join(parts), level
