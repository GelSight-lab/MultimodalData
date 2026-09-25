"""Per-stream integrity of a recorded episode: is every stream's frame rate
consistent, and were any frames lost?

`validate` answers "does this episode meet the recorder's contract" with
pass/fail checks. This module answers the operator's simpler question,
stream by stream: how many frames, at what rate, and were any dropped.

    python -m twm.recorder integrity <episode.h5>
    python -m twm.visualize <episode.h5> --check

Lost frames are counted, not guessed from single gaps, because sensor
clocks jitter (OptiTrack delivers packets in bursts; a late Arducam frame
is followed by an early one). For every stream with its own clock:

    expected = span / nominal period        (nominal = declared fps if the
                                             stream carries one, else the
                                             densest interval in its histogram)
    lost     = expected - delivered

For GelSight and Arducam, which the recorder samples once per tick, expected
is capped at one frame per tick: a camera faster than the tick rate is not
"losing" the frames the tick never asked for. Ticks whose sensor timestamp
repeats the previous tick's are "held" (no new frame yet); held is reported
separately and is not loss. A stream passes when lost is at most
LOST_OK_FRACTION of expected or LOST_OK_FRAMES, whichever is larger; the
count is always shown, so a tolerated hiccup is still visible. The tick clock itself is the recorder's
scheduler and barely jitters, so a tick is lost when an interval reaches
1.5/fps.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

try:  # register the BLOSC filter when available
    import hdf5plugin  # noqa: F401
except ImportError:  # pragma: no cover
    pass
import h5py

LOST_GAP_FACTOR = 1.5            # a tick interval this many nominal periods wide lost ticks
LOST_OK_FRACTION = 0.005         # a stream passes when lost <= this fraction of expected...
LOST_OK_FRAMES = 2               # ...or at most this many frames, whichever is larger (short clips)
BIG_GAP_FACTOR = 3.0             # a gap this wide is reported as a warning even if within budget
OPTITRACK_EDGE_TOLERANCE_S = 0.25   # tracker data may start/stop this far from the tick span


@dataclass
class StreamStat:
    name: str
    kind: str                     # ticks | frames | sensor | optitrack
    count: int
    expected: Optional[int] = None    # T for frame/sensor datasets
    rate_hz: Optional[float] = None       # samples per second of the stored stream
    native_hz: Optional[float] = None     # distinct sensor frames per second
    median_dt_ms: Optional[float] = None
    max_dt_ms: Optional[float] = None
    lost: int = 0                 # frames missing from the stream's own clock
    held: Optional[int] = None    # ticks where a sensor repeated its previous frame
    ok: bool = True
    note: str = ""


@dataclass
class IntegrityReport:
    path: str
    T: int
    fps: float
    streams: List[StreamStat] = field(default_factory=list)
    problems: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def to_dict(self) -> Dict[str, Any]:
        return {"path": self.path, "T": self.T, "fps": self.fps, "ok": self.ok,
                "problems": list(self.problems), "warnings": list(self.warnings),
                "streams": [asdict(s) for s in self.streams]}

    def table(self) -> str:
        def num(v, fmt):
            return "-" if v is None else format(v, fmt)
        rows = [("stream", "frames", "rate Hz", "native Hz", "median dt", "max dt",
                 "lost", "held", "status")]
        for s in self.streams:
            frames = str(s.count) if s.expected is None or s.count == s.expected \
                else f"{s.count}/{s.expected}"
            status = "ok" if s.ok else ("LOST %d" % s.lost if s.lost else "FAIL")
            if s.note:
                status += f"  {s.note}"
            rows.append((s.name, frames, num(s.rate_hz, ".2f"), num(s.native_hz, ".1f"),
                         "-" if s.median_dt_ms is None else f"{s.median_dt_ms:.1f} ms",
                         "-" if s.max_dt_ms is None else f"{s.max_dt_ms:.1f} ms",
                         str(s.lost), "-" if s.held is None else str(s.held), status))
        widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]) - 1)]
        lines = []
        for r in rows:
            cells = [r[0].ljust(widths[0])] + [r[i].rjust(widths[i]) for i in range(1, len(widths))]
            lines.append("  ".join(cells) + "  " + r[-1])
        verdict = "INTEGRITY OK" if self.ok else "INTEGRITY FAILED"
        tail = [f"{verdict}: {self.path}  (T={self.T}, {self.fps:g} fps)"]
        tail += [f"  problem: {p}" for p in self.problems]
        tail += [f"  warning: {w}" for w in self.warnings]
        return "\n".join(lines + tail)


def _tick_gaps(ts: np.ndarray, nominal_dt: float):
    """(median_dt, max_dt, lost) for the tick clock: every interval at least
    LOST_GAP_FACTOR nominal periods wide lost round(dt/nominal) - 1 ticks."""
    if len(ts) < 2:
        return None, None, 0
    dt = np.diff(ts)
    wide = dt[dt >= LOST_GAP_FACTOR * nominal_dt]
    lost = int(np.sum(np.rint(wide / nominal_dt) - 1)) if wide.size else 0
    return float(np.median(dt)), float(dt.max()), lost


def _nominal_dt(dt: np.ndarray) -> Optional[float]:
    """The stream's typical period: the densest 0.5 ms bin of its interval
    histogram, refined by the median of the intervals within 15 % of it.
    Robust to burst jitter and to dropouts as long as most intervals are
    on-period."""
    dt = dt[dt > 0]
    if dt.size == 0:
        return None
    # A near-zero interval is a burst delivery (two packets at once), never
    # the stream's period; keep it out of the histogram. The mean interval is
    # the threshold's anchor because bursts cannot move it (they leave the
    # span and the count unchanged).
    bulk = dt[dt >= 0.25 * np.mean(dt)]
    dt = bulk if bulk.size else dt
    edges = np.arange(0.0, float(dt.max()) + 0.0005, 0.0005)
    if len(edges) < 2:
        return float(np.median(dt))
    hist, edges = np.histogram(dt, bins=edges)
    k = int(np.argmax(hist))
    mode = 0.5 * (edges[k] + edges[k + 1])
    near = dt[(dt > 0.85 * mode) & (dt < 1.15 * mode)]
    return float(np.median(near)) if near.size else float(mode)


def _count_loss(ts: np.ndarray, declared_fps: Optional[float] = None,
                cap_intervals: Optional[int] = None):
    """(nominal_dt, expected_intervals, lost, median_dt, max_dt) for a stream
    with its own clock, by count against its nominal period."""
    if len(ts) < 2:
        return None, 0, 0, None, None
    dt = np.diff(ts)
    nominal = (1.0 / declared_fps) if declared_fps else _nominal_dt(dt)
    if not nominal or nominal <= 0:
        return None, 0, 0, float(np.median(dt)), float(dt.max())
    expected = int(round(float(ts[-1] - ts[0]) / nominal))
    if cap_intervals is not None:
        expected = min(expected, cap_intervals)
    lost = max(0, expected - (len(ts) - 1))
    return nominal, expected, lost, float(np.median(dt)), float(dt.max())


def _within_budget(lost: int, expected: int) -> bool:
    return lost <= max(LOST_OK_FRACTION * max(expected, 1), LOST_OK_FRAMES)


def _rate(ts: np.ndarray) -> Optional[float]:
    if len(ts) < 2 or ts[-1] <= ts[0]:
        return None
    return float((len(ts) - 1) / (ts[-1] - ts[0]))


def _ms(v: Optional[float]) -> Optional[float]:
    return None if v is None else v * 1e3


def _tick_stat(ts: np.ndarray, fps: float) -> StreamStat:
    median, mx, lost = _tick_gaps(ts, 1.0 / fps)
    note = ""
    if len(ts) > 1 and np.any(np.diff(ts) <= 0):
        note = "clock not strictly increasing"
    return StreamStat("timestamps", "ticks", len(ts), None, _rate(ts), None,
                      _ms(median), _ms(mx), lost, None,
                      ok=(lost == 0 and not note), note=note)


def _frames_stat(name: str, count: int, T: int, rate: Optional[float]) -> StreamStat:
    ok = count == T
    return StreamStat(name, "frames", count, T, rate, None, None, None, 0, None,
                      ok=ok, note="" if ok else f"{T - count} frame(s) short of T")


def _sensor_stat(name: str, count: int, sensor_ts: np.ndarray, T: int,
                 tick_rate: Optional[float], declared_fps: Optional[float] = None):
    """Returns (stat, warning or None)."""
    notes = []
    if count != T:
        notes.append(f"{T - count} frame(s) short of T")
    if len(sensor_ts) != T:
        notes.append(f"{len(sensor_ts)} timestamps for {T} ticks")
    if len(sensor_ts) > 1 and np.any(np.diff(sensor_ts) < 0):
        notes.append("sensor clock goes backwards")
    distinct = np.unique(sensor_ts) if len(sensor_ts) else sensor_ts
    held = int(len(sensor_ts) - len(distinct))
    nominal, expected, lost, median, mx = _count_loss(distinct, declared_fps,
                                                      cap_intervals=max(T - 1, 0))
    warning = None
    if nominal and mx is not None and mx >= BIG_GAP_FACTOR * nominal:
        warning = f"{name}: longest gap between new frames {mx * 1e3:.0f} ms"
    stat = StreamStat(name, "sensor", count, T, tick_rate, _rate(distinct),
                      _ms(median), _ms(mx), lost, held,
                      ok=(not notes and _within_budget(lost, expected)), note="; ".join(notes))
    return stat, warning


def _optitrack_stat(name: str, ts: np.ndarray, t0: float, t1: float):
    """Returns (stat, warning or None)."""
    if len(ts) == 0:
        return (StreamStat(name, "optitrack", 0, None, None, None, None, None, 0, None,
                           ok=True, note="no samples"),
                f"{name}: no samples (rigid body not broadcast)")
    nominal, expected, lost, median, mx = _count_loss(ts)
    notes, warning = [], None
    if np.any(np.diff(ts) < 0):
        notes.append("clock goes backwards")
    if ts[0] > t0 + OPTITRACK_EDGE_TOLERANCE_S or ts[-1] < t1 - OPTITRACK_EDGE_TOLERANCE_S:
        warning = (f"{name}: covers {ts[0] - t0:+.2f}s .. {ts[-1] - t1:+.2f}s "
                   f"relative to the tick span")
    elif nominal and mx is not None and mx >= BIG_GAP_FACTOR * nominal:
        warning = f"{name}: longest gap {mx * 1e3:.0f} ms"
    return (StreamStat(name, "optitrack", len(ts), None, _rate(ts), None,
                       _ms(median), _ms(mx), lost, None,
                       ok=(not notes and _within_budget(lost, expected)), note="; ".join(notes)),
            warning)


def check_integrity(path: str, fps: Optional[float] = None) -> IntegrityReport:
    with h5py.File(path, "r") as f:
        ts = f["timestamps"][:].astype(np.float64)
        T = len(ts)
        fps = float(fps or f["metadata"].attrs.get("fps", 30))
        report = IntegrityReport(str(path), T, fps)
        if T == 0:
            report.problems.append("timestamps: episode has 0 ticks")
            return report

        tick = _tick_stat(ts, fps)
        report.streams.append(tick)

        if "realsense" in f:
            for cam in sorted(f["realsense"]):
                for ds in ("color", "depth"):
                    name = f"realsense/{cam}/{ds}"
                    if name in f:
                        report.streams.append(
                            _frames_stat(name, int(f[name].shape[0]), T, tick.rate_hz))

        for group in ("gelsight", "arducam"):
            if group not in f:
                continue
            for sub in sorted(f[group]):
                g = f[f"{group}/{sub}"]
                if "frames" not in g:
                    continue
                sensor_ts = g["timestamps"][:].astype(np.float64) if "timestamps" in g \
                    else np.zeros(0)
                declared = g.attrs.get("fps")
                stat, warning = _sensor_stat(
                    f"{group}/{sub}", int(g["frames"].shape[0]), sensor_ts, T,
                    tick.rate_hz, float(declared) if declared else None)
                report.streams.append(stat)
                if warning:
                    report.warnings.append(warning)

        if "optitrack" in f:
            for tr in sorted(f["optitrack"]):
                g = f[f"optitrack/{tr}"]
                if "timestamps" not in g:
                    continue
                stat, warning = _optitrack_stat(
                    f"optitrack/{tr}", g["timestamps"][:].astype(np.float64),
                    float(ts[0]), float(ts[-1]))
                report.streams.append(stat)
                if warning:
                    report.warnings.append(warning)

    for s in report.streams:
        if not s.ok:
            what = f"{s.lost} lost frame(s)" if s.lost and not s.note else s.note or "failed"
            report.problems.append(f"{s.name}: {what}")
    return report


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    p = argparse.ArgumentParser(prog="python -m twm.recorder integrity",
                                description="Per-stream frame rate and lost-frame check.")
    p.add_argument("path")
    p.add_argument("--fps", type=float, default=None,
                   help="Nominal tick rate; defaults to metadata.attrs['fps'].")
    p.add_argument("--json", action="store_true", help="Print the JSON report instead of a table.")
    a = p.parse_args(argv)
    report = check_integrity(a.path, fps=a.fps)
    print(json.dumps(report.to_dict(), indent=2) if a.json else report.table())
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
