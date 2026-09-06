"""Episode validator: judges a recorded HDF5 episode against the format,
timing, and content invariants the recorder is supposed to guarantee.

Each check takes the open (read-only) file and a mutable `stats` dict, and
returns a `Check`. A check that raises is caught by `validate_episode` and
turned into a failing `Check` naming the exception, so one bad dataset
never hides the results of the other checks.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import h5py
import numpy as np

from twm.recorder.episode import VALID_ENDINGS
from twm.recorder.frames import COLOR_SHAPE, DEPTH_SHAPE
from twm.recorder.schema import ARDUCAM_SLOTS, GELSIGHT_SIDES, count_optitrack_samples

logger = logging.getLogger(__name__)

CHECK_NAMES = ("metadata", "shapes", "tick_rate", "duration", "sensor_sync",
              "content", "optitrack", "writer")


def _to_jsonable(value: Any) -> Any:
    """Cast numpy scalars/arrays (and nested dict/list) to plain JSON types."""
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return str(value)


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


@dataclass
class ValidationReport:
    path: str
    checks: List[Check] = field(default_factory=list)
    stats: Dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "ok": self.ok,
            "checks": [{"name": c.name, "ok": c.ok, "detail": c.detail}
                      for c in self.checks],
            "stats": _to_jsonable(self.stats),
        }


def _frame_streams(f: h5py.File):
    """Yield (label, dataset, is_depth) for every configured frame dataset."""
    if "realsense" in f:
        for name in sorted(f["realsense"]):
            g = f[f"realsense/{name}"]
            if "color" in g:
                yield f"realsense/{name}/color", g["color"], False
            if "depth" in g:
                yield f"realsense/{name}/depth", g["depth"], True
    if "gelsight" in f:
        for side in GELSIGHT_SIDES:
            ds_path = f"gelsight/{side}/frames"
            if ds_path in f:
                yield ds_path, f[ds_path], False
    if "arducam" in f:
        for slot in ARDUCAM_SLOTS:
            ds_path = f"arducam/{slot}/frames"
            if ds_path in f:
                yield ds_path, f[ds_path], False


def _sensor_streams(f: h5py.File):
    """Yield (label, timestamps_dataset) for every configured sensor-timestamp set."""
    if "gelsight" in f:
        for side in GELSIGHT_SIDES:
            ds_path = f"gelsight/{side}/timestamps"
            if ds_path in f:
                yield f"gelsight/{side}", f[ds_path]
    if "arducam" in f:
        for slot in ARDUCAM_SLOTS:
            ds_path = f"arducam/{slot}/timestamps"
            if ds_path in f:
                yield f"arducam/{slot}", f[ds_path]


def check_metadata(f: h5py.File, stats: Dict[str, Any]) -> Check:
    attrs = f["metadata"].attrs
    T = stats.get("T", int(f["timestamps"].shape[0]))
    problems = []

    valid = bool(attrs.get("valid", False))
    if not valid:
        problems.append(f"valid=False ({attrs.get('invalid_reason', 'unknown')})")

    ended_by = attrs.get("ended_by", "")
    if ended_by not in VALID_ENDINGS:
        problems.append(f"ended_by={ended_by!r} not in {VALID_ENDINGS}")

    frame_count = int(attrs.get("frame_count", -1))
    if frame_count != T:
        problems.append(f"frame_count={frame_count} != T={T}")

    gap_count = int(attrs.get("gap_count", -1))
    if gap_count != 0:
        problems.append(f"gap_count={gap_count} != 0")

    if problems:
        return Check("metadata", False, "; ".join(problems))
    return Check("metadata", True,
                f"valid=True, ended_by={ended_by}, frame_count={frame_count}, gap_count=0")


def check_shapes(f: h5py.File, stats: Dict[str, Any]) -> Check:
    T = stats.get("T", int(f["timestamps"].shape[0]))
    problems = []
    checked = 0
    for label, ds, is_depth in _frame_streams(f):
        checked += 1
        expected_shape = (T, *DEPTH_SHAPE) if is_depth else (T, *COLOR_SHAPE)
        expected_dtype = np.uint16 if is_depth else np.uint8
        if ds.shape != expected_shape:
            problems.append(f"{label}: shape {ds.shape} != {expected_shape}")
        elif ds.dtype != expected_dtype:
            problems.append(f"{label}: dtype {ds.dtype} != {expected_dtype}")
    for label, ds in _sensor_streams(f):
        checked += 1
        if ds.shape != (T,):
            problems.append(f"{label}/timestamps: shape {ds.shape} != ({T},)")
    if "optitrack" in f:
        for name in sorted(f["optitrack"]):
            ts = f[f"optitrack/{name}/timestamps"]
            pose = f[f"optitrack/{name}/pose"]
            checked += 1
            if pose.shape != (ts.shape[0], 7):
                problems.append(f"optitrack/{name}: pose shape {pose.shape} "
                                f"!= ({ts.shape[0]}, 7)")

    if problems:
        return Check("shapes", False, "; ".join(problems))
    return Check("shapes", True, f"{checked} dataset(s) match (T={T}) and expected dtype")


def check_tick_rate(f: h5py.File, stats: Dict[str, Any], fps: int,
                    max_tick_gap_s: float) -> Check:
    ts = f["timestamps"][:]
    T = ts.shape[0]
    stats["T"] = T
    if T < 2:
        stats["median_dt"] = 0.0
        stats["max_dt"] = 0.0
        stats["late_fraction"] = 0.0
        return Check("tick_rate", False, f"only {T} tick(s), can't measure rate")

    dt = np.diff(ts)
    expected_dt = 1.0 / fps
    median_dt = float(np.median(dt))
    max_dt = float(np.max(dt))
    late_fraction = float(np.mean(dt > 1.5 * expected_dt))
    stats["median_dt"] = median_dt
    stats["max_dt"] = max_dt
    stats["late_fraction"] = late_fraction

    problems = []
    if not np.all(dt > 0):
        n_bad = int(np.sum(dt <= 0))
        problems.append(f"timestamps not strictly increasing ({n_bad} non-positive dt)")
    if abs(median_dt - expected_dt) > 0.1 * expected_dt:
        problems.append(f"median dt {median_dt * 1000:.2f} ms not within 10% of "
                        f"expected {expected_dt * 1000:.2f} ms")
    if max_dt > max_tick_gap_s:
        problems.append(f"max dt {max_dt:.3f}s > max_tick_gap_s {max_tick_gap_s:.3f}s")
    if late_fraction >= 0.01:
        problems.append(f"late_fraction {late_fraction * 100:.2f}% >= 1%")

    if problems:
        return Check("tick_rate", False, "; ".join(problems))
    return Check("tick_rate", True,
                f"median dt {median_dt * 1000:.2f} ms, max dt {max_dt * 1000:.2f} ms, "
                f"late {late_fraction * 100:.2f}%")


def check_duration(f: h5py.File, stats: Dict[str, Any], fps: int,
                   expected_duration: Optional[float]) -> Check:
    T = stats.get("T", int(f["timestamps"].shape[0]))
    if expected_duration is None:
        return Check("duration", True, "no --expected-duration given, skipped")
    required = 0.97 * expected_duration * fps
    if T < required:
        return Check("duration", False,
                     f"T={T} < required {required:.1f} "
                     f"(0.97 x {expected_duration}s x {fps}fps)")
    return Check("duration", True,
                f"T={T} >= required {required:.1f} "
                f"(0.97 x {expected_duration}s x {fps}fps)")


def check_sensor_sync(f: h5py.File, stats: Dict[str, Any]) -> Check:
    tick_ts = f["timestamps"][:]
    T = tick_ts.shape[0]
    streams = list(_sensor_streams(f))
    if not streams:
        return Check("sensor_sync", True, "no sensor timestamp streams present")

    problems = []
    details = []
    span = float(tick_ts[-1] - tick_ts[0]) if T >= 2 else 0.0
    for label, ds in streams:
        sensor_ts = ds[:]
        if sensor_ts.shape[0] != T:
            problems.append(f"{label}: {sensor_ts.shape[0]} timestamps != T={T}")
            continue
        if T == 0:
            continue

        if not np.all(np.diff(sensor_ts) >= 0):
            n_bad = int(np.sum(np.diff(sensor_ts) < 0))
            problems.append(f"{label}: timestamps not non-decreasing ({n_bad} decreases)")

        lag = tick_ts - sensor_ts
        lag_min_val = float(np.min(lag))
        lag_max_val = float(np.max(lag))
        stats[f"sensor_lag_max_s.{label}"] = lag_max_val
        out_of_bounds = np.sum((lag < -0.005) | (lag > 0.25))
        if out_of_bounds > 0:
            problems.append(f"{label}: lag out of [-5ms, 250ms] for {int(out_of_bounds)} "
                            f"tick(s) (min={lag_min_val * 1000:.1f}ms, "
                            f"max={lag_max_val * 1000:.1f}ms)")

        distinct = int(np.unique(sensor_ts).shape[0])
        if span > 1.0:
            distinct_hz = distinct / span
            stats[f"distinct_sensor_hz.{label}"] = distinct_hz
            if distinct_hz < 10.0:
                problems.append(f"{label}: distinct sensor rate {distinct_hz:.1f} Hz < 10 Hz")
            else:
                details.append(f"{label}: lag [{lag_min_val * 1000:.1f}, "
                               f"{lag_max_val * 1000:.1f}] ms, {distinct_hz:.1f} Hz distinct")
        else:
            stats[f"distinct_sensor_hz.{label}"] = 0.0
            details.append(f"{label}: lag [{lag_min_val * 1000:.1f}, "
                           f"{lag_max_val * 1000:.1f}] ms, too short to judge rate")

    if problems:
        return Check("sensor_sync", False, "; ".join(problems))
    return Check("sensor_sync", True, "; ".join(details) if details else "no ticks to check")


def _sample_indices(T: int, n_samples: int) -> np.ndarray:
    if T <= 0:
        return np.array([], dtype=int)
    return np.unique(np.linspace(0, T - 1, min(n_samples, T)).astype(int))


def check_content(f: h5py.File, stats: Dict[str, Any], sample_frames: int) -> Check:
    T = stats.get("T", int(f["timestamps"].shape[0]))
    idx = _sample_indices(T, sample_frames)
    if idx.shape[0] < 2:
        return Check("content", True, f"only {idx.shape[0]} sample(s), too short to judge")

    problems = []
    checked = []
    for label, ds, is_depth in _frame_streams(f):
        frames = [np.asarray(ds[int(i)]) for i in idx]
        stds = [float(np.std(fr)) for fr in frames]
        low_std = [i for i, s in zip(idx, stds) if s <= 1.0]
        if low_std:
            problems.append(f"{label}: std <= 1.0 at {len(low_std)}/{len(stds)} "
                            f"sampled frame(s) (min std {min(stds):.2f})")

        n_diff = sum(0 if np.array_equal(frames[i], frames[i + 1]) else 1
                    for i in range(len(frames) - 1))
        n_pairs = len(frames) - 1
        if n_pairs > 0 and n_diff < 0.5 * n_pairs:
            problems.append(f"{label}: only {n_diff}/{n_pairs} consecutive sample pairs differ "
                            f"(frozen stream?)")

        if is_depth:
            nonzero_counts = [int(np.count_nonzero(fr)) for fr in frames]
            if any(c == 0 for c in nonzero_counts):
                n_zero = sum(1 for c in nonzero_counts if c == 0)
                problems.append(f"{label}: {n_zero}/{len(frames)} sampled frame(s) all-zero")

        checked.append(f"{label} (min std {min(stds):.2f}, {n_diff}/{max(n_pairs, 1)} differ)")

    if not checked:
        return Check("content", True, "no frame streams present")
    if problems:
        return Check("content", False, "; ".join(problems))
    return Check("content", True, "; ".join(checked))


def check_optitrack(f: h5py.File, stats: Dict[str, Any]) -> Check:
    if "optitrack" not in f:
        return Check("optitrack", True, "no optitrack group present")

    n_samples = count_optitrack_samples(f)
    stats["optitrack_samples"] = n_samples
    if n_samples == 0:
        return Check("optitrack", True, "no samples (ok if recorded with --no_optitrack)")

    tick_ts = f["timestamps"][:]
    if tick_ts.shape[0] == 0:
        lo, hi = -np.inf, np.inf
    else:
        lo, hi = float(tick_ts[0]) - 1.0, float(tick_ts[-1]) + 1.0

    problems = []
    for name in sorted(f["optitrack"]):
        ts = f[f"optitrack/{name}/timestamps"][:]
        if ts.shape[0] == 0:
            continue
        if not np.all(np.diff(ts) >= 0):
            n_bad = int(np.sum(np.diff(ts) < 0))
            problems.append(f"optitrack/{name}: timestamps not non-decreasing ({n_bad})")
        out = np.sum((ts < lo) | (ts > hi))
        if out > 0:
            problems.append(f"optitrack/{name}: {int(out)} sample(s) outside "
                            f"[{lo:.3f}, {hi:.3f}]")

    if problems:
        return Check("optitrack", False, "; ".join(problems))
    return Check("optitrack", True, f"{n_samples} sample(s) across tracker(s), in range")


def check_writer(f: h5py.File, stats: Dict[str, Any]) -> Check:
    attrs = f["metadata"].attrs
    problems = []
    if "queue_peak_fraction" not in attrs:
        problems.append("queue_peak_fraction attr missing")
    else:
        qpf = float(attrs["queue_peak_fraction"])
        stats["queue_peak_fraction"] = qpf
        if not qpf < 0.5:
            problems.append(f"queue_peak_fraction {qpf:.3f} not < 0.5")

    if "writer_mean_mb_s" not in attrs:
        problems.append("writer_mean_mb_s attr missing")
    else:
        mbs = float(attrs["writer_mean_mb_s"])
        stats["writer_mean_mb_s"] = mbs
        if not mbs > 0:
            problems.append(f"writer_mean_mb_s {mbs:.3f} not > 0")

    if problems:
        return Check("writer", False, "; ".join(problems))
    return Check("writer", True,
                f"queue_peak_fraction={stats['queue_peak_fraction']:.3f}, "
                f"writer_mean_mb_s={stats['writer_mean_mb_s']:.1f}")


def validate_episode(path: str, fps: int = 30, expected_duration: Optional[float] = None,
                     max_tick_gap_s: float = 0.5, sample_frames: int = 20) -> ValidationReport:
    """Open `path` read-only and run every check in order.

    A check that raises never aborts the report: the exception is recorded
    as a failing `Check` naming the exception type and message.
    """
    stats: Dict[str, Any] = {}
    checks: List[Check] = []

    def run(name, fn):
        try:
            checks.append(fn())
        except Exception as exc:  # noqa: BLE001 - deliberately broad, see docstring
            logger.exception("check %s raised", name)
            checks.append(Check(name, False, f"{type(exc).__name__}: {exc}"))

    with h5py.File(path, "r") as f:
        # tick_rate computes T first; other checks read stats["T"] rather
        # than re-reading the dataset shape.
        run("tick_rate", lambda: check_tick_rate(f, stats, fps, max_tick_gap_s))
        stats.setdefault("T", int(f["timestamps"].shape[0]))
        run("metadata", lambda: check_metadata(f, stats))
        run("shapes", lambda: check_shapes(f, stats))
        run("duration", lambda: check_duration(f, stats, fps, expected_duration))
        run("sensor_sync", lambda: check_sensor_sync(f, stats))
        run("content", lambda: check_content(f, stats, sample_frames))
        run("optitrack", lambda: check_optitrack(f, stats))
        run("writer", lambda: check_writer(f, stats))

    # Reorder to the canonical name order for readability, regardless of the
    # run order above (tick_rate runs first so `stats["T"]` is available).
    order = {name: i for i, name in enumerate(CHECK_NAMES)}
    checks.sort(key=lambda c: order.get(c.name, len(order)))

    return ValidationReport(path=str(path), checks=checks, stats=stats)
