"""Episode validator: judges a recorded HDF5 episode against the format,
timing, and content invariants the recorder is supposed to guarantee.

Each check takes the open (read-only) file and a mutable `stats` dict, and
returns a `Check`. A check that raises is caught by `validate_episode` and
turned into a failing `Check` naming the exception, so one bad dataset
never hides the results of the other checks. A file that cannot even be
opened (truncated, zero-byte, corrupted) never raises out of
`validate_episode` either: it comes back as a report with every check
failing.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import h5py
import numpy as np

from twm.recorder.episode import VALID_ENDINGS
from twm.recorder.frames import decode_arducam, COLOR_SHAPE, DEPTH_SHAPE
from twm.recorder.schema import ARDUCAM_SLOTS, GELSIGHT_SIDES, count_optitrack_samples

logger = logging.getLogger(__name__)

CHECK_NAMES = ("metadata", "shapes", "tick_rate", "duration", "sensor_sync",
              "content", "optitrack", "writer")

# Sensor-clock fallback bounds (see check_sensor_sync).
_LAG_MIN_S = -0.005
# "Late sample" threshold: a tick's sensor timestamp lagging the tick clock
# by more than this is unusual but not, on its own, a defect — isolated
# scheduling jitter produces the occasional late sample even on a healthy
# rig. Tolerate up to _MAX_LATE_FRACTION of a stream's ticks past this bound
# before failing (see check_sensor_sync); every one is still counted and
# reported.
_LAG_MAX_S = 0.25
# Hard bound: the drivers' own contract (`max_age` / `max_no_update_time`
# in rig.py — the age at which a driver itself gives up on a stale sample).
# A tick lagging past this could not have come from a driver honoring that
# contract, so even a single occurrence fails outright, regardless of the
# late-sample tolerance above.
_LAG_HARD_MAX_S = 0.5
_MAX_LATE_FRACTION = 0.01
_MIN_DISTINCT_HZ = 10.0


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


def _safe_T(f: h5py.File) -> int:
    """T from the timestamps dataset, or 0 if it's missing/unreadable.

    Used only for the fallback assignment after `check_tick_rate` (which
    normally sets stats["T"] itself); this must never raise, because it
    runs outside the per-check `run()` safety net in `validate_episode`.
    """
    try:
        return int(f["timestamps"].shape[0])
    except Exception:
        return 0


def _frame_streams(f: h5py.File):
    """Yield (label, dataset, is_depth, expected_frame_shape, expected_dtype)
    for every configured frame dataset. Arducam's expected per-frame shape
    comes from its own group's height/width attrs (its resolution need not
    match the RealSense/GelSight default)."""
    if "realsense" in f:
        for name in sorted(f["realsense"]):
            g = f[f"realsense/{name}"]
            if "color" in g:
                yield f"realsense/{name}/color", g["color"], False, COLOR_SHAPE, np.uint8
            if "depth" in g:
                yield f"realsense/{name}/depth", g["depth"], True, DEPTH_SHAPE, np.uint16
    if "gelsight" in f:
        for side in GELSIGHT_SIDES:
            ds_path = f"gelsight/{side}/frames"
            if ds_path in f:
                yield ds_path, f[ds_path], False, COLOR_SHAPE, np.uint8
    if "arducam" in f:
        for slot in ARDUCAM_SLOTS:
            ds_path = f"arducam/{slot}/frames"
            if ds_path in f:
                g = f[f"arducam/{slot}"]
                if str(g.attrs.get("encoding", "bgr8")) == "mjpeg":
                    # Ragged: one JPEG per tick, so there is no per-frame
                    # shape to check. `None` means "length only".
                    yield ds_path, f[ds_path], False, None, None
                    continue
                height = int(g.attrs.get("height", COLOR_SHAPE[0]))
                width = int(g.attrs.get("width", COLOR_SHAPE[1]))
                yield ds_path, f[ds_path], False, (height, width, 3), np.uint8


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


def _missing_promised_streams(f: h5py.File) -> List[str]:
    """Groups metadata.attrs promises (by serial/config lists) that aren't there."""
    attrs = f["metadata"].attrs
    missing = []

    realsense_serials = attrs.get("realsense_serials", [])
    for i in range(len(realsense_serials)):
        if f"realsense/cam{i}" not in f:
            missing.append(f"realsense/cam{i}")

    gelsight_serials = attrs.get("gelsight_serials", [])
    if len(gelsight_serials) > 0:
        for side in GELSIGHT_SIDES:
            if f"gelsight/{side}" not in f:
                missing.append(f"gelsight/{side}")

    arducam_json = attrs.get("arducam_config", None)
    if arducam_json:
        try:
            arducam_list = json.loads(arducam_json)
        except (TypeError, ValueError) as exc:
            return missing + [f"arducam_config: unparseable ({exc})"]
        for c in arducam_list:
            slot = c.get("slot")
            if slot and f"arducam/{slot}" not in f:
                missing.append(f"arducam/{slot}")

    return missing


def check_shapes(f: h5py.File, stats: Dict[str, Any]) -> Check:
    T = stats.get("T", int(f["timestamps"].shape[0]))
    problems = []
    checked = 0

    missing = _missing_promised_streams(f)
    if missing:
        problems.append("missing stream(s) promised by metadata: " + ", ".join(missing))

    for label, ds, is_depth, frame_shape, dtype in _frame_streams(f):
        checked += 1
        if frame_shape is None:                 # ragged: one JPEG per tick
            if ds.shape != (T,):
                problems.append(f"{label}: {ds.shape[0]} JPEG frame(s) for {T} tick(s)")
            continue
        expected_shape = (T, *frame_shape)
        if ds.shape != expected_shape:
            problems.append(f"{label}: shape {ds.shape} != {expected_shape}")
        elif ds.dtype != dtype:
            problems.append(f"{label}: dtype {ds.dtype} != {dtype}")
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
        stats["late_count"] = 0
        return Check("tick_rate", False, f"only {T} tick(s), can't measure rate")

    dt = np.diff(ts)
    expected_dt = 1.0 / fps
    median_dt = float(np.median(dt))
    max_dt = float(np.max(dt))
    late_mask = dt > 1.5 * expected_dt
    late_count = int(np.sum(late_mask))
    late_fraction = float(np.mean(late_mask))
    stats["median_dt"] = median_dt
    stats["max_dt"] = max_dt
    stats["late_fraction"] = late_fraction
    stats["late_count"] = late_count

    problems = []
    if not np.all(dt > 0):
        n_bad = int(np.sum(dt <= 0))
        problems.append(f"timestamps not strictly increasing ({n_bad} non-positive dt)")
    if abs(median_dt - expected_dt) > 0.1 * expected_dt:
        problems.append(f"median dt {median_dt * 1000:.2f} ms not within 10% of "
                        f"expected {expected_dt * 1000:.2f} ms")
    if max_dt > max_tick_gap_s:
        problems.append(f"max dt {max_dt:.3f}s > max_tick_gap_s {max_tick_gap_s:.3f}s")
    # A fraction cannot resolve below 1/T. On a 31-tick recording 1% is 0.31,
    # so a single late tick scored 3.23% and the check demanded ZERO of them
    # while its message still said "1%" -- a rounding artefact presented as a
    # tolerance. One late tick is tolerated whatever T is; on any recording
    # long enough for 1% to mean something (100 ticks or more) 0.01 * T is
    # already above 1, so nothing real changes.
    if late_fraction >= 0.01 and late_count > 1:
        problems.append(f"late_fraction {late_fraction * 100:.2f}% >= 1% ({late_count} tick(s))")

    if problems:
        return Check("tick_rate", False, "; ".join(problems))
    return Check("tick_rate", True,
                f"median dt {median_dt * 1000:.2f} ms, max dt {max_dt * 1000:.2f} ms, "
                f"late {late_fraction * 100:.2f}% ({late_count} tick(s))")


def check_duration(f: h5py.File, stats: Dict[str, Any], fps: int,
                   expected_duration: Optional[float], warmup_frames: int) -> Check:
    T = stats.get("T", int(f["timestamps"].shape[0]))
    if expected_duration is None:
        logger.warning("duration check skipped: no --expected-duration given for %s",
                       f.filename)
        return Check("duration", True, "skipped (no --expected-duration)")
    required = 0.97 * (expected_duration * fps - warmup_frames)
    term = (f"0.97 x ({expected_duration}s x {fps}fps - {warmup_frames} warmup) "
           f"= {required:.1f}")
    if T < required:
        return Check("duration", False, f"T={T} < required {term}")
    return Check("duration", True, f"T={T} >= required {term}")


_MAX_OUTAGE_FRACTION = 0.05   # sensor outages (stalls/restarts) may cover this much of an episode


def _runs(mask: np.ndarray):
    """[(start, end)) index pairs of consecutive True values."""
    runs = []
    start = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


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
        stats[f"sensor_lag_min_s.{label}"] = lag_min_val
        stats[f"sensor_lag_max_s.{label}"] = lag_max_val

        if np.array_equal(sensor_ts, tick_ts):
            # The rig substitutes the tick clock when a stream reports no
            # capture timestamp (rig.py's `frame_with_timestamp` fallback).
            # That produces lag == 0 everywhere, which would otherwise pass
            # every bound below without ever having measured a real sensor
            # clock at all.
            problems.append(f"{label}: timestamps identical to tick clock (fallback?)")
            continue

        ahead = int(np.sum(lag < _LAG_MIN_S))
        if ahead > 0:
            problems.append(f"{label}: sensor clock ahead of the tick clock by more than "
                            f"{-_LAG_MIN_S * 1000:.0f}ms on {ahead} tick(s)")

        # Outage: consecutive ticks whose sensor frame is older than the
        # drivers' 0.5 s staleness contract. The recorder keeps ticking through
        # a stalled sensor (recording the last frame with its OLD capture time)
        # while its supervisor restarts the stream, so an outage is expected
        # to look exactly like this. It is reported, and tolerated when short
        # and resolved; it fails the episode when it covers more than
        # _MAX_OUTAGE_FRACTION of the ticks or the sensor never came back.
        # A run of consecutive ticks with lag > 250 ms is one event: it is an
        # outage if the lag ever exceeds the 500 ms contract inside it (the
        # ramp into a stall belongs to the stall), otherwise a late spell.
        late_any = lag > _LAG_MAX_S
        outage_mask = np.zeros(T, dtype=bool)
        for a, b in _runs(late_any):
            if np.max(lag[a:b]) > _LAG_HARD_MAX_S:
                outage_mask[a:b] = True
        outages = _runs(outage_mask)
        outage_ticks = int(np.sum(outage_mask))
        outage_fraction = outage_ticks / T if T else 0.0
        longest_s = max((float(tick_ts[b - 1] - tick_ts[a]) + float(np.median(np.diff(tick_ts))) if T > 1 else 0.0
                         for a, b in outages), default=0.0)
        stats[f"sensor_outage_count.{label}"] = len(outages)
        stats[f"sensor_outage_longest_s.{label}"] = longest_s
        stats[f"sensor_outage_fraction.{label}"] = outage_fraction
        if outage_fraction > _MAX_OUTAGE_FRACTION:
            problems.append(f"{label}: {len(outages)} outage(s) covering "
                            f"{outage_fraction * 100:.1f}% of ticks (> "
                            f"{_MAX_OUTAGE_FRACTION * 100:.0f}%), longest {longest_s:.1f}s")
        elif outage_mask[-1]:
            problems.append(f"{label}: sensor still stale at the end of the episode "
                            f"(lag {lag[-1]:.1f}s) — it never resumed")

        late_mask = (lag > _LAG_MAX_S) & ~outage_mask
        late_count = int(np.sum(late_mask))
        late_fraction = late_count / T if T else 0.0
        stats[f"sensor_late_count.{label}"] = late_count
        if late_fraction >= _MAX_LATE_FRACTION:
            problems.append(f"{label}: {late_count} tick(s) ({late_fraction * 100:.2f}%) "
                            f"lag > {_LAG_MAX_S * 1000:.0f}ms, exceeds "
                            f"{_MAX_LATE_FRACTION * 100:.0f}% tolerance")

        late_note = (f", {late_count} late ({late_fraction * 100:.2f}%)"
                    if late_count else "")
        if outages:
            late_note += (f", {len(outages)} outage(s) ({outage_ticks} ticks, "
                          f"{outage_fraction * 100:.1f}%, longest {longest_s:.1f}s)")
        distinct = int(np.unique(sensor_ts).shape[0])
        if span > 1.0:
            distinct_hz = distinct / span
            stats[f"distinct_sensor_hz.{label}"] = distinct_hz
            if distinct_hz < _MIN_DISTINCT_HZ:
                problems.append(f"{label}: distinct sensor rate {distinct_hz:.1f} Hz < "
                                f"{_MIN_DISTINCT_HZ:.0f} Hz")
            else:
                details.append(f"{label}: lag [{lag_min_val * 1000:.1f}, "
                               f"{lag_max_val * 1000:.1f}] ms, {distinct_hz:.1f} Hz "
                               f"distinct{late_note}")
        else:
            stats[f"distinct_sensor_hz.{label}"] = None
            details.append(f"{label}: lag [{lag_min_val * 1000:.1f}, "
                           f"{lag_max_val * 1000:.1f}] ms, too short to judge "
                           f"rate{late_note}")

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
    for label, ds, is_depth, _frame_shape, _dtype in _frame_streams(f):
        # decode_arducam is a no-op on already-decoded frames, so the content
        # check reads both encodings without branching on which it has.
        frames = [decode_arducam(np.asarray(ds[int(i)])) if _frame_shape is None
                  else np.asarray(ds[int(i)]) for i in idx]
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


def _unreadable_file_report(path: str, exc: Exception) -> ValidationReport:
    checks = [Check("metadata", False, f"could not open file: {type(exc).__name__}: {exc}")]
    checks += [Check(name, False, "file not readable")
              for name in CHECK_NAMES if name != "metadata"]
    return ValidationReport(path=str(path), checks=checks, stats={})


def validate_episode(path: str, fps: Optional[int] = 30, expected_duration: Optional[float] = None,
                     max_tick_gap_s: float = 0.5, sample_frames: int = 20,
                     warmup_frames: int = 10) -> ValidationReport:
    """Open `path` read-only and run every check in order.

    A check that raises never aborts the report: the exception is recorded
    as a failing `Check` naming the exception type and message. A file that
    cannot be opened at all (truncated, zero-byte, corrupted) never raises
    either: it comes back as a report with a failing `metadata` check
    carrying the exception text and every other check failed as "file not
    readable".

    `fps=None` reads the recording fps from `metadata.attrs["fps"]`
    (falling back to 30 if that's unavailable too) — useful for the CLI,
    which passes None when `--fps` was not given.
    """
    try:
        f = h5py.File(path, "r")
    except Exception as exc:  # noqa: BLE001 - deliberately broad, see docstring
        logger.exception("could not open %s", path)
        return _unreadable_file_report(path, exc)

    stats: Dict[str, Any] = {}
    checks: List[Check] = []

    def run(name, fn):
        try:
            checks.append(fn())
        except Exception as exc:  # noqa: BLE001 - deliberately broad, see docstring
            logger.exception("check %s raised", name)
            checks.append(Check(name, False, f"{type(exc).__name__}: {exc}"))

    try:
        if fps is None:
            try:
                fps = int(f["metadata"].attrs.get("fps", 30))
            except Exception:
                logger.warning("could not read fps from metadata attrs for %s; "
                               "defaulting to 30", path)
                fps = 30

        # tick_rate computes T first; other checks read stats["T"] rather
        # than re-reading the dataset shape.
        run("tick_rate", lambda: check_tick_rate(f, stats, fps, max_tick_gap_s))
        stats.setdefault("T", _safe_T(f))
        run("metadata", lambda: check_metadata(f, stats))
        run("shapes", lambda: check_shapes(f, stats))
        run("duration", lambda: check_duration(f, stats, fps, expected_duration, warmup_frames))
        run("sensor_sync", lambda: check_sensor_sync(f, stats))
        run("content", lambda: check_content(f, stats, sample_frames))
        run("optitrack", lambda: check_optitrack(f, stats))
        run("writer", lambda: check_writer(f, stats))
    finally:
        f.close()

    # Reorder to the canonical name order for readability, regardless of the
    # run order above (tick_rate runs first so `stats["T"]` is available).
    order = {name: i for i, name in enumerate(CHECK_NAMES)}
    checks.sort(key=lambda c: order.get(c.name, len(order)))

    return ValidationReport(path=str(path), checks=checks, stats=stats)
