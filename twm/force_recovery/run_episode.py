"""Batch normal-force estimation over release episodes.

Row alignment: published parquet row ``i`` corresponds to source H5 frame
``trim + i + 15`` for the legacy recordings (the +15 tactile latency shift
baked into the release — verified frame-exactly by
``react_preprocess.backfill.verify_against_h5``).

Only rows flagged ``tactile_<side>_is_new`` are reconstructed; duplicated
rows reuse the previous estimate, which is exact (identical pixels give an
identical estimate) and cuts the work by ~3.6x on legacy recordings.

Reference (no-contact) frames for calibration are the 15 lowest-intensity
fresh rows of the episode, spread apart in time; the estimator's median/MAD
calibration tolerates a minority of them being lightly in contact.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import pyarrow.parquet as pq

from .depth_force import DepthForceEstimator

DATA_ROOT = Path("/media/yxma/Disk1/twm/data")
STAGE_ROOT = Path("/media/yxma/Disk1/twm/release")
OUT_ROOT = Path("/media/yxma/Disk1/twm/force_recovery")

LEGACY_SHIFT = 15
N_REFERENCE = 15
FIELDS = ("force_normal_n", "volume_mm3", "contact_area_mm2", "max_depth_mm")


def _reference_rows(intensity: np.ndarray, is_new: np.ndarray,
                    n: int = N_REFERENCE) -> np.ndarray:
    """Lowest-intensity fresh rows, at most one per second of recording."""
    fresh = np.where(is_new)[0]
    order = fresh[np.argsort(intensity[fresh])]
    picked: list[int] = []
    for row in order:
        if all(abs(row - p) >= 30 for p in picked):
            picked.append(int(row))
        if len(picked) == n:
            break
    return np.array(sorted(picked))


def process_side(task: str, date: str, ep: str, side: str,
                 keep_top_depths: int = 3) -> dict:
    """Estimate per-row normal force for one sensor of one episode."""
    table = pq.read_table(str(STAGE_ROOT / task / "meta" / date / f"{ep}.parquet"))
    inten = np.asarray(table[f"tactile_{side}_intensity"].to_numpy())
    is_new = np.asarray(table[f"tactile_{side}_is_new"].to_numpy())
    trim = int(np.asarray(table["source_h5_frame"].to_numpy())[0])
    T = len(inten)

    h5_path = DATA_ROOT / task / date / f"{ep}.h5"
    ref_rows = _reference_rows(inten, is_new)

    out = {k: np.zeros(T, np.float32) for k in FIELDS}
    kept_depths: list[tuple[int, np.ndarray]] = []

    with h5py.File(str(h5_path), "r") as f:
        frames = f[f"gelsight/{side}/frames"]
        n_frames = len(frames)

        def src(row: int) -> int:
            return min(trim + row + LEGACY_SHIFT, n_frames - 1)

        est = DepthForceEstimator([frames[src(int(r))] for r in ref_rows])

        last = None
        for row in range(T):
            if is_new[row] or last is None:
                last = est.estimate(frames[src(row)])
            for key, value in zip(FIELDS, (last.normal_n, last.volume_mm3,
                                           last.contact_area_mm2,
                                           last.max_depth_mm)):
                out[key][row] = value

        # re-run the strongest rows with the depth map kept, for figures
        top = np.argsort(out["force_normal_n"])[::-1][:keep_top_depths]
        for row in top:
            r = est.estimate(frames[src(int(row))], keep_depth=True)
            kept_depths.append((int(row), r.depth_map.astype(np.float32)))

    meta = {
        "task": task, "date": date, "episode": ep, "side": side,
        "trim": trim, "shift": LEGACY_SHIFT,
        "contact_threshold_mm": est.contact_threshold_mm,
        "noise_std_mm": est.noise_std_mm,
        "reference_rows": ref_rows,
    }
    out_dir = OUT_ROOT / task / date
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / f"{ep}_{side}.npz",
        **out, **{f"depth_row_{row}": d for row, d in kept_depths},
        **{k: v for k, v in meta.items() if not isinstance(v, str)},
        )
    meta["out"] = str(out_dir / f"{ep}_{side}.npz")
    meta["force_max_n"] = float(out["force_normal_n"].max())
    meta["force_p50_contact"] = float(np.percentile(
        out["force_normal_n"][out["force_normal_n"] > 0.02], 50)
        if (out["force_normal_n"] > 0.02).any() else 0.0)
    return meta


def process_episode(task: str, date: str, ep: str) -> list[dict]:
    return [process_side(task, date, ep, side) for side in ("left", "right")]
