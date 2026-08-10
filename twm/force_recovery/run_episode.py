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
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import pyarrow.parquet as pq

from .depth_force import DepthForceEstimator

DATA_ROOT = Path("/media/yxma/Disk1/twm/data")
STAGE_ROOT = Path("/media/yxma/Disk1/twm/release")
OUT_ROOT = Path("/media/yxma/Disk1/twm/force_recovery")

# Re-exported from the single definition; do not redeclare the value here.
from twm.tactile_align import LEGACY_SHIFT  # noqa: E402,F401
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


def process_side(task: str, date: str, ep: str, side: str, *,
                 out_dir=None,
                 keep_top_depths: int = 3) -> dict:
    """Estimate per-row normal force for one sensor of one episode."""
    table = pq.read_table(str(STAGE_ROOT / task / "meta" / date / f"{ep}.parquet"))
    inten = np.asarray(table[f"tactile_{side}_intensity"].to_numpy())
    is_new = np.asarray(table[f"tactile_{side}_is_new"].to_numpy())
    trim = int(np.asarray(table["source_h5_frame"].to_numpy())[0])
    T = len(inten)

    h5_path = DATA_ROOT / task / date / f"{ep}.h5"
    ref_rows = _reference_rows(inten, is_new)

    # Force comes from the SAME calibration the rest of the project uses.
    # What was here — a single N-per-mm3 constant times the v1 MLP volume —
    # scored rho 0.297 on GlowTact `round` and mapped a true 0.16-8 N range
    # onto 0.01-103 N. See react_calib for the held-out numbers.
    from .react_calib import (CALIBRATION_NAME,
                              FORCE_RECONSTRUCTION as _FORCE_RECON,
                              fit as _fit_calib)
    predict_force = _fit_calib(report=False)
    scale_source = CALIBRATION_NAME

    out = {k: np.zeros(T, np.float32) for k in FIELDS}
    kept_depths: list[tuple[int, np.ndarray]] = []

    with h5py.File(str(h5_path), "r") as f:
        frames = f[f"gelsight/{side}/frames"]
        n_frames = len(frames)

        def src(row: int) -> int:
            return min(trim + row + LEGACY_SHIFT, n_frames - 1)

        # flatfield: validated on cnc_Mini ground truth (held-out rho
        # 0.34 -> 0.65 with edge filtering); normalizes the vignette the
        # depth MLP was never trained under.
        # Reconstruction is `stages()` — the same function the studies and the
        # site use — not the retired v1 MLP estimator.
        from .debug_gallery import stages
        from .lut_calibration import crop

        ref = np.median(np.stack([crop(frames[src(int(r))]).astype(np.float32)
                                  for r in ref_rows[:12]]), 0)

        # Force from `react_calib.force_stages` (the calibration-free solve,
        # 0.812 held-out rho against the LUT's 0.763 on the same presses and
        # split); the exported vol/area/maxd stay in the LUT's MILLIMETRES,
        # because those columns are read as geometry and the calibration-free
        # depth has no millimetre scale. Two reconstructions, two purposes,
        # both named in the metadata below.
        from .react_calib import force_stages
        last = None
        for row in range(T):
            if is_new[row] or last is None:
                img = crop(frames[src(row)]).astype(np.float32)
                st_mm = stages(img, ref)
                ft = st_mm["feats"]
                last = (predict_force(force_stages(img, ref)),
                        ft["vol"], ft["area"], ft["maxd"])
            for key, value in zip(FIELDS, last):
                out[key][row] = value

        # re-run the strongest rows with the depth map kept, for figures
        top = np.argsort(out["force_normal_n"])[::-1][:keep_top_depths]
        for row in top:
            st = stages(crop(frames[src(int(row))]).astype(np.float32), ref)
            kept_depths.append((int(row), st["depth"].astype(np.float32)))

    meta = {
        "task": task, "date": date, "episode": ep, "side": side,
        "trim": trim, "shift": LEGACY_SHIFT,
        # thresholds now live in stages(): |dI|>8 for the valid mask and
        # depth>0.05 mm for the contact mask
        "contact_threshold_mm": 0.05,
        "valid_mask_dI": 8.0,
        # which reconstruction produced which column
        "force_reconstruction": _FORCE_RECON,
        "geometry_reconstruction": "stages (LUT, millimetres)",
        "reference_rows": ref_rows,
        "force_calibration": scale_source,
        "scale_source": scale_source,
    }
    # An explicit destination so a reprocess can be COMPARED against the
    # published npz before replacing it. Without this the only way to try a
    # pipeline change was to overwrite the released force channel and hope.
    out_dir = Path(out_dir) if out_dir is not None else OUT_ROOT / task / date
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / f"{ep}_{side}.npz",
        **out, **{f"depth_row_{row}": d for row, d in kept_depths},
        # Strings included ON PURPOSE. The previous filter dropped every
        # str-valued field, which meant `force_calibration` — the identity of
        # the map turning mm^3 into newtons — existed only in the log. An npz
        # that cannot say which calibration produced its newtons is how a
        # pixel-unit weight vector went on scoring mm-unit features for weeks.
        **{k: (np.str_(v) if isinstance(v, str) else v)
           for k, v in meta.items()},
        )
    meta["out"] = str(out_dir / f"{ep}_{side}.npz")
    meta["force_max_n"] = float(out["force_normal_n"].max())
    meta["force_p50_contact"] = float(np.percentile(
        out["force_normal_n"][out["force_normal_n"] > 0.02], 50)
        if (out["force_normal_n"] > 0.02).any() else 0.0)
    return meta


def process_episode(task: str, date: str, ep: str) -> list[dict]:
    return [process_side(task, date, ep, side) for side in ("left", "right")]
