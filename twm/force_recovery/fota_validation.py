"""Validate against FoTa (T3) — GelSight Mini presses with pose labels.

FoTa's panda_warped subset has no force ground truth (verified: f_z is NaN
throughout; the raw WebDataset JSONs carry only x,y,z + quaternion). What it
does have is the end-effector pose of every still while a Panda presses 13
household objects against the sensor — and within one capture (fixed object,
fixed initial pose) the advance along the pressing axis is a monotone proxy
of applied force. So the check is rank correlation, not absolute error:

    per capture:  Spearman( F_estimate , pose-derived press depth )

Two things make this complementary to the FEATS validation:
- third-party rig, real household objects instead of CNC indenters;
- captures come in BOTH gel types (46 markerless / 18 markered in val),
  so the pipeline is tested on its native domain and the foreign one
  under identical conditions.

Press-axis convention per capture: depth = projection of position onto the
first PCA axis of the still poses. The free end is identified without the
force estimate: the stills at the free extreme are nearly identical images
(no contact), so the end whose 5 extreme stills have the smaller internal
pixel spread is free; references come from there.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from .depth_force import DepthForceEstimator
from .run_episode import OUT_ROOT

FOTA = Path("/media/yxma/Disk1/yuxiang/mini_data_parquet/fota_labeled")

N_REF = 5
N_EVAL = 40


def _decode(row_image) -> np.ndarray:
    data = row_image["bytes"] if isinstance(row_image, dict) else row_image
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))


def _press_depth(xyz: np.ndarray) -> np.ndarray:
    """Projection of each still onto the capture's dominant motion axis."""
    c = xyz - xyz.mean(axis=0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    return c @ vt[0]


def _internal_spread(images: list[np.ndarray]) -> float:
    stack = np.stack([i.astype(np.float32) for i in images])
    return float(np.abs(stack - stack.mean(0)).mean())


def validate_capture(group, use_gpu: bool = True) -> dict | None:
    from scipy.stats import spearmanr

    xyz = group[["x_mm", "y_mm", "z_mm"]].to_numpy()
    if len(group) < 2 * N_REF + 10:
        return None
    s = _press_depth(xyz)
    order = np.argsort(s)

    lo_idx, hi_idx = order[:N_REF], order[-N_REF:]
    lo_imgs = [_decode(group.image.iloc[i]) for i in lo_idx]
    hi_imgs = [_decode(group.image.iloc[i]) for i in hi_idx]
    # free end = visually static end
    if _internal_spread(lo_imgs) <= _internal_spread(hi_imgs):
        refs, depth = lo_imgs, s - s[lo_idx].mean()
    else:
        refs, depth = hi_imgs, s[hi_idx].mean() - s

    est = DepthForceEstimator(refs, use_gpu=use_gpu,
                              inpaint=bool(group.markered.iloc[0]))
    pick = np.linspace(0, len(group) - 1, min(N_EVAL, len(group))).astype(int)
    forces = np.array([est.estimate(_decode(group.image.iloc[i])).normal_n
                       for i in pick])
    d = depth[pick]
    rho = float(spearmanr(forces, d).statistic)
    return {
        "capture": str(group.capture.iloc[0]),
        "markered": bool(group.markered.iloc[0]),
        "n_stills": int(len(group)),
        "press_range_mm": float(s.max() - s.min()),
        "spearman_force_vs_depth": rho,
        "force_max_n": float(forces.max()),
        "contact_fraction": float((forces > 0.1).mean()),
    }


def run(split: str = "val", max_captures: int | None = None,
        use_gpu: bool = True) -> dict:
    df = pq.read_table(str(FOTA / f"{split}-00000-of-00001.parquet")).to_pandas()
    results = []
    for i, (_, group) in enumerate(df.groupby("capture")):
        if max_captures and i >= max_captures:
            break
        r = validate_capture(group.reset_index(drop=True), use_gpu)
        if r:
            results.append(r)
            print(f"  {r['capture'][:46]:48s} markered={r['markered']} "
                  f"rho={r['spearman_force_vs_depth']:+.2f} "
                  f"Fmax={r['force_max_n']:.1f}N", flush=True)

    def _summ(rows):
        rhos = np.array([r["spearman_force_vs_depth"] for r in rows])
        return {"n": len(rows), "rho_p25": float(np.percentile(rhos, 25)),
                "rho_median": float(np.median(rhos)),
                "rho_p75": float(np.percentile(rhos, 75)),
                "rho_positive_frac": float((rhos > 0).mean())} if rows else {}

    report = {
        "split": split, "n_captures": len(results),
        "markerless": _summ([r for r in results if not r["markered"]]),
        "markered": _summ([r for r in results if r["markered"]]),
        "all": _summ(results),
        "per_capture": results,
    }
    (OUT_ROOT / f"fota_validation_{split}.json").write_text(
        json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    rep = run()
    for k in ("markerless", "markered", "all"):
        print(k, rep[k])
