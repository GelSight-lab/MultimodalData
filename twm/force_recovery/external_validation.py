"""Validate the depth->force estimator against FEATS ground truth.

FEATS (arXiv:2411.03315) pressed 24 indenters into a GelSight Mini with a
CNC machine and recorded FEA-derived force labels; the local parquet mirror
carries per-frame total forces (``f_z`` <= 0, compression). That gives what
React lacks: frames with known normal force, from the same sensor model.

Two caveats make this a *transfer* validation, not a same-domain one:
- FEATS gel carries marker dots; the depth network is run with the SDK's
  marker masking (dots detected by gray level, gradients interpolated).
- The absolute Winkler scale depends on gel constants (E, h) that differ
  between pads, so the meaningful outputs are (a) correlation between
  estimated and true force and (b) a fitted scale factor, which then
  *calibrates* the React estimates — turning the assumed E*/h into a
  measured one, under the stated same-gel-modulus assumption.

Reference (zero) frames per capture are the rows with |f_z| < 0.05 N.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from .depth_force import DepthForceEstimator, FORCE_PER_MM3
from .run_episode import OUT_ROOT

FEATS_PARQUET = Path("/media/yxma/Disk1/yuxiang/mini_data_parquet/feats")
CALIB_OUT = OUT_ROOT / "scale_calibration.json"

ZERO_FORCE_N = 0.05


def _decode(row_image) -> np.ndarray:
    """HF image column stores either raw bytes or a {'bytes': ...} struct."""
    data = row_image["bytes"] if isinstance(row_image, dict) else row_image
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))


def _marker_gray_max(img_rgb: np.ndarray, dot_fraction: float = 0.056) -> float:
    import cv2

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 2)
    return float(np.percentile(blurred, dot_fraction * 100.0))


def run(split: str = "val", max_frames: int = 300,
        use_gpu: bool = True) -> dict:
    """Estimate force on FEATS frames and compare with ground truth.

    Every FEATS row is one independent pressed configuration (no time
    series), and near-zero-force frames are rare, so the zero map is
    global: the lightest-touch frames of the split stand in for a
    no-contact reference. All frames come from one recording campaign
    ('black_dot' gel), which is what makes a shared zero map defensible.
    """
    table = pq.read_table(str(FEATS_PARQUET / f"{split}-00000-of-00001.parquet"))
    df = table.to_pandas()
    df["fz_true"] = -df["f_z"]                       # compression -> positive
    df = df.sort_values("fz_true").reset_index(drop=True)

    refs = [_decode(r) for r in df.head(10).image]
    est = DepthForceEstimator(refs, use_gpu=use_gpu, already_cropped=True,
                              inpaint=True)

    # Scope: pure normal loading, below gel saturation. Shear-loaded
    # captures measured spearman = -0.15 (shear churns the image without
    # adding indentation volume) and above ~30 N the pad bottoms out
    # (volume plateaus at ~7 mm^3) — both are outside what a normal-force
    # estimator can represent and are reported as limitations, not hidden.
    press = df[(df.fz_true >= ZERO_FORCE_N)
               & (df.fz_true <= 30.0)
               & ~df.capture.str.contains("shear")]
    idx = np.linspace(0, len(press) - 1,
                      min(max_frames, len(press))).astype(int)
    results = []
    for _, row in press.iloc[idx].iterrows():
        r = est.estimate(_decode(row.image))
        results.append({
            "capture": str(row.capture),
            "indenter": str(row.capture).split("_", 2)[-1],
            "f_true": float(row.fz_true), "volume_mm3": r.volume_mm3,
            "f_winkler": r.normal_n,
        })

    f_true = np.array([r["f_true"] for r in results])
    vol = np.array([r["volume_mm3"] for r in results])
    f_wink = np.array([r["f_winkler"] for r in results])

    # scale fit through the origin, robust to the tail
    good = vol > 0
    scale = float(np.median(f_true[good] / np.maximum(vol[good], 1e-9)))
    f_cal = vol * scale
    def _pearson(a, b):
        return float(np.corrcoef(a, b)[0, 1])

    report = {
        "split": split,
        "n_frames": len(results),
        "n_captures": len({r["capture"] for r in results}),
        "f_true_range_n": [float(f_true.min()), float(f_true.max())],
        "pearson_r": _pearson(vol, f_true),
        "spearman_rho": _spearman(vol, f_true),
        "scale_n_per_mm3": scale,
        "winkler_theoretical_n_per_mm3": FORCE_PER_MM3,
        "scale_vs_theory": scale / FORCE_PER_MM3,
        "mae_calibrated_n": float(np.abs(f_cal - f_true).mean()),
        "mae_relative": float((np.abs(f_cal - f_true) / f_true).mean()),
        "per_frame": results,
    }
    CALIB_OUT.parent.mkdir(parents=True, exist_ok=True)
    # Only the val split defines the canonical calibration — a test-split
    # run must never silently swap the scale under the batch pipeline.
    if split == "val":
        CALIB_OUT.write_text(json.dumps({k: v for k, v in report.items()
                                         if k != "per_frame"}, indent=2))
    (OUT_ROOT / f"feats_validation_{split}.json").write_text(
        json.dumps(report, indent=2))
    return report


def _spearman(a, b) -> float:
    from scipy.stats import spearmanr

    return float(spearmanr(a, b).statistic)


def calibrated_scale() -> float | None:
    """N per mm^3 fitted on FEATS, or None if validation hasn't run."""
    if CALIB_OUT.exists():
        return json.loads(CALIB_OUT.read_text())["scale_n_per_mm3"]
    return None
