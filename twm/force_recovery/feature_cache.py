"""Per-frame feature cache for method optimization on GT datasets.

Decoding an 8-MP JPEG and reconstructing depth costs ~250 ms; regressing a
force model costs microseconds. So every candidate feature is computed once
per frame and cached, and the method comparison iterates on the cache.

Features per frame (from the plane-removed indentation map δ):
  vol        Σ max(δ-θ, over threshold) · dA         (current method)
  vol_soft   Σ max(δ-θ/3, 0) · dA                    (lower threshold)
  vol15      Σ δ^1.5 · dA  over contact              (Hertz-flavoured)
  area       contact area
  maxd       robust max depth (p99 over contact)
  cx, cy     contact centroid (mm, gel frame)        (edge filtering)
  border_mm  centroid distance to the nearest pad edge
plus ground truth force, commanded z_mm, probe id, and the FEATS U-net's
total normal force on the same frame (markerless input — its failure mode
is part of the comparison).
"""
from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .depth_force import (MARGIN_X, MARGIN_Y, MM_PER_PIXEL, PIXEL_AREA_MM2,
                          DepthForceEstimator, remove_background_plane)
from .fota_cnc_validation import _group_by_tar, build_index, load_images
from .run_episode import OUT_ROOT

CACHE = OUT_ROOT / "feature_cache"


def indent_map(est: DepthForceEstimator, img: np.ndarray) -> np.ndarray:
    depth_px = est._raw_depth(est._prep(img))
    ind = remove_background_plane(depth_px * MM_PER_PIXEL, est._interior)
    return cv2.GaussianBlur(ind, (5, 5), 1.5)


def features_from_indent(ind: np.ndarray, thr: float) -> dict:
    interior = np.zeros_like(ind, bool)
    interior[MARGIN_Y:-MARGIN_Y, MARGIN_X:-MARGIN_X] = True
    contact = (ind > thr) & interior
    soft = (ind > thr / 3.0) & interior
    H, W = ind.shape
    out = {
        "vol": float(ind[contact].sum()) * PIXEL_AREA_MM2,
        "vol_soft": float(ind[soft].sum()) * PIXEL_AREA_MM2,
        "vol15": float((ind[contact] ** 1.5).sum()) * PIXEL_AREA_MM2,
        "area": float(contact.sum()) * PIXEL_AREA_MM2,
        "maxd": float(np.percentile(ind[contact], 99)) if contact.any() else 0.0,
    }
    if contact.any():
        ys, xs = np.nonzero(contact)
        w = ind[ys, xs]
        cx = float((xs * w).sum() / w.sum()) * MM_PER_PIXEL
        cy = float((ys * w).sum() / w.sum()) * MM_PER_PIXEL
        out["cx"], out["cy"] = cx, cy
        out["border_mm"] = float(min(cx, W * MM_PER_PIXEL - cx,
                                     cy, H * MM_PER_PIXEL - cy))
    else:
        out["cx"] = out["cy"] = float("nan")
        out["border_mm"] = float("nan")
    return out


class FeatsNet:
    """The FEATS U-net, run on the same frames for the head-to-head."""

    def __init__(self, device="cuda:0"):
        from .feats_infer import FeatsPredictor, preprocess

        self.pred = FeatsPredictor(device=device, shift=None)
        self.preprocess = preprocess

    def total_normal(self, full_frame_rgb: np.ndarray) -> float:
        g = self.pred.predict(self.preprocess(full_frame_rgb)[None])[0]
        return -g.total_normal        # their z is compression-negative


def build_cnc_cache(split: str, max_frames: int = 700,
                    use_gpu: bool = True) -> Path:
    index = build_index(split)
    forces = np.array([e["force"] for e in index])
    order = np.argsort(forces)

    # references: the truly-free frames; zero map: median over 60 scattered
    near_zero = [index[i] for i in order if forces[i] < 0.5][:6]
    refs = load_images(near_zero)
    est = DepthForceEstimator(refs, use_gpu=use_gpu, already_cropped=False,
                              inpaint=False)
    rng = np.random.default_rng(0)
    rand_rows = [index[i] for i in rng.choice(len(index), 60, replace=False)]
    est.recon.depth_map_zero = 0.0
    raws = []
    for tar, rows in _group_by_tar(rand_rows).items():
        with tarfile.open(tar) as tf:
            for p in rows:
                img = np.asarray(Image.open(io.BytesIO(
                    tf.extractfile(p["jpg"]).read())).convert("RGB"))
                raws.append(est._raw_depth(est._prep(img)))
    est.recon.depth_map_zero = np.median(np.stack(raws), axis=0)
    resid = np.stack([remove_background_plane(
        est._raw_depth(est._prep(r)) * MM_PER_PIXEL, est._interior)
        for r in refs])
    inner = resid[:, MARGIN_Y:-MARGIN_Y, MARGIN_X:-MARGIN_X]
    thr = max(5.0 * 1.4826 * float(np.median(np.abs(inner))), 0.01)

    feats_net = FeatsNet() if use_gpu else None
    pick = [index[i] for i in
            order[np.linspace(0, len(order) - 1,
                              min(max_frames, len(order))).astype(int)]]
    rows_out = []
    for tar, rows in _group_by_tar(pick).items():
        with tarfile.open(tar) as tf:
            for p in rows:
                img = np.asarray(Image.open(io.BytesIO(
                    tf.extractfile(p["jpg"]).read())).convert("RGB"))
                ind = indent_map(est, img)
                f = features_from_indent(ind, thr)
                f.update({"force_true": float(p["force"]),
                          "z_mm": float(p["z_mm"]),
                          "x_mm": float(p["x_mm"]), "y_mm": float(p["y_mm"]),
                          "probe": str(p["obj_name"]),
                          "feats_pred_n": feats_net.total_normal(img)
                          if feats_net else float("nan")})
                rows_out.append(f)
                if len(rows_out) % 100 == 0:
                    print(f"  {len(rows_out)}/{len(pick)}", flush=True)

    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / f"cnc_mini_{split}.json"
    out.write_text(json.dumps({"threshold_mm": thr, "rows": rows_out}))
    print(f"cached {len(rows_out)} -> {out}", flush=True)
    return out


if __name__ == "__main__":
    import sys

    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 700
    build_cnc_cache(split, n)
