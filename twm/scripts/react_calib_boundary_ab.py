"""Did the boundary-condition change help or hurt REACT's newton calibration?

The dataset-level evaluation said the free boundary is worth +0.108 rho to the
LUT on cnc_mini_26. Rebuilding React's own calibration cache then moved its
held-out rho 0.739 -> 0.533. Both cannot be dismissed, and the second one is
the number React's exported force column depends on, so it gets its own test.

The first comparison was NOT matched: the cache keeps a frame only if its
thresholded depth has >= 30 pixels, and changing the solver changes which
frames pass (471 vs 478). This builds both feature sets over the SAME frames,
keeps the intersection, and runs them through `react_calib.fit` itself — the
production fitting code, not a copy — with the identical split.

    python -m scripts.react_calib_boundary_ab
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from force_recovery import react_calib as RC
from force_recovery.debug_gallery import LUT, CNT
from force_recovery.lut_calibration import (BINS, DI_RANGE, GLOWTACT,
                                            MM_PER_PIXEL, PAT, crop)
from force_recovery.poisson import poisson_dirichlet, poisson_neumann

F_MAX_N = RC.F_MAX_N
# The clamped solver suppresses low-frequency drift as a side effect of pinning
# the border. If that is why it wins here, removing the drift explicitly should
# let the physically-correct boundary keep its geometry AND the robustness.
ARMS = ("dirichlet", "neumann", "neumann_plane", "neumann_quad")


def _feats(img, ref, solver: str):
    """The stages() feature block, with the solver as an explicit argument.

    Duplicated here on purpose rather than adding a global switch to the
    production path: a module-level "which solver" flag is exactly the kind of
    hidden state that makes two figures of the same frame disagree later.
    """
    import cv2
    dI = img - ref
    q = np.clip((dI + DI_RANGE) / (2 * DI_RANGE) * (BINS - 1),
                0, BINS - 1).astype(np.int32)
    g = LUT[q[..., 0], q[..., 1], q[..., 2]].copy()
    mag = cv2.GaussianBlur(np.abs(dI).max(2), (5, 5), 1.5)
    valid = cv2.morphologyEx((mag > 8.0).astype(np.uint8), cv2.MORPH_OPEN,
                             np.ones((3, 3), np.uint8)).astype(bool)
    g[~valid] = 0.0
    if solver.startswith("neumann"):
        from force_recovery.poisson import anchor_region, detrend_flat
        far = anchor_region(valid)
        depth = poisson_neumann(g[..., 0], g[..., 1])
        if far.mean() <= 0.05:
            depth = depth - np.median(depth)
        elif solver == "neumann":
            depth = depth - np.median(depth[far])
        else:                       # neumann_plane / neumann_quad
            depth = detrend_flat(depth, far,
                                 order=2 if solver.endswith("quad") else 1)
    else:
        depth = poisson_dirichlet(g[..., 0], g[..., 1])
    if depth[valid].size and np.median(depth[valid]) < 0:
        depth = -depth
    d = np.maximum(depth, 0.0)
    m = d > 0.05
    if m.sum() < 30:
        return None
    px = MM_PER_PIXEL ** 2
    yy, xx = np.nonzero(m)
    w = d[m]
    out = {"vol": float(d[m].sum() * px), "vol2": float((d[m] ** 2).sum() * px),
           "maxd": float(np.percentile(d, 99.8)), "area": float(m.sum() * px),
           "cx": float((xx * w).sum() / w.sum()),
           "cy": float((yy * w).sum() / w.sum())}
    out["h1"] = np.sqrt(out["area"]) * out["maxd"]
    return out


def main() -> int:
    ref = crop(np.asarray(Image.open(GLOWTACT / "round" / "initial.jpg")
                          .convert("RGB"))).astype(np.float32)
    rows = {k: [] for k in ARMS}
    files = sorted((GLOWTACT / "round").glob("*.jpg"))
    kept = dropped = 0
    for i, p in enumerate(files):
        m = PAT.search(p.name)
        if not m or not (0.15 < float(m["f"]) <= F_MAX_N):
            continue
        img = crop(np.asarray(Image.open(p).convert("RGB"))).astype(np.float32)
        got = {k: _feats(img, ref, k) for k in ARMS}
        if any(v is None for v in got.values()):    # matched set only
            dropped += 1
            continue
        kept += 1
        meta = {"f": float(m["f"]), "x": float(m["x"]), "y": float(m["y"]),
                "z": -float(m["z"])}
        for k, v in got.items():
            rows[k].append({**v, **meta})
        if kept % 100 == 0:
            print(f"  {kept} matched frames", flush=True)

    print(f"\nmatched {kept} frames ({dropped} dropped by one solver or "
          f"the other)\n")
    orig = RC.CACHE
    out = {}
    for name, rs in rows.items():
        with tempfile.NamedTemporaryFile("w", suffix=".json",
                                         delete=False) as fh:
            json.dump(rs, fh)
            RC.CACHE = Path(fh.name)
        print(f"== {name}")
        _, h = RC.fit(report=True, holdout=True)
        from scipy.stats import spearmanr
        inview = h["clip"] <= 0
        for label, m in (("fully in view", inview), ("clipped", ~inview)):
            if m.sum() >= 12:
                r = spearmanr(h["pred"][m], h["f"][m]).statistic
                print(f"     {label:14s} n={m.sum():3d}  rho={r:.3f}  "
                      f"MAE={np.abs(h['pred'][m]-h['f'][m]).mean():.3f} N")
            else:
                print(f"     {label:14s} n={m.sum():3d}  too few to score")
        out[name] = h
    RC.CACHE = orig
    print("\nSame frames, same split, same fitting code — only the boundary "
          "condition differs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
