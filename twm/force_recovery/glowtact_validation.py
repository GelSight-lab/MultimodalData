"""Validate against GlowTact's GelSight subset — cleaned Mini presses, F/T GT.

14,715 frames of a markerless GelSight Mini pressed by controlled probes
(A-E, quad, round, star, triangle) and deformable objects (balloon, bulb,
pipe, rope), forces 0-20 N encoded in the filenames, full-view 640x480.
Conditions are friendlier than cnc_Mini: 13 genuinely free frames for
referencing and a cleaned selection. Protocol identical to the other GT
validations (flatfield pipeline, vol15 feature, train-half scale fit,
edge stratification).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

from .depth_force import DepthForceEstimator
from .feature_cache import features_from_indent, indent_map
from .run_episode import OUT_ROOT

ROOT = OUT_ROOT / "glowtact" / "GelSight_force_final_14716"
PAT = re.compile(r"^(?P<obj>.+?)\|(?P<idx>\d+)pos\|(?P<x>[-\d.]+), "
                 r"(?P<y>[-\d.]+), (?P<z>[-\d.]+) f\|(?P<f>[\d.]+)\.jpg$")


def build_index() -> list[dict]:
    rows = []
    for p in ROOT.rglob("*.jpg"):
        m = PAT.match(p.name)
        if not m:
            continue
        rows.append({"path": str(p), "family": p.parent.name,
                     "force": float(m.group("f")),
                     "x_mm": float(m.group("x")), "y_mm": float(m.group("y")),
                     "z_mm": float(m.group("z"))})
    return rows


def run(max_eval: int = 600, use_gpu: bool = True) -> dict:
    from scipy.stats import pearsonr, spearmanr

    index = build_index()
    forces = np.array([r["force"] for r in index])
    order = np.argsort(forces)
    free = [index[i] for i in order if forces[i] < 0.1][:10]
    refs = [np.asarray(Image.open(r["path"]).convert("RGB")) for r in free]
    print(f"{len(index)} frames | free refs: "
          f"{[round(r['force'], 2) for r in free]}", flush=True)

    est = DepthForceEstimator(refs, use_gpu=use_gpu, already_cropped=False,
                              inpaint=False, flatfield=True)
    print(f"threshold {est.contact_threshold_mm*1e3:.0f} um", flush=True)

    picks = [index[i] for i in
             order[np.linspace(0, len(order) - 1,
                               min(max_eval, len(order))).astype(int)]]
    rows = []
    for i, p in enumerate(picks):
        img = np.asarray(Image.open(p["path"]).convert("RGB"))
        ind = indent_map(est, img)
        f = features_from_indent(ind, est.contact_threshold_mm)
        rows.append({**{k: p[k] for k in
                        ("force", "family", "x_mm", "y_mm", "z_mm")}, **f})
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(picks)}", flush=True)

    ft = np.array([r["force"] for r in rows])
    v = np.array([r["vol15"] for r in rows])
    bd = np.array([r["border_mm"] for r in rows])
    inner = np.isfinite(bd) & (bd > 3.0)

    # split-half scale fit (odd fit, even eval) to keep the MAE honest
    fit = np.arange(len(rows)) % 2 == 1
    good = fit & (v > 0.2) & (ft > 0.5)
    scale = float(np.median(ft[good] / v[good]))
    ev = ~fit
    fcal = v * scale

    per_family = {}
    fams = np.array([r["family"] for r in rows])
    for fam in np.unique(fams):
        m = fams == fam
        if m.sum() >= 10:
            per_family[fam] = {
                "n": int(m.sum()),
                "rho": float(spearmanr(v[m], ft[m]).statistic)}

    report = {
        "n_frames": len(rows),
        "force_range_n": [float(ft.min()), float(ft.max())],
        "spearman_rho": float(spearmanr(v, ft).statistic),
        "pearson_r": float(pearsonr(v, ft).statistic),
        "spearman_nonedge": float(spearmanr(v[inner], ft[inner]).statistic),
        "n_nonedge": int(inner.sum()),
        "scale_n_per_mm3": scale,
        "mae_split_half_n": float(np.abs(fcal[ev] - ft[ev]).mean()),
        "per_family": per_family,
        "per_frame": rows,
    }
    (OUT_ROOT / "glowtact_validation.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    rep = run()
    for k, val in rep.items():
        if k not in ("per_frame", "per_family"):
            print(k, val)
    for fam, m in sorted(rep["per_family"].items()):
        print(f"  {fam:12s} n={m['n']:3d} rho={m['rho']:+.3f}")
