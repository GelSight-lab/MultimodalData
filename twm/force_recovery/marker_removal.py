"""Marker-dot removal for the DEPTH / 3D product path (opt-in, adopted).

This is the *pipeline* home of the marker step whose evidence lives in
``marker_study.py``. The split it encodes is the whole finding, so it is
stated once here and repeated wherever a number appears on the site:

    ADOPTED for depth / 3D geometry.   NOT adopted for force features.

Why the split, in measured numbers (FEATS, our only marker gel):

* GEOMETRY — the depth map carries a dimple lattice at the marker pitch
  (f0 = 1/31.9 px). Power at f0 relative to its spectral shoulders drops
  1.523 -> 0.890 (x0.65) with ``img_telea``: cv2.inpaint(Telea) applied to
  BOTH the reference and the frame *before* differencing. Lower on 91% of
  120 frames above 1 N, Wilcoxon p = 2.6e-19. ``grad_inpaint`` is close
  (0.924); gradient ZEROING is nearly useless (1.251, x0.93) because g := 0
  puts a dipole layer on every hole boundary, exactly at the lattice
  frequency the step is meant to remove.
* FORCE — nothing clears the baseline. Per-group half/half + isotonic,
  5 seeds, 186 eval frames: base rho 0.7747 / MAE 5.03 N (seed sd 0.029),
  every variant's median paired delta negative, best variant wins 2 of 5
  seeds. Marker removal costs 0.00-0.06 rho.
* Both falsification controls behave: area-matched RANDOM masks inpainted at
  non-marker positions give 0.7697 (no gain, so the small changes are not
  "inpainting = smoothing = better"), and the detector returns exactly 0
  blobs on the two markerless references (GlowTact, cnc), so it is
  marker-specific rather than a generic blob finder.

Prior art: GelSight Wedge (Wang/She/Dong/Adelson, ICRA 2021) Fig. 10 gives
the marker algorithm we build on — zero-fill vs griddata nearest / linear /
cubic interpolation over the dot holes, nearest costing 10 ms at 200x150 and
linear/cubic 60/70 ms. We use OpenCV's Telea inpainting in image space
instead of gradient-space interpolation because, measured above, filling the
IMAGE beats filling the GRADIENT and both beat masking.

Entry points:
    marker_mask(ref)                   -> dilated dot mask, or None if the gel
                                          has no dots (markerless = exact no-op)
    stages_depth(img, ref)             -> debug_gallery.stages() on inpainted
                                          inputs; wraps, never edits, the force
                                          path
    python -m force_recovery.marker_removal figure   -> the site figure
                                          (raw | markers | inpainted |
                                           depth before/after | mesh
                                           before/after), needs xvfb-run
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .debug_gallery import stages

OUT = Path("/media/yxma/Disk1/twm/force_recovery/marker_study")
SITE_ASSETS = Path("/media/yxma/Disk1/twm/force_recovery/site/assets")


# ------------------------------------------------------------------ detector
def detect_markers(ref: np.ndarray, thr: float = 8.0, sigma: float = 12.0,
                   a_lo: int = 15, a_hi: int = 400,
                   dilate: int = 2) -> tuple[np.ndarray, np.ndarray, dict]:
    """Dark-dot detector on the reference frame.

    Markers are dark blobs on a smoothly shaded gel: a large-sigma Gaussian
    estimates the shading, and the dots are what sits `thr` grey levels below
    it. Area gating drops shading residue. Returns (mask, centres, stats).

    Verified specific, not a blob finder: 63/63 dots and 0 rejects on the
    FEATS reference, the same 63 for every threshold in 3..16 grey levels,
    radius 6.3-7.4 px, pitch 31.9 px, dilated mask 20.6% of the frame — and
    exactly 0 blobs on the GlowTact and cnc_Mini references.
    """
    g = ref.mean(2).astype(np.float32)
    resid = cv2.GaussianBlur(g, (0, 0), sigma) - g
    m = cv2.morphologyEx((resid > thr).astype(np.uint8), cv2.MORPH_OPEN,
                         np.ones((3, 3), np.uint8))
    n, lab, st, cent = cv2.connectedComponentsWithStats(m)
    keep = [i for i in range(1, n) if a_lo <= st[i, 4] <= a_hi]
    mask = np.isin(lab, keep)
    radii = np.sqrt(st[keep, 4] / np.pi) if keep else np.zeros(0)
    if dilate:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (2 * dilate + 1, 2 * dilate + 1))
        mask = cv2.dilate(mask.astype(np.uint8), k).astype(bool)
    info = {"n": len(keep), "rejected": n - 1 - len(keep),
            "radius_med": float(np.median(radii)) if len(radii) else 0.0,
            "coverage": float(mask.mean())}
    if len(keep) > 1:
        from scipy.spatial import cKDTree
        d, _ = cKDTree(cent[keep]).query(cent[keep], k=2)
        info["spacing"] = float(np.median(d[:, 1]))
    return mask, cent[keep], info


def inpaint_img(img: np.ndarray, mask: np.ndarray, method: str = "telea",
                radius: float = 5.0) -> np.ndarray:
    flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
    u8 = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.inpaint(u8, mask.astype(np.uint8), radius, flag).astype(np.float32)


# ------------------------------------------------------------------ pipeline
_MASK_CACHE: dict[bytes, tuple[np.ndarray | None, dict]] = {}


def marker_mask(ref: np.ndarray, **kw):
    """Cached dot mask for a reference frame; None when the gel is markerless.

    Returning None (rather than an all-false mask) is what makes the step a
    strict no-op on markerless gels: `stages_depth` then calls `stages()`
    with the untouched arrays, so GlowTact / cnc / React output is bit-exact.
    """
    key = ref.tobytes()
    if key not in _MASK_CACHE:
        mask, _, info = detect_markers(ref, **kw)
        _MASK_CACHE[key] = (mask if info["n"] else None, info)
    return _MASK_CACHE[key][0]


def marker_info(ref: np.ndarray, **kw) -> dict:
    marker_mask(ref, **kw)
    return _MASK_CACHE[ref.tobytes()][1]


def stages_depth(img: np.ndarray, ref: np.ndarray,
                 inpaint_markers: bool = True,
                 mask: np.ndarray | None = None,
                 method: str = "telea") -> dict:
    """`debug_gallery.stages()` for the depth / 3D product.

    A wrapper, deliberately: `stages()` is the FORCE path and marker removal
    measurably does not help force, so that function stays untouched and
    byte-identical. Only the geometry we display goes through here.

    Identical by construction to the `img_telea` variant of
    `marker_study.stages_m` (inpaint BOTH reference and frame, then
    difference) — `test_units.test_stages_depth_matches_marker_study` asserts
    that equivalence on a real FEATS frame.
    """
    if not inpaint_markers:
        return stages(img, ref)
    if mask is None:
        mask = marker_mask(ref)
    if mask is None:
        return stages(img, ref)
    return stages(inpaint_img(img, mask, method),
                  inpaint_img(ref, mask, method))


# ------------------------------------------------------------------ geometry
def dimple_power(depth: np.ndarray, spacing: float) -> float:
    """Depth power at the marker lattice frequency vs its shoulders.

    Radially averaged power spectrum of the (Hann-windowed, mean-removed)
    depth map; the marker grid puts a bump at f0 = 1/spacing cyc/px. The
    statistic is P(0.9-1.1 f0) / P(shoulders), so a surface with no marker
    signature gives ~1 and cannot go meaningfully below it.
    """
    H, W = depth.shape
    win = np.outer(np.hanning(H), np.hanning(W))
    d = (depth - depth.mean()) * win
    P = np.abs(np.fft.fftshift(np.fft.fft2(d))) ** 2
    fy = np.fft.fftshift(np.fft.fftfreq(H))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(W))[None, :]
    fr = np.hypot(fy, fx)
    f0 = 1.0 / spacing
    band = (fr > 0.90 * f0) & (fr < 1.10 * f0)
    sh = (((fr > 0.62 * f0) & (fr < 0.80 * f0))
          | ((fr > 1.28 * f0) & (fr < 1.6 * f0)))
    return float(P[band].mean() / max(P[sh].mean(), 1e-30))


# ------------------------------------------------------------------ figure
def cmd_figure(n_rows: int = 3) -> Path:
    """Site figure: what marker inpainting buys, on FEATS, end to end.

    Columns: raw | markers detected | inpainted | depth before | depth after
    | Open3D mesh before | Open3D mesh after. Needs a display:
        xvfb-run -a -s "-screen 0 1400x1000x24" \\
            python -m force_recovery.marker_removal figure
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .debug_gallery import load_feats
    from .o3d_view import has_display, mesh_view_rgb, remove_halo_pedestal

    if not has_display():
        raise SystemExit('needs: xvfb-run -a -s "-screen 0 1400x1000x24" ...')

    frames, get = load_feats()
    _, ref = get(frames[0])
    mask, cent, info = detect_markers(ref)
    sp = info["spacing"]
    print(f"FEATS reference: {info['n']} dots, {info['rejected']} rejects, "
          f"radius {info['radius_med']:.1f}px, pitch {sp:.1f}px, "
          f"dilated mask {info['coverage']*100:.1f}% of frame")

    # Row choice must not be a highlight reel. Score a candidate pool, then
    # take, in each force tercile, the frame whose after/before ratio is
    # closest to the pool MEDIAN — so the figure shows the typical effect,
    # not the best one. (The very top of FEATS is a 60 N press whose gel is
    # so sheared that a static reference mask misses most dots; that is a
    # limitation to state in words, not an example to lead with.)
    fs = np.array([fr["f"] for fr in frames])
    pool = [i for i in np.argsort(fs) if fs[i] > 1.0][::max(len(fs) // 45, 1)]
    scored = []
    for i in pool:
        img, _ = get(frames[i])
        b = dimple_power(stages_depth(img, ref, inpaint_markers=False)["depth"], sp)
        a_ = dimple_power(stages_depth(img, ref, mask=mask)["depth"], sp)
        scored.append((i, b, a_, a_ / max(b, 1e-9)))
    med = float(np.median([s[3] for s in scored]))
    print(f"candidate pool n={len(scored)}: median after/before ratio "
          f"{med:.3f} (study median over 120 frames: 0.58)")
    order = sorted(scored, key=lambda s: fs[s[0]])
    sel = []
    for t in range(n_rows):
        chunk = order[t * len(order) // n_rows:(t + 1) * len(order) // n_rows]
        sel.append(min(chunk, key=lambda s: abs(s[3] - med))[0])

    # 8 columns: the difference image was missing, and it is the only
    # panel that shows what the reconstruction is actually reading.
    fig, ax = plt.subplots(n_rows, 8, figsize=(24.5, 3.05 * n_rows))
    ax = np.atleast_2d(ax)
    rows = []
    for r, i in enumerate(sel):
        img, _ = get(frames[i])
        inp = inpaint_img(img, mask, "telea")
        before = stages_depth(img, ref, inpaint_markers=False)
        after = stages_depth(img, ref, mask=mask)
        dp0 = dimple_power(before["depth"], sp)
        dp1 = dimple_power(after["depth"], sp)
        vmax = max(before["depth"].max(), after["depth"].max(), 0.05)
        ov = img.copy()
        ov[mask] = [255, 60, 60]
        from . import eval_panel as EP
        panels = [
            (np.clip(img, 0, 255).astype(np.uint8),
             f"raw  [{frames[i]['group']}]  F = {fs[i]:.1f} N", None),
            (EP.diff_rgb(img, ref),
             "difference  dI = frame − ref  (×3, colour)", None),
            (np.clip(ov, 0, 255).astype(np.uint8),
             f"markers detected ({info['n']} dots, {info['coverage']*100:.0f}%"
             " of frame)", None),
            (np.clip(inp, 0, 255).astype(np.uint8),
             "inpainted (Telea, ref + frame)", None),
            (before["depth"], f"depth BEFORE — dimple power {dp0:.2f}",
             "inferno"),
            (after["depth"], f"depth AFTER — dimple power {dp1:.2f}",
             "inferno"),
            (mesh_view_rgb(remove_halo_pedestal(before["depth"])),
             "3D mesh BEFORE", None),
            (mesh_view_rgb(remove_halo_pedestal(after["depth"])),
             "3D mesh AFTER", None),
        ]
        for a, (data, ti, cm) in zip(ax[r], panels):
            kw = dict(cmap=cm, vmin=0, vmax=vmax) if cm else dict(cmap=cm)
            a.imshow(data, **kw)
            a.set_title(ti, fontsize=8.5)
            a.axis("off")
        rows.append({"f": float(fs[i]), "group": str(frames[i]["group"]),
                     "dimple_before": dp0, "dimple_after": dp1,
                     "ratio": dp1 / max(dp0, 1e-9),
                     "peak_before": float(before["depth"].max()),
                     "peak_after": float(after["depth"].max())})
        print(f"  F={fs[i]:6.2f}N  dimple {dp0:.3f} -> {dp1:.3f}  "
              f"peak {before['depth'].max():.2f} -> {after['depth'].max():.2f} mm")

    fig.suptitle(
        "FEATS marker gel: inpainting the dots before differencing removes the "
        "dimple lattice from the DEPTH map (adopted for 3D) — it does not help "
        "force (rho 0.775 → 0.737, not adopted there)\n"
        "Paired over 120 frames above 1 N: lattice power 1.523 → 0.890 "
        "(×0.65), lower on 91% of frames, Wilcoxon p = 2.6e-19.   Dots left in "
        "column 3 are real: they shear with the gel (median 1.7 px, >8 px on "
        "8% of frames) and a static reference mask cannot follow them.",
        fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    SITE_ASSETS.mkdir(parents=True, exist_ok=True)
    out = SITE_ASSETS / "feats_marker_removal.png"
    fig.savefig(out, dpi=105, bbox_inches="tight")
    plt.close(fig)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figure_rows.json").write_text(json.dumps(
        {"detector": info, "pool_median_ratio": med,
         "pool_n": len(scored), "rows": rows}, indent=1))
    print(f"-> {out}")
    return out


if __name__ == "__main__":
    import sys
    {"figure": cmd_figure}[sys.argv[1] if len(sys.argv) > 1 else "figure"]()
