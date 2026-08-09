"""Re-calibrate the photometric LUT on GlowTact's SELF-MADE pad subset.

Why this file exists
--------------------
The GlowTact release ships the same press protocol recorded on TWO different
sensors: `mini_cnc26/` (a commercial GelSight Mini) and
`GlowTact_force_final_14716/` (the authors' own pad). Every table we have
built so far came from the COMMERCIAL subset, but React's sensors are the
self-made kind. Measured rest-gel colour (median HSV over the reference
frame, H in degrees):

    React left / right      H = 90 / 82
    GlowTact self-made      H = 62          <- 20-28 deg away
    GelSight Mini (in use)  H = 169         <- 79-87 deg away

A photometric LUT is a map from colour change to surface slope, so a
3-4x larger hue offset is not cosmetic: it is the difference between
interpolating inside the calibrated colour volume and extrapolating outside
it. This module rebuilds the table on the self-made subset with the SAME
code path (`lut_calibration.fit_sphere_geometry` / `build_lut` /
`fill_lut_holes`) and then re-scores everything old-vs-new.

Structural difference to handle: the self-made subset has no single
`initial.jpg`; each family carries `initial/initial_XXX.jpg` x50. The
reference is the per-pixel median of those 50, and `cmd_ref` reports how much
the reference itself moves (split-half p95) because every dI in the pipeline
is measured against it.

Nothing here overwrites `glowtact_lut.npz` or any published asset: the new
table is saved beside it as `glowtact_selfmade_lut.npz` and all comparison
artefacts go to `lut_compare/`. Switching the pipeline over is a separate,
explicit decision.

What it found (so the next reader does not repeat it): the self-made pad is
NOT a photometric sensor. Its contact difference image is rank 1 -- the
contact is a uniform dark disc, all three channels dropping together -- while
the commercial GelSight Mini and React are both rank 3. A table from a scalar
to a 2-D slope is degenerate by construction, and the self-calibrated table
duly fails on its own pad (gradient angle 72 deg against a chance of 90).
The rest-gel hue is real but it describes the gel's appearance, not the
sensor's principle. `modality` is the command that settles this.

Run (`check` and `defaults` first -- they are the regression tests):
  python -m force_recovery.glowtact_selfmade check       # == debug_gallery.stages
  python -m force_recovery.glowtact_selfmade defaults    # published fit unmoved
  python -m force_recovery.glowtact_selfmade ref         # reference dispersion
  python -m force_recovery.glowtact_selfmade modality    # which sensor family?
  python -m force_recovery.glowtact_selfmade calibrate   # sphere fit + LUT
  python -m force_recovery.glowtact_selfmade degeneracy  # colour -> slope?
  python -m force_recovery.glowtact_selfmade geom        # dome / grad angle
  python -m force_recovery.glowtact_selfmade feats       # feature tables
  python -m force_recovery.glowtact_selfmade force       # force eval
  python -m force_recovery.glowtact_selfmade host react  # out-of-gamut rates
  python -m force_recovery.glowtact_selfmade calib_audit # shipped React model
  xvfb-run -a -s "-screen 0 1400x1000x24" \
      python -m force_recovery.glowtact_selfmade panel   # React depth panels
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import visualize as _V
from .lut_calibration import (BINS, CAL_OUT, DI_RANGE, MINI_CNC26, MM_PER_PIXEL,
                              PAT, W, H, build_lut, crop, detect_circle,
                              fill_lut_holes, fit_sphere_geometry)
from .run_episode import OUT_ROOT

SELFMADE = OUT_ROOT / "glowtact" / "GlowTact_force_final_14716"
COMMERCIAL = MINI_CNC26                     # GelSight Mini CNC presses
CMP = OUT_ROOT / "lut_compare"
OLD_LUT = CAL_OUT / "glowtact_lut.npz"
NEW_LUT = CAL_OUT / "glowtact_selfmade_lut.npz"

VALID_THR = 8.0
# depth datum and indenter radius fitted per pad (`calibrate`); the two pads
# log commanded z from data of their own, 3.3 mm apart, and their `round`
# indenters are not the same size
Z0 = {"commercial": 0.806, "selfmade": 4.121}
R_PAD = {"commercial": 3.351, "selfmade": 1.877}
# the six controlled indenter families present on BOTH pads (the self-made
# subset spells the small square `small_quad`, the commercial one
# `quad_small`) -- keeping the intersection is what makes old-vs-new a
# same-geometry comparison rather than a different-object comparison
FAMILIES = ("round", "quad", "star", "triangle", "B", "small_quad")
FAM_ALIAS = {"small_quad": "quad_small"}    # self-made name -> commercial name


# ---------------------------------------------------------------- loading
def family_dir(pad: str, fam: str) -> Path:
    root = SELFMADE if pad == "selfmade" else COMMERCIAL
    if pad != "selfmade":
        fam = FAM_ALIAS.get(fam, fam)
    return root / fam


def initial_frames(pad: str, fam: str) -> list[Path]:
    """Every rest-gel frame shipped for a family, whatever the layout."""
    d = family_dir(pad, fam)
    if (d / "initial").is_dir():
        return sorted((d / "initial").glob("*.jpg"))
    p = d / "initial.jpg"
    return [p] if p.exists() else []


def reference(pad: str, fam: str) -> np.ndarray:
    """Per-pixel median of the rest-gel frames, cropped to the pipeline view."""
    fr = initial_frames(pad, fam)
    if not fr:
        raise FileNotFoundError(f"no rest-gel frame for {pad}/{fam}")
    stack = [crop(np.asarray(Image.open(p).convert("RGB"))).astype(np.float32)
             for p in fr]
    return np.median(np.stack(stack), 0) if len(stack) > 1 else stack[0]


def press_rows(pad: str, fam: str) -> list[dict]:
    rows = []
    for p in family_dir(pad, fam).glob("*.jpg"):
        m = PAT.search(p.name)
        if m:
            rows.append({"path": p, "fam": fam, "z": -float(m["z"]),
                         "f": float(m["f"]), "x": float(m["x"]),
                         "y": float(m["y"])})
    return sorted(rows, key=lambda r: r["path"].name)


def load_img(p: Path) -> np.ndarray:
    return crop(np.asarray(Image.open(p).convert("RGB"))).astype(np.float32)


# ---------------------------------------------------------------- pipeline
def _poisson(gx, gy):
    sys.path.insert(0, str(Path.home() / "gelsight_heightmap_reconstruction"
                           / "python_version"))
    from fast_poisson import fast_poisson
    return fast_poisson(gx, gy)


def stages_lut(img, ref, lut, cnt) -> dict:
    """`debug_gallery.stages`, with the table passed in instead of imported.

    Identical arithmetic (same quantisation, same |dI|>8 opened valid mask,
    same fast_poisson, same five features) so that swapping the table is the
    ONLY variable in every comparison below. Features stay in raw pixel units
    (sum of depth over pixels, pixel count) exactly like the cached
    `lut_full.json`, so a model fit on one table can be read against the
    other without a hidden unit change.
    """
    dI = img - ref
    q = np.clip((dI + DI_RANGE) / (2 * DI_RANGE) * (BINS - 1),
                0, BINS - 1).astype(np.int32)
    g = lut[q[..., 0], q[..., 1], q[..., 2]].copy()
    observed = cnt[q[..., 0], q[..., 1], q[..., 2]] > 0
    mag = cv2.GaussianBlur(np.abs(dI).max(2), (5, 5), 1.5)
    valid = mag > VALID_THR
    valid = cv2.morphologyEx(valid.astype(np.uint8), cv2.MORPH_OPEN,
                             np.ones((3, 3), np.uint8)).astype(bool)
    g[~valid] = 0.0
    depth = _poisson(g[..., 0], g[..., 1])
    if depth[valid].size and np.median(depth[valid]) < 0:
        depth = -depth
    d = np.maximum(depth, 0.0)
    m = d > 0.05
    feats = {"vol": float(d[m].sum()), "vol2": float((d[m] ** 2).sum()),
             "maxd": float(np.percentile(d, 99.8)), "area": float(m.sum())}
    feats["h1"] = float(np.sqrt(feats["area"]) * feats["maxd"])
    if m.any():
        ys, xs = np.nonzero(m)
        wgt = d[m]
        feats["cx"] = float((xs * wgt).sum() / wgt.sum())
        feats["cy"] = float((ys * wgt).sum() / wgt.sum())
    else:
        feats["cx"] = feats["cy"] = float("nan")
    unobs = float((~observed[valid]).mean()) if valid.any() else float("nan")
    return {"dI": dI, "gx": g[..., 0], "gy": g[..., 1], "valid": valid,
            "observed": observed, "depth": d, "feats": feats,
            "unobserved": unobs, "n_valid": int(valid.sum())}


def cmd_check(n: int = 4) -> dict:
    """`stages_lut` must be `debug_gallery.stages` with the table injected.

    Every old-vs-new number below is produced by this function, so if it had
    drifted from the shipped path the comparison would be against a pipeline
    nobody runs. Checked on real frames, bit-for-bit, not by reading the code.
    """
    from . import debug_gallery as dg

    L = load_luts()["old_commercial"]
    assert np.array_equal(L["lut"], dg.LUT) and np.array_equal(L["count"],
                                                               dg.CNT)
    ref = reference("commercial", "round")
    rows = [r for r in press_rows("commercial", "round") if r["f"] > 1][:n]
    px = MM_PER_PIXEL ** 2
    wd = wf = 0.0
    for r in rows:
        img = load_img(r["path"])
        a, b = dg.stages(img, ref), stages_lut(img, ref, L["lut"], L["count"])
        wd = max(wd, float(np.abs(a["depth"] - b["depth"]).max()))
        wf = max(wf, abs(a["feats"]["vol"] - b["feats"]["vol"] * px),
                 abs(a["feats"]["area"] - b["feats"]["area"] * px),
                 abs(a["feats"]["maxd"] - b["feats"]["maxd"]))
    print(f"  {len(rows)} frames: max |depth delta| {wd:.3e}, "
          f"max |feature delta| (after the px -> mm^2 factor) {wf:.3e}")
    assert wd < 1e-12 and wf < 1e-9, "stages_lut drifted from stages()"
    return {"n": len(rows), "max_depth_delta": wd, "max_feat_delta": wf}


def cmd_defaults() -> dict:
    """`fit_sphere_geometry` on its DEFAULT bracket must not have moved.

    This module widened `z0_bounds` and changed the solver's starting z0 to
    the middle of the bracket. The midpoint of the shipped +/-2 mm bracket is
    0.0, i.e. exactly the old start -- but "should be identical" is not the
    same as "is identical", and the published commercial calibration
    (`lut_calibration/geometry.json`) is downstream of it. Slow (it
    re-detects 1152 circles) and worth it.
    """
    from .lut_calibration import load_family

    ref, rows = load_family("round")
    geom, _ = fit_sphere_geometry(ref, rows)
    pub = json.loads((CAL_OUT / "geometry.json").read_text())
    got = vars(geom)
    same = {k: got[k] == pub[k] for k in pub}
    print(f"  published  {pub}")
    print(f"  recomputed {got}")
    print(f"  exact equality per key: {same}")
    assert all(same.values()), "the default sphere fit changed"
    return got


# ------------------------------------------------------- geometry metrics
def radial_profile(depth, R_mm, cx, cy):
    """Reconstructed radial profile vs the analytic cap, about a GIVEN centre.

    `recon_study.sphere_check` re-centres on the reconstructed depth's own
    core, which is fine when the reconstruction is a clean dome and
    self-serving when it is not (a bilobed map recentres onto one lobe).
    Passing the centre of the contact circle detected in dI -- which no table
    touches -- makes the metric table-independent.
    """
    d = depth
    pk = float(d.max())
    if pk <= 1e-6:
        return None
    yy, xx = np.mgrid[0:d.shape[0], 0:d.shape[1]]
    r = np.hypot(xx - cx, yy - cy) * MM_PER_PIXEL
    m = r < 0.95 * R_mm
    nb = 28
    edges = np.linspace(0, float(r[m].max()), nb + 1)
    rc, hp = [], []
    for i in range(nb):
        sel = m & (r >= edges[i]) & (r < edges[i + 1])
        if sel.sum() >= 5:
            rc.append(0.5 * (edges[i] + edges[i + 1]))
            hp.append(float(np.median(d[sel])))
    if len(rc) < 6:
        return None
    rc, hp = np.array(rc), np.array(hp)
    cap = np.maximum(hp[0] - (R_mm - np.sqrt(
        np.clip(R_mm ** 2 - rc ** 2, 1e-9, None))), 0.0)
    # dip: h(centre)/h(max) -- 1.0 for a single dome, <1 for the bilobed
    # failure the Sparsh study caught
    return dict(r=rc, h=hp, cap=cap,
                rms=float(np.sqrt(np.mean((hp - cap) ** 2))),
                dip=float(hp[0] / max(hp.max(), 1e-9)), peak=pk)


def grad_angle(gx, gy, valid, cx, cy, R_mm):
    """Mean angle between the LUT gradient and the analytic sphere gradient.

    Chance is 90 deg; this is the statistic that exposed a foreign table at
    93.3 deg on Sparsh before any integration, so it separates a bad table
    from a bad solver.
    """
    yy, xx = np.mgrid[0:gx.shape[0], 0:gx.shape[1]]
    dx, dy = (xx - cx) * MM_PER_PIXEL, (yy - cy) * MM_PER_PIXEL
    r = np.hypot(dx, dy)
    m = valid & (r > 0.15 * R_mm) & (r < 0.9 * R_mm)
    if m.sum() < 50:
        return float("nan"), float("nan"), 0
    slope = -(r / np.sqrt(np.clip(R_mm ** 2 - r ** 2, 1e-9, None))) * MM_PER_PIXEL
    ax = np.where(r > 0, slope * dx / np.maximum(r, 1e-9), 0)
    ay = np.where(r > 0, slope * dy / np.maximum(r, 1e-9), 0)
    u = np.stack([gx[m], gy[m]], 1)
    v = np.stack([ax[m], ay[m]], 1)
    nu, nv = np.linalg.norm(u, axis=1), np.linalg.norm(v, axis=1)
    ok = (nu > 1e-9) & (nv > 1e-9)
    if ok.sum() < 30:
        return float("nan"), float("nan"), 0
    cos = np.clip((u[ok] * v[ok]).sum(1) / (nu[ok] * nv[ok]), -1, 1)
    ang = np.degrees(np.arccos(cos))
    return float(ang.mean()), float((ang < 30).mean()), int(ok.sum())


# ---------------------------------------------------------------- cmd: ref
def cmd_ref() -> dict:
    """How stable is the 50-frame rest-gel reference? It gates everything.

    Two numbers per family: the split-half p95 disagreement in grey levels
    (median of frames 0,2,4,... vs 1,3,5,...) and, more decisive, the
    fraction of pixels that would clear the pipeline's own |dI| > 8 valid
    test from reference noise alone. A p95 that is small next to 8 means the
    reference cannot manufacture contact.
    """
    out = {}
    for fam in FAMILIES:
        fr = initial_frames("selfmade", fam)
        st = np.stack([crop(np.asarray(Image.open(p).convert("RGB"))
                            ).astype(np.float32) for p in fr])
        a, b = np.median(st[0::2], 0), np.median(st[1::2], 0)
        dd = np.abs(a - b).max(2)
        # single-frame spread about the full median, for scale
        med = np.median(st, 0)
        per_frame = np.abs(st - med).max(3).reshape(len(st), -1)
        out[fam] = {
            "n_initial": len(fr),
            "halfsplit_p95": float(np.percentile(dd, 95)),
            "halfsplit_p999": float(np.percentile(dd, 99.9)),
            "halfsplit_max": float(dd.max()),
            "halfsplit_frac_over_valid_thr": float((dd > VALID_THR).mean()),
            "single_frame_p95_median": float(np.median(
                np.percentile(per_frame, 95, axis=1))),
            "ref_rgb_mean": [float(v) for v in med.reshape(-1, 3).mean(0)],
        }
        print(f"  {fam:11s} n={len(fr):3d}  half-split p95 {out[fam]['halfsplit_p95']:5.2f}"
              f"  p99.9 {out[fam]['halfsplit_p999']:5.2f}"
              f"  max {out[fam]['halfsplit_max']:6.1f}"
              f"  frac>|dI|{VALID_THR:g} {out[fam]['halfsplit_frac_over_valid_thr']*100:6.3f}%"
              f"  single-frame p95 {out[fam]['single_frame_p95_median']:5.2f}",
              flush=True)
    # commercial pad, for scale: it ships one frame for these families, so the
    # comparison is "50-frame median" vs "1 frame" -- reported, not hidden
    for fam in FAMILIES:
        n = len(initial_frames("commercial", fam))
        out.setdefault("_commercial_n_initial", {})[fam] = n
    CMP.mkdir(parents=True, exist_ok=True)
    (CMP / "reference_dispersion.json").write_text(json.dumps(out, indent=1))
    print(f"-> {CMP / 'reference_dispersion.json'}")
    return out


# ----------------------------------------------------------- cmd: modality
def contact_dI_rank(dI: np.ndarray, thr: float = 20.0) -> tuple | None:
    """Is the contact signal 3-channel (photometric) or 1-channel (shadow)?

    A photometric gel is lit by three differently coloured LEDs from three
    directions, so a slope changes the three channels by different amounts
    and the contact pixels fill a 3-D cloud in dI space. A gel that only
    occludes light darkens all three channels together: the cloud collapses
    onto one line, and a table from dI to a 2-D gradient is then degenerate
    by construction -- one scalar cannot name a direction.

    Two numbers, because either alone can be fooled: the PCA variance
    fractions of the contact dI cloud (rank), and the chromatic fraction
    |dI - mean_c(dI)| / |dI| (how much of the change is colour rather than
    brightness). A single straight-edge contact can look rank-1 on a good
    sensor because the true slope really does point one way -- the chromatic
    fraction does not have that failure mode, and pooling over many frames
    removes it from the rank too.
    """
    m = np.abs(dI).max(2) > thr
    if m.sum() < 200:
        return None
    X = dI[m] - dI[m].mean(0)
    s = np.linalg.svd(X, compute_uv=False)
    var = s ** 2 / (s ** 2).sum()
    ch = dI - dI.mean(2, keepdims=True)
    chroma = (np.percentile(np.abs(ch).max(2)[m], 95)
              / max(np.percentile(np.abs(dI).max(2)[m], 95), 1e-6))
    return [float(v) for v in var], float(chroma), int(m.sum())


def cmd_modality_pads(n_pad: int = 60) -> dict:
    """The two GlowTact pads only (cheap); React needs `modality`."""
    out = _modality_pads(n_pad)
    fp = CMP / "modality.json"
    old = json.loads(fp.read_text()) if fp.exists() else {}
    old.update(out)
    fp.write_text(json.dumps(old, indent=1))
    _print_modality(out)
    return out


def _print_modality(out: dict) -> None:
    for k, v in out.items():
        p = v["pca_median"]
        print(f"  {k:26s} n={v['n']:3d}  contact-dI PCA "
              f"{p[0]:.3f}/{p[1]:.3f}/{p[2]:.3f}  chromatic fraction "
              f"{v['chroma_median']:.3f}  rest-gel HSV "
              f"{v['ref_hsv'][0]:.0f}/{v['ref_hsv'][1]:.0f}/"
              f"{v['ref_hsv'][2]:.0f}", flush=True)


def _modality_pads(n_pad: int = 60) -> dict:
    out = {}
    for pad in ("selfmade", "commercial"):
        ref = reference(pad, "round")
        R = R_PAD[pad]
        # spread over the whole in-band depth range and the whole pad, not a
        # handful of centred presses: a single contact can look rank-1 on a
        # good sensor just because its slope points one way
        cand = [r for r in press_rows(pad, "round")
                if 0.15 <= r["z"] - Z0[pad] <= 0.9 * R]
        cand.sort(key=lambda q: (q["z"], q["x"], q["y"]))
        recs = []
        for i in np.linspace(0, len(cand) - 1, min(n_pad, len(cand))
                             ).astype(int):
            r = cand[int(i)]
            got = contact_dI_rank(load_img(r["path"]) - ref)
            if got:
                recs.append({"pca": got[0], "chroma": got[1], "z": r["z"]})
        pca = np.array([r["pca"] for r in recs])
        ch = np.array([r["chroma"] for r in recs])
        out[f"GlowTact {pad} pad"] = {
            "n": len(recs),
            "pca_median": [float(v) for v in np.median(pca, 0)],
            "pca_p10": [float(v) for v in np.percentile(pca, 10, axis=0)],
            "pca_p90": [float(v) for v in np.percentile(pca, 90, axis=0)],
            "chroma_median": float(np.median(ch)),
            "chroma_min": float(ch.min()), "chroma_max": float(ch.max()),
            "ref_hsv": _ref_hsv(ref)}
    return out


def cmd_modality(per_side: int = 8, n_pad: int = 60) -> dict:
    """Which sensor family is React actually in? Measured, not assumed.

    The hue argument for switching pads compares the REST gel colour. This
    compares what the pipeline actually consumes -- the contact difference
    image -- on all three sensors, over many frames each (60 presses per pad
    spread across depth and position, every React episode-side).
    """
    from .showcase import _react_context

    out = _modality_pads(n_pad)
    sides = react_sides()
    recs = []
    for i, (task, date, ep, side) in enumerate(sides):
        try:
            z, is_new, inten, _, frame, ref, h5 = _react_context(
                task, date, ep, side)
        except Exception as exc:                        # noqa: BLE001
            print(f"    skip {task}/{date}/{ep}_{side}: {exc}", flush=True)
            continue
        fresh = np.where(is_new)[0]
        per = []
        for row in fresh[np.argsort(-inten[fresh])][:per_side]:
            got = contact_dI_rank(crop(frame(int(row))).astype(np.float32)
                                  - ref)
            if got:
                per.append(got)
        h5.close()
        if per:
            recs.append({"side": f"{task}/{date}/{ep}_{side}", "n": len(per),
                         "pca": [float(v) for v in
                                 np.median([p[0] for p in per], 0)],
                         "chroma": float(np.median([p[1] for p in per])),
                         "ref_hsv": _ref_hsv(ref)})
        if (i + 1) % 12 == 0:
            print(f"    react {i+1}/{len(sides)}", flush=True)
    P = np.array([r["pca"] for r in recs])
    C = np.array([r["chroma"] for r in recs])
    out["React (all sides)"] = {
        "n": len(recs),
        "pca_median": [float(v) for v in np.median(P, 0)],
        "pca_p10": [float(v) for v in np.percentile(P, 10, axis=0)],
        "pca_p90": [float(v) for v in np.percentile(P, 90, axis=0)],
        "chroma_median": float(np.median(C)),
        "chroma_min": float(C.min()), "chroma_max": float(C.max()),
        "ref_hsv": [float(v) for v in
                    np.median([r["ref_hsv"] for r in recs], 0)],
        "per_side": recs}
    _print_modality(out)
    CMP.mkdir(parents=True, exist_ok=True)
    (CMP / "modality.json").write_text(json.dumps(out, indent=1))
    print(f"-> {CMP / 'modality.json'}")
    return out


def _ref_hsv(ref: np.ndarray) -> list[float]:
    h = cv2.cvtColor(np.clip(ref, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    m = np.median(h.reshape(-1, 3), 0)
    return [float(m[0]) * 2, float(m[1]), float(m[2])]     # H in degrees


# ---------------------------------------------------------- cmd: calibrate
def cmd_calibrate(z0_bounds=(0.0, 6.0)) -> dict:
    """Sphere fit + LUT on the self-made pad.

    `z0_bounds` is widened from the shipped default (+/-2 mm): the self-made
    subset logs commanded z from a datum ~3.3 mm lower than the commercial
    one (its z spans 1.1-7.9 mm and contact starts near 3.3), so with the
    default bracket the solver sits on the boundary, returns R^2 = 0 and
    builds an EMPTY table. That failure is silent unless the bin count is
    checked, which is why it is printed here.
    """
    ref = reference("selfmade", "round")
    rows = press_rows("selfmade", "round")
    print(f"  round: {len(rows)} press frames, reference = median of "
          f"{len(initial_frames('selfmade', 'round'))} rest frames", flush=True)
    geom, fits = fit_sphere_geometry(ref, rows, z0_bounds=z0_bounds)
    print(f"  sphere fit: R={geom.R_mm:.3f} mm  z0={geom.z0_mm:.3f} mm  "
          f"n={geom.n_used}/{len(rows)}  R^2={geom.fit_r2:.4f}", flush=True)
    for v, (lo, hi), nm in ((geom.z0_mm, z0_bounds, "z0"),
                            (geom.R_mm, (0.5, 15.0), "R")):
        if min(abs(v - lo), abs(v - hi)) < 1e-3:
            print(f"  WARNING: {nm} sits on its bound ({lo}, {hi}) — the fit "
                  f"did not converge to an interior optimum", flush=True)
    raw = build_lut(ref, fits, geom)
    lut = fill_lut_holes(raw["lut"], raw["count"])
    used = sum(1 for f in fits
               if 0.15 <= f["z"] - geom.z0_mm <= 0.9 * geom.R_mm)
    print(f"  LUT: {raw['filled_frac']*100:.2f}% of {BINS}^3 bins observed "
          f"({int((raw['count'] > 0).sum())} bins, "
          f"{int(raw['count'].sum())} pixels) from {used} frames", flush=True)
    CAL_OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(NEW_LUT, lut=lut, count=raw["count"],
                        R_mm=geom.R_mm, z0_mm=geom.z0_mm,
                        bins=BINS, di_range=DI_RANGE)
    meta = {"pad": "selfmade", "R_mm": geom.R_mm, "z0_mm": geom.z0_mm,
            "n_circles": geom.n_used, "n_rows": len(rows),
            "fit_r2": geom.fit_r2, "frames_in_lut": used,
            "observed_bin_frac": raw["filled_frac"],
            "observed_bins": int((raw["count"] > 0).sum()),
            "pixels": int(raw["count"].sum()),
            "n_reference_frames": len(initial_frames("selfmade", "round"))}
    CMP.mkdir(parents=True, exist_ok=True)
    (CMP / "selfmade_geometry.json").write_text(json.dumps(meta, indent=1))
    print(f"-> {NEW_LUT}")
    return meta


def cmd_degeneracy() -> dict:
    """Does one colour still name one slope on this pad?

    `build_lut` writes the MEAN target gradient of every pixel that landed in
    a colour bin. If a colour is produced by many different slopes, those
    targets point in different directions and cancel, so the stored vector is
    short compared with the individual targets that made it. The ratio

        |mean(g)| / sqrt(mean(|g|^2))            per bin, weighted by count

    is 1 when the colour determines the slope and 0 when it says nothing.
    This is measured on the SAME frames each table was built from, so it is a
    property of the sensor, not of a test set.
    """
    from scipy.stats import spearmanr           # noqa: F401  (kept explicit)

    out = {}
    for pad, key in (("commercial", "old_commercial"),
                     ("selfmade", "new_selfmade")):
        z = np.load(OLD_LUT if pad == "commercial" else NEW_LUT)
        R, z0 = float(z["R_mm"]), float(z["z0_mm"])
        ref = reference(pad, "round")
        rows = press_rows(pad, "round")
        ssum = np.zeros((BINS, BINS, BINS, 2), np.float64)
        sq = np.zeros((BINS, BINS, BINS), np.float64)
        cnt = np.zeros((BINS, BINS, BINS), np.int64)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        used = 0
        for r in rows:
            d = r["z"] - z0
            if not (0.15 <= d <= 0.9 * R):
                continue
            img = load_img(r["path"])
            det = detect_circle(img - ref)
            if det is None:
                continue
            cx, cy, a = det
            if not (18 + a < cx < W - 18 - a and 18 + a < cy < H - 18 - a):
                continue
            used += 1
            dI = img - ref
            rx, ry = xx - cx, yy - cy
            r_px = np.sqrt(rx ** 2 + ry ** 2)
            r_mm = r_px * MM_PER_PIXEL
            inside = (r_px < 0.97 * a) & (r_mm < 0.985 * R)
            denom = np.sqrt(np.clip(R ** 2 - r_mm ** 2, 1e-6, None))
            slope = -(r_mm / denom) * MM_PER_PIXEL
            gx = np.where(r_px > 1e-6, slope * rx / np.maximum(r_px, 1e-6), 0.)
            gy = np.where(r_px > 1e-6, slope * ry / np.maximum(r_px, 1e-6), 0.)
            q = np.clip((dI[inside] + DI_RANGE) / (2 * DI_RANGE) * (BINS - 1),
                        0, BINS - 1).astype(np.int32)
            idx = (q[:, 0], q[:, 1], q[:, 2])
            np.add.at(ssum, idx + (0,), gx[inside])
            np.add.at(ssum, idx + (1,), gy[inside])
            np.add.at(sq, idx, gx[inside] ** 2 + gy[inside] ** 2)
            np.add.at(cnt, idx, 1)
        have = cnt >= 10          # only bins with enough votes to average
        mean_mag = np.linalg.norm(ssum[have] / cnt[have, None], axis=1)
        rms_mag = np.sqrt(sq[have] / cnt[have])
        w = cnt[have].astype(float)
        ratio = float((mean_mag / np.maximum(rms_mag, 1e-12) * w).sum()
                      / w.sum())
        out[pad] = {"frames": used, "bins_ge10": int(have.sum()),
                    "pixels": int(cnt.sum()),
                    "coherence": ratio,
                    "coherence_unweighted": float(np.mean(
                        mean_mag / np.maximum(rms_mag, 1e-12)))}
        print(f"  {pad:11s} {used:4d} frames, {int(cnt.sum()):8d} px in "
              f"{int(have.sum()):6d} bins (>=10 votes): direction coherence "
              f"|mean g| / rms|g| = {ratio:.3f}", flush=True)
    CMP.mkdir(parents=True, exist_ok=True)
    (CMP / "lut_degeneracy.json").write_text(json.dumps(out, indent=1))
    print(f"-> {CMP / 'lut_degeneracy.json'}")
    return out


def load_luts() -> dict:
    out = {}
    for name, p in (("old_commercial", OLD_LUT), ("new_selfmade", NEW_LUT)):
        z = np.load(p)
        out[name] = {"lut": z["lut"], "count": z["count"],
                     "R_mm": float(z["R_mm"]), "z0_mm": float(z["z0_mm"])}
    return out


# --------------------------------------------------------------- cmd: geom
def _geom_frames(pad: str, n: int, seed: int = 0) -> list[dict]:
    """Clean, fully-interior sphere presses spanning the depth range."""
    rows = press_rows(pad, "round")
    rows = [r for r in rows if r["f"] > 0.3]
    rows.sort(key=lambda r: r["z"])
    idx = np.linspace(0, len(rows) - 1, min(n * 3, len(rows))).astype(int)
    return [rows[i] for i in idx]


def cmd_geom(pad: str = "selfmade", n: int = 120) -> dict:
    """Old vs new table on the SAME sphere presses of one pad.

    Both tables see the same frames and the same reference, so the only
    variable is the table. Two rules keep this honest:

    * the contact circle both metrics are measured about comes from dI
      (`detect_circle`), never from a reconstruction -- a bilobed depth map
      would otherwise re-centre onto one lobe and flatter itself;
    * the analytic sphere both tables are scored against uses the radius
      fitted on THIS pad, not the radius stored in each table. The indenter
      is a property of the data; scoring a table against its own R would let
      a wrong table pick a convenient target.
    """
    luts = load_luts()
    R_pad = R_PAD[pad]
    ref = reference(pad, "round")
    cand = _geom_frames(pad, n)
    res = {k: [] for k in luts}
    used = []
    for r in cand:
        if len(used) >= n:
            break
        img = load_img(r["path"])
        det = detect_circle(img - ref)
        if det is None:
            continue
        cx, cy, a = det
        if not (18 + a < cx < W - 18 - a and 18 + a < cy < H - 18 - a):
            continue
        used.append(dict(z=r["z"], f=r["f"], a_px=a))
        for k, L in luts.items():
            st = stages_lut(img, ref, L["lut"], L["count"])
            R = R_pad
            ang, frac30, npx = grad_angle(st["gx"], st["gy"], st["valid"],
                                          cx, cy, R)
            prof = radial_profile(st["depth"], R, cx, cy)
            res[k].append(dict(
                z=r["z"], f=r["f"], grad_angle=ang, frac_lt30=frac30,
                n_px=npx, peak=float(st["depth"].max()),
                unobserved=st["unobserved"],
                rms_um=(prof["rms"] * 1000) if prof else float("nan"),
                dip=prof["dip"] if prof else float("nan")))
        if len(used) % 20 == 0:
            print(f"    {len(used)}/{n} frames", flush=True)

    def med(v):
        v = np.array(v, float)
        v = v[np.isfinite(v)]
        return float(np.median(v)) if len(v) else float("nan")

    summary = {"pad": pad, "R_pad_mm": R_pad, "n_frames": len(used),
               "z_range_mm": [min(u["z"] for u in used),
                              max(u["z"] for u in used)]}
    for k, rr in res.items():
        summary[k] = {
            "R_mm": R_pad,
            "grad_angle_deg_median": med([x["grad_angle"] for x in rr]),
            "grad_angle_deg_p90": float(np.nanpercentile(
                [x["grad_angle"] for x in rr], 90)),
            "frac_within_30deg_median": med([x["frac_lt30"] for x in rr]),
            "profile_rms_um_median": med([x["rms_um"] for x in rr]),
            "profile_rms_um_p90": float(np.nanpercentile(
                [x["rms_um"] for x in rr], 90)),
            "centre_dip_median": med([x["dip"] for x in rr]),
            "peak_depth_mm_median": med([x["peak"] for x in rr]),
            "unobserved_frac_median": med([x["unobserved"] for x in rr]),
        }
        s = summary[k]
        print(f"  {k:16s} R={s['R_mm']:.2f}mm  grad-angle "
              f"{s['grad_angle_deg_median']:6.2f} deg (p90 "
              f"{s['grad_angle_deg_p90']:6.2f}, <30deg "
              f"{s['frac_within_30deg_median']*100:5.1f}%)  profile RMS "
              f"{s['profile_rms_um_median']:6.1f} um  dip "
              f"{s['centre_dip_median']:.3f}  peak "
              f"{s['peak_depth_mm_median']:.3f} mm  unobs "
              f"{s['unobserved_frac_median']*100:5.1f}%", flush=True)
    CMP.mkdir(parents=True, exist_ok=True)
    fp = CMP / f"geometry_old_vs_new_{pad}.json"
    fp.write_text(json.dumps({"summary": summary, "per_frame": res}, indent=1))
    print(f"-> {fp}")
    return summary


def cmd_geom_both() -> dict:
    """Both pads: the home control is what makes the new number readable."""
    return {p: cmd_geom(p) for p in ("selfmade", "commercial")}


# -------------------------------------------------------------- cmd: feats
_G: dict = {}


def _feat_init():
    _G.update(load_luts())


def _feat_one(job):
    pad, fam, path, z, f, x, y = job
    key = (pad, fam)
    if key not in _G:
        _G[key] = reference(pad, fam)
    ref = _G[key]
    img = load_img(Path(path))
    out = {"pad": pad, "fam": fam, "z": z, "f": f, "x": x, "y": y}
    for k in ("old_commercial", "new_selfmade"):
        st = stages_lut(img, ref, _G[k]["lut"], _G[k]["count"])
        out[k] = {**st["feats"], "unobserved": st["unobserved"],
                  "n_valid": st["n_valid"]}
    return out


def cmd_feats(pads=("selfmade", "commercial"), procs: int = 8) -> None:
    """Both tables' features on every controlled press of both pads."""
    from multiprocessing import Pool

    CMP.mkdir(parents=True, exist_ok=True)
    for pad in pads:
        jobs = []
        for fam in FAMILIES:
            for r in press_rows(pad, fam):
                jobs.append((pad, fam, str(r["path"]), r["z"], r["f"],
                             r["x"], r["y"]))
        print(f"  {pad}: {len(jobs)} frames x 2 tables", flush=True)
        rows = []
        with Pool(procs, initializer=_feat_init) as pool:
            for i, rec in enumerate(pool.imap_unordered(_feat_one, jobs, 16)):
                rows.append(rec)
                if (i + 1) % 1000 == 0:
                    print(f"    {i+1}/{len(jobs)}", flush=True)
        fp = CMP / f"features_{pad}.json"
        fp.write_text(json.dumps(rows))
        print(f"  -> {fp}  ({len(rows)} rows)", flush=True)


# -------------------------------------------------------------- cmd: force
def _basis(x, y):
    return np.column_stack([np.ones_like(x), x, y, x * x, y * y, x * y])


def force_matrix(rows, lut_key: str, scope: str = "published") -> dict:
    """`force_eval_all.ds_glowtact`, verbatim protocol, table as a parameter.

    Same spatial gain field fitted on `round`, same five features, same
    per-family half/half + isotonic, 5 seeds, same within-group shuffle
    control. Two scopes, because the published one is not table-independent:

    `published` reproduces `ds_glowtact` exactly -- contact fully in view is
    decided from the RECONSTRUCTED contact centroid and area, so a table that
    reconstructs a bloated halo disqualifies its own frames. That is fine for
    reporting one table's ceiling (it reproduces 0.9864 on the commercial
    pad) and useless for comparing two, because the two tables then score
    different frames. On the self-made pad the commercial table keeps ZERO
    frames under it.

    `physical` scopes on the ROBOT only -- commanded position interior and
    indentation depth d = z - z0(pad) in [0.15, 3.4] mm -- so all four
    pad x table cells score the identical press set and the table is the
    only variable.
    """
    from .force_eval_all import evaluate

    rows = [r for r in rows
            if r["f"] > 0.15 and np.isfinite(r[lut_key].get("cx", np.nan))]
    a = lambda k: np.array([r[k] for r in rows])                 # noqa: E731
    b = lambda k: np.array([r[lut_key][k] for r in rows])        # noqa: E731
    x, y, z, f = a("x"), a("y"), a("z"), a("f")
    V, V2, A, D = b("vol"), b("vol2"), b("area"), b("maxd")
    cx, cy = b("cx"), b("cy")
    grp = a("fam")
    m = (grp == "round") & (x > 3.5) & (x < 14.5) & (y > 3.0) & (y < 13.5)
    PHI = _basis(x[m], y[m])
    w, *_ = np.linalg.lstsq(np.hstack([PHI * z[m][:, None], -PHI]), D[m],
                            rcond=None)
    u = 1.0 / np.clip(_basis(x, y) @ w[:6], 0.15, 3.0)
    X = np.column_stack([V * u, V2 * u ** 2, D * u,
                         np.sqrt(np.clip(A, 0, None)) * D * u, A])
    interior = (x > 3.5) & (x < 14.5) & (y > 3.0) & (y < 13.5)
    if scope == "published":
        r_eff = np.sqrt(np.clip(A, 0, None) / np.pi)
        sc = ((cx - r_eff > 24) & (cx + r_eff < 296) & (cy - r_eff > 20)
              & (cy + r_eff < 220) & (z <= 4.2) & interior)
    else:
        d = z - Z0[rows[0]["pad"]]
        sc = interior & (d >= 0.15) & (d <= 3.4)
    if sc.sum() < 20:
        return {"rho": float("nan"), "mae": float("nan"),
                "rho_min": float("nan"), "rho_max": float("nan"),
                "rho_sd": float("nan"), "shuffle_rho": float("nan"),
                "n_eval": 0, "n_groups": 0, "per_group_rho": {},
                "note": f"only {int(sc.sum())} frames survive this scope"}
    return evaluate(X[sc], f[sc], grp[sc])


def cmd_force() -> dict:
    out = {}
    for scope in ("published", "physical"):
        for pad in ("selfmade", "commercial"):
            fp = CMP / f"features_{pad}.json"
            if not fp.exists():
                print(f"  (skip {pad}: run `feats` first)")
                continue
            rows = json.loads(fp.read_text())
            for key in ("old_commercial", "new_selfmade"):
                out[f"[{scope}] {pad} pad / {key} LUT"] = force_matrix(
                    rows, key, scope)
    print(f"\n{'scope / pad / LUT':56s} {'n':>6s} {'rho':>7s} "
          f"{'[min,max]':>15s} {'MAE [N]':>9s} {'shuffle':>8s}")
    for k, v in out.items():
        print(f"{k:56s} {v['n_eval']:6d} {v['rho']:7.4f} "
              f"[{v['rho_min']:.3f},{v['rho_max']:.3f}]".ljust(89)
              + f"{v['mae']:9.3f} {v['shuffle_rho']:8.3f}")
        if v["per_group_rho"]:
            print("      per-family: " + "  ".join(
                f"{g} {rr:.3f}" for g, rr in sorted(v["per_group_rho"].items())))
    (CMP / "force_old_vs_new.json").write_text(json.dumps(out, indent=1))
    print(f"-> {CMP / 'force_old_vs_new.json'}")
    return out


# -------------------------------------------------------------- cmd: react
def react_sides(limit: int | None = None) -> list[tuple[str, str, str, str]]:
    from .run_episode import DATA_ROOT, STAGE_ROOT

    out = []
    for npz in sorted(OUT_ROOT.rglob("episode_*_*.npz")):
        task, date = npz.parent.parent.name, npz.parent.name
        ep, side = npz.stem.rsplit("_", 1)
        if not (DATA_ROOT / task / date / f"{ep}.h5").exists():
            continue
        if not (STAGE_ROOT / task / "meta" / date / f"{ep}.parquet").exists():
            continue
        out.append((task, date, ep, side))
    return out[:limit] if limit else out


def cmd_react(per_side: int = 24) -> dict:
    """Out-of-gamut rate on React, old table vs new, over EVERY episode-side.

    Sampling: the `per_side` strongest fresh contacts of each side (strongest
    = largest valid-mask area), which is where the table is actually used and
    where an out-of-gamut colour costs the most. Reported per side and
    pooled, never as a single frame.
    """
    from .showcase import _react_context

    luts = load_luts()
    sides = react_sides()
    print(f"  {len(sides)} episode-sides with h5 + parquet", flush=True)
    per, allrec = [], []
    for si, (task, date, ep, side) in enumerate(sides):
        try:
            z, is_new, inten, _, frame, ref, h5 = _react_context(
                task, date, ep, side)
        except Exception as exc:                        # noqa: BLE001
            print(f"    skip {task}/{date}/{ep}_{side}: {exc}", flush=True)
            continue
        fresh = np.where(is_new)[0]
        # rank fresh rows by contact strength using intensity (the parquet's
        # own contact proxy), take the strongest `per_side`
        order = fresh[np.argsort(-inten[fresh])][:per_side]
        recs = []
        for row in order:
            img = crop(frame(int(row))).astype(np.float32)
            rec = {"task": task, "date": date, "ep": ep, "side": side,
                   "row": int(row)}
            for k, L in luts.items():
                st = stages_lut(img, ref, L["lut"], L["count"])
                rec[k] = {"unobserved": st["unobserved"],
                          "n_valid": st["n_valid"],
                          "peak_mm": float(st["depth"].max())}
                rec[f"_g_{k}"] = (st["gx"], st["gy"], st["valid"])
            # how differently do the two tables decode the SAME pixels?
            gx0, gy0, v0 = rec.pop("_g_old_commercial")
            gx1, gy1, v1 = rec.pop("_g_new_selfmade")
            m = v0 & v1
            if m.sum() > 50:
                u = np.stack([gx0[m], gy0[m]], 1)
                w = np.stack([gx1[m], gy1[m]], 1)
                nu = np.linalg.norm(u, axis=1)
                nw = np.linalg.norm(w, axis=1)
                ok = (nu > 1e-9) & (nw > 1e-9)
                if ok.sum() > 30:
                    cos = np.clip((u[ok] * w[ok]).sum(1) / (nu[ok] * nw[ok]),
                                  -1, 1)
                    ang = np.degrees(np.arccos(cos))
                    rec["disagree_deg_median"] = float(np.median(ang))
                    rec["disagree_frac_gt30"] = float((ang > 30).mean())
                    rec["mag_ratio_new_over_old"] = float(
                        np.median(nw[ok]) / max(np.median(nu[ok]), 1e-12))
            if rec[list(luts)[0]]["n_valid"] >= 200:    # real contact only
                recs.append(rec)
        h5.close()
        if not recs:
            continue
        row = {"side": f"{task}/{date}/{ep}_{side}", "n": len(recs)}
        for k in luts:
            row[k] = float(np.median([r[k]["unobserved"] for r in recs]))
            row[f"{k}_peak"] = float(np.median([r[k]["peak_mm"]
                                                for r in recs]))
        row["disagree_deg"] = float(np.median(
            [r.get("disagree_deg_median", np.nan) for r in recs]))
        per.append(row)
        allrec += recs
        print(f"    [{si+1}/{len(sides)}] {row['side']:44s} n={row['n']:3d}  "
              f"unobs old {row['old_commercial']*100:5.1f}%  new "
              f"{row['new_selfmade']*100:5.1f}%  decode-diff "
              f"{row['disagree_deg']:5.1f} deg", flush=True)

    def pool(k):
        v = np.array([r[k]["unobserved"] for r in allrec], float)
        return dict(median=float(np.median(v)), mean=float(v.mean()),
                    p90=float(np.percentile(v, 90)))

    summary = {"n_sides": len(per), "n_frames": len(allrec),
               "old_commercial": pool("old_commercial"),
               "new_selfmade": pool("new_selfmade"),
               "per_side_median_old": float(np.median(
                   [r["old_commercial"] for r in per])),
               "per_side_median_new": float(np.median(
                   [r["new_selfmade"] for r in per])),
               "decode_disagreement_deg_median": float(np.nanmedian(
                   [r.get("disagree_deg_median", np.nan) for r in allrec])),
               "per_side": per}
    CMP.mkdir(parents=True, exist_ok=True)
    (CMP / "react_out_of_gamut.json").write_text(json.dumps(summary, indent=1))
    print(f"\n  pooled over {len(allrec)} contact frames on {len(per)} sides:"
          f"\n    old (commercial-pad table) unobserved "
          f"{summary['old_commercial']['median']*100:.1f}% median / "
          f"{summary['old_commercial']['p90']*100:.1f}% p90"
          f"\n    new (self-made-pad table) unobserved "
          f"{summary['new_selfmade']['median']*100:.1f}% median / "
          f"{summary['new_selfmade']['p90']*100:.1f}% p90")
    print(f"-> {CMP / 'react_out_of_gamut.json'}")
    return summary


def cmd_host() -> dict:
    """The same out-of-gamut statistic each table gets on its OWN pad.

    Without this the React number has no scale: 18% is only alarming next to
    what the table scores at home.
    """
    luts = load_luts()
    out = {}
    for pad, key in (("commercial", "old_commercial"),
                     ("selfmade", "new_selfmade")):
        ref = reference(pad, "round")
        rows = [r for r in press_rows(pad, "round") if r["f"] > 0.3]
        idx = np.linspace(0, len(rows) - 1, 200).astype(int)
        vals = []
        for i in idx:
            img = load_img(rows[i]["path"])
            st = stages_lut(img, ref, luts[key]["lut"], luts[key]["count"])
            if st["n_valid"] >= 200:
                vals.append(st["unobserved"])
        out[f"{pad}/{key}"] = {"n": len(vals),
                               "median": float(np.median(vals)),
                               "p90": float(np.percentile(vals, 90))}
        print(f"  {pad:11s} pad + own table: unobserved "
              f"{out[f'{pad}/{key}']['median']*100:.1f}% median / "
              f"{out[f'{pad}/{key}']['p90']*100:.1f}% p90 (n={len(vals)})",
              flush=True)
    (CMP / "host_out_of_gamut.json").write_text(json.dumps(out, indent=1))
    return out


def cmd_calib_audit() -> dict:
    """Incidental finding, reported not fixed: the shipped React force model.

    `showcase._glowtact_calib` fits its weights on the cached
    `feature_cache/lut_full.json` and then predicts from
    `debug_gallery.stages()`. Those two are not the same features:

      * units -- the cache stores `vol` and `area` in PIXEL units (area
        6815 px for a press that `stages()` calls 10.86 mm^2, and
        10.86 / MM_PER_PIXEL^2 = 6301 px), while `stages()` returns mm^2.
        The five features pick up different powers of that factor, so it is
        not a rescale the isotonic stage can absorb;
      * vintage -- the same press has maxd 0.613 in the cache and 0.468
        through today's `stages()`, so the cache predates the current
        reconstruction as well.

    The symptom is not subtle. Run end-to-end through `stages()` on the very
    presses it was calibrated on (GlowTact round, in view, 0.19-4.75 N), the
    shipped model scores rho 0.143 / MAE 1.23 N and compresses its output to
    0.31-1.86 N, while the same model FORM refit on matching features gets
    rho 0.997 in sample. React's published newtons come through this path.

    Not fixed here: `showcase.py` is out of scope for this task, and
    re-fitting it changes numbers already published on the site.
    """
    from scipy.stats import spearmanr

    from . import debug_gallery as dg
    from .showcase import _glowtact_calib

    pub = _glowtact_calib()
    rows = json.loads((CMP / "features_commercial.json").read_text())
    keep = set()
    for r in rows:
        b = r["old_commercial"]
        if r["fam"] != "round" or not np.isfinite(b.get("cx", np.nan)):
            continue
        if not (3.5 < r["x"] < 14.5 and 3.0 < r["y"] < 13.5):
            continue
        if not (r["z"] - Z0["commercial"] <= 3.4 and 0.15 < r["f"] <= 8.0):
            continue
        re_ = np.sqrt(b["area"] / np.pi)
        if (b["cx"] - re_ > 24 and b["cx"] + re_ < 296
                and b["cy"] - re_ > 20 and b["cy"] + re_ < 220):
            keep.add((round(r["z"], 3), round(r["f"], 3)))
    ref = reference("commercial", "round")
    pred, truth = [], []
    for p in press_rows("commercial", "round"):
        if (round(p["z"], 3), round(p["f"], 3)) not in keep:
            continue
        pred.append(pub(dg.stages(load_img(p["path"]), ref)))
        truth.append(p["f"])
    pred, truth = np.array(pred), np.array(truth)
    out = {"n": len(pred), "rho": float(spearmanr(pred, truth).statistic),
           "mae_n": float(np.abs(pred - truth).mean()),
           "pred_range": [float(pred.min()), float(pred.max())],
           "true_range": [float(truth.min()), float(truth.max())]}
    print(f"  shipped _glowtact_calib on its own calibration objects "
          f"(n={out['n']}): rho {out['rho']:.3f}  MAE {out['mae_n']:.3f} N  "
          f"predicts {out['pred_range'][0]:.2f}-{out['pred_range'][1]:.2f} N "
          f"for a true {out['true_range'][0]:.2f}-"
          f"{out['true_range'][1]:.2f} N")
    (CMP / "shipped_calib_audit.json").write_text(json.dumps(out, indent=1))
    return out


# -------------------------------------------------------------- cmd: panel
def _calib_from(rows, lut_key: str, pad: str):
    """`showcase._glowtact_calib`, refit on a chosen pad+table feature table.

    Same form as the published model (sphere presses only, strictly in view,
    0-8 N, pixel-space spatial gain field, least squares + isotonic), but fit
    on features produced by the table it will be used with -- a model fit on
    one table and applied through another is not a comparison, it is a
    mismatch.
    """
    from sklearn.isotonic import IsotonicRegression

    rows = [r for r in rows
            if r["f"] > 0.15 and np.isfinite(r[lut_key].get("cx", np.nan))]
    a = lambda k: np.array([r[k] for r in rows])                 # noqa: E731
    b = lambda k: np.array([r[lut_key][k] for r in rows])        # noqa: E731
    x, y, z, f = a("x"), a("y"), a("z"), a("f")
    V, V2, A, D = b("vol"), b("vol2"), b("area"), b("maxd")
    cx, cy, grp = b("cx"), b("cy"), a("fam")
    m = (grp == "round") & (x > 3.5) & (x < 14.5) & (y > 3.0) & (y < 13.5)
    PHI = _basis(cx[m] / 100, cy[m] / 100)
    w, *_ = np.linalg.lstsq(np.hstack([PHI * z[m][:, None], -PHI]), D[m],
                            rcond=None)
    gain = w[:6]

    def u_at(px, py):
        return 1.0 / np.clip(_basis(np.atleast_1d(px / 100),
                                    np.atleast_1d(py / 100)) @ gain,
                             0.15, 3.0)

    u = u_at(cx, cy)
    X = np.column_stack([V * u, V2 * u ** 2, D * u,
                         np.sqrt(np.clip(A, 0, None)) * D * u, A])
    r_eff = np.sqrt(np.clip(A, 0, None) / np.pi)
    # depth band, not raw commanded z: the published `z <= 4.2` encodes "gel
    # not bottomed out" against the COMMERCIAL pad's datum (z0 = 0.81 mm).
    # Applied literally to the self-made pad (z0 = 4.12 mm) it keeps a single
    # frame, and the model is then fit on n=1 without erroring.
    sc = ((cx - r_eff > 24) & (cx + r_eff < 296) & (cy - r_eff > 20)
          & (cy + r_eff < 220) & (z - Z0[pad] <= 3.4) & (grp == "round")
          & (x > 3.5) & (x < 14.5) & (y > 3.0) & (y < 13.5) & (f <= 8.0))
    if sc.sum() < 20:
        raise SystemExit(f"{lut_key}: only {int(sc.sum())} calibration "
                         f"presses survive the scope — refusing to fit")
    wl, *_ = np.linalg.lstsq(X[sc], f[sc], rcond=None)
    iso = IsotonicRegression(out_of_bounds="clip").fit(X[sc] @ wl, f[sc])

    def predict(st: dict) -> float:
        ft = st["feats"]
        if ft["area"] < 1.0:
            return 0.0
        uu = float(u_at(ft["cx"], ft["cy"])[0])
        v = np.array([ft["vol"] * uu, ft["vol2"] * uu ** 2, ft["maxd"] * uu,
                      np.sqrt(max(ft["area"], 0.0)) * ft["maxd"] * uu,
                      ft["area"]])
        return float(max(0.0, iso.predict([float(v @ wl)])[0]))

    return predict, int(sc.sum())


def cmd_panel(task="motherboard", date="2026-05-10", ep="episode_000",
              side="left") -> dict:
    """React depth panel under each table, saved side by side in lut_compare/.

    This deliberately does NOT call `showcase.react_showcase`: that entry
    point copies its output over the published site assets, and which table
    the site should ship is exactly the question still open. Same three-press
    selection, same four columns (raw | |frame-ref| | LUT depth | Open3D
    mesh), same renderer.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .evaluate import median3_fresh
    from .o3d_view import has_display
    from .showcase import _react_context, mesh_view

    if not has_display():
        raise SystemExit('needs: xvfb-run -a -s "-screen 0 1400x1000x24" ...')
    luts = load_luts()
    # each table's force model is fit on ITS OWN home pad, which is the
    # fairest version of the comparison the site would actually ship
    home = {"old_commercial": "commercial", "new_selfmade": "selfmade"}
    feat = {k: json.loads((CMP / f"features_{p}.json").read_text())
            for k, p in home.items()}
    calib = {k: _calib_from(feat[k], k, home[k]) for k in luts}
    for k, (_, n) in calib.items():
        print(f"  {k}: force model fit on {n} in-view sphere presses",
              flush=True)

    z, is_new, _, _, frame, ref, h5 = _react_context(task, date, ep, side)
    rows_all = list(range(len(is_new)))
    out = {}
    CMP.mkdir(parents=True, exist_ok=True)
    for k, L in luts.items():
        pred = calib[k][0]
        force = np.zeros(len(is_new))
        last, peaks, unobs = 0.0, [], []
        for row in rows_all:
            if is_new[row] or row == 0:
                st = stages_lut(crop(frame(row)).astype(np.float32), ref,
                                L["lut"], L["count"])
                last = pred(st)
                # scalars only: keeping every stage dict of a full episode is
                # ~1 GB of dI arrays
                peaks.append(float(st["depth"].max()))
                unobs.append(st["unobserved"])
            force[row] = last
        force = median3_fresh(force, is_new)
        order = np.argsort(-force)
        picks, taken = [], []
        for r in order:
            if is_new[r] and all(abs(int(r) - t) > 150 for t in taken):
                picks.append(int(r)); taken.append(int(r))
            if len(picks) == 3:
                break
        fig, axes = plt.subplots(len(picks), 4,
                                 figsize=(11.2, 2.15 * len(picks)))
        axes = np.atleast_2d(axes)
        stats = []
        for i, row in enumerate(picks):
            img = crop(frame(row)).astype(np.float32)
            st = stages_lut(img, ref, L["lut"], L["count"])
            diff = _V.diff_rgb(img, ref)
            stats.append({"row": row, "force_n": float(force[row]),
                          "peak_mm": float(st["depth"].max()),
                          "unobserved": st["unobserved"],
                          "n_valid": st["n_valid"],
                          "area_px": st["feats"]["area"]})
            for c, (data, cmap, title) in enumerate((
                    (np.clip(img, 0, 255).astype(np.uint8), None,
                     f"row {row}  F={force[row]:.2f} N"),
                    (diff, "gray", "|frame - reference|"),
                    (st["depth"], "inferno",
                     f"LUT depth (max {st['depth'].max():.2f} mm)"),
                    (mesh_view(st["depth"]), None,
                     "3D reconstruction (Open3D mesh)"))):
                ax = axes[i, c]
                ax.imshow(data, cmap=cmap)
                ax.set_title(title, fontsize=8)
                ax.axis("off")
        fig.suptitle(f"{task}/{ep} {side} - {k} LUT", fontsize=10)
        fp = CMP / f"depth_validation_panel_{k}.png"
        fig.tight_layout(); fig.savefig(fp, dpi=130); plt.close(fig)
        out[k] = {"panel": str(fp), "rows": picks, "stats": stats,
                  "force_max": float(force.max()),
                  "peak_mm_median_fresh": float(np.median(peaks)),
                  "peak_mm_p99_fresh": float(np.percentile(peaks, 99)),
                  "unobserved_median_fresh": float(np.nanmedian(unobs)),
                  "n_fresh": len(peaks)}
        print(f"  {k}: rows {picks}  force max {force.max():.2f} N  "
              f"peak(fresh) {out[k]['peak_mm_median_fresh']:.3f} mm  "
              f"unobs(fresh) {out[k]['unobserved_median_fresh']*100:.1f}%",
              flush=True)
    h5.close()
    (CMP / "react_panel_stats.json").write_text(json.dumps(out, indent=1))
    print(f"-> {CMP}")
    return out


CMDS = {"check": cmd_check, "defaults": cmd_defaults,
        "ref": cmd_ref, "modality": cmd_modality,
        "modality_pads": cmd_modality_pads,
        "calibrate": cmd_calibrate, "degeneracy": cmd_degeneracy,
        "geom": cmd_geom_both, "feats": cmd_feats, "force": cmd_force,
        "react": cmd_react, "host": cmd_host, "panel": cmd_panel,
        "calib_audit": cmd_calib_audit}


if __name__ == "__main__":
    for c in (sys.argv[1:] or ["ref"]):
        print(f"== {c}", flush=True)
        CMDS[c]()
