"""Does removing the marker dots before photometric reconstruction help?

FEATS is our only marker-dot gel and our worst dataset (rho 0.772); the three
markerless sets score 0.94-0.99. The debug gallery shows *why* this could be
the markers' fault: the |dI|>8 valid mask is speckled by the dots, the LUT
gradient panel rings at every dot, and the integrated depth carries a dimple
grid. Markers occlude the gel, so the photometric LUT has no valid colour
there and the Poisson solve integrates garbage.

Prior art implemented here (GelSight Wedge, Wang/She/Dong/Adelson ICRA 2021,
and the standard GelSight marker pipelines): detect the dark dots on the
REFERENCE frame (they are static there), build + dilate a mask, then either
inpaint the images before differencing, or drop the markers out of the
gradient field and let the Poisson solve fill the holes harmonically.

Strategies compared, all on identical frames/splits:
  base           current pipeline (debug_gallery.stages)
  img_telea      cv2.inpaint Telea on ref AND frame, then dI
  img_ns         cv2.inpaint Navier-Stokes on ref AND frame
  grad_zero      raw dI, but g:=0 on the marker mask -> div g = 0 inside the
                 holes -> fast_poisson interpolates harmonically across them
  grad_inpaint   raw dI, gx/gy inpainted (Telea) from surrounding valid pixels
  img+grad       img_telea followed by grad_zero

Controls that could falsify a gain:
  rand_mask      inpaint the SAME NUMBER of fake markers at random non-marker
                 positions (if this reproduces the gain, it is just smoothing)
  markerless     run detector+inpaint on GlowTact / cnc (no dots): a gain
                 there means the detector is a blur, not a marker remover

Geometry check independent of the force fit: power of the depth map at the
marker lattice frequency (1/31.9 px) relative to its spectral shoulders.

RESULT — marker removal fixes the GEOMETRY and does NOT fix the FORCE.
Do not adopt it for the force pipeline; keep it if the depth map is the
product. That is exactly how it shipped: `marker_removal.stages_depth`
wraps (never edits) `debug_gallery.stages` and is used only where the site
shows depth or a 3D mesh; the force features on every dataset still come
from the untouched `stages()`.

1. The detector is good and specific. 63/63 dots on the FEATS reference,
   0 rejects, the same 63 for every threshold in 3..16 grey levels; radius
   6.3-7.4 px, pitch 31.9 px, dilated mask = 20.6% of the frame. On the two
   MARKERLESS references (GlowTact, cnc) it returns exactly 0 blobs, so it
   is not a generic blob finder. It does not eat the contact either: the
   per-frame detector still returns 63 at F = 60 N. But the dots MOVE with
   the gel (median 1.7 px vs the reference grid, >4 px on 20% of frames,
   >8 px on 8%), so a static reference mask leaves 4-15% of the contact
   area covered by an unmasked, shifted dot.

2. Force, per-group half/half + isotonic, 5 seeds, 186 eval frames.
   Baseline rho 0.7747 / MAE 5.03 N, seed sd 0.029 — nothing clears it:

       img_telea    0.7371 / 4.93     grad_zero     0.7620 / 5.12
       img_ns       0.7368 / 5.02     grad_zero_pf  0.7164 / 5.59
       img_telea_pf 0.7313 / 5.31     grad_inpaint  0.7408 / 5.01
       img+grad     0.7708 / 5.14     img+grad_pf   0.7323 / 5.44

   Paired on identical splits the best variant wins 2 of 5 seeds and every
   median delta is negative. Removing the markers costs 0.00-0.06 rho.

3. Geometry, 120 frames above 1 N, paired per frame — this part works:
       base 1.523  ->  img_telea 0.890 (x0.65 of base, lower on 91% of
       frames, Wilcoxon p 2.6e-19), img_telea_pf 0.860 (x0.58, 92%),
       grad_inpaint 0.924 (x0.64, 92%).
   Gradient ZEROING barely helps (grad_zero 1.251, x0.93, only 61% of
   frames): g:=0 in the holes is not a clean harmonic fill, it puts a
   dipole layer on every hole boundary at exactly the lattice frequency.
   Inpainting beats masking, and the dimple grid is real and removable.

4. Both falsification controls behave: see cmd_controls.

5. What actually caps FEATS is not the dots (cmd_reference): on the 20
   LIGHTEST presses |dI| already sits at 11 grey levels off-dot and 88% of
   off-dot pixels pass the |dI|>8 valid test, so "valid" is nearly the whole
   frame and the depth features integrate reference mismatch rather than
   contact. Swapping in a per-indenter light-press reference does not fix
   it either (0.7747 -> 0.7261).

Run:
  python -m force_recovery.marker_study detect     # task 1: detector figure
  python -m force_recovery.marker_study force      # tasks 2/3: rho + MAE
  python -m force_recovery.marker_study stats      # paired per-seed deltas
  python -m force_recovery.marker_study geometry   # task 4: dimple power
  python -m force_recovery.marker_study controls   # task 5
  python -m force_recovery.marker_study reference  # what the real cap is
  python -m force_recovery.marker_study all
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression

sys.path.append(str(Path(__file__).resolve().parents[1]))
from force_recovery.debug_gallery import (  # noqa: E402
    LUT, CNT, load_feats, load_glowtact, load_cnc)
from force_recovery.lut_calibration import (  # noqa: E402
    BINS, DI_RANGE, MM_PER_PIXEL)
# the detector / inpainter / geometry statistic now live in the pipeline
# module that ships the adopted step; this file keeps only the STUDY around
# them, so there is exactly one implementation of each.
from force_recovery.marker_removal import (  # noqa: E402,F401
    detect_markers, dimple_power, inpaint_img)

OUT = Path("/media/yxma/Disk1/twm/force_recovery/marker_study")
FE = ("vol", "vol2", "maxd", "area", "h1")
SEEDS = 5


def _fast_poisson():
    sys.path.insert(0, str(Path.home() / "gelsight_heightmap_reconstruction"
                           / "python_version"))
    from fast_poisson import fast_poisson
    return fast_poisson


# ------------------------------------------------------------------ controls
def random_mask(shape, n, radius, real_mask, seed=0, target_cov=None):
    """Control: n fake markers at random non-marker spots.

    `target_cov` area-matches the control to the real dilated marker mask —
    without it the control inpaints less of the frame than the real one and
    the comparison is not apples to apples.
    """
    H, W = shape[:2]

    def draw(rad):
        rng = np.random.default_rng(seed)
        out = np.zeros((H, W), np.uint8)
        far = cv2.dilate(real_mask.astype(np.uint8),
                         np.ones((2 * int(rad) + 5,) * 2, np.uint8)).astype(bool)
        placed = 0
        for _ in range(50000):
            if placed >= n:
                break
            y = int(rng.integers(rad + 2, H - rad - 2))
            x = int(rng.integers(rad + 2, W - rad - 2))
            if far[y, x]:
                continue
            cv2.circle(out, (x, y), int(round(rad)), 1, -1)
            placed += 1
        return out.astype(bool), placed

    if target_cov is None:
        return draw(radius)
    best = None
    for rad in np.arange(radius, radius * 2.2, 0.25):
        m, placed = draw(rad)
        if best is None or abs(m.mean() - target_cov) < abs(best[0].mean()
                                                            - target_cov):
            best = (m, placed)
        if m.mean() >= target_cov:
            break
    return best


# ------------------------------------------------------------------ pipeline
def stages_m(img, ref, marker=None, mode="base", inpaint="telea", m_img=None):
    """debug_gallery.stages() with a marker-removal hook.

    mode: base | img | grad_zero | grad_inpaint | img_grad
    `marker` masks the reference, `m_img` (optional) the current frame — the
    dots move with the gel, so a per-frame mask is not the reference mask.
    The base branch is byte-identical in behaviour to debug_gallery.stages().
    """
    fast_poisson = _fast_poisson()
    m_ref = marker
    if m_img is None:
        m_img = m_ref
    both = None if m_ref is None else (m_ref | m_img)
    if mode in ("img", "img_grad") and m_ref is not None:
        img = inpaint_img(img, m_img, inpaint)
        ref = inpaint_img(ref, m_ref, inpaint)
    dI = img - ref
    q = np.clip((dI + DI_RANGE) / (2 * DI_RANGE) * (BINS - 1),
                0, BINS - 1).astype(np.int32)
    g = LUT[q[..., 0], q[..., 1], q[..., 2]].copy()
    observed = CNT[q[..., 0], q[..., 1], q[..., 2]] > 0
    mag = cv2.GaussianBlur(np.abs(dI).max(2), (5, 5), 1.5)
    valid = mag > 8.0
    valid = cv2.morphologyEx(valid.astype(np.uint8), cv2.MORPH_OPEN,
                             np.ones((3, 3), np.uint8)).astype(bool)
    g[~valid] = 0.0
    if both is not None and mode in ("grad_zero", "img_grad"):
        # div g = 0 inside the holes -> the Poisson solve is harmonic there,
        # i.e. it interpolates the surface across the occluded dots.
        g[both] = 0.0
    elif both is not None and mode == "grad_inpaint":
        mk = both.astype(np.uint8)
        for c in (0, 1):
            lo, hi = float(g[..., c].min()), float(g[..., c].max())
            sc = 254.0 / max(hi - lo, 1e-6)
            u8 = np.clip((g[..., c] - lo) * sc, 0, 255).astype(np.uint8)
            g[..., c] = cv2.inpaint(u8, mk, 5, cv2.INPAINT_TELEA
                                    ).astype(np.float32) / sc + lo
    depth = fast_poisson(g[..., 0], g[..., 1])
    if depth[valid].size and np.median(depth[valid]) < 0:
        depth = -depth
    d = np.maximum(depth, 0.0)
    m = d > 0.05
    px = MM_PER_PIXEL ** 2
    feats = {"vol": float(d[m].sum() * px), "vol2": float((d[m] ** 2).sum() * px),
             "maxd": float(np.percentile(d, 99.8)), "area": float(m.sum() * px)}
    feats["h1"] = np.sqrt(feats["area"]) * feats["maxd"]
    return {"dI": dI, "gmag": np.hypot(g[..., 0], g[..., 1]), "valid": valid,
            "depth": d, "feats": feats,
            "lut_coverage": float(observed[valid].mean()) if valid.any() else 0.}


# pf=True -> detect the dots on the CURRENT frame too (they move with the gel)
STRATEGIES = {
    "base":          dict(mode="base"),
    "img_telea":     dict(mode="img", inpaint="telea"),
    "img_ns":        dict(mode="img", inpaint="ns"),
    "img_telea_pf":  dict(mode="img", inpaint="telea", pf=True),
    "grad_zero":     dict(mode="grad_zero"),
    "grad_zero_pf":  dict(mode="grad_zero", pf=True),
    "grad_inpaint":  dict(mode="grad_inpaint"),
    "img+grad":      dict(mode="img_grad", inpaint="telea"),
    "img+grad_pf":   dict(mode="img_grad", inpaint="telea", pf=True),
}


def run_strategy(img, ref, mask, name, pf_cache=None):
    kw = dict(STRATEGIES[name])
    pf = kw.pop("pf", False)
    if name == "base":
        return stages_m(img, ref, None, **kw)
    m_img = None
    if pf:
        if pf_cache is None:
            m_img, _, _ = detect_markers(img)
        else:
            m_img = pf_cache
    return stages_m(img, ref, mask, m_img=m_img, **kw)


# ------------------------------------------------------------------ eval
def fit_eval(rows, seeds=SEEDS):
    """Per-group half/half least squares on 5 features + isotonic, n seeds."""
    X = np.array([[r[k] for k in FE] for r in rows])
    f = np.array([r["f"] for r in rows])
    grp = np.array([r["group"] for r in rows])
    rr, mm = [], []
    for s in range(seeds):
        pred = np.full(len(f), np.nan)
        for gi, gname in enumerate(sorted(set(grp))):
            idx = np.where(grp == gname)[0]
            # deterministic across processes (str hash is salted in python3)
            idx = idx[np.random.default_rng(1000 * s + 7 * gi)
                      .permutation(len(idx))]
            half = len(idx) // 2
            fi, ei = idx[:half], idx[half:]
            if len(fi) < 8:
                continue
            w, *_ = np.linalg.lstsq(X[fi], f[fi], rcond=None)
            iso = IsotonicRegression(out_of_bounds="clip").fit(X[fi] @ w, f[fi])
            pred[ei] = iso.predict(X[ei] @ w)
        ok = np.isfinite(pred)
        rr.append(spearmanr(pred[ok], f[ok]).statistic)
        mm.append(float(np.abs(pred[ok] - f[ok]).mean()))
    return (float(np.median(rr)), float(np.median(mm)),
            float(np.std(rr)), int(ok.sum()))


# ------------------------------------------------------------------ commands
def _feats_setup():
    frames, get = load_feats()
    _, ref = get(frames[0])
    mask, cent, info = detect_markers(ref)
    return frames, get, ref, mask, cent, info


def cmd_detect():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUT.mkdir(parents=True, exist_ok=True)
    frames, get, ref, mask, cent, info = _feats_setup()
    print(f"detector on FEATS reference: {info}")

    # threshold sweep -> is the count stable, i.e. is the detector a knob?
    for thr in (3, 5, 8, 12, 16):
        _, _, i2 = detect_markers(ref, thr=thr)
        print(f"  thr={thr:2d}  n={i2['n']:3d} rejected={i2['rejected']} "
              f"radius={i2['radius_med']:.2f}px coverage={i2['coverage']*100:.1f}%")

    # false positives on the contact: how much of the CONTACT region would a
    # per-frame detector eat, and how much does the static ref mask overlap it?
    fs = np.array([fr["f"] for fr in frames])
    picks = np.argsort(fs)[np.linspace(0, len(fs) - 1, 6).astype(int)]
    fp_rows = []
    for i in picks:
        img, _ = get(frames[i])
        contact = cv2.GaussianBlur(np.abs(img - ref).max(2), (5, 5), 1.5) > 8.0
        pf_mask, _, pf_info = detect_markers(img)     # per-frame detector
        fp_rows.append({
            "f": float(fs[i]), "contact_frac": float(contact.mean()),
            "ref_mask_in_contact": float((mask & contact).sum()
                                         / max(contact.sum(), 1)),
            "perframe_n": pf_info["n"],
            "perframe_extra_in_contact": float(((pf_mask & ~mask) & contact).sum()
                                               / max(contact.sum(), 1))})
    print("\nfalse-positive audit (per-frame detector would eat the contact):")
    for r in fp_rows:
        print(f"  F={r['f']:6.2f}N contact={r['contact_frac']*100:5.1f}% of frame"
              f" | static ref mask covers {r['ref_mask_in_contact']*100:5.1f}%"
              f" of contact | per-frame detector n={r['perframe_n']:3d},"
              f" EXTRA {r['perframe_extra_in_contact']*100:5.1f}% of contact")

    # do the dots stay put? (they shear with the gel; a static mask then
    # inpaints bare gel and leaves the real dot in place)
    from scipy.spatial import cKDTree
    tree = cKDTree(cent)
    ns, off = [], []
    for fr in frames[:150]:
        im, _ = get(fr)
        _, c, i2 = detect_markers(im)
        ns.append(i2["n"])
        if len(c):
            off.append(float(np.median(tree.query(c)[0])))
    ns, off = np.array(ns), np.array(off)
    print(f"\nper-frame detector on 150 frames: n in [{ns.min()},{ns.max()}], "
          f"median {np.median(ns):.0f}")
    print(f"  dot displacement vs reference grid: median {np.median(off):.2f}px,"
          f"  >4px on {(off>4).mean()*100:.0f}% of frames,"
          f"  >8px on {(off>8).mean()*100:.0f}%")

    fig, ax = plt.subplots(6, 4, figsize=(15, 17))
    for k, i in enumerate(picks):
        img, _ = get(frames[i])
        inp = inpaint_img(img, mask, "telea")
        ov = img.copy()
        ov[mask] = [255, 0, 0]
        for a, (im, t) in zip(ax[k], [
                (img / 255, f"raw  F={fs[i]:.2f}N"),
                (ov / 255, f"marker mask (n={info['n']}, dilated)"),
                (inp / 255, "inpainted (Telea)"),
                (np.clip((img - inp) * 4 + 128, 0, 255) / 255,
                 "removed content (x4)")]):
            a.imshow(im)
            a.set_title(t, fontsize=9)
            a.axis("off")
    fig.suptitle("FEATS marker detection + inpainting", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "detector.png", dpi=95, bbox_inches="tight")
    plt.close(fig)
    json.dump({"info": info, "fp": fp_rows}, open(OUT / "detector.json", "w"),
              indent=1)
    print(f"\n-> {OUT/'detector.png'}")


def _run_strategies(frames, get, mask, names, spacing=None, tag="feats"):
    """Feature (and optional dimple-power) pass for each strategy."""
    res = {n: [] for n in names}
    dp = {n: [] for n in names}
    need_pf = any(STRATEGIES[n].get("pf") for n in names)
    for j, fr in enumerate(frames):
        img, ref = get(fr)
        pfm = detect_markers(img)[0] if need_pf else None
        for n in names:
            st = run_strategy(img, ref, mask, n, pf_cache=pfm)
            res[n].append({"group": fr["group"], "f": fr["f"], **st["feats"]})
            if spacing:
                dp[n].append(dimple_power(st["depth"], spacing))
        if (j + 1) % 100 == 0:
            print(f"    {tag}: {j+1}/{len(frames)}", flush=True)
    return res, dp


def cmd_force():
    OUT.mkdir(parents=True, exist_ok=True)
    frames, get, ref, mask, cent, info = _feats_setup()
    names = list(STRATEGIES)
    print(f"FEATS n={len(frames)} markers={info['n']} "
          f"coverage={info['coverage']*100:.1f}% spacing={info['spacing']:.1f}px")
    res, dp = _run_strategies(frames, get, mask, names,
                              spacing=info["spacing"])
    out = {}
    print(f"\n{'strategy':14s} {'rho':>7s} {'MAE(N)':>8s} {'d rho':>7s}"
          f" {'dimple':>7s}")
    base_rho = None
    for n in names:
        rho, mae, sd, ne = fit_eval(res[n])
        base_rho = rho if n == "base" else base_rho
        out[n] = {"rho": rho, "mae": mae, "rho_sd": sd, "n_eval": ne,
                  "dimple": float(np.median(dp[n]))}
        print(f"{n:14s} {rho:7.4f} {mae:8.3f} {rho-base_rho:+7.4f}"
              f" {np.median(dp[n]):7.3f}")
    json.dump({"info": info, "res": out,
               "features": {n: res[n] for n in names}},
              open(OUT / "force.json", "w"))
    print(f"-> {OUT/'force.json'}")


def cmd_stats():
    """Paired read of force.json: same splits, so compare seed-by-seed."""
    d = json.load(open(OUT / "force.json"))
    feats = d["features"]
    per = {}
    for n, rows in feats.items():
        X = np.array([[r[k] for k in FE] for r in rows])
        f = np.array([r["f"] for r in rows])
        grp = np.array([r["group"] for r in rows])
        rr, mm = [], []
        for s in range(SEEDS):
            pred = np.full(len(f), np.nan)
            for gi, gname in enumerate(sorted(set(grp))):
                idx = np.where(grp == gname)[0]
                idx = idx[np.random.default_rng(1000 * s + 7 * gi)
                          .permutation(len(idx))]
                h = len(idx) // 2
                fi, ei = idx[:h], idx[h:]
                if len(fi) < 8:
                    continue
                w, *_ = np.linalg.lstsq(X[fi], f[fi], rcond=None)
                iso = IsotonicRegression(out_of_bounds="clip").fit(
                    X[fi] @ w, f[fi])
                pred[ei] = iso.predict(X[ei] @ w)
            ok = np.isfinite(pred)
            rr.append(spearmanr(pred[ok], f[ok]).statistic)
            mm.append(float(np.abs(pred[ok] - f[ok]).mean()))
        per[n] = (np.array(rr), np.array(mm))
    b = per["base"][0]
    print("paired per-seed rho (identical splits), 5 seeds:")
    print(f"  {'strategy':14s} {'rho med':>8s} {'d rho med':>10s} "
          f"{'d rho all seeds':>22s}  win")
    for n, (rr, mm) in per.items():
        dd = rr - b
        print(f"  {n:14s} {np.median(rr):8.4f} {np.median(dd):+10.4f} "
              f"{np.array2string(dd, precision=3, floatmode='fixed'):>22s}"
              f"  {int((dd>0).sum())}/5")
    print(f"\n  baseline rho across seeds: "
          f"{np.array2string(b, precision=3, floatmode='fixed')}"
          f"  (sd {b.std():.3f}) -> a real effect must exceed ~0.03")


def cmd_geometry():
    """Dimple power in more detail + a spectrum figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUT.mkdir(parents=True, exist_ok=True)
    frames, get, ref, mask, cent, info = _feats_setup()
    sp = info["spacing"]
    sel = [fr for fr in frames if fr["f"] > 1.0][:120]
    names = ["base", "img_telea", "img_telea_pf", "grad_zero", "grad_zero_pf",
             "grad_inpaint", "img+grad_pf"]
    prof = {n: [] for n in names}
    vals = {n: [] for n in names}
    for j, fr in enumerate(sel):
        img, r = get(fr)
        pfm = detect_markers(img)[0]
        for n in names:
            st = run_strategy(img, r, mask, n, pf_cache=pfm)
            vals[n].append(dimple_power(st["depth"], sp))
            d = st["depth"]
            H, W = d.shape
            win = np.outer(np.hanning(H), np.hanning(W))
            P = np.abs(np.fft.fftshift(np.fft.fft2((d - d.mean()) * win))) ** 2
            fy = np.fft.fftshift(np.fft.fftfreq(H))[:, None]
            fx = np.fft.fftshift(np.fft.fftfreq(W))[None, :]
            fr_ = np.hypot(fy, fx)
            bins = np.linspace(0, 0.12, 60)
            k = np.digitize(fr_.ravel(), bins)
            prof[n].append([P.ravel()[k == b].mean() if (k == b).any() else np.nan
                            for b in range(1, len(bins))])
        if (j + 1) % 40 == 0:
            print(f"  geometry {j+1}/{len(sel)}", flush=True)
    bins = np.linspace(0, 0.12, 60)
    ctr = 0.5 * (bins[1:] + bins[:-1])
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    print(f"\ndimple power ratio at f0=1/{sp:.1f}px "
          f"(1.0 = no marker signature), n={len(sel)} frames F>1N")
    for n in names:
        p = np.nanmedian(np.array(prof[n]), 0)
        ax[0].loglog(ctr, p, label=n)
        v = np.array(vals[n])
        print(f"  {n:14s} median {np.median(v):6.3f}  mean {v.mean():6.3f}"
              f"  p90 {np.percentile(v,90):6.3f}")
        ax[1].hist(v, bins=np.linspace(0, max(3, np.percentile(v, 99)), 40),
                   histtype="step", label=n)
    for a, t in ((ax[0], "radial power spectrum of depth"),
                 (ax[1], "dimple power ratio per frame")):
        a.axvline(1 / sp, color="r", ls="--", lw=1) if a is ax[0] else None
        a.set_title(t)
        a.legend(fontsize=8)
    ax[0].set_xlabel("cycles / pixel")
    fig.tight_layout()
    fig.savefig(OUT / "geometry.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    json.dump({k: list(map(float, v)) for k, v in vals.items()},
              open(OUT / "geometry.json", "w"))
    print(f"-> {OUT/'geometry.png'}")


def cmd_controls():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}

    # (ii) random fake markers on FEATS: same count, same radius, elsewhere
    frames, get, ref, mask, cent, info = _feats_setup()
    rm, placed = random_mask(ref.shape, info["n"], info["radius_med"], mask,
                             target_cov=info["coverage"])
    print(f"random-mask control: {placed} fake markers, "
          f"coverage {rm.mean()*100:.1f}% vs real {mask.mean()*100:.1f}% "
          f"(area-matched)")
    r_base, r_rand = [], []
    rows_b, rows_r, rows_t = [], [], []
    for j, fr in enumerate(frames):
        img, r = get(fr)
        rows_b.append({"group": fr["group"], "f": fr["f"],
                       **stages_m(img, r, None, mode="base")["feats"]})
        rows_r.append({"group": fr["group"], "f": fr["f"],
                       **stages_m(img, r, rm, mode="img")["feats"]})
        rows_t.append({"group": fr["group"], "f": fr["f"],
                       **stages_m(img, r, mask, mode="img")["feats"]})
        if (j + 1) % 100 == 0:
            print(f"    rand-control {j+1}/{len(frames)}", flush=True)
    for nm, rows in (("feats base", rows_b), ("feats real-marker inpaint", rows_t),
                     ("feats RANDOM-mask inpaint", rows_r)):
        rho, mae, sd, ne = fit_eval(rows)
        res[nm] = {"rho": rho, "mae": mae}
        print(f"  {nm:28s} rho={rho:.4f} MAE={mae:.3f}N")

    # (i) markerless datasets: same detector + same inpainting
    for name, loader in (("glowtact", load_glowtact), ("cnc", load_cnc)):
        fr2, get2 = loader()
        fr2 = fr2[:300]
        _, ref2 = get2(fr2[0])
        m2, _, i2 = detect_markers(ref2)
        print(f"\n{name}: detector finds n={i2['n']} blobs "
              f"(coverage {i2['coverage']*100:.1f}%)")
        if i2["n"] == 0:
            # nothing to remove -> force the same amount of inpainting anyway
            m2, p = random_mask(ref2.shape, info["n"], info["radius_med"],
                                np.zeros(ref2.shape[:2], bool),
                                target_cov=info["coverage"])
            print(f"  no dots found; using {p} fake markers so the SAME amount "
                  f"of inpainting is applied ({m2.mean()*100:.1f}% of frame, "
                  f"area-matched to FEATS {info['coverage']*100:.1f}%)")
        rb, ri = [], []
        for j, fr in enumerate(fr2):
            img, r = get2(fr)
            rb.append({"group": fr["group"], "f": fr["f"],
                       **stages_m(img, r, None, mode="base")["feats"]})
            ri.append({"group": fr["group"], "f": fr["f"],
                       **stages_m(img, r, m2, mode="img")["feats"]})
            if (j + 1) % 100 == 0:
                print(f"    {name} {j+1}/{len(fr2)}", flush=True)
        for nm, rows in ((f"{name} base", rb), (f"{name} inpainted", ri)):
            rho, mae, sd, ne = fit_eval(rows)
            res[nm] = {"rho": rho, "mae": mae}
            print(f"  {nm:28s} rho={rho:.4f} MAE={mae:.3f}N (n={ne})")
    json.dump(res, open(OUT / "controls.json", "w"), indent=1)
    print(f"-> {OUT/'controls.json'}")


def cmd_reference():
    """If markers are not what caps FEATS, what is? The reference.

    FEATS ships no per-press reference: debug_gallery.load_feats() uses the
    median of 200 frames. Consequence measured here: |dI| > 8 already covers
    the majority of the frame on the LIGHTEST presses, so "valid" is close to
    global and the depth features integrate reference mismatch, not contact.
    Marker removal cannot touch that term — this run bounds how big it is by
    swapping in a per-indenter light-press reference (the cnc recipe).
    """
    frames, get, ref, mask, cent, info = _feats_setup()
    dI0 = []
    fs = np.array([fr["f"] for fr in frames])
    for i in np.argsort(fs)[:20]:
        im, _ = get(frames[i])
        d = np.abs(im - ref).max(2)
        dI0.append([float(np.median(d[~mask])), float((d[~mask] > 8).mean())])
    dI0 = np.array(dI0)
    print(f"20 lightest presses (F={fs[np.argsort(fs)[:20]].max():.2f}N or less):"
          f" |dI| median off-dot {np.median(dI0[:,0]):.1f} grey levels,"
          f" {np.median(dI0[:,1])*100:.0f}% of off-dot pixels already 'valid'")

    # per-indenter reference from that group's own lightest presses
    imgs = {}
    for fr in frames:
        imgs[id(fr)] = None
    grefs = {}
    for gname in sorted({fr["group"] for fr in frames}):
        sel = sorted([fr for fr in frames if fr["group"] == gname],
                     key=lambda r: r["f"])[:8]
        grefs[gname] = np.median(np.stack([get(s)[0] for s in sel]), 0)
    out = {}
    for tag, use_group in (("global median ref (current)", False),
                           ("per-indenter light-press ref", True)):
        rows = []
        for fr in frames:
            img, r = get(fr)
            if use_group:
                r = grefs[fr["group"]]
            rows.append({"group": fr["group"], "f": fr["f"],
                         **stages_m(img, r, None, mode="base")["feats"]})
        rho, mae, sd, ne = fit_eval(rows)
        out[tag] = {"rho": rho, "mae": mae}
        print(f"  {tag:30s} rho={rho:.4f} MAE={mae:.3f}N (n={ne})")
    json.dump(out, open(OUT / "reference.json", "w"), indent=1)


def cmd_all():
    for c in (cmd_detect, cmd_force, cmd_geometry, cmd_controls):
        print("=" * 74)
        c()


if __name__ == "__main__":
    {"detect": cmd_detect, "force": cmd_force, "geometry": cmd_geometry,
     "controls": cmd_controls, "stats": cmd_stats, "reference": cmd_reference, "all": cmd_all}[
        sys.argv[1] if len(sys.argv) > 1 else "all"]()
