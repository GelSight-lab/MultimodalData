"""Every knob in the 3D reconstruction, laid out for 20 samples.

Purpose: this is a WORKBENCH, not a results figure. Each sample shows the full
chain with the intermediate quantities that are actually tunable, so a defect
can be attributed to a stage instead of guessed at:

  1  raw                     the frame
  2  reference               rest gel (per-family initial.jpg / median of
                             contact-free frames)
  3  dI = img - ref          x3 gain; the ONLY input the LUT sees
  4  |dI| max over channels   what the valid mask thresholds on
  5  valid mask              |dI| > VALID_THR, opened 3x3   <-- knob
  6  LUT bin coverage        were these colours ever observed in calibration?
                             (blue = unobserved -> nearest-filled, i.e. made up)
  7  gx                      LUT surface gradient, x            <-- knob: LUT
  8  gy                      LUT surface gradient, y
  9  |grad|                  magnitude
 10  div(g)                  the Poisson right-hand side
 11  depth (fast_poisson)    Neumann/DCT solve                  <-- knob: solver
 12  depth, halo removed     annulus-median pedestal subtraction <-- knob
 13  radial profile          reconstructed vs analytic sphere cap where the
                             indenter is a sphere of known R
 14  Open3D mesh             what the site shows

Plus a per-sample diagnostics line: peak depth, contact area, fraction of
contact pixels landing in UNOBSERVED LUT bins, median angle between the LUT
gradient and the analytic sphere gradient (the measure that caught the
93.3-degree foreign-sensor failure — chance is 90), and the profile residual.

Run (needs a display for Open3D):
  xvfb-run -a -s "-screen 0 1400x1000x24" \
      python -m force_recovery.recon_study glowtact   # or: sparsh, cnc, feats
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .lut_calibration import (BINS, CAL_OUT, DI_RANGE, GLOWTACT, MM_PER_PIXEL,
                              PAT, crop)
from .o3d_view import has_display, remove_halo_pedestal, mesh_view_rgb
from .run_episode import OUT_ROOT

OUT = OUT_ROOT / "recon_study"
VALID_THR = 8.0           # knob: valid-mask threshold on |dI|
N_SAMPLES = 20


def _poisson(gx, gy):
    sys.path.insert(0, str(Path.home() / "gelsight_heightmap_reconstruction"
                           / "python_version"))
    from fast_poisson import fast_poisson
    return fast_poisson(gx, gy)


def stages_full(img, ref, lut, cnt, inpaint_markers: bool = False):
    """Every intermediate array, nothing collapsed.

    `inpaint_markers` runs the adopted depth-path marker step
    (`marker_removal.stages_depth`'s preprocessing) before differencing. It is
    off by default because this workbench's sources are markerless, where the
    detector finds nothing and the step is a bit-exact no-op anyway; the flag
    exists so a marker gel can be walked through the same 14 panels.
    """
    if inpaint_markers:
        from .marker_removal import inpaint_img, marker_mask
        mk = marker_mask(ref)
        if mk is not None:
            img, ref = inpaint_img(img, mk), inpaint_img(ref, mk)
    dI = img - ref
    q = np.clip((dI + DI_RANGE) / (2 * DI_RANGE) * (BINS - 1),
                0, BINS - 1).astype(np.int32)
    g = lut[q[..., 0], q[..., 1], q[..., 2]].copy()
    observed = cnt[q[..., 0], q[..., 1], q[..., 2]] > 0
    absdI = np.abs(dI).max(axis=2)
    mag = cv2.GaussianBlur(absdI, (5, 5), 1.5)
    valid = mag > VALID_THR
    valid = cv2.morphologyEx(valid.astype(np.uint8), cv2.MORPH_OPEN,
                             np.ones((3, 3), np.uint8)).astype(bool)
    g[~valid] = 0.0
    gx, gy = g[..., 0], g[..., 1]
    div = np.zeros_like(gx)
    div[:, 1:] += gx[:, 1:] - gx[:, :-1]
    div[1:, :] += gy[1:, :] - gy[:-1, :]
    depth = _poisson(gx, gy)
    if valid.any() and np.median(depth[valid]) < 0:
        depth = -depth
    depth = np.maximum(depth, 0.0)
    return dict(dI=dI, absdI=absdI, valid=valid, observed=observed,
                gx=gx, gy=gy, div=div, depth=depth,
                depth_flat=remove_halo_pedestal(depth))


def sphere_check(depth, R_mm):
    """Radial profile vs the analytic cap, and the gradient-angle diagnostic."""
    d = depth
    pk = float(d.max())
    if pk <= 1e-6:
        return None
    core = d > 0.30 * pk
    if core.sum() < 60:
        return None
    ys, xs = np.nonzero(core)
    cy, cx = ys.mean(), xs.mean()
    yy, xx = np.mgrid[0:d.shape[0], 0:d.shape[1]]
    r = np.hypot(xx - cx, yy - cy) * MM_PER_PIXEL
    m = core & (r < 0.95 * R_mm)
    if m.sum() < 40:
        return None
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
    cap = hp[0] - (R_mm - np.sqrt(np.clip(R_mm ** 2 - rc ** 2, 1e-9, None)))
    return dict(r=rc, h=hp, cap=np.maximum(cap, 0),
                rms=float(np.sqrt(np.mean((hp - np.maximum(cap, 0)) ** 2))),
                peak=pk, cx=cx, cy=cy)


def flat_top_sag(depth, contact_frac=0.35):
    """Centre/rim height ratio of a contact. Descriptive, NOT a defect score.

    RETRACTED criterion: we used to read this against "a flat-topped indenter
    should reconstruct as a plateau, so 1.0 is correct". That premise is
    wrong. The gel is compliant and wraps around a flat edge, so the TRUE
    surface already domes and c/r > 1 is expected for any contact.

    Per-pixel ground truth (`mnist_validation`, Tactile MNIST meshes) puts
    numbers on it: the true depth map of a pressed digit has c/r 1.400
    (1.334 for the gel surface itself) while we reconstruct 1.539-1.562, and
    on an enclosed filleted-plateau control whose truth is exactly 1.000 we
    measure 1.069. The over-doming artefact is therefore +7% to +12%, not the
    +23-42% the raw ratios below suggest.

    Returns h_centre / h_rim_median; compare it against the GT ratio for the
    same geometry, never against 1.0.
    """
    import cv2

    pk = float(depth.max())
    if pk <= 1e-6:
        return float("nan")
    m = (depth > contact_frac * pk).astype(np.uint8)
    if m.sum() < 80:
        return float("nan")
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    if dist.max() < 4:
        return float("nan")
    core = dist > 0.75 * dist.max()           # deep interior
    rim = (dist > 0.15 * dist.max()) & (dist < 0.4 * dist.max())
    if core.sum() < 10 or rim.sum() < 20:
        return float("nan")
    return float(np.median(depth[core]) / max(np.median(depth[rim]), 1e-9))


def grad_angle(gx, gy, valid, cx, cy, R_mm):
    """Median angle between LUT gradient and the analytic sphere gradient.

    Chance is 90 deg. This is the statistic that exposed the foreign-sensor
    failure (93.3 deg) BEFORE any integration, so it separates a bad table
    from a bad solver.
    """
    yy, xx = np.mgrid[0:gx.shape[0], 0:gx.shape[1]]
    dx, dy = (xx - cx) * MM_PER_PIXEL, (yy - cy) * MM_PER_PIXEL
    r = np.hypot(dx, dy)
    m = valid & (r > 0.15 * R_mm) & (r < 0.9 * R_mm)
    if m.sum() < 50:
        return float("nan")
    slope = -(r / np.sqrt(np.clip(R_mm ** 2 - r ** 2, 1e-9, None))) * MM_PER_PIXEL
    ax = np.where(r > 0, slope * dx / np.maximum(r, 1e-9), 0)
    ay = np.where(r > 0, slope * dy / np.maximum(r, 1e-9), 0)
    u = np.stack([gx[m], gy[m]], 1)
    v = np.stack([ax[m], ay[m]], 1)
    nu = np.linalg.norm(u, axis=1)
    nv = np.linalg.norm(v, axis=1)
    ok = (nu > 1e-9) & (nv > 1e-9)
    if ok.sum() < 30:
        return float("nan")
    cos = np.clip((u[ok] * v[ok]).sum(1) / (nu[ok] * nv[ok]), -1, 1)
    return float(np.degrees(np.arccos(cos)).mean())


# ------------------------------------------------------------------ sources
def src_glowtact():
    lut = np.load(CAL_OUT / "glowtact_lut.npz")
    R = float(lut["R_mm"])
    picks = []
    for fam in ("round", "star", "triangle", "quad", "quad_small", "B"):
        ref = crop(np.asarray(Image.open(GLOWTACT / fam / "initial.jpg")
                              .convert("RGB"))).astype(np.float32)
        rows = []
        for p in (GLOWTACT / fam).glob("*.jpg"):
            m = PAT.search(p.name)
            if m and 6 < float(m["x"]) < 12 and 5 < float(m["y"]) < 11:
                rows.append((p, -float(m["z"]), float(m["f"])))
        rows.sort(key=lambda t: t[1])
        # 4 depth quantiles on the two families whose LUT coverage degrades
        # fastest with depth (star, quad_small), 3 elsewhere -> 20 samples
        qs = (0.15, 0.4, 0.62, 0.85) if fam in ("star", "quad_small") \
            else (0.2, 0.45, 0.7)
        for q in qs:
            p, z, f = rows[int(len(rows) * q)]
            picks.append(dict(path=p, ref=ref, z=z, f=f,
                              tag=f"{fam} z={z:.2f}mm F={f:.1f}N",
                              sphere=(fam == "round")))
    return lut["lut"], lut["count"], R, picks[:N_SAMPLES]


def src_sparsh():
    lut = np.load(CAL_OUT / "sparsh_lut.npz")
    from . import sparsh_data as SD
    R = float(lut["R_mm"]) if "R_mm" in lut else 2.438
    picks = []
    for b in (1, 2, 3):
        fr, lab, ref = SD.load_batch("sphere", b, with_ref=True)
        idxs = sorted(lab.keys())[:400]
        for j in np.linspace(0, len(idxs) - 1, 7).astype(int):
            i = idxs[j]
            picks.append(dict(img=fr(i), ref=ref, z=float("nan"),
                              f=abs(lab[i][2]),
                              tag=f"sphere b{b} F={abs(lab[i][2]):.2f}N",
                              sphere=True))
    return lut["lut"], lut["count"], R, picks[:N_SAMPLES]


SOURCES = {"glowtact": src_glowtact, "sparsh": src_sparsh}


def build(which="glowtact"):
    if not has_display():
        raise SystemExit('needs: xvfb-run -a -s "-screen 0 1400x1000x24" ...')
    lut, cnt, R, picks = SOURCES[which]()
    OUT.mkdir(parents=True, exist_ok=True)
    diag = []
    for k, s in enumerate(picks):
        img = s.get("img")
        if img is None:
            img = crop(np.asarray(Image.open(s["path"]).convert("RGB"))
                       ).astype(np.float32)
        st = stages_full(img, s["ref"], lut, cnt)
        prof = sphere_check(st["depth_flat"], R) if s.get("sphere") else None
        ang = (grad_angle(st["gx"], st["gy"], st["valid"], prof["cx"],
                          prof["cy"], R) if prof else float("nan"))
        unobs = float((~st["observed"][st["valid"]]).mean()) \
            if st["valid"].any() else float("nan")
        sag = flat_top_sag(st["depth_flat"])

        # Two rows of 7, not one row of 14. Measured: at the page's 1900 px
        # container a 14-wide strip gives each panel 136 px and renders its
        # title at ~6.5 px effective — unreadable. 2x7 doubles both.
        NC, NR = 7, 2
        # figsize follows the CELL aspect, it is not guessed: the panels are
        # 320x240 (4:3), so 7 cols x 2 rows wants 7*4 : 2*3 = 4.67:1, plus
        # ~28% height for titles and colourbars. A 1.79:1 figure (the first
        # attempt) letterboxed every image and left half the canvas empty.
        fig = plt.figure(figsize=(17.0, 17.0 / 4.67 * 1.28))

        def add(i, data, ttl, cmap=None, cb=False):
            a = fig.add_subplot(NR, NC, i)
            im = a.imshow(data, cmap=cmap)
            a.set_title(ttl, fontsize=10.5)
            a.axis("off")
            if cb:
                fig.colorbar(im, ax=a, fraction=0.046)
            return a

        add(1, np.clip(img, 0, 255).astype(np.uint8), "1 raw")
        add(2, np.clip(s["ref"], 0, 255).astype(np.uint8), "2 reference")
        add(3, np.clip(st["dI"] * 3 + 128, 0, 255).astype(np.uint8),
            "3  dI = img − ref  (×3)")
        add(4, st["absdI"], "4  max|dI|", "viridis", True)
        add(5, st["valid"], f"5  valid mask  |dI|>{VALID_THR:g}", "gray")
        add(6, st["observed"], "6  LUT bin seen?\n(dark = invented)",
            "gray")
        v = np.abs(np.concatenate([st["gx"].ravel(), st["gy"].ravel()])).max()
        add(7, st["gx"], "7  gx (LUT)", "coolwarm", True)
        add(8, st["gy"], "8  gy (LUT)", "coolwarm", True)
        add(9, np.hypot(st["gx"], st["gy"]), "9  |grad|", "magma", True)
        add(10, st["div"], "10  div(g)", "coolwarm", True)
        add(11, st["depth"], "11  depth [mm]", "inferno", True)
        add(12, st["depth_flat"], "12  depth, halo removed",
            "inferno", True)
        a = fig.add_subplot(NR, NC, 13)
        if prof:
            a.plot(prof["r"], prof["h"], "-o", ms=2.5, label="reconstructed")
            a.plot(prof["r"], prof["cap"], "k--", lw=1, label="analytic cap")
            a.set_xlabel("r [mm]", fontsize=8.5)   # no ylabel: it lands on
            # panel 12's colourbar; the units are already in the title
            a.legend(fontsize=6)
            a.set_title(f"13  radial profile  h [mm]\nRMS "
                        f"{prof['rms']*1000:.0f} µm", fontsize=10.5)
        else:
            a.text(.5, .5, "13  no sphere\n(profile check n/a)", ha="center",
                   va="center", fontsize=11)
            a.axis("off")
        a.tick_params(labelsize=8.5)
        add(14, mesh_view_rgb(st["depth_flat"]), "14  Open3D mesh")

        peak = float(st["depth_flat"].max())
        area = float((st["depth_flat"] > 0.05).sum()) * MM_PER_PIXEL ** 2
        fig.suptitle(
            f"[{k+1}/{len(picks)}] {s['tag']}   |   peak {peak:.2f} mm · "
            f"area {area:.1f} mm² · unobserved-LUT {unobs*100:.0f}% · "
            # no verdict word here: 1.4 is CORRECT for a sphere and WRONG
            # for a flat punch, so the label would have to know the indenter
            + (f"centre/rim {sag:.2f} · " if sag == sag else "")
            + (f"grad-angle {ang:.1f}° (chance 90°) · profile RMS "
               f"{prof['rms']*1000:.0f} µm" if prof else "no sphere check"),
            fontsize=13, fontweight="bold")
        fig.subplots_adjust(left=.012, right=.988, top=.86, bottom=.03,
                            wspace=.30, hspace=.26)
        fp = OUT / f"{which}_{k:02d}.png"
        fig.savefig(fp, dpi=110, bbox_inches="tight")
        plt.close(fig)
        diag.append(dict(sample=k, tag=s["tag"], peak_mm=peak,
                         area_mm2=area, unobserved_lut_frac=unobs,
                         grad_angle_deg=ang, flat_top_ratio=sag,
                         profile_rms_um=(prof["rms"] * 1000) if prof else None))
        print(f"  [{k+1:2d}/{len(picks)}] {s['tag']:34s} peak {peak:5.2f}mm  "
              f"unobs {unobs*100:4.0f}%  c/r "
              + (f"{sag:4.2f}" if sag == sag else " n/a") + "  ang "
              + (f"{ang:5.1f}°" if ang == ang else "  n/a") + "  rms "
              + (f"{prof['rms']*1000:4.0f}µm" if prof else " n/a"), flush=True)
    (OUT / f"{which}_diagnostics.json").write_text(json.dumps(diag, indent=1))
    html = ["<html><body style='background:#111;color:#eee;"
            "font-family:sans-serif'>",
            f"<h2>{which}: raw → mesh, every stage ({len(picks)} samples)</h2>",
            "<p>1 raw · 2 reference · 3 dI · 4 max|dI| · 5 valid mask · "
            "6 LUT coverage · 7 gx · 8 gy · 9 |grad| · 10 div(g) · "
            "11 depth · 12 depth halo-removed · 13 radial profile vs "
            "analytic cap · 14 Open3D mesh</p>"]
    html += [f"<img src='{which}_{i:02d}.png' style='width:100%'><br>"
             for i in range(len(picks))]
    (OUT / f"{which}.html").write_text("\n".join(html) + "</body></html>")
    print(f"-> {OUT / (which + '.html')}")


# ------------------------------------------------------------------ site page
SITE = OUT_ROOT / "site"

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>3D reconstruction workbench — every stage, 20 samples</title>
<style>body{background:#0f1420;color:#e8ecf4;font-family:system-ui,sans-serif;
margin:0;padding:28px}.wrap{max-width:1900px;margin:0 auto}
h1{font-size:26px;margin:0 0 6px}h2{color:#ffd9a0;font-size:18px;margin:26px 0 8px}
p{color:#b9c2d0;line-height:1.55;max-width:1000px}
img{width:100%;background:#fff;border-radius:5px;margin:5px 0}
table{border-collapse:collapse;margin:10px 0;font-size:13px}
th,td{border:1px solid #2c3648;padding:5px 10px;text-align:left}
th{color:#ffd9a0;font-weight:600}code{background:#1b2334;padding:1px 5px;border-radius:3px}
.note{color:#8e99ab;font-size:13px}
.good{color:#7be0a0}.bad{color:#ff9a8a}</style></head><body><div class="wrap">
<h1>3D reconstruction workbench — every stage, 20 samples</h1>
<p>Not a results figure: this is the full chain with every tunable quantity
exposed, so a defect can be attributed to a stage. Columns:
<b>1</b> raw · <b>2</b> reference · <b>3</b> dI = img−ref (×3) ·
<b>4</b> max|dI| · <b>5</b> valid mask · <b>6</b> LUT bin observed? ·
<b>7</b> gx · <b>8</b> gy · <b>9</b> |grad| · <b>10</b> div(g) = Poisson RHS ·
<b>11</b> depth · <b>12</b> depth halo-removed · <b>13</b> radial profile vs
analytic cap · <b>14</b> Open3D mesh.
Reproduce: <code>xvfb-run -a -s "-screen 0 1400x1000x24" python -m
force_recovery.recon_study glowtact</code>, then
<code>python -m force_recovery.recon_study page</code>.
The source here (GlowTact) is markerless, so the adopted marker step
(<code>marker_removal.stages_depth</code>) is a bit-exact no-op on these
frames; on a marker gel it inpaints the dots out of the reference and the
frame before column 3.</p>

<h2>What external ground truth did to this page</h2>
<p>Everything below used to be scored against our own expectations. It is now
scored against exact per-pixel ground truth — ray-cast mesh depth on 420
Tactile MNIST touches (<code>mnist_validation</code>) — and two of the three
defects we published moved:</p>
<table><tr><th>claim</th><th>status after per-pixel GT</th></tr>
<tr><td>Flat-topped indenters over-dome badly (centre/rim 1.23–1.42 where
1.0 is correct)</td>
<td class="bad"><b>retracted and re-measured.</b> Compliant gel wraps a flat
edge, so c/r &gt; 1 is <i>expected</i>: the true depth map of a pressed digit
has c/r <b>1.400</b> (gel surface 1.334) and we reconstruct 1.539–1.562
(<b>+10–12%</b>); on an enclosed plateau control whose truth is exactly 1.000
we measure <b>1.069</b> (<b>+7%</b>). Most of the 1.23–1.42 in the table
below is real curvature, not artefact.</td></tr>
<tr><td>Up to 22% of contact pixels land in unobserved LUT bins — a top
defect</td>
<td class="bad"><b>demoted to a minor factor.</b> On the GT set 13.8% / 16.9%
of contact pixels are unobserved, and their correlation with per-touch
Type-2 error is only <b>0.098 / 0.294</b>. Still worth showing (column 6),
no longer worth blaming.</td></tr>
<tr><td>The valid mask is halo-dominated</td>
<td class="good"><b>confirmed and quantified.</b> Against the true contact
region the <code>|dI| &gt; 8</code> mask scores IoU <b>0.614</b>, recall
<b>0.917</b>, over-segmentation <b>0.531</b> — it finds nearly all of the
contact and then adds half as much again in halo.</td></tr>
<tr><td>(new) the photometric table is the weak link</td>
<td class="good"><b>ruled out.</b> LUT gradient direction vs true gel
gradient is <b>24.4°</b> on the GT renders against <b>26.1°</b> for this
table on its own real sensor — the same statistic, no worse off-domain.</td>
</tr></table>

<h2>The real headline: accuracy is a function of press depth</h2>
<p>Same digit meshes, re-rendered at five penetrations, no per-frame fitting
of any kind (the photometric table is calibrated once on that sensor's own
sphere presses, the recipe we already use per sensor):</p>
<table><tr><th>press depth [mm]</th><th>0.30</th><th>0.60</th><th>1.00</th>
<th>1.50</th><th>2.25 (what the dataset ships)</th></tr>
<tr><td>MAE [µm]</td><td class="good"><b>11.2</b></td><td class="good">35.0</td>
<td>67.8</td><td>127.4</td><td class="bad">281.1</td></tr>
<tr><td>Type-2 error [µm]</td><td class="good"><b>96.5</b></td><td>186.3</td>
<td>308.6</td><td>514.6</td><td class="bad">961.8</td></tr>
<tr><td>peak ours / GT</td><td>1.00</td><td>0.97</td><td>0.77</td><td>0.68</td>
<td class="bad">0.55</td></tr></table>
<p>So the honest public claim is a <b>range</b>, not a number: at ≤0.6 mm this
reconstruction is accurate on non-spherical ground truth with zero fitting; by
2.25 mm it recovers barely half the peak. No accuracy figure on this site
should be quoted without the press depth it was measured at.</p>

<h2>Per-sample diagnostics</h2>
<p class="note">centre/rim is reported <b>descriptively</b> — read it against
the GT ratio for that geometry (1.33–1.40 for a compliant press), never
against 1.0.</p>
@@TABLE@@
<p class="note">Grad angle and profile RMS are only defined where the indenter
is a sphere of known radius (the <code>round</code> family): the LUT gradient
sits @@ANGRANGE@@ from the analytic sphere gradient and the radial profile
matches the cap to @@RMSRANGE@@, so on this sensor the table and the solver
are both sound. Every one of these frames is a deep press (@@ZRANGE@@ mm) —
i.e. the regime the sweep above shows is our worst.</p>
<footer style="margin-top:30px;color:#6f7a8c;font-size:13px">
React force recovery · <a href="index.html" style="color:#ffd9a0">overview</a> ·
<a href="method.html" style="color:#ffd9a0">method</a> ·
<a href="results.html" style="color:#ffd9a0">results</a></footer>
@@IMAGES@@
</div></body></html>"""


def build_page(which: str = "glowtact") -> Path:
    """Emit recon_workbench.html from the diagnostics JSON (no typed numbers)."""
    import shutil

    diag = json.loads((OUT / f"{which}_diagnostics.json").read_text())
    dst = SITE / "assets" / "recon"
    dst.mkdir(parents=True, exist_ok=True)
    for p in sorted(OUT.glob(f"{which}_*.png")):
        shutil.copyfile(p, dst / p.name)

    def cell(v, fmt, scale=1.0):
        return "" if v is None or v != v else format(v * scale, fmt)

    rows = "".join(
        f"<tr><td>{d['tag']}</td><td>{d['peak_mm']:.2f}</td>"
        f"<td>{d['unobserved_lut_frac']*100:.0f}%</td>"
        f"<td>{cell(d['flat_top_ratio'], '.2f')}</td>"
        f"<td>{cell(d['grad_angle_deg'], '.1f')}</td>"
        f"<td>{cell(d['profile_rms_um'], '.0f')}</td></tr>" for d in diag)
    table = ("<table><tr><th>sample</th><th>peak [mm]</th>"
             "<th>unobserved LUT</th><th>centre/rim<br>"
             "<span class='note'>GT for a compliant press: 1.33–1.40</span>"
             "</th><th>grad angle<br><span class='note'>chance 90°</span></th>"
             f"<th>profile RMS</th></tr>{rows}</table>")
    ang = [d["grad_angle_deg"] for d in diag if d["grad_angle_deg"] == d["grad_angle_deg"]]
    rms = [d["profile_rms_um"] for d in diag if d["profile_rms_um"]]
    zs = [float(t.split("z=")[1].split("mm")[0]) for d in diag
          if "z=" in (t := d["tag"])]
    imgs = "".join(f"<img src='assets/recon/{p.name}' loading='lazy'><br>"
                   for p in sorted(dst.glob(f"{which}_*.png")))
    page = (PAGE.replace("@@TABLE@@", table)
                .replace("@@ANGRANGE@@", f"{min(ang):.1f}–{max(ang):.1f}°")
                .replace("@@RMSRANGE@@", f"{min(rms):.0f}–{max(rms):.0f} µm")
                .replace("@@ZRANGE@@", f"{min(zs):.1f}–{max(zs):.1f}")
                .replace("@@IMAGES@@", imgs))
    out = SITE / "recon_workbench.html"
    out.write_text(page)
    print(f"-> {out}  ({len(diag)} samples, {len(list(dst.glob('*.png')))} images)")
    return out


if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv) > 1 else "glowtact"
    if a == "page":
        build_page(sys.argv[2] if len(sys.argv) > 2 else "glowtact")
    else:
        build(a)
