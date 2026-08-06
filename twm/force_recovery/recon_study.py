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


def stages_full(img, ref, lut, cnt):
    """Every intermediate array, nothing collapsed."""
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
    """Crater artefact: how far the CENTRE sits below the rim of a contact.

    A flat-topped indenter (star/quad/triangle/letter) should reconstruct as a
    plateau. Its interior carries no colour change — flat gel, zero gradient —
    so Poisson can only raise it by integrating inward from the rim, and when
    the rim gradient is under-recovered the middle sags into a crater.

    Returns (h_centre / h_rim_max): 1.0 is a clean plateau, < 1 is a crater.
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

        fig = plt.figure(figsize=(30, 4.1))
        n = 14

        def add(i, data, ttl, cmap=None, cb=False):
            a = fig.add_subplot(1, n, i)
            im = a.imshow(data, cmap=cmap)
            a.set_title(ttl, fontsize=7.5)
            a.axis("off")
            if cb:
                fig.colorbar(im, ax=a, fraction=0.046)
            return a

        add(1, np.clip(img, 0, 255).astype(np.uint8), "1 raw")
        add(2, np.clip(s["ref"], 0, 255).astype(np.uint8), "2 reference")
        add(3, np.clip(st["dI"] * 3 + 128, 0, 255).astype(np.uint8),
            "3  dI = img − ref  (×3)")
        add(4, st["absdI"], "4  max|dI| per pixel", "viridis", True)
        add(5, st["valid"], f"5  valid mask  |dI|>{VALID_THR:g}", "gray")
        add(6, st["observed"], "6  LUT bin observed?\n(dark = made-up value)",
            "gray")
        v = np.abs(np.concatenate([st["gx"].ravel(), st["gy"].ravel()])).max()
        add(7, st["gx"], "7  gx (LUT)", "coolwarm", True)
        add(8, st["gy"], "8  gy (LUT)", "coolwarm", True)
        add(9, np.hypot(st["gx"], st["gy"]), "9  |grad|", "magma", True)
        add(10, st["div"], "10  div(g) = Poisson RHS", "coolwarm", True)
        add(11, st["depth"], "11  depth, fast_poisson [mm]", "inferno", True)
        add(12, st["depth_flat"], "12  depth, halo pedestal removed",
            "inferno", True)
        a = fig.add_subplot(1, n, 13)
        if prof:
            a.plot(prof["r"], prof["h"], "-o", ms=2.5, label="reconstructed")
            a.plot(prof["r"], prof["cap"], "k--", lw=1, label="analytic cap")
            a.set_xlabel("r [mm]", fontsize=7)
            a.set_ylabel("h [mm]", fontsize=7)
            a.legend(fontsize=6)
            a.set_title(f"13  radial profile\nRMS {prof['rms']*1000:.0f} µm",
                        fontsize=7.5)
        else:
            a.text(.5, .5, "13  no sphere\n(profile check n/a)", ha="center",
                   va="center", fontsize=8)
            a.axis("off")
        a.tick_params(labelsize=6)
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
            fontsize=11, fontweight="bold")
        fig.tight_layout()
        fp = OUT / f"{which}_{k:02d}.png"
        fig.savefig(fp, dpi=78, bbox_inches="tight")
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


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "glowtact")
