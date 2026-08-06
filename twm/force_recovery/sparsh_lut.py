"""Sensor-general test: self-calibrate the photometric LUT on Sparsh's OWN
sphere presses, then re-evaluate force on the same frames.

Motivation (measured, see task_plan ledger):
  Our LUT was calibrated on the GlowTact sensor. Applied to Sparsh's GelSight
  Mini it produces INVALID geometry — a sphere press reconstructs as two lobes
  with a central dip. The Sparsh rho we previously reported (raw 0.451 /
  calibrated 0.558 pooled) was therefore driven by dI MAGNITUDE, not by any
  reconstructed shape.

  CORRECTION to the earlier reading of that failure: the Sparsh dI is NOT a
  "clean dipole" / degenerate illumination. PCA of 534k contact-pixel dI
  vectors gives variance fractions 0.670 / 0.258 / 0.072 — a full-rank
  three-LED signal, as a GelSight Mini should be. The foreign table simply
  decodes those colours into the WRONG directions: median angle between the
  GlowTact-LUT gradient and the analytic sphere gradient is 93.3 deg (chance
  is 90), with only 15% of pixels within 30 deg. The Sparsh-native table gives
  4.5 deg and 99.2%. Nothing was missing from the data; the map was wrong.

The question here is narrow and falsifiable: if the ONLY thing we change is to
re-derive the dI -> gradient table from Sparsh's own sphere presses (the same
procedure the sensor's manufacturer would run once), does the reconstruction
become physical, and does force recovery improve?

Ground truth for calibration comes from the dataset's own robot log:
`poses[:, 2]` is the probe z (verified: pooled rho(z, |Fz|) = -0.888,
per-trajectory median -0.927; the other two translation axes are the 2 mm
lateral slide of the protocol and correlate < 0.28). Pressing DOWN lowers z,
so indentation depth d = z0 - z with z0 the unknown contact datum, recovered
together with the sphere radius R by regressing the exact cap relation
a^2 = d (2R - d) against the detected contact radius a.

Commands (run from twm/):
    python -m force_recovery.sparsh_lut geom      # circles + sphere fit
    python -m force_recovery.sparsh_lut build     # accumulate the LUT
    python -m force_recovery.sparsh_lut verify    # IS THE DOME PHYSICAL?
    python -m force_recovery.sparsh_lut features  # feature pass, new LUT
    python -m force_recovery.sparsh_lut eval      # force + shuffle controls
    python -m force_recovery.sparsh_lut angle     # gradient-direction error
    python -m force_recovery.sparsh_lut cross     # reverse control on GlowTact
    python -m force_recovery.sparsh_lut depth     # recon vs robot z
    python -m force_recovery.sparsh_lut inview    # contact circles on eval set
    python -m force_recovery.sparsh_lut subsets   # clipping control
    python -m force_recovery.sparsh_lut inview_eval

Headline: pooled calibrated rho 0.558 -> 0.676 (MAE 0.138 -> 0.113 N) on all
frames; restricted to presses whose contact disc is fully inside the cropped
field of view, 0.878 -> 0.968 (MAE 0.079 -> 0.042 N), within-batch shuffle
-0.02..0.05 throughout. The pipeline reconstructs the robot's own indentation
depth at rho 0.92-0.94 in view, vs 0.15 over all frames — 36% of Sparsh
presses land outside the visible pad, the same defect already root-caused on
FOTA/cnc.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from force_recovery.lut_calibration import (  # noqa: E402
    BINS, DI_RANGE, CAL_OUT, W, H, detect_circle, fill_lut_holes)
from force_recovery.run_episode import OUT_ROOT  # noqa: E402
from force_recovery import sparsh_data as SD  # noqa: E402

OUT = OUT_ROOT / "sparsh_probe"
CACHE = OUT_ROOT / "feature_cache_sparshlut"
LUT_PATH = CAL_OUT / "sparsh_lut.npz"
CIRC_PATH = OUT / "sparsh_circles.json"

# GelSight Mini sensing area 18.6 x 14.3 mm on a 320x240 raw frame; sparsh_data
# rotates to landscape and `crop` keeps 230/320 of the long axis and 170/240 of
# the short one, then resizes to 320x240. That gives 0.04178 mm/px along x and
# 0.04222 mm/px along y — 1.1% anisotropic, so a single isotropic scale is used
# and the residual anisotropy is carried as a known 1% error on R.
MMPP = 0.5 * (18.6 * (230 / 320) / W + 14.3 * (170 / 240) / H)

CAL_BATCHES = [("sphere", b) for b in (1, 2, 3)]   # LUT-held-out: sphere 4-6,
N_CAL = 420                                        # flat 1-2, sharp 1-2
SHEAR_MAX = 0.15        # N; the protocol slides 2 mm AFTER loading, and a
                        # sheared contact is not the axisymmetric cap the
                        # geometry model assumes
MARGIN = 10             # px clearance from the frame border


# --------------------------------------------------------------- selection
def cal_select(n: int = N_CAL, seed: int = 0, shear_max: float = SHEAR_MAX):
    """Low-shear frames stratified over indentation DEPTH (not force).

    Stratifying on |Fz| (what the evaluation loader does) would over-sample the
    force range; the LUT needs the dI->gradient map populated evenly over
    depth, which is what sets the range of surface slopes seen.
    """
    def pick(tab):
        F, P = tab["F"], tab["P"]
        sh = np.hypot(F[:, 0], F[:, 1])
        z = P[:, 2]
        ok = np.where((sh < shear_max) & (np.abs(F[:, 2]) > 0.05))[0]
        if len(ok) == 0:
            return ok
        rng = np.random.default_rng(seed)
        edges = np.quantile(z[ok], np.linspace(0, 1, 13))
        per = int(np.ceil(n / 12))
        take = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = ok[(z[ok] >= lo) & (z[ok] <= hi)]
            if len(m):
                take.append(rng.choice(m, min(per, len(m)), replace=False))
        out = np.unique(np.concatenate(take))
        return out if len(out) <= n else rng.choice(out, n, replace=False)
    return pick


# ------------------------------------------------------------- geometry
def cmd_geom() -> None:
    """Detect contact circles on the calibration frames and fit a^2=d(2R-d)."""
    from scipy.optimize import least_squares
    from scipy.stats import spearmanr

    OUT.mkdir(parents=True, exist_ok=True)
    rec = []
    for probe, b in CAL_BATCHES:
        rows, _ = SD.load_frames(probe, b, select=cal_select())
        nd = 0
        for r in rows:
            det = detect_circle(r["img"] - r["ref"])
            if det is None:
                continue
            cx, cy, a = det
            nd += 1
            interior = (MARGIN + a < cx < W - MARGIN - a
                        and MARGIN + a < cy < H - MARGIN - a)
            rec.append({"probe": probe, "batch": b, "index": r["index"],
                        "cx": cx, "cy": cy, "a_px": a,
                        "a_mm": a * MMPP, "z": r["pz"], "f": r["f"],
                        "shear": r["shear"], "tid": r["tid"],
                        "interior": bool(interior)})
        print(f"{probe}/b{b}: loaded={len(rows)} circles={nd} "
              f"interior={sum(x['interior'] for x in rec if x['batch'] == b and x['probe'] == probe)}",
              flush=True)
        del rows

    # sanity: the detected contact area must grow with indentation, else the
    # circle detector is reading noise and the whole fit is meaningless.
    a2 = np.array([x["a_mm"] ** 2 for x in rec])
    z = np.array([x["z"] for x in rec])
    f = np.array([x["f"] for x in rec])
    print(f"\nall detections n={len(rec)}  rho(a^2, -z)={spearmanr(a2, -z).statistic:.3f}"
          f"  rho(a^2, |Fz|)={spearmanr(a2, f).statistic:.3f}")

    use = [x for x in rec if x["interior"]]
    # z0 is per batch (the gel is re-referenced between batches) but the
    # indenter is one physical sphere, so R is shared across batches.
    keys = sorted({(x["probe"], x["batch"]) for x in use})
    zz = np.array([x["z"] for x in use])
    aa = np.array([x["a_mm"] ** 2 for x in use])
    gid = np.array([keys.index((x["probe"], x["batch"])) for x in use])

    def model(par, zv, g):
        R = par[0]
        z0 = np.asarray(par[1:])[g]
        d = np.clip(z0 - zv, 1e-3, R)
        return d * (2 * R - d)

    x0 = [3.0] + [float(np.percentile(zz[gid == k], 98)) for k in range(len(keys))]
    lo = [0.5] + [zz[gid == k].min() for k in range(len(keys))]
    hi = [15.0] + [zz[gid == k].max() + 1.0 for k in range(len(keys))]
    sol = least_squares(lambda p: aa - model(p, zz, gid), x0=x0, bounds=(lo, hi))
    res = aa - model(sol.x, zz, gid)
    keep = np.abs(res) < 3 * np.std(res)
    sol = least_squares(lambda p: aa[keep] - model(p, zz[keep], gid[keep]),
                        x0=sol.x, bounds=(lo, hi))
    pred = model(sol.x, zz[keep], gid[keep])
    r2 = 1 - np.sum((aa[keep] - pred) ** 2) / np.sum((aa[keep] - aa[keep].mean()) ** 2)
    R = float(sol.x[0])
    z0 = {f"{k[0]}_b{k[1]}": float(v) for k, v in zip(keys, sol.x[1:])}
    print(f"\nSHARED-R fit: R={R:.3f} mm (diameter {2*R:.2f} mm)  "
          f"n={int(keep.sum())}/{len(use)}  R^2={r2:.3f}")
    for k, v in z0.items():
        print(f"   z0[{k}] = {v:.3f} mm   depth range "
              f"{(v - zz[gid == keys.index((k.split('_b')[0], int(k.split('_b')[1])))]).min():.3f}"
              f"..{(v - zz[gid == keys.index((k.split('_b')[0], int(k.split('_b')[1])))]).max():.3f} mm")

    # per-batch independent fit, as a consistency check on R
    per = {}
    for k in range(len(keys)):
        m = gid == k
        s = least_squares(lambda p: aa[m] - np.clip(p[1] - zz[m], 1e-3, p[0]) *
                          (2 * p[0] - np.clip(p[1] - zz[m], 1e-3, p[0])),
                          x0=[3.0, float(np.percentile(zz[m], 98))],
                          bounds=([0.5, zz[m].min()], [15.0, zz[m].max() + 1]))
        pr = np.clip(s.x[1] - zz[m], 1e-3, s.x[0])
        pr = pr * (2 * s.x[0] - pr)
        per[f"{keys[k][0]}_b{keys[k][1]}"] = {
            "R": float(s.x[0]), "z0": float(s.x[1]),
            "r2": float(1 - np.sum((aa[m] - pr) ** 2) /
                        np.sum((aa[m] - aa[m].mean()) ** 2)), "n": int(m.sum())}
    print("per-batch independent fits:", json.dumps(per, indent=None))

    for x, kp in zip(use, keep):
        x["keep"] = bool(kp)
    CIRC_PATH.write_text(json.dumps(
        {"R_mm": R, "z0": z0, "r2": float(r2), "n_used": int(keep.sum()),
         "n_detected": len(rec), "mmpp": MMPP, "per_batch": per,
         "circles": use}))
    print(f"-> {CIRC_PATH}")


# ---------------------------------------------------------------- build
def cmd_build() -> None:
    """Accumulate dI -> analytic sphere gradient into a 90^3 RGB table."""
    meta = json.loads(CIRC_PATH.read_text())
    R, z0 = meta["R_mm"], meta["z0"]
    by_batch = {}
    for c in meta["circles"]:
        if c.get("keep", True):
            by_batch.setdefault((c["probe"], c["batch"]), {})[c["index"]] = c

    ssum = np.zeros((BINS, BINS, BINS, 2), np.float64)
    cnt = np.zeros((BINS, BINS, BINS), np.int64)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    n_used = 0
    for probe, b in CAL_BATCHES:
        circ = by_batch.get((probe, b), {})
        rows, _ = SD.load_frames(probe, b, select=cal_select())
        for r in rows:
            c = circ.get(r["index"])
            if c is None:
                continue
            depth = z0[f"{probe}_b{b}"] - r["pz"]
            if not (0.10 <= depth <= 0.9 * R):
                continue
            dI = r["img"] - r["ref"]
            rx, ry = xx - c["cx"], yy - c["cy"]
            r_px = np.sqrt(rx ** 2 + ry ** 2)
            r_mm = r_px * MMPP
            inside = (r_px < 0.97 * c["a_px"]) & (r_mm < 0.985 * R)
            if inside.sum() < 50:
                continue
            denom = np.sqrt(np.clip(R ** 2 - r_mm ** 2, 1e-6, None))
            slope = -(r_mm / denom) * MMPP
            gx = np.where(r_px > 1e-6, slope * rx / np.maximum(r_px, 1e-6), 0.0)
            gy = np.where(r_px > 1e-6, slope * ry / np.maximum(r_px, 1e-6), 0.0)
            q = np.clip((dI[inside] + DI_RANGE) / (2 * DI_RANGE) * (BINS - 1),
                        0, BINS - 1).astype(np.int32)
            idx = (q[:, 0], q[:, 1], q[:, 2])
            np.add.at(ssum, idx + (0,), gx[inside])
            np.add.at(ssum, idx + (1,), gy[inside])
            np.add.at(cnt, idx, 1)
            n_used += 1
        print(f"{probe}/b{b}: accumulated (running frames={n_used})", flush=True)
        del rows

    have = cnt > 0
    lut = np.zeros((BINS, BINS, BINS, 2), np.float32)
    lut[have] = (ssum[have] / cnt[have, None]).astype(np.float32)
    filled = float(have.mean())
    print(f"frames used={n_used}  observed bins={have.sum()} "
          f"({filled*100:.3f}% of {BINS}^3)  pixels={cnt.sum()}")
    lut = fill_lut_holes(lut, cnt)
    np.savez_compressed(LUT_PATH, lut=lut, count=cnt, R_mm=R,
                        z0=json.dumps(z0), bins=BINS, di_range=DI_RANGE,
                        mmpp=MMPP)
    print(f"-> {LUT_PATH}")


# ------------------------------------------------------------- pipeline
_TABLES: dict[str, tuple] = {}


def table(name: str) -> tuple[np.ndarray, np.ndarray]:
    if name not in _TABLES:
        p = {"glowtact": CAL_OUT / "glowtact_lut.npz", "sparsh": LUT_PATH}[name]
        d = np.load(p)
        _TABLES[name] = (d["lut"], d["count"])
    return _TABLES[name]


def stages_lut(img, ref, lut, cnt, mmpp: float = MMPP) -> dict:
    """debug_gallery.stages with an injectable table (that module is frozen).

    Verified bit-identical to debug_gallery.stages when handed the GlowTact
    table and its MM_PER_PIXEL, so the head-to-head compares only the table.
    """
    sys.path.insert(0, str(Path.home() / "gelsight_heightmap_reconstruction"
                           / "python_version"))
    from fast_poisson import fast_poisson

    dI = img - ref
    q = np.clip((dI + DI_RANGE) / (2 * DI_RANGE) * (BINS - 1),
                0, BINS - 1).astype(np.int32)
    g = lut[q[..., 0], q[..., 1], q[..., 2]].copy()
    observed = cnt[q[..., 0], q[..., 1], q[..., 2]] > 0
    mag = cv2.GaussianBlur(np.abs(dI).max(2), (5, 5), 1.5)
    valid = mag > 8.0
    valid = cv2.morphologyEx(valid.astype(np.uint8), cv2.MORPH_OPEN,
                             np.ones((3, 3), np.uint8)).astype(bool)
    g[~valid] = 0.0
    depth = fast_poisson(g[..., 0], g[..., 1])
    if depth[valid].size and np.median(depth[valid]) < 0:
        depth = -depth
    d = np.maximum(depth, 0.0)
    m = d > 0.05
    px = mmpp ** 2
    feats = {"vol": float(d[m].sum() * px), "vol2": float((d[m] ** 2).sum() * px),
             "maxd": float(np.percentile(d, 99.8)), "area": float(m.sum() * px)}
    feats["h1"] = np.sqrt(feats["area"]) * feats["maxd"]
    return {"dI": dI, "gmag": np.hypot(g[..., 0], g[..., 1]), "valid": valid,
            "depth": d, "raw_depth": depth, "feats": feats,
            "lut_coverage": float(observed[valid].mean()) if valid.any() else 0.0}


# -------------------------------------------------------------- verify
def _profile(depth, cx, cy, a_px, nbin=40):
    """Radial mean profile about the contact centre, background-levelled."""
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    ring = (r > 1.25 * a_px) & (r < 1.7 * a_px)
    d = depth - (np.median(depth[ring]) if ring.sum() > 20 else 0.0)
    edges = np.linspace(0, a_px, nbin + 1)
    rr, hh = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (r >= lo) & (r < hi)
        if m.sum() > 3:
            rr.append(0.5 * (lo + hi))
            hh.append(float(d[m].mean()))
    return np.array(rr), np.array(hh), d


def cmd_verify() -> None:
    """THE CRUX: does a sphere press reconstruct as a single dome?

    Two scalars per frame, both sign-free:
      dip   = h(centre) / max h(r)   -> 1.0 for a dome, <<1 (or <0) if bilobed
      rms   = RMS residual [mm] of the radial profile against the analytic cap
              h(r) = d - R + sqrt(R^2 - r^2), amplitude-free comparison is NOT
              used: the amplitude is exactly what a LUT is supposed to get right
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    meta = json.loads(CIRC_PATH.read_text())
    R, z0 = meta["R_mm"], meta["z0"]
    circ = {(c["probe"], c["batch"], c["index"]): c for c in meta["circles"]}
    probe, b = "sphere", 1
    rows, _ = SD.load_frames(probe, b, select=cal_select())
    rows = [r for r in rows if (probe, b, r["index"]) in circ]
    dep = np.array([z0[f"{probe}_b{b}"] - r["pz"] for r in rows])
    order = np.argsort(dep)
    rows = [rows[i] for i in order]
    dep = dep[order]
    sel = [i for i in range(len(rows)) if dep[i] > 0.15]

    stats = {"glowtact": [], "sparsh": []}
    for name in ("glowtact", "sparsh"):
        lut, cnt = table(name)
        for i in sel:
            r = rows[i]
            c = circ[(probe, b, r["index"])]
            st = stages_lut(r["img"], r["ref"], lut, cnt)
            rr, hh, _ = _profile(st["raw_depth"], c["cx"], c["cy"], c["a_px"])
            if len(rr) < 8:
                continue
            s = 1.0 if hh[np.argmax(np.abs(hh))] > 0 else -1.0
            hh = hh * s
            d = dep[i]
            r_mm = rr * MMPP
            cap = d - R + np.sqrt(np.clip(R ** 2 - r_mm ** 2, 0, None))
            cap = np.maximum(cap, 0)
            stats[name].append({
                "d": float(d), "dip": float(hh[0] / max(hh.max(), 1e-6)),
                "peak": float(hh.max()), "peak_r": float(r_mm[np.argmax(hh)]),
                "rms": float(np.sqrt(np.mean((hh - cap) ** 2))),
                "amp_ratio": float(hh.max() / max(d, 1e-6))})
    print(f"n frames = {len(stats['sparsh'])}   R={R:.2f} mm, "
          f"depths {dep[sel].min():.2f}..{dep[sel].max():.2f} mm")
    print(f"{'LUT':10s} {'dip h(0)/hmax':>14s} {'peak r/a':>10s} "
          f"{'resid RMS mm':>13s} {'peak/true depth':>16s}")
    for name in ("glowtact", "sparsh"):
        s = stats[name]
        dip = np.array([x["dip"] for x in s])
        pr = np.array([x["peak_r"] for x in s])
        print(f"{name:10s} {np.median(dip):14.3f} "
              f"{np.median(pr) / (np.median([x['d'] for x in s]) and 1):10.3f} "
              f"{np.median([x['rms'] for x in s]):13.4f} "
              f"{np.median([x['amp_ratio'] for x in s]):16.3f}")
    (OUT / "dome_stats.json").write_text(json.dumps(stats))

    # figure: 3 depths x (dI, glowtact depth, sparsh depth, profiles)
    show = [sel[int(k)] for k in np.linspace(0, len(sel) - 1, 3)]
    fig, ax = plt.subplots(3, 4, figsize=(17, 10))
    for row, i in enumerate(show):
        r = rows[i]
        c = circ[(probe, b, r["index"])]
        ax[row, 0].imshow(np.clip((r["img"] - r["ref"]) * 3 + 128, 0, 255) / 255)
        ax[row, 0].set_title(f"dI x3   d={dep[i]:.2f} mm  |Fz|={r['f']:.2f} N",
                             fontsize=9)
        for col, name in ((1, "glowtact"), (2, "sparsh")):
            lut, cnt = table(name)
            st = stages_lut(r["img"], r["ref"], lut, cnt)
            rr, hh, dd = _profile(st["raw_depth"], c["cx"], c["cy"], c["a_px"])
            s = 1.0 if hh[np.argmax(np.abs(hh))] > 0 else -1.0
            im = ax[row, col].imshow(dd * s, cmap="viridis")
            ax[row, col].add_patch(plt.Circle((c["cx"], c["cy"]), c["a_px"],
                                              fill=False, color="r", lw=1))
            ax[row, col].set_title(f"{name} LUT depth [mm]", fontsize=9)
            plt.colorbar(im, ax=ax[row, col], fraction=0.04)
            ax[row, 3].plot(rr * MMPP, hh * s, label=f"{name} LUT")
        r_mm = np.linspace(0, c["a_px"] * MMPP, 60)
        cap = np.maximum(dep[i] - R + np.sqrt(np.clip(R ** 2 - r_mm ** 2, 0, None)), 0)
        ax[row, 3].plot(r_mm, cap, "k--", label="analytic cap")
        ax[row, 3].set_xlabel("r [mm]"), ax[row, 3].set_ylabel("h [mm]")
        ax[row, 3].legend(fontsize=8)
        for k in range(3):
            ax[row, k].axis("off")
    fig.suptitle("Sparsh sphere press: GlowTact LUT vs Sparsh-native LUT "
                 f"(R={R:.2f} mm fitted from Sparsh's own presses)")
    fig.tight_layout()
    fig.savefig(OUT / "dome_before_after.png", dpi=95, bbox_inches="tight")
    print(f"-> {OUT / 'dome_before_after.png'}")


# ------------------------------------------------------------- features
def cmd_features() -> None:
    from scipy.stats import spearmanr
    CACHE.mkdir(parents=True, exist_ok=True)
    lut, cnt = table("sparsh")
    for probe, b in SD.BATCHES:
        path = CACHE / f"sparsh_{probe}_batch_{b}.json"
        if path.exists():
            print(f"cached {path.name}")
            continue
        rows, ref = SD.load_frames(probe, b)     # SAME seed/selection as the
        out = []                                 # GlowTact-LUT feature cache
        for r in rows:
            st = stages_lut(r["img"], r["ref"], lut, cnt)
            out.append({k: r[k] for k in
                        ("index", "fx", "fy", "fz", "f", "fmag", "shear",
                         "tid", "slip")}
                       | st["feats"] | {"cov": st["lut_coverage"]})
        json.dump({"probe": probe, "batch": b, "rot": SD.ROT, "lut": "sparsh",
                   "rows": out}, open(path, "w"))
        f = [r["f"] for r in out]
        print(f"{probe}/batch_{b}: n={len(out)} "
              f"rho(vol,|Fz|)={spearmanr([r['vol'] for r in out], f).statistic:.3f} "
              f"cov={np.median([r['cov'] for r in out]):.3f}", flush=True)
        del rows


def _cache(which: str) -> dict:
    root = {"sparsh": CACHE, "glowtact": SD.CACHE}[which]
    out = {}
    for probe, b in SD.BATCHES:
        p = root / f"sparsh_{probe}_batch_{b}.json"
        if p.exists():
            out[f"{probe}_b{b}"] = json.load(open(p))
    return out


# ---------------------------------------------------------------- eval
def cmd_eval() -> None:
    from scipy.stats import spearmanr
    rho = lambda a, b: float(spearmanr(a, b).statistic)
    G, S = _cache("glowtact"), _cache("sparsh")
    assert set(G) == set(S)
    for k in G:                       # identical frames, only the table differs
        assert [r["index"] for r in G[k]["rows"]] == [r["index"] for r in S[k]["rows"]], k
    HELD = {"sphere_b4", "sphere_b5", "sphere_b6", "flat_b1", "flat_b2",
            "sharp_b1", "sharp_b2"}

    print("\n=== (a) raw rho(vol,|Fz|): GlowTact LUT vs Sparsh-native LUT ===")
    print(f"{'batch':12s} {'n':>4s} {'GT-LUT':>7s} {'SP-LUT':>7s} {'delta':>7s}"
          f" | {'covGT':>6s} {'covSP':>6s} | {'in LUT cal?':>11s}")
    pg, ps = [], []
    for k in G:
        g, s = G[k]["rows"], S[k]["rows"]
        pg += g, ; ps += s,
        f = [x["f"] for x in g]
        a = rho([x["vol"] for x in g], f)
        c = rho([x["vol"] for x in s], f)
        print(f"{k:12s} {len(g):4d} {a:7.3f} {c:7.3f} {c-a:+7.3f} | "
              f"{np.median([x['cov'] for x in g]):6.3f} "
              f"{np.median([x['cov'] for x in s]):6.3f} | "
              f"{'held-out' if k in HELD else 'CALIBRATED':>11s}")
    fg = [x["f"] for d in pg for x in d]
    print(f"{'POOLED':12s} {len(fg):4d} "
          f"{rho([x['vol'] for d in pg for x in d], fg):7.3f} "
          f"{rho([x['vol'] for d in ps for x in d], fg):7.3f}")

    print("\n=== (b) calibrated rho + MAE, with WITHIN-BATCH and "
          "WITHIN-TRAJECTORY shuffle controls ===")
    print(f"{'batch':12s} | {'GT rho':>7s} {'GT MAE':>7s} {'shufB':>6s} "
          f"{'shufT':>6s} | {'SP rho':>7s} {'SP MAE':>7s} {'shufB':>6s} "
          f"{'shufT':>6s} | {'d rho':>6s} {'d MAE':>6s}")
    acc = {"glowtact": [[], []], "sparsh": [[], []]}
    accs = {"glowtact": [[], []], "sparsh": [[], []]}
    for k in G:
        line = [f"{k:12s} |"]
        deltas = []
        for name, D in (("glowtact", G), ("sparsh", S)):
            p, y = SD._split_eval(D[k]["rows"])
            pb, yb = SD._split_eval(D[k]["rows"], shuffle="batch")
            pt, yt = SD._split_eval(D[k]["rows"], shuffle="traj")
            acc[name][0].append(p), acc[name][1].append(y)
            accs[name][0].append(pb), accs[name][1].append(yb)
            line.append(f" {rho(p, y):7.3f} {np.abs(p-y).mean():7.3f} "
                        f"{rho(pb, yb):6.3f} {rho(pt, yt):6.3f} |")
            deltas.append((rho(p, y), float(np.abs(p - y).mean())))
        line.append(f" {deltas[1][0]-deltas[0][0]:+6.3f} "
                    f"{deltas[1][1]-deltas[0][1]:+6.3f}")
        print("".join(line))
    out = []
    for name in ("glowtact", "sparsh"):
        p = np.concatenate(acc[name][0]); y = np.concatenate(acc[name][1])
        pb = np.concatenate(accs[name][0]); yb = np.concatenate(accs[name][1])
        out.append((name, rho(p, y), float(np.abs(p - y).mean()),
                    rho(pb, yb), float(np.abs(pb - yb).mean())))
        print(f"POOLED {name:9s} rho={out[-1][1]:.3f} MAE={out[-1][2]:.3f} N "
              f"| within-batch shuffle rho={out[-1][3]:.3f} "
              f"MAE={out[-1][4]:.3f} N")

    print("\n=== (c) cross-batch transfer with the Sparsh-native LUT "
          "(fit row -> eval column), rho (MAE N) ===")
    for name, D in (("glowtact", G), ("sparsh", S)):
        ks = [k for k in D if k.startswith("sphere")]
        print(f"\n-- {name} LUT, sphere --")
        print(f"{'fit|eval':10s}" + "".join(f"{k.split('_')[1]:>16s}" for k in ks))
        for a in ks:
            fa = SD.fit(D[a]["rows"])
            cells = []
            for bb in ks:
                pr = fa(D[bb]["rows"])
                gt = np.array([r["f"] for r in D[bb]["rows"]])
                cells.append(f"{rho(pr, gt):.3f} ({np.abs(pr-gt).mean():.3f})")
            print(f"{a.split('_')[1]:10s}" + "".join(f"{c:>16s}" for c in cells))
    print("\n-- cross-PROBE (fit sphere_b1) --")
    print(f"{'target':12s} {'GT-LUT rho':>11s} {'SP-LUT rho':>11s} "
          f"{'GT MAE':>7s} {'SP MAE':>7s}")
    fg_, fs_ = SD.fit(G["sphere_b1"]["rows"]), SD.fit(S["sphere_b1"]["rows"])
    for k in G:
        gt = np.array([r["f"] for r in G[k]["rows"]])
        a, c = fg_(G[k]["rows"]), fs_(S[k]["rows"])
        print(f"{k:12s} {rho(a, gt):11.3f} {rho(c, gt):11.3f} "
              f"{np.abs(a-gt).mean():7.3f} {np.abs(c-gt).mean():7.3f}")

    print("\n=== (d) high-shear outliers under each LUT ===")
    print(f"{'batch':12s} {'q90 shear':>9s} | {'GT |res| lo':>11s} "
          f"{'hi':>6s} {'ratio':>6s} | {'SP |res| lo':>11s} {'hi':>6s} "
          f"{'ratio':>6s}")
    tot = {"glowtact": [], "sparsh": []}
    for k in G:
        sh = np.array([x["shear"] for x in G[k]["rows"]])
        hi = sh > np.quantile(sh, 0.90)
        cells = []
        for name, D in (("glowtact", G), ("sparsh", S)):
            r = D[k]["rows"]
            f = np.array([x["f"] for x in r])
            res = np.abs(SD.fit(r)(r) - f)
            tot[name].append(res[hi].mean() / res[~hi].mean())
            cells.append(f" {res[~hi].mean():11.3f} {res[hi].mean():6.3f} "
                         f"{res[hi].mean()/res[~hi].mean():6.2f} |")
        print(f"{k:12s} {np.quantile(sh, .9):9.3f} |" + "".join(cells))
    for name in tot:
        print(f"  {name:9s} median hi/lo residual ratio = "
              f"{np.median(tot[name]):.2f}")


def cmd_depth() -> None:
    """Separate the two things a force number confounds.

    The pipeline's actual job is depth: dI -> height map. The dataset's label
    is force. If the reconstruction now tracks the robot's own indentation
    depth but force rho does not move, the remaining gap is the depth->force
    map on THIS dataset (shear, rate, contact history), not our optics.
    Joins the cached features back to poses[:,2] by frame index; no decoding.
    """
    from scipy.stats import spearmanr
    rho = lambda a, b: float(spearmanr(a, b).statistic)
    G, S = _cache("glowtact"), _cache("sparsh")
    print(f"{'batch':12s} {'rho(vol,d) GT':>13s} {'SP':>7s} | "
          f"{'rho(maxd,d) GT':>14s} {'SP':>7s} | {'rho(d,|Fz|)':>11s} "
          f"{'rho(vol,|Fz|) SP':>17s}")
    for probe, b in SD.BATCHES:
        k = f"{probe}_b{b}"
        if k not in S:
            continue
        tab = SD.label_table(probe, b)
        z = {int(i): float(p) for i, p in zip(tab["index"], tab["P"][:, 2])}
        d = np.array([-z[r["index"]] for r in S[k]["rows"]])   # deeper = larger
        f = np.array([r["f"] for r in S[k]["rows"]])
        print(f"{k:12s} {rho([r['vol'] for r in G[k]['rows']], d):13.3f} "
              f"{rho([r['vol'] for r in S[k]['rows']], d):7.3f} | "
              f"{rho([r['maxd'] for r in G[k]['rows']], d):14.3f} "
              f"{rho([r['maxd'] for r in S[k]['rows']], d):7.3f} | "
              f"{rho(d, f):11.3f} {rho([r['vol'] for r in S[k]['rows']], f):17.3f}")


EVAL_CIRC = OUT / "eval_circles.json"


def cmd_inview() -> None:
    """Detect the contact circle on the EVALUATION frames (LUT-independent).

    The protocol presses over robot x 192-212 mm, y -8..8 mm, which is wider
    than the visible pad, so a large share of presses is clipped by the frame
    edge. A clipped press loses part of its contact volume, which is exactly
    the failure already root-caused on FOTA/cnc (2/3 of that press grid was
    out of view; strictly in-view cnc went 0.63 -> 0.94). Written separately
    from the feature caches so the same flags apply to both tables.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    out = {}
    for probe, b in SD.BATCHES:
        rows, _ = SD.load_frames(probe, b)
        rec = {}
        for r in rows:
            det = detect_circle(r["img"] - r["ref"])
            if det is None:
                rec[str(r["index"])] = None
                continue
            cx, cy, a = det
            rec[str(r["index"])] = {
                "cx": cx, "cy": cy, "a_px": a,
                "inview": bool(MARGIN + a < cx < W - MARGIN - a
                               and MARGIN + a < cy < H - MARGIN - a)}
        out[f"{probe}_b{b}"] = rec
        n = sum(1 for v in rec.values() if v)
        print(f"{probe}/b{b}: detected {n}/{len(rec)}  in-view "
              f"{sum(1 for v in rec.values() if v and v['inview'])}", flush=True)
        del rows
    EVAL_CIRC.write_text(json.dumps(out))
    print(f"-> {EVAL_CIRC}")


def cmd_inview_eval() -> None:
    """Re-run the headline numbers on strictly in-view presses only."""
    from scipy.stats import spearmanr
    rho = lambda a, b: float(spearmanr(a, b).statistic)
    circ = json.loads(EVAL_CIRC.read_text())
    G, S = _cache("glowtact"), _cache("sparsh")

    print(f"{'batch':12s} {'n_all':>6s} {'n_inview':>8s} | "
          f"{'rho(vol,d) all':>14s} {'in-view':>8s} | "
          f"{'rho(vol,|Fz|) GT':>16s} {'SP':>6s} | {'cal rho GT':>10s} "
          f"{'SP':>6s} {'shufB':>6s} {'shufT':>6s} | {'SP MAE':>7s}")
    keep_rows = {}
    accg, accs, accsh = [[], []], [[], []], [[], []]
    for probe, b in SD.BATCHES:
        k = f"{probe}_b{b}"
        c = circ[k]
        tab = SD.label_table(probe, b)
        z = {int(i): float(p) for i, p in zip(tab["index"], tab["P"][:, 2])}
        g, s = G[k]["rows"], S[k]["rows"]
        m = np.array([bool(c.get(str(r["index"])) and c[str(r["index"])]["inview"])
                      for r in s])
        d = np.array([-z[r["index"]] for r in s])
        vs = np.array([r["vol"] for r in s])
        f = np.array([r["f"] for r in s])
        gi = [g[i] for i in np.where(m)[0]]
        si = [s[i] for i in np.where(m)[0]]
        if len(si) < 60:
            # `detect_circle` demands a near-circular disc, which the sharp
            # indenter never makes: sharp/b1 yields 0 in-view discs out of 750.
            # The in-view filter is therefore only defined on sphere and flat.
            print(f"{k:12s} {len(s):6d} {int(m.sum()):8d} |  (no circular "
                  f"contact — sharp indenter, filter undefined)")
            continue
        p1, y1 = SD._split_eval(gi)
        p2, y2 = SD._split_eval(si)
        pb, yb = SD._split_eval(si, shuffle="batch")
        pt, yt = SD._split_eval(si, shuffle="traj")
        accg[0].append(p1), accg[1].append(y1)
        accs[0].append(p2), accs[1].append(y2)
        accsh[0].append(pb), accsh[1].append(yb)
        keep_rows[k] = si
        print(f"{k:12s} {len(s):6d} {int(m.sum()):8d} | {rho(vs, d):14.3f} "
              f"{rho(vs[m], d[m]):8.3f} | "
              f"{rho([r['vol'] for r in gi], [r['f'] for r in gi]):16.3f} "
              f"{rho(vs[m], f[m]):6.3f} | {rho(p1, y1):10.3f} "
              f"{rho(p2, y2):6.3f} {rho(pb, yb):6.3f} {rho(pt, yt):6.3f} | "
              f"{np.abs(p2 - y2).mean():7.3f}")
    for nm, a in (("glowtact", accg), ("sparsh", accs), ("sparsh-SHUFFLED", accsh)):
        p, y = np.concatenate(a[0]), np.concatenate(a[1])
        print(f"POOLED in-view {nm:16s} rho={rho(p, y):.3f} "
              f"MAE={np.abs(p - y).mean():.3f} N  (n={len(p)})")

    print("\n-- in-view cross-batch transfer, Sparsh LUT: rho (MAE N) --")
    ks = [k for k in keep_rows if k.startswith("sphere")]
    print(f"{'fit|eval':10s}" + "".join(f"{k.split('_')[1]:>16s}" for k in ks))
    for a in ks:
        fa = SD.fit(keep_rows[a])
        cells = []
        for bb in ks:
            pr = fa(keep_rows[bb])
            gt = np.array([r["f"] for r in keep_rows[bb]])
            cells.append(f"{rho(pr, gt):.3f} ({np.abs(pr-gt).mean():.3f})")
        print(f"{a.split('_')[1]:10s}" + "".join(f"{c:>16s}" for c in cells))
    print("-- in-view cross-PROBE, fit sphere_b1 --")
    fa = SD.fit(keep_rows["sphere_b1"])
    for k, rs in keep_rows.items():
        gt = np.array([r["f"] for r in rs])
        pr = fa(rs)
        print(f"   sphere_b1 -> {k:12s} rho={rho(pr, gt):.3f} "
              f"MAE={np.abs(pr - gt).mean():.3f} N")

    print("\n-- in-view high-shear decile, Sparsh LUT --")
    for k, rs in keep_rows.items():
        sh = np.array([x["shear"] for x in rs])
        hi = sh > np.quantile(sh, 0.90)
        f = np.array([x["f"] for x in rs])
        res = np.abs(SD.fit(rs)(rs) - f)
        print(f"   {k:12s} |res| lo={res[~hi].mean():.3f} hi={res[hi].mean():.3f}"
              f"  ratio={res[hi].mean()/res[~hi].mean():.2f}")


def cmd_subsets() -> None:
    """Separate the two things the in-view filter does.

    `detect_circle` needs BOTH a strong enough contact (|dI| threshold) and a
    circular, unclipped disc. So the in-view subset could be better simply
    because near-zero-force frames were dropped. The decisive comparison is
    the middle row: frames whose contact WAS detected but is clipped by the
    frame edge. Same force regime, only the visibility differs.
    """
    from scipy.stats import spearmanr
    rho = lambda a, b: float(spearmanr(a, b).statistic)
    circ = json.loads(EVAL_CIRC.read_text())
    S, G = _cache("sparsh"), _cache("glowtact")
    print(f"{'subset':22s} {'n':>5s} {'|Fz| med':>9s} {'|Fz| p95':>9s} "
          f"{'shear med':>9s} {'rho(vol,d)':>11s} {'rho(vol,|Fz|)':>14s} "
          f"{'cal rho':>8s} {'MAE':>6s} {'shufB':>6s}")
    for name, want in (("all frames", None), ("no circle detected", "none"),
                       ("detected, CLIPPED", "clip"),
                       ("detected, in-view", "inview")):
        rows, ds = [], []
        for probe, b in SD.BATCHES:
            if probe == "sharp":
                continue                   # no circular contact, see cmd_inview
            k = f"{probe}_b{b}"
            c = circ[k]
            tab = SD.label_table(probe, b)
            z = {int(i): float(p) for i, p in zip(tab["index"], tab["P"][:, 2])}
            for r in S[k]["rows"]:
                v = c.get(str(r["index"]))
                ok = {None: True, "none": v is None,
                      "clip": bool(v) and not v["inview"],
                      "inview": bool(v) and v["inview"]}[want]
                if ok:
                    rows.append(r), ds.append(-z[r["index"]])
        f = np.array([r["f"] for r in rows])
        sh = np.array([r["shear"] for r in rows])
        v = np.array([r["vol"] for r in rows])
        p, y = SD._split_eval(rows)
        pb, yb = SD._split_eval(rows, shuffle="batch")
        print(f"{name:22s} {len(rows):5d} {np.median(f):9.3f} "
              f"{np.quantile(f, .95):9.3f} {np.median(sh):9.3f} "
              f"{rho(v, ds):11.3f} {rho(v, f):14.3f} {rho(p, y):8.3f} "
              f"{np.abs(p - y).mean():6.3f} {rho(pb, yb):6.3f}")


def cmd_angle() -> None:
    """WHY the foreign table fails: it maps the same colours to wrong slopes.

    On calibration frames the true surface gradient is known analytically, so
    the error of a table is measurable directly, without any Poisson step:
    angle between predicted and true (gx, gy), and |g_pred| / |g_true|.
    """
    meta = json.loads(CIRC_PATH.read_text())
    R, z0 = meta["R_mm"], meta["z0"]
    circ = {c["index"]: c for c in meta["circles"]
            if c["probe"] == "sphere" and c["batch"] == 1 and c.get("keep")}
    rows, _ = SD.load_frames("sphere", 1, select=cal_select())
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    acc = {"glowtact": [[], []], "sparsh": [[], []]}
    nrgb = []
    for r in rows:
        c = circ.get(r["index"])
        if c is None:
            continue
        d = z0["sphere_b1"] - r["pz"]
        if not (0.15 <= d <= 0.9 * R):
            continue
        dI = r["img"] - r["ref"]
        rx, ry = xx - c["cx"], yy - c["cy"]
        rp = np.sqrt(rx ** 2 + ry ** 2)
        rm = rp * MMPP
        inside = (rp < 0.97 * c["a_px"]) & (rm < 0.985 * R) & (rp > 3)
        if inside.sum() < 50:
            continue
        sl = -(rm / np.sqrt(np.clip(R ** 2 - rm ** 2, 1e-6, None))) * MMPP
        tx = (sl * rx / np.maximum(rp, 1e-6))[inside]
        ty = (sl * ry / np.maximum(rp, 1e-6))[inside]
        nrgb.append(dI[inside])
        q = np.clip((dI + DI_RANGE) / (2 * DI_RANGE) * (BINS - 1),
                    0, BINS - 1).astype(np.int32)
        for name in acc:
            lut, _ = table(name)
            g = lut[q[..., 0], q[..., 1], q[..., 2]]
            px, py = g[..., 0][inside], g[..., 1][inside]
            ang = np.degrees(np.abs(np.arctan2(px * ty - py * tx,
                                               px * tx + py * ty)))
            acc[name][0].append(ang)
            acc[name][1].append(np.hypot(px, py) / np.maximum(np.hypot(tx, ty), 1e-9))
    print(f"contact pixels = {sum(len(a) for a in acc['sparsh'][0])}")
    print(f"{'LUT':10s} {'median angle err':>17s} {'frac<30deg':>11s} "
          f"{'median |g|ratio':>16s}")
    for name in ("glowtact", "sparsh"):
        a = np.concatenate(acc[name][0]); m = np.concatenate(acc[name][1])
        print(f"{name:10s} {np.median(a):16.1f}d {np.mean(a < 30):11.3f} "
              f"{np.median(m):16.3f}")
    # is Sparsh's dI really 1-D (a "dipole")? PCA of the contact-pixel colours
    X = np.concatenate(nrgb); X = X - X.mean(0)
    ev = np.linalg.svd(X, compute_uv=False) ** 2
    print(f"Sparsh dI colour PCA variance fractions: "
          f"{np.round(ev / ev.sum(), 3)}")


def cmd_cross() -> None:
    """Reverse control: does the SPARSH table also fail on GlowTact frames?

    If tables were interchangeable in one direction only, the story would be
    'our table is bad'. If each table only works at home, the conclusion is
    that the map is a per-sensor property, which is the claim being tested.
    """
    from PIL import Image
    from force_recovery.lut_calibration import (
        load_family, fit_sphere_geometry, crop as gcrop, MM_PER_PIXEL)

    ref, rows = load_family("round")
    geom, fits = fit_sphere_geometry(ref, rows)
    print(f"glowtact sphere: R={geom.R_mm:.2f} mm z0={geom.z0_mm:.2f} "
          f"n={geom.n_used} R^2={geom.fit_r2:.3f}")
    fits = [f for f in fits if 0.15 <= f["z"] - geom.z0_mm <= 0.9 * geom.R_mm]
    fits = fits[:180]
    out = {}
    for name in ("glowtact", "sparsh"):
        lut, cnt = table(name)
        dip, rms, amp = [], [], []
        for f in fits:
            img = gcrop(np.asarray(Image.open(f["path"]).convert("RGB"))
                        ).astype(np.float32)
            st = stages_lut(img, ref, lut, cnt, mmpp=MM_PER_PIXEL)
            rr, hh, _ = _profile(st["raw_depth"], f["cx"], f["cy"], f["a_px"])
            if len(rr) < 8:
                continue
            hh = hh * (1.0 if hh[np.argmax(np.abs(hh))] > 0 else -1.0)
            d = f["z"] - geom.z0_mm
            rm = rr * MM_PER_PIXEL
            cap = np.maximum(d - geom.R_mm +
                             np.sqrt(np.clip(geom.R_mm ** 2 - rm ** 2, 0, None)), 0)
            dip.append(hh[0] / max(hh.max(), 1e-6))
            rms.append(np.sqrt(np.mean((hh - cap) ** 2)))
            amp.append(hh.max() / max(d, 1e-6))
        out[name] = (np.median(dip), np.median(rms), np.median(amp), len(dip))
    print(f"GLOWTACT frames  {'dip':>8s} {'RMS mm':>8s} {'amp':>6s} {'n':>5s}")
    for k, v in out.items():
        print(f"  {k:14s} {v[0]:8.3f} {v[1]:8.4f} {v[2]:6.3f} {v[3]:5d}")


if __name__ == "__main__":
    {"geom": cmd_geom, "build": cmd_build, "verify": cmd_verify,
     "features": cmd_features, "eval": cmd_eval, "angle": cmd_angle,
     "cross": cmd_cross, "depth": cmd_depth, "inview": cmd_inview,
     "inview_eval": cmd_inview_eval, "subsets": cmd_subsets}[
        sys.argv[1] if len(sys.argv) > 1 else "geom"]()
