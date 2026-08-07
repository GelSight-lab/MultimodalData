"""Step-by-step pipeline debug gallery: raw image -> force, on 3 datasets.

Answers: is cnc_Mini's low rho a dataset-quality problem or a recon problem?
Shows every intermediate stage on sampled frames:

    raw -> ref -> dI (difference) -> valid mask -> LUT gradients -> Poisson
    depth -> features (vol, maxd, area) -> calibrated force vs GT

Datasets:
  glowtact  markerless, the LUT's home sensor (control: recon known good)
  cnc       markerless, FOREIGN sensor, no reference frames shipped
            (ref = per-probe median of 10 lightest presses)
  feats     dot/marker type (ref = global median of 200 frames)

Per-dataset diagnostics printed and saved:
  ref RGB mean (sensor color shift), LUT observed-bin coverage on contact
  pixels (out-of-domain measure), rho(maxd, |z| or F), rho(F_pred, F_gt).

Run:
  python -m force_recovery.debug_gallery features   # compute + diagnose
  python -m force_recovery.debug_gallery gallery    # render 18 samples/dataset
"""
from __future__ import annotations

import io
import json
import sys
import tarfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1]))
from force_recovery.lut_calibration import (  # noqa: E402
    crop, GLOWTACT, PAT, BINS, DI_RANGE, CAL_OUT, MM_PER_PIXEL, W, H)

OUT = Path("/media/yxma/Disk1/twm/force_recovery/site_assets/debug_gallery")
CNC = Path("/media/yxma/Disk1/twm/force_recovery/fota_cnc/cnc/cnc_Mini")
FEATS_PQ = Path("/media/yxma/Disk1/yuxiang/mini_data_parquet/feats")
N_FIT = 350          # frames per dataset for feature fit + diagnostics
N_GALLERY = 18
RNG = np.random.default_rng(0)

_cal = np.load(CAL_OUT / "glowtact_lut.npz")
LUT, CNT = _cal["lut"], _cal["count"]


# ---------------------------------------------------------------- pipeline
def stages(img: np.ndarray, ref: np.ndarray) -> dict:
    """All intermediate stages from raw to depth + features."""
    sys.path.insert(0, str(Path.home() / "gelsight_heightmap_reconstruction"
                           / "python_version"))
    from fast_poisson import fast_poisson

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
    depth = fast_poisson(g[..., 0], g[..., 1])
    if depth[valid].size and np.median(depth[valid]) < 0:
        depth = -depth
    d = np.maximum(depth, 0.0)
    m = d > 0.05
    px_mm2 = MM_PER_PIXEL ** 2
    feats = {
        "vol": float(d[m].sum() * px_mm2), "vol2": float((d[m] ** 2).sum() * px_mm2),
        "maxd": float(np.percentile(d, 99.8)), "area": float(m.sum() * px_mm2),
    }
    feats["h1"] = np.sqrt(feats["area"]) * feats["maxd"]
    cov = float(observed[valid].mean()) if valid.any() else 0.0
    return {"dI": dI, "gmag": np.hypot(g[..., 0], g[..., 1]), "valid": valid,
            "depth": d, "feats": feats, "lut_coverage": cov}


def feat_vec(f):
    return [f["vol"], f["vol2"], f["maxd"], f["area"], f["h1"]]


# ---------------------------------------------------------------- datasets
def _num(s):                       # cnc filenames use '-' as decimal point
    sign = -1.0 if s.startswith("-") else 1.0
    return sign * float(s.lstrip("-").replace("-", "."))


def load_glowtact():
    fams = ["round", "quad", "star", "triangle", "B", "quad_small"]
    frames, refs = [], {}
    for fam in fams:
        refs[fam] = crop(np.asarray(Image.open(
            GLOWTACT / fam / "initial.jpg").convert("RGB"))).astype(np.float32)
        rows = []
        for p in (GLOWTACT / fam).glob("*.jpg"):
            m = PAT.search(p.name)
            if m and float(m["f"]) > 0.3:
                rows.append({"path": p, "group": fam, "f": float(m["f"]),
                             "z": -float(m["z"]), "x": float(m["x"]),
                             "y": float(m["y"])})
        rows = [rows[i] for i in RNG.permutation(len(rows))]
        frames += rows[:N_FIT // len(fams) + 20]
    def get(fr):
        img = crop(np.asarray(Image.open(fr["path"]).convert("RGB"))
                   ).astype(np.float32)
        return img, refs[fr["group"]]
    return frames, get


def load_cnc():
    import re
    pat = re.compile(r"(Mini_[A-F])\|(\d+)pos\|([-\d]+), ([-\d]+), "
                     r"([-\d]+) f\|([-\d]+)\.jpg$")
    blobs, metas = {}, []
    for split in ("train", "val"):
        with tarfile.open(CNC / split / "data-000000.tar") as tf:
            for mem in tf.getmembers():
                m = pat.search(mem.name)
                if not m:
                    continue
                f = _num(m[6])
                if f <= 0.15:
                    pass                      # keep light frames for refs
                blobs[mem.name] = tf.extractfile(mem).read()
                metas.append({"key": mem.name, "group": m[1], "f": f,
                              "z": abs(_num(m[5])), "x": _num(m[3]),
                              "y": _num(m[4])})
    refs = {}
    for probe in sorted({m["group"] for m in metas}):
        light = sorted([m for m in metas if m["group"] == probe],
                       key=lambda r: r["f"])[:10]
        refs[probe] = np.median(np.stack([
            crop(np.asarray(Image.open(io.BytesIO(blobs[r["key"]]))
                            .convert("RGB"))).astype(np.float32)
            for r in light]), 0)
    frames = [m for m in metas if m["f"] > 0.3]
    import os
    n = int(os.environ.get("CNC_N", N_FIT + 40))
    frames = [frames[i] for i in RNG.permutation(len(frames))][:n]
    def get(fr):
        img = crop(np.asarray(Image.open(io.BytesIO(blobs[fr["key"]]))
                              .convert("RGB"))).astype(np.float32)
        return img, refs[fr["group"]]
    return frames, get


def load_feats():
    import pyarrow.parquet as pq
    t = pq.read_table(FEATS_PQ / "val-00000-of-00001.parquet",
                      columns=["image", "f_z", "indenter"]).to_pandas()
    imgs = t["image"].to_list()
    def decode(i):
        a = np.asarray(Image.open(io.BytesIO(imgs[i])).convert("RGB"))
        return cv2.resize(a, (W, H)).astype(np.float32)
    ridx = RNG.permutation(len(t))[:200]
    ref = np.median(np.stack([decode(i) for i in ridx]), 0)
    frames = [{"idx": int(i), "group": t["indenter"].iloc[i],
               "f": float(abs(t["f_z"].iloc[i])), "z": float("nan"),
               "x": float("nan"), "y": float("nan")}
              for i in RNG.permutation(len(t))[:N_FIT + 40]]
    def get(fr):
        return decode(fr["idx"]), ref
    return frames, get


DATASETS = {"glowtact": load_glowtact, "cnc": load_cnc, "feats": load_feats}


# ---------------------------------------------------------------- commands
def cmd_features():
    from scipy.stats import spearmanr
    OUT.mkdir(parents=True, exist_ok=True)
    for name, loader in DATASETS.items():
        frames, get = loader()
        out = []
        for fr in frames:
            img, ref = get(fr)
            st = stages(img, ref)
            out.append({**{k: fr[k] for k in ("group", "f", "z", "x", "y")},
                        **st["feats"], "cov": st["lut_coverage"],
                        "ref_rgb": [float(v) for v in ref.mean((0, 1))]})
        path = OUT / f"features_{name}.json"
        json.dump(out, open(path, "w"))
        f = np.array([r["f"] for r in out])
        z = np.array([r["z"] for r in out])
        maxd = np.array([r["maxd"] for r in out])
        cov = np.array([r["cov"] for r in out])
        print(f"== {name}: n={len(out)} ref_rgb={np.round(out[0]['ref_rgb'],1)}"
              f" LUT-coverage med={np.median(cov):.2f}")
        if np.isfinite(z).all():
            for g in sorted({r['group'] for r in out}):
                m = np.array([r['group'] == g for r in out])
                print(f"   {g:10s} rho(maxd,z)={spearmanr(maxd[m], z[m]).statistic:.3f}"
                      f" rho(maxd,F)={spearmanr(maxd[m], f[m]).statistic:.3f}")
        else:
            print(f"   rho(maxd,F)={spearmanr(maxd, f).statistic:.3f}")


def _fit_predict(rows):
    """Per-group half/half fit -> predictions for the eval half."""
    from sklearn.isotonic import IsotonicRegression
    X = np.array([feat_vec(r) for r in rows])
    f = np.array([r["f"] for r in rows])
    groups = np.array([r["group"] for r in rows])
    pred = np.full(len(f), np.nan)
    for g in sorted(set(groups)):
        m = np.where(groups == g)[0]
        half = len(m) // 2
        fi, ei = m[:half], m[half:]
        if len(fi) < 8:
            continue
        wl, *_ = np.linalg.lstsq(X[fi], f[fi], rcond=None)
        iso = IsotonicRegression(out_of_bounds="clip").fit(X[fi] @ wl, f[fi])
        pred[ei] = iso.predict(X[ei] @ wl)
    return pred


def cmd_gallery():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    for name, loader in DATASETS.items():
        frames, get = loader()
        # feature pass for calibration
        rows = []
        for fr in frames:
            img, ref = get(fr)
            rows.append({**fr, **stages(img, ref)["feats"]})
        pred = _fit_predict(rows)
        ok = np.isfinite(pred)
        f = np.array([r["f"] for r in rows])
        rho = spearmanr(pred[ok], f[ok]).statistic
        mae = np.abs(pred[ok] - f[ok]).mean()
        # stratified gallery picks among predicted rows
        cand = np.where(ok)[0]
        order = cand[np.argsort(f[cand])]
        picks = order[np.linspace(0, len(order) - 1, N_GALLERY).astype(int)]
        gdir = OUT / name
        gdir.mkdir(parents=True, exist_ok=True)
        items = []
        for k, i in enumerate(picks):
            fr = rows[i]
            img, ref = get(fr)
            st = stages(img, ref)
            # 2 rows of 4: a 7-wide strip in the 880 px page column gives
            # each panel 126 px and a ~3.4 px effective title.
            fig, axg = plt.subplots(2, 4, figsize=(15.5, 15.5 / (4 * 4 /
                                                   (2 * 3)) * 1.30))
            ax = axg.ravel()
            panels = [
                (img / 255, f"raw  [{fr['group']}]", None),
                (ref / 255, "reference", None),
                (np.clip(st["dI"] * 2 + 128, 0, 255) / 255, "dI = img - ref (x2)", None),
                (st["valid"], "valid mask (|dI|>8)", "gray"),
                (st["gmag"], "|LUT gradient|", "magma"),
                (st["depth"], "depth [mm] (Poisson)", "viridis"),
            ]
            for a, (im, ti, cm) in zip(ax, panels):
                a.imshow(im, cmap=cm)
                a.set_title(ti, fontsize=11)
                a.axis("off")
            zs = "" if not np.isfinite(fr.get("z", np.nan)) else f"z={fr['z']:.2f}mm\n"
            ax[6].text(0.02, 0.95, (
                f"{zs}F_gt={fr['f']:.2f}N\nF_pred={pred[i]:.2f}N\n\n"
                f"maxd={fr['maxd']:.2f}mm\nvol={fr['vol']:.1f}mm3\n"
                f"area={fr['area']:.0f}mm2\nLUT cov={st['lut_coverage']*100:.0f}%"),
                va="top", fontsize=12, family="monospace",
                transform=ax[6].transAxes)
            ax[6].axis("off")
            ax[7].axis("off")            # 8th cell of the 2x4 grid is spare
            fig.suptitle(f"{name} sample {k+1}/{N_GALLERY}", fontsize=13)
            fig.tight_layout()
            fp = gdir / f"sample_{k:02d}.png"
            fig.savefig(fp, dpi=90, bbox_inches="tight")
            plt.close(fig)
            items.append(fp.name)
        html = ["<html><body style='font-family:sans-serif;background:#111;color:#eee'>",
                f"<h2>{name} — raw→force step-by-step "
                f"(rho={rho:.3f}, MAE={mae:.2f}N on {ok.sum()} eval frames)</h2>"]
        html += [f"<img src='{it}' style='width:100%;max-width:1600px'><br>"
                 for it in items]
        html.append("</body></html>")
        (gdir / "index.html").write_text("\n".join(html))
        print(f"{name}: rho={rho:.3f} MAE={mae:.2f}N -> {gdir}/index.html")


if __name__ == "__main__":
    {"features": cmd_features, "gallery": cmd_gallery}[
        sys.argv[1] if len(sys.argv) > 1 else "features"]()
