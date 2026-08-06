"""Showcase gallery: raw image | depth map | 3D surface | force readout.

Two products:
  image_samples() — 20 stills from the GT datasets (cnc_Mini + FEATS),
      each panel: raw frame, plane-removed indentation heat map, 3D surface
      render, and predicted-vs-ground-truth force bars.
  video_samples() — 10 clips from React episodes around their force peaks:
      tactile stream | live depth reconstruction | running force trace.
"""
from __future__ import annotations

import io
import json
import subprocess
import tarfile
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .depth_force import DepthForceEstimator
from .feature_cache import CACHE, features_from_indent, indent_map
from .fota_cnc_validation import _group_by_tar, build_index, load_images
from .run_episode import DATA_ROOT, LEGACY_SHIFT, OUT_ROOT, STAGE_ROOT

GALLERY = OUT_ROOT / "site" / "assets" / "gallery"


def _panel(img: np.ndarray, ind: np.ndarray, f_pred: float,
           f_true: float | None, title: str, out: Path) -> None:
    fig = plt.figure(figsize=(11.5, 3.1))
    ax = fig.add_subplot(1, 4, 1)
    ax.imshow(cv2.resize(img, (320, 240)))
    ax.set_title("raw GelSight frame", fontsize=8); ax.axis("off")

    ax = fig.add_subplot(1, 4, 2)
    vmax = max(float(ind.max()), 0.05)
    im = ax.imshow(ind, cmap="inferno", vmin=0, vmax=vmax)
    ax.set_title("indentation δ [mm]", fontsize=8); ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(1, 4, 3, projection="3d")
    step = 6
    yy, xx = np.mgrid[0:ind.shape[0]:step, 0:ind.shape[1]:step]
    ax.plot_surface(xx, yy, np.clip(ind[::step, ::step], 0, None),
                    cmap="inferno", linewidth=0, antialiased=False)
    ax.set_zlim(0, vmax); ax.set_title("3D reconstruction", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([]); ax.tick_params(labelsize=6)
    ax.view_init(elev=55, azim=-60)

    ax = fig.add_subplot(1, 4, 4)
    bars = [("predicted", f_pred, "#d95f02")]
    if f_true is not None:
        bars.append(("ground truth", f_true, "#1b9e77"))
    ax.bar([b[0] for b in bars], [b[1] for b in bars],
           color=[b[2] for b in bars], width=0.55)
    ax.set_ylabel("normal force [N]", fontsize=8)
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=110); plt.close(fig)


def image_samples(n_cnc: int = 14, n_feats: int = 6) -> list[Path]:
    GALLERY.mkdir(parents=True, exist_ok=True)
    outs = []

    # --- cnc_Mini, stratified by force; flatfield pipeline (validated best)
    ff_cache = CACHE.parent / "feature_cache_ff" / "cnc_mini_val.json"
    thr = json.loads(ff_cache.read_text())["threshold_mm"]

    index = build_index("val")
    forces = np.array([e["force"] for e in index])
    order = np.argsort(forces)
    refs = load_images([index[i] for i in order if forces[i] < 0.5][:6]
                       or [index[i] for i in order[:4]])
    est = DepthForceEstimator(refs, already_cropped=False, inpaint=False,
                              flatfield=True)
    est.contact_threshold_mm = thr
    # scale for display: median F/vol_soft on the ff train cache
    tr = json.loads((CACHE.parent / "feature_cache_ff" /
                     "cnc_mini_train.json").read_text())["rows"]
    ratios = [r["force_true"] / r["vol_soft"] for r in tr
              if r["vol_soft"] > 0.2 and r["force_true"] > 0.5]
    c = float(np.median(ratios))

    picks = [index[i] for i in
             order[np.linspace(len(order) // 10, len(order) - 1, n_cnc)
                   .astype(int)]]
    k = 0
    for tar, rows in _group_by_tar(picks).items():
        with tarfile.open(tar) as tf:
            for p in rows:
                img = np.asarray(Image.open(io.BytesIO(
                    tf.extractfile(p["jpg"]).read())).convert("RGB"))
                ind = indent_map(est, img)
                f = features_from_indent(ind, thr)
                out = GALLERY / f"cnc_{k:02d}.png"
                _panel(img, np.clip(ind, 0, None), f["vol_soft"] * c,
                       p["force"],
                       f"cnc_Mini probe {p['obj_name']}  z={p['z_mm']:.1f} mm",
                       out)
                outs.append(out); k += 1

    # --- FEATS val, marker gel (inpainted)
    import pyarrow.parquet as pq

    from .external_validation import FEATS_PARQUET, _decode
    df = pq.read_table(str(FEATS_PARQUET / "val-00000-of-00001.parquet")
                       ).to_pandas()
    df["fz"] = -df["f_z"]; df = df.sort_values("fz").reset_index(drop=True)
    frefs = [_decode(r) for r in df.head(8).image]
    fest = DepthForceEstimator(frefs, already_cropped=True, inpaint=True)
    press = df[(df.fz > 1) & (df.fz < 30)
               & ~df.capture.str.contains("shear")]
    rows = press.iloc[np.linspace(0, len(press) - 1, n_feats).astype(int)]
    for j, (_, row) in enumerate(rows.iterrows()):
        img = _decode(row.image)
        ind = indent_map(fest, img)
        f = features_from_indent(ind, fest.contact_threshold_mm)
        out = GALLERY / f"feats_{j:02d}.png"
        _panel(img, np.clip(ind, 0, None), f["vol_soft"] * c, float(row.fz),
               f"FEATS {row.capture.split('_', 2)[-1]}", out)
        outs.append(out)
    return outs


def video_samples(n: int = 10, seconds: float = 8.0) -> list[Path]:
    """React episode clips: tactile | live depth | force trace."""
    from .evaluate import median3_fresh
    import h5py
    import hdf5plugin  # noqa: F401
    import pyarrow.parquet as pq

    GALLERY.mkdir(parents=True, exist_ok=True)
    npzs = sorted(Path(OUT_ROOT).rglob("episode_*_*.npz"),
                  key=lambda p: -float(np.load(p)["force_normal_n"].max()))
    outs = []
    for npz_path in npzs[:n]:
        task = npz_path.parent.parent.name
        date = npz_path.parent.name
        ep, side = npz_path.stem.rsplit("_", 1)
        z = np.load(npz_path)
        table = pq.read_table(
            str(STAGE_ROOT / task / "meta" / date / f"{ep}.parquet"),
            columns=[f"tactile_{side}_is_new", f"tactile_{side}_intensity"])
        is_new = np.asarray(table[f"tactile_{side}_is_new"].to_numpy())
        inten = np.asarray(table[f"tactile_{side}_intensity"].to_numpy())
        force = median3_fresh(z["force_normal_n"], is_new)
        trim = int(z["trim"])
        peak = int(np.argmax(force))
        half = int(seconds * 30 / 2)
        lo, hi = max(0, peak - half), min(len(force), peak + half)
        fmax = max(float(force.max()), 1e-3)

        with h5py.File(str(DATA_ROOT / task / date / f"{ep}.h5"), "r") as f:
            frames = f[f"gelsight/{side}/frames"]
            nmax = len(frames) - 1
            ref_rows = z["reference_rows"]
            refs = [frames[min(trim + int(r) + LEGACY_SHIFT, nmax)]
                    for r in ref_rows[:6]]
            est = DepthForceEstimator(refs, already_cropped=False,
                                      inpaint=False)
            est.contact_threshold_mm = float(z["contact_threshold_mm"])

            out = GALLERY / f"clip_{task}_{ep}_{side}.mp4"
            tmp = out.with_suffix(".raw.mp4")
            W_, H_ = 960, 330
            vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"),
                                 30, (W_, H_))
            last_row, ind = None, None
            for row in range(lo, hi):
                src = min(trim + row + LEGACY_SHIFT, nmax)
                img = frames[src]
                if is_new[row] or ind is None:
                    ind = np.clip(indent_map(est, img), 0, None)
                canvas = np.zeros((H_, W_, 3), np.uint8)
                canvas[:240, :320] = cv2.cvtColor(
                    cv2.resize(img, (320, 240)), cv2.COLOR_RGB2BGR)
                dn = (np.clip(ind / max(fmax / 20.0, ind.max() + 1e-9, 0.2),
                              0, 1) * 255).astype(np.uint8)
                canvas[:240, 320:640] = cv2.applyColorMap(dn, cv2.COLORMAP_INFERNO)
                # force trace panel
                xs = np.linspace(650, 950, hi - lo).astype(int)
                ys = (200 - 170 * force[lo:hi] / fmax).astype(int) + 20
                for i in range(1, len(xs)):
                    cv2.line(canvas, (xs[i - 1], ys[i - 1]), (xs[i], ys[i]),
                             (150, 150, 150), 1)
                cv2.circle(canvas, (xs[row - lo], ys[row - lo]), 4,
                           (30, 120, 240), -1)
                for x0, lab in ((0, "tactile"), (320, "depth"), (650, "force")):
                    cv2.putText(canvas, lab, (x0 + 8, 262),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (200, 200, 200), 1, cv2.LINE_AA)
                cv2.putText(canvas, f"{force[row]:.2f} N  "
                            f"({task}/{ep} {side})", (8, 300),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (240, 240, 240), 1, cv2.LINE_AA)
                vw.write(canvas)
            vw.release()
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp),
                        "-c:v", "libx264", "-crf", "24", "-pix_fmt",
                        "yuv420p", "-movflags", "+faststart", str(out)],
                       check=True)
        tmp.unlink()
        outs.append(out)
        print(f"  clip {out.name}", flush=True)
    return outs
