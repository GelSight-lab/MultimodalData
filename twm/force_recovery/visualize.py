"""Figures and clips for the force-recovery results site.

Everything renders headless into OUT_ROOT/site_assets/.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

from .dexforce import force_informed_targets, gel_axis
from .run_episode import DATA_ROOT, LEGACY_SHIFT, OUT_ROOT, STAGE_ROOT

ASSETS = OUT_ROOT / "site_assets"

plt.rcParams.update({
    "figure.dpi": 130, "font.size": 9, "axes.grid": True,
    "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False,
})


def _load(task, date, ep, side):
    npz = np.load(OUT_ROOT / task / date / f"{ep}_{side}.npz")
    table = pq.read_table(
        str(STAGE_ROOT / task / "meta" / date / f"{ep}.parquet"),
        columns=[f"tactile_{side}_intensity", f"tactile_{side}_is_new",
                 f"sensor_{side}_pose"])
    inten = np.asarray(table[f"tactile_{side}_intensity"].to_numpy())
    is_new = np.asarray(table[f"tactile_{side}_is_new"].to_numpy())
    pose = np.stack(table[f"sensor_{side}_pose"].to_numpy())
    return npz, inten, is_new, pose


def force_timeline(task: str, date: str, ep: str) -> Path:
    """Two-panel force + intensity timeline for one episode."""
    from .evaluate import median3_fresh

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 4.4), sharex=True)
    for ax, side in zip(axes, ("left", "right")):
        npz, inten, is_new, _ = _load(task, date, ep, side)
        raw = npz["force_normal_n"]
        f = median3_fresh(raw, is_new)
        t = np.arange(len(f)) / 30.0
        ax.plot(t, raw, lw=0.4, color="#d95f02", alpha=0.3)
        ax.plot(t, f, lw=0.8, color="#d95f02",
                label="normal force (median-3 on fresh frames)")
        ax.set_ylabel(f"{side}\nF [N]", color="#d95f02")
        ax2 = ax.twinx()
        ax2.plot(t, inten, lw=0.5, color="#7570b3", alpha=0.55,
                 label="tactile intensity")
        ax2.set_ylabel("intensity", color="#7570b3")
        ax2.grid(False); ax2.spines["top"].set_visible(False)
    axes[1].set_xlabel("time [s]")
    axes[0].set_title(f"{task}/{date}/{ep} — Winkler normal force vs "
                      "contact intensity (independent proxy)")
    out = ASSETS / f"timeline_{task}_{ep}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    return out


def depth_panels(task: str, date: str, ep: str, side: str) -> Path:
    """Raw | image-difference | depth map for the strongest-contact rows."""
    import h5py
    import hdf5plugin  # noqa: F401

    npz, _, _, _ = _load(task, date, ep, side)
    trim = int(npz["trim"])
    depth_keys = sorted(k for k in npz.files if k.startswith("depth_row_"))
    ref_row = int(npz["reference_rows"][0])

    with h5py.File(str(DATA_ROOT / task / date / f"{ep}.h5"), "r") as f:
        frames = f[f"gelsight/{side}/frames"]
        nmax = len(frames) - 1
        ref = frames[min(trim + ref_row + LEGACY_SHIFT, nmax)].astype(np.float32)
        rows = []
        for key in depth_keys[:3]:
            row = int(key.split("_")[-1])
            img = frames[min(trim + row + LEGACY_SHIFT, nmax)]
            depth = npz[key]
            rows.append((row, img, depth))

    fig, axes = plt.subplots(len(rows), 3, figsize=(8.4, 2.15 * len(rows)))
    axes = np.atleast_2d(axes)
    force = npz["force_normal_n"]
    for r, (row, img, depth) in enumerate(rows):
        diff = np.abs(img.astype(np.float32) - ref).mean(axis=2)
        for c, (data, cmap, title) in enumerate((
                (img, None, f"row {row}  F={force[row]:.2f} N"),
                (diff, "gray", "|frame - reference|"),
                (depth, "inferno", f"depth (max {depth.max():.2f} mm)"))):
            ax = axes[r, c]
            ax.imshow(data, cmap=cmap)
            ax.set_title(title, fontsize=8)
            ax.axis("off")
    fig.suptitle(f"{task}/{ep} {side} — photometric-stereo depth at the "
                 "strongest contacts", fontsize=10)
    out = ASSETS / f"depth_{task}_{ep}_{side}.png"
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    return out


def dexforce_figure(task: str, date: str, ep: str, side: str) -> Path:
    """Virtual-target separation along the pressing direction + histogram.

    Plotting a world coordinate hides the offset (0.2-4 mm against a
    ~100 mm trajectory), so both curves are projected onto the pressing
    direction at the force peak — there the separation IS the penetration.
    """
    from .evaluate import median3_fresh

    npz, _, is_new, pose = _load(task, date, ep, side)
    force = median3_fresh(npz["force_normal_n"].astype(np.float64), is_new)
    act = force_informed_targets(pose, force, gel_axis(task, side))
    pen_mm = act.penetration_m * 1000.0
    from .evaluate import CONTACT_N
    contact = force > CONTACT_N

    peak = int(np.argmax(force))
    lo, hi = max(0, peak - 90), min(len(force), peak + 90)
    offset_mm = pen_mm[lo:hi]           # target - observed, along the normal
    t = np.arange(lo, hi) / 30.0

    fig = plt.figure(figsize=(9.5, 3.8))
    ax = fig.add_subplot(1, 2, 1)
    ax.axhline(0, color="#1b9e77", lw=1.1, label="observed pose (offset 0)")
    ax.plot(t, offset_mm, lw=0.9, color="#d95f02",
            label="virtual target offset F/k")
    ax.fill_between(t, 0, offset_mm, color="#d95f02", alpha=0.25)
    ax2 = ax.twinx()
    ax2.plot(t, force[lo:hi], lw=0.6, color="#7570b3", alpha=0.6)
    ax2.set_ylabel("F [N]", color="#7570b3"); ax2.grid(False)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("target offset along pressing direction [mm]")
    ax.set_title("6 s around the force peak — how far past the surface\n"
                 "the action commands the impedance controller")
    ax.legend(fontsize=8, loc="upper right")

    ax = fig.add_subplot(1, 2, 2)
    ax.hist(pen_mm[contact], bins=40, color="#7570b3")
    ax.set_xlabel("penetration F/k [mm]"); ax.set_ylabel("rows")
    ax.set_title(f"penetration while in contact "
                 f"(k = {act.stiffness:.0f} N/m, max {pen_mm.max():.1f} mm)")
    fig.suptitle(f"{task}/{ep} {side} — DexForce-style virtual targets",
                 fontsize=10)
    out = ASSETS / f"dexforce_{task}_{ep}_{side}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    return out


def feats_validation_figure() -> Path:
    """Ground-truth scatter: estimated volume x calibrated force vs FEA force."""
    import json

    rep = json.loads((OUT_ROOT / "feats_validation_val.json").read_text())
    pf = rep["per_frame"]
    ft = np.array([r["f_true"] for r in pf])
    fc = np.array([r["volume_mm3"] for r in pf]) * rep["scale_n_per_mm3"]
    # capture names come in two shapes ("45_<ts>_sphere_10" and
    # "102_sphere_10"); the family is the first alphabetic token
    def _family(r):
        for tok in r["capture"].split("_"):
            if tok.isalpha():
                return tok
        return "other"
    fam = np.array([_family(r) for r in pf])

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    ax = axes[0]
    for f_ in sorted(set(fam)):
        m = fam == f_
        ax.scatter(ft[m], fc[m], s=14, alpha=0.75, label=f_)
    lim = max(ft.max(), fc.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel("FEA ground-truth normal force [N]")
    ax.set_ylabel("our estimate (calibrated) [N]")
    ax.set_title(f"FEATS val split — r={rep['pearson_r']:.2f}, "
                 f"ρ={rep['spearman_rho']:.2f} (n={rep['n_frames']})")
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1]
    order = np.argsort(ft)
    ax.plot(ft[order], np.abs(fc - ft)[order], ".", ms=4, alpha=0.6,
            color="#7570b3")
    ax.set_xlabel("FEA ground-truth force [N]")
    ax.set_ylabel("|error| [N]")
    ax.set_title(f"per-frame error — MAE {rep['mae_calibrated_n']:.1f} N "
                 "(transfer setting)")
    fig.suptitle("External validation on FEATS (marker gel, CNC-pressed, "
                 "FEA force labels)", fontsize=10)
    out = ASSETS / "feats_validation.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    return out


def overlay_clip(task: str, date: str, ep: str, side: str,
                 seconds: float = 12.0, out_fps: int = 30) -> Path:
    """Tactile view + live force bar + timeline cursor around the peak."""
    import h5py
    import hdf5plugin  # noqa: F401

    from .evaluate import median3_fresh

    npz, _, is_new, _ = _load(task, date, ep, side)
    force = median3_fresh(npz["force_normal_n"], is_new)
    trim = int(npz["trim"])
    peak = int(np.argmax(force))
    half = int(seconds * 30 / 2)
    lo, hi = max(0, peak - half), min(len(force), peak + half)
    fmax = max(float(force.max()), 1e-3)

    out = ASSETS / f"clip_{task}_{ep}_{side}.mp4"
    tmp = out.with_suffix(".raw.mp4")
    W_, H_ = 640, 560
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"),
                         out_fps, (W_, H_))
    with h5py.File(str(DATA_ROOT / task / date / f"{ep}.h5"), "r") as f:
        frames = f[f"gelsight/{side}/frames"]
        nmax = len(frames) - 1
        for row in range(lo, hi):
            img = frames[min(trim + row + LEGACY_SHIFT, nmax)]
            canvas = np.zeros((H_, W_, 3), np.uint8)
            canvas[:480] = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # force bar
            frac = force[row] / fmax
            cv2.rectangle(canvas, (40, 520), (600, 540), (60, 60, 60), 1)
            cv2.rectangle(canvas, (40, 520), (40 + int(560 * frac), 540),
                          (30, 120, 240), -1)
            cv2.putText(canvas, f"F_n = {force[row]:.3f} N",
                        (40, 512), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (240, 240, 240), 1, cv2.LINE_AA)
            # mini timeline
            xs = np.linspace(40, 600, hi - lo).astype(int)
            ys = (556 - 12 * force[lo:hi] / fmax).astype(int)
            for i in range(1, len(xs)):
                cv2.line(canvas, (xs[i - 1], ys[i - 1]), (xs[i], ys[i]),
                         (140, 140, 140), 1)
            cv2.circle(canvas, (xs[row - lo], ys[row - lo]), 3,
                       (30, 120, 240), -1)
            vw.write(canvas)
    vw.release()
    import subprocess
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp),
                    "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(out)], check=True)
    tmp.unlink()
    return out
