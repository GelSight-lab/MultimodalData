"""Figures and clips for the force-recovery results site.

Everything renders headless into OUT_ROOT/site_assets/.
"""
from __future__ import annotations

from pathlib import Path
import os

import numpy as np

# Compatibility exports for existing figure callers; no estimator imports.
from twm.visualization.force import DIFF_GAIN, diff_rgb, diff_caption

DATA_ROOT = Path(os.environ.get("REACT_DATA_ROOT", "/media/yxma/Disk1/twm/data"))
STAGE_ROOT = Path(os.environ.get("REACT_STAGE_ROOT", "/media/yxma/Disk1/twm/release"))
OUT_ROOT = Path(os.environ.get("REACT_FORCE_RECOVERY_ROOT",
                               "/media/yxma/Disk1/twm/force_recovery"))
ASSETS = OUT_ROOT / "site_assets"


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 130, "font.size": 9, "axes.grid": True,
        "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False,
    })
    return plt


def _load(task, date, ep, side):
    import pyarrow.parquet as pq

    with np.load(OUT_ROOT / task / date / f"{ep}_{side}.npz") as archive:
        npz = {key: archive[key] for key in archive.files}
    table = pq.read_table(
        str(STAGE_ROOT / task / "meta" / date / f"{ep}.parquet"),
        columns=[f"tactile_{side}_intensity", f"tactile_{side}_is_new",
                 f"sensor_{side}_pose"])
    inten = np.asarray(table[f"tactile_{side}_intensity"].to_numpy())
    is_new = np.asarray(table[f"tactile_{side}_is_new"].to_numpy())
    if not len(inten):
        raise ValueError("Force metadata has no rows")
    pose = np.stack(table[f"sensor_{side}_pose"].to_numpy())
    return npz, inten, is_new, pose


def _index_map(path, task, side, rows):
    from twm.react_preprocess.h5io import open_episode

    indices = np.asarray(open_episode(path, task).align[side].index_map)
    if indices.shape != (rows,):
        raise ValueError(f"Tactile alignment length differs from {rows} force rows")
    return indices


def _validate_indices(indices, frames):
    if indices.dtype.kind not in "iu" or np.any((indices < 0) | (indices >= len(frames))):
        raise ValueError("Tactile alignment points outside the recording")


def _force_values(force, rows):
    force = np.asarray(force)
    if force.shape != (rows,) or not rows or not np.isfinite(force).all():
        raise ValueError("Force must be a nonempty finite vector matching the metadata rows")
    return force


def force_timeline(task: str, date: str, ep: str) -> Path:
    """Two-panel force + intensity timeline for one episode."""
    from .evaluate import median3_fresh
    plt = _pyplot()

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

    npz, inten, _, _ = _load(task, date, ep, side)
    force = _force_values(npz["force_normal_n"], len(inten))
    depth_keys = sorted(k for k in npz if k.startswith("depth_row_"))
    if not depth_keys:
        raise ValueError("No saved depth rows are available")
    refs = np.asarray(npz["reference_rows"])
    if refs.ndim != 1 or not len(refs):
        raise ValueError("No reference rows are available")
    ref_row = int(refs[0])
    selected = [int(key.split("_")[-1]) for key in depth_keys[:3]]
    if any(row < 0 or row >= len(force) for row in [ref_row, *selected]):
        raise ValueError("Depth/reference rows are outside the force timeline")
    path = DATA_ROOT / task / date / f"{ep}.h5"
    indices = _index_map(path, task, side, len(force))

    with h5py.File(str(path), "r") as f:
        frames = f[f"gelsight/{side}/frames"]
        _validate_indices(indices, frames)
        ref = frames[int(indices[ref_row])].astype(np.float32)
        rows = []
        for key in depth_keys[:3]:
            row = int(key.split("_")[-1])
            img = frames[int(indices[row])]
            depth = npz[key]
            rows.append((row, img, depth))

    plt = _pyplot()
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
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    return out


def dexforce_figure(task: str, date: str, ep: str, side: str,
                    force: np.ndarray | None = None) -> Path:
    """Virtual-target separation along the pressing direction + histogram.

    Plotting a world coordinate hides the offset (0.2-4 mm against a
    ~100 mm trajectory), so both curves are projected onto the pressing
    direction at the force peak — there the separation IS the penetration.

    `force` overrides the npz force (e.g. the LUT-v2 estimate); default
    falls back to the force stored at episode-processing time.
    """
    from .evaluate import median3_fresh
    from .dexforce import force_informed_targets, gel_axis
    plt = _pyplot()

    npz, _, is_new, pose = _load(task, date, ep, side)
    if force is None:
        force = median3_fresh(npz["force_normal_n"].astype(np.float64),
                              is_new)
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
    plt = _pyplot()

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


def fota_validation_figure() -> Path:
    """Per-capture rank correlation on FoTa, split by gel type."""
    import json
    plt = _pyplot()

    rep = json.loads((OUT_ROOT / "fota_validation_val.json").read_text())
    pc = rep["per_capture"]
    groups = [("markerless gel\n(our domain)", [r for r in pc if not r["markered"]], "#4fd8e0"),
              ("marker-dot gel\n(foreign domain)", [r for r in pc if r["markered"]], "#ffb347")]

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    rng = np.random.default_rng(0)
    for gi, (label, rows, color) in enumerate(groups):
        rhos = np.array([r["spearman_force_vs_depth"] for r in rows])
        x = gi + rng.uniform(-0.13, 0.13, len(rhos))
        ax.scatter(x, rhos, s=26, alpha=0.75, color=color, edgecolors="none")
        med = np.median(rhos)
        ax.hlines(med, gi - 0.24, gi + 0.24, color=color, lw=2.5)
        ax.text(gi + 0.27, med, f"median {med:.2f}", va="center", fontsize=9,
                color=color)
    ax.axhline(0, color="#999", lw=0.8, ls="--")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylabel("Spearman ρ: estimated force vs pose press-depth")
    ax.set_title(f"FoTa (T3) validation — {rep['n_captures']} captures, "
                 "13 household objects pressed by a Panda\n"
                 "(no force GT in FoTa; press depth is the monotone proxy)",
                 fontsize=10)
    out = ASSETS / "fota_validation.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    return out


def overlay_clip(task: str, date: str, ep: str, side: str,
                 seconds: float = 12.0, out_fps: int = 30,
                 force: np.ndarray | None = None) -> Path:
    """Tactile view + live force bar + timeline cursor around the peak.

    `force` overrides the npz force (e.g. the LUT-v2 estimate)."""
    import h5py
    import hdf5plugin  # noqa: F401

    from twm.visualization.export import write_video
    from twm.visualization.force import ForceOverlay

    npz, inten, is_new, _ = _load(task, date, ep, side)
    if force is None:
        from .evaluate import median3_fresh
        _force_values(npz["force_normal_n"], len(inten))
        force = median3_fresh(npz["force_normal_n"], is_new)
    force = _force_values(force, len(inten))
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("seconds must be finite and positive")
    peak = int(np.argmax(force))
    half = max(1, int(seconds * 30 / 2))
    lo, hi = max(0, peak - half), min(len(force), peak + half)
    fmax = max(float(force.max()), 1e-3)
    path = DATA_ROOT / task / date / f"{ep}.h5"
    indices = _index_map(path, task, side, len(force))
    # source_frame in the archive records the held force estimate's source.
    # Reference/depth reconstructions and tactile display use the canonical
    # per-row capture map, also when an overriding force estimate is supplied.
    if "source_frame" in npz and npz["source_frame"].shape != indices.shape:
        raise ValueError("Saved source_frame length differs from force rows")
    overlay = ForceOverlay(force[lo:hi], maximum=fmax)
    out = ASSETS / f"clip_{task}_{ep}_{side}.mp4"

    def panels():
        # Reopen on verification retry; generator close also releases the H5
        # when encoding fails before the final row.
        with h5py.File(str(path), "r") as f:
            frames = f[f"gelsight/{side}/frames"]
            _validate_indices(indices, frames)
            if "source_frame" in npz:
                _validate_indices(npz["source_frame"], frames)
            for row in range(lo, hi):
                yield overlay.render(frames[int(indices[row])], row - lo)

    write_video(out, panels, fps=out_fps, pixel_format="yuv420p")
    return out
