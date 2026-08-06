"""Showcase gallery: raw image | depth map | 3D surface | force readout.

LUT-v2 edition — every reconstruction here goes through the classic
per-sensor calibration (`stages()` in debug_gallery: dI = img - ref ->
RGB LUT -> valid mask -> fast_poisson depth), NOT the retired gsrobotics
MLP whose depth amplitude was broken (~25% of true indentation).

Every 3D view on the site is an Open3D TriangleMesh render (`o3d_view`), not a
matplotlib `plot_surface`: full-resolution mm-scale grid mesh with gradient-grey
shading baked into the vertex colours. The halo pedestal is removed first
(`remove_halo_pedestal`) — without it the diffuse `|dI| > 8` skirt integrates
into a broad base and every press renders as a rounded mound.

MARKER GELS: the depth and mesh panels go through `marker_removal.stages_depth`
(dots inpainted on reference AND frame before differencing), which strips the
dimple lattice from the depth map — 1.523 -> 0.890 lattice power on FEATS,
lower on 91% of frames. The FORCE numbers in the same panels do NOT: they come
from the untouched `stages()`, because on identical splits marker removal costs
0.00-0.06 rho (0.7747 -> 0.7371) and every paired median delta is negative.
On markerless gels (cnc, GlowTact, React) the detector returns zero dots and
`stages_depth` is a bit-exact no-op, so nothing there changed.

RUN REQUIREMENT: Open3D 0.15.2 segfaults offscreen without an X display here,
so every entry point that renders a 3D panel must be invoked as

    xvfb-run -a -s "-screen 0 1400x1000x24" python -m force_recovery.showcase

Products:
  image_samples()   20 stills from the GT datasets (cnc_Mini + FEATS):
                    raw frame | depth heat map | Open3D mesh | predicted
                    vs ground-truth force. Predictions use the published
                    per-group half/half least-squares + isotonic models
                    (gain-field features on cnc, plain features on FEATS).
  video_samples()   10 clips from React episodes around their force peaks:
                    tactile stream | live LUT depth | Open3D mesh |
                    running force trace, force in GlowTact-calibrated newtons.
                    The mesh panel is re-rendered on every *fresh* tactile
                    frame (the sensor's own update rate, ~15 Hz against the
                    30 Hz row clock) and held on duplicated rows, exactly like
                    the depth panel next to it.
  react_showcase()  full-episode LUT force for motherboard/episode_000 left:
                    depth_validation_panel.png (index page), dexforce figure,
                    overlay clip (actions page) and action_trace.json.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .debug_gallery import (OUT as DEBUG_OUT, feat_vec, load_cnc, load_feats,
                            stages)
from .lut_calibration import MM_PER_PIXEL, crop
from .marker_removal import marker_mask, stages_depth
from .o3d_view import (crop_to_content, has_display, remove_halo_pedestal,
                       render_depth_mesh)
from .run_episode import DATA_ROOT, LEGACY_SHIFT, OUT_ROOT, STAGE_ROOT

ASSETS = OUT_ROOT / "site_assets"
SITE_ASSETS = OUT_ROOT / "site" / "assets"
GALLERY = SITE_ASSETS / "gallery"


# ------------------------------------------------------------------ fitting
def _fit_pred(X: np.ndarray, f: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Per-group half/half least-squares + isotonic; NaN on the fit half."""
    from sklearn.isotonic import IsotonicRegression

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


def _basis(x, y):
    return np.column_stack([np.ones_like(x), x, y, x * x, y * y, x * y])


def _glowtact_calib():
    """Sphere-supervised force model on the LUT's home sensor (GlowTact).

    The published React-range calibration (method page): GlowTact `round`
    presses, strictly in view, 0-8 N, with the pixel-space spatial gain
    field — holdout rho 0.98, MAE 0.24 N. Pooled multi-object fits are much
    worse (rho 0.47): the volume->force gain is object-dependent, which is
    exactly why the sphere (known geometry) is the calibration object.

    Returns predict(stages_dict) -> force [N]; the gain field is evaluated
    at the depth-weighted contact centroid of the frame itself.
    """
    from scipy.stats import spearmanr
    from sklearn.isotonic import IsotonicRegression

    rows = json.loads(
        (OUT_ROOT / "feature_cache" / "lut_full.json").read_text())
    rows = [r for r in rows if r["f"] > 0.15
            and np.isfinite(r.get("cx", float("nan")))]
    a = lambda k: np.array([r[k] for r in rows])
    x, y, z, f = a("x"), a("y"), a("z"), a("f")
    V, V2, A, D, cx, cy = (a("vol"), a("vol2"), a("area"), a("maxd"),
                           a("cx"), a("cy"))
    grp = np.array([r["fam"] for r in rows])

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
    sc = ((cx - r_eff > 24) & (cx + r_eff < 296) & (cy - r_eff > 20)
          & (cy + r_eff < 220) & (z <= 4.2) & (grp == "round")
          & (x > 3.5) & (x < 14.5) & (y > 3.0) & (y < 13.5) & (f <= 8.0))
    wl, *_ = np.linalg.lstsq(X[sc], f[sc], rcond=None)
    iso = IsotonicRegression(out_of_bounds="clip").fit(X[sc] @ wl, f[sc])
    rho = spearmanr(iso.predict(X[sc] @ wl), f[sc]).statistic
    print(f"  glowtact sphere calibration (in-view, 0-8 N): n={sc.sum()} "
          f"in-sample rho={rho:.3f}")

    def predict(st: dict) -> float:
        ft = st["feats"]
        if ft["area"] < 1.0:                    # no measurable contact
            return 0.0
        d = st["depth"]
        mm = d > 0.05
        yy, xx = np.nonzero(mm)
        wgt = d[mm]
        pcx = float((xx * wgt).sum() / wgt.sum())
        pcy = float((yy * wgt).sum() / wgt.sum())
        uu = float(u_at(pcx, pcy)[0])
        v = np.array([ft["vol"] * uu, ft["vol2"] * uu ** 2,
                      ft["maxd"] * uu,
                      np.sqrt(max(ft["area"], 0.0)) * ft["maxd"] * uu,
                      ft["area"]])
        return float(max(0.0, iso.predict([float(v @ wl)])[0]))

    return predict


# ------------------------------------------------------------------ 3D view
# Verified on GlowTact star/triangle/quad presses: this camera and z_scale make
# the indenter geometry readable; a shallower front or z_scale=1 flattens it.
MESH_KW = dict(smooth_px=3, z_scale=1.6, front=(0.0, 0.32, 0.95), zoom=0.66)


def mesh_view(depth: np.ndarray, width: int = 700, height: int = 560,
              stride: int = 1, bg: float = 1.0) -> np.ndarray:
    """Open3D mesh render of one LUT depth map -> RGB uint8.

    `remove_halo_pedestal` is mandatory here: it is a render-only correction
    (the force features never see it), but without it the halo skirt lifts a
    broad base under the imprint and the shape reads as a mound.
    """
    dp = remove_halo_pedestal(np.clip(depth, 0.0, None).astype(np.float32))
    rgb = render_depth_mesh(dp, MM_PER_PIXEL, stride=stride, bg=bg,
                            width=width, height=height, **MESH_KW)
    # Open3D fits the camera to the scene bounding sphere, so a flat wide pad
    # renders with a large empty margin and the mesh ends up filling only
    # ~15% of its figure panel. Trim to the rendered content.
    return crop_to_content(rgb)


def mesh_tile(depth: np.ndarray, w: int = 320, h: int = 240, stride: int = 2,
              bg: float = 0.05, oversample: float = 1.35) -> np.ndarray:
    """Video-sized mesh tile: render larger, then centre-crop.

    The camera looks at the mesh centre, so a centre crop is exactly a zoom
    about the imprint — it enlarges the geometry without touching the verified
    view direction. Only the empty margin around the gel plane is lost.
    """
    import cv2

    rgb = mesh_view(depth, width=int(w * oversample),
                    height=int(h * oversample), stride=stride, bg=bg)
    # mesh_view already trims to content, so the tile is produced by fitting
    # that crop into the tile (letterboxed on the background colour) rather
    # than by a fixed centre crop, which would now cut the geometry itself.
    sc = min(w / rgb.shape[1], h / rgb.shape[0])
    rs = cv2.resize(rgb, (max(int(rgb.shape[1] * sc), 1),
                          max(int(rgb.shape[0] * sc), 1)))
    tile = np.full((h, w, 3), int(round(bg * 255)), np.uint8)
    y0, x0 = (h - rs.shape[0]) // 2, (w - rs.shape[1]) // 2
    tile[y0:y0 + rs.shape[0], x0:x0 + rs.shape[1]] = rs
    return tile


# ------------------------------------------------------------------ panels
def _panel(img: np.ndarray, depth: np.ndarray, f_pred: float,
           f_true: float | None, title: str, out: Path,
           ref: np.ndarray | None = None, st: dict | None = None,
           depth_note: str = "") -> None:
    """Every stage the force number is built from, left to right:

        raw | reference | dI = img-ref | valid mask | |LUT gradient| |
        depth [mm] | Open3D mesh | predicted vs ground truth

    The middle four are what the pipeline actually operates on, and they are
    what makes a wrong result diagnosable: on a marker gel the mask speckles
    and the gradient shows a ring at every dot, which no amount of looking at
    the raw frame reveals.

    `depth_note` labels the two rightmost geometry panels when they come from
    the marker-inpainted depth path — they do, on marker gels, while the force
    bar beside them does not (see the module docstring).
    """
    d = np.clip(depth, 0, None)
    vmax = max(float(d.max()), 0.05)
    have = ref is not None and st is not None
    ncol = 8 if have else 4
    fig = plt.figure(figsize=(2.55 * ncol, 3.1))
    k = 1

    def add(data, title_, cmap=None, colorbar=False):
        nonlocal k
        a = fig.add_subplot(1, ncol, k)
        im = a.imshow(data, cmap=cmap)
        a.set_title(title_, fontsize=8)
        a.axis("off")
        if colorbar:
            fig.colorbar(im, ax=a, fraction=0.046)
        k += 1

    add(np.clip(img, 0, 255).astype(np.uint8), "raw GelSight frame")
    if have:
        add(np.clip(ref, 0, 255).astype(np.uint8), "reference (no contact)")
        # x3 gain: dI is only a few gray levels away from contact, and the
        # site figure has to show that it is NOT zero there either. Taken
        # from `st` rather than recomputed, so on a marker gel this and every
        # panel right of it show the inpainted (dot-free) chain while the raw
        # panel still shows the dots.
        add(np.clip(st["dI"] * 3 + 128, 0, 255).astype(np.uint8),
            f"difference image  dI = img − ref  (×3){depth_note}")
        add(st["valid"], "valid mask  |dI| > 8", cmap="gray")
        add(st["gmag"], "|LUT surface gradient|", cmap="magma")
    add(d, f"LUT depth [mm]{depth_note}", cmap="inferno", colorbar=True)
    add(mesh_view(d), f"3D reconstruction (Open3D mesh){depth_note}")

    ax = fig.add_subplot(1, ncol, k)
    bars = [("predicted", f_pred, "#d95f02")]
    if f_true is not None:
        bars.append(("ground truth", f_true, "#1b9e77"))
    ax.bar([b[0] for b in bars], [b[1] for b in bars],
           color=[b[2] for b in bars], width=0.55)
    ax.set_ylabel("normal force [N]", fontsize=8)
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def _require_display() -> None:
    """Fail before the (minutes-long) feature pass, not after it."""
    if not has_display():
        raise RuntimeError(
            "the 3D panels need Open3D offscreen rendering, which segfaults "
            "without a display here — rerun under: xvfb-run -a "
            "-s '-screen 0 1400x1000x24'")


def image_samples(n_cnc: int = 14, n_feats: int = 6) -> list[Path]:
    from scipy.stats import spearmanr

    _require_display()
    GALLERY.mkdir(parents=True, exist_ok=True)
    outs = []

    # --- cnc_Mini, strictly in-view presses, gain-field features (the
    #     published LUT-v2 model: in-view rho ~0.94)
    import os
    os.environ.setdefault("CNC_N", "1400")
    frames, get = load_cnc()
    rows, depths = [], []
    for fr in frames:
        img, ref = get(fr)
        st = stages(img, ref)
        rows.append({**fr, **st["feats"]})
    a = lambda k: np.array([r[k] for r in rows])
    x, y, zc, f = a("x"), a("y"), a("z"), a("f")
    grp = np.array([r["group"] for r in rows])
    inner = (x > 5) & (x < 13) & (y > 4) & (y < 12)
    PHI = _basis(x[inner], y[inner])
    w, *_ = np.linalg.lstsq(np.hstack([PHI * zc[inner][:, None], -PHI]),
                            a("maxd")[inner], rcond=None)
    u = 1.0 / np.clip(_basis(x, y) @ w[:6], 0.15, 3.0)
    X = np.column_stack([a("vol") * u, a("vol2") * u ** 2, a("maxd") * u,
                         a("area"),
                         np.sqrt(np.clip(a("area"), 0, None)) * a("maxd") * u])
    pred = np.full(len(f), np.nan)
    ii = np.where(inner)[0]
    pred[ii] = _fit_pred(X[ii], f[ii], grp[ii])
    ok = np.isfinite(pred)
    rho = spearmanr(pred[ok], f[ok]).statistic
    mae = float(np.abs(pred[ok] - f[ok]).mean())
    print(f"  cnc in-view: n={ok.sum()} rho={rho:.3f} MAE={mae:.2f}N")

    cand = np.where(ok)[0]
    order = cand[np.argsort(f[cand])]
    picks = order[np.linspace(0, len(order) - 1, n_cnc).astype(int)]
    for k, i in enumerate(picks):
        img, ref = get(rows[i])
        # geometry panels take the depth path; cnc is markerless, so this is
        # a bit-exact no-op here and the call is kept only so both datasets
        # go through one code path
        st = stages_depth(img, ref)
        out = GALLERY / f"cnc_{k:02d}.png"
        _panel(img, st["depth"], float(pred[i]), float(f[i]),
               f"cnc_Mini {rows[i]['group']}  z={rows[i]['z']:.1f} mm",
               out, ref=ref, st=st)
        outs.append(out)

    # --- FEATS val, marker gel. Force features come from the UNCHANGED
    #     stages(); only the displayed geometry is marker-inpainted.
    frames, get = load_feats()
    rows = []
    for fr in frames:
        img, ref = get(fr)
        rows.append({**fr, **stages(img, ref)["feats"]})
    Xf = np.array([feat_vec(r) for r in rows])
    ff = np.array([r["f"] for r in rows])
    predf = _fit_pred(Xf, ff, np.array([r["group"] for r in rows]))
    okf = np.isfinite(predf)
    rhof = spearmanr(predf[okf], ff[okf]).statistic
    maef = float(np.abs(predf[okf] - ff[okf]).mean())
    print(f"  feats: n={okf.sum()} rho={rhof:.3f} MAE={maef:.2f}N")

    cand = np.where(okf)[0]
    order = cand[np.argsort(ff[cand])]
    picks = order[np.linspace(0, len(order) - 1, n_feats).astype(int)]
    _, ref0 = get(rows[picks[0]])
    dots = marker_mask(ref0)
    note = "\nmarkers inpainted (geometry only)" if dots is not None else ""
    print(f"  feats marker dots on the reference: "
          f"{'detected' if dots is not None else 'none'}")
    for k, i in enumerate(picks):
        img, ref = get(rows[i])
        st = stages_depth(img, ref)
        out = GALLERY / f"feats_{k:02d}.png"
        _panel(img, st["depth"], float(predf[i]), float(ff[i]),
               f"FEATS {rows[i]['group']}", out, ref=ref, st=st,
               depth_note=note)
        outs.append(out)

    (GALLERY / "metrics.json").write_text(json.dumps({
        "cnc_in_view": {"n": int(ok.sum()), "rho": float(rho), "mae_n": mae},
        "feats": {"n": int(okf.sum()), "rho": float(rhof), "mae_n": maef},
        "feats_geometry": "marker-inpainted depth/mesh; force from stages()",
        "pipeline": "lut_v2"}, indent=2))
    return outs


# ------------------------------------------------------------------ React
def _react_context(task: str, date: str, ep: str, side: str):
    """npz + parquet columns + reference image (cropped, float32)."""
    import h5py
    import hdf5plugin  # noqa: F401
    import pyarrow.parquet as pq

    z = np.load(OUT_ROOT / task / date / f"{ep}_{side}.npz")
    table = pq.read_table(
        str(STAGE_ROOT / task / "meta" / date / f"{ep}.parquet"),
        columns=[f"tactile_{side}_is_new", f"tactile_{side}_intensity",
                 f"sensor_{side}_pose"])
    is_new = np.asarray(table[f"tactile_{side}_is_new"].to_numpy())
    inten = np.asarray(table[f"tactile_{side}_intensity"].to_numpy())
    pose = np.stack(table[f"sensor_{side}_pose"].to_numpy())
    trim = int(z["trim"])
    h5 = h5py.File(str(DATA_ROOT / task / date / f"{ep}.h5"), "r")
    frames = h5[f"gelsight/{side}/frames"]
    nmax = len(frames) - 1

    def frame(row: int) -> np.ndarray:
        return frames[min(trim + int(row) + LEGACY_SHIFT, nmax)]

    ref = np.median(np.stack([
        crop(frame(int(r))).astype(np.float32)
        for r in z["reference_rows"][:6]]), 0)
    return z, is_new, inten, pose, frame, ref, h5


def _lut_force_rows(frame, ref, calib, rows, is_new,
                    keep_depth: bool = False):
    """LUT force for the fresh rows in `rows`; forward-fill duplicates.

    Force always comes from the untouched `stages()`. The depth kept for the
    video panels goes through the marker path instead — a bit-exact no-op on
    React's markerless gel (the detector finds no dots), present so a marker
    sensor would get clean geometry without changing the force it reports.
    """
    dots = marker_mask(ref)
    force = np.zeros(len(is_new))
    depths = {}
    last = 0.0
    for row in rows:
        if is_new[row] or row == rows[0]:
            img = crop(frame(row)).astype(np.float32)
            st = stages(img, ref)
            last = calib(st)
            if keep_depth:
                depths[row] = (stages_depth(img, ref, mask=dots)["depth"]
                               if dots is not None else st["depth"])
        force[row] = last
    return force, depths


def video_samples(n: int = 10, seconds: float = 8.0,
                  mesh_stride: int = 2) -> list[Path]:
    """React clips: tactile | live LUT depth | Open3D mesh | force trace.

    The mesh panel costs ~0.2 s/frame, so it is rendered once per *fresh*
    tactile frame (never per duplicated row) and held in between — the same
    update rate the depth panel already has. `mesh_stride` decimates the mesh
    grid (2 = every other pixel) purely for speed at the 320x256 panel size.
    """
    from .evaluate import median3_fresh

    _require_display()
    GALLERY.mkdir(parents=True, exist_ok=True)
    calib = _glowtact_calib()
    npzs = sorted(Path(OUT_ROOT).rglob("episode_*_*.npz"),
                  key=lambda p: -float(np.load(p)["force_normal_n"].max()))
    outs = []
    for npz_path in npzs[:n]:
        task = npz_path.parent.parent.name
        date = npz_path.parent.name
        ep, side = npz_path.stem.rsplit("_", 1)
        if not (DATA_ROOT / task / date / f"{ep}.h5").exists():
            continue
        z, is_new, _, _, frame, ref, h5 = _react_context(task, date, ep, side)
        old = median3_fresh(z["force_normal_n"], is_new)
        peak = int(np.argmax(old))            # window choice only
        half = int(seconds * 30 / 2)
        lo, hi = max(0, peak - half), min(len(old), peak + half)

        force, depths = _lut_force_rows(frame, ref, calib,
                                        list(range(lo, hi)), is_new,
                                        keep_depth=True)
        fwin = median3_fresh(force[lo:hi], is_new[lo:hi])
        fmax = max(float(fwin.max()), 1e-3)
        vmax = max(0.2, max(float(d.max()) for d in depths.values()))

        # one Open3D render per fresh depth frame, reused on duplicated rows
        meshes = {r: cv2.cvtColor(mesh_tile(d, stride=mesh_stride),
                                  cv2.COLOR_RGB2BGR)
                  for r, d in sorted(depths.items())}

        out = GALLERY / f"clip_{task}_{ep}_{side}.mp4"
        tmp = out.with_suffix(".raw.mp4")
        W_, H_ = 1280, 330
        vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"),
                             30, (W_, H_))
        dep = np.zeros((240, 320), np.float32)
        msh = np.full((240, 320, 3), 13, np.uint8)
        for row in range(lo, hi):
            img = frame(row)
            dep = depths.get(row, dep)
            msh = meshes.get(row, msh)
            canvas = np.zeros((H_, W_, 3), np.uint8)
            canvas[:240, :320] = cv2.cvtColor(
                cv2.resize(img, (320, 240)), cv2.COLOR_RGB2BGR)
            dn = (np.clip(dep / vmax, 0, 1) * 255).astype(np.uint8)
            canvas[:240, 320:640] = cv2.applyColorMap(dn,
                                                      cv2.COLORMAP_INFERNO)
            canvas[:240, 640:960] = msh
            xs = np.linspace(970, 1270, hi - lo).astype(int)
            ys = (200 - 170 * fwin / fmax).astype(int) + 20
            for i in range(1, len(xs)):
                cv2.line(canvas, (xs[i - 1], ys[i - 1]), (xs[i], ys[i]),
                         (150, 150, 150), 1)
            cv2.circle(canvas, (xs[row - lo], ys[row - lo]), 4,
                       (30, 120, 240), -1)
            for x0, lab in ((0, "tactile"), (320, "LUT depth"),
                            (640, "3D mesh (Open3D)"), (970, "force")):
                cv2.putText(canvas, lab, (x0 + 8, 262),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"{fwin[row - lo]:.2f} N  "
                        f"({task}/{ep} {side})", (8, 300),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (240, 240, 240), 1, cv2.LINE_AA)
            vw.write(canvas)
        vw.release()
        h5.close()
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp),
                        "-c:v", "libx264", "-crf", "24", "-pix_fmt",
                        "yuv420p", "-movflags", "+faststart", str(out)],
                       check=True)
        tmp.unlink()
        outs.append(out)
        print(f"  clip {out.name}  peak {fwin.max():.2f} N  "
              f"{len(meshes)} mesh renders / {hi - lo} rows", flush=True)
    return outs


def react_showcase(task: str = "motherboard", date: str = "2026-05-10",
                   ep: str = "episode_000", side: str = "left") -> None:
    """Full-episode LUT force -> index panel, dexforce figure, clip, trace."""
    from .dexforce import force_informed_targets, gel_axis
    from .evaluate import median3_fresh
    from .visualize import dexforce_figure, overlay_clip

    calib = _glowtact_calib()
    z, is_new, inten, pose, frame, ref, h5 = _react_context(
        task, date, ep, side)
    raw_force, _ = _lut_force_rows(frame, ref, calib,
                                   list(range(len(is_new))), is_new)
    force = median3_fresh(raw_force, is_new)
    print(f"  {task}/{date}/{ep} {side}: LUT force max {force.max():.2f} N")

    # --- depth_validation_panel.png: three strong presses >=5 s apart
    order = np.argsort(-force)
    rows, taken = [], []
    for r in order:
        if is_new[r] and all(abs(int(r) - t) > 150 for t in taken):
            rows.append(int(r)); taken.append(int(r))
        if len(rows) == 3:
            break
    fig, axes = plt.subplots(len(rows), 3,
                             figsize=(8.4, 2.15 * len(rows)))
    axes = np.atleast_2d(axes)
    for r_i, row in enumerate(rows):
        img = crop(frame(row)).astype(np.float32)
        st = stages_depth(img, ref)      # markerless React: identical to stages()
        diff = np.abs(img - ref).mean(axis=2)
        for c, (data, cmap, title) in enumerate((
                (np.clip(img, 0, 255).astype(np.uint8), None,
                 f"row {row}  F={force[row]:.2f} N"),
                (diff, "gray", "|frame - reference|"),
                (st["depth"], "inferno",
                 f"LUT depth (max {st['depth'].max():.2f} mm)"))):
            ax = axes[r_i, c]
            ax.imshow(data, cmap=cmap)
            ax.set_title(title, fontsize=8)
            ax.axis("off")
    fig.suptitle(f"{task}/{ep} {side} — LUT-calibrated depth at the "
                 "strongest contacts", fontsize=10)
    panel = ASSETS / "depth_validation_panel.png"
    fig.tight_layout(); fig.savefig(panel); plt.close(fig)
    shutil.copy2(panel, SITE_ASSETS / panel.name)
    print(f"  {panel.name}: rows {rows}")

    # --- dexforce figure + overlay clip with the new force
    fp = dexforce_figure(task, date, ep, side, force=force)
    shutil.copy2(fp, SITE_ASSETS / fp.name)
    cp = overlay_clip(task, date, ep, side, force=force)
    shutil.copy2(cp, SITE_ASSETS / cp.name)
    print(f"  {fp.name}, {cp.name}")

    # --- action_trace.json for the interactive actions page
    act = force_informed_targets(pose, force, gel_axis(task, side))
    peak = int(np.argmax(force))
    N = 2700
    t0 = int(np.clip(peak - N // 2, 0, len(force) - N))
    n_hat = gel_axis(task, side)
    s_obs = (pose[:, :3] @ n_hat) * 1000.0
    sl = slice(t0, t0 + N)
    trace = {
        "meta": {"episode": f"{task}/{date}/{ep}", "side": side, "fps": 30,
                 "stiffness": float(act.stiffness), "t0_row": t0,
                 "peak_row": peak, "force": "LUT v2, GlowTact-calibrated"},
        "force_n": [round(float(v), 3) for v in force[sl]],
        "pen_mm": [round(float(v) * 1000.0, 3)
                   for v in act.penetration_m[sl]],
        "s_obs_mm": [round(float(v - s_obs[sl].min()), 2)
                     for v in s_obs[sl]],
        "intensity": [round(float(v), 1) for v in inten[sl]],
    }
    tj = ASSETS / "action_trace.json"
    tj.write_text(json.dumps(trace))
    shutil.copy2(tj, SITE_ASSETS / tj.name)
    h5.close()
    print(f"  action_trace.json: window rows {t0}..{t0 + N}")


if __name__ == "__main__":
    import sys

    cmds = {"images": image_samples, "videos": video_samples,
            "react": react_showcase}
    for c in (sys.argv[1:] or ["images", "react", "videos"]):
        print(f"== {c}")
        cmds[c]()
