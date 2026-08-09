"""External PER-PIXEL validation of the depth reconstruction on Tactile MNIST.

Everything we have reported so far is self-validated: our reconstruction against
our own analytic sphere cap, with the cap's amplitude anchored on the
reconstruction itself (shape-only). Tactile MNIST gives the missing thing —
exact per-pixel ground-truth depth on NON-spherical geometry (3D-printed MNIST
digits, 6.5-8.2 mm tall, broad low-curvature tops) that a sphere calibration
cannot self-validate.

Stage 1 (this file's main result) uses SimTactileMNIST: Taxim-rendered GelSight
Mini images paired with object/gel poses. The GT depth map is ray-cast from the
mesh, so it is exact up to the pose bookkeeping — which is verified end-to-end
(`verify`) by re-rendering the GT height map with Taxim and comparing against
the shipped image (MAE ~2/255 grey levels).

GEOMETRY, VERIFIED NOT ASSUMED (see `verify`)
  * the touch that produced `sensor_image[i]` used `info.object_pose[i-1]`
    (poses are logged after the per-step object perturbation, sigma 1 mm/2.9 deg);
    with that offset the peak penetration is constant at 2.2490 +- 0.0023 mm
    over 145 touches, with the +0/+1 offsets it scatters by >1 mm.
  * the gel frame origin sits GEL_THICKNESS_MM above the undeformed gel surface
    and the simulator lowers the sensor until the closest object point is
    GEL_DATUM_MM below the origin => every sim touch is a 2.25 mm press.
  * image row increases along -y of the gel frame, column along +x.

SCALE / RESOLUTION (the mismatch the task asks to resolve explicitly)
  sim images are 320x240 over the full 18.88 x 14.16 mm sensing area
  (0.059 mm/px); our pipeline assumes MM_PER_PIXEL = 18.6*(5/7)/320 = 0.041518
  because it always runs on a 5/7 centre crop of a 640x480 Mini frame. Instead
  of re-scaling the pipeline (which would change every published number) we
  resample the sim frame to 455x341, at which 1 px == MM_PER_PIXEL to 0.05%,
  and render the GT on the same grid. That keeps the whole sensing surface AND
  an exact scale. Every table also carries a "deployment crop" row that repeats
  the evaluation through the pipeline's own crop() (central 13.57 x 10.03 mm,
  51% of the pad) as a robustness check.

TRACKS
  A  the shipped pipeline: debug_gallery.stages() with the GlowTact LUT.
     Answers "does a photometric table calibrated on a real Mini transfer to
     Taxim renders at all".
  B  same solver, same code path, but with the LUT re-fitted on Taxim sphere
     presses (`simlut`) exactly the way lut_calibration fits the real one.
     Removes the photometric domain gap so the geometric defects (over-doming,
     unobserved bins, halo mask) are measured against exact GT rather than
     against the domain gap. Held-out sphere renders are the in-domain control.

WHAT IT FOUND (n=420 touches, 106 objects; um)
  * the sim's photometry drives OUR real-sensor table as well as the real
    sensor does: LUT-vs-true gradient angle 24.4 deg on Taxim renders vs
    26.1 deg on GlowTact's own sphere presses (same statistic, same code).
    So the domain gap is NOT the failure mode.
  * 420/420 touches have a contact that runs off the sensor edge (a 2.25 mm
    press into an 8 mm digit whose footprint is ~3x the pad). fast_poisson's
    zero boundary cannot represent that: the C1 control moves one sphere cap
    from the middle of the pad to the edge and the peak collapses 1.39 -> 0.30
    mm against a 0.90 mm truth.
  * digits: MAE 273 / Type-1 86 / Type-2 884 unfitted (383 / - / 1407 if you
    predict zero everywhere). Enclosed sphere caps, no fitting at all:
    MAE 38 / Type-2 292 - the same order as 3D Cal's 22-49 / 153-290, which
    are quoted WITH a 2D alignment and a fitted indentation scale.
  * over-doming is real but small: centre/rim 1.07 against a plateau whose
    truth is 1.000 (+7%), 1.56 vs 1.40 on the digits (+12%) - not the
    +23..42% the flat-top indenters suggested.
  * unobserved LUT bins (14-17%) correlate only 0.10-0.29 with Type-2 error.
  * the |dI|>8 valid mask reaches IoU 0.614 / recall 0.917 against the true
    contact region while over-segmenting by 53% of its area.
  * `sweep` re-presses the SAME digit meshes at 0.3..2.25 mm (Taxim reproduces
    the shipped images to 1.76/255, and the re-rendered 2.25 mm row reproduces
    the stage1 numbers, so this is a legitimate counterfactual): MAE / Type-2
    goes 11/97 -> 35/186 -> 68/309 -> 127/515 -> 281/962 um and the peak ratio
    1.00 -> 0.55. The digits are not the problem, the 2.25 mm press is: at
    <=0.6 mm we beat 3D Cal's published numbers on non-spherical GT with no
    fitting at all.

Run:
  python -m force_recovery.mnist_validation verify    # geometry + Taxim repro
  python -m force_recovery.mnist_validation simlut    # fit the sim-domain LUT
  python -m force_recovery.mnist_validation stage1    # main evaluation
  python -m force_recovery.mnist_validation report    # aggregate + print table
  python -m force_recovery.mnist_validation diagnose  # gradient angle + FOV
  python -m force_recovery.mnist_validation controls  # C1 clipping, C2 plateau
  python -m force_recovery.mnist_validation sweep     # same digits, 0.3-2.25mm
  python -m force_recovery.mnist_validation figures   # 8-sample comparison PNG
"""
from __future__ import annotations

import io
import json
import sys
import types
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1]))
from force_recovery.debug_gallery import stages                    # noqa: E402
from force_recovery.recon_study import flat_top_sag, stages_full   # noqa: E402
from force_recovery.lut_calibration import (                       # noqa: E402
    BINS, CAL_OUT, DI_RANGE, MM_PER_PIXEL, crop, fill_lut_holes)

OUT = Path("/media/yxma/Disk1/twm/force_recovery/mnist_validation")
SIM_DIR = Path("/media/yxma/Disk1/yuxiang/mini_data/markerless/"
               "SimTactileMNIST/data")
REAL_DIR = Path("/media/yxma/Disk1/yuxiang/mini_data/markerless/"
                "RealTactileMNIST/data")
MESH_PQ = OUT / "meshes/data/printed_train-00000-of-00001.parquet"
MESH_PQ_TEST = OUT / "meshes/data/printed_test-00000-of-00001.parquet"

SENSOR_MM = np.array([18.88, 14.16])       # GelSight Mini sensing area
SIM_W, SIM_H = 320, 240                    # native sim frame
W_P, H_P = 455, 341                        # 1 px == MM_PER_PIXEL (0.05% off)
GEL_THICKNESS_MM = 4.25                    # tactile_mnist constant
GEL_DATUM_MM = 2.0                         # verified from the data (see above)
PRESS_MM = GEL_THICKNESS_MM - GEL_DATUM_MM  # 2.25 mm, constant for every touch
N_TOUCHES = 420
RNG = np.random.default_rng(0)


# ------------------------------------------------------------------ taxim
def _shim_torch_scatter():
    """taxim imports torch_scatter for one scatter_min in the shadow path."""
    if "torch_scatter" in sys.modules:
        return
    import torch
    mod = types.ModuleType("torch_scatter")

    def scatter_min(src, index, dim=-1, out=None, dim_size=None):
        idx = index.expand(src.shape) if index.dim() < src.dim() else index
        out.scatter_reduce_(dim, idx, src, reduce="amin", include_self=True)
        return out, None

    mod.scatter_min = scatter_min
    sys.modules["torch_scatter"] = mod
    _ = torch


_TAXIM = {}


def taxim(device: str = "cpu"):
    """The exact renderer tactile_mnist uses (Mini calib, contact_scale 0.6)."""
    if device not in _TAXIM:
        _shim_torch_scatter()
        import torch
        from taxim import CALIB_GELSIGHT_MINI
        from taxim.taxim_torch import TaximTorch
        _TAXIM[device] = TaximTorch(
            calib_folder=CALIB_GELSIGHT_MINI, device=torch.device(device),
            params={"simulator": {"contact_scale": 0.6}})
    return _TAXIM[device]


def taxim_render(height_map: np.ndarray, device: str = "cpu") -> np.ndarray:
    return (taxim(device).render(height_map.astype(np.float32),
                                 with_shadow=True) * 255.0).astype(np.float32)


def taxim_deformation(height_map: np.ndarray, device: str = "cpu"
                      ) -> np.ndarray:
    """The gel surface Taxim actually rendered, in mm of indentation (>=0).

    Our pipeline reconstructs the GEL surface, not the object; Taxim's pyramid
    soft-body model makes the two differ. Reported alongside the geometric GT.
    """
    import torch
    tx = taxim(device)
    hm = torch.from_numpy(height_map[None].astype(np.float32)).to(tx.device)
    with torch.no_grad():
        deformed, _ = tx._TaximTorch__compute_gel_pad_deformation(hm)
    return -deformed[0].cpu().numpy()


_BG = {}


def background(shape=(H_P, W_P), device: str = "cpu") -> np.ndarray:
    """Reference frame: the no-contact Taxim image.

    Chosen over a median-of-N-frames reference because it is exact — Taxim adds
    the calibrated background to a zero-gradient field, so a touch with no
    contact reproduces it bit-for-bit. `verify` reports the agreement with the
    median of 200 shipped frames (the reference we would be forced to use if we
    only had the images).
    """
    key = (shape, device)
    if key not in _BG:
        _BG[key] = taxim_render(np.full(shape, 50.0, np.float32), device)
    return _BG[key]


# ------------------------------------------------------------------ geometry
def quat_to_R(q) -> np.ndarray:
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def load_meshes(pq_path: Path = MESH_PQ) -> dict:
    import pyarrow.parquet as pq
    d = pq.read_table(pq_path).to_pandas()
    return {int(r["id"]): (np.stack(r["mesh.vertices"]).astype(np.float64),
                           np.stack(r["mesh.faces"]).astype(np.uint32))
            for _, r in d.iterrows()}


def _grid(w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """Pixel-centre (x, y) in the gel frame [m]; row -> -y, col -> +x."""
    px, py = SENSOR_MM[0] / w * 1e-3, SENSOR_MM[1] / h * 1e-3
    X, Y = np.meshgrid((np.arange(w) + 0.5 - w / 2) * px,
                       -(np.arange(h) + 0.5 - h / 2) * py)
    return X, Y


def gt_height_map(V, F, obj_pos, obj_quat, gel_pos, gel_quat,
                  w: int = W_P, h: int = H_P) -> np.ndarray:
    """Taxim-convention height map [mm]: <=0 is in contact with the gel.

    Orthographic ray cast of the mesh along the gel -z axis. Value is the
    distance of the object below the undeformed gel surface, negated:
    hm = (gel_z - z_surface)*1000 - GEL_THICKNESS_MM.
    """
    import open3d as o3d
    Vc = (V @ quat_to_R(obj_quat).T) + obj_pos
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.core.Tensor(Vc.astype(np.float32)),
                     o3d.core.Tensor(F))
    X, Y = _grid(w, h)
    pts = np.stack([X, Y, np.zeros_like(X)], -1) @ quat_to_R(gel_quat).T \
        + np.asarray(gel_pos)
    org = (pts + np.array([0, 0, 0.06])).astype(np.float32)
    rays = np.concatenate(
        [org, np.tile(np.float32([0, 0, -1]), (h, w, 1))], -1)
    t_hit = sc.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
    hit = np.isfinite(t_hit)
    z_surf = np.where(hit, org[..., 2] - t_hit, -1e3)
    hm = (np.asarray(gel_pos)[2] - z_surf) * 1000.0 - GEL_THICKNESS_MM
    return np.clip(hm, None, 50.0).astype(np.float32)


def up(img: np.ndarray) -> np.ndarray:
    """Sim frame (0.059 mm/px) -> pipeline scale (MM_PER_PIXEL/px)."""
    return cv2.resize(img, (W_P, H_P), interpolation=cv2.INTER_LINEAR)


# ------------------------------------------------------------------ touches
def iter_touches(shards, rows_per_shard=40, touches_per_row=4, meshes=None,
                 min_contact=0.02, rng=RNG):
    """Yield dicts with the sim image and the exact GT for one touch."""
    import pyarrow.parquet as pq
    cols = ["object_id", "label", "info.object_pose.position",
            "info.object_pose.quaternion", "gel_pose_cell_frame.position",
            "gel_pose_cell_frame.quaternion", "sensor_image"]
    for sh in shards:
        pf = pq.ParquetFile(sh)
        n_groups = pf.metadata.num_row_groups
        for gi in range(n_groups):
            t = pf.read_row_group(gi, columns=cols).to_pandas()
            idx = rng.permutation(len(t))[:rows_per_shard]
            for ri in idx:
                r = t.iloc[int(ri)]
                oid = int(r["object_id"])
                if meshes is None or oid not in meshes:
                    continue
                V, F = meshes[oid]
                op = np.stack([np.asarray(x)
                               for x in r["info.object_pose.position"]])
                oq = np.stack([np.asarray(x)
                               for x in r["info.object_pose.quaternion"]])
                gp = np.stack([np.asarray(x)
                               for x in r["gel_pose_cell_frame.position"]])
                gq = np.stack([np.asarray(x)
                               for x in r["gel_pose_cell_frame.quaternion"]])
                order = rng.permutation(np.arange(1, len(gp)))
                taken = 0
                for ti in order:
                    ti = int(ti)
                    hm = gt_height_map(V, F, op[ti - 1], oq[ti - 1],
                                       gp[ti], gq[ti])
                    if (hm < 0).mean() < min_contact:
                        continue          # off-object: flat platform contact
                    img = np.asarray(Image.open(io.BytesIO(
                        r["sensor_image"][ti]["bytes"])).convert("RGB")
                    ).astype(np.float32)
                    yield dict(shard=sh.name, row=int(ri), touch=ti,
                               object_id=oid, label=int(r["label"]),
                               img=img, hm=hm, gel_z=float(gp[ti][2]))
                    taken += 1
                    if taken >= touches_per_row:
                        break
            del t


# ------------------------------------------------------------------ metrics
def align_shift(pred: np.ndarray, gt: np.ndarray, max_shift: int = 25):
    """3D Cal's 2D cross-correlation alignment (integer pixels)."""
    from numpy.fft import fft2, ifft2
    a = pred - pred.mean()
    b = gt - gt.mean()
    cc = np.real(ifft2(fft2(b) * np.conj(fft2(a))))
    h, w = cc.shape
    win = np.zeros_like(cc, bool)
    win[:max_shift + 1, :max_shift + 1] = True
    win[-max_shift:, :max_shift + 1] = True
    win[:max_shift + 1, -max_shift:] = True
    win[-max_shift:, -max_shift:] = True
    cc = np.where(win, cc, -np.inf)
    dy, dx = np.unravel_index(np.argmax(cc), cc.shape)
    dy = dy - h if dy > h // 2 else dy
    dx = dx - w if dx > w // 2 else dx
    return int(dy), int(dx)


def shift_img(a: np.ndarray, dy: int, dx: int) -> np.ndarray:
    out = np.zeros_like(a)
    ys, ye = max(0, dy), min(a.shape[0], a.shape[0] + dy)
    xs, xe = max(0, dx), min(a.shape[1], a.shape[1] + dx)
    out[ys:ye, xs:xe] = a[ys - dy:ye - dy, xs - dx:xe - dx]
    return out


def err_split(pred: np.ndarray, gt: np.ndarray, thr: float = 0.0) -> dict:
    """3D Cal's protocol: overall MAE plus the Type-1/Type-2 split [um]."""
    e = np.abs(pred - gt) * 1000.0
    m2 = gt > thr
    m1 = ~m2
    return dict(mae=float(e.mean()),
                t1=float(e[m1].mean()) if m1.any() else float("nan"),
                t2=float(e[m2].mean()) if m2.any() else float("nan"),
                rmse=float(np.sqrt((e ** 2).mean())))


def sample_metrics(depth, gt, valid, observed, coverage, tag) -> dict:
    """All the numbers we report for one (prediction, GT) pair."""
    out = {}
    for k, v in err_split(depth, gt).items():
        out[f"{tag}_{k}"] = v
    # per-sample scale fit (3D Cal fit the indentation-depth scale too).
    # num/den are kept so `report` can also fit ONE scale for the whole set.
    num = float((gt * depth).sum())
    den = float((depth * depth).sum())
    s = num / max(den, 1e-12)
    out[f"{tag}_scale"] = s
    out[f"{tag}_num"], out[f"{tag}_den"] = num, den
    for k, v in err_split(s * depth, gt).items():
        out[f"{tag}_fit_{k}"] = v
    dy, dx = align_shift(depth, gt)
    sd = shift_img(depth, dy, dx)
    s2 = float((gt * sd).sum() / max((sd * sd).sum(), 1e-12))
    for k, v in err_split(s2 * sd, gt).items():
        out[f"{tag}_fitshift_{k}"] = v
    out[f"{tag}_dy"], out[f"{tag}_dx"] = dy, dx
    out[f"{tag}_corr"] = float(np.corrcoef(depth.ravel(), gt.ravel())[0, 1])
    out[f"{tag}_peak"] = float(depth.max())
    out[f"{tag}_cov"] = float(coverage)
    # defect (c): mask against the GT contact region
    gtm = gt > 0
    for name, m in (("valid", valid), ("d05", depth > 0.05)):
        inter = float((m & gtm).sum())
        union = float((m | gtm).sum())
        out[f"{tag}_{name}_iou"] = inter / max(union, 1.0)
        out[f"{tag}_{name}_over"] = float((m & ~gtm).sum()) / max(gtm.sum(), 1)
        out[f"{tag}_{name}_recall"] = inter / max(gtm.sum(), 1)
    # defect (a): centre/rim ratio, ours vs the same statistic on the GT
    out[f"{tag}_flat_top"] = float(flat_top_sag(depth))
    # defect (b)
    out[f"{tag}_unobserved"] = float(1.0 - coverage)
    _ = observed
    return out


# ------------------------------------------------------------------ verify
def verify(n: int = 12) -> dict:
    """Re-render our GT height map with Taxim and diff against the shipped
    image; plus the object-pose index test and the reference-frame check."""
    import pyarrow.parquet as pq
    meshes = load_meshes()
    shard = sorted(SIM_DIR.glob("printed_train-*.parquet"))[0]
    t = pq.ParquetFile(shard).read_row_group(0).to_pandas()
    res = {"repro": [], "pose_offset": {}}
    for off in (-1, 0, 1):
        peaks = []
        for ri in range(8):
            r = t.iloc[ri]
            V, F = meshes[int(r["object_id"])]
            op = np.stack([np.asarray(x)
                           for x in r["info.object_pose.position"]])
            oq = np.stack([np.asarray(x)
                           for x in r["info.object_pose.quaternion"]])
            gp = np.stack([np.asarray(x)
                           for x in r["gel_pose_cell_frame.position"]])
            gq = np.stack([np.asarray(x)
                           for x in r["gel_pose_cell_frame.quaternion"]])
            for ti in range(2, 26):
                hm = gt_height_map(V, F, op[ti + off], oq[ti + off],
                                   gp[ti], gq[ti], SIM_W, SIM_H)
                if (hm < 0).mean() < 0.05:
                    continue
                peaks.append(-float(hm.min()))
        a = np.array(peaks)
        res["pose_offset"][str(off)] = dict(
            n=len(a), mean=float(a.mean()), std=float(a.std()))
    # image reproduction with the winning offset
    bg320 = background((SIM_H, SIM_W))
    ok = 0
    for ri in range(6):
        r = t.iloc[ri]
        V, F = meshes[int(r["object_id"])]
        op = np.stack([np.asarray(x) for x in r["info.object_pose.position"]])
        oq = np.stack([np.asarray(x)
                       for x in r["info.object_pose.quaternion"]])
        gp = np.stack([np.asarray(x)
                       for x in r["gel_pose_cell_frame.position"]])
        gq = np.stack([np.asarray(x)
                       for x in r["gel_pose_cell_frame.quaternion"]])
        for ti in range(1, 32):
            hm = gt_height_map(V, F, op[ti - 1], oq[ti - 1], gp[ti], gq[ti],
                               SIM_W, SIM_H)
            if (hm < 0).mean() < 0.05:
                continue
            img = np.asarray(Image.open(io.BytesIO(
                r["sensor_image"][ti]["bytes"])).convert("RGB")
            ).astype(np.float32)
            sim = taxim_render(hm)
            res["repro"].append(dict(
                row=ri, touch=ti,
                mae=float(np.abs(sim - img).mean()),
                corr=float(np.corrcoef(sim.ravel(), img.ravel())[0, 1]),
                bg_mae=float(np.abs(bg320 - img).mean())))
            ok += 1
            break
        if ok >= n:
            break
    # reference-frame check: taxim background vs median of shipped frames
    imgs = []
    for ri in range(min(60, len(t))):
        r = t.iloc[ri]
        for ti in RNG.permutation(32)[:4]:
            imgs.append(np.asarray(Image.open(io.BytesIO(
                r["sensor_image"][int(ti)]["bytes"])).convert("RGB")
            ).astype(np.float32))
    med = np.median(np.stack(imgs), 0)
    res["ref_check"] = dict(
        n_frames=len(imgs),
        median_vs_taxim_bg_mae=float(np.abs(med - bg320).mean()),
        median_vs_taxim_bg_p99=float(np.percentile(np.abs(med - bg320), 99)))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "verify.json").write_text(json.dumps(res, indent=2))
    a = np.array([x["mae"] for x in res["repro"]])
    print(json.dumps(res["pose_offset"], indent=2))
    print(f"taxim re-render vs shipped image: MAE {a.mean():.2f}/255 "
          f"(n={len(a)}), corr "
          f"{np.mean([x['corr'] for x in res['repro']]):.4f}")
    print(f"reference frame: median-of-{len(imgs)} vs taxim background "
          f"MAE {res['ref_check']['median_vs_taxim_bg_mae']:.2f}/255")
    return res


# ------------------------------------------------------------------ sim LUT
def sphere_presses(n_per_r=45, radii=(2.5, 4.0, 6.0), rng=None):
    """Analytic sphere caps pressed into the sim's gel, Taxim-rendered.

    Same protocol as lut_calibration on the real sensor: known R, known press
    depth, gel assumed to conform to the cap inside the contact circle.
    """
    rng = rng or np.random.default_rng(7)
    pix = SENSOR_MM[0] / SIM_W                    # mm per sim pixel
    yy, xx = np.mgrid[0:SIM_H, 0:SIM_W].astype(np.float32)
    for R in radii:
        for _ in range(n_per_r):
            d = float(rng.uniform(0.25, min(1.6, 0.85 * R)))
            a = np.sqrt(max(d * (2 * R - d), 1e-9))       # contact radius mm
            m = 0.75 * a / pix
            cx = float(rng.uniform(m, SIM_W - m))
            cy = float(rng.uniform(m, SIM_H - m))
            r_mm = np.hypot(xx - cx, yy - cy) * pix
            cap = d - (R - np.sqrt(np.clip(R ** 2 - r_mm ** 2, 1e-9, None)))
            cap = np.where(r_mm < a, np.maximum(cap, 0.0), 0.0)
            hm = (-cap).astype(np.float32)
            yield dict(R=R, d=d, a=a, cx=cx, cy=cy, hm=hm,
                       img=taxim_render(hm))


def fit_sim_lut(n_per_r=45, holdout=12) -> dict:
    """Fit a LUT on Taxim sphere renders, exactly like lut_calibration does."""
    ssum = np.zeros((BINS, BINS, BINS, 2), np.float64)
    cnt = np.zeros((BINS, BINS, BINS), np.int64)
    bg = background()
    yy, xx = np.mgrid[0:H_P, 0:W_P].astype(np.float32)
    test = []
    presses = list(sphere_presses(n_per_r))
    rng = np.random.default_rng(11)
    ho = set(rng.permutation(len(presses))[:holdout].tolist())
    for i, p in enumerate(presses):
        img = up(p["img"])
        if i in ho:
            test.append(dict(img=img, hm=p["hm"], R=p["R"], d=p["d"]))
            continue
        dI = img - bg
        # centre in the resampled frame
        cx = (p["cx"] + 0.5) * W_P / SIM_W - 0.5
        cy = (p["cy"] + 0.5) * H_P / SIM_H - 0.5
        rx, ry = xx - cx, yy - cy
        r_px = np.hypot(rx, ry)
        r_mm = r_px * MM_PER_PIXEL
        R = p["R"]
        inside = (r_mm < 0.97 * p["a"]) & (r_mm < 0.985 * R)
        denom = np.sqrt(np.clip(R ** 2 - r_mm ** 2, 1e-6, None))
        slope = -(r_mm / denom) * MM_PER_PIXEL
        gx = np.where(r_px > 1e-6, slope * rx / np.maximum(r_px, 1e-6), 0.0)
        gy = np.where(r_px > 1e-6, slope * ry / np.maximum(r_px, 1e-6), 0.0)
        q = np.clip((dI[inside] + DI_RANGE) / (2 * DI_RANGE) * (BINS - 1),
                    0, BINS - 1).astype(np.int32)
        idx = (q[:, 0], q[:, 1], q[:, 2])
        np.add.at(ssum, idx + (0,), gx[inside])
        np.add.at(ssum, idx + (1,), gy[inside])
        np.add.at(cnt, idx, 1)
    have = cnt > 0
    lut = np.zeros((BINS, BINS, BINS, 2), np.float32)
    lut[have] = (ssum[have] / cnt[have, None]).astype(np.float32)
    lut = fill_lut_holes(lut, cnt)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "sim_lut.npz", lut=lut, count=cnt)
    print(f"sim LUT: {have.mean()*100:.2f}% of bins observed, "
          f"{len(presses)-holdout} presses")
    # in-domain control: held-out sphere renders through the same solver
    ctrl = []
    for t in test:
        st = stages_full(t["img"], bg, lut, cnt)
        gt = np.maximum(-up(t["hm"]), 0.0)
        gel = np.maximum(up(taxim_deformation(t["hm"])), 0.0)
        ctrl.append(dict(R=t["R"], d=t["d"], peak=float(st["depth"].max()),
                         gt_peak=float(gt.max()), gel_peak=float(gel.max()),
                         **{f"geo_{k}": v
                            for k, v in err_split(st["depth"], gt).items()},
                         **{f"gel_{k}": v
                            for k, v in err_split(st["depth"], gel).items()}))
    (OUT / "simlut_control.json").write_text(json.dumps(ctrl, indent=2))
    m = {k: float(np.mean([c[k] for c in ctrl]))
         for k in ctrl[0] if k not in ("R",)}
    print("held-out sphere control (sim domain):",
          json.dumps({k: round(v, 1) for k, v in m.items()}, indent=1))
    return dict(lut=lut, count=cnt, control=ctrl)


def load_sim_lut():
    z = np.load(OUT / "sim_lut.npz")
    return z["lut"], z["count"]


# ------------------------------------------------------------------ stage 1
def stage1(n=N_TOUCHES, save_examples=10):
    from force_recovery.lut_calibration import CAL_OUT as _C
    cal = np.load(_C / "glowtact_lut.npz")
    lut_a, cnt_a = cal["lut"], cal["count"]
    lut_b, cnt_b = load_sim_lut()
    meshes = load_meshes()
    bg = background()
    bg_crop = crop(background((SIM_H, SIM_W)))
    shards = sorted(SIM_DIR.glob("printed_train-*.parquet"))
    rows, ex = [], []
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "examples").mkdir(exist_ok=True)
    # every depth map is cached so that report() can re-derive any metric
    # (global instead of per-sample scale, other thresholds, ...) without
    # re-running the ray casting.
    tens = np.lib.format.open_memmap(
        OUT / "tensors.npy", mode="w+", dtype=np.float16,
        shape=(n, 4, H_P, W_P))
    for k, s in enumerate(iter_touches(shards, meshes=meshes)):
        img = up(s["img"])
        gt = np.maximum(-up(s["hm"]), 0.0)                 # object geometry
        gel = np.maximum(up(taxim_deformation(s["hm"])), 0.0)  # gel surface
        rec = dict(shard=s["shard"], row=s["row"], touch=s["touch"],
                   object_id=s["object_id"], label=s["label"],
                   gt_peak=float(gt.max()), gel_peak=float(gel.max()),
                   gt_area=float((gt > 0).mean()),
                   gt_flat_top=float(flat_top_sag(gt)),
                   gel_flat_top=float(flat_top_sag(gel)),
                   zero_mae=float(np.abs(gt).mean() * 1000),
                   zero_t2=float(gt[gt > 0].mean() * 1000))
        # the conformance ceiling: the gel surface Taxim actually rendered is
        # not the object geometry, and NO gel-surface method can beat this.
        rec.update({f"gap_{kk}": vv
                    for kk, vv in err_split(gel, gt).items()})
        # track A: the shipped pipeline, GlowTact LUT
        st_a = stages(img, bg)
        rec.update(sample_metrics(st_a["depth"], gt, st_a["valid"], None,
                                  st_a["lut_coverage"], "A"))
        rec.update({f"Agel_{kk}": vv for kk, vv
                    in err_split(st_a["depth"], gel, thr=0.01).items()})
        # track B: same solver, LUT re-fitted on Taxim spheres
        st_b = stages_full(img, bg, lut_b, cnt_b)
        cov_b = float(st_b["observed"][st_b["valid"]].mean()) \
            if st_b["valid"].any() else 0.0
        rec.update(sample_metrics(st_b["depth"], gt, st_b["valid"], None,
                                  cov_b, "B"))
        rec.update({f"Bgel_{kk}": vv for kk, vv
                    in err_split(st_b["depth"], gel, thr=0.01).items()})
        # deployment crop (the pipeline's own 5/7 view), track B
        st_c = stages_full(crop(s["img"]), bg_crop, lut_b, cnt_b)
        gt_c = np.maximum(crop(-s["hm"]), 0.0)
        rec.update({f"Bcrop_{kk}": vv
                    for kk, vv in err_split(st_c["depth"], gt_c).items()})
        # gradient-direction diagnostic against the true gel surface
        for tag, st in (("A", st_a), ("B", st_b)):
            gxx = st.get("gx", None)
            if gxx is None:
                continue
            tgx = cv2.Sobel(gel, cv2.CV_32F, 1, 0, ksize=3) / 8.0
            tgy = cv2.Sobel(gel, cv2.CV_32F, 0, 1, ksize=3) / 8.0
            sel = (gel > 0.05) & (np.hypot(tgx, tgy) > 0.005) \
                & (np.hypot(st["gx"], st["gy"]) > 1e-4)
            if sel.sum() > 200:
                dot = (st["gx"][sel] * tgx[sel] + st["gy"][sel] * tgy[sel]) / (
                    np.hypot(st["gx"][sel], st["gy"][sel])
                    * np.hypot(tgx[sel], tgy[sel]))
                rec[f"{tag}_grad_angle"] = float(np.degrees(
                    np.arccos(np.clip(dot, -1, 1))).mean())
                rec[f"{tag}_grad_ratio"] = float(np.median(
                    np.hypot(st["gx"][sel], st["gy"][sel])
                    / np.hypot(tgx[sel], tgy[sel])))
        tens[len(rows)] = np.stack([gt, gel, st_a["depth"], st_b["depth"]]
                                   ).astype(np.float16)
        rows.append(rec)
        if len(ex) < save_examples and rec["gt_area"] > 0.15:
            np.savez_compressed(
                OUT / "examples" / f"ex_{len(ex):02d}.npz",
                img=s["img"].astype(np.uint8), gt=gt.astype(np.float32),
                gel=gel.astype(np.float32),
                depth_a=st_a["depth"].astype(np.float32),
                depth_b=st_b["depth"].astype(np.float32),
                valid_b=st_b["valid"], label=s["label"],
                object_id=s["object_id"])
            ex.append(len(ex))
        if (k + 1) % 25 == 0:
            print(f"{k+1}/{n} touches", flush=True)
        if len(rows) >= n:
            break
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "stage1.parquet")
    print(f"saved {len(df)} touches -> {OUT/'stage1.parquet'}")
    report()


def _agg(df, keys):
    out = {}
    for k in keys:
        if k in df:
            out[k] = float(np.nanmean(df[k]))
    return out


def _mean(df, k, scale=1.0):
    import pandas as pd
    if k not in df:
        return float("nan")
    return float(np.nanmean(pd.to_numeric(df[k], errors="coerce"))) * scale


def global_scale_metrics(df):
    """One scale for the WHOLE set (much stricter than a per-sample fit) plus
    the halo-pedestal-removed variant, from the cached depth maps."""
    from force_recovery.o3d_view import remove_halo_pedestal
    tens = np.load(OUT / "tensors.npy", mmap_mode="r")
    n = min(len(df), tens.shape[0])
    out = {}
    for tag, ch in (("A", 2), ("B", 3)):
        s = float(df[f"{tag}_num"].sum() / df[f"{tag}_den"].sum())
        acc = {k: [] for k in ("mae", "t1", "t2")}
        acc_p = {k: [] for k in ("mae", "t1", "t2")}
        for i in range(n):
            gt = tens[i, 0].astype(np.float32)
            d = tens[i, ch].astype(np.float32)
            for k, v in err_split(s * d, gt).items():
                if k in acc:
                    acc[k].append(v)
            dp = remove_halo_pedestal(d)
            sp = float((gt * dp).sum() / max((dp * dp).sum(), 1e-12))
            for k, v in err_split(sp * dp, gt).items():
                if k in acc_p:
                    acc_p[k].append(v)
        out[f"{tag}_globalscale"] = s
        for k in acc:
            out[f"{tag}_gfit_{k}"] = float(np.nanmean(acc[k]))
            out[f"{tag}_pedestal_{k}"] = float(np.nanmean(acc_p[k]))
    return out


def report():
    import pandas as pd
    df = pd.read_parquet(OUT / "stage1.parquet")
    g = global_scale_metrics(df)
    print(f"\n=== Stage 1: SimTactileMNIST, n={len(df)} touches, "
          f"{df.object_id.nunique()} objects, {df.label.nunique()} digits ===")
    print(f"GT: peak penetration {df.gt_peak.mean():.3f} mm (every sim touch "
          f"is the same 2.25 mm press), contact area "
          f"{df.gt_area.mean()*100:.1f}% of the pad")
    print(f"context: predicting ZERO everywhere gives MAE "
          f"{df.zero_mae.mean():.0f} um / Type-2 {df.zero_t2.mean():.0f} um; "
          f"the Taxim gel surface itself differs from the object geometry by "
          f"MAE {df.gap_mae.mean():.0f} um / Type-2 {df.gap_t2.mean():.0f} um "
          f"(conformance ceiling for ANY gel-surface method)")
    print(f"\n{'variant':38s} {'MAE':>8s} {'Type-1':>8s} {'Type-2':>8s}")
    for tag, name in (("A", "A GlowTact LUT (shipped)"),
                      ("B", "B sim-refit LUT")):
        rowspec = [("raw (no fitting)", (df[f"{tag}_mae"].mean(),
                                         df[f"{tag}_t1"].mean(),
                                         df[f"{tag}_t2"].mean())),
                   (f"+1 global scale s={g[tag+'_globalscale']:.2f}",
                    (g[f"{tag}_gfit_mae"], g[f"{tag}_gfit_t1"],
                     g[f"{tag}_gfit_t2"])),
                   ("+per-sample scale", (df[f"{tag}_fit_mae"].mean(),
                                          df[f"{tag}_fit_t1"].mean(),
                                          df[f"{tag}_fit_t2"].mean())),
                   ("+per-sample scale +2D shift",
                    (df[f"{tag}_fitshift_mae"].mean(),
                     df[f"{tag}_fitshift_t1"].mean(),
                     df[f"{tag}_fitshift_t2"].mean())),
                   ("+pedestal removed +scale",
                    (g[f"{tag}_pedestal_mae"], g[f"{tag}_pedestal_t1"],
                     g[f"{tag}_pedestal_t2"]))]
        for lbl, (a, b, c) in rowspec:
            print(f"{name+' '+lbl:38s} {a:8.1f} {b:8.1f} {c:8.1f}")
    print(f"{'B vs Taxim gel surface (raw)':38s} "
          f"{df['Bgel_mae'].mean():8.1f} {df['Bgel_t1'].mean():8.1f} "
          f"{df['Bgel_t2'].mean():8.1f}")
    print(f"{'B deployment crop 5/7 (raw)':38s} "
          f"{df['Bcrop_mae'].mean():8.1f} {df['Bcrop_t1'].mean():8.1f} "
          f"{df['Bcrop_t2'].mean():8.1f}")
    print("\ndefects vs exact GT")
    print(f"  (a) centre/rim ratio  ours A {_mean(df,'A_flat_top'):.3f} "
          f"B {_mean(df,'B_flat_top'):.3f}  GT(object) "
          f"{_mean(df,'gt_flat_top'):.3f} GT(gel) {_mean(df,'gel_flat_top'):.3f}"
          f"  -> over-doming factor B/GT "
          f"{_mean(df,'B_flat_top')/_mean(df,'gt_flat_top'):.3f}")
    print(f"  (b) unobserved LUT bins A {_mean(df,'A_unobserved')*100:.1f}% "
          f"B {_mean(df,'B_unobserved')*100:.1f}%;  corr(unobserved, Type-2) "
          f"A {df.A_unobserved.corr(df.A_t2):.3f} "
          f"B {df.B_unobserved.corr(df.B_t2):.3f}")
    print(f"  (c) valid-mask IoU vs GT contact  A {_mean(df,'A_valid_iou'):.3f}"
          f" B {_mean(df,'B_valid_iou'):.3f}; over-segmentation "
          f"A {_mean(df,'A_valid_over'):.3f} B {_mean(df,'B_valid_over'):.3f}; "
          f"recall {_mean(df,'B_valid_recall'):.3f}; "
          f"depth>0.05 IoU B {_mean(df,'B_d05_iou'):.3f}")
    if "B_grad_angle" in df:
        print(f"  gradient angle vs true gel surface B "
              f"{_mean(df,'B_grad_angle'):.1f} deg, amplitude ratio "
              f"{df.B_grad_ratio.median():.2f}")
    print(f"  peak depth: ours A {_mean(df,'A_peak'):.3f} "
          f"B {_mean(df,'B_peak'):.3f} vs GT object {df.gt_peak.mean():.3f} / "
          f"gel {_mean(df,'gel_peak'):.3f} mm")
    print(f"  whole-image corr with GT: A {_mean(df,'A_corr'):.3f} "
          f"B {_mean(df,'B_corr'):.3f}; frac of touches with corr<0: "
          f"A {(df.A_corr<0).mean():.2f} B {(df.B_corr<0).mean():.2f}")
    print(f"  per-sample scale (median) A {df.A_scale.median():.3f} "
          f"B {df.B_scale.median():.3f}; |2D shift| median B "
          f"{np.median(np.hypot(df.B_dy, df.B_dx)):.1f} px "
          f"({np.median(np.hypot(df.B_dy, df.B_dx))*MM_PER_PIXEL:.2f} mm)")
    q = pd.qcut(df.gt_area, 3, labels=["small", "mid", "large"])
    print("\n  by GT contact area (fraction of pad):")
    for lvl in ["small", "mid", "large"]:
        s = df[q == lvl]
        print(f"   {lvl:6s} area {s.gt_area.mean()*100:4.1f}%  "
              f"B raw MAE {s.B_mae.mean():6.1f} T2 {s.B_t2.mean():7.1f}  "
              f"gap(ceiling) MAE {s.gap_mae.mean():6.1f} T2 {s.gap_t2.mean():7.1f}")
    summary = {"n": int(len(df)), "gt_peak_mm": float(df.gt_peak.mean()),
               "zero_baseline_mae_um": float(df.zero_mae.mean()),
               "zero_baseline_t2_um": float(df.zero_t2.mean()), **g}
    for tag in ("A", "B"):
        for suf in ("", "fit_", "fitshift_"):
            for m in ("mae", "t1", "t2"):
                summary[f"{tag}_{suf}{m}"] = float(df[f"{tag}_{suf}{m}"].mean())
    summary.update(_agg(df, [
        "gap_mae", "gap_t1", "gap_t2", "A_flat_top", "B_flat_top",
        "gt_flat_top", "gel_flat_top", "A_unobserved", "B_unobserved",
        "A_valid_iou", "B_valid_iou", "A_valid_over", "B_valid_over",
        "B_valid_recall", "B_d05_iou", "B_grad_angle", "B_grad_ratio",
        "A_corr", "B_corr", "A_peak", "B_peak", "gel_peak", "Bgel_mae",
        "Bgel_t2", "Bcrop_mae", "Bcrop_t2"]))
    (OUT / "stage1_summary.json").write_text(json.dumps(summary, indent=2))
    return df


# ------------------------------------------------------------------ diagnose
def diagnose(n_img=80):
    """Two things the headline table cannot answer.

    1. does Taxim's photometry drive OUR table at all? -> angle between the
       LUT gradient and the true gel-surface gradient, for both tables, with
       the GlowTact sphere presses (in-domain, real sensor) as the control.
    2. is the error driven by contacts that overflow the field of view?
       Poisson integration needs the contact to be enclosed; a digit pressed
       2.25 mm deep usually is not.
    """
    import pandas as pd
    cal = np.load(CAL_OUT / "glowtact_lut.npz")
    lut_a, cnt_a = cal["lut"], cal["count"]
    lut_b, cnt_b = load_sim_lut()
    meshes = load_meshes()
    bg = background()
    rows = []
    for k, s in enumerate(iter_touches(
            sorted(SIM_DIR.glob("printed_train-*.parquet")), meshes=meshes,
            rng=np.random.default_rng(3))):
        img = up(s["img"])
        gel = np.maximum(up(taxim_deformation(s["hm"])), 0.0)
        tgx = cv2.Sobel(gel, cv2.CV_32F, 1, 0, ksize=3) / 8.0
        tgy = cv2.Sobel(gel, cv2.CV_32F, 0, 1, ksize=3) / 8.0
        rec = {}
        for tag, lut, cnt in (("A", lut_a, cnt_a), ("B", lut_b, cnt_b)):
            st = stages_full(img, bg, lut, cnt)
            sel = (gel > 0.05) & (np.hypot(tgx, tgy) > 0.005) \
                & (np.hypot(st["gx"], st["gy"]) > 1e-4)
            if sel.sum() < 200:
                continue
            gm = np.hypot(st["gx"][sel], st["gy"][sel])
            tm = np.hypot(tgx[sel], tgy[sel])
            dot = (st["gx"][sel] * tgx[sel] + st["gy"][sel] * tgy[sel]) \
                / (gm * tm)
            rec[f"{tag}_angle"] = float(np.degrees(
                np.arccos(np.clip(dot, -1, 1))).mean())
            rec[f"{tag}_within30"] = float(np.mean(
                np.degrees(np.arccos(np.clip(dot, -1, 1))) < 30))
            rec[f"{tag}_ratio"] = float(np.median(gm / tm))
        if rec:
            rows.append(rec)
        if len(rows) >= n_img:
            break
    dg = pd.DataFrame(rows)
    print(f"gradient direction vs the TRUE gel surface (n={len(dg)} touches)")
    for tag in ("A", "B"):
        print(f"  {tag}: mean angle {dg[tag+'_angle'].mean():5.1f} deg, "
              f"{dg[tag+'_within30'].mean()*100:4.1f}% of contact pixels "
              f"within 30 deg, |g|ratio {dg[tag+'_ratio'].median():.2f}")
    # in-domain control on the REAL sensor, same statistic
    ctrl = glowtact_grad_control()
    print(f"  control, GlowTact sphere presses (real Mini, its own LUT): "
          f"mean angle {ctrl['angle']:.1f} deg, "
          f"{ctrl['within30']*100:.1f}% within 30 deg (n={ctrl['n']})")
    # field of view
    df = pd.read_parquet(OUT / "stage1.parquet")
    tens = np.load(OUT / "tensors.npy", mmap_mode="r")
    border, shape_corr = [], []
    for i in range(min(len(df), tens.shape[0])):
        gt = tens[i, 0].astype(np.float32)
        d = tens[i, 3].astype(np.float32)
        m = gt > 0
        edge = m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any()
        border.append(bool(edge))
        if m.sum() > 500:
            shape_corr.append(float(np.corrcoef(d[m], gt[m])[0, 1]))
        else:
            shape_corr.append(np.nan)
    df["border"] = border
    df["shape_corr"] = shape_corr
    print(f"\nGT contact touches the image border in "
          f"{np.mean(border)*100:.0f}% of touches")
    for lbl, sub in (("enclosed", df[~df.border]), ("clipped", df[df.border])):
        if len(sub) == 0:
            continue
        print(f"  {lbl:9s} n={len(sub):3d} area {sub.gt_area.mean()*100:4.1f}% "
              f"| B raw MAE {sub.B_mae.mean():6.1f} T2 {sub.B_t2.mean():7.1f} "
              f"| whole-image corr {sub.B_corr.mean():5.2f} "
              f"| in-contact shape corr {np.nanmean(sub.shape_corr):5.2f} "
              f"| corr<0 {(sub.B_corr<0).mean():.2f}")
    out = dict(grad=dg.mean(numeric_only=True).to_dict(), control=ctrl,
               border_frac=float(np.mean(border)),
               enclosed_mae=float(df[~df.border].B_mae.mean())
               if (~df.border).any() else None,
               clipped_mae=float(df[df.border].B_mae.mean()),
               enclosed_corr=float(df[~df.border].B_corr.mean())
               if (~df.border).any() else None,
               clipped_corr=float(df[df.border].B_corr.mean()),
               shape_corr=float(np.nanmean(df.shape_corr)))
    (OUT / "diagnose.json").write_text(json.dumps(out, indent=2))
    return out


def controls():
    """Two synthetic controls that decide WHY the digits are hard.

    C1 border clipping: the same sphere cap, once enclosed by flat gel and once
       cut by the image edge. fast_poisson imposes a zero boundary, so a
       contact that leaves the field of view cannot integrate. Every one of the
       420 MNIST touches is clipped (a 2.25 mm press into an 8 mm digit whose
       footprint is 3x the pad), so this has to be measured, not assumed.
    C2 plateau: a filleted flat-top disc, fully enclosed. Its true centre/rim
       ratio is 1.000 by construction, so it isolates the over-doming defect
       from both clipping and from the digits' own curvature.
    """
    lut, cnt = load_sim_lut()
    bg = background()
    pix = SENSOR_MM[0] / SIM_W
    yy, xx = np.mgrid[0:SIM_H, 0:SIM_W].astype(np.float32)
    rows = []
    for R in (3.0, 4.5, 6.0):
        for d in (0.6, 1.2):
            a = np.sqrt(d * (2 * R - d))
            for place, (cx, cy) in (("enclosed", (SIM_W / 2, SIM_H / 2)),
                                    ("clipped", (0.15 * a / pix, SIM_H / 2))):
                r_mm = np.hypot(xx - cx, yy - cy) * pix
                cap = d - (R - np.sqrt(np.clip(R ** 2 - r_mm ** 2, 1e-9, None)))
                cap = np.where(r_mm < a, np.maximum(cap, 0.0), 0.0)
                hm = (-cap).astype(np.float32)
                img = up(taxim_render(hm))
                st = stages_full(img, bg, lut, cnt)
                gt = np.maximum(up(cap), 0.0)
                e = err_split(st["depth"], gt)
                rows.append(dict(control="C1_sphere", place=place, R=R, d=d,
                                 a_mm=float(a), peak=float(st["depth"].max()),
                                 gt_peak=float(gt.max()), **e))
    for r0, h0 in ((2.0, 0.6), (3.0, 0.6), (3.0, 1.2)):
        f = 0.4                                   # fillet radius mm
        r_mm = np.hypot(xx - SIM_W / 2, yy - SIM_H / 2) * pix
        cap = np.where(r_mm <= r0, h0, 0.0)
        ring = (r_mm > r0) & (r_mm < r0 + f)
        cap[ring] = h0 - (f - np.sqrt(np.clip(
            f ** 2 - (r_mm[ring] - r0) ** 2, 0, None))) * h0 / f
        hm = (-cap).astype(np.float32)
        img = up(taxim_render(hm))
        st = stages_full(img, bg, lut, cnt)
        gt = np.maximum(up(cap), 0.0)
        rows.append(dict(control="C2_plateau", place="enclosed", R=r0, d=h0,
                         a_mm=float(r0), peak=float(st["depth"].max()),
                         gt_peak=float(gt.max()),
                         flat_top=float(flat_top_sag(st["depth"])),
                         gt_flat_top=float(flat_top_sag(gt)),
                         **err_split(st["depth"], gt)))
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_json(OUT / "controls.json", orient="records", indent=1)
    print(df.to_string(index=False))
    c1 = df[df.control == "C1_sphere"]
    print("\nC1 border clipping (sphere caps, sim-refit LUT):")
    for place in ("enclosed", "clipped"):
        s = c1[c1.place == place]
        print(f"  {place:9s} MAE {s.mae.mean():7.1f} um  Type-1 "
              f"{s.t1.mean():6.1f}  Type-2 {s.t2.mean():7.1f}  "
              f"peak {s.peak.mean():.2f} / {s.gt_peak.mean():.2f} mm")
    c2 = df[df.control == "C2_plateau"]
    print(f"C2 plateau (true centre/rim ratio = "
          f"{c2.gt_flat_top.mean():.3f}): ours {c2.flat_top.mean():.3f} "
          f"-> over-doming +{(c2.flat_top.mean()/c2.gt_flat_top.mean()-1)*100:.0f}%"
          f", MAE {c2.mae.mean():.1f} um, Type-2 {c2.t2.mean():.1f} um")
    return df


def depth_sweep(n_touch=40, depths=(0.3, 0.6, 1.0, 1.5, 2.25)):
    """The decisive experiment: the SAME digit geometry, re-pressed shallower.

    Every shipped touch is a 2.25 mm press, and every one of them is clipped by
    the sensor edge, so the dataset cannot separate "digits are hard" from
    "contacts that leave the field of view are hard". Taxim reproduces the
    shipped images to 1.76/255 (see `verify`), so re-rendering the same mesh at
    a smaller press depth is a legitimate counterfactual: identical geometry
    and photometry, only the penetration changes.
    """
    import pandas as pd
    lut, cnt = load_sim_lut()
    cal = np.load(CAL_OUT / "glowtact_lut.npz")
    bg = background()
    bg320 = background((SIM_H, SIM_W))
    meshes = load_meshes()
    rows = []
    touches = []
    for s in iter_touches(sorted(SIM_DIR.glob("printed_train-*.parquet")),
                          meshes=meshes, rng=np.random.default_rng(5)):
        touches.append(s["hm"])
        if len(touches) >= n_touch:
            break
    for hm0 in touches:
        for d in depths:
            hm = hm0 + (PRESS_MM - d)          # raise the sensor
            if (hm < 0).mean() < 0.002:
                continue
            img = up(taxim_render(hm))
            gt = np.maximum(up(-hm), 0.0)
            m = gt > 0
            edge = bool(m[0].any() or m[-1].any() or m[:, 0].any()
                        or m[:, -1].any())
            for tag, (L, C) in (("A", (cal["lut"], cal["count"])),
                                ("B", (lut, cnt))):
                st = stages_full(img, bg, L, C)
                rows.append(dict(press=d, tag=tag, clipped=edge,
                                 area=float(m.mean()),
                                 peak=float(st["depth"].max()),
                                 gt_peak=float(gt.max()),
                                 xcorr=float(np.corrcoef(st["depth"].ravel(),
                                                         gt.ravel())[0, 1]),
                                 iou=float((st["valid"] & m).sum()
                                           / max((st["valid"] | m).sum(), 1)),
                                 **err_split(st["depth"], gt)))
    _ = bg320
    df = pd.DataFrame(rows)
    df.to_json(OUT / "depth_sweep.json", orient="records")
    print(f"{'press':>6} {'tag':>3} {'n':>4} {'clipped':>8} {'area%':>6} "
          f"{'MAE':>7} {'T1':>6} {'T2':>8} {'peak/GT':>8} {'corr':>6} "
          f"{'IoU':>5}")
    for d in depths:
        for tag in ("A", "B"):
            s = df[(df.press == d) & (df.tag == tag)]
            if not len(s):
                continue
            print(f"{d:6.2f} {tag:>3} {len(s):4d} {s.clipped.mean()*100:7.0f}% "
                  f"{s.area.mean()*100:5.1f} {s.mae.mean():7.1f} "
                  f"{s.t1.mean():6.1f} {s.t2.mean():8.1f} "
                  f"{(s.peak/s.gt_peak).mean():8.2f} {s.xcorr.mean():6.2f} "
                  f"{s.iou.mean():5.2f}")
    sub = df[df.tag == "A"]
    for lbl, q in (("enclosed", ~sub.clipped), ("clipped", sub.clipped)):
        t = sub[q]
        if len(t):
            print(f"  pooled {lbl:9s} n={len(t):3d} MAE {t.mae.mean():6.1f} "
                  f"T2 {t.t2.mean():7.1f} peak/GT {(t.peak/t.gt_peak).mean():.2f} "
                  f"corr {t.xcorr.mean():.2f}")
    return df


def glowtact_grad_control(n=10):
    """Same gradient-angle statistic on the real sensor the LUT came from."""
    from force_recovery.lut_calibration import (GLOWTACT, PAT, detect_circle)
    cal = np.load(CAL_OUT / "glowtact_lut.npz")
    lut, cnt = cal["lut"], cal["count"]
    R, z0 = float(cal["R_mm"]), float(cal["z0_mm"])
    ref = crop(np.asarray(Image.open(GLOWTACT / "round" / "initial.jpg")
                          .convert("RGB"))).astype(np.float32)
    angs, w30, k = [], [], 0
    for p in sorted((GLOWTACT / "round").glob("*.jpg")):
        m = PAT.search(p.name)
        if not m:
            continue
        d = -float(m["z"]) - z0
        if not (0.3 < d < 0.9 * R):
            continue
        img = crop(np.asarray(Image.open(p).convert("RGB"))).astype(np.float32)
        st = stages_full(img, ref, lut, cnt)
        det = detect_circle(img - ref)
        if det is None:
            continue
        cx, cy, a = det
        yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
        r_mm = np.hypot(xx - cx, yy - cy) * MM_PER_PIXEL
        h = np.where(r_mm < 0.97 * a * MM_PER_PIXEL,
                     d - (R - np.sqrt(np.clip(R ** 2 - r_mm ** 2, 1e-9, None))),
                     0.0)
        h = np.maximum(h, 0).astype(np.float32)
        tgx = cv2.Sobel(h, cv2.CV_32F, 1, 0, ksize=3) / 8.0
        tgy = cv2.Sobel(h, cv2.CV_32F, 0, 1, ksize=3) / 8.0
        sel = (h > 0.05) & (np.hypot(tgx, tgy) > 0.005) \
            & (np.hypot(st["gx"], st["gy"]) > 1e-4)
        if sel.sum() < 200:
            continue
        dot = (st["gx"][sel] * tgx[sel] + st["gy"][sel] * tgy[sel]) / (
            np.hypot(st["gx"][sel], st["gy"][sel])
            * np.hypot(tgx[sel], tgy[sel]))
        ang = np.degrees(np.arccos(np.clip(dot, -1, 1)))
        angs.append(float(ang.mean()))
        w30.append(float((ang < 30).mean()))
        k += 1
        if k >= n:
            break
    return dict(angle=float(np.mean(angs)), within30=float(np.mean(w30)),
                n=k)


# ------------------------------------------------------------------ figures
def figures(n=8):
    """Six columns, in the site's one convention — see `eval_panel`.

    The figure used to show five: sim image, GT depth, two LUT depths, error.
    Two things were missing and both mattered. There was no DIFFERENCE image,
    so a reader could not see what the reconstruction was reading; and there
    was no MESH, so a depth map that is the wrong SHAPE looked the same as one
    that is right — which is precisely the failure this validation exists to
    catch, and precisely what a heat map hides.

    The reference is Taxim's no-contact render, recomputed here rather than
    stored, because it is exact by construction (`background`).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from . import eval_panel as EP

    files = sorted((OUT / "examples").glob("ex_*.npz"))[:n]
    if not files:
        raise SystemExit(f"no examples under {OUT/'examples'}")
    bg = crop(background((SIM_H, SIM_W)))
    ncol = 6
    fig, axes = plt.subplots(len(files), ncol,
                             figsize=(3.0 * ncol, 2.5 * len(files)))
    axes = np.atleast_2d(axes)
    for i, f in enumerate(files):
        z = np.load(f)
        gt, da, db = z["gt"], z["depth_a"], z["depth_b"]
        vmax = max(float(gt.max()), 1e-3)
        img = z["img"]
        cells = [
            (img, None, "sim image (Taxim)", None),
            (EP.diff_rgb(img, bg), None,
             "difference  dI = frame − ref  (×3, colour)", None),
            (gt, "viridis", f"GT depth (mesh) max {gt.max():.2f} mm", vmax),
            (da, "viridis", f"ours, cnc_mini_26 LUT  max {da.max():.2f}",
             vmax),
            (db, "viridis", f"ours, sim-refit LUT max {db.max():.2f}", vmax),
            (EP.mesh(db) if EP.available()
             else np.zeros((240, 320, 3), np.uint8), None,
             "3D reconstruction (Open3D mesh)"
             if EP.available() else "mesh — no display", None),
        ]
        for j, (a, cmap, t, vm) in enumerate(cells):
            ax = axes[i, j]
            if vm is not None:
                ax.imshow(a, cmap=cmap, vmin=0, vmax=vm)
            else:
                ax.imshow(a, cmap=cmap)
            ax.set_title(t, fontsize=8)
            ax.axis("off")
    fig.suptitle("SimTactileMNIST per-pixel validation: exact mesh GT vs our "
                 "reconstruction (mm). Every touch is a 2.25 mm press whose "
                 "contact leaves the field of view.", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    p = OUT / "mnist_examples.png"
    fig.savefig(p, dpi=110)
    print("wrote", p)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    {"verify": verify, "simlut": fit_sim_lut, "stage1": stage1,
     "report": report, "figures": figures, "diagnose": diagnose,
     "controls": controls, "sweep": depth_sweep}[cmd]()
