"""Validate against FoTa's cnc_Mini — CNC-probed GelSight Mini with F/T labels.

The subset the FoTa paper describes as the 3-DoF gantry platform: textured
probes attached to a force/torque sensor on a desktop CNC pressed into a
GelSight Mini. Each frame's JSON carries a measured scalar ``force`` (N)
plus the probe position — real force ground truth, on a third-party rig,
independent of FEATS.

Protocol mirrors the FEATS validation: references are the lowest-force
frames (the closest thing to no-contact this subset offers — if the minimum
recorded force is materially above zero, the zero map absorbs that contact
and the estimate is biased low; the report states the reference floor),
volume is regressed against measured force, and correlation + a fitted
scale are reported per probe and pooled.
"""
from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image

from .depth_force import DepthForceEstimator
from .run_episode import OUT_ROOT

CNC_MINI = OUT_ROOT / "fota_cnc" / "cnc" / "cnc_Mini"

N_REF = 10
MAX_EVAL = 300


def _iter_samples(split: str, limit: int | None = None):
    """(meta, image) pairs — used only for small peeks; see build_index."""
    n = 0
    for tar in sorted((CNC_MINI / split).glob("*.tar")):
        with tarfile.open(tar) as tf:
            pending: dict[str, dict] = {}
            for m in tf:
                stem, _, ext = m.name.rpartition(".")
                if ext not in ("jpg", "json"):
                    continue
                entry = pending.setdefault(stem, {})
                entry[ext] = tf.extractfile(m).read()
                if len(entry) == 2:
                    meta = json.loads(entry["json"])
                    img = np.asarray(Image.open(
                        io.BytesIO(entry["jpg"])).convert("RGB"))
                    yield meta, img
                    del pending[stem]
                    n += 1
                    if limit and n >= limit:
                        return


def build_index(split: str) -> list[dict]:
    """Metadata for every frame without decoding a single 8-MP image.

    The frames are full-resolution 3280x2464 JPEGs (~24 MB decoded), so
    loading everything to pick 300 of them would need ~100 GB; instead the
    JSONs are read in one pass and images are fetched by member name later.
    """
    index = []
    for tar in sorted((CNC_MINI / split).glob("*.tar")):
        with tarfile.open(tar) as tf:
            for m in tf:
                if m.name.endswith(".json"):
                    meta = json.loads(tf.extractfile(m).read())
                    index.append({"tar": str(tar),
                                  "jpg": m.name[:-5] + ".jpg", **meta})
    return index


def load_images(picks: list[dict]) -> list[np.ndarray]:
    """Decode only the requested members, grouped per tar."""
    by_tar: dict[str, list[dict]] = {}
    for p in picks:
        by_tar.setdefault(p["tar"], []).append(p)
    out: dict[int, np.ndarray] = {}
    for tar, rows in by_tar.items():
        with tarfile.open(tar) as tf:
            for p in rows:
                img = np.asarray(Image.open(io.BytesIO(
                    tf.extractfile(p["jpg"]).read())).convert("RGB"))
                out[id(p)] = img
    return [out[id(p)] for p in picks]


def _group_by_tar(rows: list[dict]) -> dict[str, list[dict]]:
    by_tar: dict[str, list[dict]] = {}
    for p in rows:
        by_tar.setdefault(p["tar"], []).append(p)
    return by_tar


def _detect_geometry(img: np.ndarray) -> bool:
    """True if the image is already the cropped gel view (no LED border).

    On the full frame the extreme corners hold the dark outside-gel border,
    so corner luminance sits far below the centre; a cropped view is
    near-uniform.
    """
    h, w = img.shape[:2]
    corners = np.mean([img[:h // 12, :w // 12].mean(),
                       img[:h // 12, -w // 12:].mean(),
                       img[-h // 12:, :w // 12].mean(),
                       img[-h // 12:, -w // 12:].mean()])
    center = img[h // 3:-h // 3, w // 3:-w // 3].mean()
    return corners > 0.55 * center


def run(split: str = "train", max_eval: int = MAX_EVAL,
        use_gpu: bool = True) -> dict:
    from scipy.stats import pearsonr, spearmanr

    index = build_index(split)
    forces = np.array([e["force"] for e in index])
    order = np.argsort(forces)
    print(f"{split}: {len(index)} frames, force "
          f"[{forces.min():.2f}, {forces.max():.2f}] N, "
          f"p50={np.median(forces):.2f}", flush=True)

    # This subset has almost no free frames: only 4 of 2686 sit below 0.1 N,
    # the next-lowest is already 1.59 N, and presses are scattered over the
    # full 20x16 mm pad. So the zero map cannot come from "the lowest-force
    # frames" (they'd bake 1.7 N of contact into the reference). Instead:
    #   zero map   = per-pixel MEDIAN of ~60 random frames — any single
    #                pixel is contacted in only a minority of presses, so
    #                the median sees the uncontacted gel;
    #   threshold  = 5-sigma MAD of the genuinely-free frames' residuals.
    near_zero = [index[i] for i in order if forces[i] < 0.5][:6]
    refs = load_images(near_zero)
    already_cropped = _detect_geometry(refs[0])
    print(f"geometry: already_cropped={already_cropped} "
          f"(shape {refs[0].shape}) | free frames: "
          f"{[round(e['force'],2) for e in near_zero]}", flush=True)
    est = DepthForceEstimator(refs, use_gpu=use_gpu,
                              already_cropped=already_cropped,
                              inpaint=False)

    from .depth_force import (MARGIN_X, MARGIN_Y, MM_PER_PIXEL,
                              remove_background_plane)
    rng = np.random.default_rng(0)
    rand_rows = [index[i] for i in rng.choice(len(index), 60, replace=False)]
    est.recon.depth_map_zero = 0.0
    raw = []
    for tar, tar_rows in _group_by_tar(rand_rows).items():
        with tarfile.open(tar) as tf:
            for p in tar_rows:
                img = np.asarray(Image.open(io.BytesIO(
                    tf.extractfile(p["jpg"]).read())).convert("RGB"))
                raw.append(est._raw_depth(est._prep(img)))
    est.recon.depth_map_zero = np.median(np.stack(raw), axis=0)

    resid = np.stack([remove_background_plane(
        (est._raw_depth(est._prep(r)) ) * MM_PER_PIXEL, est._interior)
        for r in refs])
    interior = resid[:, MARGIN_Y:-MARGIN_Y, MARGIN_X:-MARGIN_X]
    sigma = 1.4826 * float(np.median(np.abs(interior)))
    est.contact_threshold_mm = max(5.0 * sigma, 0.01)
    print(f"median-zero calibration: threshold="
          f"{est.contact_threshold_mm*1000:.1f} um", flush=True)

    pick_rows = [index[i] for i in
                 order[np.linspace(0, len(order) - 1,
                                   min(max_eval, len(order))).astype(int)]]
    rows = []
    by_tar: dict[str, list[dict]] = {}
    for p in pick_rows:
        by_tar.setdefault(p["tar"], []).append(p)
    for tar, tar_rows in by_tar.items():
        with tarfile.open(tar) as tf:
            for i, p in enumerate(tar_rows):
                img = np.asarray(Image.open(io.BytesIO(
                    tf.extractfile(p["jpg"]).read())).convert("RGB"))
                r = est.estimate(img)
                rows.append({"force_true": float(p["force"]),
                             "volume_mm3": r.volume_mm3,
                             "probe": str(p.get("obj_name", "?")),
                             "x_mm": float(p["x_mm"]),
                             "y_mm": float(p["y_mm"])})
                if (len(rows)) % 50 == 0:
                    print(f"  {len(rows)}/{len(pick_rows)}", flush=True)

    ft = np.array([r["force_true"] for r in rows])
    v = np.array([r["volume_mm3"] for r in rows])
    good = v > 0
    scale = float(np.median(ft[good] / np.maximum(v[good], 1e-9))) \
        if good.any() else float("nan")
    fcal = v * scale

    per_probe = {}
    for p in sorted({r["probe"] for r in rows}):
        m = np.array([r["probe"] == p for r in rows])
        if m.sum() >= 8:
            per_probe[p] = {
                "n": int(m.sum()),
                "spearman": float(spearmanr(v[m], ft[m]).statistic)}

    # Position stratification: presses near the pad border sit where the
    # raw frames' illumination falloff is steepest and the imprint can clip
    # the sensor edge — measured on probe D, central rho is 2.4x edge rho.
    xs = np.array([r["x_mm"] for r in rows])
    ys = np.array([r["y_mm"] for r in rows])
    central = (xs > 4) & (xs < 16) & (ys > 3) & (ys < 13)

    report = {
        "split": split, "n_frames": len(rows),
        "force_range_n": [float(ft.min()), float(ft.max())],
        "reference_floor_n": float(forces[order[:N_REF]].max()),
        "already_cropped": bool(already_cropped),
        "spearman_rho": float(spearmanr(v, ft).statistic),
        "pearson_r": float(pearsonr(v, ft).statistic),
        "spearman_central": float(spearmanr(v[central], ft[central]).statistic)
            if central.sum() >= 8 else None,
        "spearman_edge": float(spearmanr(v[~central], ft[~central]).statistic)
            if (~central).sum() >= 8 else None,
        "n_central": int(central.sum()),
        "scale_n_per_mm3": scale,
        "mae_calibrated_n": float(np.abs(fcal - ft).mean()),
        "per_probe": per_probe,
        "per_frame": rows,
    }
    (OUT_ROOT / f"fota_cnc_validation_{split}.json").write_text(
        json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    rep = run()
    for k, val in rep.items():
        if k != "per_frame":
            print(k, val)
