"""Stage-1 builder: H5 -> per-episode .pt under `processed/episodes/`.

Used for any *new* raw H5 episodes that don't yet have a corresponding
mode1_v1 .pt. Produces the same schema that
`build_episodes_multicam.py` produces for old data (which transplants from
the legacy `mode1_v1/` instead of building from scratch).

Schema produced
---------------
  view_left, view_middle, view_right   (T, 3, 128, 128) uint8 BGR-CHW
  tactile_left, tactile_right          (T, 3, 128, 128) uint8 RGB-CHW
  sensor_left_pose, sensor_right_pose  (T, 7)           float32  cam-aligned
  timestamps                           (T,)             float64  cam epoch sec
  tactile_*_intensity / area / mixed   (T,)             float32  contact scalars
  _contact_meta                        dict             curation provenance

Recipe (verified bit-for-bit against existing mode1_v1 .pt files)
-----------------------------------------------------------------
  View per cam (cam_idx 0/1/2 = right/left/middle by calibration serial):
      H5 (480, 640, 3) BGR uint8
        -> center-crop columns to (480, 480, 3)
        -> cv2.resize to (128, 128, 3) with INTER_AREA
        -> transpose HWC -> CHW
        -> keep BGR
  Tactile per side:
      same recipe but keep RGB (gelsight is natively RGB)
  Pose per side:
      nearest-timestamp lookup in `optitrack/sensor_<side>/pose` (N, 7)
        (xyz + quat in float64 -> float32)
  Contact scalars (per side, given ref_p01 frame in same 128x128 RGB form,
                   tau=8.0, L2 norm across RGB channels):
      d[h, w] = sqrt( sum_c (frame[c, h, w] - ref_p01[c, h, w])^2 )
      intensity[t] = mean(d)
      area[t]      = mean(d > tau)
      mixed[t]     = mean(d * (d > tau))

p01 reference selection (clean reproducible recipe, not bit-identical with
the original builder which used an opaque smoothed criterion)
-----------------------------------------------------------------
  ref_first = gelsight processed from H5 frame 0 (pre-trim)
  intensity_vs_first[t] = mean(d(tactile[t], ref_first))  computed for
      t in trimmed-pt range
  smoothed = uniform-window mean of intensity_vs_first with window=30 frames
  ref_p01_idx_pt = argmin(smoothed)
  ref_p01_idx (H5 coords) = ref_p01_idx_pt + trim_offset

drift = mean(d(ref_first, ref_p01))     (per side)

Trim logic
----------
  first_valid_cam = first cam-frame index where every active sensor has at
                    least one OT sample at or before that frame's timestamp.
  Active sensors auto-detected: a side is active if H5 has OT data on it.
"""
from __future__ import annotations

import argparse
import gc
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import cv2
import h5py
import hdf5plugin  # noqa: F401 (required for compressed H5 datasets)
import numpy as np
import torch


H5_ROOT  = Path("/media/yxma/Disk1/twm/data/motherboard")
OUT_ROOT = Path("/media/yxma/Disk1/twm/processed/episodes/motherboard")

THUMB = 128
TAU = 8.0
RGB_NORM = "L2"
FORMULA_VERSION = "v1"
P01_SMOOTH_WIN = 30          # frames for the smoothed-intensity argmin

# Cam_idx -> output field name (verified via T_mocap_to_cam_*.json serials)
CAM_FIELD = {0: "view_right", 1: "view_left", 2: "view_middle"}


# ──────────────────────────────────────────────────────────────────────────
# Image conversion
# ──────────────────────────────────────────────────────────────────────────

def to_thumb_chw(img_hwc: np.ndarray) -> np.ndarray:
    """640x480 HWC uint8 -> 128x128 CHW uint8 via center-crop+INTER_AREA.
    Channel ordering preserved as-is (caller decides BGR vs RGB)."""
    H, W = img_hwc.shape[:2]
    c = (W - H) // 2
    sq = img_hwc[:, c:c + H]
    rs = cv2.resize(sq, (THUMB, THUMB), interpolation=cv2.INTER_AREA)
    return rs.transpose(2, 0, 1)


# ──────────────────────────────────────────────────────────────────────────
# OT cam-alignment
# ──────────────────────────────────────────────────────────────────────────

def cam_align_poses(cam_ts: np.ndarray, ot_ts: np.ndarray, ot_pose: np.ndarray) -> np.ndarray:
    """For each cam_ts[i], return ot_pose at nearest ot_ts. (T, 7) float32."""
    idx = np.searchsorted(ot_ts, cam_ts)
    idx = np.clip(idx, 0, len(ot_ts) - 1)
    idx_minus = np.clip(idx - 1, 0, len(ot_ts) - 1)
    # Prefer the closer of idx vs idx-1
    pick_minus = np.abs(ot_ts[idx_minus] - cam_ts) < np.abs(ot_ts[idx] - cam_ts)
    final = np.where(pick_minus, idx_minus, idx)
    return ot_pose[final].astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────
# Trim logic
# ──────────────────────────────────────────────────────────────────────────

def find_first_valid(cam_ts: np.ndarray, sl_ts: np.ndarray | None,
                     sr_ts: np.ndarray | None) -> int:
    """First cam frame at which every active sensor has >=1 OT sample."""
    thresholds = []
    if sl_ts is not None and len(sl_ts) > 0:
        thresholds.append(float(sl_ts[0]))
    if sr_ts is not None and len(sr_ts) > 0:
        thresholds.append(float(sr_ts[0]))
    if not thresholds:
        return 0
    return int(np.searchsorted(cam_ts, max(thresholds)))


# ──────────────────────────────────────────────────────────────────────────
# Contact scalars
# ──────────────────────────────────────────────────────────────────────────

def contact_scalars(frame_chw: np.ndarray, ref_chw: np.ndarray):
    """Per-frame contact scalars vs reference. frame/ref are (3, H, W) uint8."""
    d = np.sqrt(((frame_chw.astype(np.float32) - ref_chw.astype(np.float32)) ** 2).sum(axis=0))
    above = d > TAU
    return float(d.mean()), float(above.mean()), float((d * above).mean())


# ──────────────────────────────────────────────────────────────────────────
# Per-episode worker
# ──────────────────────────────────────────────────────────────────────────

def process_one(args):
    h5_path_str, force = args
    h5_path = Path(h5_path_str)
    date = h5_path.parent.name
    ep_stem = h5_path.stem
    out_path = OUT_ROOT / date / f"{ep_stem}.pt"
    if out_path.exists() and not force:
        return ep_stem, "skipped (exists)"

    t0 = time.time()
    with h5py.File(str(h5_path), "r") as f:
        cam_ts = f["timestamps"][:]
        T_h5 = len(cam_ts)

        active = []
        sl_ts = sr_ts = None
        sl_pose = sr_pose = None
        if "optitrack/sensor_left" in f:
            sl_ts   = f["optitrack/sensor_left/timestamps"][:]
            sl_pose = f["optitrack/sensor_left/pose"][:]
            if len(sl_ts) > 0:
                active.append("left")
        if "optitrack/sensor_right" in f:
            sr_ts   = f["optitrack/sensor_right/timestamps"][:]
            sr_pose = f["optitrack/sensor_right/pose"][:]
            if len(sr_ts) > 0:
                active.append("right")

        trim_offset = find_first_valid(cam_ts, sl_ts, sr_ts)
        T = T_h5 - trim_offset
        if T <= 0:
            return ep_stem, "FAIL T<=0 after trim"

        trim_reason = ("OT not yet streaming at recording start"
                       if trim_offset > 0 else "no trim required")

        # ── Cams ──
        views = {}
        for cam_idx, fname in CAM_FIELD.items():
            ds = f[f"realsense/cam{cam_idx}/color"]
            buf = np.empty((T, 3, THUMB, THUMB), np.uint8)
            CHUNK = 256
            for s in range(0, T, CHUNK):
                e = min(s + CHUNK, T)
                chunk = ds[trim_offset + s : trim_offset + e]   # (n, 480, 640, 3) BGR
                for i in range(chunk.shape[0]):
                    buf[s + i] = to_thumb_chw(chunk[i])
            views[fname] = torch.from_numpy(buf)

        # ── Tactiles + p01 refs + scalars ──
        tactiles  = {}
        cm: dict = {
            "formula_version":    FORMULA_VERSION,
            "tau":                TAU,
            "rgb_norm":           RGB_NORM,
            "reference_strategy": "p01",
        }
        scalars_out = {}
        for side in ("left", "right"):
            ds = f[f"gelsight/{side}/frames"]
            buf = np.empty((T, 3, THUMB, THUMB), np.uint8)
            CHUNK = 256
            for s in range(0, T, CHUNK):
                e = min(s + CHUNK, T)
                chunk = ds[trim_offset + s : trim_offset + e]   # (n, 480, 640, 3) RGB
                for i in range(chunk.shape[0]):
                    buf[s + i] = to_thumb_chw(chunk[i])
            tactiles[f"tactile_{side}"] = torch.from_numpy(buf)

            # ref_first := gelsight processed from H5 frame 0 (pre-trim)
            ref_first = to_thumb_chw(ds[0])
            cm[f"ref_first_{side}"] = torch.from_numpy(ref_first)

            # intensity vs ref_first across trimmed range
            intens_vs_first = np.zeros(T, np.float32)
            for t in range(T):
                d = np.sqrt(((buf[t].astype(np.float32) - ref_first.astype(np.float32)) ** 2).sum(axis=0))
                intens_vs_first[t] = d.mean()
            # Smoothed argmin (P01_SMOOTH_WIN-frame uniform window)
            w = min(P01_SMOOTH_WIN, T)
            kernel = np.ones(w, dtype=np.float32) / w
            smoothed = np.convolve(intens_vs_first, kernel, mode="same")
            ref_p01_idx_pt = int(smoothed.argmin())
            ref_p01 = buf[ref_p01_idx_pt].copy()
            cm[f"ref_p01_{side}"]     = torch.from_numpy(ref_p01)
            cm[f"ref_p01_idx_{side}"] = ref_p01_idx_pt + trim_offset
            # drift
            cm[f"drift_{side}"] = float(np.sqrt(
                ((ref_first.astype(np.float32) - ref_p01.astype(np.float32)) ** 2).sum(axis=0)
            ).mean())

            # contact scalars vs ref_p01
            I = np.zeros(T, np.float32); A = np.zeros(T, np.float32); M = np.zeros(T, np.float32)
            for t in range(T):
                I[t], A[t], M[t] = contact_scalars(buf[t], ref_p01)
            scalars_out[f"tactile_{side}_intensity"] = torch.from_numpy(I)
            scalars_out[f"tactile_{side}_area"]      = torch.from_numpy(A)
            scalars_out[f"tactile_{side}_mixed"]     = torch.from_numpy(M)

        cm["drift_warning"]     = bool(max(cm["drift_left"], cm["drift_right"]) > 5.0)
        cm["trim_offset"]       = int(trim_offset)
        cm["trim_reason"]       = trim_reason
        cm["pre_trim_n_frames"] = int(T_h5)
        cm["active_sensors"]    = active

        # ── Poses (cam-aligned to trimmed cam_ts) ──
        trim_cam_ts = cam_ts[trim_offset:]
        if sl_pose is not None:
            sensor_left_pose = cam_align_poses(trim_cam_ts, sl_ts, sl_pose)
        else:
            sensor_left_pose = np.zeros((T, 7), np.float32)
        if sr_pose is not None:
            sensor_right_pose = cam_align_poses(trim_cam_ts, sr_ts, sr_pose)
        else:
            sensor_right_pose = np.zeros((T, 7), np.float32)

    out_dict = {
        **views,
        **tactiles,
        "timestamps":        torch.from_numpy(trim_cam_ts.astype(np.float64)),
        "sensor_left_pose":  torch.from_numpy(sensor_left_pose),
        "sensor_right_pose": torch.from_numpy(sensor_right_pose),
        **scalars_out,
        "_contact_meta":     cm,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".pt.tmp")
    torch.save(out_dict, str(tmp))
    os.replace(tmp, out_path)

    elapsed = time.time() - t0
    size_gb = out_path.stat().st_size / 1e9
    del out_dict, views, tactiles, scalars_out
    gc.collect()
    return ep_stem, (f"OK  T_h5={T_h5}  trim={trim_offset}  T={T}  active={active}  "
                     f"{size_gb:.2f} GB  in {elapsed:.1f}s")


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--date", required=True, help="Date subfolder, e.g. 2026-05-19")
    ap.add_argument("--workers", type=int, default=2,
                    help="Number of parallel episodes (memory-heavy; default 2)")
    ap.add_argument("--force", action="store_true",
                    help="Reprocess episodes whose output already exists.")
    args = ap.parse_args()

    h5_dir = H5_ROOT / args.date
    if not h5_dir.is_dir():
        print(f"No such date dir: {h5_dir}", file=sys.stderr); sys.exit(1)
    h5_files = sorted(h5_dir.glob("episode_*.h5"))
    if not h5_files:
        print(f"No episode_*.h5 under {h5_dir}", file=sys.stderr); sys.exit(1)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"[stage1] {len(h5_files)} episode(s) in {args.date} with {args.workers} worker(s)",
          flush=True)
    t_start = time.time()

    work = [(str(p), args.force) for p in h5_files]
    results = []
    if args.workers <= 1:
        for w in work:
            stem, msg = process_one(w)
            print(f"  {stem}: {msg}", flush=True)
            results.append((stem, msg))
    else:
        with mp.Pool(processes=args.workers) as pool:
            for stem, msg in pool.imap_unordered(process_one, work):
                print(f"  {stem}: {msg}", flush=True)
                results.append((stem, msg))

    n_ok   = sum(1 for _, m in results if m.startswith("OK"))
    n_skip = sum(1 for _, m in results if m.startswith("skipped"))
    n_fail = len(results) - n_ok - n_skip
    print(f"\n[stage1] done in {time.time()-t_start:.1f}s — "
          f"{n_ok} OK, {n_skip} skipped, {n_fail} failed.", flush=True)


if __name__ == "__main__":
    main()
