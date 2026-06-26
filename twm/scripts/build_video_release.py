"""Task-first LeRobot-style video release builder.

For each source H5 episode, produce:
  <stage>/<task>/videos/<date>/episode_NNN/{view_left,view_middle,view_right,
                                            tactile_left,tactile_right}.mp4
  <stage>/<task>/meta/<date>/episode_NNN.parquet
  <stage>/<task>/meta/<date>/episode_NNN._detect.pt   (poses+scalars only,
        so the validated detect_bad_intervals.py can run unchanged)

Video: 640x480 ORIGINAL resolution, libx264 yuv444p CRF18 30fps. Frame i in
every MP4 == parquet row i == source H5 frame (trim_offset + i).

Cam mapping (verified by calibration serials):
  cam0 -> view_right (143322063538)
  cam1 -> view_left  (104122062574)
  cam2 -> view_middle(217222066989)

Pose / contact-scalar logic is identical to build_episodes_from_h5.py;
contact scalars are computed on the FULL 640x480 GelSight frames.
"""
from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

H5_ROOTS = {
    "motherboard": Path("/media/yxma/Disk1/twm/data/motherboard"),
    "pushT":       Path("/media/yxma/Disk1/twm/data/pushT"),
}
STAGE_ROOT = Path("/media/yxma/Disk1/twm/release")

W, H = 640, 480
TAU = 8.0
P01_SMOOTH_WIN = 30
CRF = "18"
FPS = 30.0
CHUNK = 128

# H5 cam_idx -> stream name
CAM_STREAM = {0: "view_right", 1: "view_left", 2: "view_middle"}
GEL_STREAM = {"left": "tactile_left", "right": "tactile_right"}

# 2026-05-19 world-frame offset (already applied to published motherboard poses)
WORLD_OFFSET = {("motherboard", "2026-05-19"): (0.23, 0.0, 0.175)}


# ── pose alignment (same as build_episodes_from_h5) ──────────────────────────
def cam_align_poses(cam_ts, ot_ts, ot_pose):
    if len(ot_ts) == 0:
        return np.zeros((len(cam_ts), 7), np.float32)
    idx = np.clip(np.searchsorted(ot_ts, cam_ts), 0, len(ot_ts) - 1)
    idxm = np.clip(idx - 1, 0, len(ot_ts) - 1)
    pick_minus = np.abs(ot_ts[idxm] - cam_ts) < np.abs(ot_ts[idx] - cam_ts)
    final = np.where(pick_minus, idxm, idx)
    return ot_pose[final].astype(np.float32)


def find_first_valid(cam_ts, sl_ts, sr_ts):
    th = []
    if sl_ts is not None and len(sl_ts) > 0: th.append(float(sl_ts[0]))
    if sr_ts is not None and len(sr_ts) > 0: th.append(float(sr_ts[0]))
    if not th: return 0
    return int(np.searchsorted(cam_ts, max(th)))


def contact_scalars_full(frame_hwc, ref_hwc):
    """frame/ref are (H,W,3) uint8 (RGB). Returns (intensity, area, mixed)."""
    d = np.sqrt(((frame_hwc.astype(np.float32) - ref_hwc.astype(np.float32)) ** 2).sum(axis=2))
    above = d > TAU
    return float(d.mean()), float(above.mean()), float((d * above).mean())


def open_ffmpeg(out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS),
        "-i", "-", "-c:v", "libx264", "-profile:v", "high444",
        "-preset", "medium", "-crf", CRF, "-pix_fmt", "yuv444p",
        "-movflags", "+faststart", "-an", str(out_path),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def process_one(args):
    task, h5_path_str, force = args
    h5_path = Path(h5_path_str)
    date = h5_path.parent.name
    ep_stem = h5_path.stem
    vid_dir = STAGE_ROOT / task / "videos" / date / ep_stem
    meta_dir = STAGE_ROOT / task / "meta" / date
    pq_path = meta_dir / f"{ep_stem}.parquet"
    det_path = meta_dir / f"{ep_stem}._detect.pt"
    if pq_path.exists() and not force:
        return ep_stem, "skipped (exists)"

    t0 = time.time()
    off = WORLD_OFFSET.get((task, date), (0.0, 0.0, 0.0))
    try:
        _probe = h5py.File(str(h5_path), "r"); _probe.close()
    except Exception as e:
        return ep_stem, f"FAIL corrupt H5 ({str(e)[:40]})"
    with h5py.File(str(h5_path), "r") as f:
        cam_ts = f["timestamps"][:]
        T_h5 = len(cam_ts)
        sl_ts = f["optitrack/sensor_left/timestamps"][:] if "optitrack/sensor_left" in f else None
        sl_pose = f["optitrack/sensor_left/pose"][:] if "optitrack/sensor_left" in f else None
        sr_ts = f["optitrack/sensor_right/timestamps"][:] if "optitrack/sensor_right" in f else None
        sr_pose = f["optitrack/sensor_right/pose"][:] if "optitrack/sensor_right" in f else None
        active = []
        if sl_ts is not None and len(sl_ts) > 0: active.append("left")
        if sr_ts is not None and len(sr_ts) > 0: active.append("right")

        trim = find_first_valid(cam_ts, sl_ts, sr_ts)
        T = T_h5 - trim
        if T <= 0:
            return ep_stem, "FAIL T<=0"

        # ── encode cams ──
        vid_dir.mkdir(parents=True, exist_ok=True)
        for cam_idx, name in CAM_STREAM.items():
            ds = f[f"realsense/cam{cam_idx}/color"]   # (T,480,640,3) BGR
            proc = open_ffmpeg(vid_dir / f"{name}.mp4")
            for s in range(0, T, CHUNK):
                e = min(s + CHUNK, T)
                proc.stdin.write(np.ascontiguousarray(ds[trim + s:trim + e]).tobytes())
            proc.stdin.close(); proc.wait()

        # ── encode gelsight + p01 ref + scalars (two-pass, bounded memory) ──
        scalars = {}
        for side, name in GEL_STREAM.items():
            ds = f[f"gelsight/{side}/frames"]          # (T,480,640,3) RGB
            ref_first = ds[0].astype(np.float32)       # full-res ref (pre-trim frame 0)
            # PASS 1: intensity vs ref_first -> smoothed argmin = p01 idx
            intens = np.zeros(T, np.float32)
            for s in range(0, T, CHUNK):
                e = min(s + CHUNK, T)
                chunk = ds[trim + s:trim + e].astype(np.float32)
                d = np.sqrt(((chunk - ref_first) ** 2).sum(axis=3))
                intens[s:e] = d.mean(axis=(1, 2))
            w = min(P01_SMOOTH_WIN, T)
            sm = np.convolve(intens, np.ones(w)/w, mode="same")
            p01_idx = int(sm.argmin())
            ref_p01_f = ds[trim + p01_idx].astype(np.float32)   # RGB full-res
            # PASS 2: scalars vs ref_p01 + encode (re-read from H5, no full cache)
            I = np.zeros(T, np.float32); A = np.zeros(T, np.float32); M = np.zeros(T, np.float32)
            proc = open_ffmpeg(vid_dir / f"{name}.mp4")
            for s in range(0, T, CHUNK):
                e = min(s + CHUNK, T)
                chunk = ds[trim + s:trim + e]                    # uint8 RGB
                cf = chunk.astype(np.float32)
                dd = np.sqrt(((cf - ref_p01_f) ** 2).sum(axis=3))
                ab = dd > TAU
                I[s:e] = dd.mean(axis=(1, 2))
                A[s:e] = ab.mean(axis=(1, 2))
                M[s:e] = (dd * ab).mean(axis=(1, 2))
                proc.stdin.write(np.ascontiguousarray(chunk[..., ::-1]).tobytes())  # RGB->BGR
            proc.stdin.close(); proc.wait()
            scalars[side] = (I, A, M, p01_idx + trim)
            gc.collect()

        # ── poses ──
        trim_cam_ts = cam_ts[trim:]
        pose_L = cam_align_poses(trim_cam_ts, sl_ts, sl_pose) if sl_pose is not None else np.zeros((T,7),np.float32)
        pose_R = cam_align_poses(trim_cam_ts, sr_ts, sr_pose) if sr_pose is not None else np.zeros((T,7),np.float32)
        pose_L = pose_L.copy(); pose_R = pose_R.copy()
        pose_L[:, 0] += off[0]; pose_L[:, 1] += off[1]; pose_L[:, 2] += off[2]
        pose_R[:, 0] += off[0]; pose_R[:, 1] += off[1]; pose_R[:, 2] += off[2]

    # ── parquet ──
    meta_dir.mkdir(parents=True, exist_ok=True)
    IL, AL, ML, refL = scalars["left"]
    IR, AR, MR, refR = scalars["right"]
    tbl = pa.table({
        "frame_idx": np.arange(T, dtype=np.int32),
        "timestamp": trim_cam_ts.astype(np.float64),
        "sensor_left_pose":  list(pose_L),
        "sensor_right_pose": list(pose_R),
        "tactile_left_intensity": IL, "tactile_left_area": AL, "tactile_left_mixed": ML,
        "tactile_right_intensity": IR, "tactile_right_area": AR, "tactile_right_mixed": MR,
        "source_h5_frame": (np.arange(T) + trim).astype(np.int32),
    })
    pq.write_table(tbl, str(pq_path))

    # ── tiny _detect.pt for the validated detector ──
    torch.save({
        "timestamps": torch.from_numpy(trim_cam_ts.astype(np.float64)),
        "sensor_left_pose": torch.from_numpy(pose_L),
        "sensor_right_pose": torch.from_numpy(pose_R),
        "tactile_left_intensity": torch.from_numpy(IL),
        "tactile_right_intensity": torch.from_numpy(IR),
        "_contact_meta": {"trim_offset": int(trim), "active_sensors": active,
                          "ref_p01_idx_left": int(refL), "ref_p01_idx_right": int(refR),
                          "world_frame_offset_applied": list(off)},
    }, str(det_path))

    return ep_stem, f"OK T={T} trim={trim} active={active} {time.time()-t0:.0f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(H5_ROOTS))
    ap.add_argument("--date", default=None)
    ap.add_argument("--episodes", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    root = H5_ROOTS[args.task]
    h5s = sorted(root.rglob("episode_*.h5"))
    h5s = [p for p in h5s if p.parent.name != "2026-03-23"]
    if args.date: h5s = [p for p in h5s if p.parent.name == args.date]
    if args.episodes: h5s = [p for p in h5s if p.stem in set(args.episodes)]
    if not h5s:
        print("no episodes", file=sys.stderr); sys.exit(1)
    print(f"[release] {args.task}: {len(h5s)} episodes, {args.workers} workers", flush=True)

    work = [(args.task, str(p), args.force) for p in h5s]
    if args.workers <= 1:
        for w in work:
            s, m = process_one(w); print(f"  {s}: {m}", flush=True)
    else:
        with mp.Pool(args.workers) as pool:
            for s, m in pool.imap_unordered(process_one, work):
                print(f"  {s}: {m}", flush=True)
    print("[release] done", flush=True)


if __name__ == "__main__":
    main()
