"""Encode per-camera depth as lossless FFV1 16-bit video for the release.

For each episode (using the same trim as the RGB release, read from the
release `_detect.pt` sidecar):
    data/<task>/depth/<date>/episode_NNN/depth_{left,middle,right}.mkv

FFV1 / gray16le, lossless, ~10% of raw uint16. Depth is in millimeters;
0 = no return / invalid. Frame i aligns to the RGB video frame i and
parquet row i (source H5 frame trim_offset + i).

Cam mapping matches RGB: cam0->right, cam1->left, cam2->middle.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import subprocess
import sys
import time
from pathlib import Path

import h5py
import hdf5plugin  # noqa
import numpy as np
import torch

H5_ROOTS = {
    "motherboard": Path("/media/yxma/Disk1/twm/data/motherboard"),
    "pushT":       Path("/media/yxma/Disk1/twm/data/pushT"),
}
STAGE = Path("/media/yxma/Disk1/twm/release")
W, H, CHUNK = 640, 480, 128
CAM_DEPTH = {0: "depth_right", 1: "depth_left", 2: "depth_middle"}


def _trim(task, date, ep_stem):
    det = STAGE / task / "meta" / date / f"{ep_stem}._detect.pt"
    if not det.exists():
        return None
    d = torch.load(str(det), weights_only=False, map_location="cpu")
    return int(d["_contact_meta"].get("trim_offset", 0)), int(d["timestamps"].shape[0])


def _ffv1(out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "gray16le", "-s", f"{W}x{H}", "-r", "30",
           "-i", "-", "-c:v", "ffv1", "-level", "3", str(out_path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def process_one(args):
    task, h5_path_str, force = args
    h5_path = Path(h5_path_str)
    date = h5_path.parent.name
    ep_stem = h5_path.stem
    tr = _trim(task, date, ep_stem)
    if tr is None:
        return ep_stem, "skip (no release sidecar)"
    trim, T = tr
    out_dir = STAGE / task / "depth" / date / ep_stem
    if (out_dir / "depth_middle.mkv").exists() and not force:
        return ep_stem, "skipped (exists)"
    try:
        _p = h5py.File(str(h5_path), "r"); _p.close()
    except Exception as e:
        return ep_stem, f"FAIL corrupt ({str(e)[:30]})"

    t0 = time.time()
    with h5py.File(str(h5_path), "r") as f:
        for cam_idx, name in CAM_DEPTH.items():
            ds = f[f"realsense/cam{cam_idx}/depth"]   # (T_h5,480,640) uint16
            proc = _ffv1(out_dir / f"{name}.mkv")
            for s in range(0, T, CHUNK):
                e = min(s + CHUNK, T)
                chunk = ds[trim + s:trim + e].astype("<u2")
                proc.stdin.write(np.ascontiguousarray(chunk).tobytes())
            proc.stdin.close(); proc.wait()
    sz = sum(p.stat().st_size for p in out_dir.glob("*.mkv")) / 1e6
    return ep_stem, f"OK T={T} {sz:.0f}MB {time.time()-t0:.0f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(H5_ROOTS))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    root = H5_ROOTS[args.task]
    h5s = [p for p in sorted(root.rglob("episode_*.h5")) if p.parent.name != "2026-03-23"]
    # only episodes that have a release sidecar (i.e. were encoded)
    h5s = [p for p in h5s if (STAGE / args.task / "meta" / p.parent.name / f"{p.stem}._detect.pt").exists()]
    print(f"[depth] {args.task}: {len(h5s)} episodes, {args.workers} workers", flush=True)
    work = [(args.task, str(p), args.force) for p in h5s]
    with mp.Pool(args.workers) as pool:
        for s, m in pool.imap_unordered(process_one, work):
            print(f"  {s}: {m}", flush=True)
    print("[depth] done", flush=True)


if __name__ == "__main__":
    main()
