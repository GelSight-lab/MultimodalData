"""Add per-camera views (left/middle/right) to existing per-episode .pt files.

The current `processed/mode1_v1/` .pt files only store one RealSense camera
as `view` (from H5 cam_idx 0 = right). This script:

  - reads the corresponding source H5
  - extracts cam_idx 0/1/2 (right/left/middle by serial)
  - applies the same center-crop + resize + BGR-keep recipe used to build
    the original `view`
  - writes a new .pt under `processed/episodes/<task>/<date>/episode_NNN.pt`
    with `view_left` / `view_middle` / `view_right` replacing the old single
    `view`. All other keys carry over unchanged from mode1_v1.

Recipe (per camera, per frame), verified against the existing `view`:
    H5 (480, 640, 3) BGR uint8
      -> center-crop columns to (480, 480, 3)
      -> cv2.resize to (128, 128, 3) with INTER_AREA
      -> transpose HWC -> CHW
      -> torch.uint8

Cam mapping (verified via T_mocap_to_cam_*.json serials):
    cam_idx 0  serial 143322063538  =  right    (= existing `view`)
    cam_idx 1  serial 104122062574  =  left
    cam_idx 2  serial 217222066989  =  middle

After this, `build_segments.py` slices the resulting per-episode .pt files
into contiguous clean training segments under `processed/segments/`.
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
import hdf5plugin  # noqa: F401 -- required for compressed H5 datasets
import numpy as np
import torch


MODE1_ROOT = Path("/media/yxma/Disk1/twm/processed/mode1_v1/motherboard")
H5_ROOT    = Path("/media/yxma/Disk1/twm/data/motherboard")
OUT_ROOT   = Path("/media/yxma/Disk1/twm/processed/episodes/motherboard")

THUMB = 128
CHUNK_FRAMES = 256

# H5 cam_idx -> output field-name suffix (verified via calibration serials)
CAM_FIELD = {0: "view_right", 1: "view_left", 2: "view_middle"}


def cam_to_view_chw(frame_bgr_hwc: np.ndarray) -> np.ndarray:
    """(H, W, 3) BGR uint8  ->  (3, 128, 128) BGR uint8 CHW.
    Matches the original `view` build recipe (verified by reproducing it
    against the existing mode1_v1 .pt's at < 1.1 mean abs diff)."""
    H, W = frame_bgr_hwc.shape[:2]
    c = (W - H) // 2                       # center-crop columns to square
    sq = frame_bgr_hwc[:, c:c + H]
    rs = cv2.resize(sq, (THUMB, THUMB), interpolation=cv2.INTER_AREA)
    return rs.transpose(2, 0, 1)           # HWC -> CHW


def extract_cam_views(h5_path: Path, trim_offset: int, T: int) -> dict:
    """Read cams 0/1/2 from the H5 in chunks, build (T, 3, 128, 128) uint8
    BGR-CHW tensors for each. Returns dict keyed by output field name."""
    out: dict[str, np.ndarray] = {
        name: np.empty((T, 3, THUMB, THUMB), dtype=np.uint8)
        for name in CAM_FIELD.values()
    }
    with h5py.File(h5_path, "r") as f:
        cams = {idx: f[f"realsense/cam{idx}/color"] for idx in CAM_FIELD}
        n_h5 = cams[0].shape[0]
        if trim_offset + T > n_h5:
            raise RuntimeError(
                f"H5 has only {n_h5} frames but trim_offset={trim_offset} + T={T} "
                f"= {trim_offset + T} requested")
        for s in range(0, T, CHUNK_FRAMES):
            e = min(s + CHUNK_FRAMES, T)
            for cam_idx, ds in cams.items():
                chunk = ds[trim_offset + s : trim_offset + e]   # (n, 480, 640, 3)
                buf = out[CAM_FIELD[cam_idx]]
                for i in range(chunk.shape[0]):
                    buf[s + i] = cam_to_view_chw(chunk[i])
    return out


def process_one(args) -> tuple:
    pt_path_str, force = args
    pt_path = Path(pt_path_str)
    date = pt_path.parent.name
    ep_stem = pt_path.stem
    h5_path = H5_ROOT / date / f"{ep_stem}.h5"
    out_path = OUT_ROOT / date / f"{ep_stem}.pt"

    if out_path.exists() and not force:
        return ep_stem, "skipped (exists)"
    if not h5_path.exists():
        return ep_stem, f"FAIL no H5 at {h5_path}"

    t0 = time.time()
    ep = torch.load(str(pt_path), weights_only=False, map_location="cpu")
    T = int(ep["view"].shape[0])
    cm = ep.get("_contact_meta", {})
    trim_offset = int(cm.get("trim_offset", 0))

    views_np = extract_cam_views(h5_path, trim_offset, T)

    new_ep: dict = {}
    for k, v in ep.items():
        if k == "view":
            continue        # superseded by view_left/middle/right below
        new_ep[k] = v
    for name, arr in views_np.items():
        new_ep[name] = torch.from_numpy(arr)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".pt.tmp")
    torch.save(new_ep, str(tmp_path))
    os.replace(tmp_path, out_path)

    elapsed = time.time() - t0
    size_gb = out_path.stat().st_size / 1e9
    del new_ep, ep, views_np
    gc.collect()
    return ep_stem, f"OK  T={T}  trim={trim_offset}  {size_gb:.2f} GB  in {elapsed:.1f}s"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=2,
                    help="Number of parallel episodes (memory-heavy; default 2)")
    ap.add_argument("--force", action="store_true",
                    help="Reprocess episodes whose output already exists.")
    ap.add_argument("--episodes", nargs="*", default=None,
                    help="Optional whitelist of <date>/<ep_stem> keys to process.")
    args = ap.parse_args()

    files = sorted(MODE1_ROOT.rglob("episode_*.pt"))
    files = [p for p in files if p.parent.name != "2026-03-23"]    # excluded
    if args.episodes:
        wanted = set(args.episodes)
        files = [p for p in files if f"{p.parent.name}/{p.stem}" in wanted]
    if not files:
        print("No episodes selected.", file=sys.stderr); sys.exit(1)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"[multicam] processing {len(files)} episodes with {args.workers} worker(s)",
          flush=True)
    t_start = time.time()

    work = [(str(p), args.force) for p in files]
    if args.workers <= 1:
        results = [process_one(w) for w in work]
    else:
        with mp.Pool(processes=args.workers) as pool:
            results = []
            for r in pool.imap_unordered(process_one, work):
                results.append(r)
                stem, msg = r
                print(f"  {stem}: {msg}", flush=True)
    if args.workers <= 1:
        for stem, msg in results:
            print(f"  {stem}: {msg}", flush=True)

    n_ok = sum(1 for _, m in results if m.startswith("OK"))
    n_skip = sum(1 for _, m in results if m.startswith("skipped"))
    n_fail = len(results) - n_ok - n_skip
    print(f"\n[multicam] done in {time.time() - t_start:.1f}s — "
          f"{n_ok} OK, {n_skip} skipped, {n_fail} failed.", flush=True)


if __name__ == "__main__":
    main()
