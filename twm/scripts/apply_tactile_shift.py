"""Bake the tactile latency correction into the release: shift each tactile
stream EARLIER by SHIFT frames so tactile[i] aligns with view[i].

Per episode (operating on the already-published release, no H5 needed):
  - decode tactile_{left,right}.mp4  (T frames)
  - shifted[i] = frames[min(i+SHIFT, T-1)]   (tail repeats last frame)
  - re-encode tactile_{left,right}.mp4  (yuv444p CRF18, matches dataset)
  - recompute tactile_{L,R}_{intensity,area,mixed} on shifted frames vs the
    p01 reference (frame ref_p01_idx-trim of the original stream)
  - rewrite those 4 parquet columns; update the _detect.pt intensity arrays
RGB / depth / poses / lengths are untouched (only tactile is lagged).

After this, re-run detect_bad_intervals + build_release_curation (spikes follow
the shifted tactile; pose-based ot_loss/teleport unchanged), then re-upload.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import subprocess
import sys
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

REL = Path("/media/yxma/Disk1/twm/release")
TAU = 8.0
W, H = 640, 480


def decode_all(mp4):
    c = av.open(str(mp4))
    out = [f.to_ndarray(format="rgb24") for f in c.decode(c.streams.video[0])]
    c.close()
    return out                      # list of (H,W,3) RGB uint8


def encode_all(frames_rgb, dst):
    tmp = dst.with_suffix(".shift.mp4")
    p = subprocess.Popen(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo",
         "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", "30", "-i", "-",
         "-c:v", "libx264", "-profile:v", "high444", "-preset", "medium",
         "-crf", "18", "-pix_fmt", "yuv444p", "-movflags", "+faststart", "-an", str(tmp)],
        stdin=subprocess.PIPE)
    for f in frames_rgb:
        p.stdin.write(np.ascontiguousarray(f[..., ::-1]).tobytes())   # RGB->BGR
    p.stdin.close(); p.wait()
    tmp.replace(dst)


def scalars(frames_rgb, ref_rgb):
    ref = ref_rgb.astype(np.float32)
    I = np.empty(len(frames_rgb), np.float32); A = np.empty_like(I); M = np.empty_like(I)
    for i, fr in enumerate(frames_rgb):
        d = np.sqrt(((fr.astype(np.float32) - ref) ** 2).sum(axis=2))
        ab = d > TAU
        I[i] = d.mean(); A[i] = ab.mean(); M[i] = (d * ab).mean()
    return I, A, M


def process(args):
    task, date, stem, shift = args
    base = REL / task
    det_path = base / "meta" / date / f"{stem}._detect.pt"
    det = torch.load(str(det_path), weights_only=False)
    if int(det["_contact_meta"].get("tactile_latency_corrected", 0)) != 0:
        return f"{task}/{date}/{stem} SKIP (already corrected)"
    trim = int(det["_contact_meta"]["trim_offset"])
    vd = base / "videos" / date / stem
    pqf = base / "meta" / date / f"{stem}.parquet"
    tbl = pq.read_table(pqf)
    T = tbl.num_rows
    cols = {c: tbl.column(c).to_pylist() for c in tbl.column_names}
    for side, scal_idx_key in (("left", "ref_p01_idx_left"), ("right", "ref_p01_idx_right")):
        frames = decode_all(vd / f"tactile_{side}.mp4")
        if len(frames) != T:
            # tolerate ±1 from decoder; clamp
            T2 = min(T, len(frames)); frames = frames[:T2]
        n = len(frames)
        ref_rel = int(det["_contact_meta"].get(scal_idx_key, trim)) - trim
        ref_rel = max(0, min(n - 1, ref_rel))
        ref = frames[ref_rel]
        shifted = [frames[min(i + shift, n - 1)] for i in range(n)]
        encode_all(shifted, vd / f"tactile_{side}.mp4")
        I, A, M = scalars(shifted, ref)
        S = "left" if side == "left" else "right"
        # pad to T if decoder returned fewer
        def padto(a):
            if len(a) < T:
                a = np.concatenate([a, np.repeat(a[-1], T - len(a))])
            return a[:T]
        cols[f"tactile_{S}_intensity"] = padto(I).tolist()
        cols[f"tactile_{S}_area"] = padto(A).tolist()
        cols[f"tactile_{S}_mixed"] = padto(M).tolist()
        det[f"tactile_{S}_intensity"] = torch.from_numpy(padto(I))
    pq.write_table(pa.table(cols), str(pqf))
    det["_contact_meta"]["tactile_latency_corrected"] = int(shift)
    torch.save(det, str(det_path))
    return f"{task}/{date}/{stem} shifted {shift}f (T={T})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--shift", type=int, default=15)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    dets = sorted((REL / args.task / "meta").rglob("*._detect.pt"))
    work = [(args.task, d.parent.name, d.name.replace("._detect.pt", ""), args.shift) for d in dets]
    print(f"[shift] {args.task}: {len(work)} episodes, shift={args.shift}, workers={args.workers}", flush=True)
    with mp.Pool(args.workers) as pool:
        for msg in pool.imap_unordered(process, work):
            print("  " + msg, flush=True)
    print("[shift] done", flush=True)


if __name__ == "__main__":
    main()
