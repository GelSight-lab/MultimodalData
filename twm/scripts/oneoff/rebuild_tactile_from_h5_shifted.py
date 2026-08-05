"""Rebuild tactile streams with +SHIFT alignment DIRECTLY from raw H5 (single
h264 generation, full-quality scalars). Replaces the earlier shift-from-MP4
approach which double-compressed video and recomputed scalars from lossy MP4.

Corrected output position i shows what the raw sensor saw at i+SHIFT:
    tactile_video[i]   = H5 gelsight[trim + min(i+SHIFT, Tproc-1)]
    tactile_scalar[i]  = contact(H5 gelsight[trim+min(i+SHIFT,Tproc-1)], ref_p01)
Tail (last SHIFT frames) repeats the final real frame. RGB/depth/pose/length
unchanged. Re-does ALL episodes (uniform, idempotent via flag overwrite).
"""
from __future__ import annotations

import argparse, multiprocessing as mp, subprocess
from pathlib import Path
import h5py, hdf5plugin  # noqa
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, torch

REL = Path("/media/yxma/Disk1/twm/release")
H5_ROOTS = {"motherboard": Path("/media/yxma/Disk1/twm/data/motherboard"),
            "pushT": Path("/media/yxma/Disk1/twm/data/pushT")}
TAU, W, H, CHUNK = 8.0, 640, 480, 128


def encode(frames_rgb, dst):
    tmp = dst.with_suffix(".reb.mp4")
    p = subprocess.Popen(["ffmpeg","-y","-hide_banner","-loglevel","error","-f","rawvideo",
        "-pix_fmt","bgr24","-s",f"{W}x{H}","-r","30","-i","-","-c:v","libx264",
        "-profile:v","high444","-preset","medium","-crf","18","-pix_fmt","yuv444p",
        "-movflags","+faststart","-an",str(tmp)], stdin=subprocess.PIPE)
    for f in frames_rgb:
        p.stdin.write(np.ascontiguousarray(f[..., ::-1]).tobytes())   # RGB->BGR
    p.stdin.close(); p.wait(); tmp.replace(dst)


def process(args):
    task, date, stem, shift = args
    det_p = REL/task/"meta"/date/f"{stem}._detect.pt"
    det = torch.load(str(det_p), weights_only=False)
    cm = det["_contact_meta"]; trim = int(cm["trim_offset"])
    pqf = REL/task/"meta"/date/f"{stem}.parquet"
    tbl = pq.read_table(pqf); Tproc = tbl.num_rows
    cols = {c: tbl.column(c).to_pylist() for c in tbl.column_names}
    h5 = H5_ROOTS[task]/date/f"{stem}.h5"
    with h5py.File(str(h5), "r") as f:
        for side, ridx_key, Skey in (("left","ref_p01_idx_left","left"),
                                     ("right","ref_p01_idx_right","right")):
            ds = f[f"gelsight/{side}/frames"]; N = ds.shape[0]
            ref = ds[int(cm.get(ridx_key, trim))].astype(np.float32)   # raw H5 ref
            # Corrected pos i -> H5 frame trim+min(i+shift,Tproc-1), clamped to N-1.
            # This is a CONTIGUOUS slab [trim+shift .. N-1] then the last frame
            # repeated `shift` times -> read in fast chunked slabs, not per-frame.
            lo = min(trim + shift, N - 1); hi = N        # slab [lo, N)
            n_slab = hi - lo                              # = Tproc - shift (typically)
            frames = []
            I = np.empty(Tproc, np.float32); A = np.empty_like(I); M = np.empty_like(I)
            pos = 0
            def _accum(fr, k):
                d = np.sqrt(((fr.astype(np.float32) - ref) ** 2).sum(axis=2)); ab = d > TAU
                I[k] = d.mean(); A[k] = ab.mean(); M[k] = (d * ab).mean()
            for s in range(lo, hi, CHUNK):
                e = min(s + CHUNK, hi)
                slab = ds[s:e]                            # contiguous chunked read
                for j in range(slab.shape[0]):
                    if pos >= Tproc: break
                    fr = slab[j]; frames.append(fr); _accum(fr, pos); pos += 1
            # tail: repeat last real frame to fill Tproc
            last = frames[-1]
            while pos < Tproc:
                frames.append(last); _accum(last, pos); pos += 1
            encode(frames, REL/task/"videos"/date/stem/f"tactile_{side}.mp4")
            cols[f"tactile_{Skey}_intensity"] = I.tolist()
            cols[f"tactile_{Skey}_area"] = A.tolist()
            cols[f"tactile_{Skey}_mixed"] = M.tolist()
            det[f"tactile_{Skey}_intensity"] = torch.from_numpy(I)
    pq.write_table(pa.table(cols), str(pqf))
    cm["tactile_latency_corrected"] = int(shift)
    cm["tactile_latency_method"] = "rebuilt_from_h5"
    torch.save(det, str(det_p))
    return f"{task}/{date}/{stem} rebuilt+{shift} (T={Tproc})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True); ap.add_argument("--shift", type=int, default=15)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    dets = sorted((REL/a.task/"meta").rglob("*._detect.pt"))
    work = [(a.task, d.parent.name, d.name.replace("._detect.pt",""), a.shift) for d in dets]
    print(f"[rebuild] {a.task}: {len(work)} eps, shift={a.shift}, workers={a.workers}", flush=True)
    with mp.Pool(a.workers) as pool:
        for m in pool.imap_unordered(process, work): print("  "+m, flush=True)
    print("[rebuild] done", flush=True)


if __name__ == "__main__":
    main()
