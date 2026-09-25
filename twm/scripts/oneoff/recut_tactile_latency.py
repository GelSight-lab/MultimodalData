"""DESTRUCTIVE re-alignment: shift the tactile streams forward by N frames to
compensate the recording-side GelSight acquisition lag, baking the correction
into the published data (so even non-loader / lerobot consumers get aligned
tactile).

Pairing: view[i] aligns with tactile[i+N]. So the corrected episode of length
T-N has:
    view_*/depth_*/poses[i]  = original[i]          for i in 0 .. T-N-1
    tactile_*[i]             = original[i+N]
    tactile scalars[i]       = original[i+N]

This re-encodes the 2 tactile MP4s per episode (shifted+trimmed), rewrites the
parquet (view-side cols trimmed to T-N; tactile cols shifted), and trims depth
+ view by N frames at the tail. Segments.json frame ranges are clamped to T-N.

RUN ONLY after confirming N on the rig with camera_stream/measure_gelsight_latency.py.
This rewrites the release staging dir; re-publish afterwards.

    python scripts/recut_tactile_latency.py --task motherboard --n 15 --dry_run
    python scripts/recut_tactile_latency.py --task motherboard --n 15
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REL = Path("/media/yxma/Disk1/twm/release")
W, H, FPS = 640, 480, 30
TACT = ("tactile_left", "tactile_right")
VIEW = ("view_left", "view_middle", "view_right")


def reencode_shift(src_mp4, dst_mp4, n, T):
    """Write frames [n .. T-1] of src to dst (drops first n => shifts earlier)."""
    c = av.open(str(src_mp4))
    frames = []
    for i, fr in enumerate(c.decode(c.streams.video[0])):
        if i >= n:
            frames.append(fr.to_ndarray(format="bgr24"))
    c.close()
    dst_mp4.parent.mkdir(parents=True, exist_ok=True)
    p = subprocess.Popen(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo",
         "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-profile:v", "high", "-pix_fmt", "yuv420p", "-crf", "18",
         "-movflags", "+faststart", "-an", str(dst_mp4)], stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close(); p.wait()
    return len(frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--n", type=int, required=True, help="tactile latency in frames")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    n = args.n
    meta_root = REL / args.task / "meta"
    vid_root = REL / args.task / "videos"
    print(f"[recut] {args.task} N={n} frames  (view[i] <-> tactile[i+{n}])", flush=True)

    for pqf in sorted(meta_root.rglob("episode_*.parquet")):
        date, stem = pqf.parent.name, pqf.stem
        tbl = pq.read_table(pqf)
        T = tbl.num_rows
        Tn = T - n
        if Tn <= 0:
            print(f"  skip {date}/{stem}: T={T} <= N"); continue
        print(f"  {date}/{stem}: T={T} -> {Tn}", flush=True)
        if args.dry_run:
            continue
        # rewrite parquet: view cols [0:Tn], tactile cols [n:T]
        cols = {}
        for c in tbl.column_names:
            arr = tbl.column(c).to_pylist()
            if c.startswith("tactile_") and ("intensity" in c or "area" in c or "mixed" in c):
                cols[c] = arr[n:T]
            else:
                cols[c] = arr[0:Tn]
        # fix frame_idx/index continuity
        cols["frame_idx"] = list(range(Tn))
        if "frame_index" in cols:
            cols["frame_index"] = list(range(Tn))
        pq.write_table(pa.table(cols), str(pqf))
        # re-encode view + depth: trim tail by n (frames [0:Tn]) ; tactile: shift [n:T]
        vdir = vid_root / date / stem
        for s in VIEW:
            tmp = vdir / f"{s}.shift.mp4"
            # view: keep first Tn frames -> drop last n
            c = av.open(str(vdir / f"{s}.mp4")); frames = []
            for i, fr in enumerate(c.decode(c.streams.video[0])):
                if i < Tn: frames.append(fr.to_ndarray(format="bgr24"))
            c.close()
            pp = subprocess.Popen(["ffmpeg","-y","-hide_banner","-loglevel","error","-f","rawvideo",
              "-pix_fmt","bgr24","-s",f"{W}x{H}","-r",str(FPS),"-i","-","-c:v","libx264",
              "-profile:v","high","-pix_fmt","yuv420p","-crf","18","-movflags","+faststart","-an",str(tmp)],
              stdin=subprocess.PIPE)
            for f in frames: pp.stdin.write(np.ascontiguousarray(f).tobytes())
            pp.stdin.close(); pp.wait(); tmp.replace(vdir / f"{s}.mp4")
        for s in TACT:
            tmp = vdir / f"{s}.shift.mp4"
            reencode_shift(vdir / f"{s}.mp4", tmp, n, T); tmp.replace(vdir / f"{s}.mp4")

    # clamp segments.json ranges
    if not args.dry_run:
        sjp = REL / args.task / "segments.json"
        sj = json.loads(sjp.read_text())
        for s in sj["segments"]:
            a, b = s["frame_range"]
            s["frame_range"] = [a, min(b, b - n)]   # shrink each episode by n at tail
            s["n_frames"] = s["frame_range"][1] - s["frame_range"][0] + 1
        sjp.write_text(json.dumps(sj, indent=2))
    print(f"[recut] done (depth NOT shifted — it is a view-side cam).", flush=True)
    print("[recut] NOTE: re-run build_release_curation? segments clamped in place; "
          "re-publish videos+meta+segments after this.", flush=True)


if __name__ == "__main__":
    main()
