"""Render real-time (1x) clips around tactile contact-onset events so the
tactile latency (gelsight deformation lag vs visual contact) can be eyeballed.

Per task: pick 5 contact-onset windows from distinct episodes, render ~10s
(300 frames) at 1x in the canonical panel (3 cams + OptiTrack + GelSight
raw/diff + projection overlay) with a frame counter + contact value overlay.

Output: <task>/latency_clips/<date>_<ep>_<start>.mp4
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import h5py
import hdf5plugin  # noqa
import numpy as np
import pyarrow.parquet as pq
import subprocess

sys.path.insert(0, "/home/yxma/MultimodalData")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_episode_previews as BEP  # reuse calib loader + panel path

REL = Path("/media/yxma/Disk1/twm/release")
CALIB = Path("/home/yxma/MultimodalData/twm/calibration")
from twm.calib_epoch import calib_dir
from twm.viz import build_preview_panel, draw_projection_overlay, load_optitrack, optitrack_at

TASK_CFG = {
    "motherboard": {"h5_root": Path("/media/yxma/Disk1/twm/data/motherboard"),
                    "calib_dir": calib_dir("motherboard")},
    "pushT": {"h5_root": Path("/media/yxma/Disk1/twm/data/pushT"),
              "calib_dir": calib_dir("pushT")},
}
WORLD_OFFSET = {("motherboard", "2026-05-19"): (0.23, 0.0, 0.175)}
FPS = 30
CLIP = 300          # 10 s
PRE = 60            # 2 s before onset
PANEL_W, PANEL_H = 1280, 480


def find_onsets(task):
    """Return up to 5 (date, ep_stem, start) sharp contact-onset windows from
    distinct episodes. Adaptive per-episode thresholds; score = onset sharpness."""
    cands = []
    for pqf in sorted((REL / task / "meta").rglob("episode_*.parquet")):
        t = pq.read_table(pqf)
        T = t.num_rows
        if T < CLIP:
            continue
        mix = (t.column("tactile_left_mixed").to_numpy()
               + t.column("tactile_right_mixed").to_numpy()).astype(np.float32)
        # light smooth
        k = np.ones(5, np.float32) / 5
        sm = np.convolve(mix, k, mode="same")
        lo, hi = np.percentile(sm, 25), np.percentile(sm, 75)
        if hi - lo < 1.0:        # episode with no real contact dynamics
            continue
        best = None
        for i in range(PRE + 1, T - (CLIP - PRE)):
            # crossing up to hi, having been near lo just before  => onset
            if sm[i] >= hi and sm[i - 10:i - 2].mean() <= lo:
                sharp = float(sm[i + 3] - sm[i - 7])     # rise magnitude
                if best is None or sharp > best[1]:
                    best = (i, sharp)
        if best:
            cands.append((pqf.parent.name, pqf.stem, max(0, best[0] - PRE), best[1]))
    cands.sort(key=lambda c: -c[3])
    # distinct episodes, top 5
    out, seen = [], set()
    for date, ep, start, sc in cands:
        if ep in seen:
            continue
        seen.add(ep)
        out.append((date, ep, start))
        if len(out) == 5:
            break
    return out


def render_clip(task, date, ep_stem, start, project_cams, glc, grc):
    h5 = TASK_CFG[task]["h5_root"] / date / f"{ep_stem}.h5"
    det = REL / task / "meta" / date / f"{ep_stem}._detect.pt"
    import torch
    trim = int(torch.load(str(det), weights_only=False)["_contact_meta"]["trim_offset"])
    off = WORLD_OFFSET.get((task, date), (0.0, 0.0, 0.0))
    out_mp4 = REL / task / "latency_clips" / f"{date}_{ep_stem}_f{start}.mp4"
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(str(h5), "r") as f:
        cam_ts = f["timestamps"][:]
        ot = load_optitrack(f)
        for nm, data in ot.items():
            if data is not None and any(off):
                ts, ps = data; ps[:, 0] += off[0]; ps[:, 1] += off[1]; ps[:, 2] += off[2]
        end = start + CLIP
        gs_ref_L = f["gelsight/left/frames"][trim + start]
        gs_ref_R = f["gelsight/right/frames"][trim + start]
        proc = subprocess.Popen(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo",
             "-pix_fmt", "bgr24", "-s", f"{PANEL_W}x{PANEL_H}", "-r", str(FPS), "-i", "-",
             "-c:v", "libx264", "-profile:v", "high444", "-preset", "medium", "-crf", "18",
             "-pix_fmt", "yuv444p", "-movflags", "+faststart", "-an", str(out_mp4)],
            stdin=subprocess.PIPE)
        for fi in range(start, end):
            h5f = trim + fi
            color = [f[f"realsense/cam{c}/color"][h5f] for c in range(3)]
            gsL = f["gelsight/left/frames"][h5f]; gsR = f["gelsight/right/frames"][h5f]
            opp = optitrack_at(ot, float(cam_ts[h5f]))
            panel = build_preview_panel(color, [gsL, gsR], [gs_ref_L, gs_ref_R], opp,
                recording=False, frame_count=fi, elapsed=(fi - start) / FPS, fps=FPS,
                task_name=task, status_override=(
                    f"{task} {date}/{ep_stem}  frame {fi}  t={(fi-start)/FPS:.2f}s  "
                    f"[1x REAL-TIME -- compare cam contact vs GelSight diff]"))
            if project_cams:
                try:
                    draw_projection_overlay(panel, opp, project_cams, glc, grc)
                except Exception:
                    pass
            proc.stdin.write(panel.tobytes())
        proc.stdin.close(); proc.wait()
    return out_mp4


def main():
    task = sys.argv[1]
    pc, glc, grc = BEP._load_proj_calibs(task)
    print(f"[latency] {task}: calib={TASK_CFG[task]['calib_dir'].name} cams={len(pc)}", flush=True)
    onsets = find_onsets(task)
    print(f"[latency] {task}: {len(onsets)} onset clips: {[(d,e,s) for d,e,s in onsets]}", flush=True)
    for date, ep, start in onsets:
        p = render_clip(task, date, ep, start, pc, glc, grc)
        print(f"  -> {p.name} ({p.stat().st_size/1024:.0f} KB)", flush=True)
    print("[latency] done", flush=True)


if __name__ == "__main__":
    main()
