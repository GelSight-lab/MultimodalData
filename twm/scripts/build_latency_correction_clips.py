"""Verify the tactile-latency correction visually: render 20s clips at
early/mid/late positions of 2 episodes per task, as a vertical 2-up:

    TOP    = RAW       (cam[i] paired with GelSight[i])      -> tactile lags
    BOTTOM = CORRECTED (cam[i] paired with GelSight[i+SHIFT]) -> should align

If SHIFT is right, in the bottom panel the GelSight diff lights up on the SAME
frame the camera shows contact; in the top panel it lights up ~SHIFT frames late.

Output: <task>/latency_check/<ep>__pos{p}.mp4  (yuv420p, browser-safe)
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import h5py
import hdf5plugin  # noqa
import numpy as np
import subprocess
import torch

sys.path.insert(0, "/home/yxma/MultimodalData")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_episode_previews as BEP
from twm.calib_epoch import calib_dir
from twm.viz import build_preview_panel, draw_projection_overlay, load_optitrack, optitrack_at

REL = Path("/media/yxma/Disk1/twm/release")
CALIB = Path("/home/yxma/MultimodalData/twm/calibration")
TASK_CFG = {
    "motherboard": {"h5_root": Path("/media/yxma/Disk1/twm/data/motherboard"),
                    "calib_dir": calib_dir("motherboard"),
                    "eps": [("2026-05-11", "episode_005"), ("2026-05-11", "episode_012")]},
    "pushT": {"h5_root": Path("/media/yxma/Disk1/twm/data/pushT"),
              "calib_dir": calib_dir("pushT"),
              "eps": [("2026-06-18", "episode_001"), ("2026-06-18", "episode_002")]},
}
WORLD_OFFSET = {("motherboard", "2026-05-19"): (0.23, 0.0, 0.175)}
FPS = 30
LEN = 600          # 20 s
from twm.tactile_align import LEGACY_SHIFT as SHIFT
POSITIONS = [0.10, 0.50, 0.85]
PW, PH = 1280, 480


def label(img, text, color=(0, 255, 180)):
    cv2.rectangle(img, (0, 0), (PW, 22), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return img


def render(task, date, ep, start, pc, glc, grc, out):
    h5 = TASK_CFG[task]["h5_root"] / date / f"{ep}.h5"
    trim = int(torch.load(str(REL/task/'meta'/date/f'{ep}._detect.pt'),
                          weights_only=False)["_contact_meta"]["trim_offset"])
    off = WORLD_OFFSET.get((task, date), (0.0, 0.0, 0.0))
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(h5), "r") as f:
        cam_ts = f["timestamps"][:]
        ot = load_optitrack(f)
        for nm, data in ot.items():
            if data is not None and any(off):
                ts, ps = data; ps[:, 0] += off[0]; ps[:, 1] += off[1]; ps[:, 2] += off[2]
        # refs = clip-start gelsight for each variant
        ref_raw = [f["gelsight/left/frames"][trim+start], f["gelsight/right/frames"][trim+start]]
        ref_cor = [f["gelsight/left/frames"][trim+start+SHIFT], f["gelsight/right/frames"][trim+start+SHIFT]]
        proc = subprocess.Popen(
            ["ffmpeg","-y","-hide_banner","-loglevel","error","-f","rawvideo","-pix_fmt","bgr24",
             "-s",f"{PW}x{PH*2}","-r",str(FPS),"-i","-","-c:v","libx264","-profile:v","high",
             "-pix_fmt","yuv420p","-crf","20","-movflags","+faststart","-an",str(out)],
            stdin=subprocess.PIPE)
        for k in range(LEN):
            i = trim + start + k
            color = [f[f"realsense/cam{c}/color"][i] for c in range(3)]
            opp = optitrack_at(ot, float(cam_ts[i]))
            # RAW: cam[i] + gs[i]
            gsR = [f["gelsight/left/frames"][i], f["gelsight/right/frames"][i]]
            top = build_preview_panel(color, gsR, ref_raw, opp, recording=False,
                frame_count=start+k, elapsed=k/FPS, fps=FPS, task_name=task,
                status_override=f"RAW  {task} {date}/{ep}  frame {start+k}  t={k/FPS:.2f}s  (cam[i] + tactile[i])")
            # CORRECTED: cam[i] + gs[i+SHIFT]
            gsC = [f["gelsight/left/frames"][i+SHIFT], f["gelsight/right/frames"][i+SHIFT]]
            bot = build_preview_panel(color, gsC, ref_cor, opp, recording=False,
                frame_count=start+k, elapsed=k/FPS, fps=FPS, task_name=task,
                status_override=f"CORRECTED +{SHIFT}f  (cam[i] + tactile[i+{SHIFT}])  -- diff should match cam contact")
            if pc:
                try:
                    draw_projection_overlay(top, opp, pc, glc, grc)
                    draw_projection_overlay(bot, opp, pc, glc, grc)
                except Exception:
                    pass
            label(top, "RAW  (tactile lags ~15f)", (120,180,255))
            label(bot, f"CORRECTED +{SHIFT}f", (0,255,180))
            proc.stdin.write(np.vstack([top, bot]).tobytes())
        proc.stdin.close(); proc.wait()


def main():
    task = sys.argv[1]
    pc, glc, grc = BEP._load_proj_calibs(task)
    print(f"[lat-check] {task} calib={TASK_CFG[task]['calib_dir'].name} cams={len(pc)}", flush=True)
    for date, ep in TASK_CFG[task]["eps"]:
        T = int(torch.load(str(REL/task/'meta'/date/f'{ep}._detect.pt'),
                           weights_only=False)["timestamps"].shape[0])
        usable = T - LEN - SHIFT
        for p in POSITIONS:
            start = max(0, int(p * usable))
            out = REL/task/"latency_check"/f"{date}_{ep}__p{int(p*100):02d}.mp4"
            render(task, date, ep, start, pc, glc, grc, out)
            print(f"  {out.name}  start={start} ({p:.0%})  {out.stat().st_size//1024}KB", flush=True)
    print("[lat-check] done", flush=True)


if __name__ == "__main__":
    main()
