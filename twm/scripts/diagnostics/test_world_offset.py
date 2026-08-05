"""One-shot: render a 2026-05-19 episode preview with a manually-applied
world-frame offset (dx, dy, 0) added to every OptiTrack pose. Use this to
test whether the projection overlay re-aligns with the actual sensor
positions in the cam images.

Usage:
    python scripts/test_world_offset.py --episode 2026-05-19/episode_004 --dx 0.3 --dy 0.3
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "/home/yxma/MultimodalData")

import cv2  # noqa
import h5py
import hdf5plugin  # noqa
import numpy as np

from twm.data_collection import REALSENSE_SERIALS
from twm.viz import (
    build_preview_panel,
    draw_projection_overlay,
    load_optitrack,
    optitrack_at,
    load_calibrations,
)


CALIB_DIR = Path("/home/yxma/MultimodalData/twm/calibration/result")
H5_ROOT   = Path("/media/yxma/Disk1/twm/data/motherboard")

PANEL_W, PANEL_H = 1280, 480
SOURCE_FPS = 30.0


def load_proj_calibs():
    cam_calibs, gel_center_left, gel_center_right = load_calibrations(
        [str(CALIB_DIR / "T_mocap_to_cam_middle.json"),
         str(CALIB_DIR / "T_mocap_to_cam_left.json"),
         str(CALIB_DIR / "T_mocap_to_cam_right.json")],
        str(CALIB_DIR / "T_gel_to_rigid_left.json"),
        str(CALIB_DIR / "T_gel_to_rigid_right.json"),
    )
    project_cams = []
    for c in cam_calibs:
        try:
            c_idx = REALSENSE_SERIALS.index(c["camera_serial"])
        except ValueError:
            continue
        project_cams.append({
            "index":          c_idx,
            "T_mocap_to_cam": c["T_mocap_to_cam"],
            "intrinsics":     c["intrinsics"],
            "serial":         c["camera_serial"],
            "rmse":           c.get("rmse_mm", 0.0),
        })
    return project_cams, gel_center_left, gel_center_right


def apply_offset_to_lookup(ot_lookup: dict, dx: float, dy: float, dz: float,
                           verbose: bool = True):
    """`load_optitrack` returns {name: (ts, poses)} tuples (or None). The
    second element `poses` is an (N, 7) numpy array with columns
    [x, y, z, qx, qy, qz, qw]. Mutate columns 0/1/2 in place so subsequent
    `optitrack_at` lookups (which do `poses[idx].tolist()`) see the offset."""
    n_total = 0
    for name, data in ot_lookup.items():
        if data is None:
            if verbose:
                print(f"  {name}: no OT data; skipping", flush=True)
            continue
        ts, poses = data
        if poses.ndim != 2 or poses.shape[1] < 3:
            if verbose:
                print(f"  {name}: unexpected pose shape {poses.shape}; skipping",
                      flush=True)
            continue
        poses[:, 0] += dx
        poses[:, 1] += dy
        poses[:, 2] += dz
        n_total += poses.shape[0]
        if verbose:
            print(f"  {name}: offset (+{dx:.3f}, +{dy:.3f}, +{dz:.3f}) m "
                  f"applied to {poses.shape[0]} samples  "
                  f"(post: x[0]={poses[0, 0]:+.3f} z[0]={poses[0, 2]:+.3f})",
                  flush=True)
    if verbose:
        print(f"  TOTAL pose samples mutated: {n_total}", flush=True)


def render(episode: str, dx: float, dy: float, dz: float,
           clip_s: float, speed: float, out_mp4: Path):
    date, ep_stem = episode.split("/")
    h5_path = H5_ROOT / date / f"{ep_stem}.h5"
    project_cams, glc, grc = load_proj_calibs()

    with h5py.File(str(h5_path), "r") as f:
        cam_ts = f["timestamps"][:]
        T_h5 = len(cam_ts)
        n_target = int(round(clip_s * SOURCE_FPS))
        sample_idx = np.arange(0, min(T_h5, n_target))

        ot_lookup = load_optitrack(f)
        apply_offset_to_lookup(ot_lookup, dx, dy, dz, verbose=True)

        ref_idx = int(sample_idx[0])
        gs_ref_L = f["gelsight/left/frames"][ref_idx]
        gs_ref_R = f["gelsight/right/frames"][ref_idx]

        panels = []
        for f_idx_int in sample_idx:
            f_idx_int = int(f_idx_int)
            color_frames = [f[f"realsense/cam{c}/color"][f_idx_int] for c in range(3)]
            gs_L = f["gelsight/left/frames"][f_idx_int]
            gs_R = f["gelsight/right/frames"][f_idx_int]
            opt_poses = optitrack_at(ot_lookup, float(cam_ts[f_idx_int]))

            panel = build_preview_panel(
                color_frames=color_frames,
                gs_frames=[gs_L, gs_R],
                gs_ref=[gs_ref_L, gs_ref_R],
                optitrack_poses=opt_poses,
                recording=False,
                frame_count=f_idx_int,
                elapsed=float(cam_ts[f_idx_int] - cam_ts[0]),
                fps=SOURCE_FPS,
                task_name="motherboard",
                status_override=(
                    f"OFFSET dx={dx:+.3f} dy={dy:+.3f} dz={dz:+.3f}  {episode}  "
                    f"H5 frame {f_idx_int}/{T_h5}  "
                    f"({float(cam_ts[f_idx_int] - cam_ts[0]):.1f}s)"
                ),
            )
            if project_cams:
                try:
                    draw_projection_overlay(
                        panel, opt_poses, project_cams, glc, grc,
                    )
                except Exception:
                    pass
            panels.append(panel)

    output_fps = SOURCE_FPS * speed
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{PANEL_W}x{PANEL_H}",
        "-r", f"{output_fps}",
        "-i", "-",
        "-c:v", "libx264",
        "-profile:v", "high444",
        "-preset", "medium",
        "-crf", "20",
        "-pix_fmt", "yuv444p",
        "-movflags", "+faststart",
        "-an",
        str(out_mp4),
    ]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for panel in panels:
        p.stdin.write(panel.tobytes())
    p.stdin.close()
    ret = p.wait()
    print(f"ffmpeg exit={ret}; wrote {out_mp4} "
          f"({out_mp4.stat().st_size / 1024:.0f} KB, {len(panels)} frames)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", default="2026-05-19/episode_004")
    ap.add_argument("--dx", type=float, default=0.0)
    ap.add_argument("--dy", type=float, default=0.0)
    ap.add_argument("--dz", type=float, default=0.0)
    ap.add_argument("--clip_s", type=float, default=30.0)
    ap.add_argument("--speed", type=float, default=2.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.out is None:
        ep_id = args.episode.replace("/", "_")
        args.out = (
            f"/media/yxma/Disk1/twm/figures/world_offset_test/"
            f"{ep_id}__dx{args.dx:+.3f}_dy{args.dy:+.3f}_dz{args.dz:+.3f}.mp4"
        )

    render(args.episode, args.dx, args.dy, args.dz, args.clip_s, args.speed, Path(args.out))


if __name__ == "__main__":
    main()
