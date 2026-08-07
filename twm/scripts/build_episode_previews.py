"""Generate per-episode preview MP4s in the canonical viewer layout.

Each preview shows the **first 30 seconds of usable data** (i.e. frames
after the OT-uninitialized trim prefix) played back at **2x speed**, so
each preview is 15 s long regardless of episode length (or shorter if the
recording itself is < 30 s). Compared to the previous "flipbook" recipe
that sampled 60 frames evenly across the whole episode and played at
50-120x speed, this gives users a real sense of the actual motion.

  - Source frames consumed: trim_offset .. trim_offset + 900 (or end of H5)
  - Output encoding: 60 fps (= 2x the 30 fps recording rate), yuv444p H.264 CRF 20
  - Output length: 15 s (or T/60 s if T < 900)
  - File size: roughly 10-25 MB per preview

Reuses `twm.viz.build_preview_panel` + `draw_projection_overlay` for the
exact same layout as the live viewer / `play_react_pt.py`.

Usage
-----
    python scripts/build_episode_previews.py --date 2026-05-19
    python scripts/build_episode_previews.py --date 2026-05-10 --clip_s 30 --speed 2

Output: /media/yxma/Disk1/twm/figures/episode_previews/<task>/<date>/episode_NNN.mp4
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import h5py
import hdf5plugin  # noqa: F401
import numpy as np

sys.path.insert(0, "/home/yxma/MultimodalData")

from twm.data_collection import REALSENSE_SERIALS    # noqa
from twm.viz import (
    build_preview_panel,
    draw_projection_overlay,
    cam_aligned_pose,
    load_optitrack,
    optitrack_at,
    load_calibrations,
)


H5_ROOT  = Path("/media/yxma/Disk1/twm/data/motherboard")
OUT_ROOT = Path("/media/yxma/Disk1/twm/figures/episode_previews/motherboard")
CALIB_DIR = Path("/home/yxma/MultimodalData/twm/calibration/result")

PANEL_W, PANEL_H = 1280, 480
SOURCE_FPS       = 30.0       # recording frame rate
CLIP_S_DEFAULT   = 30.0       # seconds of usable data to sample (real time)
SPEED_DEFAULT    = 2.0        # playback speed (output_fps = SOURCE_FPS * speed)
EPISODES_ROOT    = Path("/media/yxma/Disk1/twm/processed/episodes/motherboard")


def _apply_world_offset(ot_lookup, dx, dy, dz):
    """Apply (dx, dy, dz) m offset to every OT sample (in-place mutation
    of the `poses` array inside the (ts, poses) tuple)."""
    if dx == 0 and dy == 0 and dz == 0:
        return
    for name, data in ot_lookup.items():
        if data is None:
            continue
        ts, poses = data
        if poses.ndim == 2 and poses.shape[1] >= 3:
            poses[:, 0] += dx
            poses[:, 1] += dy
            poses[:, 2] += dz


def _load_proj_calibs():
    cam_calib = [
        str(CALIB_DIR / "T_mocap_to_cam_middle.json"),
        str(CALIB_DIR / "T_mocap_to_cam_left.json"),
        str(CALIB_DIR / "T_mocap_to_cam_right.json"),
    ]
    gel_L = str(CALIB_DIR / "T_gel_to_rigid_left.json")
    gel_R = str(CALIB_DIR / "T_gel_to_rigid_right.json")
    try:
        cam_calibs, gel_center_left, gel_center_right = load_calibrations(
            cam_calib, gel_L, gel_R)
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
    except Exception as e:
        print(f"  WARN: calibration load failed ({e}); previews will lack projection overlay")
        return [], None, None


from twm.force_overlay import (draw_force_dot, draw_legend,
                               load_forces, row_for_h5_frame)

FORCE_ROOT = Path("/media/yxma/Disk1/twm/force_recovery")


def _parquet_trim_and_rows(task, date, ep):
    """Force rows are indexed by the RELEASE parquet, not by the .pt trim the
    preview uses elsewhere — see force_overlay for why mixing them shifts the
    dot ~15 frames."""
    import pyarrow.parquet as pq
    f = Path("/media/yxma/Disk1/twm/release")/task/"meta"/date/f"{ep}.parquet"
    if not f.exists():
        return None, 0
    t = pq.read_table(str(f), columns=["source_h5_frame"])
    import numpy as _np
    return int(_np.asarray(t["source_h5_frame"].to_numpy())[0]), t.num_rows


def _get_trim_offset(h5_path: Path) -> int:
    """Read trim_offset from the matching processed/episodes/.pt's _contact_meta.
    Falls back to 0 if the .pt isn't built yet."""
    date = h5_path.parent.name
    ep_stem = h5_path.stem
    pt_path = EPISODES_ROOT / date / f"{ep_stem}.pt"
    if not pt_path.exists():
        return 0
    import torch
    try:
        d = torch.load(str(pt_path), weights_only=False, map_location="cpu", mmap=True)
        return int(d.get("_contact_meta", {}).get("trim_offset", 0))
    except Exception:
        return 0


def build_one_preview(h5_path: Path, out_mp4: Path,
                      clip_s: float, speed: float,
                      project_cams, gel_center_left, gel_center_right,
                      dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> None:
    output_fps = SOURCE_FPS * speed
    n_frames_target = int(round(clip_s * SOURCE_FPS))   # e.g. 30s * 30fps = 900
    trim_offset = _get_trim_offset(h5_path)
    with h5py.File(str(h5_path), "r") as f:
        cam_ts = f["timestamps"][:]
        T_h5 = len(cam_ts)
        # First 30s of usable data, starting at trim_offset
        start = trim_offset
        end   = min(T_h5, trim_offset + n_frames_target)
        sample_idx = np.arange(start, end)

        # Pre-load OT once
        ot_lookup = load_optitrack(f)
        _apply_world_offset(ot_lookup, dx, dy, dz)

        # Use the first sampled frame's gelsight as the static diff reference
        ref_idx = int(sample_idx[0])
        gs_ref_L = f["gelsight/left/frames"][ref_idx]
        gs_ref_R = f["gelsight/right/frames"][ref_idx]

        task_name = h5_path.parent.parent.name
        date_name = h5_path.parent.name
        forces = load_forces(task_name, date_name, h5_path.stem, FORCE_ROOT)
        trim_pq, n_rows = _parquet_trim_and_rows(task_name, date_name,
                                                 h5_path.stem)

        panels = []
        for f_idx_int in sample_idx:
            f_idx_int = int(f_idx_int)
            color_frames = [
                f[f"realsense/cam{cam_idx}/color"][f_idx_int]
                for cam_idx in range(3)
            ]
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
                fps=30.0,
                task_name="motherboard",
                status_override=(
                    f"[motherboard] {h5_path.parent.name}/{h5_path.stem}  "
                    f"H5 frame {f_idx_int}/{T_h5}  "
                    f"({float(cam_ts[f_idx_int] - cam_ts[0]):.1f}s)"
                ),
            )

            if project_cams:
                try:
                    draw_projection_overlay(
                        panel, opt_poses,
                        project_cams,
                        gel_center_left, gel_center_right,
                    )
                except Exception:
                    pass

            if forces and trim_pq is not None:
                row = row_for_h5_frame(f_idx_int, trim_pq, n_rows)
                if row is not None:
                    for side, arr in forces.items():
                        if row < len(arr):
                            draw_force_dot(panel, side, float(arr[row]))
                    draw_legend(panel, 4 * 240 + 24, 240 + 110)

            panels.append(panel)

    # Write MP4 via ffmpeg (rawvideo BGR -> H.264 yuv444p)
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
    if ret != 0:
        raise RuntimeError(f"ffmpeg failed with code {ret}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--date", required=True,
                    help="Date subfolder under H5_ROOT, e.g. 2026-05-19")
    ap.add_argument("--clip_s", type=float, default=CLIP_S_DEFAULT,
                    help=f"Seconds of usable (post-trim) data to sample (default {CLIP_S_DEFAULT})")
    ap.add_argument("--speed", type=float, default=SPEED_DEFAULT,
                    help=f"Playback speed (default {SPEED_DEFAULT}x; output_fps = "
                         f"source_fps * speed)")
    ap.add_argument("--dx", type=float, default=0.0,
                    help="World-frame X offset (m) added to every OT pose before projection.")
    ap.add_argument("--dy", type=float, default=0.0,
                    help="World-frame Y offset (m) added to every OT pose before projection.")
    ap.add_argument("--dz", type=float, default=0.0,
                    help="World-frame Z offset (m) added to every OT pose before projection.")
    ap.add_argument("--episodes", nargs="*", default=None,
                    help="Optional list of episode_NNN stems to process. "
                         "Defaults to all episode_*.h5 under <date>/.")
    args = ap.parse_args()

    h5_dir = H5_ROOT / args.date
    if not h5_dir.is_dir():
        print(f"No such date: {h5_dir}", file=sys.stderr); sys.exit(1)
    h5_files = sorted(h5_dir.glob("episode_*.h5"))
    if args.episodes:
        wanted = set(args.episodes)
        h5_files = [p for p in h5_files if p.stem in wanted]
    if not h5_files:
        print(f"No episodes selected.", file=sys.stderr); sys.exit(1)

    print(f"Building previews for {len(h5_files)} episode(s) in {args.date} "
          f"(first {args.clip_s:.0f}s of post-trim data @ {args.speed:.1f}x speed -> "
          f"{args.clip_s / args.speed:.0f}s output)", flush=True)

    project_cams, glc, grc = _load_proj_calibs()
    if project_cams:
        print(f"  projection overlay: ON ({len(project_cams)} cameras)", flush=True)

    out_dir = OUT_ROOT / args.date
    for h5 in h5_files:
        out_mp4 = out_dir / f"{h5.stem}.mp4"
        print(f"  {h5.stem}: ", end="", flush=True)
        try:
            build_one_preview(h5, out_mp4, args.clip_s, args.speed,
                              project_cams, glc, grc,
                              dx=args.dx, dy=args.dy, dz=args.dz)
            print(f"OK  -> {out_mp4.relative_to(OUT_ROOT.parent.parent.parent)}  "
                  f"({out_mp4.stat().st_size / 1024:.0f} KB)", flush=True)
        except Exception as e:
            print(f"FAIL ({type(e).__name__}: {e})", flush=True)


if __name__ == "__main__":
    main()
