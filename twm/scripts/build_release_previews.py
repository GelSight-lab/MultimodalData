"""Generate canonical-layout preview MP4s for the task-first release.

Reuses build_episode_previews.build_one_preview (the exact "same as before"
1280x480 panel: 3 cams + OptiTrack + GelSight raw/diff + projection overlay,
first 30s @ 2x). Per task it swaps in:
  - the correct calibration dir (motherboard=May-12, pushT=June-26)
  - the correct H5 root
  - output under data/<task>/previews/<date>/episode_NNN.mp4
  - trim offset read from the release `_detect.pt` sidecar
  - per-(task,date) world-frame offset (motherboard/2026-05-19)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_episode_previews as BEP

CALIB = Path("/home/yxma/MultimodalData/twm/calibration")
STAGE = Path("/media/yxma/Disk1/twm/release")

TASK_CFG = {
    "motherboard": {
        "h5_root": Path("/media/yxma/Disk1/twm/data/motherboard"),
        "calib_dir": CALIB / "result backup",       # May-12
    },
    "pushT": {
        "h5_root": Path("/media/yxma/Disk1/twm/data/pushT"),
        "calib_dir": CALIB / "result",              # June-26
    },
}
WORLD_OFFSET = {("motherboard", "2026-05-19"): (0.23, 0.0, 0.175)}


def _release_trim(task, date, ep_stem):
    import torch
    det = STAGE / task / "meta" / date / f"{ep_stem}._detect.pt"
    if not det.exists():
        return 0
    d = torch.load(str(det), weights_only=False, map_location="cpu")
    return int(d["_contact_meta"].get("trim_offset", 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASK_CFG))
    ap.add_argument("--clip_s", type=float, default=30.0)
    ap.add_argument("--speed", type=float, default=2.0)
    args = ap.parse_args()
    cfg = TASK_CFG[args.task]

    # Point the reused module at this task's calibration + roots.
    BEP.CALIB_DIR = cfg["calib_dir"]
    BEP.H5_ROOT = cfg["h5_root"]
    BEP.OUT_ROOT = STAGE / args.task / "previews"
    # Override trim source to use the release sidecar.
    BEP._get_trim_offset = lambda h5_path: _release_trim(
        args.task, Path(h5_path).parent.name, Path(h5_path).stem)

    project_cams, glc, grc = BEP._load_proj_calibs()
    print(f"[previews] {args.task}: calib={cfg['calib_dir'].name}  "
          f"cams={len(project_cams)}", flush=True)

    # Only episodes that actually have a release video (skip corrupt H5).
    vids_root = STAGE / args.task / "videos"
    for date_dir in sorted(vids_root.iterdir()):
        if not date_dir.is_dir():
            continue
        date = date_dir.name
        off = WORLD_OFFSET.get((args.task, date), (0.0, 0.0, 0.0))
        out_dir = BEP.OUT_ROOT / date
        for ep_dir in sorted(date_dir.iterdir()):
            ep_stem = ep_dir.name
            h5 = cfg["h5_root"] / date / f"{ep_stem}.h5"
            if not h5.exists():
                print(f"  {date}/{ep_stem}: no H5, skip", flush=True); continue
            out_mp4 = out_dir / f"{ep_stem}.mp4"
            try:
                BEP.build_one_preview(h5, out_mp4, args.clip_s, args.speed,
                                      project_cams, glc, grc,
                                      dx=off[0], dy=off[1], dz=off[2])
                print(f"  {date}/{ep_stem}: OK ({out_mp4.stat().st_size/1024:.0f} KB)",
                      flush=True)
            except Exception as e:
                print(f"  {date}/{ep_stem}: FAIL ({type(e).__name__}: {e})", flush=True)
    print("[previews] done", flush=True)


if __name__ == "__main__":
    main()
