"""One-shot: apply a (dx, dy, dz) world-frame offset to all 5 episodes of
2026-05-19 and re-publish.

Steps
-----
  1. Add (dx, dy, dz) m to `sensor_{left,right}_pose[:, :3]` in each of the
     5 episode .pt files under processed/episodes/motherboard/2026-05-19/.
     Record `_contact_meta.world_frame_offset_applied = [dx, dy, dz]`.
  2. Re-run build_segments.py --date 2026-05-19 to slice with the new poses.
  3. Re-run build_episode_previews.py --date 2026-05-19 --dx X --dz Z
     so the projection overlay reflects the same offset (the preview reads
     poses from H5, not .pt, so we apply the offset live in the preview
     pipeline as well).
  4. Patch tasks.json (per_date_notes[2026-05-19].world_frame_offset_applied)
  5. Refresh the previews index page.
  6. Commit everything to HF in a single operation.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

import torch

REPO_ROOT          = Path("/home/yxma/MultimodalData/twm")
EPISODES_ROOT      = Path("/media/yxma/Disk1/twm/processed/episodes/motherboard")
SEGMENTS_ROOT      = Path("/media/yxma/Disk1/twm/processed/segments/motherboard")
PREVIEWS_ROOT      = Path("/media/yxma/Disk1/twm/figures/episode_previews/motherboard")
NEW_DATE           = "2026-05-19"


def step1_bake_offset(dx: float, dy: float, dz: float):
    """Idempotent: reads the offset already applied (per `_contact_meta.
    world_frame_offset_applied`, default [0,0,0]) and applies only the delta
    so iterative tuning doesn't double-shift."""
    print(f"\n=== STEP 1: bake target offset ({dx:.3f}, {dy:.3f}, {dz:.3f}) into 5 episodes ===", flush=True)
    target = (float(dx), float(dy), float(dz))
    for p in sorted((EPISODES_ROOT / NEW_DATE).glob("episode_*.pt")):
        ep = torch.load(str(p), weights_only=False, map_location="cpu")
        cm = ep["_contact_meta"]
        cur = cm.get("world_frame_offset_applied", [0.0, 0.0, 0.0])
        cur = [float(c) for c in cur]
        delta = (target[0] - cur[0], target[1] - cur[1], target[2] - cur[2])
        for side in ("left", "right"):
            pose = ep[f"sensor_{side}_pose"]
            pose[:, 0] += delta[0]
            pose[:, 1] += delta[1]
            pose[:, 2] += delta[2]
        cm["world_frame_offset_applied"] = list(target)
        cm["world_frame_offset_note"] = (
            "On 2026-05-19 the OptiTrack world origin was redefined relative to "
            "earlier sessions; this offset has been added to sensor_{left,right}_pose "
            "translation (x, y, z columns) so that all dates share one world frame."
        )
        tmp = p.with_suffix(".pt.tmp")
        torch.save(ep, str(tmp))
        os.replace(tmp, p)
        print(f"  {p.name}: current={cur}  target={list(target)}  "
              f"delta={list(delta)}  -> baked", flush=True)


def step2_rebuild_segments():
    print("\n=== STEP 2: rebuild segments for 2026-05-19 ===", flush=True)
    # Fetch current bad_frames.json from HF (already has 32-episode coverage)
    bf_local = Path("/tmp/hf_current_bad_frames.json")
    with urllib.request.urlopen(
        "https://huggingface.co/datasets/yxma/React/raw/main/bad_frames.json"
    ) as r:
        bf_local.write_bytes(r.read())
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts/build_segments.py"),
         "--date", NEW_DATE,
         "--bad_frames", str(bf_local),
         "--manifest", "/tmp/segments_2026_05_19.json",
         "--workers", "2"],
        check=True)
    return Path("/tmp/segments_2026_05_19.json")


def step3_regen_previews(dx: float, dy: float, dz: float):
    print("\n=== STEP 3: regenerate 2026-05-19 previews with offset ===", flush=True)
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts/build_episode_previews.py"),
         "--date", NEW_DATE,
         "--dx", str(dx), "--dy", str(dy), "--dz", str(dz)],
        check=True)


def step4_patch_tasks(dx: float, dy: float, dz: float) -> Path:
    print("\n=== STEP 4: patch tasks.json ===", flush=True)
    with urllib.request.urlopen(
        "https://huggingface.co/datasets/yxma/React/raw/main/tasks.json"
    ) as r:
        tj = json.loads(r.read())
    entry = tj["tasks"]["motherboard"]["per_date_notes"][NEW_DATE]
    entry["world_frame_offset_applied"] = [float(dx), float(dy), float(dz)]
    entry["note"] = (
        "Recorded with all 3 RealSense views and both GelSight sensors. "
        f"OptiTrack world origin was redefined on {NEW_DATE} relative to "
        f"earlier sessions; an offset of (dx={dx:.3f}, dy={dy:.3f}, dz={dz:.3f}) m has been "
        "added to sensor_{left,right}_pose translation columns of every .pt "
        "(and to OT samples during preview rendering) so all 32 episodes "
        "share one world frame. See `_contact_meta.world_frame_offset_applied` "
        "in each 2026-05-19 .pt for the per-file record."
    )
    out = Path("/tmp/tasks_offset_patched.json")
    out.write_text(json.dumps(tj, indent=2))
    return out


def step5_refresh_index() -> Path:
    print("\n=== STEP 5: refresh previews index ===", flush=True)
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts/build_previews_index.py"),
         "--out", "/tmp/episode_previews_index.md"],
        check=True)
    return Path("/tmp/episode_previews_index.md")


def step6_merge_segments_manifest(date_manifest: Path) -> Path:
    """Replace 2026-05-19 segments in the published segments.json."""
    print("\n=== STEP 6: merge segments.json ===", flush=True)
    with urllib.request.urlopen(
        "https://huggingface.co/datasets/yxma/React/raw/main/segments.json"
    ) as r:
        pub = json.loads(r.read())
    new = json.loads(date_manifest.read_text())
    keep = [s for s in pub["segments"]
            if not s.get("source_episode", "").startswith(f"{NEW_DATE}/")]
    new_segs = []
    for s in new["segments"]:
        s2 = dict(s); s2["path"] = s2["path"].replace("processed/segments/", "segments/")
        new_segs.append(s2)
    combined = sorted(keep + new_segs,
                      key=lambda s: (s["source_episode"], s["source_segment_idx"]))
    total_frames = sum(s["n_frames"] for s in combined)
    pub.update({
        "n_source_episodes":    len({s["source_episode"] for s in combined}),
        "n_segments":           len(combined),
        "total_frames":         total_frames,
        "total_duration_min":   round(total_frames / 30.0 / 60.0, 2),
        "segments":             combined,
    })
    out = Path("/tmp/segments_merged.json")
    out.write_text(json.dumps(pub, indent=2))
    print(f"  combined manifest: {len(combined)} segments, "
          f"{total_frames:,} frames, "
          f"{pub['total_duration_min']} min", flush=True)
    return out


def step7_commit(tasks_path: Path, index_path: Path, segments_manifest: Path,
                 dx: float = 0.0, dy: float = 0.0, dz: float = 0.0):
    print("\n=== STEP 7: HF commit ===", flush=True)
    from huggingface_hub import HfApi, CommitOperationAdd
    ops = []
    # 5 updated episode .pt
    for p in sorted((EPISODES_ROOT / NEW_DATE).glob("episode_*.pt")):
        ops.append(CommitOperationAdd(
            path_in_repo=f"episodes/motherboard/{NEW_DATE}/{p.name}",
            path_or_fileobj=str(p)))
    # 5 updated segment .pt
    for p in sorted((SEGMENTS_ROOT / NEW_DATE).glob("episode_*.pt")):
        ops.append(CommitOperationAdd(
            path_in_repo=f"segments/motherboard/{NEW_DATE}/{p.name}",
            path_or_fileobj=str(p)))
    # 5 updated preview MP4
    for p in sorted((PREVIEWS_ROOT / NEW_DATE).glob("episode_*.mp4")):
        ops.append(CommitOperationAdd(
            path_in_repo=f"figures/episode_previews/motherboard/{NEW_DATE}/{p.name}",
            path_or_fileobj=str(p)))
    # Metadata
    ops.append(CommitOperationAdd(path_in_repo="tasks.json",
                                  path_or_fileobj=str(tasks_path)))
    ops.append(CommitOperationAdd(path_in_repo="segments.json",
                                  path_or_fileobj=str(segments_manifest)))
    ops.append(CommitOperationAdd(path_in_repo="figures/episode_previews/index.md",
                                  path_or_fileobj=str(index_path)))
    print(f"  {len(ops)} ops queued; pushing...", flush=True)
    HfApi().create_commit(
        repo_id="yxma/React", repo_type="dataset", operations=ops,
        commit_message=(
            f"Apply world-frame offset (dx, dy, dz) = "
            f"({dx:.3f}, {dy:.3f}, {dz:.3f}) m to all 5 episodes of {NEW_DATE} so "
            f"all 32 episodes share one OptiTrack world frame. Per-episode "
            f"_contact_meta.world_frame_offset_applied documents the bake. "
            f"Segments, previews, tasks.json, segments.json refreshed."
        ),
    )
    print("  committed.", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", type=float, required=True)
    ap.add_argument("--dy", type=float, required=True)
    ap.add_argument("--dz", type=float, required=True)
    args = ap.parse_args()

    step1_bake_offset(args.dx, args.dy, args.dz)
    seg_mani  = step2_rebuild_segments()
    step3_regen_previews(args.dx, args.dy, args.dz)
    tasks_p   = step4_patch_tasks(args.dx, args.dy, args.dz)
    index_p   = step5_refresh_index()
    combined  = step6_merge_segments_manifest(seg_mani)
    step7_commit(tasks_p, index_p, combined, args.dx, args.dy, args.dz)
    print("\n[apply_world_offset_2026_05_19] DONE.", flush=True)


if __name__ == "__main__":
    main()
