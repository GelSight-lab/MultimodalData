"""Build `processed/segments/<task>/` — per-episode .pt files sliced into
contiguous clean segments at every bad-frames boundary. Parallel across
episodes.

Each output `.pt`:
  - same schema as the per-episode .pt under `processed/episodes/` (i.e.
    `view_left`, `view_middle`, `view_right`, `tactile_left`, `tactile_right`,
    `sensor_*_pose`, `timestamps`, `tactile_*_{intensity,area,mixed}`)
  - `_contact_meta` is the source episode's `_contact_meta` extended with
    segmentation metadata (`source_episode`, `source_segment_idx`,
    `source_pt_frame_range`, `source_h5_frame_range`).

Also writes `segments.json` at the configured manifest path with a flat
manifest indexed by segment.

Notes
-----
Compared to the old `build_mode2_segments.py`:
  * Reads from `processed/episodes/` (multi-cam, naming `view_{left,middle,right}`)
    rather than from `processed/mode1_v1/` (single-cam `view`).
  * Writes to `processed/segments/` instead of `processed/mode2_v1/`.
  * Schema string is `segments_v1`.
  * Slicing logic is field-agnostic: anything with leading dim == T is
    sliced, so adding/removing per-frame fields (e.g. depth) requires no
    code change here.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
from copy import deepcopy
from pathlib import Path

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

import numpy as np
import torch


PT_ROOT           = Path("/media/yxma/Disk1/twm/processed/episodes/motherboard")
OUT_ROOT          = Path("/media/yxma/Disk1/twm/processed/segments/motherboard")
BAD_FRAMES_PATH   = Path("/media/yxma/Disk1/twm/figures/dataset_figures/bad_frames.json")
SEGMENTS_MANIFEST = Path("/media/yxma/Disk1/twm/processed/segments/segments.json")

MIN_SEGMENT_FRAMES = 16          # drop fragments too short to host any useful window
TASK = "motherboard"
TACTILE_THRESHOLD = 0.4          # for contact_pct stat in segments.json
SCHEMA_NAME = "segments_v1"


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping / touching [a, b] (inclusive) intervals."""
    if not intervals:
        return []
    ivs = sorted(intervals)
    out = [list(ivs[0])]
    for a, b in ivs[1:]:
        if a <= out[-1][1] + 1:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def find_clean_segments(T: int, bad_intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Given T total frames and the union of bad [a, b] intervals,
    return the complement — list of inclusive [a, b] for clean spans."""
    merged = merge_intervals(bad_intervals)
    segments: list[tuple[int, int]] = []
    prev_end = -1
    for a, b in merged:
        if a > prev_end + 1:
            segments.append((prev_end + 1, a - 1))
        prev_end = max(prev_end, b)
    if prev_end < T - 1:
        segments.append((prev_end + 1, T - 1))
    return segments


def _episode_T(ep: dict) -> int:
    """Determine T from any (T, ...) per-frame tensor, preferring `timestamps`
    so we don't have to page in giant view tensors under mmap."""
    if "timestamps" in ep and hasattr(ep["timestamps"], "shape"):
        return int(ep["timestamps"].shape[0])
    for k, v in ep.items():
        if hasattr(v, "shape") and len(v.shape) >= 1:
            return int(v.shape[0])
    raise RuntimeError("could not determine T from episode dict")


def process_episode(args):
    """Worker: load one .pt, slice into clean segments, save each to segments/."""
    pt_path_str, ep_key, bad_lookup, trim_offsets_lookup = args
    pt_path = Path(pt_path_str)
    date, ep_stem = ep_key.split("/")

    ep = torch.load(str(pt_path), weights_only=False, map_location="cpu")
    T = _episode_T(ep)

    bad_ep = bad_lookup.get(ep_key, {})
    bad_intervals = (
        bad_ep.get("intensity_spikes", [])
        + bad_ep.get("pose_teleports_L", [])
        + bad_ep.get("pose_teleports_R", [])
        + bad_ep.get("ot_loss_L", [])
        + bad_ep.get("ot_loss_R", [])
    )
    bad_intervals = [(int(a), int(b)) for a, b in bad_intervals]
    segments = find_clean_segments(T, bad_intervals)

    src_trim_offset = int(trim_offsets_lookup.get(ep_key, 0))
    base_meta = ep.get("_contact_meta", {})
    out_dir = OUT_ROOT / date
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    seg_idx_next = 0
    for s_a, s_b in segments:
        L = s_b - s_a + 1
        if L < MIN_SEGMENT_FRAMES:
            continue
        seg_meta = dict(base_meta)
        seg_meta["source_episode"]        = ep_key
        seg_meta["source_segment_idx"]    = seg_idx_next
        seg_meta["source_pt_frame_range"] = [s_a, s_b]
        seg_meta["source_h5_frame_range"] = [s_a + src_trim_offset, s_b + src_trim_offset]
        seg_meta["trim_offset"]           = s_a + src_trim_offset
        seg_meta["pre_trim_n_frames"]     = T + src_trim_offset

        # Slice every per-frame tensor (field-agnostic).
        new_ep: dict = {}
        for k, v in ep.items():
            if hasattr(v, "shape") and len(v.shape) >= 1 and v.shape[0] == T:
                new_ep[k] = v[s_a:s_b + 1].clone()
            else:
                new_ep[k] = v
        new_ep["_contact_meta"] = seg_meta

        out_pt = out_dir / f"{ep_stem}.segment_{seg_idx_next:02d}.pt"
        tmp_pt = out_pt.with_suffix(".pt.tmp")
        torch.save(new_ep, str(tmp_pt))
        os.replace(tmp_pt, out_pt)

        # Stats for segments.json
        mL = new_ep["tactile_left_mixed"].numpy()
        mR = new_ep["tactile_right_mixed"].numpy()
        contact_L = float((mL > TACTILE_THRESHOLD).mean() * 100)
        contact_R = float((mR > TACTILE_THRESHOLD).mean() * 100)
        pL = new_ep["sensor_left_pose"][:, :3].numpy()
        pR = new_ep["sensor_right_pose"][:, :3].numpy()
        speed_L_mps = float(np.linalg.norm(np.diff(pL, axis=0), axis=1).mean() * 30) if L > 1 else 0.0
        speed_R_mps = float(np.linalg.norm(np.diff(pR, axis=0), axis=1).mean() * 30) if L > 1 else 0.0

        written.append({
            "path":                   str(out_pt.relative_to(OUT_ROOT.parent.parent)),
            "source_episode":         ep_key,
            "source_segment_idx":     seg_idx_next,
            "n_frames":               L,
            "duration_s":             round(L / 30.0, 3),
            "source_pt_frame_range":  [s_a, s_b],
            "source_h5_frame_range":  [s_a + src_trim_offset, s_b + src_trim_offset],
            "contact_pct_left":       round(contact_L, 2),
            "contact_pct_right":      round(contact_R, 2),
            "mean_speed_left_mps":    round(speed_L_mps, 4),
            "mean_speed_right_mps":   round(speed_R_mps, 4),
            "file_size_mb":           round(out_pt.stat().st_size / 1024 / 1024, 2),
        })
        seg_idx_next += 1

    n_dropped = len(segments) - len(written)
    return ep_key, written, T, len(segments), n_dropped


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None,
                    help="Optional: process only this date subfolder (e.g. 2026-05-19). "
                         "If omitted, processes all dates under PT_ROOT.")
    ap.add_argument("--bad_frames", default=str(BAD_FRAMES_PATH),
                    help="Path to bad_frames.json (default: published copy on disk).")
    ap.add_argument("--manifest", default=str(SEGMENTS_MANIFEST))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no_wipe", action="store_true",
                    help="Do not wipe existing segments before rebuilding.")
    args = ap.parse_args()

    bad = json.loads(Path(args.bad_frames).read_text())
    bad_lookup = bad.get("episodes", {})
    trim_offsets_lookup = bad.get("trim_offsets", {})

    pt_files = sorted(PT_ROOT.rglob("episode_*.pt"))
    work = []
    for pt in pt_files:
        date = pt.parent.name
        if date == "2026-03-23":     # excluded
            continue
        if args.date and date != args.date:
            continue
        ep_key = f"{date}/{pt.stem}"
        work.append((str(pt), ep_key, bad_lookup, trim_offsets_lookup))
    if not work:
        print(f"No episodes selected (date filter: {args.date!r})."); sys.exit(1)
    print(f"Processing {len(work)} episodes with {args.workers} workers"
          + (f" (date={args.date})" if args.date else ""), flush=True)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    # Wipe prior segments (only for the target date if --date given).
    if not args.no_wipe:
        for old in OUT_ROOT.rglob("*.pt"):
            if args.date and old.parent.name != args.date:
                continue
            old.unlink()

    all_segments = []
    with mp.Pool(processes=args.workers) as pool:
        for ep_key, written, T, n_total_segs, n_dropped in pool.imap_unordered(process_episode, work):
            print(f"  {ep_key}: T={T:>6d}  segments_kept={len(written):>2d}  "
                  f"(of {n_total_segs} clean spans; {n_dropped} dropped as too short)",
                  flush=True)
            all_segments.extend(written)

    all_segments.sort(key=lambda s: (s["source_episode"], s["source_segment_idx"]))

    # Per-episode active-sensors lookup from HF tasks.json
    try:
        import urllib.request
        t = json.loads(urllib.request.urlopen(
            "https://huggingface.co/datasets/yxma/React/raw/main/tasks.json").read())
        per_date_active = {
            d: notes.get("active_sensors", ["left", "right"])
            for d, notes in t["tasks"][TASK]["per_date_notes"].items()
        }
    except Exception:
        per_date_active = {}
    for s in all_segments:
        d = s["source_episode"].split("/")[0]
        s["active_sensors"] = per_date_active.get(d, ["left", "right"])

    total_frames = sum(s["n_frames"] for s in all_segments)
    total_min = total_frames / 30.0 / 60.0
    src_episodes = sorted({s["source_episode"] for s in all_segments})
    seg_lens = sorted(s["n_frames"] for s in all_segments)
    manifest = {
        "schema": SCHEMA_NAME,
        "description": (
            "Each .pt is a contiguous clean segment of a source recording. "
            "Bad intervals from bad_frames.json (intensity_spikes, "
            "pose_teleports_*, ot_loss_*) are excluded by construction, so "
            "a dataloader can iterate windows here without any per-window "
            "quality filter. Use `source_h5_frame_range` to map back to the "
            "original H5 recording. Each segment includes three RealSense "
            "views (`view_left`, `view_middle`, `view_right`) and two "
            "GelSight tactile streams (`tactile_left`, `tactile_right`)."
        ),
        "n_source_episodes":      len(src_episodes),
        "n_segments":             len(all_segments),
        "total_frames":           total_frames,
        "total_duration_min":     round(total_min, 2),
        "median_segment_frames":  int(seg_lens[len(seg_lens) // 2]) if seg_lens else 0,
        "median_segment_s":       round(seg_lens[len(seg_lens) // 2] / 30.0, 3) if seg_lens else 0,
        "min_segment_frames_kept":           MIN_SEGMENT_FRAMES,
        "tactile_threshold_used_for_contact_pct": TACTILE_THRESHOLD,
        "segments":               all_segments,
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {len(all_segments)} segments  →  {total_frames:,} frames "
          f"({total_min:.1f} min). Manifest at {manifest_path}.", flush=True)


if __name__ == "__main__":
    main()
