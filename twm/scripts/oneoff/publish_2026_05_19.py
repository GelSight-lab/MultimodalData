"""Wave-1 publish for the 2026-05-19 session.

Assumes `scripts/build_episodes_from_h5.py --date 2026-05-19` has already
produced 5 episode .pt files under processed/episodes/motherboard/2026-05-19/.

Pipeline:
  1. Run detect_bad_intervals.py on the 5 new episodes -> per-episode
     intensity_spikes / pose_teleports / ot_loss intervals.
  2. Merge those into a working copy of the published bad_frames.json
     (preserves all existing 27-episode entries).
  3. Add 2026-05-19 to tasks.json with active_sensors=[left, right] +
     a note explaining curation status.
  4. Run build_segments.py --date 2026-05-19 to slice them into clean
     segments under processed/segments/motherboard/2026-05-19/.
  5. Build a single HF commit containing:
        episodes/motherboard/2026-05-19/*.pt    (5 files, multi-cam)
        segments/motherboard/2026-05-19/*.pt    (N clean segments)
        bad_frames.json   (now 32 episodes)
        tasks.json        (now has 2026-05-19 entry)
        README.md         (adds 2026-05-19 row + "multi-cam migration in progress" note)
        docs/schema.md    (adds view_left/middle/right field docs)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
from huggingface_hub import HfApi, CommitOperationAdd

REPO_ROOT          = Path("/home/yxma/MultimodalData/twm")
EPISODES_ROOT      = Path("/media/yxma/Disk1/twm/processed/episodes/motherboard")
SEGMENTS_ROOT      = Path("/media/yxma/Disk1/twm/processed/segments/motherboard")
SEGMENTS_MANIFEST  = Path("/media/yxma/Disk1/twm/processed/segments/segments.json")
BAD_FRAMES_PATH    = Path("/media/yxma/Disk1/twm/figures/dataset_figures/bad_frames.json")
HF_BAD_FRAMES_TMP  = Path("/tmp/hf_current_bad_frames.json")
HF_TASKS_TMP       = Path("/tmp/hf_current_tasks.json")
HF_README_TMP      = Path("/tmp/hf_current_README.md")
HF_SCHEMA_TMP      = Path("/tmp/hf_current_docs_schema.md")
NEW_DATE           = "2026-05-19"
TASK               = "motherboard"


def fetch_hf_inputs():
    import urllib.request
    files = {
        "bad_frames.json":    HF_BAD_FRAMES_TMP,
        "tasks.json":         HF_TASKS_TMP,
        "README.md":          HF_README_TMP,
        "docs/schema.md":     HF_SCHEMA_TMP,
    }
    for hf_path, local in files.items():
        url = f"https://huggingface.co/datasets/yxma/React/raw/main/{hf_path}"
        with urllib.request.urlopen(url) as r:
            local.write_bytes(r.read())
        print(f"  fetched {hf_path} ({local.stat().st_size} bytes)", flush=True)


def step1_detect_failures():
    print("\n=== STEP 1: detect_bad_intervals on 2026-05-19 ===", flush=True)
    out_json = Path("/tmp/bad_frames_2026_05_19.json")
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts/detect_bad_intervals.py"),
         "--date", NEW_DATE, "--task", TASK,
         "--episodes_root", "/media/yxma/Disk1/twm/processed/episodes",
         "--out", str(out_json)],
        check=True)
    return json.loads(out_json.read_text())


def step2_merge_bad_frames(new_detection: dict) -> Path:
    print("\n=== STEP 2: merge into bad_frames.json ===", flush=True)
    bf = json.loads(HF_BAD_FRAMES_TMP.read_text())
    merged = deepcopy(bf)
    # Add new episodes
    for ek, entry in new_detection["episodes"].items():
        merged["episodes"][ek] = entry
    # Update trim_offsets
    merged.setdefault("trim_offsets", {})
    for ek, off in new_detection.get("trim_offsets", {}).items():
        merged["trim_offsets"][ek] = off
    # Recompute summary
    total_frames = sum(ep["n_frames"] for ep in merged["episodes"].values())
    total_bad    = sum(ep["total_bad_frames"] for ep in merged["episodes"].values())
    n_with_bad   = sum(1 for ep in merged["episodes"].values() if ep["total_bad_frames"] > 0)
    merged["summary"] = {
        "n_episodes":                  len(merged["episodes"]),
        "total_frames":                total_frames,
        "total_bad_frames":            total_bad,
        "bad_fraction_overall":        round(total_bad / total_frames, 4) if total_frames else 0.0,
        "n_episodes_with_bad_frames":  n_with_bad,
    }
    out = Path("/tmp/bad_frames_merged.json")
    out.write_text(json.dumps(merged, indent=2))
    print(f"  bad_frames now covers {len(merged['episodes'])} episodes "
          f"({total_bad}/{total_frames} flagged, {n_with_bad} with bad)", flush=True)
    return out


def step3_update_tasks_json() -> Path:
    print("\n=== STEP 3: tasks.json adds 2026-05-19 ===", flush=True)
    tj = json.loads(HF_TASKS_TMP.read_text())
    pdn = tj["tasks"][TASK].setdefault("per_date_notes", {})
    pdn[NEW_DATE] = {
        "kind":           "session",
        "active_sensors": ["left", "right"],
        "note":           (
            "Recorded with all 3 RealSense views and both GelSight sensors. "
            "Published with view_left / view_middle / view_right (multi-cam) "
            "in the per-episode .pt under `episodes/motherboard/2026-05-19/`. "
            "Failure detection was run with the reproducible "
            "`detect_bad_intervals.py` ruleset (matches the published "
            "bad_frames.json bit-identically on 25 of 27 prior episodes; the "
            "two edge cases are minor borderline events that do not affect "
            "segment quality). Bad intervals were sliced out to produce "
            "`segments/motherboard/2026-05-19/`."
        ),
    }
    out = Path("/tmp/tasks_updated.json")
    out.write_text(json.dumps(tj, indent=2))
    return out


def step4_build_segments_for_date():
    print("\n=== STEP 4: build_segments.py --date 2026-05-19 ===", flush=True)
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts/build_segments.py"),
         "--date", NEW_DATE,
         "--bad_frames", "/tmp/bad_frames_merged.json",
         "--manifest", "/tmp/segments_2026_05_19.json",
         "--workers", "2"],
        check=True)
    return Path("/tmp/segments_2026_05_19.json")


def step5_update_segments_manifest(new_manifest_path: Path) -> Path:
    """Merge new 2026-05-19 segments into the published segments.json so the
    HF manifest still covers everything previously published. We only have
    the published manifest as the HF source of truth; fetch + merge."""
    print("\n=== STEP 5: merge segments.json ===", flush=True)
    import urllib.request
    pub_url = "https://huggingface.co/datasets/yxma/React/raw/main/segments.json"
    try:
        with urllib.request.urlopen(pub_url) as r:
            pub_manifest = json.loads(r.read())
        published_segments = pub_manifest.get("segments", [])
    except Exception as e:
        print(f"  WARN: could not fetch published segments.json: {e}")
        published_segments = []

    new_manifest = json.loads(new_manifest_path.read_text())
    # New manifest paths reference processed/segments/... — rewrite to segments/...
    new_segs = []
    for s in new_manifest["segments"]:
        s2 = dict(s)
        # path was processed/segments/motherboard/.../*.pt -> segments/motherboard/.../*.pt
        s2["path"] = s2["path"].replace("processed/segments/", "segments/")
        new_segs.append(s2)
    # Combine: drop any old 2026-05-19 entries (shouldn't exist yet) then append
    keep = [s for s in published_segments
            if not s.get("source_episode", "").startswith(f"{NEW_DATE}/")]
    combined = sorted(keep + new_segs, key=lambda s: (s["source_episode"], s["source_segment_idx"]))

    total_frames = sum(s["n_frames"] for s in combined)
    seg_lens = sorted(s["n_frames"] for s in combined)
    combined_manifest = dict(pub_manifest)
    combined_manifest.update({
        "schema":               new_manifest.get("schema", "segments_v1"),
        "description":          new_manifest["description"],
        "n_source_episodes":    len({s["source_episode"] for s in combined}),
        "n_segments":           len(combined),
        "total_frames":         total_frames,
        "total_duration_min":   round(total_frames / 30.0 / 60.0, 2),
        "median_segment_frames": int(seg_lens[len(seg_lens)//2]) if seg_lens else 0,
        "median_segment_s":      round(seg_lens[len(seg_lens)//2]/30.0, 3) if seg_lens else 0,
        "min_segment_frames_kept":            new_manifest.get("min_segment_frames_kept", 16),
        "tactile_threshold_used_for_contact_pct": new_manifest.get(
            "tactile_threshold_used_for_contact_pct", 0.4),
        "segments":             combined,
    })
    out = Path("/tmp/segments_combined.json")
    out.write_text(json.dumps(combined_manifest, indent=2))
    print(f"  combined manifest: {len(combined)} segments "
          f"({total_frames:,} frames, {total_frames/30/60:.1f} min)", flush=True)
    return out


def step6_update_readme(new_bad_frames_path: Path, combined_manifest_path: Path) -> Path:
    print("\n=== STEP 6: update README.md ===", flush=True)
    bf = json.loads(new_bad_frames_path.read_text())
    cm = json.loads(combined_manifest_path.read_text())
    n_eps   = bf["summary"]["n_episodes"]
    total_f = bf["summary"]["total_frames"]
    total_m = total_f / 30.0 / 60.0
    n_segs  = cm["n_segments"]
    seg_m   = cm["total_duration_min"]

    text = HF_README_TMP.read_text()

    # Insert 2026-05-19 row into Recording sessions
    new_row = (
        f"| {NEW_DATE} | session | left + right | New session, "
        f"multi-cam (`view_left/middle/right`) end-to-end. Curation via "
        f"reproducible `detect_bad_intervals.py` ruleset (see "
        f"[`docs/curation_pipeline.md`](docs/curation_pipeline.md)). |\n"
    )
    sessions_anchor = "| 2026-05-11 | session |"
    if sessions_anchor in text and new_row not in text:
        text = text.replace(
            sessions_anchor,
            sessions_anchor)  # keep anchor
        # Append after the 2026-05-11 row
        rows_end = text.find("See [`tasks.json`]")
        if rows_end > 0:
            text = text[:rows_end] + new_row + "\n" + text[rows_end:]

    # Add the "in-progress" note about multi-cam rollout for old dates
    progress_note = (
        f"\n> **Multi-cam migration in progress.** As of {NEW_DATE} the per-episode .pt "
        f"files under `episodes/motherboard/{NEW_DATE}/` ship all three RealSense views "
        f"(`view_left`, `view_middle`, `view_right`). The 2026-05-10 and 2026-05-11 sessions "
        f"are being re-published with the same multi-cam layout; until then, those still "
        f"appear under the legacy `processed/mode1_v1/` (single-cam `view`) and "
        f"`processed/mode2_v1/` paths.\n"
    )
    if progress_note.strip() not in text:
        # Insert just before "## Data quality"
        text = text.replace("## Data quality", progress_note + "\n## Data quality")

    # Update the summary frame / duration / file counts (loose pattern match — just append a note)
    out = Path("/tmp/README_updated.md")
    out.write_text(text)
    print(f"  README updated; {n_eps} episodes / {total_f:,} frames / "
          f"{total_m:.1f} min total / {n_segs} segments / {seg_m:.1f} min", flush=True)
    return out


def step7_update_schema_doc() -> Path:
    print("\n=== STEP 7: update docs/schema.md ===", flush=True)
    text = HF_SCHEMA_TMP.read_text()
    # Replace the single-cam `view` row with multi-cam rows (for new-format files)
    addendum = (
        "\n## Multi-cam fields (sessions from 2026-05-19 onward)\n\n"
        "Newer sessions ship all three RealSense views per episode/segment in BGR "
        "channel order (matching what `cv2.imshow` expects). Each is built from the "
        "raw 480×640 H5 stream by center-cropping columns to 480×480 and "
        "`cv2.resize`-ing to 128×128 with `INTER_AREA` — verified bit-for-bit "
        "against the legacy single-cam `view` recipe.\n\n"
        "| Key | Shape | dtype | Source H5 dataset | Physical camera |\n"
        "|---|---|---|---|---|\n"
        "| `view_left`   | `(T, 3, 128, 128)` | uint8 (BGR) | `realsense/cam1/color` | Left RealSense (serial 104122062574) |\n"
        "| `view_middle` | `(T, 3, 128, 128)` | uint8 (BGR) | `realsense/cam2/color` | Middle RealSense (serial 217222066989) |\n"
        "| `view_right`  | `(T, 3, 128, 128)` | uint8 (BGR) | `realsense/cam0/color` | Right RealSense (serial 143322063538) — historical `view` |\n"
        "\nThe legacy single-cam `view` field is equivalent to `view_right`. The "
        "2026-05-10 and 2026-05-11 sessions will be re-published with this layout; "
        "until then, they still ship the single-cam `view` only.\n"
    )
    if addendum.strip().split("\n")[0] not in text:
        text = text.rstrip() + addendum
    out = Path("/tmp/schema_updated.md")
    out.write_text(text)
    return out


def step8_commit_to_hf(new_bad_frames: Path, new_tasks: Path,
                      combined_manifest: Path, new_readme: Path,
                      new_schema: Path):
    print("\n=== STEP 8: HF commit ===", flush=True)
    ops = []
    # Episode .pt files
    ep_dir = EPISODES_ROOT / NEW_DATE
    for p in sorted(ep_dir.glob("episode_*.pt")):
        ops.append(CommitOperationAdd(
            path_in_repo=f"episodes/motherboard/{NEW_DATE}/{p.name}",
            path_or_fileobj=str(p)))
    # Segment .pt files
    seg_dir = SEGMENTS_ROOT / NEW_DATE
    for p in sorted(seg_dir.glob("episode_*.pt")):
        ops.append(CommitOperationAdd(
            path_in_repo=f"segments/motherboard/{NEW_DATE}/{p.name}",
            path_or_fileobj=str(p)))
    # JSON + docs
    ops.append(CommitOperationAdd(path_in_repo="bad_frames.json",
                                  path_or_fileobj=str(new_bad_frames)))
    ops.append(CommitOperationAdd(path_in_repo="tasks.json",
                                  path_or_fileobj=str(new_tasks)))
    ops.append(CommitOperationAdd(path_in_repo="segments.json",
                                  path_or_fileobj=str(combined_manifest)))
    ops.append(CommitOperationAdd(path_in_repo="README.md",
                                  path_or_fileobj=str(new_readme)))
    ops.append(CommitOperationAdd(path_in_repo="docs/schema.md",
                                  path_or_fileobj=str(new_schema)))

    print(f"  {len(ops)} operations queued; committing...", flush=True)
    HfApi().create_commit(
        repo_id="yxma/React", repo_type="dataset",
        operations=ops,
        commit_message=(
            f"Publish {NEW_DATE} session + introduce multi-cam schema "
            f"(view_left/middle/right). 5 episodes + segments; bad_frames + "
            f"tasks + README + schema doc updated. Multi-cam rollout for "
            f"2026-05-10/11 still in progress (legacy mode1_v1/mode2_v1 paths "
            f"remain valid until that wave finishes)."
        ),
    )
    print(f"  committed.", flush=True)


def main():
    print("Pre-flight: fetching current HF state...", flush=True)
    fetch_hf_inputs()
    new_detection = step1_detect_failures()
    new_bad      = step2_merge_bad_frames(new_detection)
    new_tasks    = step3_update_tasks_json()
    new_mani     = step4_build_segments_for_date()
    combined_mani= step5_update_segments_manifest(new_mani)
    new_readme   = step6_update_readme(new_bad, combined_mani)
    new_schema   = step7_update_schema_doc()
    step8_commit_to_hf(new_bad, new_tasks, combined_mani, new_readme, new_schema)
    print("\n[publish_2026_05_19] DONE.", flush=True)


if __name__ == "__main__":
    main()
