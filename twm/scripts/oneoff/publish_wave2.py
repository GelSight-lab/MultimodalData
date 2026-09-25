"""Wave-2 publish: multi-cam migration for the existing 27 episodes
(2026-05-10 + 2026-05-11).

Preconditions
-------------
  * `build_episodes_multicam.py` has finished, producing per-episode multi-cam
    .pt files under processed/episodes/motherboard/{2026-05-10,2026-05-11}/.
  * 2026-05-19 has already been published in Wave 1.

Pipeline
--------
  1. Sanity-check: 27 .pt files for the legacy dates exist under
     processed/episodes/motherboard/.
  2. Run build_segments.py over ALL dates (using the published bad_frames.json
     which already has 32-episode coverage from Wave 1). Wipes prior segments
     under processed/segments/ for the legacy dates.
  3. Build single HF commit:
        ADD:
          episodes/motherboard/{2026-05-10,2026-05-11}/*.pt  (27 files, multi-cam)
          segments/motherboard/{2026-05-10,2026-05-11}/*.pt  (~73 files, multi-cam)
          segments.json   (regenerated, all 78 entries multi-cam)
          README.md       (drop "migration in progress" note)
          docs/schema.md  (multi-cam is now default, not "from 2026-05-19 onward")
        DELETE:
          processed/mode1_v1/**       (legacy single-cam episodes)
          processed/mode2_v1/**       (legacy single-cam segments)
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationDelete

REPO_ROOT          = Path("/home/yxma/MultimodalData/twm")
EPISODES_ROOT      = Path("/media/yxma/Disk1/twm/processed/episodes/motherboard")
SEGMENTS_ROOT      = Path("/media/yxma/Disk1/twm/processed/segments/motherboard")
DATES_LEGACY       = ("2026-05-10", "2026-05-11")
DATES_ALL          = (*DATES_LEGACY, "2026-05-19")


def step1_sanity():
    print("=== STEP 1: sanity-check multicam outputs ===", flush=True)
    expected = {"2026-05-10": 12, "2026-05-11": 15}
    for d, n in expected.items():
        actual = len(list((EPISODES_ROOT / d).glob("episode_*.pt")))
        ok = actual == n
        print(f"  {d}: {actual}/{n} {'OK' if ok else 'MISSING'}", flush=True)
        if not ok:
            sys.exit(f"Aborting: {d} has {actual} but expected {n}")
    print("  all 27 multi-cam episodes present.", flush=True)


def step2_build_segments_all():
    print("\n=== STEP 2: build_segments.py over all 32 episodes ===", flush=True)
    # No --date filter, so we rebuild everything. Use the bad_frames.json
    # currently published on HF (already has 32-episode coverage from Wave 1).
    bf_local = Path("/tmp/hf_current_bad_frames.json")
    if not bf_local.exists():
        with urllib.request.urlopen(
            "https://huggingface.co/datasets/yxma/React/raw/main/bad_frames.json"
        ) as r:
            bf_local.write_bytes(r.read())
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts/build_segments.py"),
         "--bad_frames", str(bf_local),
         "--manifest", "/tmp/segments_full.json",
         "--workers", "4"],
        check=True)
    return Path("/tmp/segments_full.json")


def step3_finalize_segments_manifest(local_manifest: Path) -> Path:
    """Strip the `processed/segments/` prefix so paths in segments.json are
    relative to HF root (`segments/...`)."""
    print("\n=== STEP 3: rewrite segments.json paths for HF ===", flush=True)
    m = json.loads(local_manifest.read_text())
    for s in m["segments"]:
        s["path"] = s["path"].replace("processed/segments/", "segments/")
    out = Path("/tmp/segments_full_hf.json")
    out.write_text(json.dumps(m, indent=2))
    print(f"  {len(m['segments'])} segments, "
          f"{m['total_frames']:,} frames, "
          f"{m['total_duration_min']} min", flush=True)
    return out


def step4_update_readme() -> Path:
    print("\n=== STEP 4: finalize README.md ===", flush=True)
    # Re-fetch the README that Wave 1 just pushed
    url = "https://huggingface.co/datasets/yxma/React/raw/main/README.md"
    with urllib.request.urlopen(url) as r:
        text = r.read().decode()

    # Remove the "Multi-cam migration in progress" callout that Wave 1 added.
    # The pattern was inserted before `## Data quality`; find + strip the
    # blockquote that starts with "> **Multi-cam migration in progress."
    text = re.sub(
        r"\n> \*\*Multi-cam migration in progress\.\*\*[^\n]*(\n>[^\n]*)*\n",
        "\n",
        text,
    )

    # Update the "Two schemas" section to drop mode1_v1/mode2_v1 terminology
    old_section = re.search(
        r"## Two schemas: mode1_v1 vs mode2_v1.*?(?=\n## |\Z)",
        text, flags=re.DOTALL)
    if old_section:
        new_section = (
            "## Two layouts: `episodes/` vs `segments/`\n\n"
            "The same recordings are shipped two ways depending on what your code\n"
            "wants to do:\n\n"
            "- **`episodes/<task>/<date>/episode_*.pt`** — one file per recording.\n"
            "  Includes bad intervals (LED flicker, pose teleport, OT track loss)\n"
            "  inside; downstream code is expected to filter them out using\n"
            "  `bad_frames.json`. Each file carries all three RealSense views\n"
            "  (`view_left`, `view_middle`, `view_right`) plus both GelSights.\n"
            "- **`segments/<task>/<date>/episode_*.segment_*.pt`** — same\n"
            "  recordings, but **pre-sliced into contiguous clean segments at\n"
            "  every bad-frames boundary**. No `bad_frames.json` lookup needed;\n"
            "  the data is clean by construction. Index lookup via\n"
            "  [`segments.json`](segments.json). Each segment's\n"
            "  `_contact_meta.source_h5_frame_range` maps it back to the\n"
            "  original recording. The example `ReactSegmentDataset`\n"
            "  ([`examples/react_segment_dataset.py`](examples/react_segment_dataset.py))\n"
            "  consumes these.\n\n"
            "Both layouts have identical content (same source recordings, same\n"
            "frame data); only the file boundaries differ.\n"
        )
        text = text.replace(old_section.group(0), new_section)

    # Update HF config paths from processed/mode1_v1/ and processed/mode2_v1/
    text = text.replace("processed/mode1_v1/motherboard/**/episode_*.pt",
                        "episodes/motherboard/**/episode_*.pt")
    text = text.replace("processed/mode2_v1/motherboard/**/episode_*.segment_*.pt",
                        "segments/motherboard/**/episode_*.segment_*.pt")
    text = text.replace("processed/mode1_v1/**/episode_*.pt",
                        "episodes/**/episode_*.pt")
    # Casual textual references in body
    text = text.replace("processed/mode1_v1/motherboard", "episodes/motherboard")
    text = text.replace("processed/mode2_v1/motherboard", "segments/motherboard")
    text = text.replace("processed/mode1_v1/", "episodes/")
    text = text.replace("processed/mode2_v1/", "segments/")

    # Update the Repository-layout block
    text = re.sub(
        r"```\nREADME\.md.*?processed/mode1_v1/<task>/<date>/episode_\*\.pt.*?\n```",
        "```\n"
        "README.md                                        # this file\n"
        "tasks.json                                       # task / session registry\n"
        "bad_frames.json                                  # data-quality skip-list\n"
        "segments.json                                    # flat segments manifest\n"
        "episodes/<task>/<date>/episode_*.pt              # one .pt per recording (multi-cam)\n"
        "segments/<task>/<date>/episode_*.segment_*.pt    # pre-cut clean segments (multi-cam)\n"
        "figures/                                         # previews + analysis figures\n"
        "docs/                                            # extended documentation\n"
        "```",
        text, flags=re.DOTALL)

    out = Path("/tmp/README_wave2.md")
    out.write_text(text)
    return out


def step5_update_schema_doc() -> Path:
    print("\n=== STEP 5: finalize docs/schema.md ===", flush=True)
    url = "https://huggingface.co/datasets/yxma/React/raw/main/docs/schema.md"
    with urllib.request.urlopen(url) as r:
        text = r.read().decode()

    # Replace the "sessions from 2026-05-19 onward" caveat — multi-cam is now default
    text = text.replace(
        "## Multi-cam fields (sessions from 2026-05-19 onward)",
        "## Multi-cam fields (all sessions)")
    text = text.replace(
        "The 2026-05-10 and 2026-05-11 sessions will be re-published with this layout; "
        "until then, they still ship the single-cam `view` only.",
        "All published sessions ship this multi-cam layout.")
    # Mark the legacy `view` row as obsolete in the original table by patching
    # the first paragraph that referenced `processed/mode1_v1/`.
    text = text.replace(
        "Layout of each per-episode `.pt` file in the published "
        "`processed/mode1_v1/` slice.",
        "Layout of each per-episode `.pt` file in the published "
        "`episodes/` slice (and analogously each per-segment `.pt` in "
        "`segments/`).")
    # Drop the single-cam `view` row from the table — superseded by view_left/middle/right.
    text = re.sub(
        r"\| `view` \| `\(T, 3, 128, 128\)` \| uint8 \| .*?\n",
        "",
        text)
    out = Path("/tmp/schema_wave2.md")
    out.write_text(text)
    return out


def step6_commit(seg_manifest: Path, readme: Path, schema: Path):
    print("\n=== STEP 6: build HF commit ===", flush=True)
    api = HfApi()

    # Enumerate existing HF files we need to delete (legacy mode1_v1 / mode2_v1)
    repo_files = api.list_repo_files("yxma/React", repo_type="dataset")
    to_delete = [f for f in repo_files
                 if f.startswith("processed/mode1_v1/")
                 or f.startswith("processed/mode2_v1/")]
    print(f"  legacy files to delete: {len(to_delete)}", flush=True)

    ops: list = []
    # New per-episode multi-cam .pt files
    n_ep_uploaded = 0
    for d in DATES_LEGACY:
        for p in sorted((EPISODES_ROOT / d).glob("episode_*.pt")):
            ops.append(CommitOperationAdd(
                path_in_repo=f"episodes/motherboard/{d}/{p.name}",
                path_or_fileobj=str(p)))
            n_ep_uploaded += 1

    # New segments .pt for the legacy dates (overwrites Wave-1 mode2_v1 too,
    # since path is different — old ones explicitly deleted above)
    n_seg_uploaded = 0
    for d in DATES_LEGACY:
        for p in sorted((SEGMENTS_ROOT / d).glob("episode_*.pt")):
            ops.append(CommitOperationAdd(
                path_in_repo=f"segments/motherboard/{d}/{p.name}",
                path_or_fileobj=str(p)))
            n_seg_uploaded += 1

    # Updated metadata / docs
    ops.append(CommitOperationAdd(path_in_repo="segments.json",
                                  path_or_fileobj=str(seg_manifest)))
    ops.append(CommitOperationAdd(path_in_repo="README.md",
                                  path_or_fileobj=str(readme)))
    ops.append(CommitOperationAdd(path_in_repo="docs/schema.md",
                                  path_or_fileobj=str(schema)))

    # Deletions
    for f in to_delete:
        ops.append(CommitOperationDelete(path_in_repo=f))

    print(f"  episodes adds = {n_ep_uploaded}", flush=True)
    print(f"  segments adds = {n_seg_uploaded}", flush=True)
    print(f"  metadata adds = 3", flush=True)
    print(f"  deletions     = {len(to_delete)}", flush=True)
    print(f"  TOTAL ops     = {len(ops)}", flush=True)
    print("  committing (large upload, may take several minutes)...", flush=True)

    api.create_commit(
        repo_id="yxma/React", repo_type="dataset", operations=ops,
        commit_message=(
            "Wave 2: multi-cam migration complete for 2026-05-10 + 2026-05-11. "
            "All 32 episodes now ship under episodes/ and segments/ with "
            "view_left / view_middle / view_right + tactile_{left,right}. "
            "Legacy processed/mode1_v1/ and processed/mode2_v1/ removed."
        ),
    )
    print("  committed.", flush=True)


def main():
    step1_sanity()
    local_manifest = step2_build_segments_all()
    seg_manifest   = step3_finalize_segments_manifest(local_manifest)
    readme         = step4_update_readme()
    schema         = step5_update_schema_doc()
    step6_commit(seg_manifest, readme, schema)
    print("\n[publish_wave2] DONE.", flush=True)


if __name__ == "__main__":
    main()
