"""Build `figures/episode_previews/index.md` — a per-episode preview picker.

The page is intentionally short: each episode is a one-line summary inside
a collapsed `<details>` block; clicking the row expands to show that
episode's MP4 preview inline. HF renders this as a clickable navigation
page.

Run after pushing new preview MP4s so we know which episodes have
previews available on HF.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

REPO = "yxma/React"
TASK = "motherboard"
RAW_BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main"


def discover_previews_on_hf() -> dict[str, list[str]]:
    """Returns {date: [episode_stem, ...]} for every preview MP4 on HF."""
    from huggingface_hub import HfApi
    files = HfApi().list_repo_files(REPO, repo_type="dataset")
    out: dict[str, list[str]] = {}
    prefix = f"figures/episode_previews/{TASK}/"
    for f in files:
        if not f.startswith(prefix) or not f.endswith(".mp4"):
            continue
        parts = f[len(prefix):].split("/")
        if len(parts) != 2:
            continue
        date, fname = parts
        stem = fname.removesuffix(".mp4")
        out.setdefault(date, []).append(stem)
    for d in out:
        out[d].sort()
    return out


def fetch_segments_summary() -> dict[str, dict]:
    """Returns {ep_key: {n_frames, duration_s, n_segments}} from segments.json."""
    url = f"{RAW_BASE}/segments.json"
    with urllib.request.urlopen(url) as r:
        sj = json.loads(r.read())
    by_ep: dict[str, dict] = {}
    for s in sj.get("segments", []):
        ek = s["source_episode"]
        d = by_ep.setdefault(ek, {"n_frames": 0, "duration_s": 0.0, "n_segments": 0})
        d["n_frames"]   += s["n_frames"]
        d["duration_s"] += s["duration_s"]
        d["n_segments"] += 1
    return by_ep


def fetch_bad_summary() -> dict[str, dict]:
    """Returns {ep_key: bad_frames entry} from bad_frames.json."""
    url = f"{RAW_BASE}/bad_frames.json"
    with urllib.request.urlopen(url) as r:
        bf = json.loads(r.read())
    return bf.get("episodes", {})


def fetch_tasks_notes() -> dict[str, dict]:
    url = f"{RAW_BASE}/tasks.json"
    with urllib.request.urlopen(url) as r:
        tj = json.loads(r.read())
    return tj["tasks"][TASK].get("per_date_notes", {})


def render_episode_block(ep_key: str, mp4_url: str,
                         bad: dict, seg: dict) -> str:
    """One <details> block per episode."""
    duration = (seg.get("duration_s")
                or bad.get("duration_s")
                or 0.0)
    n_frames = (seg.get("n_frames")
                or bad.get("n_frames")
                or 0)
    n_segments = seg.get("n_segments", 0)
    bad_frames = bad.get("total_bad_frames", 0)
    bad_pct = bad.get("bad_fraction", 0.0) * 100.0

    # Compact summary on the visible row
    summary = (
        f"<b>{ep_key}</b>  &nbsp;&middot;&nbsp; "
        f"{n_frames:,} frames "
        f"({duration:.1f}s) "
        f"&nbsp;&middot;&nbsp; "
        f"{n_segments} segment{'s' if n_segments != 1 else ''}"
    )
    if bad_frames > 0:
        summary += f" &nbsp;&middot;&nbsp; {bad_pct:.2f}% flagged"

    # Raw <video controls> — HF's markdown renderer turns markdown ![](*.mp4)
    # into an <img> tag (broken), so we render the video element explicitly.
    body = (
        f"\n\n"
        f"<video controls preload=\"metadata\" width=\"100%\" "
        f"style=\"max-width:1280px\">\n"
        f"  <source src=\"{mp4_url}\" type=\"video/mp4\">\n"
        f"  Your browser cannot display this video. "
        f"<a href=\"{mp4_url}\">Download MP4</a>.\n"
        f"</video>\n\n"
        f"_Direct link: [`{mp4_url.split('/')[-1]}`]({mp4_url})_\n"
    )

    return (
        f"<details>\n"
        f"<summary>{summary}</summary>\n"
        f"{body}"
        f"</details>\n"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/episode_previews_index.md")
    args = ap.parse_args()

    previews = discover_previews_on_hf()
    # Exclude outdated dates (e.g. 2026-03-23 used different calibration; excluded from publication)
    EXCLUDED_DATES = {"2026-03-23", "2026-05-15"}
    previews = {d: eps for d, eps in previews.items() if d not in EXCLUDED_DATES}
    if not previews:
        print("No previews found on HF.", file=sys.stderr); sys.exit(1)

    bad      = fetch_bad_summary()
    seg      = fetch_segments_summary()
    notes    = fetch_tasks_notes()

    lines: list[str] = []
    lines.append("# Episode previews\n")
    lines.append(
        "One short flipbook video per recording, sampled at 60 frames per "
        "episode in the canonical viewer layout (3 RealSense cams + "
        "OptiTrack pose panel + 2 GelSight raw/diff + projection overlay).\n"
    )
    lines.append(
        "Each row below collapses by default to keep this page short. "
        "**Click any row to expand and preview that episode inline**, "
        "or follow the direct MP4 link.\n"
    )
    lines.append("---\n")

    total_eps = sum(len(v) for v in previews.values())
    lines.append(
        f"_Sessions with previews: {len(previews)} dates "
        f"&nbsp;&middot;&nbsp; "
        f"Total previews: {total_eps} episodes_\n"
    )

    for date in sorted(previews):
        note = notes.get(date, {})
        sensors = note.get("active_sensors", ["left", "right"])
        kind = note.get("kind", "session")
        lines.append(f"\n## {date}  &nbsp;&middot;&nbsp; "
                     f"{kind}  &nbsp;&middot;&nbsp; "
                     f"{', '.join(sensors)} sensor(s)\n")
        if note.get("note"):
            lines.append(f"> {note['note']}\n")
        for stem in previews[date]:
            ep_key = f"{date}/{stem}"
            mp4_url = (
                f"{RAW_BASE}/figures/episode_previews/{TASK}/{date}/{stem}.mp4"
            )
            ep_bad = bad.get(ep_key, {})
            ep_seg = seg.get(ep_key, {})
            lines.append(render_episode_block(ep_key, mp4_url, ep_bad, ep_seg))

    lines.append("\n---\n\n_Generated by `scripts/build_previews_index.py`._\n")
    out = "\n".join(lines)
    Path(args.out).write_text(out)
    print(f"Wrote {args.out} ({len(out)} bytes, {total_eps} episodes across "
          f"{len(previews)} dates)")


if __name__ == "__main__":
    main()
