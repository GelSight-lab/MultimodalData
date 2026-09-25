"""Merge the index files of the three groups now living in `data/validation`.

`episodes.jsonl`, `segments.json` and `bad_frames.json` were each written for a
single task by a single run. The folder now holds two tasks and three sessions,
so each index gains the field that distinguishes them -- `task` on every
episodes row, `task` on every segment (it was a single value at the top), and a
`source_recording` naming the file the episode came from.

That last one is not decoration. The recorder reuses episode numbers within a
date: `motherboard/2026-09-09/episode_000.h5` was written four times that day,
and the copy on disk is not the one published as `validation/episode_000`.
Renumbering is what keeps them apart; `source_recording` is what still says
which recording each number means once they are apart.

`pushT_2026-09-09/segments.json` shipped `n_segments: 79` over an EMPTY list,
while its own `episodes.jsonl` said 62. Recomputing the clean spans from
`bad_frames.json` at the documented floor of 16 frames reproduces 62 and the
published `total_bad_frames` of 1147 exactly, so the list is rebuilt here
rather than carried across with its header.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

FPS = 30.0                # the convention every published index uses (n_frames/duration_s)
DATE = "2026-09-09"
OLD_FLOOR = 16        # the floor the 04:00 session and pushT were cut at
CUT_FLOOR = 900       # 30 s -- the floor the 17:20/17:41 segments were cut at


def clean_spans(entry: dict, floor: int) -> list[tuple[int, int]]:
    """Contiguous runs of frames no detector flagged, at least `floor` long."""
    n = entry["n_frames"]
    bad = set()
    for v in entry.values():
        if isinstance(v, list):
            for a, b in v:
                bad.update(range(int(a), int(b) + 1))
    spans, start = [], None
    for i in range(n):
        if i in bad:
            if start is not None:
                spans.append((start, i - 1)); start = None
        elif start is None:
            start = i
    if start is not None:
        spans.append((start, n - 1))
    return [s for s in spans if s[1] - s[0] + 1 >= floor]


def merge(existing: dict, arrivals: list[dict], pushT_bad: dict) -> dict:
    """`existing` holds the three published index files; `arrivals` the manifest."""
    # ── episodes.jsonl ──────────────────────────────────────────────────────
    rows = []
    for line in existing["episodes.jsonl"].splitlines():
        if line.strip():
            r = json.loads(line)
            r.setdefault("task", existing["segments.json"]["task"])
            r.setdefault("source_recording", f"{r['task']}/{r['episode']}")
            rows.append(r)

    segments = []
    for s in existing["segments.json"]["segments"]:
        s.setdefault("task", existing["segments.json"]["task"])
        s.setdefault("min_frames_policy", OLD_FLOOR)
        segments.append(s)

    bad = dict(existing["bad_frames.json"]["episodes"])

    for a in arrivals:
        ep, n = a["new_episode"], a["n_frames"]
        key = f"{DATE}/{ep}"
        is_cut = a["src"] == "cut"
        floor = CUT_FLOOR if is_cut else OLD_FLOOR
        if is_cut:
            # A cut segment IS a clean span: `segment` published it because no
            # detector flagged any frame in it. Its own defect record is empty,
            # and it indexes as one segment covering the whole file.
            b = {"n_frames": n, "duration_s": round(n / FPS, 3),
                 **{k: [] for k, v in next(iter(bad.values())).items()
                    if isinstance(v, list)},
                 "total_bad_frames": 0, "bad_fraction": 0.0}
            spans = [(0, n - 1)]
        else:
            b = dict(pushT_bad)
            spans = clean_spans(b, floor)
        bad[key] = b
        for i, (lo, hi) in enumerate(spans):
            segments.append({
                "task": a["task"], "source_episode": key, "segment_idx": i,
                "frame_range": [lo, hi], "n_frames": hi - lo + 1,
                "duration_s": round((hi - lo + 1) / FPS, 3),
                "min_frames_policy": floor,
            })
        rows.append({
            "episode": key, "date": DATE, "task": a["task"],
            "n_frames": n, "duration_s": a["duration_s"],
            "active_sensors": ["left", "right"],
            "trim_offset": a.get("source_h5_range", [0])[0],
            "world_frame_offset": [0.0, 0.0, 0.0],
            "n_segments": len(spans),
            "total_bad_frames": b["total_bad_frames"],
            "up_axis": "z",
            "source_recording": a["source_recording"],
        })

    seg_out = dict(existing["segments.json"])
    seg_out.pop("task", None)          # no longer one task
    seg_out.update({
        "tasks": sorted({s["task"] for s in segments}),
        "n_segments": len(segments),
        "total_frames": sum(s["n_frames"] for s in segments),
        "total_duration_min": round(sum(s["n_frames"] for s in segments) / FPS / 60, 2),
        "min_segment_frames_kept": {str(OLD_FLOOR): "2026-09-09 04:00 session and pushT",
                                    str(CUT_FLOOR): "2026-09-09 17:20/17:41 session (cut release)"},
        "segments": segments,
    })
    bad_out = dict(existing["bad_frames.json"])
    bad_out.pop("task", None)
    bad_out["tasks"] = seg_out["tasks"]
    bad_out["episodes"] = bad
    bad_out["summary"] = {
        "n_episodes": len(bad),
        "total_bad_frames": sum(v["total_bad_frames"] for v in bad.values()),
    }
    return {"episodes.jsonl": "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
            "segments.json": json.dumps(seg_out, indent=2, ensure_ascii=False) + "\n",
            "bad_frames.json": json.dumps(bad_out, indent=2, ensure_ascii=False) + "\n"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/media/yxma/Disk1/twm/publish/validation")
    a = ap.parse_args()
    from huggingface_hub import hf_hub_download
    out = Path(a.out)
    manifest = json.loads((out / "_manifest.json").read_text())
    existing = {}
    for f in ("episodes.jsonl", "segments.json", "bad_frames.json"):
        txt = Path(hf_hub_download("yxma/React", f"data/validation/{f}",
                                   repo_type="dataset")).read_text()
        existing[f] = txt if f.endswith(".jsonl") else json.loads(txt)
    pushT_bad = json.loads(Path(hf_hub_download(
        "yxma/React", "data/pushT_2026-09-09/bad_frames.json",
        repo_type="dataset")).read_text())["episodes"][f"{DATE}/episode_000"]

    for name, text in merge(existing, manifest, pushT_bad).items():
        (out / name).write_text(text)
    seg = json.loads((out / "segments.json").read_text())
    print(f"  episodes.jsonl  {len((out/'episodes.jsonl').read_text().strip().splitlines())} 行")
    print(f"  segments.json   {seg['n_segments']} 段  {seg['total_duration_min']} 分钟  tasks={seg['tasks']}")
    print(f"  bad_frames.json {json.loads((out/'bad_frames.json').read_text())['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
