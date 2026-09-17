#!/usr/bin/env python3
"""Build a read-only review inventory for ambiguous OptiTrack pose events.

The default command scans metadata and writes JSON/CSV under a separate review
directory.  It never rewrites a parquet or a released video.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from twm.react_preprocess.pose_anomaly import (  # noqa: E402
    detect_pose_events,
    transition_metrics,
)
from twm.viz import build_preview_panel, draw_projection_overlay  # noqa: E402


SIDES = ("left", "right")


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _event_id(date: str, episode: str, side: str, source_start: int,
              source_end: int, kind: str) -> str:
    return (f"{date}__{episode}__{side}__{source_start}-{source_end}__{kind}")


def scan_parquet(path: Path,
                 known_gaps: dict[str, list[tuple[int, int]]] | None = None,
                 chart_context: int = 60) -> list[dict[str, Any]]:
    """Return explainable event dictionaries for one release parquet."""
    path = Path(path)
    date, episode = path.parent.name, path.stem
    schema = pq.read_schema(path).names
    columns = ["source_h5_frame"] + [
        f"sensor_{side}_pose" for side in SIDES
        if f"sensor_{side}_pose" in schema
    ]
    table = pq.read_table(path, columns=columns)
    source = np.asarray(table["source_h5_frame"].to_numpy(), np.int64)
    gaps = known_gaps or {}
    out: list[dict[str, Any]] = []
    for side in SIDES:
        col = f"sensor_{side}_pose"
        if col not in table.column_names:
            continue
        pose = np.asarray(table[col].to_pylist(), float)
        rot, trans = transition_metrics(pose)
        for event in detect_pose_events(pose, side, gaps.get(side, ())):
            start = max(0, min(event.start, len(source) - 1))
            end = max(start, min(event.end, len(source) - 1))
            source_start, source_end = int(source[start]), int(source[end])
            lo = max(0, start - chart_context)
            hi = min(len(rot), end + chart_context + 1)
            item = event.to_dict()
            item.update({
                "date": date,
                "episode": episode,
                "row_start": start,
                "row_end": end,
                "source_start": source_start,
                "source_end": source_end,
                "source_bad_frames": [int(source[i]) for i in event.bad_frames
                                      if 0 <= i < len(source)],
                "event_id": _event_id(date, episode, side, source_start,
                                      source_end, event.kind),
                "chart": {
                    "transition_source_frames": source[lo:hi].astype(int).tolist(),
                    "rotation_deg": rot[lo:hi].astype(float).tolist(),
                    "translation_mm": trans[lo:hi].astype(float).tolist(),
                },
            })
            out.append(_jsonable(item))
    return sorted(out, key=lambda x: (x["date"], x["episode"],
                                      x["source_start"], x["side"], x["kind"]))


def scan_tree(task_root: Path) -> list[dict[str, Any]]:
    """Scan every parquet under ``task_root/meta/<date>``."""
    task_root = Path(task_root)
    gap_path = task_root / "pose_gaps.json"
    gap_data = json.loads(gap_path.read_text()) if gap_path.exists() else {}
    out: list[dict[str, Any]] = []
    for path in sorted((task_root / "meta").glob("*/*.parquet")):
        key = f"{path.parent.name}/{path.stem}"
        known = {
            side: [tuple(map(int, span)) for span in spans]
            for side, spans in gap_data.get(key, {}).items()
        }
        out.extend(scan_parquet(path, known))
    return out


def merge_review_events(events: list[dict[str, Any]],
                        max_gap: int = 15) -> list[dict[str, Any]]:
    """Merge nearby unresolved events so one tracking loss gets one clip."""
    unresolved = sorted((e for e in events if not e["repairable"]),
                        key=lambda x: (x["date"], x["episode"], x["side"],
                                       x["source_start"], x["source_end"]))
    groups: list[list[dict[str, Any]]] = []
    for event in unresolved:
        if (groups
                and all(event[k] == groups[-1][-1][k]
                        for k in ("date", "episode", "side"))
                and event["source_start"] - groups[-1][-1]["source_end"] <= max_gap):
            groups[-1].append(event)
        else:
            groups.append([event])
    out = []
    for group in groups:
        first, last = group[0], group[-1]
        merged = dict(first)
        merged["source_start"] = min(e["source_start"] for e in group)
        merged["source_end"] = max(e["source_end"] for e in group)
        if all("row_start" in e for e in group):
            merged["row_start"] = min(e["row_start"] for e in group)
            merged["row_end"] = max(e["row_end"] for e in group)
        merged["component_ids"] = [e["event_id"] for e in group]
        merged["component_kinds"] = [e["kind"] for e in group]
        samples: dict[int, tuple[float, float]] = {}
        for event in group:
            chart = event.get("chart", {})
            for frame, rotation, translation in zip(
                    chart.get("transition_source_frames", []),
                    chart.get("rotation_deg", []),
                    chart.get("translation_mm", [])):
                samples[int(frame)] = (float(rotation), float(translation))
        if samples:
            frames = sorted(samples)
            merged["chart"] = {
                "transition_source_frames": frames,
                "rotation_deg": [samples[f][0] for f in frames],
                "translation_mm": [samples[f][1] for f in frames],
            }
        if len(group) > 1:
            merged["event_id"] = _event_id(
                first["date"], first["episode"], first["side"],
                merged["source_start"], merged["source_end"], "review")
        out.append(merged)
    return out


def _open_streams(video_dir: Path) -> tuple[list[cv2.VideoCapture], list[str]]:
    names = ["view_right", "view_left", "view_middle",
             "tactile_left", "tactile_right"]
    wrist = [name for name in ("wrist_left", "wrist_right")
             if (video_dir / f"{name}.mp4").exists()]
    names += wrist
    captures = []
    for name in names:
        path = video_dir / f"{name}.mp4"
        if not path.exists():
            raise FileNotFoundError(path)
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video: {path}")
        captures.append(cap)
    return captures, names


def _decoded_frames(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    n = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        n += 1
    cap.release()
    return n


def _load_review_calibration(task: str, date: str):
    """Reuse the canonical preview's date-specific calibration routing."""
    try:
        from build_episode_previews import _load_proj_calibs
        cameras, gel_left, gel_right, _ = _load_proj_calibs(task, date)
        return cameras, gel_left, gel_right
    except Exception as exc:  # a review clip remains useful without projection
        print(f"  WARN: no projection calibration for {task}/{date}: {exc}")
        return [], None, None


def render_event_clip(event: dict[str, Any], task_root: Path, output: Path,
                      context_frames: int = 60, fps: float = 30.0) -> dict[str, Any]:
    """Render one row-aligned released-video window in the canonical panel."""
    task_root, output = Path(task_root), Path(output)
    date, episode = event["date"], event["episode"]
    parquet = task_root / "meta" / date / f"{episode}.parquet"
    video_dir = task_root / "videos" / date / episode
    table = pq.read_table(parquet, columns=[
        "source_h5_frame", "sensor_left_pose", "sensor_right_pose"])
    source = np.asarray(table["source_h5_frame"].to_numpy(), np.int64)
    poses = {side: np.asarray(table[f"sensor_{side}_pose"].to_pylist(), float)
             for side in SIDES}
    project_cams, gel_left, gel_right = _load_review_calibration(
        task_root.name, date)
    row_start = int(event.get("row_start", np.searchsorted(source, event["source_start"])))
    row_end = int(event.get("row_end", np.searchsorted(source, event["source_end"])))
    lo = max(0, row_start - int(context_frames))
    hi = min(len(source), row_end + int(context_frames) + 1)
    captures, names = _open_streams(video_dir)
    try:
        n_video = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in captures)
        hi = min(hi, n_video)
        if hi <= lo:
            raise ValueError(f"empty clip window {lo}:{hi} for {event['event_id']}")
        for cap in captures:
            cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
        first: dict[str, np.ndarray] = {}
        panels = []
        for row in range(lo, hi):
            frames = {}
            for cap, name in zip(captures, names):
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f"{name} ended at row {row}, expected {hi}")
                frames[name] = frame
                first.setdefault(name, frame.copy())
            opt = {f"sensor_{side}": (row / fps, poses[side][row]) for side in SIDES}
            wrists = [frames[name] for name in ("wrist_left", "wrist_right")
                      if name in frames]
            panel = build_preview_panel(
                color_frames=[frames["view_right"], frames["view_left"],
                              frames["view_middle"]],
                gs_frames=[frames["tactile_left"], frames["tactile_right"]],
                gs_ref=[first["tactile_left"], first["tactile_right"]],
                optitrack_poses=opt, recording=False, frame_count=int(source[row]),
                elapsed=row / fps, fps=fps,
                status_override=(
                    f"MOCAP REVIEW  {date}/{episode}  {event['side']}  "
                    f"{event['kind']}  source frame {int(source[row])}  "
                    f"confidence={float(event.get('confidence', 0)):.2f}"),
                arducam_frames=wrists or None,
                arducam_labels=[x.replace("_", " ") for x in
                                ("wrist_left", "wrist_right") if x in frames] or None,
            )
            if project_cams:
                try:
                    draw_projection_overlay(panel, opt, project_cams,
                                            gel_left, gel_right)
                except Exception as exc:
                    if row == lo:
                        print(f"  WARN: projection overlay failed for "
                              f"{event['event_id']}: {exc}")
            if row_start <= row <= row_end:
                cv2.rectangle(panel, (1, 1), (panel.shape[1] - 2, panel.shape[0] - 2),
                              (0, 0, 255), 4)
                cv2.putText(panel, "CANDIDATE EVENT", (8, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 255), 2,
                            cv2.LINE_AA)
            panels.append(panel)
    finally:
        for cap in captures:
            cap.release()

    output.parent.mkdir(parents=True, exist_ok=True)
    height, width = panels[0].shape[:2]
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
           "-r", str(float(fps)), "-i", "-", "-c:v", "libx264",
           "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", "-an", str(output)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    for panel in panels:
        proc.stdin.write(panel.tobytes())
    proc.stdin.close()
    code = proc.wait()
    if code != 0:
        raise RuntimeError(f"ffmpeg failed with code {code}: {output}")
    decoded = _decoded_frames(output)
    if decoded != len(panels):
        raise RuntimeError(f"clip decodes {decoded}/{len(panels)} frames: {output}")
    return {"path": str(output), "frames": decoded, "row_start": lo, "row_end": hi - 1}


def write_metadata(events: list[dict[str, Any]], output: Path,
                   review_events: list[dict[str, Any]] | None = None) -> dict[str, Path]:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "events.json"
    csv_path = output / "unresolved.csv"
    review = merge_review_events(events) if review_events is None else review_events
    json_path.write_text(json.dumps({
        "schema_version": 1,
        "events": events,
        "review_events": review,
    }, indent=2, sort_keys=True))
    fields = ["event_id", "date", "episode", "side", "kind",
              "source_start", "source_end", "confidence", "repairable"]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(review)
    return {"json": json_path, "csv": csv_path}


def write_html(events: list[dict[str, Any]], output: Path) -> Path:
    """Write a dependency-free review page with persistent local decisions."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cards = []
    for event in events:
        event_id = html.escape(str(event["event_id"]), quote=True)
        clip = html.escape(str(event.get("clip", "")), quote=True)
        title = (f"{event['date']}/{event['episode']} · {event['side']} · "
                 f"{event['source_start']}–{event['source_end']}")
        evidence = html.escape(json.dumps(event.get("evidence", {}), sort_keys=True))
        cards.append(f"""
<article class="event" id="event-{event_id}" data-event-id="{event_id}">
  <h2>{html.escape(title)}</h2>
  <p><span class="kind">{html.escape(str(event['kind']))}</span>
     confidence {float(event.get('confidence', 0)):.2f}</p>
  <video controls preload="metadata" src="{clip}"></video>
  <svg class="chart" viewBox="0 0 900 220" role="img"
       aria-label="rotation and translation chart"></svg>
  <details><summary>Detector evidence</summary><pre>{evidence}</pre></details>
  <div class="choices">
    <button data-decision="keep">Keep as real motion</button>
    <button data-decision="repair">Repair</button>
    <button data-decision="invalidate">Invalidate</button>
    <button data-decision="unsure">Unsure</button>
    <span class="chosen"></span>
  </div>
</article>""")
    payload = json.dumps(events, separators=(",", ":")).replace("</", "<\\/")
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mocap abnormal-event review</title>
<style>
body{{font:15px system-ui,sans-serif;margin:24px;background:#111;color:#eee}}
header{{position:sticky;top:0;background:#111e;padding:12px 0;z-index:3}}
.event{{max-width:960px;margin:24px auto;padding:18px;background:#1d1d1d;border:1px solid #444}}
video{{width:100%;background:#000}} .chart{{width:100%;height:220px;background:#171717}}
.kind{{color:#ffb000;font-weight:700}} button{{margin:8px 6px 0 0;padding:8px}}
button.active{{outline:3px solid #58a6ff}} pre{{white-space:pre-wrap}} .chosen{{margin-left:8px}}
</style></head><body>
<header><h1>Mocap unresolved events</h1>
<p>{len(events)} merged intervals. Decisions stay in this browser until exported.</p>
<button id="export">Export decisions.json</button></header>
{''.join(cards)}
<script id="event-data" type="application/json">{payload}</script>
<script>
const events=JSON.parse(document.getElementById('event-data').textContent);
const key='mocap-review-decisions-v1';
let decisions=JSON.parse(localStorage.getItem(key)||'{{}}');
function save(){{localStorage.setItem(key,JSON.stringify(decisions));}}
function setChoice(card,value){{
  card.querySelectorAll('[data-decision]').forEach(b=>b.classList.toggle('active',b.dataset.decision===value));
  card.querySelector('.chosen').textContent=value?('selected: '+value):'';
}}
document.querySelectorAll('.event').forEach(card=>{{
  setChoice(card,decisions[card.dataset.eventId]);
  card.querySelectorAll('[data-decision]').forEach(button=>button.onclick=()=>{{
    decisions[card.dataset.eventId]=button.dataset.decision;save();setChoice(card,button.dataset.decision);
  }});
}});
function points(values,max,w=900,h=180,y0=15){{
  if(!values||!values.length)return '';
  return values.map((v,i)=>`${{i*w/Math.max(1,values.length-1)}},${{y0+h-(Math.min(max,v)/max*h)}}`).join(' ');
}}
document.querySelectorAll('.chart').forEach((svg,i)=>{{
  const c=events[i].chart||{{}}, r=c.rotation_deg||[], t=c.translation_mm||[];
  svg.innerHTML=`<line x1="0" y1="177" x2="900" y2="177" stroke="#ffb000" stroke-dasharray="5 5"/>
  <polyline points="${{points(r,120)}}" fill="none" stroke="#ff5c5c" stroke-width="3"/>
  <polyline points="${{points(t,60)}}" fill="none" stroke="#58a6ff" stroke-width="3"/>
  <text x="8" y="208" fill="#ff5c5c">rotation ° (red; dashed = 30°)</text>
  <text x="650" y="208" fill="#58a6ff">translation mm (blue)</text>`;
}});
document.getElementById('export').onclick=()=>{{
  const rows=events.map(e=>({{event_id:e.event_id,decision:decisions[e.event_id]||'unreviewed'}}));
  const blob=new Blob([JSON.stringify({{schema_version:1,decisions:rows}},null,2)],{{type:'application/json'}});
  const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='decisions.json';a.click();
  URL.revokeObjectURL(a.href);
}};
</script></body></html>"""
    output.write_text(document)
    return output


def build_review(task_root: Path, output: Path, render: bool = False,
                 context_frames: int = 60) -> dict[str, Any]:
    """Scan, optionally render, and write one complete review package."""
    task_root, output = Path(task_root), Path(output)
    events = scan_tree(task_root)
    # A clip already includes `context_frames` on both sides.  Merge events
    # whose clip windows would overlap so the operator never reviews the same
    # frames twice under different filenames.
    review = merge_review_events(events, max_gap=2 * int(context_frames))
    if render:
        for i, event in enumerate(review, 1):
            filename = event["event_id"].replace("/", "_") + ".mp4"
            relative = Path("clips") / filename
            print(f"[{i}/{len(review)}] {event['event_id']}", flush=True)
            info = render_event_clip(event, task_root, output / relative,
                                     context_frames=context_frames)
            event["clip"] = relative.as_posix()
            event["clip_frames"] = info["frames"]
    paths = write_metadata(events, output, review)
    html_path = write_html(review, output / "index.html") if render else None
    return {"events": events, "review_events": review,
            "json": paths["json"], "csv": paths["csv"], "html": html_path}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path,
                    default=Path("/media/yxma/Disk1/twm/release/motherboard"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--render", action="store_true",
                    help="render review clips after scanning")
    ap.add_argument("--context-frames", type=int, default=60,
                    help="video frames before and after each event (default: 60)")
    args = ap.parse_args()
    result = build_review(args.root, args.output, args.render, args.context_frames)
    print(f"events={len(result['events'])} "
          f"unresolved_clips={len(result['review_events'])}")
    print(result["json"])
    if result["html"]:
        print(result["html"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
