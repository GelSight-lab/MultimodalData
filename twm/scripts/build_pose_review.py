#!/usr/bin/env python3
"""Build a read-only review inventory for ambiguous OptiTrack pose events.

The default command scans metadata and writes JSON/CSV under a separate review
directory.  It never rewrites a parquet or a released video.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import cv2
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from twm.react_preprocess.pose_anomaly import (  # noqa: E402
    detect_pose_events,
    transition_metrics,
)
from twm.viz import build_preview_panel, draw_projection_overlay  # noqa: E402


SIDES = ("left", "right")


def _confidence_text(value: Any) -> str:
    if isinstance(value, str):
        return value.upper()
    return f"{float(value):.2f}"


def _pose_residual(raw: np.ndarray,
                   candidate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    trans = np.linalg.norm(candidate[:, :3] - raw[:, :3], axis=1) * 1000.0
    rot = np.degrees((Rotation.from_quat(candidate[:, 3:])
                      * Rotation.from_quat(raw[:, 3:]).inv()).magnitude())
    return trans, rot


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


def load_candidate_events(task_root: Path, candidate_task_root: Path,
                          chart_context: int = 60,
                          fps: float = 30.0) -> list[dict[str, Any]]:
    """Load candidate sidecars and attach raw/candidate motion evidence."""
    task_root = Path(task_root)
    candidate_task_root = Path(candidate_task_root)
    out: list[dict[str, Any]] = []
    for event_path in sorted(
            (candidate_task_root / "repair_events").glob("*/*.json")):
        payload = json.loads(event_path.read_text())
        date, episode = str(payload["date"]), str(payload["episode"])
        raw_path = task_root / "meta" / date / f"{episode}.parquet"
        candidate_path = (candidate_task_root / "meta" / date
                          / f"{episode}.parquet")
        if not raw_path.is_file() or not candidate_path.is_file():
            raise FileNotFoundError(
                f"candidate event has no raw/candidate parquet: {event_path}")
        raw_table = pq.read_table(raw_path)
        candidate_table = pq.read_table(candidate_path)
        source = np.asarray(raw_table["source_h5_frame"].to_numpy(), np.int64)
        for event in payload.get("events", []):
            side = str(event["side"])
            raw = np.asarray(raw_table[f"sensor_{side}_pose"].to_pylist(), float)
            candidate = np.asarray(
                candidate_table[f"sensor_{side}_pose"].to_pylist(), float)
            start, end = int(event["start"]), int(event["end"])
            lo = max(0, start - int(chart_context))
            hi_row = min(len(raw), end + int(chart_context) + 1)
            hi_transition = max(lo, min(len(raw) - 1, hi_row - 1))
            raw_rot, raw_trans = transition_metrics(raw)
            candidate_rot, candidate_trans = transition_metrics(candidate)
            trans_residual, rot_residual = _pose_residual(raw, candidate)
            candidate_velocity = candidate_trans * float(fps)
            candidate_angular_velocity = candidate_rot * float(fps)
            if len(candidate_velocity):
                linear_accel = np.diff(
                    candidate_velocity, prepend=candidate_velocity[0]) * float(fps)
                angular_accel = np.diff(
                    candidate_angular_velocity,
                    prepend=candidate_angular_velocity[0]) * float(fps)
            else:
                linear_accel = angular_accel = np.empty(0)
            repaired_name = f"pose_{side}_repaired"
            repaired = (np.asarray(candidate_table[repaired_name], bool)
                        if repaired_name in candidate_table.column_names
                        else np.zeros(len(raw), bool))
            chart_rows = np.arange(lo, hi_row)
            evidence = event.get("evidence", {})
            left_anchor = event.get(
                "left_context_end", evidence.get("left_anchor", start - 1))
            right_anchor = event.get(
                "right_context_start", evidence.get("right_anchor", end + 1))
            anchor_mask = np.isin(chart_rows, [left_anchor, right_anchor])
            retained_mask = ((chart_rows >= start) & (chart_rows <= end)
                             & ~repaired[lo:hi_row])
            item = dict(event)
            item.update({
                "date": date,
                "episode": episode,
                "row_start": start,
                "row_end": end,
                "source_start": int(source[start]),
                "source_end": int(source[end]),
                "manifest_digest": payload.get("manifest_digest", ""),
                "candidate_parquet": str(candidate_path),
                "repairable": str(event.get("confidence", "")).upper() == "HIGH",
                "chart": {
                    "transition_source_frames": source[lo:hi_transition].astype(int).tolist(),
                    "rotation_deg": raw_rot[lo:hi_transition].astype(float).tolist(),
                    "translation_mm": raw_trans[lo:hi_transition].astype(float).tolist(),
                    "rotation_raw_deg": raw_rot[lo:hi_transition].astype(float).tolist(),
                    "rotation_candidate_deg": candidate_rot[lo:hi_transition].astype(float).tolist(),
                    "translation_raw_mm": raw_trans[lo:hi_transition].astype(float).tolist(),
                    "translation_candidate_mm": candidate_trans[lo:hi_transition].astype(float).tolist(),
                    "linear_velocity_candidate_mm_s": candidate_velocity[lo:hi_transition].astype(float).tolist(),
                    "angular_velocity_candidate_deg_s": candidate_angular_velocity[lo:hi_transition].astype(float).tolist(),
                    "linear_acceleration_candidate_mm_s2": linear_accel[lo:hi_transition].astype(float).tolist(),
                    "angular_acceleration_candidate_deg_s2": angular_accel[lo:hi_transition].astype(float).tolist(),
                    "pose_translation_residual_mm": trans_residual[lo:hi_row].astype(float).tolist(),
                    "pose_rotation_residual_deg": rot_residual[lo:hi_row].astype(float).tolist(),
                    "anchor_rows": anchor_mask.astype(bool).tolist(),
                    "retained_rows": retained_mask.astype(bool).tolist(),
                    "replaced_rows": repaired[lo:hi_row].astype(bool).tolist(),
                },
            })
            out.append(_jsonable(item))
    return sorted(out, key=lambda event: (
        event["date"], event["episode"], event["row_start"], event["side"]))


def select_review_events(events: list[dict[str, Any]],
                         high_sample_rate: float = 0.1,
                         seed: int = 0) -> list[dict[str, Any]]:
    """Keep all MEDIUM/LOW events and a stable sample of HIGH events."""
    if not 0.0 <= float(high_sample_rate) <= 1.0:
        raise ValueError("high_sample_rate must be between 0 and 1")
    selected = []
    for event in sorted(events, key=lambda row: str(row["event_id"])):
        confidence = str(event.get("confidence", "")).upper()
        if confidence in {"MEDIUM", "LOW"}:
            selected.append(event)
            continue
        if confidence == "HIGH":
            token = f"{int(seed)}:{event['event_id']}".encode("utf-8")
            fraction = int.from_bytes(hashlib.sha256(token).digest()[:8], "big") / 2**64
            if fraction < float(high_sample_rate):
                selected.append(event)
            continue
        # Backward compatibility for the original review inventory.
        if not bool(event.get("repairable", False)):
            selected.append(event)
    return selected


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
        from twm.scripts.build_episode_previews import _load_proj_calibs
        cameras, gel_left, gel_right, _ = _load_proj_calibs(task, date)
        return cameras, gel_left, gel_right
    except Exception as exc:  # a review clip remains useful without projection
        print(f"  WARN: no projection calibration for {task}/{date}: {exc}")
        return [], None, None


def render_event_clip(event: dict[str, Any], task_root: Path, output: Path,
                      context_frames: int = 60, fps: float = 30.0,
                      candidate_parquet: Path | None = None) -> dict[str, Any]:
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
    candidate_poses = poses
    if candidate_parquet is not None:
        candidate_table = pq.read_table(Path(candidate_parquet), columns=[
            "sensor_left_pose", "sensor_right_pose"])
        if candidate_table.num_rows != table.num_rows:
            raise ValueError(
                f"candidate rows {candidate_table.num_rows} != raw rows "
                f"{table.num_rows}: {candidate_parquet}")
        candidate_poses = {
            side: np.asarray(
                candidate_table[f"sensor_{side}_pose"].to_pylist(), float)
            for side in SIDES
        }
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
            opt = {f"sensor_{side}": (row / fps, candidate_poses[side][row])
                   for side in SIDES}
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
                    f"confidence={_confidence_text(event.get('confidence', 0))}"),
                arducam_frames=wrists or None,
                arducam_labels=[x.replace("_", " ") for x in
                                ("wrist_left", "wrist_right") if x in frames] or None,
            )
            if project_cams:
                try:
                    if candidate_parquet is None:
                        draw_projection_overlay(panel, opt, project_cams,
                                                gel_left, gel_right)
                    else:
                        side = str(event["side"])
                        raw_opt = {
                            f"sensor_{side}": (row / fps, poses[side][row])}
                        candidate_opt = {
                            f"sensor_{side}": (
                                row / fps, candidate_poses[side][row])}
                        draw_projection_overlay(
                            panel, raw_opt, project_cams, gel_left, gel_right,
                            frozen_side=side)
                        draw_projection_overlay(
                            panel, candidate_opt, project_cams,
                            gel_left, gel_right)
                        trans_residual, rot_residual = _pose_residual(
                            poses[side][row:row + 1],
                            candidate_poses[side][row:row + 1])
                        cv2.putText(
                            panel,
                            f"RAW(red ring) -> CANDIDATE  residual "
                            f"{trans_residual[0]:.1f} mm / "
                            f"{rot_residual[0]:.1f} deg",
                            (8, 44), cv2.FONT_HERSHEY_SIMPLEX, .52,
                            (30, 220, 255), 2, cv2.LINE_AA)
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
        confidence = html.escape(_confidence_text(event.get("confidence", 0)))
        cards.append(f"""
<article class="event" id="event-{event_id}" data-event-id="{event_id}">
  <h2>{html.escape(title)}</h2>
  <p><span class="kind">{html.escape(str(event['kind']))}</span>
     confidence {confidence}</p>
  <video controls preload="metadata" src="{clip}"></video>
  <svg class="chart" viewBox="0 0 900 220" role="img"
       aria-label="rotation and translation chart"></svg>
  <details><summary>Detector evidence</summary><pre>{evidence}</pre></details>
  <div class="choices">
    <button data-decision="keep_raw">Keep raw as real motion</button>
    <button data-decision="accept_repair">Accept repair</button>
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
const manifestDigest=(events[0]&&events[0].manifest_digest)||'';
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
  const c=events[i].chart||{{}}, rr=c.rotation_raw_deg||c.rotation_deg||[],
        rc=c.rotation_candidate_deg||rr, tr=c.translation_raw_mm||c.translation_mm||[],
        tc=c.translation_candidate_mm||tr, pr=c.pose_rotation_residual_deg||[],
        pt=c.pose_translation_residual_mm||[];
  svg.innerHTML=`<line x1="0" y1="177" x2="900" y2="177" stroke="#ffb000" stroke-dasharray="5 5"/>
  <polyline points="${{points(rr,120)}}" fill="none" stroke="#ff5c5c" stroke-width="2"/>
  <polyline points="${{points(rc,120)}}" fill="none" stroke="#60d394" stroke-width="2"/>
  <polyline points="${{points(tr,60)}}" fill="none" stroke="#58a6ff" stroke-width="2"/>
  <polyline points="${{points(tc,60)}}" fill="none" stroke="#f4d35e" stroke-width="2"/>
  <polyline points="${{points(pr,120)}}" fill="none" stroke="#d77aff" stroke-width="1"/>
  <polyline points="${{points(pt,60)}}" fill="none" stroke="#ffffff" stroke-width="1"/>
  <text x="8" y="208" fill="#ddd">raw/candidate rotation, translation, residuals</text>`;
}});
document.getElementById('export').onclick=()=>{{
  const rows=events.map(e=>({{event_id:e.event_id,decision:decisions[e.event_id]||'unreviewed'}}));
  const blob=new Blob([JSON.stringify({{schema_version:1,manifest_digest:manifestDigest,decisions:rows}},null,2)],{{type:'application/json'}});
  const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='decisions.json';a.click();
  URL.revokeObjectURL(a.href);
}};
</script></body></html>"""
    output.write_text(document)
    return output


def build_review(task_root: Path, output: Path, render: bool = False,
                 context_frames: int = 60,
                 candidate_task_root: Path | None = None,
                 high_sample_rate: float = 0.1,
                 sample_seed: int = 0) -> dict[str, Any]:
    """Scan, optionally render, and write one complete review package."""
    task_root, output = Path(task_root), Path(output)
    if candidate_task_root is None:
        events = scan_tree(task_root)
        # A clip already includes `context_frames` on both sides. Merge old
        # unresolved detections; candidate events stay distinct because their
        # deterministic event IDs are the unit of a replayable decision.
        review = merge_review_events(events, max_gap=2 * int(context_frames))
    else:
        events = load_candidate_events(
            task_root, candidate_task_root, chart_context=context_frames)
        review = select_review_events(
            events, high_sample_rate=high_sample_rate, seed=sample_seed)
    if render:
        for i, event in enumerate(review, 1):
            filename = event["event_id"].replace("/", "_") + ".mp4"
            relative = Path("clips") / filename
            print(f"[{i}/{len(review)}] {event['event_id']}", flush=True)
            kwargs = {}
            if candidate_task_root is not None:
                kwargs["candidate_parquet"] = Path(event["candidate_parquet"])
            info = render_event_clip(
                event, task_root, output / relative,
                context_frames=context_frames, **kwargs)
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
    ap.add_argument("--candidate-task-root", type=Path,
                    help="candidate <task> root containing meta/ and repair_events/")
    ap.add_argument("--high-sample-rate", type=float, default=0.1,
                    help="fraction of HIGH events rendered for audit")
    ap.add_argument("--sample-seed", type=int, default=0)
    args = ap.parse_args()
    result = build_review(
        args.root, args.output, args.render, args.context_frames,
        candidate_task_root=args.candidate_task_root,
        high_sample_rate=args.high_sample_rate,
        sample_seed=args.sample_seed)
    print(f"events={len(result['events'])} "
          f"unresolved_clips={len(result['review_events'])}")
    print(result["json"])
    if result["html"]:
        print(result["html"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
