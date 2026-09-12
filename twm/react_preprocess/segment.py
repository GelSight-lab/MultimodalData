"""Cut published episodes down to their clean spans.

    python -m react_preprocess segment --task pushT

An episode built by ``pipeline`` is whatever the operator recorded, defects and
all; ``curation`` then writes the defects to ``bad_frames.json`` and their
complement to ``segments.json``. That is an honest description, but it is only
a description: a reader who loads the parquet and the MP4s and never opens the
sidecar trains on the frozen tactile and the teleported poses without ever
being told. Nine of nineteen 2026-09 episodes carry at least one such span, one
of them for its last two minutes.

This stage makes the description structural instead. Each clean span becomes
its own published episode, so every frame that ships is a frame that passed
every detector, and there is no annotation left for a reader to skip. What was
``bad_frames.json`` becomes the gaps between episodes.

The cut is the LAST stage, after force recovery and the Z-up conversion, so
every column those stages added is carried through by the same row slice and
neither has to learn about segments.

Cost, measured over the 2026-09 sessions: 132.2 min of recording becomes
122.5 min across 38 episodes — 7.3% discarded, of which 4.7% is defective and
2.6% is clean-but-too-short (see ``MIN_PUBLISH_SECONDS``).
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from . import curation, detect as D
from .config import FPS, STAGE_ROOT

# A published segment must be long enough to be a demonstration, not a
# fragment. The measured span lengths are strongly bimodal: 19 spans under 1.3
# seconds (the slivers between two nearby defects) and then a clean jump to
# 2.5s, 11.6s and up. Anything below the floor is dropped rather than shipped,
# because a 0.4-second "episode" costs a reader more to notice and exclude than
# it can possibly contribute.
#
# 30s over 10s was the operator's call on 2026-09-11: at 10s the release keeps
# 125.9 min in 48 episodes, at 30s it keeps 122.5 min in 38, and every one of
# those 38 is long enough to hold a complete manipulation attempt.
MIN_PUBLISH_SECONDS = 30.0
MIN_PUBLISH_FRAMES = int(round(MIN_PUBLISH_SECONDS * FPS))

# The streams every complete episode has. Named here so a partial build is
# caught at cut time rather than silently shipping an episode missing a view.
EXPECTED_STREAMS = ("view_left", "view_middle", "view_right",
                    "tactile_left", "tactile_right",
                    "wrist_left", "wrist_right")


def segment_name(episode: str, idx: int) -> str:
    """``episode_003`` + span 1 -> ``episode_003_seg01``.

    The source episode stays legible in the name on purpose. A renumbered flat
    sequence would make the release tidier and make it impossible to ask "which
    recording did this come from" without a lookup table.
    """
    return f"{episode}_seg{idx:02d}"


def publishable_spans(report: dict, min_frames: int = MIN_PUBLISH_FRAMES):
    """The clean spans of one episode report that are worth publishing.

    Inclusive ``[a, b]`` in episode-video coordinates, which is what
    ``segments.json`` uses, so these index the built MP4s and parquet directly.
    """
    T = int(report["n_frames"])
    return [(a, b) for a, b in D.find_clean_segments(T, curation._bad_intervals(report))
            if b - a + 1 >= min_frames]


# ── video ────────────────────────────────────────────────────────────────────

def dimensions(path: Path) -> tuple[int, int]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height", "-of", "csv=p=0:s=x", str(path)],
        check=True, capture_output=True, text=True).stdout.strip()
    w, h = out.split("x")[:2]
    return int(w), int(h)


def cut_video(src: Path, spans, dsts) -> None:
    """Write one output per span, frame-exactly, from a single decode pass.

    The routing is done here, on decoded frames, rather than by an ffmpeg
    filter graph. The obvious formulation —
    ``select=between(n\\,a\\,b),setpts=N/FRAME_RATE/TB`` per output — is not
    frame-exact: on a 120-frame probe, asking for [10,29] returned twenty
    frames whose contents were 10, 10, 12, 13, … . The count was right and the
    pixels were not, which is the failure mode this stage exists to remove and
    the one a count check cannot see. Frames carrying their own index made it
    visible; a real cut would just have shipped a duplicated frame.

    Counting frames off the decoder in Python has no such subtlety, and reuses
    ``encode.rgb_writer``, so a cut stream cannot drift from the encoder
    settings the rest of the release is published with.

    The input is still decoded ONCE for all of a stream's spans: on the 12-span
    pushT episode a call per span would mean 12 full decodes of each of its 7
    streams.

    This re-encodes, so a cut stream is a second H.264 generation. At CRF 18
    yuv444p that is visually lossless again, and it is unavoidable: the sessions
    whose source HDF5 was deleted (``config.RAW_DELETED``) have no other master,
    so cutting from the H5 would work for some episodes and not others.
    """
    from contextlib import ExitStack

    from .encode import VideoWriter

    spans = list(spans)
    dsts = list(dsts)
    w, h = dimensions(src)
    stride = w * h * 3
    last = max(b for _, b in spans)

    proc = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(src),
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        stdout=subprocess.PIPE, bufsize=stride)
    try:
        with ExitStack() as stack:
            writers = []
            for dst in dsts:
                dst.parent.mkdir(parents=True, exist_ok=True)
                writers.append(stack.enter_context(
                    VideoWriter(dst, pix_fmt="bgr24", codec="libx264",
                                width=w, height=h)))
            n = 0
            while n <= last:
                buf = proc.stdout.read(stride)
                if len(buf) < stride:
                    raise RuntimeError(
                        f"{src}: decode ended at frame {n}, but a span needs "
                        f"frame {last}")
                block = np.frombuffer(buf, np.uint8).reshape(1, h, w, 3)
                for (a, b), wr in zip(spans, writers):
                    if a <= n <= b:
                        wr.write(block)
                n += 1
    finally:
        proc.stdout.close()
        proc.wait()


def frame_count(path: Path) -> int:
    """Frames in a video, by decode. Slower than the container's count and the
    only one worth trusting: ``nb_frames`` is whatever the muxer wrote."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)],
        check=True, capture_output=True, text=True).stdout.strip()
    return int(out.rstrip(","))


# ── parquet ──────────────────────────────────────────────────────────────────

def cut_table(table, a: int, b: int, source_episode: str, name: str):
    """Rows ``[a, b]`` of an episode table, renumbered as a standalone episode.

    Per-row identity is rewritten (``frame_idx``, ``frame_index``, ``episode``)
    and per-row provenance is added, so a segment can be traced back to the
    frame of the recording it came from without consulting an index file.
    ``timestamp`` and ``source_h5_frame`` are NOT rebased: they are statements
    about when the data was recorded and where it sits in the raw file, and both
    stay true of a slice.
    """
    import pyarrow as pa

    sub = table.slice(a, b - a + 1)
    n = sub.num_rows
    cols = {
        "frame_idx": pa.array(np.arange(n, dtype=np.int32)),
        "source_episode": pa.array([source_episode] * n, pa.string()),
        "source_frame_idx": pa.array(np.arange(a, b + 1, dtype=np.int32)),
    }
    # Only rewritten if the enrichment step has already run; adding them here
    # would invent an episode_index this stage has no basis to choose.
    if "frame_index" in sub.column_names:
        cols["frame_index"] = pa.array(np.arange(n, dtype=np.int64))
    if "episode" in sub.column_names:
        cols["episode"] = pa.array([name] * n, pa.string())

    for col, arr in cols.items():
        i = sub.schema.get_field_index(col)
        sub = (sub.set_column(i, col, arr) if i >= 0
               else sub.append_column(col, arr))
    return sub


# ── one episode ──────────────────────────────────────────────────────────────

def cut_episode(task: str, date: str, episode: str, spans,
                src_root: Path, dst_root: Path, verify: bool = True,
                expected_T: int | None = None) -> list[dict]:
    """Cut one built episode into its publishable segments.

    Returns one row per emitted segment. An episode whose only span covers the
    whole recording is copied rather than re-encoded: it has nothing to cut, and
    re-encoding it would cost a generation of quality for no change.

    ``expected_T`` is the length the detector reports saw. The spans are frame
    indices, and they are computed on one tree and applied to another (the
    sidecars live with the uncut master, the cut is applied to the Z-up tree
    that is actually published). Every stage between them preserves row order
    and row count -- which is exactly the kind of assumption that holds until
    it does not, and would fail silently by cutting at the wrong frames.
    """
    src_vid = src_root / "videos" / date / episode
    src_pq = src_root / "meta" / date / f"{episode}.parquet"
    table = pq.read_table(str(src_pq))
    T = table.num_rows
    if expected_T is not None and T != expected_T:
        raise RuntimeError(
            f"{task}/{date}/{episode}: the spans were computed on {expected_T} "
            f"frames but this tree has {T} rows. The frame numbering differs "
            f"between the two trees, so the cut would land elsewhere.")

    missing = [s for s in EXPECTED_STREAMS if not (src_vid / f"{s}.mp4").is_file()]
    if missing:
        raise FileNotFoundError(
            f"{task}/{date}/{episode}: incomplete build, no {', '.join(missing)}. "
            f"Cutting it would publish an episode missing a stream.")

    whole = len(spans) == 1 and spans[0] == (0, T - 1)
    names = [episode if whole else segment_name(episode, i)
             for i in range(len(spans))]

    for stream in EXPECTED_STREAMS:
        src = src_vid / f"{stream}.mp4"
        dsts = [dst_root / "videos" / date / nm / f"{stream}.mp4" for nm in names]
        if whole:
            dsts[0].parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dsts[0])
        else:
            cut_video(src, spans, dsts)

    rows = []
    for (a, b), nm in zip(spans, names):
        sub = cut_table(table, a, b, f"{date}/{episode}", nm)
        out_pq = dst_root / "meta" / date / f"{nm}.parquet"
        out_pq.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(sub, str(out_pq))

        if verify:
            # The whole point of the stage is that a published frame index means
            # what it says, so the row count and every stream's frame count have
            # to agree before the segment counts as written.
            for stream in EXPECTED_STREAMS:
                got = frame_count(dst_root / "videos" / date / nm / f"{stream}.mp4")
                if got != sub.num_rows:
                    raise RuntimeError(
                        f"{task}/{date}/{nm}: {stream}.mp4 has {got} frames but "
                        f"the parquet has {sub.num_rows} rows")

        rows.append({
            "episode": f"{date}/{nm}", "date": date, "task": task,
            "source_episode": f"{date}/{episode}",
            "source_frame_range": [int(a), int(b)],
            "n_frames": int(sub.num_rows),
            "duration_s": round(sub.num_rows / FPS, 3),
            "recut": not whole,
        })
    return rows


# ── one task ─────────────────────────────────────────────────────────────────

def build_task(task: str, src_root: Path = STAGE_ROOT,
               dst_root: Path | None = None, dates=None,
               min_frames: int = MIN_PUBLISH_FRAMES,
               verify: bool = True, dry_run: bool = False,
               detect_root: Path | None = None) -> dict:
    """Cut every built episode of a task into its publishable segments.

    ``detect_root`` is where the ``_detect.pt`` sidecars and the videos the
    corruption detectors read live; ``src_root`` is the tree actually cut.
    They differ in the real chain: defects are a property of the recording and
    are measured once on the master, while what gets published has been through
    force recovery and the Z-up conversion. Frame numbering is checked to match
    rather than assumed (see ``cut_episode``).
    """
    src_root = Path(src_root) / task
    det_root = Path(detect_root) / task if detect_root else src_root
    dst_root = Path(dst_root) if dst_root else Path(str(STAGE_ROOT) + "_cut")
    out = dst_root / task

    # Enumerate the tree being CUT, not the tree the defects were measured on.
    # Those are the same set in a full run and are not during a partial one: a
    # wave that publishes the 20 episodes built so far has 20 parquets and 33
    # sidecars, and discovering by sidecar would try to cut 13 episodes this
    # tree does not contain.
    parquets = sorted((src_root / "meta").rglob("episode_*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"no episode parquet under {src_root/'meta'}")

    rows, dropped, raw_frames = [], [], 0
    for pq_path in parquets:
        date, episode = pq_path.parent.name, pq_path.stem
        if dates and date not in dates:
            continue
        det = det_root / "meta" / date / f"{episode}._detect.pt"
        if not det.is_file():
            raise FileNotFoundError(
                f"{task}/{date}/{episode}: no _detect.pt under {det_root}. The "
                f"spans cannot be computed, and publishing it uncut would ship "
                f"the defects this stage exists to remove.")
        report, _ = curation.episode_report(
            det, video_dir=det_root / "videos" / date / episode)
        raw_frames += int(report["n_frames"])
        spans = publishable_spans(report, min_frames)
        kept = {(a, b) for a, b in spans}
        for a, b in D.find_clean_segments(int(report["n_frames"]),
                                          curation._bad_intervals(report)):
            if (a, b) not in kept:
                dropped.append({"episode": f"{date}/{episode}",
                                "frame_range": [int(a), int(b)],
                                "n_frames": int(b - a + 1)})
        if not spans:
            dropped.append({"episode": f"{date}/{episode}", "frame_range": None,
                            "n_frames": 0, "note": "no span reached the floor"})
            continue
        if dry_run:
            rows += [{"episode": f"{date}/{segment_name(episode, i)}",
                      "n_frames": int(b - a + 1)} for i, (a, b) in enumerate(spans)]
            continue
        rows += cut_episode(task, date, episode, spans, src_root, out, verify,
                            expected_T=int(report["n_frames"]))

    kept_frames = sum(r["n_frames"] for r in rows)
    summary = {
        "task": task, "episodes": len(rows),
        "raw_frames": raw_frames, "kept_frames": kept_frames,
        "kept_minutes": round(kept_frames / FPS / 60, 2),
        "discarded_frames": raw_frames - kept_frames,
        "kept_fraction": round(kept_frames / raw_frames, 4) if raw_frames else 0.0,
        "min_publish_seconds": MIN_PUBLISH_SECONDS,
        "dropped_spans": dropped,
    }
    if not dry_run:
        out.mkdir(parents=True, exist_ok=True)
        (out / "episodes.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows))
        (out / "segment_provenance.json").write_text(
            json.dumps({"summary": {k: v for k, v in summary.items()
                                    if k != "dropped_spans"},
                        "thresholds": D.thresholds(),
                        "min_publish_seconds": MIN_PUBLISH_SECONDS,
                        "dropped_spans": dropped,
                        "segments": rows}, indent=2))
    return summary
