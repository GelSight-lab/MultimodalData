"""Does a prepared session folder match the published dataset layout?

Run it before uploading. Every check below is one that failed silently while
preparing the 2026-09-09 session — the folder uploaded cleanly each time and
was still wrong:

  * the first upload had no `previews/` at all, because nothing said it should
  * `curate` rebuilt `episodes.jsonl` from `_detect.pt` sidecars, which only
    the newly built episodes had, and dropped 29 of 32 rows
  * the force export refused an episode missing from `episodes.jsonl`, which
    is the only reason the dropped rows were noticed
  * the calibration shipped in `calibration/` had no file saying which epoch
    it is, and this session's epoch was changed after the first upload

    python -m twm.dataset_layout check <folder> --date 2026-09-09
    python -m twm.dataset_layout check-remote yxma/React data/validation --date 2026-09-09
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

VIDEO_FILES = ("view_left.mp4", "view_middle.mp4", "view_right.mp4",
               "tactile_left.mp4", "tactile_right.mp4")
DEPTH_FILES = ("depth_left.mkv", "depth_middle.mkv", "depth_right.mkv")
# Present only for sessions recorded after the wrist cameras were wired in
# (2026-09), so their absence is reported but does not fail a folder.
WRIST_FILES = ("wrist_left.mp4", "wrist_right.mp4")
CAM_CALIB = ("T_mocap_to_cam_left", "T_mocap_to_cam_middle", "T_mocap_to_cam_right")
GEL_CALIB = ("T_gel_to_rigid_left.json", "T_gel_to_rigid_right.json")
# The force channel has two halves and they are not the same kind of thing.
#
# MEASURED: the estimated newtons, and the tactile frame each number came
# from. Required whenever a task declares a force channel at all.
#
# DERIVED: a control policy computed at an ASSUMED stiffness -- penetration is
# force/k and the target pose is the observed pose displaced by it. Optional,
# because v8 measures to 15 N while the shipped k = 2 N/mm caps the exporter's
# gel-thickness gate at 8.5 N, and the operator chose on 2026-09-16 to publish
# the measurement and withhold the policy rather than invent a stiffness.
#
# Optional but ALL-OR-NOTHING: a target_pose without the penetration it was
# displaced by is a half-written export, not a deliberate choice.
FORCE_MEASURED = ("force_left_normal_n", "force_left_source_frame",
                  "force_right_normal_n", "force_right_source_frame")
FORCE_DERIVED = ("force_left_penetration_mm", "force_left_target_pose",
                 "force_right_penetration_mm", "force_right_target_pose")
FORCE_COLUMNS = FORCE_MEASURED + FORCE_DERIVED


def missing_force_columns(cols) -> list:
    """Which force columns a parquet still owes, given what it has.

    Empty for a force-only export and for a full one; non-empty for an episode
    with no force at all, and for a partially written derived half.
    """
    have = set(cols)
    missing = [c for c in FORCE_MEASURED if c not in have]
    present_derived = [c for c in FORCE_DERIVED if c in have]
    if present_derived:
        missing += [c for c in FORCE_DERIVED if c not in have]
    return missing
INDEX_FILES = ("episodes.jsonl", "segments.json", "bad_frames.json", "splits.json")


@dataclass
class Problem:
    check: str
    message: str


@dataclass
class LayoutReport:
    root: str
    date: str
    episodes: List[str] = field(default_factory=list)
    problems: List[Problem] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    facts: Dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.problems

    def fail(self, check: str, message: str) -> None:
        self.problems.append(Problem(check, message))

    def to_dict(self):
        return {"root": self.root, "date": self.date, "ok": self.ok,
                "episodes": self.episodes, "facts": self.facts,
                "warnings": list(self.warnings),
                "problems": [{"check": p.check, "message": p.message} for p in self.problems]}

    def table(self) -> str:
        lines = [f"{'LAYOUT OK' if self.ok else 'LAYOUT FAILED'}: {self.root} ({self.date})",
                 f"  episodes: {', '.join(self.episodes) or '(none)'}"]
        for k, v in self.facts.items():
            lines.append(f"  {k}: {v}")
        for p in self.problems:
            lines.append(f"  problem [{p.check}] {p.message}")
        for w in self.warnings:
            lines.append(f"  warning: {w}")
        return "\n".join(lines)


class Unreadable(Exception):
    """A file that exists but will not parse."""


def _read_json(path: Path):
    """Raises rather than returning a sentinel: a corrupt index reported as
    'missing' sends the reader looking for the wrong thing."""
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        raise Unreadable(f"{path.name} is present but unparseable "
                         f"({type(exc).__name__}: {exc})") from None


def _parquet_rows_and_columns(path: Path):
    import pandas as pd
    df = pd.read_parquet(path)
    return len(df), list(df.columns)


def _stream_frames(path) -> int:
    """Frames the container can actually deliver, by counting PACKETS.

    Three ways to ask, measured on a real 1995-frame published video:

        nb_read_frames  (-count_frames)   1.95 s   decodes every frame
        nb_read_packets (-count_packets)  0.06 s   reads the index
        nb_frames       (metadata only)   0.06 s   believes the header

    All three see a re-encoded short file. Only the first two see a TRUNCATED
    one — `nb_frames` cheerfully reports 1995 for a file whose second half is
    gone, which is the exact case this check exists for. Packets cost 1/32nd of
    a decode and catch both, so a whole task is seconds rather than a quarter
    of an hour, and the gate can stay on by default.

    Decoding is still what `segment.py` does on the cut path, where frames are
    being rewritten rather than merely counted.
    """
    import subprocess
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True)
    if r.returncode != 0 or not r.stdout.strip().isdigit():
        raise RuntimeError((r.stderr or r.stdout).strip()[:200] or "no frame count")
    return int(r.stdout.strip())


def check_layout(root, date: str, *, require_force: bool = True,
                 require_previews: bool = True,
                 count_frames: bool = False) -> LayoutReport:
    """Validate one session inside a task-shaped folder.

    ``count_frames`` DECODES every published video and compares its length to
    the parquet beside it. Off by default because `check_remote` never
    downloads the videos — it writes empty placeholders and checks presence —
    so it has no bytes to count. Turn it on for a local tree, which is where
    the gate runs before an upload anyway.

    The cut path has guarded this since it existed (`segment.py`); the
    whole-episode path only ever checked that the files were THERE.
    """
    root = Path(root)
    rep = LayoutReport(str(root), date)
    meta = root / "meta" / date
    if not meta.is_dir():
        rep.fail("meta", f"no meta/{date}/ directory")
        return rep

    parquets = sorted(meta.glob("episode_*.parquet"))
    rep.episodes = [p.stem for p in parquets]
    if not parquets:
        rep.fail("meta", f"meta/{date}/ holds no episode parquet")
        return rep

    rows_by_ep = {}
    for pq in parquets:
        ep = pq.stem
        try:
            n, cols = _parquet_rows_and_columns(pq)
        except Exception as exc:
            rep.fail("meta", f"{ep}: parquet unreadable ({type(exc).__name__}: {exc})")
            continue
        rows_by_ep[ep] = n
        # A FAILURE, not a warning. These were a warning, so 84 parquets
        # shipped without them and the gate still said pass; a warning nobody
        # fails on is a comment. Report every missing column rather than the
        # first — the previous `break` hid `frame_index` behind the four that
        # had just been added.
        # A WARNING, not a failure, and not "the LeRobot index columns".
        #
        # Two things were checked and both came back negative. These are not
        # LeRobot's set -- the standard one (verified against lerobot/pusht and
        # lerobot/aloha_sim_insertion_human, both codebase_version v3.0) is
        # `episode_index, frame_index, timestamp, index, task_index`; `task`
        # and `episode` are strings we invented and `index`/`timestamp` are
        # missing here. And nothing reads them: `ReactVideoDataset`, the loader
        # this dataset actually ships, locates data by PATH
        # (videos/<date>/<ep>/*.mp4 beside meta/<date>/<ep>.parquet, row i =
        # frame i) and never opens these columns. `build_lerobot_dataset`
        # recomputes its own indices rather than carrying these across.
        #
        # They were raised to a failure on 2026-09-13 on the grounds that every
        # older published folder has them. That is consistency, which is not
        # the same as necessity -- and a gate that blocks new data over
        # metadata with no consumer costs more than it protects. The columns
        # stay for consistency with what is already published; the gate does
        # not.
        missing_idx = [c for c in ("task", "task_index", "episode",
                                   "episode_index", "frame_index")
                       if c not in cols]
        if missing_idx:
            rep.warnings.append(
                f"{ep}: parquet has no {', '.join(missing_idx)} — carried by "
                f"every older published folder, but no known consumer reads "
                f"them (react_preprocess.meta.add_index_columns adds them)")
        if require_force:
            missing = missing_force_columns(cols)
            if missing:
                rep.fail("force columns",
                         f"{ep}: parquet has no {missing[0]} — the force export "
                         f"has not been run over it")
        for name, files in (("videos", VIDEO_FILES), ("depth", DEPTH_FILES)):
            d = root / name / date / ep
            for f in files:
                if not (d / f).is_file():
                    rep.fail(name, f"{ep}: missing {name}/{date}/{ep}/{f}")
        if count_frames:
            for f in VIDEO_FILES + WRIST_FILES:
                v = root / "videos" / date / ep / f
                if not v.is_file():
                    continue                      # already reported above
                try:
                    got = _stream_frames(v)
                except Exception as exc:          # noqa: BLE001
                    rep.fail("videos", f"{ep}: {f} will not read "
                                       f"({type(exc).__name__}: {exc})")
                    continue
                if got != n:
                    rep.fail("videos", f"{ep}: {f} has {got} frames but the "
                                       f"parquet has {n} rows")
        missing_wrist = [w for w in WRIST_FILES
                         if not (root / "videos" / date / ep / w).is_file()]
        if len(missing_wrist) == len(WRIST_FILES):
            rep.warnings.append(f"{ep}: no wrist-camera videos (recorded before the "
                                f"Arducams were wired in, or the build predates them)")
        elif missing_wrist:
            rep.fail("videos", f"{ep}: has one wrist video but not {missing_wrist[0]}")
        if require_previews and not (root / "previews" / date / f"{ep}.mp4").is_file():
            rep.fail("previews", f"{ep}: missing preview previews/{date}/{ep}.mp4")

    # ── the indices ──────────────────────────────────────────────────────────
    for name in INDEX_FILES:
        if not (root / name).is_file():
            rep.fail("indices", f"missing {name}")

    jsonl = root / "episodes.jsonl"
    if jsonl.is_file():
        listed = {}
        for line in jsonl.read_text().splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                rep.fail("episodes.jsonl", f"unparseable row ({type(exc).__name__}: {exc})")
                continue
            if row.get("date") == date:
                listed[row["episode"].split("/")[-1]] = row
        for ep in rows_by_ep:
            if ep not in listed:
                rep.fail("episodes.jsonl", f"{ep} has a parquet but no episodes.jsonl row")
            elif listed[ep].get("n_frames") not in (None, rows_by_ep[ep]):
                rep.fail("episodes.jsonl",
                         f"{ep}: episodes.jsonl says n_frames={listed[ep]['n_frames']}, "
                         f"the parquet has {rows_by_ep[ep]} rows")
        for ep in listed:
            if ep not in rows_by_ep:
                rep.fail("episodes.jsonl", f"{ep} is listed but has no parquet")

    if (root / "splits.json").is_file():
        try:
            splits = _read_json(root / "splits.json")
        except Unreadable as exc:
            rep.fail("indices", str(exc))
        else:
            covered = set((splits or {}).get("episodes") or {})
            for ep in rows_by_ep:
                if f"{date}/{ep}" not in covered:
                    # Not a warning. A loader that cannot find an episode in
                    # splits.json treats it as TRAINING data and says nothing
                    # (ReactVideoDataset._split_filter), so an uncovered
                    # episode is a leak with no trace in any metric.
                    rep.fail("splits.json",
                             f"{date}/{ep} is not in splits.json — a loader "
                             f"silently treats an unlisted episode as train. "
                             f"Rebuild with build_splits.py, using --hold-out "
                             f"if this session is held out whole.")

    _check_calibration(root, date, rep)
    return rep


def _check_calibration(root: Path, date: str, rep: LayoutReport) -> None:
    cal = root / "calibration"
    if not cal.is_dir():
        rep.fail("calibration", "no calibration/ directory")
        return
    created = set()
    for c in CAM_CALIB:
        j = cal / f"{c}.json"
        if not j.is_file():
            rep.fail("calibration", f"missing calibration/{c}.json")
            continue
        if not (cal / f"{c}.npy").is_file():
            rep.fail("calibration", f"missing calibration/{c}.npy")
        try:
            d = _read_json(j)
        except Unreadable as exc:
            rep.fail("calibration", str(exc))
            continue
        if d.get("created_at"):
            created.add(str(d["created_at"])[:10])
    for g in GEL_CALIB:
        if not (cal / g).is_file():
            rep.fail("calibration", f"missing calibration/{g}")
    if len(created) > 1:
        rep.fail("calibration",
                 f"the three cameras were solved in different epochs {sorted(created)}; "
                 f"a folder must ship one epoch")
    epoch = created.pop() if len(created) == 1 else None
    if epoch:
        rep.facts["calibration epoch"] = epoch

    doc_path = cal / "calibration.json"
    if not doc_path.is_file():
        rep.fail("calibration.json",
                 "missing calibration/calibration.json — nothing in the folder "
                 "then says which epoch these extrinsics are")
        return
    try:
        doc = _read_json(doc_path)
    except Unreadable as exc:
        rep.fail("calibration.json", str(exc))
        return
    if epoch and str(doc.get("created", ""))[:10] != epoch:
        rep.fail("calibration.json",
                 f"calibration.json says created={doc.get('created')} but the "
                 f"T_mocap_to_cam files carry {epoch}")
    applies = doc.get("applies_to_dates") or []
    if date not in applies:
        rep.fail("calibration.json",
                 f"applies_to_dates {applies} does not include this session {date}")


def check_remote(repo_id: str, prefix: str, date: str, **kw) -> LayoutReport:
    """Same checks against a folder already on the Hub.

    Downloads only the small files (parquet, json, jsonl); videos and depth
    are checked by presence in the file listing.
    """
    import tempfile

    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    files = [f for f in api.list_repo_files(repo_id, repo_type="dataset")
             if f.startswith(prefix.rstrip("/") + "/")]
    if not files:
        rep = LayoutReport(f"{repo_id}:{prefix}", date)
        rep.fail("remote", f"{prefix} holds no files in {repo_id}")
        return rep
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "root"
        for f in files:
            rel = f[len(prefix.rstrip("/")) + 1:]
            dest = root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            if rel.endswith((".parquet", ".json", ".jsonl")):
                local = hf_hub_download(repo_id, f, repo_type="dataset")
                dest.write_bytes(Path(local).read_bytes())
            else:
                dest.write_bytes(b"")          # presence is what is checked
        rep = check_layout(root, date, **kw)
    rep.root = f"{repo_id}:{prefix}"
    return rep


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="python -m twm.dataset_layout",
                                description="Check a prepared session folder against "
                                            "the published dataset layout.")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="a local folder")
    c.add_argument("root")
    # Local only: check-remote never downloads the videos, so it has no bytes
    # to count. This is the gate that runs before an upload anyway.
    c.add_argument("--no-frame-count", action="store_true",
                   help="skip decoding each video to compare its length with "
                        "the parquet (decoding a session takes minutes)")
    r = sub.add_parser("check-remote", help="a folder already on the Hub")
    r.add_argument("repo_id")
    r.add_argument("prefix")
    for q in (c, r):
        q.add_argument("--date", required=True)
        q.add_argument("--no-force", action="store_true", help="skip the force-column check")
        q.add_argument("--no-previews", action="store_true", help="skip the preview check")
        q.add_argument("--json", action="store_true")
    a = p.parse_args(argv)
    kw = {"require_force": not a.no_force, "require_previews": not a.no_previews}
    rep = (check_layout(a.root, a.date, count_frames=not a.no_frame_count, **kw)
           if a.cmd == "check"
           else check_remote(a.repo_id, a.prefix, a.date, **kw))
    print(json.dumps(rep.to_dict(), indent=2) if a.json else rep.table())
    return 0 if rep.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
