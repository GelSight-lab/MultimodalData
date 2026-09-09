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
CAM_CALIB = ("T_mocap_to_cam_left", "T_mocap_to_cam_middle", "T_mocap_to_cam_right")
GEL_CALIB = ("T_gel_to_rigid_left.json", "T_gel_to_rigid_right.json")
FORCE_COLUMNS = ("force_left_normal_n", "force_left_penetration_mm",
                 "force_left_target_pose", "force_left_source_frame",
                 "force_right_normal_n", "force_right_penetration_mm",
                 "force_right_target_pose", "force_right_source_frame")
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


def check_layout(root, date: str, *, require_force: bool = True,
                 require_previews: bool = True) -> LayoutReport:
    """Validate one session inside a task-shaped folder."""
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
        if require_force:
            missing = [c for c in FORCE_COLUMNS if c not in cols]
            if missing:
                rep.fail("force columns",
                         f"{ep}: parquet has no {missing[0]} — the force export "
                         f"has not been run over it")
        for name, files in (("videos", VIDEO_FILES), ("depth", DEPTH_FILES)):
            d = root / name / date / ep
            for f in files:
                if not (d / f).is_file():
                    rep.fail(name, f"{ep}: missing {name}/{date}/{ep}/{f}")
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
                    rep.warnings.append(f"splits.json does not cover {date}/{ep}")

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
    rep = (check_layout(a.root, a.date, **kw) if a.cmd == "check"
           else check_remote(a.repo_id, a.prefix, a.date, **kw))
    print(json.dumps(rep.to_dict(), indent=2) if a.json else rep.table())
    return 0 if rep.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
