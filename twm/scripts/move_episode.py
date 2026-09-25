"""Refile a recording under the task it actually belongs to.

Four 2026-09-11 recordings were filed as `rope` and are pushT: the overhead
view shows the blue T. Renaming the H5 is the small part. The recording has
products in five trees, its identity is stamped into five parquet columns, and
two tasks' index files describe it -- and a segment cannot be left behind,
because its `source_episode` would then name a recording in another task.

    python -m twm.scripts.move_episode rope/2026-09-11/episode_008 \
                                       pushT/2026-09-11/episode_005 [--dry-run]

The destination episode number is given, not computed: the recorder restarts
numbering daily, so "the next free one" depends on which trees you look at,
and a wrong answer silently overwrites a different recording. The move refuses
if anything is already there.

This does NOT touch the hub. Publish afterwards, and delete the old paths
there -- `--no_delete` means an upload never removes what it no longer covers.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from twm.react_preprocess.meta import task_index          # noqa: E402

DISK = Path("/media/yxma/Disk1/twm")

# tree -> (subpath template relative to <tree>/<task>, is_per_segment)
# The five trees a recording lands in, named once. A tree missing from a given
# installation is skipped rather than failing: rope has no cut segments for the
# recordings that produced none, and that is not an error.
LAYOUT = (
    ("data",           "{date}",                 False),
    ("release",        "meta/{date}",            False),
    ("release",        "videos/{date}",          False),
    ("release_force",  "meta/{date}",            False),
    ("release_cut",    "meta/{date}",            True),
    ("release_cut",    "videos/{date}",          True),
    ("release_cut",    "previews/{date}",        True),
    ("force_recovery", "{date}",                 False),
)

IDENTITY_COLUMNS = ("task", "task_index", "episode", "episode_index",
                    "source_episode")


@dataclass(frozen=True)
class Move:
    src_task: str
    src_date: str
    src_episode: str
    dst_task: str
    dst_date: str
    dst_episode: str

    @classmethod
    def parse(cls, src: str, dst: str) -> "Move":
        a, b = [x.strip("/").split("/") for x in (src, dst)]
        if len(a) != 3 or len(b) != 3:
            raise SystemExit("give both as <task>/<date>/<episode>")
        return cls(*a, *b)

    def rename(self, name: str) -> str:
        """`episode_008_seg01.parquet` -> `episode_005_seg01.parquet`.

        Only the leading episode name is replaced, so the segment number and
        the suffix survive -- `_seg01`, `._detect.pt`, `_left.npz` all mean
        something and none of them is part of the identity being changed.
        """
        if not name.startswith(self.src_episode):
            return name
        return self.dst_episode + name[len(self.src_episode):]


def plan_files(mv: Move, root: Path = DISK) -> list[tuple[Path, Path]]:
    """Every path that must move, and where it goes."""
    root = Path(root)
    pairs: list[tuple[Path, Path]] = []
    for tree, sub, _per_seg in LAYOUT:
        d = root / tree / mv.src_task / sub.format(date=mv.src_date)
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            # `episode_008` must not match `episode_0080`; the separator after
            # the episode name is always `.` or `_`, never a digit.
            rest = p.name[len(mv.src_episode):]
            if not p.name.startswith(mv.src_episode) or (rest and rest[0].isdigit()):
                continue
            dst = (root / tree / mv.dst_task / sub.format(date=mv.dst_date)
                   / mv.rename(p.name))
            pairs.append((p, dst))
    return pairs


def rewrite_parquet(path: Path, mv: Move, episode_index: int) -> None:
    """Stamp the destination identity into a cut parquet."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    t = pq.read_table(str(path))
    if not any(c in t.column_names for c in IDENTITY_COLUMNS):
        return                                  # an uncut tree carries none
    n = t.num_rows
    # Idempotent: `rename` leaves a name that is already the destination's
    # alone, so this does not depend on whether the file has been moved yet.
    # Deriving the identity from the filename AND from the move, in an order
    # that mattered, is the kind of coupling this module exists to remove.
    stem = mv.rename(path.stem)
    new = {
        "task": pa.array([mv.dst_task] * n, pa.string()),
        "task_index": pa.array([task_index(mv.dst_task)] * n, pa.int64()),
        "episode": pa.array([f"{mv.dst_date}/{stem}"] * n, pa.string()),
        "episode_index": pa.array([episode_index] * n, pa.int64()),
        "source_episode": pa.array(
            [f"{mv.dst_date}/{mv.dst_episode}"] * n, pa.string()),
    }
    for name, arr in new.items():
        if name in t.column_names:
            t = t.set_column(t.schema.get_field_index(name), name, arr)
    pq.write_table(t, str(path))


def _move_index_rows(mv: Move, root: Path) -> None:
    """Carry the recording's rows from the source index to the destination's."""
    for tree in ("release", "release_cut"):
        src = root / tree / mv.src_task / "episodes.jsonl"
        dst = root / tree / mv.dst_task / "episodes.jsonl"
        if not src.is_file():
            continue
        keep, moved = [], []
        for line in src.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = row.get("episode", "")
            ep = key.split("/")[-1]
            rest = ep[len(mv.src_episode):]
            if (key.startswith(f"{mv.src_date}/{mv.src_episode}")
                    and not (rest and rest[0].isdigit())):
                row["episode"] = f"{mv.dst_date}/{mv.rename(ep)}"
                row["date"] = mv.dst_date
                if "task" in row:
                    row["task"] = mv.dst_task
                if "source_episode" in row:
                    row["source_episode"] = f"{mv.dst_date}/{mv.dst_episode}"
                moved.append(row)
            else:
                keep.append(row)
        src.write_text("".join(json.dumps(r) + "\n" for r in keep))
        if moved:
            dst.parent.mkdir(parents=True, exist_ok=True)
            rows = []
            if dst.is_file():
                rows = [json.loads(l) for l in dst.read_text().splitlines()
                        if l.strip()]
            rows += moved
            rows.sort(key=lambda r: r.get("episode", ""))
            dst.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _belongs(mv: Move, key: str) -> bool:
    """Is `<date>/<episode>[_segNN]` this recording's? `episode_008` must not
    match `episode_0080`, so the character after the name is checked."""
    if not key.startswith(f"{mv.src_date}/{mv.src_episode}"):
        return False
    rest = key.split("/")[-1][len(mv.src_episode):]
    return not (rest and rest[0].isdigit())


def _retag(mv: Move, row: dict) -> dict:
    row = dict(row)
    for field in ("episode", "source_episode"):
        if field in row:
            ep = row[field].split("/")[-1]
            row[field] = f"{mv.dst_date}/{mv.rename(ep)}"
    if "date" in row:
        row["date"] = mv.dst_date
    if "task" in row:
        row["task"] = mv.dst_task
    return row


def _move_provenance(mv: Move, root: Path) -> None:
    """Carry the segment provenance and the dropped spans across.

    `segment_provenance.json` records WHY each segment exists and which spans
    were dropped. Left behind, the source task claims segments no longer in
    its tree -- and `pipeline_stages.coverage` reads `dropped_spans` to tell a
    deliberately-empty episode from an unprocessed one, so a stale entry there
    makes a recording look already handled in the task it left.
    """
    src = root / "release_cut" / mv.src_task / "segment_provenance.json"
    dst = root / "release_cut" / mv.dst_task / "segment_provenance.json"
    if not src.is_file():
        return
    a = json.loads(src.read_text())
    moved = {k: [r for r in a.get(k, []) if _belongs(mv, r.get("episode", ""))]
             for k in ("segments", "dropped_spans")}
    for k in ("segments", "dropped_spans"):
        a[k] = [r for r in a.get(k, []) if not _belongs(mv, r.get("episode", ""))]
    src.write_text(json.dumps(a, indent=2))
    if not any(moved.values()):
        return
    b = json.loads(dst.read_text()) if dst.is_file() else \
        {"summary": {}, "thresholds": a.get("thresholds", {}),
         "dropped_spans": [], "segments": []}
    for k in ("segments", "dropped_spans"):
        rows = b.get(k, []) + [_retag(mv, r) for r in moved[k]]
        rows.sort(key=lambda r: r.get("episode", ""))
        b[k] = rows
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(b, indent=2))


def apply_move(mv: Move, root: Path = DISK, *, episode_index: int,
               dry_run: bool = False) -> list[tuple[Path, Path]]:
    root = Path(root)
    pairs = plan_files(mv, root)
    taken = [d for _, d in pairs if d.exists()]
    if taken:
        raise FileExistsError(
            f"{len(taken)} destination path(s) already exist, first: {taken[0]} "
            f"-- pick an unused episode number rather than overwriting a "
            f"different recording")
    if dry_run:
        return pairs
    for s, d in pairs:
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(s), str(d))
    for _, d in pairs:
        if d.suffix == ".parquet":
            rewrite_parquet(d, mv, episode_index)
    _move_index_rows(mv, root)
    _move_provenance(mv, root)
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src", help="<task>/<date>/<episode>")
    ap.add_argument("dst", help="<task>/<date>/<episode>")
    ap.add_argument("--episode-index", type=int, required=True,
                    help="the destination task's next free episode_index")
    ap.add_argument("--root", default=str(DISK))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    mv = Move.parse(a.src, a.dst)
    pairs = apply_move(mv, Path(a.root), episode_index=a.episode_index,
                       dry_run=a.dry_run)
    head = "would move" if a.dry_run else "moved"
    for s, d in pairs:
        print(f"  {head} {s.relative_to(Path(a.root))} -> {d.relative_to(Path(a.root))}")
    print(f"{head}: {len(pairs)} path(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
