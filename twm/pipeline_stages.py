"""The release pipeline as one ordered chain, with its dependencies declared.

`dataset_prep.run_stages` covered four of the eight stages a release needs:
build, force, export, previews. The other four — curate, the Z-up conversion,
the cut, publish — lived in `react_preprocess.__main__`, in a standalone
script, and in the publisher. Nothing knew the order, and nothing knew what a
stage required before it could run.

What that cost on the 2026-09 release, in the order the failures surfaced:

  * `build` ran without ``--with-depth``; found ten hours later, at the layout
    check, and every episode had to be rebuilt;
  * `force` was never run at all; found four stages later when the publish
    gate refused, because `export` will not invent a force column;
  * `zup` was rebuilt from the 29 episodes that existed at the time while the
    tree had grown to 72, and `segment` then cut the stale half without
    complaint.

Every one of those is a missing prerequisite that only announced itself when
something downstream broke. A list of commands cannot catch them. A stage that
declares what it needs, and a planner that refuses to start it when the
prerequisite has produced nothing, can.

    from twm import pipeline_stages as PS
    for stage in PS.plan(skip={"build"}, until="segment"):
        stage.run(task="pushT", date="2026-09-12")
"""
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

TWM = Path(__file__).resolve().parent
REPO = TWM.parent
DATA_ROOT = Path("/media/yxma/Disk1/twm/data")
RELEASE = Path("/media/yxma/Disk1/twm/release")
RELEASE_ZUP = Path("/media/yxma/Disk1/twm/release_zup")
RELEASE_CUT = Path("/media/yxma/Disk1/twm/release_cut")
FORCE_ROOT = Path("/media/yxma/Disk1/twm/force_recovery")

# Only sessions from here on are in scope. The 2026-05/06 sessions are already
# published, their source H5 was deleted on 2026-09-09, and they predate the
# wrist cameras — so every gate has to argue about them (uncertifiable
# alignment, missing streams) for no gain, since re-cutting cannot improve data
# that is already on the Hub. Operator's decision, 2026-09-14.
# main carries ONE WEEK. 2026-09-09 is excluded as well: it was recorded with
# a different wrist camera and was published separately as `validation`. The
# May/June sessions and that epoch belong on an old-data branch, not here.
SCOPE_SINCE = "2026-09-10"

# Every task with a published task_index. This was a default argument buried in
# `run_all`'s signature, which is not a list anything can check -- and rope,
# never being in it, was published by hand and reached the Hub with no
# bad_frames.json, no segments.json and no splits.json. The last of those made
# every rope segment read as TRAIN, because `_split_filter` treats an unknown
# key that way.
TASKS = ("motherboard", "pushT", "rope", "toy")


@dataclass(frozen=True)
class Stage:
    """One step of the chain.

    `needs` is the stage that must have produced something first; `produced`
    answers "did it?" for a given task, so a plan can refuse to start a stage
    whose input is not there rather than discovering it three stages later.
    """
    name: str
    what: str
    # A tuple when a stage genuinely has more than one prerequisite.
    # `export` writes force columns (needs the npz) AND stamps a
    # world-frame declaration read from episodes.jsonl (needs curate).
    # Expressing only one of those is how the missing force estimation
    # surfaced four stages later, at the publish gate.
    needs: str | tuple[str, ...] | None = None
    produced: Callable[[str], bool] = field(default=lambda task: True, repr=False)
    argv: Callable[..., list[list[str]]] = field(default=lambda **kw: [], repr=False)
    # Assets the stage needs that NO stage produces. Callables, because the
    # roots are monkeypatched in tests and must be read at call time.
    assets: tuple = ()

    def commands(self, **kw) -> list[list[str]]:
        return self.argv(**kw)

    def run(self, *, cwd: Path = REPO, check: bool = True, **kw) -> int:
        for cmd in self.commands(**kw):
            r = subprocess.run(cmd, cwd=str(cwd))
            if r.returncode and check:
                return r.returncode
        return 0


def _has(root: Path, task: str, pattern: str) -> bool:
    d = root / task
    return d.is_dir() and any(d.rglob(pattern))


def _built(task):        return _has(RELEASE, task, "episode_*.parquet")
def _forced(task):       return _has(FORCE_ROOT, task, "*.npz")
def _exported(task):     return _has(RELEASE, task, "episode_*.parquet")
def _curated(task):      return (RELEASE / task / "bad_frames.json").is_file()
def _zupped(task):       return _has(RELEASE_ZUP, task, "episode_*.parquet")
def _segmented(task):    return _has(RELEASE_CUT, task, "*_seg*.parquet")


def _build(task, date=None, episodes=(), **_):
    """Build the recordings that have no release episode yet.

    `date` was REQUIRED and `run_all` passes none, so the scheduled path died
    on TypeError every time while the manual path -- a date typed by an
    operator -- worked. The stage was in the plan and had never run.

    With no date it asks `coverage` what is missing and builds exactly that,
    one command per date. Rebuilding an episode that already has a parquet
    costs hours and produces the same bytes.

    `twm.react_preprocess`, not `react_preprocess`: the latter only imports
    with cwd=twm/, which is how the old run_stages invoked it. Every other
    stage here runs from the repo root, and a module path that resolves in one
    and not the other is how a stage dies on its first line.
    """
    if date:
        wanted = {date: list(episodes)}
    else:
        wanted = {}
        for key in coverage("build", task).missing:
            d, ep = key.split("/", 1)
            wanted.setdefault(d, []).append(ep)
    out = []
    for d, eps in sorted(wanted.items()):
        ep = ["--episodes", *eps] if eps else []
        out.append([sys.executable, "-m", "twm.react_preprocess", "build",
                    # No --with-depth: the operator turned the depth channel
                    # off on 2026-09-16. It is the largest single cost in this
                    # stage (a second lossless FFV1 stream per camera). The
                    # flag still exists — leaving it off by ACCIDENT is what
                    # cost ten hours once — it is simply not asked for.
                    "--task", task, "--date", d, *ep])
    return out


def _force(task=None, workers: int = 2, **_):
    # `twm.force_recovery`, not `force_recovery` — the same fix `_build` carries.
    # The bare name resolves only with cwd=twm/, and every stage here runs from
    # the repo root, so this died with ModuleNotFoundError on its first line
    # while the three hours behind it went unused.
    return [[sys.executable, "-m", "twm.force_recovery.batch_worker",
             str(i), str(workers)] for i in range(workers)]


def _export(**_):
    return [[sys.executable, "-m", "twm.force_recovery.export_force_columns", "export"]]


def _curate(task, **_):
    return [[sys.executable, "-m", "twm.react_preprocess", "curate", "--task", task]]


def _zup(task, **_):
    return [[sys.executable, str(TWM / "scripts" / "convert_release_zup.py"),
             "--task", task]]


def _segment(task, **_):
    return [[sys.executable, "-m", "twm.react_preprocess", "segment", "--task", task,
             "--src", str(RELEASE_ZUP), "--out", str(RELEASE_CUT),
             "--detect-root", str(RELEASE)]]


def _index(task, **_):
    """The CUT tree's own indices. Cutting creates new publishing units, and
    an episode in episodes.jsonl but absent from splits.json is read as TRAIN
    by `ReactVideoDataset._split_filter` — a silent training leak."""
    return [[sys.executable, "-m", "twm.react_preprocess", "curate",
             "--task", task, "--root", str(RELEASE_CUT)],
            [sys.executable, str(TWM / "scripts" / "build_splits.py"),
             "--root", str(RELEASE_CUT / task), "--seed", "0"]]


def _indexed(task):
    return (RELEASE_CUT / task / "splits.json").is_file()


def _verify(**_):
    # The CUT tree — the one `publish` uploads. Certifying the uncut tree and
    # shipping the cut one means every check answered about files nobody gets.
    return [[sys.executable, "-m", "twm.pipeline_guard"],
            [sys.executable, str(TWM / "scripts" / "certify_release.py"),
             "--src", str(RELEASE_CUT), "--since", SCOPE_SINCE]]


def _publish(**_):
    # The CUT tree: the Hub holds segments, and uploading the uncut tree
    # beside them would put `episode_001` and `episode_001_seg00` in one
    # folder with the same frames counted twice.
    return [[sys.executable, str(TWM / "scripts" / "build_release_publish.py"),
             "--src", str(RELEASE_CUT), "--no_delete"]]


STAGES: tuple[Stage, ...] = (
    Stage("build", "source H5 -> videos + parquet (with depth)",
          None, _built, _build),
    Stage("force", "per-row normal force from the tactile frames",
          "build", _forced, _force,
          assets=((lambda: FORCE_ROOT / "feature_cache" / "glowtact_round_mm.json",
                   "the fitted-features cache measured in the calibration "
                   "experiment — no stage builds it and it is not in git; "
                   "restore it from the data disk"),
                  (lambda: FORCE_ROOT / "feature_cache" / "glowtact_round_8_15_di4.json",
                   "the v8 measured 8-15 N calibration cache; restore the "
                   "reviewed asset from the data disk"),
                  (lambda: FORCE_ROOT / "lut_calibration" / "glowtact_lut.npz",
                   "the LUT used for the geometry columns written with force; "
                   "restore it from the data disk"))),
    Stage("curate", "bad_frames / segments / episodes indices",
          "build", _curated, _curate),
    # The cut must come after force recovery AND the frame conversion, so that
    # every column those added is carried through by the same row slice; see
    # `react_preprocess.segment`.
    Stage("export", "force columns into the published parquet",
          ("force", "curate"), _exported, _export,
          assets=((lambda: FORCE_ROOT / "lut_calibration" / "glowtact_lut.npz",
                   "the depth lookup table the force reconstruction reads — "
                   "no stage builds it and it is not in git; restore it from "
                   "the data disk"),)),
    Stage("zup", "rotate the release from the recorded Y-up to Z-up",
          "curate", _zupped, _zup),
    Stage("segment", "cut each episode down to its publishable spans",
          "zup", _segmented, _segment),
    Stage("index", "the cut tree's own bad_frames / segments / splits",
          "segment", _indexed, _index),
    Stage("verify", "pipeline invariants + release certification",
          "index", lambda task: True, _verify),
    Stage("publish", "upload to the Hub (runs the gates again itself)",
          "verify", lambda task: True, _publish),
)

BY_NAME = {s.name: s for s in STAGES}


def plan(skip: Sequence[str] = (), until: str | None = None) -> list[Stage]:
    """The stages to run, in order, minus `skip`, stopping after `until`.

    An unknown name raises rather than being ignored: a typo in `--skip` that
    silently did nothing would run a stage the operator believed was skipped.
    """
    # export exists to write the force columns; without the npz it fails on
    # its first episode, after everything ahead of it has run. Skipping force
    # therefore skips it too — the operator paused force estimation because a
    # new algorithm is coming, not to salvage half of it.
    skip = set(skip)
    if "force" in skip:
        skip.add("export")
    unknown = set(skip) - set(BY_NAME)
    if unknown:
        raise KeyError(f"unknown stage(s): {', '.join(sorted(unknown))}; "
                       f"known: {', '.join(BY_NAME)}")
    if until is not None and until not in BY_NAME:
        raise KeyError(f"unknown stage: {until}")
    out = []
    for s in STAGES:
        if s.name not in skip:
            out.append(s)
        if until is not None and s.name == until:
            break
    return out


@dataclass(frozen=True)
class Coverage:
    """How much of a stage's input it has actually processed."""
    stage: str
    done: int
    total: int
    missing: list[str]

    @property
    def complete(self) -> bool:
        return self.total > 0 and self.done >= self.total


def _recordings_in(root: Path, task: str, since: str | None = None) -> set[str]:
    """The RAW recordings of a task, as `date/episode`.

    `build`'s source is the recorder's output, not the release tree. Taking it
    from the release tree made the denominator the output: `done == total`
    always, and a recording that had never been built was not missing, it was
    invisible -- rope/2026-09-14 read 11/11 with three of its six recordings
    untouched.
    """
    d = root / task
    if not d.is_dir():
        return set()
    return {f"{p.parent.name}/{p.stem}"
            for p in d.glob("*/episode_*.h5")
            if not since or p.parent.name >= since}


def _episodes_in(root: Path, task: str) -> set[str]:
    d = root / task / "meta"
    if not d.is_dir():
        return set()
    # date/episode, not episode: the recorder restarts numbering every day, so
    # `episode_000` exists under many dates. Keying on the stem alone collapsed
    # 47 motherboard files into 18 names, which reads as a two-thirds-done
    # stage — and would hide a genuinely two-thirds-done one just as well.
    return {f"{p.parent.name}/{p.stem.rsplit('_seg', 1)[0]}"
            for p in d.rglob("episode_*.parquet")
            if p.parent.name >= SCOPE_SINCE}


def coverage(stage_name: str, task: str) -> Coverage:
    """Which of the source episodes this stage has produced output for.

    "Produced something" is not "produced everything". `_zupped` returned True
    on finding ONE parquet, so a Z-up tree built from 29 of 72 episodes read as
    done and the cut ran on the stale half. Counted, not sampled.
    """
    # `build` turns recordings into release episodes, so its source is the
    # recorder's tree. Every later stage consumes what build produced.
    src = (_recordings_in(DATA_ROOT, task, SCOPE_SINCE) if stage_name == "build"
           else _episodes_in(RELEASE, task))
    out_root = {"zup": RELEASE_ZUP, "segment": RELEASE_CUT,
                "index": RELEASE_CUT}.get(stage_name, RELEASE)
    out = _episodes_in(out_root, task)
    missing = set(src) - set(out)
    if stage_name == "segment" and missing:
        # An episode the cut deliberately dropped produced nothing BY DESIGN —
        # 110 frames cannot yield a 20 s span. That is a result, not an
        # omission. The cut already writes what it dropped; this reads that
        # record instead of re-deriving the rule.
        missing -= _deliberately_dropped(task)
    missing = sorted(missing)
    return Coverage(stage_name, len(src) - len(missing), len(src), missing)


def _deliberately_dropped(task: str) -> set[str]:
    """Episodes the cut examined and published nothing from, per its own log."""
    import json
    p = RELEASE_CUT / task / "segment_provenance.json"
    if not p.is_file():
        return set()
    try:
        doc = json.loads(p.read_text())
    except (OSError, ValueError):
        return set()
    return {str(d.get("episode", "")) for d in doc.get("dropped_spans", [])}


def blocked(stage: Stage, task: str) -> str | None:
    """Why `stage` cannot start for `task`, or None.

    Checks the PREREQUISITE's output, not the stage's own — the point is to
    refuse before doing work, not to discover it afterwards.
    """
    # Assets the stage needs that NO stage produces. The force estimator reads
    # a fitted-features cache measured in an August calibration experiment; it
    # is not in git and no stage builds it. Without this check the chain dies
    # four stages in, AFTER the build has spent hours, with a message that
    # sends the reader to `build`.
    for asset, what in stage.assets:
        if not asset().exists():
            return (f"{stage.name} needs {asset()}, which no stage produces — "
                    f"{what}")
    if stage.needs is None:
        return None
    names = (stage.needs,) if isinstance(stage.needs, str) else stage.needs
    for name in names:
        prereq = BY_NAME[name]
        if not prereq.produced(task):
            return (f"{stage.name} needs {prereq.name} ({prereq.what}), which "
                    f"has produced nothing for {task}")
        # Partial output is the dangerous case: it looks done and is not.
        if prereq.name in ("zup", "segment"):
            cov = coverage(prereq.name, task)
            if not cov.complete:
                head = ", ".join(cov.missing[:3])
                more = "" if len(cov.missing) <= 3 else f" and {len(cov.missing)-3} more"
                return (f"{stage.name} needs {prereq.name}, which covers only "
                        f"{cov.done}/{cov.total} of {task} — missing {head}{more}")
    return None


def status(task: str) -> list[tuple[str, bool]]:
    """(stage, has produced something) for every stage, for one task."""
    return [(s.name, bool(s.produced(task))) for s in STAGES]


def run_all(tasks: Sequence[str] = TASKS,
            skip: Sequence[str] = (), until: str | None = None,
            log=print, **kw) -> int:
    """Walk the plan for each task. Returns 0, or the first non-zero exit.

    A stage graph nobody runs is a diagram. This is the part that runs it, and
    it is the only place that decides what to do when a stage refuses: stop.
    Carrying a failure downstream is how a half-built Z-up tree got cut and a
    force-less parquet reached the publish gate.

    `publish` uploads every task in one commit, so it runs once, after the
    per-task stages are through.
    """
    plan_ = plan(skip=skip, until=until)
    per_task = [s for s in plan_ if s.name != "publish"]
    # STAGE-major, not task-major. `verify` certifies every task in one run, so
    # running one task's whole chain first put verify ahead of the other task's
    # index — and it died on a bad_frames.json no stage had yet been given the
    # chance to write.
    for stage in per_task:
        if stage.name == "verify":
            log(f"\n### verify (all tasks) — {stage.what}")
            rc = stage.run(task=tasks[0], **kw)
            log(f"### verify exit {rc}")
            if rc:
                return rc
            continue
        for task in tasks:
            why = blocked(stage, task)
            if why:
                log(f"[阻塞] {task}/{stage.name}: {why}")
                return 1
            log(f"\n### {task} / {stage.name} — {stage.what}")
            rc = stage.run(task=task, **kw)
            log(f"### {task}/{stage.name} exit {rc}")
            if rc:
                return rc
    if any(s.name == "publish" for s in plan_):
        stage = BY_NAME["publish"]
        log(f"\n### publish — {stage.what}")
        rc = stage.run(task=tasks[0], **kw)
        log(f"### publish exit {rc}")
        return rc
    return 0
