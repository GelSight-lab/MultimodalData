"""Prepare one recording session for publication, and refuse to publish it
until the folder is right.

The stages are the ones `PIPELINE_WORKFLOW.md` documents; this module runs
them in order for a single session, assembles the result into a folder shaped
like a published task, checks that folder with `twm.dataset_layout`, and only
then uploads. Doing it by hand for the 2026-09-09 session produced, in order:
a folder with no previews, an `episodes.jsonl` missing 29 of 32 rows, a force
export that refused because of it, and calibration files with nothing saying
which epoch they were. The check is what makes those loud.

    python -m twm.dataset_prep motherboard 2026-09-09 \\
        --episodes episode_000 episode_001 episode_002 \\
        --publish-as data/validation            # add --upload to actually push
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from twm.react_preprocess.meta import TASK_INDEX, task_index

REPO_ROOT = Path(__file__).resolve().parent.parent
TWM = REPO_ROOT / "twm"
RELEASE = Path("/media/yxma/Disk1/twm/release")
RELEASE_FORCE = Path("/media/yxma/Disk1/twm/release_force")
DEFAULT_REPO = "yxma/React"
INDEX_EPISODE_KEYS = {"bad_frames.json": "episodes", "segments.json": "segments"}


# ── stage 1: assemble ────────────────────────────────────────────────────────

def _filter_index(doc, key: str, session: str):
    """One task-level index, cut down to a single session.

    `bad_frames.json` keys its episodes by `<date>/<episode>`; `segments.json`
    is a list whose rows carry `source_episode`. The surrounding fields (the
    detector thresholds, the schema) travel with the subset: they are what
    makes the numbers readable.
    """
    out = dict(doc)
    value = doc.get(key)
    if isinstance(value, dict):
        out[key] = {k: v for k, v in value.items() if k.startswith(session + "/")}
    elif isinstance(value, list):
        out[key] = [r for r in value
                    if str(r.get("source_episode", "")).startswith(session + "/")]
    return out


def _stamp_index_columns(pq_path: Path, task: str, date: str, episode: str,
                         episode_index: int) -> None:
    """The LeRobot-style keys every published parquet carries.

    `react_preprocess.meta.add_index_columns` exists and is called from
    nowhere, so a freshly built parquet has 14 columns where a published one
    has 19. They are stamped here, at publication, rather than at build time:
    `episode_index` numbers an episode within the folder it ships in, and a
    per-episode build cannot know that.
    """
    import pyarrow.parquet as pq_mod

    from twm.react_preprocess.meta import add_index_columns
    table = pq_mod.read_table(pq_path)
    table = add_index_columns(table, task, task_index(task),
                              f"{date}/{episode}", episode_index)
    pq_mod.write_table(table, pq_path)


def assemble_session(stage, task: str, date: str, episodes: Sequence[str], *,
                     release: Path = None, release_force: Path = None,
                     calib_dir: Path = None) -> Path:
    """Copy one session out of the release trees into a task-shaped folder."""
    release = Path(release or RELEASE / task)
    release_force = Path(release_force or RELEASE_FORCE / task)
    stage = Path(stage)
    if stage.exists():
        shutil.rmtree(stage)
    (stage / "meta" / date).mkdir(parents=True)

    for ep in episodes:
        pq = release_force / "meta" / date / f"{ep}.parquet"
        if not pq.is_file():
            raise FileNotFoundError(
                f"{ep}: no force-exported parquet at {pq}. Run the force export "
                f"first — a session published without it has no newtons.")
        shutil.copy2(pq, stage / "meta" / date / pq.name)
        _stamp_index_columns(stage / "meta" / date / pq.name, task, date, ep,
                             episode_index=list(episodes).index(ep))
        side = pq.with_suffix("").with_suffix(".force.json")
        if side.is_file():
            shutil.copy2(side, stage / "meta" / date / side.name)
        for kind in ("videos", "depth"):
            src = release / kind / date / ep
            if src.is_dir():
                shutil.copytree(src, stage / kind / date / ep)
        preview = release / "previews" / date / f"{ep}.mp4"
        if preview.is_file():
            (stage / "previews" / date).mkdir(parents=True, exist_ok=True)
            shutil.copy2(preview, stage / "previews" / date / preview.name)

    wanted = {f"{date}/{ep}" for ep in episodes}
    jsonl = release / "episodes.jsonl"
    if jsonl.is_file():
        rows = [json.loads(l) for l in jsonl.read_text().splitlines() if l.strip()]
        keep = [r for r in rows if r.get("episode") in wanted]
        (stage / "episodes.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in keep))
    for name, key in INDEX_EPISODE_KEYS.items():
        src = release / name
        if src.is_file():
            (stage / name).write_text(json.dumps(
                _filter_index(json.loads(src.read_text()), key, date), indent=1))

    if calib_dir is not None:
        shutil.copytree(Path(calib_dir), stage / "calibration", dirs_exist_ok=True)
    return stage


def write_splits(stage: Path) -> Path:
    """The session's own train/test split, from its own indices."""
    out = subprocess.run([sys.executable, str(TWM / "scripts" / "build_splits.py"),
                          "--root", str(stage)], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"build_splits failed: {out.stderr.strip()[:400]}")
    return stage / "splits.json"


def write_calibration_doc(stage: Path, task: str, date: str, epoch: str,
                          note: str = "") -> Path:
    """The file that says which epoch `calibration/` holds. Without it a
    folder's extrinsics are anonymous, which is how a session ends up
    published under another session's calibration."""
    cal = stage / "calibration"
    cams = {}
    for name in ("left", "middle", "right"):
        p = cal / f"T_mocap_to_cam_{name}.json"
        if p.is_file():
            d = json.loads(p.read_text())
            cams[name] = {"camera_serial": d.get("camera_serial"),
                          "rmse": d.get("rmse_px") or d.get("rmse_mm"),
                          "created_at": d.get("created_at")}
    doc = {"task": task, "calibration_id": epoch, "created": epoch,
           "method": "PnP", "applies_to_dates": [date], "rmse_unit": "px",
           "cameras": cams,
           "projection_chain": "sensor_pose (mocap mm) -> T_mocap_to_cam -> camera "
                               "intrinsics -> pixel; gel center via T_gel_to_rigid",
           "note": note or f"Calibration epoch {epoch}, declared for {task}/{date} "
                           f"in twm/calib_epoch.py CALIB_SESSIONS."}
    path = cal / "calibration.json"
    path.write_text(json.dumps(doc, indent=1))
    return path


# ── the stages that call the existing tools ──────────────────────────────────

def _run(cmd: List[str], cwd: Path) -> None:
    print(f"$ {' '.join(str(c) for c in cmd)}", flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=str(cwd))
    if r.returncode != 0:
        raise RuntimeError(f"stage failed ({r.returncode}): {' '.join(str(c) for c in cmd)}")


def run_stages(task: str, date: str, episodes: Sequence[str], *, workers: int = 2,
               skip: Sequence[str] = ()) -> None:
    if "build" not in skip:
        _run([sys.executable, "-m", "react_preprocess", "build", "--task", task,
              "--date", date, "--with-depth", "--episodes", *episodes], TWM)
    if "force" not in skip:
        procs = [subprocess.Popen([sys.executable, "-m", "force_recovery.batch_worker",
                                   str(i), str(workers)], cwd=str(TWM))
                 for i in range(workers)]
        if any(p.wait() for p in procs):
            raise RuntimeError("force estimation failed")
    if "export" not in skip:
        _run([sys.executable, "-m", "force_recovery.export_force_columns", "export"], TWM)
    if "previews" not in skip:
        _run([sys.executable, str(TWM / "scripts" / "build_episode_previews.py"),
              "--task", task, "--date", date, "--episodes", *episodes], TWM)


def upload(stage: Path, repo_id: str, prefix: str) -> None:
    from huggingface_hub import HfApi
    HfApi().upload_folder(folder_path=str(stage), path_in_repo=prefix.rstrip("/"),
                          repo_id=repo_id, repo_type="dataset",
                          commit_message=f"{prefix}: prepared session")


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    from twm.calib_epoch import calib_dir as epoch_dir
    from twm.calib_epoch import session_epoch
    from twm.dataset_layout import check_layout

    p = argparse.ArgumentParser(prog="python -m twm.dataset_prep",
                                description="Prepare and publish one recording session.")
    p.add_argument("task")
    p.add_argument("date")
    p.add_argument("--episodes", nargs="+", required=True)
    p.add_argument("--publish-as", default=None,
                   help="repo prefix, e.g. data/validation (default: data/<task>)")
    p.add_argument("--repo", default=DEFAULT_REPO)
    p.add_argument("--stage-dir", default=None)
    p.add_argument("--skip", nargs="*", default=(),
                   choices=("build", "force", "export", "previews"),
                   help="stages already done")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--upload", action="store_true",
                   help="push after the layout check passes (default: stop before)")
    a = p.parse_args(argv)

    prefix = a.publish_as or f"data/{a.task}"
    stage = Path(a.stage_dir or (Path.home() / ".cache" / "twm_prep" / f"{a.task}_{a.date}"))
    epoch = session_epoch(a.task, a.date)      # raises if the session is undeclared
    print(f"session {a.task}/{a.date}: calibration epoch {epoch}")

    run_stages(a.task, a.date, a.episodes, workers=a.workers, skip=a.skip)
    assemble_session(stage, a.task, a.date, a.episodes,
                     calib_dir=epoch_dir(a.task, date=a.date))
    write_calibration_doc(stage, a.task, a.date, epoch)
    write_splits(stage)

    report = check_layout(stage, a.date)
    print(report.table())
    if not report.ok:
        print("\nrefusing to upload a folder that does not match the layout")
        return 1
    if not a.upload:
        print(f"\nstaged at {stage} — re-run with --upload to push to "
              f"{a.repo}:{prefix}")
        return 0
    upload(stage, a.repo, prefix)
    print(f"uploaded to {a.repo}:{prefix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
