"""Publish the force-overlay episode previews to the React dataset repo.

Target: `data/<task>/previews/<date>/<episode>.mp4` in `yxma/React`.

EVERY task, discovered from disk — not a hard-coded one. The first version
pointed at `motherboard` alone, so when the overlay and the newtons both
changed, pushT's four clips stayed on the dataset showing the old tile dot and
force values that had been declared void. Nothing reported it: the uploader
was doing exactly what it was written to do, and its manifest said "the clips"
while describing a subset. A publishing tool whose scope is narrower than the
thing it publishes will keep something stale every single time.

The one thing a reader must not be misled about: some clips come from dates
with **no force estimation**, so they carry no disc. That is absence of input,
not a rendering failure, and the manifest says so per clip rather than leaving
someone to wonder why some previews have a disc and others do not.

Run: python -m force_recovery.upload_previews [--dry-run]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from react_preprocess.config import STAGE_ROOT

FORCE_ROOT = Path("/media/yxma/Disk1/twm/force_recovery")
REPO = "yxma/React"

# Where the renderer writes, asked of the renderer — not restated here.
# This module used to name its own directory, `figures/episode_previews`, while
# `react_preprocess.previews.plan()` wrote to `<STAGE_ROOT>/<task>/previews`.
# Both held 36 clips with identical names, so a full re-render produced fresh
# output in one tree and left the other untouched, and the uploader published
# the untouched one. The freshness gate below is what caught it — every clip
# reported stale on a run that had just re-rendered every clip. Had the gate
# not existed, this would have shipped a silent no-op.
PREVIEW_ROOT = STAGE_ROOT


def clip_dir(task: str) -> Path:
    """The directory `previews.plan()` renders `task` into."""
    return STAGE_ROOT / task / "previews"


def tasks() -> list[str]:
    """Tasks that have rendered previews, from disk — never a literal list."""
    return sorted(d.name for d in STAGE_ROOT.iterdir()
                  if d.is_dir() and any(clip_dir(d.name).rglob("*.mp4")))


def clip_paths() -> list[tuple[str, Path]]:
    return [(t, p) for t in tasks()
            for p in sorted(clip_dir(t).rglob("*.mp4"))]


def check_decodable(paths: list[Path]) -> list[str]:
    """Every clip must decode to its LAST frame before any of them ship.

    A preview run interrupted mid-encode leaves a file that is the right size
    on disk and unplayable past the truncation point. `ls` cannot see it.

    Two checks, because the obvious one does not work. `ffmpeg -f null -`
    prints "partial file" / "Invalid NAL unit size" for a truncated clip but
    **exits 0**, so a gate reading the return code passes a file that decodes
    79 of its 900 frames — mine did, until a deliberately truncated copy was
    fed to it. So:

      1. under `-v error` a healthy file prints NOTHING; any stderr is a fault;
      2. frames actually decoded must equal the frames the container declares.

    (2) is the one that cannot be argued with: the moov survives truncation
    and keeps advertising 900.
    """
    import subprocess

    def _probe(p: Path, *entries: str, count: bool = False) -> str:
        cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0"]
        if count:
            cmd.append("-count_frames")
        cmd += ["-show_entries", f"stream={','.join(entries)}",
                "-of", "csv=p=0", str(p)]
        return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()

    bad = []
    for p in paths:
        r = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(p), "-f", "null", "-"],
            capture_output=True, text=True)
        if r.stderr.strip():
            bad.append(f"{p.parent.name}/{p.name}: "
                       f"{r.stderr.strip().splitlines()[0][:90]}")
            continue
        declared = _probe(p, "nb_frames")
        decoded = _probe(p, "nb_read_frames", count=True)
        if not declared.isdigit() or declared != decoded:
            bad.append(f"{p.parent.name}/{p.name}: decoded {decoded or '?'} "
                       f"of {declared or '?'} declared frames")
    return bad


def check_mirrors_release(pairs: list[tuple[str, Path]]) -> tuple[list[str], list[str]]:
    """Previews must mirror the release: same episodes, nothing more.

    Returns (orphans, missing). An orphan — a clip for an episode the release
    does not publish — is REFUSED: it asserts data that does not exist, and it
    can never be refreshed because there is nothing to refresh it from. Four
    such clips (2026-03-23 x3, 2026-05-15 x1) survived every previous upload
    and were reported by a reader as "not updated". A missing episode is only
    WARNED: a gap is visible, an orphan misleads.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from twm.calib_epoch import release_episodes

    orphans, missing = [], []
    for task in sorted({t for t, _ in pairs}):
        rel = release_episodes(task)
        if not rel:                    # no release tree on this machine
            continue
        have = {f"{p.parent.name}/{p.stem}" for t, p in pairs if t == task}
        orphans += [f"{task}/{k}" for k in sorted(have - rel)]
        missing += [f"{task}/{k}" for k in sorted(rel - have)]
    return orphans, missing


def check_fresh(paths: list[Path]) -> list[str]:
    """Every clip must be newer than the code that renders it.

    The calibration-epoch fix went in at 21:36; 31 of 32 motherboard clips on
    disk were rendered 20:43-21:06 with the wrong extrinsics, and nothing
    distinguished them from good output — same names, same sizes, decode
    cleanly. mtime-vs-source is a blunt instrument, but it errs toward
    re-rendering ~40 clips, and it would have caught both of today's reports.
    """
    repo = Path(__file__).resolve().parents[1]
    srcs = [repo / "scripts" / "build_episode_previews.py", repo / "viz.py",
            repo / "calib_epoch.py", repo / "force_overlay.py",
            repo / "tactile_align.py"]
    # ... and the curation the FLAGGED banners are drawn from: re-curating
    # changes what a correct preview looks like just as surely as a code edit.
    srcs += list(Path("/media/yxma/Disk1/twm/release").glob("*/bad_frames.json"))
    cut = max(s.stat().st_mtime for s in srcs if s.exists())
    stale = [p for p in paths if p.stat().st_mtime < cut]
    return [f"{p.parent.parent.name}/{p.parent.name}/{p.name}: rendered "
            f"before the newest renderer source — re-render it" for p in stale]


def manifest() -> dict:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from twm.force_overlay import ALPHA, F_FULL_N, R_MAX_PX, R_MIN_PX

    import numpy as np

    clips, calibs = [], set()
    for task, p in clip_paths():
        date, name = p.parent.name, p.stem
        npzs = [FORCE_ROOT / task / date / f"{name}_{s}.npz"
                for s in ("left", "right")]
        present = [q for q in npzs if q.exists()]
        # Read the calibration out of the npz the clip was drawn from, rather
        # than naming it here. A description of the force that lives beside
        # the video instead of inside the artifact is the same mistake that
        # let the site advertise "LUT v2, GlowTact-calibrated" for weeks after
        # that map had been replaced.
        for q in present:
            with np.load(q, allow_pickle=True) as d:
                if "force_calibration" in d:
                    calibs.add(str(d["force_calibration"]))
        clips.append({"path": f"{task}/{date}/{p.name}", "task": task,
                      "date": date, "episode": name,
                      "force_overlay": bool(present),
                      "size_bytes": p.stat().st_size})
    with_dot = sum(c["force_overlay"] for c in clips)
    return {
        "what": "Episode preview clips: 3 RealSense views with OptiTrack "
                "projection, both GelSight tiles, and a semi-transparent disc "
                "whose AREA is linear in estimated normal force.",
        "force_overlay": {
            "drawn_on": "the three camera views, centred on that sensor's "
                        "OptiTrack pose projected into each view — the same "
                        "projection that draws the pose axes, so the disc and "
                        "the axes cannot disagree about where the sensor is",
            "encoding": "area proportional to force; radius = "
                        f"{R_MIN_PX} + {R_MAX_PX - R_MIN_PX} * sqrt(F / "
                        f"{F_FULL_N} N), saturating at {F_FULL_N} N",
            "alpha": ALPHA,
            "legend_in_frame": "0.5 / 2 / 8 N, drawn with the same law",
            "no_dot_below_n": 0.02,
            "note": "Force is estimated from the GelSight images alone "
                    "(no force sensor on this rig); see "
                    "https://huggingface.co/spaces/yxma/react-force-recovery",
            "calibration": sorted(calibs) or ["unknown"],
            "supersedes": "Clips uploaded before 2026-08-07 carry (a) a dot "
                          "pinned to the GelSight tile corner and (b) forces "
                          "from a calibration that mixed pixel-unit weights "
                          "with mm-unit features (end-to-end rho 0.143). "
                          "Motherboard clips additionally used pushT's "
                          "June-26 camera extrinsics instead of the May-12 "
                          "epoch (35-73 px projection error), and 2026-05-19 "
                          "missed its (0.23, 0, 0.175) m world-frame offset. "
                          "All replaced; each frame's status bar now names "
                          "the epoch and offset it was projected with.",
        },
        "tasks": tasks(),
        "clips_total": len(clips),
        "clips_with_force_overlay": with_dot,
        "clips_without_force_overlay": len(clips) - with_dot,
        "why_some_have_no_dot": sorted({c["date"] for c in clips
                                        if not c["force_overlay"]}) and
            (", ".join(sorted({c["date"] for c in clips
                               if not c["force_overlay"]}))
             + " have no force estimation, so those clips show no disc. "
               "Absence of input, not a rendering failure.")
            or "every clip has force data",
        "clips": clips,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    m = manifest()
    print(f"{m['clips_total']} clips, {m['clips_with_force_overlay']} with a "
          f"force dot, {m['clips_without_force_overlay']} without")
    total = sum(c["size_bytes"] for c in m["clips"]) / 1e6
    print(f"tasks: {', '.join(m['tasks'])}")
    print(f"total {total:.0f} MB -> {REPO}:data/<task>/previews/")

    pairs = clip_paths()
    paths = [p for _, p in pairs]

    orphans, missing = check_mirrors_release(pairs)
    if orphans:
        print(f"REFUSING TO UPLOAD: {len(orphans)} clip(s) for episodes the "
              f"release does not publish — delete them, they mislead:")
        for o in orphans:
            print("   ", o)
        raise SystemExit(1)
    for k in missing:
        print(f"WARN: release episode {k} has no preview clip")

    stale = check_fresh(paths)
    if stale:
        print(f"REFUSING TO UPLOAD: {len(stale)} clip(s) older than the "
              f"renderer sources:")
        for s in stale:
            print("   ", s)
        raise SystemExit(1)
    print("all clips mirror the release and post-date the renderer")

    print(f"decoding all {len(paths)} clips ...", flush=True)
    bad = check_decodable(paths)
    if bad:
        print(f"REFUSING TO UPLOAD: {len(bad)} unplayable clip(s)")
        for b in bad:
            print("   ", b)
        raise SystemExit(1)
    print("all clips decode")

    if args.dry_run:
        for c in m["clips"][:5]:
            print("   ", c["path"], "dot" if c["force_overlay"] else "NO DOT")
        return

    from huggingface_hub import HfApi
    api = HfApi()
    mf = STAGE_ROOT / "previews_manifest.json"
    mf.write_text(json.dumps(m, indent=1))
    api.upload_file(path_or_fileobj=str(mf), repo_id=REPO, repo_type="dataset",
                    path_in_repo="data/previews_manifest.json")
    for task in m["tasks"]:
        # clip_dir(task), NOT the task root: the task root also holds the
        # release's own `videos/`, and `allow_patterns=["*.mp4"]` would have
        # published every raw episode video into the previews folder.
        api.upload_folder(
            folder_path=str(clip_dir(task)), repo_id=REPO,
            repo_type="dataset", path_in_repo=f"data/{task}/previews",
            allow_patterns=["*.mp4"],
            commit_message=(
                f"previews [{task}]: re-rendered against the task's own "
                "calibration epoch and the current curation — FLAGGED frames "
                "are outlined and named, and the clip window starts at the "
                "episode's release trim offset rather than at H5 frame 0"))
    print("uploaded")


if __name__ == "__main__":
    main()
