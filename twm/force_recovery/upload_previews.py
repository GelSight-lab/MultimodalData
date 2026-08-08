"""Publish the force-overlay episode previews to the React dataset repo.

Target: `data/motherboard/previews/<date>/<episode>.mp4` in `yxma/React`,
which already holds 32 previews from before the force overlay existed — this
replaces them and adds 4 more.

The one thing a reader must not be misled about: 4 of the 36 clips
(2026-03-23 x3, 2026-05-15 x1) come from dates with **no force estimation**,
so they carry no dot. That is absence of input, not a rendering failure, and
the manifest written alongside says so per clip rather than leaving someone to
wonder why some previews have a dot and others do not.

Run: python -m force_recovery.upload_previews [--dry-run]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PREVIEWS = Path("/media/yxma/Disk1/twm/figures/episode_previews/motherboard")
FORCE = Path("/media/yxma/Disk1/twm/force_recovery/motherboard")
REPO = "yxma/React"
PREFIX = "data/motherboard/previews"


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


def manifest() -> dict:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from twm.force_overlay import ALPHA, F_FULL_N, R_MAX_PX, R_MIN_PX

    import numpy as np

    clips, calibs = [], set()
    for p in sorted(PREVIEWS.rglob("*.mp4")):
        date, name = p.parent.name, p.stem
        npzs = [FORCE / date / f"{name}_{s}.npz" for s in ("left", "right")]
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
        clips.append({"path": f"{date}/{p.name}", "date": date,
                      "episode": name, "force_overlay": bool(present),
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
                          "with mm-unit features (end-to-end rho 0.143). Both "
                          "are replaced; the old newton values are void.",
        },
        "clips_total": len(clips),
        "clips_with_force_overlay": with_dot,
        "clips_without_force_overlay": len(clips) - with_dot,
        "why_some_have_no_dot": "2026-03-23 and 2026-05-15 have no force "
                                "estimation, so those clips show no dot. "
                                "Absence of input, not a rendering failure.",
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
    print(f"total {total:.0f} MB -> {REPO}:{PREFIX}/")

    paths = sorted(PREVIEWS.rglob("*.mp4"))
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
    mf = PREVIEWS / "previews_manifest.json"
    mf.write_text(json.dumps(m, indent=1))
    api.upload_folder(
        folder_path=str(PREVIEWS), repo_id=REPO, repo_type="dataset",
        path_in_repo=PREFIX, allow_patterns=["*.mp4", "*.json"],
        commit_message="previews: force disc moved onto the camera views at "
                       "the sensor's projected position (was pinned to the "
                       "GelSight tile) and shrunk to annotate rather than "
                       "cover; forces recomputed with react_calib — the "
                       "previous newton values are void (end-to-end rho "
                       "0.143 from pixel-unit weights on mm-unit features)")
    print("uploaded")


if __name__ == "__main__":
    main()
