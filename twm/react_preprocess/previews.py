"""Release policy for the preview panels.

Previews are a presentation artifact, not part of reproducing the data: they
render a 3-camera + OptiTrack + GelSight panel from the *source* recordings and
need rig-local calibration that the release does not ship. So this module owns
only the release-specific decisions —

* which calibration set belongs to which task (they were recalibrated between
  the motherboard and pushT sessions, and using the wrong one silently
  misprojects the overlay)
* the trim offset, read from the release sidecar so previews start on the same
  frame as the published video
* the per-(task, date) world-frame offset
* the output layout

— and takes the renderer as a parameter. The previous version reached into the
renderer module and reassigned its globals, which meant preview settings could
not be reasoned about without reading both files, and two tasks could not be
rendered in one process.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterator

from .config import H5_ROOTS, STAGE_ROOT, WORLD_OFFSET

CALIB_ROOT = Path(__file__).resolve().parent.parent / "calibration"

# The rigs were recalibrated between sessions; each task must use the set that
# was current when it was recorded.
CALIB_DIRS = {
    "motherboard": CALIB_ROOT / "result backup",   # May 12
    "pushT": CALIB_ROOT / "result",                # June 26
}

CLIP_SECONDS = 30.0
SPEED = 2.0


def trim_offset(task: str, date: str, episode: str,
                stage_root: Path = STAGE_ROOT) -> int:
    """Trim offset for an episode, from its release sidecar (0 if absent)."""
    det = Path(stage_root) / task / "meta" / date / f"{episode}._detect.pt"
    if not det.exists():
        return 0
    import torch

    meta = torch.load(str(det), weights_only=False, map_location="cpu")
    return int(meta["_contact_meta"].get("trim_offset", 0))


def plan(task: str, stage_root: Path = STAGE_ROOT) -> Iterator[dict]:
    """One job per published episode that still has its source recording.

    Driven by the published videos rather than by the source tree, so episodes
    excluded from the release (e.g. the corrupt pushT recording) do not
    reappear here.
    """
    stage_root = Path(stage_root)
    h5_root = H5_ROOTS[task]
    videos = stage_root / task / "videos"
    if not videos.exists():
        return
    for date_dir in sorted(p for p in videos.iterdir() if p.is_dir()):
        date = date_dir.name
        dx, dy, dz = WORLD_OFFSET.get((task, date), (0.0, 0.0, 0.0))
        for ep_dir in sorted(p for p in date_dir.iterdir() if p.is_dir()):
            episode = ep_dir.name
            h5 = h5_root / date / f"{episode}.h5"
            if not h5.exists():
                continue
            yield {
                "task": task, "date": date, "episode": episode,
                "h5": h5,
                "out": stage_root / task / "previews" / date / f"{episode}.mp4",
                "calib_dir": CALIB_DIRS[task],
                "trim_offset": trim_offset(task, date, episode, stage_root),
                "world_offset": (dx, dy, dz),
            }


def build_task(task: str, render: Callable[[dict], None],
               stage_root: Path = STAGE_ROOT,
               overwrite: bool = False) -> list[dict]:
    """Render every planned preview with the supplied renderer.

    ``render`` receives one job dict. A failure is recorded against that
    episode and the rest continue — one bad recording should not cost the whole
    batch.
    """
    results = []
    for job in plan(task, stage_root):
        if job["out"].exists() and not overwrite:
            results.append({**job, "status": "SKIP"})
            continue
        job["out"].parent.mkdir(parents=True, exist_ok=True)
        try:
            render(job)
            size = job["out"].stat().st_size if job["out"].exists() else 0
            results.append({**job, "status": "OK", "bytes": size})
        except Exception as exc:                     # noqa: BLE001
            results.append({**job, "status": "FAIL",
                            "error": f"{type(exc).__name__}: {exc}"})
    return results
