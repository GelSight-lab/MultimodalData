"""One-shot: regenerate all figures, evaluation, and the site after batch."""
from __future__ import annotations

import sys

sys.path.insert(0, ".")

from force_recovery import visualize as V           # noqa: E402
from force_recovery.site import collect_and_build   # noqa: E402

EPISODES = [
    ("pushT", "2026-06-18", "episode_000"),
    ("motherboard", "2026-05-10", "episode_000"),
]

if __name__ == "__main__":
    for task, date, ep in EPISODES:
        print(V.force_timeline(task, date, ep), flush=True)
        for side in ("left", "right"):
            print(V.depth_panels(task, date, ep, side), flush=True)
            print(V.dexforce_figure(task, date, ep, side), flush=True)
        side = "right" if task == "pushT" else "left"
        print(V.overlay_clip(task, date, ep, side), flush=True)
    print(collect_and_build(), flush=True)
