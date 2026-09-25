"""Is an episode's build actually finished?

Counting files does not answer this. `build_episode` writes its parquet LAST,
so a parquet beside seven mp4 normally means a finished episode -- but during a
`--force` rebuild the PREVIOUS run's parquet is still on disk while the new
videos are being written, and ffmpeg creates each output file the moment it
opens it. So "7 mp4 + a parquet" is satisfied throughout a rebuild, by a stale
parquet and seven files that are still growing.

That is not hypothetical: the cut ran against
motherboard/2026-09-11/episode_006 at 05:21 while its tactile_right.mp4 was
still being written (it finished at 05:46), read a half-written file, and
reported the episode as having an undecodable stream.

The invariant that survives a rebuild is the ORDER: the parquet is written
after every video, so a parquet newer than the newest mp4 means this build
finished. A stale one is always older than the videos currently being written.
"""
from __future__ import annotations

from pathlib import Path

STREAMS = ("view_left", "view_middle", "view_right", "tactile_left",
           "tactile_right", "wrist_left", "wrist_right")


def is_complete(release_root, task: str, date: str, episode: str) -> bool:
    root = Path(release_root) / task
    pq = root / "meta" / date / f"{episode}.parquet"
    if not pq.is_file():
        return False
    vids = [root / "videos" / date / episode / f"{s}.mp4" for s in STREAMS]
    if not all(v.is_file() for v in vids):
        return False
    # `>=`, not `>`: the parquet is written immediately after the last video,
    # and on a coarse-timestamp filesystem the two can land in the same tick.
    # A STALE parquet is never merely equal -- it predates the rebuild by
    # however long the build has been running (9 seconds apart on the real
    # episode_006, but minutes for the videos written earlier in the same run).
    return pq.stat().st_mtime >= max(v.stat().st_mtime for v in vids)
