"""Is an episode's build actually finished?

Counting files does not answer this. `build_episode` publishes its parquet LAST,
after all requested videos and the detection sidecar. Older builders left a
previous run's parquet on disk during a `--force` rebuild. Since ffmpeg creates
each output file the moment it opens it, "7 mp4 + a parquet" was satisfied
throughout a rebuild, by a stale
parquet and seven files that are still growing.

That is not hypothetical: the cut ran against
motherboard/2026-09-11/episode_006 at 05:21 while its tactile_right.mp4 was
still being written (it finished at 05:46), read a half-written file, and
reported the episode as having an undecodable stream.

The pipeline removes the old completion parquet before replacing any output,
then atomically publishes a new one after every required artifact succeeds.
Timestamp checks also reject stale completion files from older builders.
"""
from __future__ import annotations

from pathlib import Path

from .config import CAM_STREAM, GEL_STREAM, WRIST_STREAM

STREAMS = ("view_left", "view_middle", "view_right", "tactile_left",
           "tactile_right", "wrist_left", "wrist_right")


def is_complete(release_root, task: str, date: str, episode: str, *,
                source_h5=None, encode_video: bool = True,
                with_depth: bool = False) -> bool:
    """Check the requested build artifacts, using source camera availability.

    Callers without a source (such as a scheduler) still require all seven
    streams. Metadata-only builds require metadata and the detection sidecar;
    requesting video or depth later therefore cannot skip missing outputs.
    """
    root = Path(release_root) / task
    pq = root / "meta" / date / f"{episode}.parquet"
    if not pq.is_file():
        return False
    streams = list(STREAMS)
    depths = list(CAM_STREAM.values())
    if source_h5 is not None:
        import h5py

        with h5py.File(source_h5, "r") as f:
            streams = list(GEL_STREAM.values())
            streams += [name for index, name in CAM_STREAM.items()
                        if f"realsense/cam{index}/color" in f]
            streams += [name for slot, name in WRIST_STREAM.items()
                        if f"arducam/{slot}/frames" in f]
            depths = [name for index, name in CAM_STREAM.items()
                      if f"realsense/cam{index}/depth" in f]
    artifacts = [root / "meta" / date / f"{episode}._detect.pt"]
    if encode_video:
        artifacts += [root / "videos" / date / episode / f"{s}.mp4"
                      for s in streams]
    if with_depth:
        artifacts += [root / "depth" / date / episode /
                      f"{s.replace('view_', 'depth_')}.mkv" for s in depths]
    try:
        if not all(path.is_file() for path in artifacts):
            return False
        stats = [path.stat() for path in artifacts]
        pq_stat = pq.stat()
    except FileNotFoundError:  # a rebuild may invalidate output during this check
        return False
    # Equality is valid on filesystems with coarse timestamp resolution.
    return (pq_stat.st_size > 0 and all(s.st_size > 0 for s in stats)
            and pq_stat.st_mtime_ns >= max(s.st_mtime_ns for s in stats))
