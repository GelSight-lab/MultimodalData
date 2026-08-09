"""Reading source H5 recordings, with correct cross-modal time alignment.

This module owns the one piece of logic that used to be wrong: how a GelSight
frame is paired with a camera frame.

Two recording formats exist
---------------------------
**legacy** (up to 2026-06-18) — the rig stored whatever ``self.frame`` each
stream happened to hold at the 30 Hz tick, so tactile was *index*-aligned to
the cameras. Because a full 8 MP MJPG decode cost ~71 ms, the tactile thread
only really ran at ~8 fps, so those recordings carry both a systematic lag
(~15 frames, corrected downstream by a constant shift) and ~72 % duplicated
tactile frames.

**timestamped** (2026-06-27 onward) — the rig writes
``gelsight/<side>/timestamps``, the true capture time of each tactile frame.
We pair each camera tick with the *nearest-in-time* tactile frame, which
removes the systematic lag at the source. A constant shift must NOT also be
applied to these recordings, or they get corrected twice.

``TactileAlignment.needs_legacy_shift`` is the guard for exactly that.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

from . import repair
from .config import EXCLUDE_DATES, SIDES, WORLD_OFFSET


# ── pose alignment (unchanged from the validated build_episodes_from_h5) ─────
def cam_align_poses(cam_ts: np.ndarray, ot_ts, ot_pose) -> np.ndarray:
    """Nearest-timestamp OptiTrack pose for every camera tick."""
    if ot_ts is None or len(ot_ts) == 0:
        return np.zeros((len(cam_ts), 7), np.float32)
    idx = np.clip(np.searchsorted(ot_ts, cam_ts), 0, len(ot_ts) - 1)
    idxm = np.clip(idx - 1, 0, len(ot_ts) - 1)
    pick_minus = np.abs(ot_ts[idxm] - cam_ts) < np.abs(ot_ts[idx] - cam_ts)
    return ot_pose[np.where(pick_minus, idxm, idx)].astype(np.float32)


def find_first_valid(cam_ts, sl_ts, sr_ts) -> int:
    """First camera tick at which every active OptiTrack body has a sample."""
    starts = [float(t[0]) for t in (sl_ts, sr_ts) if t is not None and len(t) > 0]
    return int(np.searchsorted(cam_ts, max(starts))) if starts else 0


def nearest_index(src_ts: np.ndarray, target_ts: np.ndarray) -> np.ndarray:
    """For each `target_ts`, index of the nearest entry in sorted `src_ts`."""
    idx = np.clip(np.searchsorted(src_ts, target_ts), 0, len(src_ts) - 1)
    idxm = np.clip(idx - 1, 0, len(src_ts) - 1)
    pick_minus = np.abs(src_ts[idxm] - target_ts) < np.abs(src_ts[idx] - target_ts)
    return np.where(pick_minus, idxm, idx)


@dataclass
class TactileAlignment:
    """How camera ticks map onto GelSight frames for one side."""
    side: str
    index_map: np.ndarray           # (T,) int — gel frame index per camera tick
    timestamped: bool               # True when per-sensor capture times existed
    residual_ms: np.ndarray | None  # (T,) signed gel_ts - cam_ts, else None

    @property
    def needs_legacy_shift(self) -> bool:
        """Legacy recordings still need the constant +N frame latency shift.

        Timestamped recordings are already aligned here; applying a shift on
        top would double-correct them.
        """
        return not self.timestamped

    def summary(self) -> str:
        if not self.timestamped:
            return f"{self.side}: index-aligned (legacy, needs latency shift)"
        r = self.residual_ms
        return (f"{self.side}: timestamp-aligned "
                f"(residual mean {r.mean():+.1f} ms, |max| {np.abs(r).max():.0f} ms)")


@dataclass
class EpisodeSource:
    """One source H5 recording, with everything the pipeline needs from it."""
    path: Path
    task: str
    date: str
    episode: str

    cam_ts: np.ndarray = field(repr=False)
    trim: int
    active: list[str]
    world_offset: tuple[float, float, float]
    pose_left: np.ndarray = field(repr=False)
    pose_right: np.ndarray = field(repr=False)
    align: dict[str, TactileAlignment] = field(repr=False)

    @property
    def T(self) -> int:
        return len(self.cam_ts) - self.trim

    @property
    def trimmed_cam_ts(self) -> np.ndarray:
        return self.cam_ts[self.trim:]

    @property
    def timestamped(self) -> bool:
        """True when this recording carries per-sensor GelSight capture times."""
        return any(a.timestamped for a in self.align.values())

    def describe(self) -> str:
        kind = "timestamped" if self.timestamped else "legacy"
        lines = [f"{self.episode} [{kind}] T={self.T} trim={self.trim} active={self.active}"]
        lines += ["    " + self.align[s].summary() for s in SIDES if s in self.align]
        return "\n".join(lines)


def _read_body(f, name):
    grp = f"optitrack/{name}"
    if grp not in f:
        return None, None
    return f[f"{grp}/timestamps"][:], f[f"{grp}/pose"][:]


def open_episode(h5_path: Path, task: str) -> EpisodeSource:
    """Read metadata + build the tactile alignment for one recording.

    Frame pixels are *not* loaded here; the encoder streams them in blocks.
    """
    h5_path = Path(h5_path)
    date, episode = h5_path.parent.name, repair.source_stem(h5_path)
    offset = WORLD_OFFSET.get((task, date), (0.0, 0.0, 0.0))

    with h5py.File(str(h5_path), "r") as f:
        cam_ts = f["timestamps"][:]
        sl_ts, sl_pose = _read_body(f, "sensor_left")
        sr_ts, sr_pose = _read_body(f, "sensor_right")
        active = [s for s, t in (("left", sl_ts), ("right", sr_ts))
                  if t is not None and len(t) > 0]

        trim = find_first_valid(cam_ts, sl_ts, sr_ts)
        if len(cam_ts) - trim <= 0:
            raise ValueError(f"{episode}: nothing left after trim")
        tcam = cam_ts[trim:]
        T = len(tcam)

        align = {}
        for side in SIDES:
            n_gel = len(f[f"gelsight/{side}/frames"])
            ts_key = f"gelsight/{side}/timestamps"
            if ts_key in f and len(f[ts_key]) == n_gel and n_gel > 0:
                gel_ts = f[ts_key][:]
                idx = nearest_index(gel_ts, tcam)
                align[side] = TactileAlignment(
                    side, idx.astype(np.int64), True,
                    (gel_ts[idx] - tcam) * 1000.0)
            else:
                # legacy: tactile stored at the same tick index as the cameras
                idx = np.clip(np.arange(trim, trim + T), 0, n_gel - 1)
                align[side] = TactileAlignment(side, idx.astype(np.int64), False, None)

        pose_l = cam_align_poses(tcam, sl_ts, sl_pose).copy()
        pose_r = cam_align_poses(tcam, sr_ts, sr_pose).copy()

    for p in (pose_l, pose_r):
        p[:, 0] += offset[0]; p[:, 1] += offset[1]; p[:, 2] += offset[2]

    return EpisodeSource(h5_path, task, date, episode, cam_ts, trim, active,
                         offset, pose_l, pose_r, align)


def discover(task: str, root: Path, date=None, episodes=None) -> list[Path]:
    """All publishable source recordings for a task, sorted."""
    # A `.recovered.h5` is not an episode of its own — it is reached through
    # the source it was rebuilt from. Left in, it would publish as
    # `episode_004.recovered` beside `episode_004`.
    paths = sorted(p for p in root.rglob("episode_*.h5")
                   if p.parent.name not in EXCLUDE_DATES
                   and not repair.is_recovered(p))
    if date:
        paths = [p for p in paths if p.parent.name == date]
    if episodes:
        want = set(episodes)
        paths = [p for p in paths if p.stem in want]
    return paths
