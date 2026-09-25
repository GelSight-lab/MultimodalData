"""One PUBLISHED episode, read back the way a dataset user gets it.

The viewer could only open the raw H5. That is the wrong artefact to review
before publishing: the H5 holds 30 GB of raw pixels, while what ships is seven
MP4s plus a parquet — re-encoded, tone-curved on the wrist streams, trimmed to
the release window. Reviewing the source and shipping the derivative means the
thing you looked at is not the thing you published.

Everything the overlay needs is already in the release:

* poses — the parquet's `sensor_{left,right}_pose` and `object_pose` columns
  are row-aligned to the video, so there is no timestamp search to redo and no
  way for this path to disagree with the published data about where a sensor
  was;
* forces — the force estimator's per-episode npz, whose rows are the parquet's
  rows;

and both are handed back in the shapes `twm.viz` already speaks, so the overlay
code is reused rather than restated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from twm.react_preprocess.config import CAM_STREAM, FPS, GEL_STREAM, STAGE_ROOT, WRIST_STREAM

# Where the force estimator writes `<task>/<date>/<episode>_<side>.npz`.
FORCE_ROOT = Path("/media/yxma/Disk1/twm/force_recovery")

# The parquet column each OptiTrack body is published under. `motherboard` is
# the body the rig broadcasts for the manipulated object, whatever the task.
POSE_COLUMN = {"sensor_left": "sensor_left_pose",
               "sensor_right": "sensor_right_pose",
               "motherboard": "object_pose"}


class _VideoStream:
    """One MP4, read sequentially, seeking only when the caller jumps.

    Decoding forward one frame is cheap; `CAP_PROP_POS_FRAMES` is not (it
    re-seeks to a keyframe and decodes forward). Playback is overwhelmingly
    sequential, so the position is tracked and a seek issued only when the
    requested index is not the next one.
    """

    def __init__(self, path: Path):
        import cv2
        self.path = Path(path)
        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            raise FileNotFoundError(f"cannot open {path}")
        self._next = 0
        self._last: tuple[int, np.ndarray] | None = None

    def frame(self, index: int) -> np.ndarray:
        import cv2
        if self._last is not None and self._last[0] == index:
            return self._last[1]
        if index != self._next:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            self._next = index
        ok, img = self._cap.read()
        if not ok:
            # Past the end, or a decode failure: a grey frame says "nothing
            # here" without pretending the previous frame is this one.
            img = np.full((48, 64, 3), 128, np.uint8) if self._last is None \
                else np.full_like(self._last[1], 128)
        self._next = index + 1
        self._last = (index, img)
        return img

    def close(self):
        self._cap.release()


@dataclass
class ReleaseEpisode:
    task: str
    date: str
    episode: str
    video_dir: Path
    parquet: Path
    force_root: Path = FORCE_ROOT
    fps: float = FPS
    _table: object = field(default=None, repr=False)
    _streams: dict = field(default_factory=dict, repr=False)
    _forces: dict = field(default=None, repr=False)

    # ── the parquet ─────────────────────────────────────────────────────────
    @property
    def table(self):
        if self._table is None:
            import pyarrow.parquet as pq
            self._table = pq.read_table(str(self.parquet))
        return self._table

    @property
    def n_frames(self) -> int:
        """From the parquet, not the video: the parquet is what says how long
        the published episode is, and a video that disagrees is a build defect
        that has to surface rather than be papered over."""
        return self.table.num_rows

    def _column(self, name):
        return self.table[name] if name in self.table.column_names else None

    def poses_at(self, index: int) -> dict:
        """`{body: (timestamp, [x,y,z,qx,qy,qz,qw])}`, or None per body.

        The shape `twm.viz.optitrack_at` returns, so the overlay does not care
        which source it is drawing.
        """
        i = int(np.clip(index, 0, max(0, self.n_frames - 1)))
        ts_col = self._column("timestamp")
        ts = float(ts_col[i].as_py()) if ts_col is not None else i / self.fps
        out = {}
        for body, col_name in POSE_COLUMN.items():
            col = self._column(col_name)
            if col is None:
                out[body] = None
                continue
            vec = np.asarray(col[i].as_py(), dtype=float)
            # All-NaN is how the build records a body that was never
            # broadcast. Passing it through as zeros would draw an
            # observation at the world origin.
            out[body] = None if vec.size != 7 or not np.isfinite(vec).all() \
                else (ts, vec.tolist())
        return out

    def tactile_intensity_at(self, index: int) -> dict:
        i = int(np.clip(index, 0, max(0, self.n_frames - 1)))
        out = {}
        for side in ("left", "right"):
            col = self._column(f"tactile_{side}_intensity")
            if col is not None:
                out[side] = float(col[i].as_py())
        return out

    # ── the force npz ───────────────────────────────────────────────────────
    @property
    def forces(self) -> dict:
        """`{side: per-row newtons}` for the sides that have force.

        The PARQUET first. It is what a dataset user reads, it is already
        row-aligned to the video, and since 2026-09-17 every published segment
        carries `force_<side>_normal_n`.

        The npz lookup below only ever worked for uncut episodes: published
        units are SEGMENTS (`episode_006_seg00`) while the npz are named for
        the source recording (`episode_006_left.npz`), so it missed on every
        segment and returned {} -- and an empty dict is indistinguishable from
        "this episode has no force channel", so `twm.visualize` played
        published segments with no force overlay and said nothing.
        """
        if self._forces is None:
            self._forces = {}
            for side in ("left", "right"):
                col = self._column(f"force_{side}_normal_n")
                if col is not None:
                    self._forces[side] = np.asarray(col.to_numpy(), dtype=float)
            if self._forces:
                return self._forces
            # No columns: an uncut episode, where the npz is the only source.
            for side in ("left", "right"):
                p = self.force_root / self.task / self.date / f"{self.episode}_{side}.npz"
                if p.exists():
                    self._forces[side] = np.asarray(
                        np.load(p)["force_normal_n"], dtype=float)
        return self._forces

    def forces_at(self, index: int) -> dict:
        """`{side: newtons}` for the sides the estimator has run on.

        A side whose array is shorter than the episode is simply absent past
        its end — clamping to the last value would assert a contact force for
        frames nobody measured.
        """
        return {side: float(arr[index])
                for side, arr in self.forces.items()
                if 0 <= index < len(arr) and np.isfinite(arr[index])}

    # ── the videos ──────────────────────────────────────────────────────────
    def _stream(self, name: str):
        if name not in self._streams:
            self._streams[name] = _VideoStream(self.video_dir / f"{name}.mp4")
        return self._streams[name]

    def has(self, name: str) -> bool:
        return (self.video_dir / f"{name}.mp4").is_file()

    def frames(self, index: int):
        """(color[3], gelsight[2], wrist[2]) at `index`, in viewer order.

        Colour is ordered by H5 camera index the way the viewer expects, so
        `DISPLAY_POSITION` keeps placing each thumbnail where it belongs.
        """
        colour = [self._stream(CAM_STREAM[i]).frame(index) for i in sorted(CAM_STREAM)]
        gel = [self._stream(GEL_STREAM[s]).frame(index) for s in ("left", "right")]
        wrist = [self._stream(WRIST_STREAM[s]).frame(index)
                 for s in sorted(WRIST_STREAM) if self.has(WRIST_STREAM[s])]
        return colour, gel, wrist

    def close(self):
        for s in self._streams.values():
            s.close()
        self._streams.clear()


def resolve(path, *, root: Path | None = None,
            force_root: Path = FORCE_ROOT) -> ReleaseEpisode:
    """Find a published episode from any of the three names it has.

    A parquet path, its video directory, or the `<task>/<date>/<episode>` key
    that `episodes.jsonl` uses — whichever the user's shell completed.
    """
    root = Path(root) if root is not None else STAGE_ROOT
    p = Path(path)
    task = date = episode = None

    if p.suffix == ".parquet" and p.is_file():
        task, date, episode = p.parts[-4], p.parent.name, p.stem
        base = p.parents[3]
    elif p.is_dir() and (p.parent.parent.name == "videos"
                         or any(p.glob("view_*.mp4"))):
        task, date, episode = p.parts[-4], p.parent.name, p.name
        base = p.parents[3]
    else:
        parts = [x for x in str(path).split("/") if x]
        if len(parts) != 3:
            raise FileNotFoundError(
                f"not a published episode: {path!r} — give the parquet, its "
                f"video directory, or '<task>/<date>/<episode>'")
        task, date, episode = parts
        base = root

    video_dir = Path(base) / task / "videos" / date / episode
    parquet = Path(base) / task / "meta" / date / f"{episode}.parquet"
    if not parquet.is_file():
        raise FileNotFoundError(f"no published parquet for {task}/{date}/{episode} "
                                f"(looked for {parquet})")
    if not video_dir.is_dir():
        raise FileNotFoundError(f"no published videos for {task}/{date}/{episode} "
                                f"(looked for {video_dir})")
    return ReleaseEpisode(task, date, episode, video_dir, parquet,
                          force_root=Path(force_root))


def shipped_calibration(root, task: str):
    """The calibration that SHIPS with a release tree, or None.

    `(camera_json_paths, gel_left_json, gel_right_json)`.

    Published poses are Z-up and `convert_release_zup` rotates the poses AND
    the calibration together -- the shipped matrix equals the repo one composed
    with the Y->Z rotation -- so the two are interchangeable only in matched
    pairs. Playing a published episode against the repo epoch pairs Z-up poses
    with a Y-up extrinsic; the translation is identical either way, so the
    overlay drifts by a rotation rather than breaking visibly.

    All five files or nothing. Mixing a shipped camera matrix with a repo gel
    file pairs a Z-up extrinsic with a Y-up one, which is worse than using
    neither.
    """
    d = Path(root) / task / "calibration"
    cams = sorted(d.glob("T_mocap_to_cam_*.json"))
    gl, gr = d / "T_gel_to_rigid_left.json", d / "T_gel_to_rigid_right.json"
    if not cams or not gl.is_file() or not gr.is_file():
        return None
    return [str(p) for p in cams], str(gl), str(gr)


def looks_like_release(path) -> bool:
    """True when `path` names a published episode rather than a recording."""
    p = Path(path)
    if p.suffix in (".h5", ".hdf5"):
        return False
    return p.suffix == ".parquet" or p.is_dir() or str(path).count("/") == 2
