"""Interactive playback for React data -- accepts a single `.pt` OR a
directory of segments / episodes, concatenates same-episode segments
into one continuous timeline, and lets you switch between source
episodes with N / P.

The viewer reuses `twm.viz.build_preview_panel`'s 1280x480 grid:

    Row 1 (y=0..240):    [left cam | middle cam | right cam | OptiTrack]
    Row 2 (y=240..480):  [gs_L_raw | gs_L_diff | gs_R_raw | gs_R_diff | Controls]

The three RealSense thumbnails and both GelSight raw streams are pulled
from the source HDF5 file (referenced by each .pt segment's
`_contact_meta.source_episode` + `source_h5_frame_range`). When the
source H5 cannot be located, those cells are filled with a "no H5"
placeholder.

Usage
-----
Single .pt file (mode1_v1 episode OR one mode2_v1 segment):
    python scripts/play_react_pt.py \\
        processed/mode2_v1/motherboard/2026-05-11/episode_012.segment_00.pt

Episode stem (no .pt suffix) -- loads ALL segments of that episode and
seeds N/P navigation across the rest of the date folder:
    python scripts/play_react_pt.py \\
        processed/mode2_v1/motherboard/2026-05-11/episode_005

Date folder -- all episodes recorded that day:
    python scripts/play_react_pt.py \\
        processed/mode2_v1/motherboard/2026-05-11

Whole task -- every episode across every date:
    python scripts/play_react_pt.py \\
        processed/mode2_v1/motherboard

  In all multi-episode modes, same-episode segments are concatenated
  into one playback timeline; N / P jumps to the next / previous
  source episode.

Headless export (one MP4 per episode):
    python scripts/play_react_pt.py <root> --save_video_dir /tmp/out

Override the source-H5 root (otherwise inferred by replacing
`processed/<mode>` with `data` in the input path):
    python scripts/play_react_pt.py <root> --h5_root /path/to/data/<task>

Controls
--------
    space         pause / resume
    -> / d        next frame
    <- / a        previous frame
    1..6          playback speed 1x / 2x / 5x / 10x / 25x / 50x
    r             reset GelSight diff reference to current frame
    n             next episode
    p             previous episode
    q             quit
"""
import argparse
import re
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import h5py
import numpy as np
import torch

try:
    from twm.viz import (
        load_optitrack as _viz_load_optitrack,
        optitrack_at as _viz_optitrack_at,
        DISPLAY_ORDER,
        draw_projection_overlay as _viz_draw_projection_overlay,
        load_calibrations as _viz_load_calibrations,
    )
except Exception:
    _viz_load_optitrack = None
    _viz_optitrack_at = None
    _viz_draw_projection_overlay = None
    _viz_load_calibrations = None
    DISPLAY_ORDER = [1, 2, 0]   # H5 cam_idx order for [left, middle, right]

try:
    from twm.data_collection import REALSENSE_SERIALS
except Exception:
    REALSENSE_SERIALS = ["143322063538", "104122062574", "217222066989"]


# ──────────────────────────────────────────────────────────────────────────────
# Layout constants (matches twm.viz.build_preview_panel)
# ──────────────────────────────────────────────────────────────────────────────
PANEL_W, PANEL_H = 1280, 480
RS_THUMB_W, RS_THUMB_H = 320, 240     # RealSense cam cell
GS_THUMB_W, GS_THUMB_H = 240, 240     # GelSight raw/diff cell
CONTROLS_W, CONTROLS_H = 320, 240     # Controls panel (replaces blank in row 2)

TRACKER_COLORS = {
    "sensor_left":  (  0, 255, 120),
    "sensor_right": (  0, 180, 255),
}

SEGMENT_FNAME_RE = re.compile(r"^(episode_\d+)\.segment_(\d+)\.pt$")
PLAIN_EPISODE_RE = re.compile(r"^(episode_\d+)\.pt$")
DATE_RE          = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# The calibration epoch is PER TASK (May-12 for motherboard, June-26 for
# pushT); it is resolved from the input path at parse time via calib_epoch.
# The old module constant here defaulted every task to June-26.


# ──────────────────────────────────────────────────────────────────────────────
def _tactile_to_bgr(tac_chw):
    rgb = tac_chw.permute(1, 2, 0).numpy()
    return rgb[..., ::-1]


def _missing_cell(w, h, label):
    panel = np.full((h, w, 3), 32, np.uint8)
    cv2.putText(panel, label, (10, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1, cv2.LINE_AA)
    return panel


# ──────────────────────────────────────────────────────────────────────────────
# H5 source resolution
# ──────────────────────────────────────────────────────────────────────────────

def infer_h5_root(pt_path: Path) -> Optional[Path]:
    """Map a .pt path to its source-H5 root by replacing the
    `processed/<mode>/` prefix with `data/`.

    Examples:
        .../twm/processed/mode2_v1/motherboard/2026-05-11/ep.pt
          -> .../twm/data/motherboard
        .../twm/processed/mode2_v1/motherboard
          -> .../twm/data/motherboard
    """
    parts = list(pt_path.parts)
    try:
        i = parts.index("processed")
    except ValueError:
        return None
    if i + 1 >= len(parts):
        return None
    base = parts[:i]
    tail = []
    for p in parts[i + 2:]:
        if DATE_RE.match(p) or p.endswith(".pt"):
            break
        tail.append(p)
    return Path(*base, "data", *tail)


def resolve_h5_path(h5_root: Optional[Path], source_episode: Optional[str]) -> Optional[Path]:
    """`source_episode` is '<date>/<episode_stem>'. Returns
    <h5_root>/<source_episode>.h5 if it exists, else None.
    """
    if not h5_root or not source_episode:
        return None
    candidate = h5_root / f"{source_episode}.h5"
    return candidate if candidate.exists() else None


class H5Source:
    """Lazy read-only access to a source H5: cams, gelsight frames, OT poses.

    Reads are served from an in-RAM chunk cache (default 32 frames). When the
    requested frame falls outside the current chunk, the next chunk for all
    5 image streams is read in a single batched slab read -- ~30x faster than
    per-frame H5 fancy-indexing.
    """

    CHUNK = 32

    def __init__(self, h5_path: Optional[Path]):
        self.path = h5_path
        self.f = None
        self.n_frames = 0
        self.timestamps = None
        self.gs_left_n = 0
        self.gs_right_n = 0
        self.optitrack = None
        self._chunk_start = -1   # inclusive
        self._chunk_end   = -1   # exclusive (for cams)
        self._chunk_cache: dict = {}
        if h5_path is None:
            return
        try:
            self.f = h5py.File(str(h5_path), "r")
            self.n_frames   = int(self.f["timestamps"].shape[0])
            self.timestamps = self.f["timestamps"][:]
            self.gs_left_n  = int(self.f["gelsight/left/frames"].shape[0])
            self.gs_right_n = int(self.f["gelsight/right/frames"].shape[0])
            if _viz_load_optitrack is not None:
                self.optitrack = _viz_load_optitrack(self.f)
        except Exception as e:
            print(f"[player]   ! failed to open H5 {h5_path}: {e}")
            self.close()

    @property
    def ok(self) -> bool:
        return self.f is not None

    def _ensure_chunk(self, h5_frame: int):
        """Load the chunk containing h5_frame into RAM if not already there."""
        if h5_frame < 0 or h5_frame >= self.n_frames:
            return
        cs = (h5_frame // self.CHUNK) * self.CHUNK
        if cs == self._chunk_start:
            return
        ce = min(cs + self.CHUNK, self.n_frames)
        cache = {
            "cam0": self.f["realsense/cam0/color"][cs:ce],
            "cam1": self.f["realsense/cam1/color"][cs:ce],
            "cam2": self.f["realsense/cam2/color"][cs:ce],
        }
        if self.gs_left_n > 0:
            ge = min(ce, self.gs_left_n)
            cache["gs_left"] = self.f["gelsight/left/frames"][cs:ge] if cs < self.gs_left_n else None
        if self.gs_right_n > 0:
            ge = min(ce, self.gs_right_n)
            cache["gs_right"] = self.f["gelsight/right/frames"][cs:ge] if cs < self.gs_right_n else None
        self._chunk_start = cs
        self._chunk_end   = ce
        self._chunk_cache = cache

    def cam(self, idx: int, h5_frame: int) -> Optional[np.ndarray]:
        if not self.ok or h5_frame < 0 or h5_frame >= self.n_frames:
            return None
        self._ensure_chunk(h5_frame)
        slab = self._chunk_cache.get(f"cam{idx}")
        if slab is None:
            return None
        return slab[h5_frame - self._chunk_start]

    def gelsight(self, side: str, h5_frame: int) -> Optional[np.ndarray]:
        if not self.ok:
            return None
        n = self.gs_left_n if side == "left" else self.gs_right_n
        if n == 0:
            return None
        clamped = min(max(h5_frame, 0), n - 1)
        self._ensure_chunk(clamped)
        slab = self._chunk_cache.get(f"gs_{side}")
        if slab is None:
            return None
        local = clamped - self._chunk_start
        if local < 0 or local >= len(slab):
            return None
        return slab[local]

    def ot_at(self, h5_frame: int) -> dict:
        if not self.ok or self.optitrack is None or _viz_optitrack_at is None:
            return {}
        idx = min(max(h5_frame, 0), self.n_frames - 1)
        return _viz_optitrack_at(self.optitrack, float(self.timestamps[idx]))

    def close(self):
        self._chunk_cache = {}
        self._chunk_start = -1
        if self.f is not None:
            try: self.f.close()
            except Exception: pass
            self.f = None


# ──────────────────────────────────────────────────────────────────────────────
# Episode discovery
# ──────────────────────────────────────────────────────────────────────────────

def discover_episodes(root: Path) -> list[dict]:
    """Walk `root`, group .pt files by source episode key '<date>/episode_NNN',
    and return one entry per episode with its segment paths sorted.

    Handles both:
      - mode2_v1: `<root>/<date>/episode_NNN.segment_MM.pt`
      - mode1_v1: `<root>/<date>/episode_NNN.pt`
    """
    pts = sorted(root.rglob("*.pt"))
    if not pts:
        raise SystemExit(f"No .pt files under {root}")

    episodes: dict[str, dict] = {}
    for p in pts:
        m_seg = SEGMENT_FNAME_RE.match(p.name)
        m_plain = PLAIN_EPISODE_RE.match(p.name)
        if m_seg:
            ep_stem, seg_idx = m_seg.group(1), int(m_seg.group(2))
        elif m_plain:
            ep_stem, seg_idx = m_plain.group(1), 0
        else:
            continue
        date = p.parent.name
        key = f"{date}/{ep_stem}"
        episodes.setdefault(key, {"date": date, "ep_stem": ep_stem, "segs": []})
        episodes[key]["segs"].append((seg_idx, p))

    out = []
    for key in sorted(episodes):
        ep = episodes[key]
        ep["segs"].sort(key=lambda x: x[0])
        out.append({"key": key, **ep})
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Episode-load
# ──────────────────────────────────────────────────────────────────────────────

class LoadedEpisode:
    """One source episode, with all its segments concatenated for playback.

    The concat timeline only carries .pt scalar/pose tensors; image cells are
    served on-demand from the bound source H5 file.
    """
    PER_FRAME_KEYS = (
        "timestamps",
        "sensor_left_pose", "sensor_right_pose",
        "tactile_left_intensity",  "tactile_right_intensity",
        "tactile_left_mixed",      "tactile_right_mixed",
    )

    def __init__(self, ep_meta: dict, h5_root: Optional[Path] = None,
                 tactile_latency: int = 0, source: str = "pt"):
        self.tactile_latency = int(tactile_latency)
        self.source = source
        self.key = ep_meta["key"]
        self.date = ep_meta["date"]
        self.ep_stem = ep_meta["ep_stem"]
        self.segment_paths = [p for _, p in ep_meta["segs"]]
        _t_load = time.time()
        n_segs = len(self.segment_paths)
        print(f"[player] loading episode {self.key}  "
              f"({n_segs} segment{'s' if n_segs > 1 else ''})...", flush=True)

        # mmap=True memory-maps the .pt without copying tensors into RAM.
        # We only touch the small scalar/pose fields below; the large
        # image tensors (view, tactile_*) are never read.
        def _load(p):
            try:
                return torch.load(str(p), weights_only=False, map_location="cpu", mmap=True)
            except (TypeError, ValueError, RuntimeError):
                return torch.load(p, weights_only=False, map_location="cpu")
        per_seg = []
        for i, p in enumerate(self.segment_paths):
            _ts = time.time()
            per_seg.append(_load(p))
            print(f"[player]   [{i + 1}/{n_segs}] {p.name}  "
                  f"({p.stat().st_size / 1e6:.0f} MB, {time.time() - _ts:.2f}s)", flush=True)

        # Clone small tensors out of the mmap so they don't keep the file open
        # any longer than necessary. When source="pt" we *do* keep the mmap'd
        # per-segment dicts alive so `view` / `tactile_*` can be served from
        # the .pt on demand.
        cat = {}
        for k in self.PER_FRAME_KEYS:
            if k in per_seg[0]:
                cat[k] = torch.cat([s[k].clone() for s in per_seg], dim=0)
        self.per_seg_data = per_seg if source == "pt" else None

        bounds = []
        cursor = 0
        source_ep = None
        for i, s in enumerate(per_seg):
            # Use shape[0] of any per-frame tensor instead of "view" to avoid
            # paging in the giant image tensor under mmap.
            n = int(s["timestamps"].shape[0])
            meta = s.get("_contact_meta", {})
            if source_ep is None:
                source_ep = meta.get("source_episode")
            bounds.append({
                "start_in_concat": cursor,
                "end_in_concat":   cursor + n - 1,
                "seg_idx":         int(meta.get("source_segment_idx", i)),
                "source_h5_range": meta.get("source_h5_frame_range"),
                "n_frames":        n,
            })
            cursor += n
        self.data = cat
        self.bounds = bounds
        self.n_frames = cursor
        self.source_episode = source_ep

        # Bind source H5
        h5_path = resolve_h5_path(h5_root, source_ep) if source_ep else None
        self.h5 = H5Source(h5_path)
        if self.h5.ok:
            print(f"[player]   H5 source: {h5_path}  ({self.h5.n_frames} frames)", flush=True)
        else:
            print(f"[player]   H5 source not available "
                  f"(source_episode={source_ep!r}, h5_root={h5_root}) -- "
                  f"cam/gelsight cells will be blank", flush=True)

        # GelSight diff references = gelsight frame at concat-frame 0 + latency
        # (shifted so the reference matches what build_panel will display).
        self.gs_ref_L = None
        self.gs_ref_R = None
        ref_idx = max(0, min(self.n_frames - 1, self.tactile_latency))
        if source == "pt":
            self.gs_ref_L = self.pt_tactile("left",  ref_idx)
            self.gs_ref_R = self.pt_tactile("right", ref_idx)
        elif self.h5.ok:
            r0 = bounds[0]["source_h5_range"]
            ref_h5_frame = (r0[0] if r0 else 0) + self.tactile_latency
            self.gs_ref_L = self.h5.gelsight("left",  ref_h5_frame)
            self.gs_ref_R = self.h5.gelsight("right", ref_h5_frame)

        print(f"[player]   total {self.n_frames} frames  ({self.n_frames / 30.0:.1f}s)  "
              f"[ready in {time.time() - _t_load:.2f}s]", flush=True)

    def segment_at(self, frame_idx: int) -> dict:
        for b in self.bounds:
            if b["start_in_concat"] <= frame_idx <= b["end_in_concat"]:
                return b
        return self.bounds[-1]

    def is_segment_start(self, frame_idx: int) -> bool:
        return any(frame_idx == b["start_in_concat"] for b in self.bounds[1:])

    def h5_frame_for(self, frame_idx: int) -> Optional[int]:
        seg = self.segment_at(frame_idx)
        r = seg["source_h5_range"]
        if r is None:
            return None
        return r[0] + (frame_idx - seg["start_in_concat"])

    def _seg_local(self, frame_idx: int):
        """Map concat frame_idx -> (segment index, local index inside segment)."""
        for i, b in enumerate(self.bounds):
            if b["start_in_concat"] <= frame_idx <= b["end_in_concat"]:
                return i, frame_idx - b["start_in_concat"]
        last = len(self.bounds) - 1
        return last, frame_idx - self.bounds[last]["start_in_concat"]

    def _pt_image(self, key: str, frame_idx: int) -> Optional[np.ndarray]:
        """Read an image tensor (shape (T, 3, H, W) uint8) from the per-segment
        .pt mmap'd dicts, return BGR HxWx3 uint8 or None.
        """
        if self.per_seg_data is None or frame_idx < 0 or frame_idx >= self.n_frames:
            return None
        seg_i, local = self._seg_local(frame_idx)
        d = self.per_seg_data[seg_i]
        if key not in d:
            return None
        chw = d[key][local]
        rgb = chw.permute(1, 2, 0).numpy()
        return np.ascontiguousarray(rgb[..., ::-1])    # BGR for OpenCV

    def pt_view(self, frame_idx: int) -> Optional[np.ndarray]:
        return self._pt_image("view", frame_idx)

    def pt_tactile(self, side: str, frame_idx: int) -> Optional[np.ndarray]:
        return self._pt_image(f"tactile_{side}", frame_idx)

    def close(self):
        self.h5.close()


# ──────────────────────────────────────────────────────────────────────────────
# Panel renderer (visualize.py-style 1280x480 grid)
# ──────────────────────────────────────────────────────────────────────────────

def _make_ot_panel(ot_poses, t_idx, t_sec, w, h):
    panel = np.zeros((h, w, 3), np.uint8)
    cv2.putText(panel, "OptiTrack (this frame)", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.line(panel, (8, 30), (w - 8, 30), (60, 60, 60), 1)
    y = 56
    for name in ("sensor_left", "sensor_right"):
        color = TRACKER_COLORS[name]
        cv2.putText(panel, name, (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        y += 18
        pose = ot_poses.get(name) if isinstance(ot_poses, dict) else None
        if pose is None:
            cv2.putText(panel, "  no data", (8, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1, cv2.LINE_AA)
            y += 36
            continue
        _, xyz_quat = pose
        x_m, y_m, z_m = xyz_quat[:3]
        qx, qy, qz, qw = xyz_quat[3:]
        cv2.putText(panel, f"  x={x_m:+.3f} y={y_m:+.3f} z={z_m:+.3f}", (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)
        y += 16
        cv2.putText(panel, f"  qx={qx:+.2f} qy={qy:+.2f}", (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1, cv2.LINE_AA)
        y += 16
        cv2.putText(panel, f"  qz={qz:+.2f} qw={qw:+.2f}", (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1, cv2.LINE_AA)
        y += 24
    cv2.putText(panel, f"t = {t_sec:.2f} s   (concat frame {t_idx})",
                (8, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    return panel


def _make_controls_panel(w, h, paused, speed, ep_idx, n_eps, tactile_latency=0):
    panel = np.zeros((h, w, 3), np.uint8)
    cv2.putText(panel, "Controls", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.line(panel, (10, 26), (w - 10, 26), (80, 80, 80), 1)
    keys = [
        ("SPACE",     "pause / resume"),
        ("-> / d",    "next frame"),
        ("<- / a",    "prev frame"),
        ("1..6",      "speed 1x/2x/5x/10x/25x/50x"),
        ("n / p",     "next / prev episode"),
        ("r",         "reset gel-diff ref"),
        ("[ / ]",     "tactile lat -/+ 1"),
        ("q",         "quit"),
    ]
    y = 44
    for key, desc in keys:
        highlight = (key == "SPACE")
        key_color  = (0, 220, 255) if highlight else (140, 200, 140)
        desc_color = (220, 220, 220) if highlight else (160, 160, 160)
        cv2.putText(panel, f"[{key}]", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, key_color, 1, cv2.LINE_AA)
        cv2.putText(panel, desc, (105, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, desc_color, 1, cv2.LINE_AA)
        y += 18
    state_text  = "|| PAUSED" if paused else "> PLAYING"
    state_color = (0, 140, 255) if paused else (0, 220, 80)
    cv2.putText(panel, state_text, (10, h - 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, state_color, 2, cv2.LINE_AA)
    cv2.putText(panel, f"speed: {speed}x   ep {ep_idx + 1}/{n_eps}",
                (10, h - 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(panel, f"tactile_latency: {tactile_latency:+d}",
                (10, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1, cv2.LINE_AA)
    return panel


def _rs_thumb(img, label=None):
    out = cv2.resize(img, (RS_THUMB_W, RS_THUMB_H))
    if label:
        cv2.putText(out, label, (6, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    return out


def _gs_thumb(img, label=None):
    out = cv2.resize(img, (GS_THUMB_W, GS_THUMB_H))
    if label:
        cv2.putText(out, label, (6, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
    return out


def _gs_diff_thumb(frame_bgr, ref_bgr, label=None):
    diff = np.clip(frame_bgr.astype(np.int16) - ref_bgr.astype(np.int16) + 128, 0, 255).astype(np.uint8)
    out = cv2.resize(diff, (GS_THUMB_W, GS_THUMB_H))
    if label:
        cv2.putText(out, label, (6, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
    return out


def build_panel(ep: LoadedEpisode, frame_idx: int,
                gs_ref_L, gs_ref_R,
                paused: bool, speed: int,
                ep_idx: int, n_eps: int,
                project_cams: Optional[list] = None,
                gel_center_left: Optional[np.ndarray] = None,
                gel_center_right: Optional[np.ndarray] = None,
                tactile_latency: int = 0):
    source = ep.source
    h5_frame = ep.h5_frame_for(frame_idx)
    gs_h5_frame = (h5_frame + tactile_latency) if h5_frame is not None else None
    pt_tac_frame = frame_idx + tactile_latency
    if pt_tac_frame < 0 or pt_tac_frame >= ep.n_frames:
        pt_tac_frame = None
    t_sec = float(ep.data["timestamps"][frame_idx] - ep.data["timestamps"][0])

    # Row 1: 3 RealSense cams (left, middle, right) + OptiTrack
    cam_thumbs = []
    cam_labels = ["left cam", "middle cam", "right cam"]
    pt_view_bgr = ep.pt_view(frame_idx) if source == "pt" else None
    for slot, label in zip(DISPLAY_ORDER, cam_labels):
        if source == "pt":
            # Only cam0 lives in the .pt; fill its slot (DISPLAY_ORDER maps it
            # to "right cam"), placeholder the others.
            if slot == 0 and pt_view_bgr is not None:
                cam_thumbs.append(_rs_thumb(pt_view_bgr, f"{label} (pt:view)"))
            else:
                cam_thumbs.append(_missing_cell(RS_THUMB_W, RS_THUMB_H,
                                                f"{label}: pt has cam0 only"))
        else:
            img = ep.h5.cam(slot, h5_frame) if h5_frame is not None else None
            if img is None:
                cam_thumbs.append(_missing_cell(RS_THUMB_W, RS_THUMB_H, f"{label}: no H5"))
            else:
                cam_thumbs.append(_rs_thumb(img, label))
    # OT poses always come from H5 if available, regardless of image source.
    ot_poses = ep.h5.ot_at(h5_frame) if h5_frame is not None else {}
    ot_panel = _make_ot_panel(ot_poses, frame_idx, t_sec, RS_THUMB_W, RS_THUMB_H)
    row1 = np.hstack(cam_thumbs + [ot_panel])

    # Row 2: GelSight raw + diff (L, R) + Controls. Gelsight reads use
    # `frame + tactile_latency` so a measured capture lag is compensated.
    if source == "pt":
        gs_L = ep.pt_tactile("left",  pt_tac_frame) if pt_tac_frame is not None else None
        gs_R = ep.pt_tactile("right", pt_tac_frame) if pt_tac_frame is not None else None
        miss_tag = "no pt"
    else:
        gs_L = ep.h5.gelsight("left",  gs_h5_frame) if gs_h5_frame is not None else None
        gs_R = ep.h5.gelsight("right", gs_h5_frame) if gs_h5_frame is not None else None
        miss_tag = "no H5"
    if gs_L is None:
        gs_L_raw = _missing_cell(GS_THUMB_W, GS_THUMB_H, f"tac_L: {miss_tag}")
        gs_L_dif = _missing_cell(GS_THUMB_W, GS_THUMB_H, f"tac_L diff: {miss_tag}")
    else:
        gs_L_raw = _gs_thumb(gs_L, "tactile_left")
        ref_L = gs_ref_L if gs_ref_L is not None else gs_L
        gs_L_dif = _gs_diff_thumb(gs_L, ref_L, "tac_L diff")
    if gs_R is None:
        gs_R_raw = _missing_cell(GS_THUMB_W, GS_THUMB_H, f"tac_R: {miss_tag}")
        gs_R_dif = _missing_cell(GS_THUMB_W, GS_THUMB_H, f"tac_R diff: {miss_tag}")
    else:
        gs_R_raw = _gs_thumb(gs_R, "tactile_right")
        ref_R = gs_ref_R if gs_ref_R is not None else gs_R
        gs_R_dif = _gs_diff_thumb(gs_R, ref_R, "tac_R diff")
    controls = _make_controls_panel(CONTROLS_W, CONTROLS_H, paused, speed, ep_idx, n_eps,
                                    tactile_latency=tactile_latency)
    row2 = np.hstack([gs_L_raw, gs_L_dif, gs_R_raw, gs_R_dif, controls])

    panel = np.vstack([row1, row2])

    # GelSight-center + axes projection on every cam thumb (matches visualize.py)
    if (project_cams and gel_center_left is not None and gel_center_right is not None
            and ot_poses and _viz_draw_projection_overlay is not None):
        _viz_draw_projection_overlay(
            panel, ot_poses, project_cams, gel_center_left, gel_center_right,
        )

    # Top status bar (visualize.py style: thicker cyan text)
    cv2.rectangle(panel, (0, 0), (PANEL_W, 22), (30, 30, 30), -1)
    state = "PAUSED" if paused else "PLAYING"
    seg = ep.segment_at(frame_idx)
    h5r = seg["source_h5_range"]
    seg_label = f"seg {seg['seg_idx']:02d}"
    if h5r:
        seg_label += f"  H5[{h5r[0]}..{h5r[1]}]"
    if h5_frame is not None:
        seg_label += f"  @{h5_frame}"
    status = (f"[{state}]  ep {ep_idx + 1}/{n_eps}  {ep.key}  |  "
              f"frame {frame_idx + 1}/{ep.n_frames}  |  t={t_sec:.2f}s  |  "
              f"{seg_label}  |  {speed}x  |  tac_lat={tactile_latency:+d}  |  src={source}")
    cv2.putText(panel, status, (10, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)

    # Segment-cut flash: red border for the first 2 frames of a non-first segment
    if ep.is_segment_start(frame_idx):
        cv2.rectangle(panel, (0, 22), (PANEL_W - 1, PANEL_H - 1), (0, 0, 220), 3)

    return panel


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Interactive React .pt player (visualize.py-style layout).")
    ap.add_argument("path", help="A single .pt file OR a directory of episodes/segments.")
    ap.add_argument("--save_video", default=None,
                    help="When `path` is a single .pt: write that one episode as MP4 here.")
    ap.add_argument("--save_video_dir", default=None,
                    help="When `path` is a directory: write one MP4 per episode into this dir.")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--h5_root", default=None,
                    help="Root directory of source H5 files (e.g. <twm>/data/<task>). "
                         "If omitted, inferred by replacing `processed/<mode>` with `data` in `path`.")
    ap.add_argument("--source", choices=("pt", "h5"), default="pt",
                    help="Where to read image cells from. 'pt' = read view/tactile_* from "
                         "the .pt segment files (default, ~1000x faster, but the .pt only "
                         "carries cam0 + both gelsights at 128x128). 'h5' = read full-res "
                         "480x640 frames from the source H5 (original behavior).")
    # ── GelSight-center projection overlay (mirrors visualize.py) ────────────
    ap.add_argument("--cam_calib", type=str, nargs="+", default=None,
                    help="Path(s) to T_mocap_to_cam_<name>.json (one per "
                         "camera); default: the input's task epoch via calib_epoch.")
    ap.add_argument("--gel_left", type=str, default=None,
                    help="Path to T_gel_to_rigid_left.json (default: task epoch).")
    ap.add_argument("--gel_right", type=str, default=None,
                    help="Path to T_gel_to_rigid_right.json (default: task epoch).")
    ap.add_argument("--no_projection", action="store_true",
                    help="Skip the GelSight-center projection overlay "
                         "(and calibration loading). Default: overlay ON.")
    ap.add_argument("--tactile_latency", type=int, default=3,
                    help="Frames to advance gelsight reads (h5_frame + N) to "
                         "compensate for tactile capture lag. Default: 3.")
    args = ap.parse_args()

    if not args.no_projection and None in (args.gel_left, args.gel_right,
                                           args.cam_calib):
        from twm.calib_epoch import calib_dir_for_path
        cdir = calib_dir_for_path(args.path)   # raises rather than guessing
        print(f"calibration epoch: {cdir.name} (from input path)")
        if args.cam_calib is None:
            args.cam_calib = [str(cdir / f"T_mocap_to_cam_{n}.json")
                              for n in ("middle", "left", "right")]
        args.gel_left = args.gel_left or str(cdir / "T_gel_to_rigid_left.json")
        args.gel_right = args.gel_right or str(cdir / "T_gel_to_rigid_right.json")

    in_path = Path(args.path)

    # Path-stem expansion: if the user typed an incomplete path like
    # `.../2026-05-11/episode_005` (no .pt suffix, possibly no .segment_NN),
    # treat it as an episode-stem filter. We load the parent date folder so
    # N/P navigation still works, then start playback at the matching episode.
    start_episode_key: Optional[str] = None
    if not in_path.exists():
        parent = in_path.parent
        stem = in_path.name
        if parent.is_dir() and stem:
            matches = sorted(parent.glob(f"{stem}*.pt"))
            # Require either an exact-name match or segment_NN siblings
            matches = [p for p in matches
                       if PLAIN_EPISODE_RE.match(p.name) and p.stem == stem
                       or (SEGMENT_FNAME_RE.match(p.name)
                           and SEGMENT_FNAME_RE.match(p.name).group(1) == stem)]
            if matches:
                start_episode_key = f"{parent.name}/{stem}"
                in_path = parent  # discover the whole date folder
                print(f"[player] interpreting '{args.path}' as episode-stem; "
                      f"will start at {start_episode_key}")
    if not in_path.exists():
        print(f"Not found: {args.path}", file=sys.stderr); sys.exit(1)

    if in_path.is_file():
        parent_root = in_path.parent.parent
        episodes = discover_episodes(parent_root)
        target_path = in_path.resolve()
        start_idx = 0
        for i, ep_meta in enumerate(episodes):
            if any(p.resolve() == target_path for _, p in ep_meta["segs"]):
                start_idx = i; break
    else:
        episodes = discover_episodes(in_path)
        start_idx = 0
        if start_episode_key is not None:
            for i, ep_meta in enumerate(episodes):
                if ep_meta["key"] == start_episode_key:
                    start_idx = i; break
            else:
                print(f"[player] WARN: stem '{start_episode_key}' not found in discovered "
                      f"episodes; starting at the first one")
    print(f"[player] discovered {len(episodes)} episode(s) under {in_path}")

    # Resolve H5 root
    h5_root: Optional[Path] = None
    if args.h5_root:
        h5_root = Path(args.h5_root)
    else:
        h5_root = infer_h5_root(in_path)
    if h5_root:
        ok = "ok" if h5_root.exists() else "missing"
        print(f"[player] H5 source root: {h5_root}  ({ok})")
    else:
        print("[player] H5 source root not specified and could not be inferred -- "
              "RealSense + GelSight cells will be blank")

    # ── Load projection calibrations (mirrors visualize.py) ──────────────────
    project_cams: list = []
    gel_center_left = None
    gel_center_right = None
    if args.no_projection or _viz_load_calibrations is None:
        if args.no_projection:
            print("[player] Projection overlay: OFF (--no_projection)")
        else:
            print("[player] Projection overlay: OFF (twm.viz unavailable)")
    else:
        try:
            cam_calibs, gel_center_left, gel_center_right = \
                _viz_load_calibrations(args.cam_calib, args.gel_left, args.gel_right)
            for calib in cam_calibs:
                serial = calib["camera_serial"]
                try:
                    c_idx = REALSENSE_SERIALS.index(serial)
                except ValueError:
                    print(f"[player] WARN: camera serial {serial} not in REALSENSE_SERIALS, skipping")
                    continue
                project_cams.append({
                    "index":          c_idx,
                    "T_mocap_to_cam": calib["T_mocap_to_cam"],
                    "intrinsics":     calib["intrinsics"],
                    "serial":         serial,
                    "rmse":           calib["rmse_mm"],
                })
            if not project_cams:
                print("[player] Projection overlay: OFF (no usable camera calibrations found -- "
                      "check --cam_calib paths)")
            else:
                print(f"[player] Projection overlay: ON ({len(project_cams)} camera(s))")
            for pc in project_cams:
                print(f"    cam{pc['index']}  serial={pc['serial']}  RMSE={pc['rmse']:.2f} mm")
            if gel_center_left is not None and gel_center_right is not None:
                print(f"    gel_L center (rigid): "
                      f"[{gel_center_left[0]:.2f}, {gel_center_left[1]:.2f}, {gel_center_left[2]:.2f}] mm")
                print(f"    gel_R center (rigid): "
                      f"[{gel_center_right[0]:.2f}, {gel_center_right[1]:.2f}, {gel_center_right[2]:.2f}] mm")
        except Exception as e:
            print(f"[player] WARN: calibration loading failed -- projection overlay disabled ({e})")
            project_cams = []
            gel_center_left = None
            gel_center_right = None

    headless_dir = Path(args.save_video_dir) if args.save_video_dir else None
    headless_single = args.save_video
    if headless_single and len(episodes) > 1 and in_path.is_dir():
        print("--save_video accepts a single output path; for a directory pass --save_video_dir instead.",
              file=sys.stderr)
        sys.exit(1)
    headless = headless_single is not None or headless_dir is not None

    if headless:
        if headless_dir:
            headless_dir.mkdir(parents=True, exist_ok=True)
        for ep_idx in range(start_idx, len(episodes)):
            ep = LoadedEpisode(episodes[ep_idx], h5_root=h5_root,
                               tactile_latency=args.tactile_latency,
                               source=args.source)
            try:
                out_path = (Path(headless_single)
                            if headless_single else
                            (headless_dir / f"{ep.date}_{ep.ep_stem}.mp4"))
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(out_path), fourcc, args.fps, (PANEL_W, PANEL_H))
                print(f"  -> {out_path}")
                for f in range(ep.n_frames):
                    panel = build_panel(ep, f, ep.gs_ref_L, ep.gs_ref_R,
                                        paused=False, speed=1,
                                        ep_idx=ep_idx, n_eps=len(episodes),
                                        project_cams=project_cams,
                                        gel_center_left=gel_center_left,
                                        gel_center_right=gel_center_right,
                                        tactile_latency=args.tactile_latency)
                    writer.write(panel)
                writer.release()
            finally:
                ep.close()
            if headless_single:
                break
        return

    # Interactive
    paused, speed = False, 1
    SPEEDS = {ord('1'): 1, ord('2'): 2, ord('3'): 5,
              ord('4'): 10, ord('5'): 25, ord('6'): 50}
    tactile_latency = int(args.tactile_latency)   # mutable; adjustable via [ / ]

    ep_idx = start_idx
    while 0 <= ep_idx < len(episodes):
        ep = LoadedEpisode(episodes[ep_idx], h5_root=h5_root,
                           tactile_latency=tactile_latency,
                           source=args.source)
        action = "stay"
        try:
            WIN = "React .pt player"
            cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
            cv2.createTrackbar("Frame", WIN, 0, max(1, ep.n_frames - 1), lambda v: None)
            gs_ref_L, gs_ref_R = ep.gs_ref_L, ep.gs_ref_R
            frame_idx = 0
            last_drawn = -1

            while True:
                frame_idx = max(0, min(frame_idx, ep.n_frames - 1))
                pos = cv2.getTrackbarPos("Frame", WIN)
                if pos != frame_idx and pos != last_drawn:
                    frame_idx = pos; paused = True

                panel = build_panel(ep, frame_idx,
                                    gs_ref_L, gs_ref_R,
                                    paused=paused, speed=speed,
                                    ep_idx=ep_idx, n_eps=len(episodes),
                                    project_cams=project_cams,
                                    gel_center_left=gel_center_left,
                                    gel_center_right=gel_center_right,
                                    tactile_latency=tactile_latency)
                cv2.imshow(WIN, panel)
                if cv2.getTrackbarPos("Frame", WIN) != frame_idx:
                    cv2.setTrackbarPos("Frame", WIN, frame_idx)
                last_drawn = frame_idx

                key = cv2.waitKey(1) & 0xFF
                if   key == ord('q'): action = "quit"; break
                elif key == ord(' '): paused = not paused
                elif key in (81, ord('a')): paused = True; frame_idx -= 1
                elif key in (83, ord('d')): paused = True; frame_idx += 1
                elif key in SPEEDS: speed = SPEEDS[key]
                elif key == ord('r'):
                    if ep.source == "pt":
                        shifted = max(0, min(ep.n_frames - 1, frame_idx + tactile_latency))
                        new_L = ep.pt_tactile("left",  shifted)
                        new_R = ep.pt_tactile("right", shifted)
                        if new_L is not None: gs_ref_L = new_L
                        if new_R is not None: gs_ref_R = new_R
                        print(f"  gel-diff ref reset to concat frame {frame_idx}  "
                              f"(pt idx {shifted}, latency {tactile_latency:+d})")
                    else:
                        h5_frame = ep.h5_frame_for(frame_idx)
                        if h5_frame is not None and ep.h5.ok:
                            shifted = h5_frame + tactile_latency
                            gs_ref_L = ep.h5.gelsight("left",  shifted)
                            gs_ref_R = ep.h5.gelsight("right", shifted)
                            print(f"  gel-diff ref reset to concat frame {frame_idx}  "
                                  f"(H5 frame {h5_frame}, gs idx {shifted})")
                elif key == ord('['):
                    tactile_latency -= 1
                    print(f"  tactile_latency = {tactile_latency:+d}")
                elif key == ord(']'):
                    tactile_latency += 1
                    print(f"  tactile_latency = {tactile_latency:+d}")
                elif key == ord('n'): action = "next"; break
                elif key == ord('p'): action = "prev"; break

                if not paused:
                    frame_idx += speed
                    if frame_idx >= ep.n_frames:
                        print(f"End of episode {ep.key}.")
                        paused = True
                        frame_idx = ep.n_frames - 1
                    time.sleep(max(0.0, 1.0 / (args.fps * speed) - 0.001))

            cv2.destroyAllWindows()
        finally:
            ep.close()
        if action == "quit": return
        if action == "next":
            new_idx = min(ep_idx + 1, len(episodes) - 1)
            if new_idx == ep_idx:
                print("Already at last episode.")
            ep_idx = new_idx
        elif action == "prev":
            new_idx = max(ep_idx - 1, 0)
            if new_idx == ep_idx:
                print("Already at first episode.")
            ep_idx = new_idx
        else:
            paused = True


if __name__ == "__main__":
    main()
