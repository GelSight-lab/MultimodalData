"""ReactVideoDataset — load the React multi-task video-format release.

Layout consumed (per task):
    data/<task>/videos/<date>/episode_NNN/{view_left,view_middle,view_right,
                                           tactile_left,tactile_right}.mp4
    data/<task>/meta/<date>/episode_NNN.parquet   # per-frame pose/scalar
    data/<task>/segments.json                     # clean-segment index
    data/<task>/bad_frames.json                   # quality intervals
    data/<task>/episodes.jsonl                    # per-episode summary
    data/<task>/calibration/                      # extrinsics (May-12 / June-26)

Decoded frames are returned as RGB uint8 (H, W, 3) — standard video-decoder
convention. (cv2 users: this is already RGB, do NOT re-swap.)

Two sampling modes:
    mode="segment": iterate clean segments from segments.json (no bad frames
                    by construction). RECOMMENDED.
    mode="window":  sliding windows over whole episodes; windows overlapping
                    bad_frames.json intervals are skipped when skip_bad=True.

Decoder backend: PyAV (`av`) by default; falls back to OpenCV. Install
`decord` for fastest random access.

Example
-------
    ds = ReactVideoDataset("data/motherboard", window_length=16, mode="segment")
    sample = ds[0]
    # sample["view_middle"]: (T, H, W, 3) uint8 RGB
    # sample["sensor_left_pose"]: (T, 7) float32
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

try:
    import av
    _BACKEND = "av"
except Exception:
    import cv2
    _BACKEND = "cv2"

VIEW_STREAMS = ("view_left", "view_middle", "view_right")
TACTILE_STREAMS = ("tactile_left", "tactile_right")
ALL_STREAMS = VIEW_STREAMS + TACTILE_STREAMS
DEPTH_STREAMS = ("depth_left", "depth_middle", "depth_right")   # optional, uint16 mm


def _decode_frames(mp4_path: Path, frame_indices, depth=False):
    """Return (N, H, W, 3) uint8 RGB, or (N, H, W) uint16 mm if depth=True."""
    want = list(frame_indices)
    fmt = None if depth else "rgb24"     # depth: native gray16le ndarray
    if _BACKEND == "av" or depth:        # depth requires PyAV (16-bit)
        container = av.open(str(mp4_path))
        stream = container.streams.video[0]
        out, wantset, got = {}, set(want), 0
        for fi, frame in enumerate(container.decode(stream)):
            if fi in wantset:
                a = frame.to_ndarray(format=fmt) if fmt else frame.to_ndarray()
                out[fi] = a
                got += 1
                if got == len(wantset):
                    break
        container.close()
        return np.stack([out[i] for i in want])
    else:  # cv2 fallback (BGR -> RGB)
        cap = cv2.VideoCapture(str(mp4_path))
        frames = []
        for i in want:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, fr = cap.read()
            frames.append(fr[..., ::-1] if ok else np.zeros((480, 640, 3), np.uint8))
        cap.release()
        return np.stack(frames)


class ReactVideoDataset:
    def __init__(self, task_root, window_length=16, stride=1, window_step=None,
                 mode="segment", streams=ALL_STREAMS, skip_bad=True,
                 which_sensors="any", load_depth=False, tactile_latency=0):
        self.root = Path(task_root)
        self.W = window_length
        self.stride = stride
        self.step = window_step or window_length
        self.mode = mode
        self.streams = tuple(streams)
        self.skip_bad = skip_bad
        self.which = which_sensors
        # depth only if requested AND present on disk for this task
        self.load_depth = load_depth and (self.root / "depth").is_dir()
        # GelSight acquisition lag (frames): tactile stream was captured
        # `tactile_latency` frames BEFORE the view at the same index, due to a
        # recording-side V4L2 buffer bug (fixed in the rig from 2026-06-27).
        # When >0, the loader pairs view[i] with tactile[i+latency] (and the
        # tactile contact scalars likewise), and trims `latency` frames from
        # the end of each window range so the shifted index stays in bounds.
        self.tactile_latency = int(tactile_latency)
        self._TACT = ("tactile_left", "tactile_right")
        self._TACT_COLS = ("tactile_left_intensity", "tactile_right_intensity",
                           "tactile_left_mixed", "tactile_right_mixed")

        self.segments = json.loads((self.root / "segments.json").read_text())["segments"]
        self.bad = json.loads((self.root / "bad_frames.json").read_text())["episodes"]
        self.index = self._build_index()

    def _video_dir(self, ep_key):
        date, ep = ep_key.split("/")
        return self.root / "videos" / date / ep

    def _parquet(self, ep_key):
        date, ep = ep_key.split("/")
        return self.root / "meta" / date / f"{ep}.parquet"

    def _bad_mask(self, ep_key, T):
        m = np.zeros(T, bool)
        e = self.bad.get(ep_key, {})
        for k in ("intensity_spikes", "pose_teleports_L", "pose_teleports_R",
                  "ot_loss_L", "ot_loss_R"):
            for a, b in e.get(k, []):
                m[max(0, a):min(T, b + 1)] = True
        return m

    def _build_index(self):
        items = []
        span = (self.W - 1) * self.stride + 1
        lat = self.tactile_latency        # tactile read at idx+lat must stay in bounds
        if self.mode == "segment":
            for s in self.segments:
                ek, a, b = s["source_episode"], s["frame_range"][0], s["frame_range"][1]
                start = a
                while start + span - 1 + lat <= b:
                    items.append((ek, start))
                    start += self.step
        else:  # window over whole episode
            eps = sorted({s["source_episode"] for s in self.segments})
            for ek in eps:
                T = self.bad.get(ek, {}).get("n_frames", 0)
                bad = self._bad_mask(ek, T) if self.skip_bad else np.zeros(T, bool)
                start = 0
                while start + span - 1 + lat < T:
                    idx = range(start, start + span, self.stride)
                    if not (self.skip_bad and bad[list(idx)].any()):
                        items.append((ek, start))
                    start += self.step
        return items

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        ek, start = self.index[i]
        lat = self.tactile_latency
        idx = list(range(start, start + (self.W - 1) * self.stride + 1, self.stride))
        idx_tac = [r + lat for r in idx]        # tactile is `lat` frames behind view
        vd = self._video_dir(ek)
        out = {}
        for s in self.streams:
            read_idx = idx_tac if s in self._TACT else idx   # shift only tactile
            out[s] = _decode_frames(vd / f"{s}.mp4", read_idx)
        if self.load_depth:                      # depth is a view-side cam, no shift
            date, ep = ek.split("/")
            dd = self.root / "depth" / date / ep
            for s in DEPTH_STREAMS:
                p = dd / f"{s}.mkv"
                if p.exists():
                    out[s] = _decode_frames(p, idx, depth=True)   # (T,H,W) uint16 mm
        # parquet: read a range covering both idx and idx_tac
        lo, hi = start, idx_tac[-1]
        tbl = pq.read_table(self._parquet(ek)).slice(lo, hi - lo + 1)
        v_rows = [r - lo for r in idx]
        t_rows = [r - lo for r in idx_tac]
        for c in ("sensor_left_pose", "sensor_right_pose"):
            out[c] = np.array(tbl.column(c).to_pylist(), np.float32)[v_rows]
        if "object_pose" in tbl.column_names:
            out["object_pose"] = np.array(tbl.column("object_pose").to_pylist(), np.float32)[v_rows]
        # tactile contact scalars follow the tactile frames -> shifted rows
        for c in self._TACT_COLS:
            out[c] = np.array(tbl.column(c).to_pylist(), np.float32)[t_rows]
        out["episode"] = ek
        out["frame_start"] = start
        out["tactile_latency"] = lat
        return out


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "data/motherboard"
    ds = ReactVideoDataset(root, window_length=8, mode="segment")
    print(f"backend={_BACKEND}  {len(ds)} windows")
    s = ds[0]
    for k, v in s.items():
        shape = getattr(v, "shape", v)
        print(f"  {k}: {shape}")
