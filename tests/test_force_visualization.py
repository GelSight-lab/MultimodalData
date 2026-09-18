"""Synthetic force figures use the same capture map as preprocessing."""
import subprocess
import json
import sys

import cv2
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


@pytest.fixture
def episode(tmp_path, monkeypatch):
    from twm.force_recovery import visualize as viz

    task, date, ep = "test", "2026-09-18", "episode_001"
    for name, folder in (("DATA_ROOT", "data"), ("STAGE_ROOT", "stage"),
                         ("OUT_ROOT", "force"), ("ASSETS", "assets")):
        monkeypatch.setattr(viz, name, tmp_path / folder)
    path = viz.DATA_ROOT / task / date / f"{ep}.h5"
    path.parent.mkdir(parents=True)
    with h5py.File(path, "w") as f:
        f["timestamps"] = np.arange(6, dtype=float)
        f["optitrack/sensor_left/timestamps"] = [2.0]
        f["optitrack/sensor_left/pose"] = np.zeros((1, 7))
        for side in ("left", "right"):
            frames = np.zeros((7, 480, 640, 3), np.uint8)
            frames[:, :, :, 0] = np.arange(7)[:, None, None] * 20
            f[f"gelsight/{side}/frames"] = frames
    force = np.array([0.1, 0.4, 0.2, 0.1])
    meta = viz.STAGE_ROOT / task / "meta" / date / f"{ep}.parquet"
    meta.parent.mkdir(parents=True)
    pq.write_table(pa.table({"tactile_left_intensity": force,
                            "tactile_left_is_new": [True] * 4,
                            "sensor_left_pose": np.zeros((4, 7)).tolist()}), meta)
    npz_path = viz.OUT_ROOT / task / date / f"{ep}_left.npz"
    npz_path.parent.mkdir(parents=True)
    arrays = dict(force_normal_n=force, trim=2, reference_rows=np.array([0]),
                  depth_row_1=np.ones((2, 3)), depth_row_3=np.ones((2, 3)))
    np.savez(npz_path, **arrays)
    return viz, (task, date, ep, "left"), path, npz_path, arrays


def test_difference_helpers_do_not_import_plotting_arrow_or_inference():
    script = """
import sys
from twm.force_recovery.visualize import diff_rgb, diff_caption
for name in ('matplotlib', 'pyarrow', 'torch', 'h5py',
             'twm.force_recovery.run_episode', 'twm.force_recovery.dexforce'):
    assert name not in sys.modules, name
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("timestamped", [False, True])
def test_depth_panels_read_authoritative_capture_map(episode, monkeypatch, timestamped):
    viz, args, path, _, _ = episode
    if timestamped:
        with h5py.File(path, "a") as f:
            f["gelsight/left/timestamps"] = [2., 3., 3.4, 4., 5., 6., 7.]
    from twm.react_preprocess.h5io import open_episode
    expected = open_episode(path, args[0]).align["left"].index_map
    from matplotlib.axes import Axes
    captured = []
    original = Axes.imshow

    def imshow(self, data, *a, **kw):
        captured.append(np.asarray(data).copy())
        return original(self, data, *a, **kw)

    monkeypatch.setattr(Axes, "imshow", imshow)
    viz.ASSETS.mkdir()
    assert viz.depth_panels(*args).exists()
    for i, row in enumerate((1, 3)):
        assert captured[3 * i][0, 0, 0] == expected[row] * 20
        np.testing.assert_allclose(captured[3 * i + 1],
                                   abs(expected[row] - expected[0]) * 20 / 3)


def test_depth_panels_reject_alignment_length_mismatch(episode):
    viz, args, _, npz_path, arrays = episode
    arrays["force_normal_n"] = np.zeros(3)
    np.savez(npz_path, **arrays)
    with pytest.raises(ValueError, match="length|rows"):
        viz.depth_panels(*args)


def test_depth_panels_reject_empty_depths(episode):
    viz, args, _, npz_path, arrays = episode
    np.savez(npz_path, **{k: v for k, v in arrays.items() if not k.startswith("depth_row_")})
    with pytest.raises(ValueError, match="depth"):
        viz.depth_panels(*args)


def _legacy_panel(image, force, row):
    """Independent reference for the established 640x560 clip layout."""
    canvas = np.zeros((560, 640, 3), np.uint8)
    canvas[:480] = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    fmax = max(float(force.max()), 1e-3)
    cv2.rectangle(canvas, (40, 520), (600, 540), (60, 60, 60), 1)
    cv2.rectangle(canvas, (40, 520), (40 + int(560 * (force[row] / fmax)), 540),
                  (30, 120, 240), -1)
    cv2.putText(canvas, f"F_n = {force[row]:.3f} N", (40, 512),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
    xs = np.linspace(40, 600, len(force)).astype(int)
    ys = (556 - 12 * force / fmax).astype(int)
    for i in range(1, len(xs)):
        cv2.line(canvas, (xs[i - 1], ys[i - 1]), (xs[i], ys[i]), (140, 140, 140), 1)
    cv2.circle(canvas, (xs[row], ys[row]), 3, (30, 120, 240), -1)
    return canvas


@pytest.mark.parametrize("timestamped", [False, True])
def test_overlay_replays_shared_export_and_preserves_pixels(episode, monkeypatch, timestamped):
    viz, args, path, _, arrays = episode
    if timestamped:
        with h5py.File(path, "a") as f:
            f["gelsight/left/timestamps"] = [2., 3., 3.4, 4., 5., 6., 7.]
    from twm.react_preprocess.h5io import open_episode
    from twm.visualization import export
    idx = open_episode(path, args[0]).align["left"].index_map
    with h5py.File(path, "r") as f:
        expected = [_legacy_panel(f["gelsight/left/frames"][int(src)],
                                  arrays["force_normal_n"], row)
                    for row, src in enumerate(idx)]
    calls = []

    def write_video(out, factory, *, fps, pixel_format="yuv444p"):
        calls.append((out, fps, pixel_format))
        for _ in range(2):
            actual = list(factory())
            assert len(actual) == len(expected)
            for frame, reference in zip(actual, expected):
                np.testing.assert_array_equal(frame, reference)
        return len(expected)

    monkeypatch.setattr(export, "write_video", write_video)
    monkeypatch.setattr(cv2, "VideoWriter", lambda *a, **kw: pytest.fail("double encoding"))
    out = viz.overlay_clip(*args, force=arrays["force_normal_n"])
    assert calls == [(out, 30, "yuv420p")]


def test_overlay_encodes_browser_compatible_h264(episode):
    viz, args, _, _, arrays = episode
    out = viz.overlay_clip(*args, force=arrays["force_normal_n"])
    result = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries",
        "stream=pix_fmt,profile,width,height,nb_frames", "-of", "json", str(out),
    ]))["streams"][0]
    assert result == {"pix_fmt": "yuv420p", "profile": "High", "width": 640,
                      "height": 560, "nb_frames": "4"}


@pytest.mark.parametrize("force", [np.array([]), np.zeros((4, 1)), np.zeros(3),
                                   np.array([0., 1., np.nan, 0.])])
def test_overlay_rejects_invalid_force(episode, force):
    viz, args, *_ = episode
    with pytest.raises(ValueError, match="force|Force"):
        viz.overlay_clip(*args, force=force)


def test_overlay_closes_h5_when_encoder_stops_early(episode, monkeypatch):
    viz, args, path, _, arrays = episode
    from twm.visualization import export
    opened = []
    original = h5py.File

    def tracked(*a, **kw):
        handle = original(*a, **kw)
        opened.append(handle)
        return handle

    def stop_encoding(path, frames, fps, pixel_format):
        next(frames)
        raise RuntimeError("encoder stopped")

    monkeypatch.setattr(h5py, "File", tracked)
    monkeypatch.setattr(export, "_encode", stop_encoding)
    with pytest.raises(RuntimeError, match="encoder stopped"):
        viz.overlay_clip(*args, force=arrays["force_normal_n"])
    assert opened
    assert all(not handle.id.valid for handle in opened)
    assert not list(viz.ASSETS.glob("*.mp4"))


def test_empty_metadata_reports_empty_force_before_loading_images(episode):
    viz, args, _, npz_path, arrays = episode
    task, date, ep, _ = args
    path = viz.STAGE_ROOT / task / "meta" / date / f"{ep}.parquet"
    table = pq.read_table(path)
    pq.write_table(table.slice(0, 0), path)
    arrays["force_normal_n"] = np.array([])
    np.savez(npz_path, **arrays)
    with pytest.raises(ValueError, match="[Ff]orce"):
        viz.overlay_clip(*args, force=np.array([]))


def test_overlay_rejects_invalid_image_shape_and_closes_h5(episode, monkeypatch):
    viz, args, path, _, arrays = episode
    from twm.visualization import export
    with h5py.File(path, "a") as f:
        del f["gelsight/left/frames"]
        f["gelsight/left/frames"] = np.zeros((7, 40, 50), np.uint8)
    opened = []
    original = h5py.File

    def tracked(*a, **kw):
        handle = original(*a, **kw)
        opened.append(handle)
        return handle

    monkeypatch.setattr(h5py, "File", tracked)
    with pytest.raises(ValueError, match="Color images"):
        viz.overlay_clip(*args, force=arrays["force_normal_n"])
    assert all(not handle.id.valid for handle in opened)


def test_timeline_draw_work_is_linear_and_pixels_match(monkeypatch):
    from twm.visualization.force import ForceOverlay
    force = np.linspace(0., 1., 30)
    image = np.zeros((480, 640, 3), np.uint8)
    original = cv2.line
    calls = []

    def line(*a, **kw):
        calls.append(True)
        return original(*a, **kw)

    monkeypatch.setattr(cv2, "line", line)
    overlay = ForceOverlay(force, maximum=1.)
    actual = [overlay.render(image, row) for row in range(len(force))]
    assert len(calls) == len(force) - 1
    monkeypatch.setattr(cv2, "line", original)
    for row, frame in enumerate(actual):
        np.testing.assert_array_equal(frame, _legacy_panel(image, force, row))
