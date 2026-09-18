"""Published streams retain source geometry, alignment, and pixel semantics."""
import shutil
import subprocess
from types import SimpleNamespace

import cv2
import h5py
import numpy as np
import pytest

from twm.react_preprocess import pipeline, tactile
from twm.wrist_tone import apply_tone_curve

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")


def _decode(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    return np.stack(frames)


@pytest.mark.parametrize("single_pass", [False, True])
@pytest.mark.parametrize("encoding", ["bgr8", "mjpeg"])
def test_colour_streams_use_each_sources_dimensions(tmp_path, monkeypatch, single_pass, encoding):
    camera = np.full((5, 24, 32, 3), (30, 80, 150), np.uint8)
    wrist = np.stack([np.full((18, 26, 3), 30 + i * 20, np.uint8) for i in range(5)])
    monkeypatch.setattr(pipeline, "CHUNK", 2)
    with h5py.File(tmp_path / "source.h5", "w") as f:
        f["realsense/cam0/color"] = camera
        f.create_group("metadata")
        if encoding == "mjpeg":
            ds = f.create_dataset("arducam/cam0/frames", (5,), dtype=h5py.vlen_dtype(np.uint8))
            for i, frame in enumerate(wrist):
                ok, jpeg = cv2.imencode(".jpg", frame)
                assert ok
                ds[i] = jpeg.ravel()
        else:
            f["arducam/cam0/frames"] = wrist
        source = SimpleNamespace(T=3, trim=1, task="pushT")
        if single_pass:
            gammas = pipeline._encode_rgb_single_pass(f, source, tmp_path)
        else:
            pipeline._encode_cameras(f, source, tmp_path)
            gammas = pipeline._encode_wrist(f, source, tmp_path)
    decoded = _decode(tmp_path / "view_right.mp4")
    assert decoded.shape == (3, 24, 32, 3)
    np.testing.assert_allclose(decoded.mean(axis=(0, 1, 2)), [30, 80, 150], atol=3)
    decoded = _decode(tmp_path / "wrist_left.mp4")
    assert decoded.shape == (3, 18, 26, 3)
    expected = apply_tone_curve(wrist[1:4], gammas["cam0"])
    np.testing.assert_allclose(decoded.mean(axis=(1, 2, 3)), expected.mean(axis=(1, 2, 3)), atol=3)


def test_depth_uses_source_dimensions_and_is_lossless(tmp_path):
    depth = np.arange(5 * 18 * 26, dtype=np.uint16).reshape(5, 18, 26)
    source = SimpleNamespace(T=3, trim=1)
    assert pipeline._encode_depth({"realsense/cam0/depth": depth}, source, tmp_path) == 1
    data = subprocess.check_output([
        "ffmpeg", "-v", "error", "-i", str(tmp_path / "depth_right.mkv"),
        "-f", "rawvideo", "-pix_fmt", "gray16le", "-",
    ])
    decoded = np.frombuffer(data, dtype="<u2").reshape(3, 18, 26)
    np.testing.assert_array_equal(decoded, depth[1:4])


def test_tactile_uses_source_dimensions_and_aligned_rgb_frames(tmp_path):
    frames = np.stack([np.full((18, 26, 3), (30 + i * 20, 80, 150), np.uint8)
                       for i in range(4)])
    align = SimpleNamespace(index_map=np.array([1, 1, 3]))
    path = tmp_path / "tactile.mp4"
    tactile.process_side({"gelsight/left/frames": frames}, "left", align, path)
    decoded = _decode(path)
    assert decoded.shape == (3, 18, 26, 3)
    np.testing.assert_allclose(decoded.mean(axis=(1, 2)),
                               frames[align.index_map, ..., ::-1].mean(axis=(1, 2)), atol=3)
