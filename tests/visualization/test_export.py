"""Small real-encoder tests for streaming, cleanup and retry behavior."""
import shutil
import json
import subprocess
import weakref

import cv2
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")


def test_export_consumes_frames_lazily_and_preserves_order(tmp_path):
    from twm.visualization.export import write_video

    refs = []

    def frames():
        for value in (20, 80, 140, 200):
            # A list-based encoder retains all four arrays. A streaming encoder
            # needs at most the last yielded array while requesting the next.
            assert sum(ref() is not None for ref in refs) <= 2
            frame = np.full((32, 48, 3), value, np.uint8)
            refs.append(weakref.ref(frame))
            yield frame

    path = tmp_path / "clip.mp4"
    assert write_video(path, frames, fps=15) == 4
    cap = cv2.VideoCapture(str(path))
    values = []
    try:
        assert cap.get(cv2.CAP_PROP_FPS) == 15
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            values.append(frame.mean())
    finally:
        cap.release()
    np.testing.assert_allclose(values, [20, 80, 140, 200], atol=3)


@pytest.mark.parametrize("failure", ["empty", "shape", "producer"])
def test_failed_export_preserves_existing_destination_and_cleans_temp(tmp_path, failure):
    from twm.visualization.export import write_video

    path = tmp_path / "clip.mp4"
    path.write_bytes(b"existing output")

    def frames():
        if failure == "empty":
            return
        yield np.zeros((32, 48, 3), np.uint8)
        if failure == "producer":
            raise RuntimeError("producer failed")
        yield np.zeros((30, 48, 3), np.uint8)

    with pytest.raises((ValueError, RuntimeError)):
        write_video(path, frames, fps=30)
    assert path.read_bytes() == b"existing output"
    assert list(tmp_path.iterdir()) == [path]


def test_failed_decode_verification_replays_factory(monkeypatch, tmp_path):
    from twm.visualization import export

    calls = []
    counts = iter((0, 2))
    monkeypatch.setattr(export, "decoded_frame_count", lambda path: next(counts))

    def frames():
        calls.append(True)
        yield np.zeros((32, 48, 3), np.uint8)
        yield np.ones((32, 48, 3), np.uint8)

    assert export.write_video(tmp_path / "clip.mp4", frames, fps=30) == 2
    assert len(calls) == 2


@pytest.mark.parametrize("fps", [0, -1, float("nan"), float("inf")])
def test_rejects_invalid_fps_before_consuming_source(tmp_path, fps):
    from twm.visualization.export import write_video

    def frames():
        pytest.fail("invalid configuration must not open source")

    with pytest.raises(ValueError, match="fps"):
        write_video(tmp_path / "clip.mp4", frames, fps=fps)


def test_episode_preview_streams_a_replayable_factory(monkeypatch, tmp_path):
    from twm.scripts import build_episode_previews as previews

    calls = []

    def panels(*args, **kwargs):
        calls.append((args, kwargs))
        yield np.zeros((32, 48, 3), np.uint8)

    monkeypatch.setattr(previews, "iter_preview_panels", panels, raising=False)
    previews.build_one_preview(
        tmp_path / "not-opened.h5", tmp_path / "clip.mp4", 4, 0.5,
        [], None, None, window_start=123, show_virtual_targets=False,
    )
    assert len(calls) == 1
    assert calls[0][1]["window_start"] == 123
    assert calls[0][1]["show_virtual_targets"] is False


@pytest.mark.parametrize("pixel_format,profile", [(None, "High 4:4:4 Predictive"),
                                                   ("yuv420p", "High")])
def test_export_preserves_selected_pixel_format(tmp_path, pixel_format, profile):
    from twm.visualization.export import write_video

    path = tmp_path / "clip.mp4"
    options = {} if pixel_format is None else {"pixel_format": pixel_format}
    assert write_video(path, lambda: iter([np.zeros((32, 48, 3), np.uint8)]),
                       **options) == 1
    result = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "stream=pix_fmt,profile",
        "-of", "json", str(path),
    ]))["streams"][0]
    assert result == {"pix_fmt": pixel_format or "yuv444p", "profile": profile}


def test_invalid_pixel_format_does_not_open_source_or_touch_destination(tmp_path):
    from twm.visualization.export import write_video

    path = tmp_path / "clip.mp4"
    path.write_bytes(b"previous")

    def frames():
        pytest.fail("invalid configuration must not open source")

    with pytest.raises(ValueError, match="pixel.format"):
        write_video(path, frames, pixel_format="not-a-format")
    assert path.read_bytes() == b"previous"
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("shape", [(31, 48, 3), (32, 47, 3)])
def test_420_rejects_odd_dimensions_and_closes_source(tmp_path, shape):
    from twm.visualization.export import write_video

    path = tmp_path / "clip.mp4"
    path.write_bytes(b"previous")
    closed = []

    def frames():
        try:
            yield np.zeros(shape, np.uint8)
            pytest.fail("invalid geometry must fail on first frame")
        finally:
            closed.append(True)

    with pytest.raises(ValueError, match="yuv420p.*even"):
        write_video(path, frames, pixel_format="yuv420p")
    assert closed == [True]
    assert path.read_bytes() == b"previous"
    assert list(tmp_path.iterdir()) == [path]
