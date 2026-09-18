"""Identical selection/error contracts for optional video decoders."""
from types import SimpleNamespace
import subprocess
import sys

import numpy as np
import pytest

from twm.react_toolbox import io


@pytest.fixture(params=["av", "cv2"])
def decoder(request, monkeypatch):
    images = [np.full((4, 6, 3), i, np.uint8) for i in range(3)]
    state = SimpleNamespace(closed=False, reads=0, fail=False)

    def read_images():
        for img in images:
            state.reads += 1
            if state.fail:
                raise RuntimeError("decode failed")
            yield img

    def close():
        state.closed = True

    if request.param == "av":
        container = SimpleNamespace(
            streams=SimpleNamespace(video=[object()]), close=close,
            decode=lambda _: (SimpleNamespace(to_ndarray=lambda format, img=img: img)
                              for img in read_images()))
        module = SimpleNamespace(open=lambda _: container)
    else:
        iterator = iter(read_images())
        def read():
            img = next(iterator, None)
            return (False, None) if img is None else (True, img[..., ::-1])
        cap = SimpleNamespace(read=read, release=close, isOpened=lambda: True)
        module = SimpleNamespace(VideoCapture=lambda _: cap)
    monkeypatch.setattr(io, "_video_backend", lambda: (request.param, module), raising=False)
    return state


def test_selected_frames_are_sorted_unique_rgb(decoder):
    result = io.load_video("synthetic.mp4", [2, 0, 2])
    assert result.shape == (2, 4, 6, 3)
    assert result[:, 0, 0, 0].tolist() == [0, 2]
    assert decoder.closed


def test_missing_frames_are_explicit_not_black_or_dropped(decoder):
    with pytest.raises(IndexError, match="5"):
        io.load_video("synthetic.mp4", [0, 5])
    assert decoder.closed


def test_decoder_is_closed_on_error(decoder):
    decoder.fail = True
    with pytest.raises(RuntimeError, match="decode failed"):
        io.load_video("synthetic.mp4")
    assert decoder.closed


def test_empty_selection_does_not_open_a_decoder(monkeypatch):
    monkeypatch.setattr(io, "_video_backend", lambda: pytest.fail("decoder opened"), raising=False)
    assert io.load_video("unused.mp4", []).shape == (0, 0, 0, 3)


@pytest.mark.parametrize("indices", [[-1], [1.5], [True]])
def test_invalid_indices_rejected_before_open(monkeypatch, indices):
    monkeypatch.setattr(io, "_video_backend", lambda: pytest.fail("decoder opened"), raising=False)
    with pytest.raises(ValueError, match="indices"):
        io.load_video("unused.mp4", indices)


def test_toolbox_image_helpers_do_not_import_arrow_or_decoders():
    subprocess.run([sys.executable, "-c",
                    "import sys; from twm.react_toolbox import difference; "
                    "assert not {'pyarrow', 'av', 'cv2'} & sys.modules.keys()"], check=True)


@pytest.mark.parametrize("backend", ["av", "cv2"])
def test_real_decoders_agree_on_rgb_and_selection(tmp_path, monkeypatch, backend):
    from twm.visualization.export import write_video
    module = pytest.importorskip(backend)
    frames = [np.full((16, 24, 3), [20 + i * 30, 80, 180], np.uint8) for i in range(3)]
    path = tmp_path / "clip.mp4"
    write_video(path, lambda: iter(frames), fps=30)
    monkeypatch.setattr(io, "_video_backend", lambda: (backend, module))
    got = io.load_video(path, [2, 0, 2])
    assert got.shape == (2, 16, 24, 3)
    np.testing.assert_allclose(got[:, 0, 0], np.array([frames[0][0, 0], frames[2][0, 0]])[:, ::-1], atol=3)
