"""Encoder failures preserve the original error and always reap the child."""
import io
from types import SimpleNamespace

import numpy as np
import pytest

from twm.react_preprocess import encode


@pytest.fixture
def process(monkeypatch):
    state = SimpleNamespace(stdin=io.BytesIO(), waited=False, returncode=0)

    def wait():
        state.waited = True
        return state.returncode

    state.wait = wait
    monkeypatch.setattr(encode.subprocess, "Popen", lambda *a, **kw: state)
    return state


def test_broken_pipe_on_close_still_reaps_and_reports_encoder_failure(tmp_path, process):
    class BrokenClose(io.BytesIO):
        def close(self):
            super().close()
            raise BrokenPipeError("encoder exited")

    process.stdin = BrokenClose()
    process.returncode = 7
    with pytest.raises(RuntimeError, match=r"ffmpeg failed \(7\).*out.mp4"):
        with encode.VideoWriter(tmp_path / "out.mp4", width=3, height=2):
            pass
    assert process.waited


def test_cleanup_does_not_replace_error_from_body(tmp_path, process):
    class BrokenClose(io.BytesIO):
        def close(self):
            super().close()
            raise BrokenPipeError("encoder exited")

    process.stdin = BrokenClose()
    with pytest.raises(ValueError, match="bad frame from caller"):
        with encode.VideoWriter(tmp_path / "out.mp4", width=3, height=2):
            raise ValueError("bad frame from caller")
    assert process.waited


def test_unsupported_pixel_format_is_rejected_before_starting_encoder(tmp_path, process):
    with pytest.raises(ValueError, match="unsupported input pixel format.*gray8"):
        encode.VideoWriter(tmp_path / "out.mp4", pix_fmt="gray8")


@pytest.mark.parametrize("shape,dtype,pix_fmt,codec", [
    ((2, 3, 3), np.uint8, "bgr24", "libx264"),
    ((1, 3, 2, 3), np.uint8, "bgr24", "libx264"),
    ((1, 2, 3, 4), np.uint8, "bgr24", "libx264"),
    ((1, 2, 3, 3), np.float32, "bgr24", "libx264"),
    ((1, 2, 3, 1), np.uint16, "gray16le", "ffv1"),
    ((1, 2, 3), np.uint8, "gray16le", "ffv1"),
])
def test_invalid_frame_layout_is_rejected_before_bytes_are_sent(
        tmp_path, process, shape, dtype, pix_fmt, codec):
    with encode.VideoWriter(tmp_path / "out", width=3, height=2,
                            pix_fmt=pix_fmt, codec=codec) as writer:
        with pytest.raises(ValueError, match="expected.*got"):
            writer.write(np.zeros(shape, dtype=dtype))
        assert process.stdin.getvalue() == b""
    assert process.waited


@pytest.mark.parametrize("pix_fmt,codec,dtype,shape", [
    ("bgr24", "libx264", np.uint8, (2, 2, 3, 3)),
    ("gray16le", "ffv1", np.uint16, (2, 2, 3)),
])
def test_valid_noncontiguous_blocks_preserve_bytes(tmp_path, process, pix_fmt, codec,
                                                  dtype, shape):
    block = np.arange(np.prod(shape), dtype=dtype).reshape(shape)[:, :, ::-1]
    with encode.VideoWriter(tmp_path / "out", width=3, height=2,
                            pix_fmt=pix_fmt, codec=codec) as writer:
        writer.write(block)
        assert process.stdin.getvalue() == block.tobytes()
    assert process.waited
