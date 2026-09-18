"""One traversal of the recording, not one per stream.

The recorder writes one chunk per frame per stream, tick by tick, so a single
stream's chunks are strided through the whole file — measured on a rope
recording, 0.62 MB chunks with the next chunk of the same stream 3.87 MB
further on. Encoding one stream at a time seeks across that stride for every
chunk. Measured, the two orders move the SAME bytes (1.09x vs 1.14x of what
is used) and differ only in time: 190 s against 351 s for 600 frames of five
streams. The cost is seeks, not wasted reads.

The fix is invisible in the output: both paths write the same frames to the
same files. So the test that matters is about the ORDER the file is read in,
and it is the only evidence that the change does anything at all.
"""
from __future__ import annotations

import numpy as np
import pytest

from twm.react_preprocess import pipeline


class FakeDataset:
    """A dataset that records the slices taken from it."""

    def __init__(self, name: str, array: np.ndarray, log: list):
        self.name, self._a, self._log = name, array, log
        self.shape = array.shape

    def __getitem__(self, sl):
        self._log.append((self.name, sl.start, sl.stop))
        return self._a[sl]


class FakeAttrs(dict):
    pass


class FakeMeta:
    def __init__(self, attrs):
        self.attrs = FakeAttrs(attrs)


class FakeWriter:
    def __init__(self, path, sink, *, width, height):
        self.path, self._sink = path, sink
        self._shape = (height, width, 3)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def write(self, block):
        assert block.shape[1:] == self._shape
        self._sink.setdefault(self.path.name, []).append(np.asarray(block).copy())


class FakeSource:
    def __init__(self, T=10, trim=3, task="pushT"):
        self.T, self.trim, self.task = T, trim, task


def _fake_file(log, n=40, wrist=True):
    f = {}
    for cam in (0, 1, 2):
        f[f"realsense/cam{cam}/color"] = FakeDataset(
            f"cam{cam}", np.arange(n * 2 * 2 * 3, dtype=np.uint8).reshape(n, 2, 2, 3) + cam,
            log)
    if wrist:
        f["arducam"] = object()
        f["metadata"] = FakeMeta({"arducam_config": '[{"serial": "TWML0001"}]'})
        for slot in ("cam0", "cam1"):
            f[f"arducam/{slot}/frames"] = FakeDataset(
                f"wrist_{slot}",
                np.arange(n * 2 * 2 * 3, dtype=np.uint8).reshape(n, 2, 2, 3), log)
    else:
        f["metadata"] = FakeMeta({})
    return f


@pytest.fixture
def patched(monkeypatch):
    """Isolate the traversal from ffmpeg and from the tone curve."""
    sink = {}
    monkeypatch.setattr(pipeline, "rgb_writer", lambda p, **kw: FakeWriter(p, sink, **kw))
    monkeypatch.setattr(pipeline, "gamma_for_camera", lambda kind, task: 2.0)
    monkeypatch.setattr(pipeline, "decode_arducam", lambda fr: fr)
    monkeypatch.setattr(pipeline, "apply_tone_curve", lambda a, g: a + 100)
    monkeypatch.setattr(pipeline, "CHUNK", 4)
    return sink


def test_the_file_is_traversed_once_not_once_per_stream(patched):
    """The point of the change, and the only place it is visible.

    Per-stream encoding reads cam0 end to end, then cam1 end to end, ... —
    five traversals whose reads are grouped by stream. One pass reads every
    stream's block for frames [a,b) before moving to [b,c), so the run of
    stream names repeats rather than being sorted into five runs.
    """
    log = []
    pipeline._encode_rgb_single_pass(_fake_file(log), FakeSource(T=10, trim=3),
                                     __import__("pathlib").Path("/tmp/x"))

    # Group the reads by the block of frames they cover.
    blocks = {}
    for name, start, stop in log:
        blocks.setdefault((start, stop), []).append(name)

    # Each frame block is read for all five streams together...
    assert list(blocks) == [(3, 7), (7, 11), (11, 13)]
    for names in blocks.values():
        assert sorted(names) == sorted(
            ["cam0", "cam1", "cam2", "wrist_cam0", "wrist_cam1"])

    # ...and no stream is read to completion before another starts, which is
    # exactly what the per-stream path does.
    first_five = [name for name, _, _ in log[:5]]
    assert len(set(first_five)) == 5


def test_the_per_stream_path_really_does_group_by_stream(patched):
    """The baseline the change is measured against — if this ever stops being
    true, the single-pass path has no reason to exist."""
    log = []
    f = _fake_file(log)
    pipeline._encode_cameras(f, FakeSource(T=10, trim=3),
                             __import__("pathlib").Path("/tmp/x"))

    names = [n for n, _, _ in log]
    # cam0 is read to completion before cam1 begins.
    assert names == ["cam0"] * 3 + ["cam1"] * 3 + ["cam2"] * 3


def test_single_pass_writes_the_same_bytes_as_the_two_pass_path(monkeypatch):
    """Same frames, same files, same order within each file. The change is an
    I/O reordering and must not be anything else."""
    def run(single: bool):
        sink = {}
        monkeypatch.setattr(pipeline, "rgb_writer", lambda p, **kw: FakeWriter(p, sink, **kw))
        monkeypatch.setattr(pipeline, "gamma_for_camera", lambda kind, task: 2.0)
        monkeypatch.setattr(pipeline, "decode_arducam", lambda fr: fr)
        monkeypatch.setattr(pipeline, "apply_tone_curve", lambda a, g: a + 100)
        monkeypatch.setattr(pipeline, "CHUNK", 4)
        from pathlib import Path
        f, src = _fake_file([]), FakeSource(T=10, trim=3)
        if single:
            gammas = pipeline._encode_rgb_single_pass(f, src, Path("/tmp/x"))
        else:
            pipeline._encode_cameras(f, src, Path("/tmp/x"))
            gammas = pipeline._encode_wrist(f, src, Path("/tmp/x"))
        return {k: np.concatenate(v) for k, v in sink.items()}, gammas

    one, g1 = run(True)
    two, g2 = run(False)

    assert sorted(one) == sorted(two) == [
        "view_left.mp4", "view_middle.mp4", "view_right.mp4",
        "wrist_left.mp4", "wrist_right.mp4"]
    for name in one:
        np.testing.assert_array_equal(one[name], two[name], err_msg=name)
    assert g1 == g2 == {"cam0": 2.0, "cam1": 2.0}


def test_every_published_frame_starts_at_the_trim(patched):
    """The trim is what makes index i mean one instant across every stream."""
    log = []
    pipeline._encode_rgb_single_pass(_fake_file(log), FakeSource(T=10, trim=3),
                                     __import__("pathlib").Path("/tmp/x"))
    assert min(start for _, start, _ in log) == 3
    assert max(stop for _, _, stop in log) == 13


def test_a_recording_with_no_wrist_camera_encodes_the_three_views(patched):
    log = []
    gammas = pipeline._encode_rgb_single_pass(
        _fake_file(log, wrist=False), FakeSource(T=10, trim=3),
        __import__("pathlib").Path("/tmp/x"))

    assert gammas == {}
    assert sorted(patched) == ["view_left.mp4", "view_middle.mp4",
                               "view_right.mp4"]


def test_both_paths_stay_reachable():
    """This asserted `off by default` while the single-pass path waited for
    evidence. It has it now — 352 s against 253 s on a real 8 GB recording,
    with 7 of 7 videos byte-identical and every parquet column equal — so the
    default moved.

    What still matters is that NEITHER path disappears: the per-stream one is
    the reference the equivalence was measured against, and the only way to
    re-measure it on a future recording.
    """
    import inspect
    from twm.react_preprocess import pipeline
    assert inspect.signature(pipeline.build_episode).parameters["single_pass"].default is True
    for name in ("_encode_rgb_single_pass", "_encode_cameras", "_encode_wrist"):
        assert hasattr(pipeline, name), f"{name} was removed"
