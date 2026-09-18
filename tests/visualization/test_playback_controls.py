"""Exercise viewer controls without replacing imported modules or opening a GUI."""
from types import SimpleNamespace

import numpy as np

from twm import release_episode, visualize


def test_reset_reference_uses_aligned_source_frames(monkeypatch):
    colors = [np.zeros((480, 640, 3), np.uint8) for _ in range(3)]
    tactile = [np.full((480, 640, 3), value, np.uint8) for value in (30, 70)]
    source = SimpleNamespace(
        n_frames=2, fps=30, task="test", date="date", episode="episode",
        gs_ref=[np.zeros_like(a) for a in tactile], labels=None,
        frames=lambda i: (colors, tactile, None), timestamp=lambda i: i / 30,
        poses_at=lambda i: {}, forces_at=lambda i: {}, close=lambda: None,
    )
    monkeypatch.setattr(release_episode, "looks_like_release", lambda p: True)
    monkeypatch.setattr(release_episode, "resolve", lambda p: source)
    monkeypatch.setattr(visualize, "_ReleasePlayback", lambda ep, args: ep)
    for name in ("namedWindow", "createTrackbar", "imshow", "setTrackbarPos", "destroyAllWindows"):
        monkeypatch.setattr(visualize.cv2, name, lambda *a, **k: None)
    monkeypatch.setattr(visualize.cv2, "getTrackbarPos", lambda *a: 0)
    keys = iter((ord("r"), ord("q")))
    monkeypatch.setattr(visualize.cv2, "waitKey", lambda *a: next(keys))
    refs = []
    original = visualize.make_preview

    def capture(*args, **kwargs):
        refs.append([a.copy() for a in args[2]])
        return original(*args, **kwargs)

    monkeypatch.setattr(visualize, "make_preview", capture)
    visualize.process_episode("published", None, SimpleNamespace(fps=None, check=False), [], None, None)
    assert len(refs) == 2
    for old, actual, expected in zip(refs[0], refs[1], tactile):
        assert not old.any()
        np.testing.assert_array_equal(actual, expected)
