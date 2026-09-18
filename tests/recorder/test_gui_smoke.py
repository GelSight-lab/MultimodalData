"""The GUI loop runs end to end with a stubbed window: builds a panel from a
real synthetic tick, draws the overlay with the newest pose, and quits on q."""
import cv2
import numpy as np

from twm.recorder import app as app_mod
from twm.recorder.capture import CaptureSnapshot
from twm.recorder.config import RecorderConfig
from twm.recorder.frames import synthetic_tick
from twm.recorder.writer import WriterStats


class FakeCapture:
    def __init__(self, snap):
        self.snap = snap

    def latest(self):
        return self.snap


class FakeRecorder:
    recording = False

    def __init__(self):
        self.ended = []

    def poll(self, snap):
        pass

    def end_episode(self, reason):
        self.ended.append(reason)


class FakeRig:
    arducam_encoding = "bgr8"
    # The real SensorRig always carries this; the preview reads it to pick the
    # tone curve it must draw the wrist frames through.
    arducam_config = ()
    calls = 0

    def latest_poses(self):
        FakeRig.calls += 1
        return {"sensor_left": (0.0, [0.075, 0.062, 0.419, -0.12, 0.52, 0.23, 0.81]),
                "sensor_right": (0.0, [0.637, 0.064, 0.389, 0.02, 0.66, -0.08, 0.75]),
                "motherboard": None}

    def arducam_labels(self):
        return ["cam0 left", "cam1 right"]


def test_gui_loop_renders_overlay_with_fresh_poses_and_quits(monkeypatch):
    shown = []
    keys = iter([255, 255, ord("p"), 255, ord("q")])
    monkeypatch.setattr(cv2, "imshow", lambda name, panel: shown.append(panel.copy()))
    monkeypatch.setattr(cv2, "waitKey", lambda ms: next(keys))
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)

    tick = synthetic_tick(100.0, seed=1, n_realsense=3, n_arducam=2)
    snap = CaptureSnapshot(tick=tick, gs_ref=tuple(g.copy() for g in tick.gelsight),
                           ot_poses=FakeRig().latest_poses(), recording=False, frame_count=0,
                           elapsed=0.0, fps_meas=30.0,
                           writer=WriterStats(0, 0, 100, 0.0, 0, 0.0, 0.0, 0, 0.0, 200.0, None, None),
                           stop_request=None)
    config = RecorderConfig(task="x")
    projection = app_mod.load_projection(config)
    assert projection and len(projection["cams"]) == 3

    rc = app_mod._gui_loop(config, FakeRecorder(), FakeCapture(snap), FakeRig(), projection)

    assert rc == 0
    assert len(shown) == 5
    assert shown[0].shape == (720 + 48, 1280, 3)     # 3 rows + text strip: wrist cameras present
    assert FakeRig.calls >= 3                          # a fresh pose per rendered frame (overlay on)
    assert not np.array_equal(shown[0][:240, :960], shown[3][:240, :960])  # p toggled the overlay off
