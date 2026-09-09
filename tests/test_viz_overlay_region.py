"""draw_projection_overlay touches only the camera row it draws on."""
import numpy as np
import hdf5plugin  # noqa: F401

from twm.viz import RS_THUMB_H, RS_THUMB_W, draw_projection_overlay, project_gel_pose, _scale_to_thumb, DISPLAY_POSITION


def _cam(index):
    return {"index": index,
            "T_mocap_to_cam": np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 800.0], [0, 0, 0, 1]]),
            "intrinsics": {"fx": 600.0, "fy": 600.0, "ppx": 320.0, "ppy": 240.0}}


def test_overlay_leaves_the_rows_below_the_cameras_untouched_and_draws_the_dot():
    panel = np.full((720, 1280, 3), 7, np.uint8)
    poses = {"sensor_left": (0.0, [0.05, 0.02, 0.0, 0, 0, 0, 1]), "sensor_right": None}
    gel = np.zeros(3)
    cam = _cam(2)
    before = panel.copy()
    draw_projection_overlay(panel, poses, [cam], gel, gel)
    assert np.array_equal(panel[RS_THUMB_H:], before[RS_THUMB_H:])
    assert np.array_equal(panel[:, 3 * RS_THUMB_W:], before[:, 3 * RS_THUMB_W:])
    (u, v), _ = project_gel_pose(poses["sensor_left"][1], gel, cam["T_mocap_to_cam"], cam["intrinsics"])
    x, y = _scale_to_thumb(u, v)
    x += DISPLAY_POSITION[2] * RS_THUMB_W
    assert not np.array_equal(panel[y - 2:y + 3, x - 2:x + 3], before[y - 2:y + 3, x - 2:x + 3])


def test_status_text_lives_in_a_strip_below_the_images():
    from twm.viz import STATUS_STRIP_H, build_preview_panel
    color = [np.full((480, 640, 3), 90, np.uint8) for _ in range(3)]
    gs = [np.full((480, 640, 3), 90, np.uint8) for _ in range(2)]
    panel = build_preview_panel(color, gs, gs, {}, False, 0, 0.0,
                                arducam_frames=gs, arducam_labels=["a", "b"])
    assert panel.shape == (3 * RS_THUMB_H + STATUS_STRIP_H, 1280, 3)
    wrist = panel[2 * RS_THUMB_H:3 * RS_THUMB_H, :RS_THUMB_W]
    assert (wrist[40:] == 90).all()                       # no status text on the wrist image (label at top only)
    strip = panel[3 * RS_THUMB_H:]
    assert strip.max() > 0                                # the status bar is drawn in the strip
