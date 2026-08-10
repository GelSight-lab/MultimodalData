"""Unit tests for the pure-function core (run: python -m force_recovery.test_units)."""
import numpy as np

from .dexforce import force_informed_targets, quat_to_matrix, roundtrip_force
from .evaluate import median3_fresh


def test_quat_identity():
    R = quat_to_matrix(np.array([[0, 0, 0, 1.0]]))
    assert np.allclose(R[0], np.eye(3)), R


def test_quat_90z():
    # 90 deg about z: x-axis maps to y-axis
    q = np.array([[0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)]])
    v = quat_to_matrix(q)[0] @ np.array([1.0, 0, 0])
    assert np.allclose(v, [0, 1, 0], atol=1e-12), v


def test_targets_free_space_identity():
    pose = np.zeros((5, 7)); pose[:, 6] = 1.0
    act = force_informed_targets(pose, np.zeros(5), np.array([0, 0, 1.0]))
    assert np.array_equal(act.target_pos, pose[:, :3])


def test_targets_penetration_direction():
    pose = np.zeros((1, 7)); pose[0, 6] = 1.0          # identity orientation
    act = force_informed_targets(pose, np.array([0.3]), np.array([0, 0, 1.0]),
                                 stiffness=300.0)
    assert np.allclose(act.target_pos[0], [0, 0, 0.001])   # 0.3N/300 = 1 mm


def test_roundtrip():
    rng = np.random.default_rng(0)
    pose = np.zeros((50, 7))
    pose[:, :3] = rng.normal(size=(50, 3))
    q = rng.normal(size=(50, 4)); pose[:, 3:] = q / np.linalg.norm(q, axis=1)[:, None]
    force = np.abs(rng.normal(size=50))
    act = force_informed_targets(pose, force, np.array([0.2, -0.9, -0.3]))
    assert np.allclose(roundtrip_force(act, pose[:, :3]), force, atol=1e-9)


def test_median3_kills_single_spike():
    force = np.zeros(30); is_new = np.ones(30, bool)
    force[10] = 5.0
    out = median3_fresh(force, is_new)
    assert out[10] == 0.0, out[10]


def test_median3_keeps_plateau():
    force = np.zeros(30); is_new = np.ones(30, bool)
    force[10:14] = 5.0
    out = median3_fresh(force, is_new)
    assert out[11] == 5.0 and out[12] == 5.0


def test_median3_respects_duplicates():
    # duplicated rows repeat the previous fresh value; a spike lasting one
    # FRESH frame but three ROWS must still be removed
    force = np.array([0, 0, 0, 5, 5, 5, 0, 0, 0], float)
    is_new = np.array([1, 0, 0, 1, 0, 0, 1, 0, 0], bool)
    out = median3_fresh(force, is_new)
    assert np.all(out == 0.0), out


def test_marker_mask_is_none_on_markerless_gel():
    """The depth-path marker step must be a strict no-op without dots.

    `marker_mask` returns None (not an empty mask) so `stages_depth` calls
    `stages()` on the untouched arrays — GlowTact / cnc / React output stays
    bit-exact. A smooth synthetic gel stands in for a markerless reference so
    the test needs no dataset on disk.
    """
    from .marker_removal import marker_mask

    yy, xx = np.mgrid[0:120, 0:160]
    ref = np.stack([120 + 30 * np.sin(xx / 90.0), 110 + 25 * (yy / 120.0),
                    np.full_like(xx, 130.0)], -1).astype(np.float32)
    assert marker_mask(ref) is None


def test_stages_depth_matches_marker_study_img_telea():
    """The adopted wrapper must reproduce the studied variant exactly.

    Skipped when the FEATS parquet is not mounted (the assertion is about
    real dots, so there is no synthetic substitute).
    """
    from .debug_gallery import FEATS_PQ

    if not FEATS_PQ.exists():
        print("  (skipped: FEATS parquet not mounted)")
        return
    from .debug_gallery import load_feats
    from .marker_removal import marker_mask, stages_depth
    from .marker_study import run_strategy

    frames, get = load_feats()
    img, ref = get(frames[3])
    mask = marker_mask(ref)
    assert mask is not None and mask.mean() > 0.15
    from .lut_calibration import GEL_THICKNESS_MM
    a = stages_depth(img, ref)["depth"]
    b = run_strategy(img, ref, mask, "img_telea")["depth"]
    # Equal UP TO THE GEL CEILING. `stages()` later gained a physical clip at
    # the 4.25 mm elastomer thickness; `marker_study.stages_m` is frozen
    # evidence from before that and does not clip. On FEATS frame 3 that is
    # the entire difference — 506 pixels, all of them the ones the study
    # reconstructs at 4.25-4.66 mm, i.e. deeper than the gel can be. The
    # assertion is kept exact everywhere else, because the point of it is that
    # these two code paths have not otherwise forked.
    assert np.array_equal(a, np.minimum(b, GEL_THICKNESS_MM)), \
        float(np.abs(a - np.minimum(b, GEL_THICKNESS_MM)).max())


def test_force_dot_area_is_linear_in_force():
    """The dot's AREA must track force; a radius-proportional dot would
    exaggerate large forces quadratically to the eye."""
    import numpy as np
    from twm.force_overlay import radius_px, F_FULL_N, R_MIN_PX, R_MAX_PX
    span = (R_MAX_PX - R_MIN_PX) ** 2
    for f in np.linspace(0.05, F_FULL_N, 25):
        area = (radius_px(f) - R_MIN_PX) ** 2 / span
        assert abs(area - f / F_FULL_N) < 1e-9, f
    assert radius_px(10 * F_FULL_N) <= R_MAX_PX      # saturates, stays in tile


def test_force_row_mapping_uses_parquet_trim():
    """Off-by-15 here would put the dot half a second from its own frame."""
    from twm.force_overlay import row_for_h5_frame
    assert row_for_h5_frame(16, 1, 100) == 0
    assert row_for_h5_frame(15, 1, 100) is None      # before the first row
    assert row_for_h5_frame(115, 1, 100) == 99
    assert row_for_h5_frame(116, 1, 100) is None     # past the last row


def test_tactile_lag_matches_the_published_dataset_contract():
    """15 frames is not ours to choose: yxma/React tasks.json documents it as
    frames_estimate 15 @ 30 fps, applies_to recordings <= 2026-06-18."""
    from twm.tactile_align import LEGACY_SHIFT, RIG_FIXED_DATE
    assert LEGACY_SHIFT == 15
    assert RIG_FIXED_DATE == "2026-06-27"


def test_tactile_lag_is_detected_per_file_not_assumed():
    """A timestamped recording must get 0 — applying the constant on top of an
    already-aligned stream is the mirror-image bug of forgetting it."""
    from twm.tactile_align import gel_lag_frames, LEGACY_SHIFT

    class _Node(dict):
        def keys(self):
            return super().keys()

    class _Fake:
        def __init__(self, ts):
            self._n = {"gelsight/left": _Node({"frames": 1, **({"timestamps": 1} if ts else {})}),
                       "gelsight/right": None}

        def get(self, k):
            return self._n.get(k)

    assert gel_lag_frames(_Fake(ts=True)) == 0
    assert gel_lag_frames(_Fake(ts=False)) == LEGACY_SHIFT


def test_force_disc_annotates_rather_than_covers():
    """The disc lives on a camera view now, so it is bounded by that view.

    The first version was sized for a 240 px tactile tile (R_MAX 74) and, moved
    as-is onto the 320x240 camera thumbnail, would have spanned 46% of its
    width — hiding the workpiece the video exists to show.
    """
    from twm.force_overlay import R_MAX_PX
    from twm.viz import RS_THUMB_W, RS_THUMB_H
    assert 2 * R_MAX_PX <= 0.20 * RS_THUMB_W, "disc too wide for a cam view"
    assert 2 * R_MAX_PX <= 0.25 * RS_THUMB_H, "disc too tall for a cam view"


def test_force_halo_stays_inside_the_view_it_annotates():
    """A sensor out of a camera's frustum must draw NOTHING in that view.

    `project_gel_pose` returns coordinates outside the 640x480 image when the
    sensor is not visible; scaled and offset by the thumbnail slot, those land
    on a *neighbouring* tile — or below the camera row entirely, on the
    tactile tiles. The first version clipped only to the canvas and did
    exactly that: a confident orange disc on a panel region where no sensor
    was. Only the render verifier caught it.
    """
    import numpy as np
    from twm.force_overlay import draw_force_halo

    panel = np.full((480, 1280, 3), 40, np.uint8)
    view = (320, 0, 640, 240)                      # the middle camera view
    for outside in [(700, 120), (300, 120), (480, 300), (480, -10)]:
        draw_force_halo(panel, outside, 8.0, bounds=view)
        assert (panel == 40).all(), f"drew ink for out-of-view {outside}"

    draw_force_halo(panel, (480, 120), 8.0, bounds=view)
    ink = (panel != 40).any(axis=2)
    assert ink.sum() > 0, "in-view centre drew nothing"
    assert ink[:, :320].sum() == 0 and ink[:, 640:].sum() == 0, "spilled sideways"
    assert ink[240:].sum() == 0, "spilled into the tactile row"


def test_force_halo_tracks_the_supersample_scale():
    """`viz` blends the disc on a 2x buffer; radius must scale with it.

    Ignoring `scale` draws a half-size dot that survives every other check —
    it looks deliberate, just wrong — so assert the 2x ink is ~4x the 1x ink.
    """
    import numpy as np
    from twm.force_overlay import draw_force_halo

    def ink(scale):
        c = np.full((240 * scale, 320 * scale, 3), 40, np.uint8)
        draw_force_halo(c, (160, 120), 6.0, scale=scale, label=False)
        return int((c != 40).any(axis=2).sum())

    one, two = ink(1), ink(2)
    assert one > 0 and abs(two / one - 4.0) < 0.15, (one, two)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
    print("all unit tests passed")
