"""The wrist cameras record a dark image, and it cannot be fixed with a gain.

Measured on the 2026-09-10/11 pushT sessions (20 frames per episode, grey =
channel mean):

    stream        p50        p95        p99.5
    wrist USB     60-76      123-128    176-213
    RealSense     134-138    153        163-166

The wrist highlights are BRIGHTER than the RealSense's, so there is no unused
headroom to scale into — a gain chosen to lift the bulk would crush that tail,
and a gain chosen to preserve the tail (0.77-0.95x) would do nothing. What is
wrong is the distribution: a thin specular tail with everything else pressed
into the shadows.

A power curve fixes exactly that. It lifts the midtones, leaves 0 at 0 and 255
at 255, and clips nothing. The exponent is chosen per episode-stream so the
wrist median lands on the median of that episode's own RealSense view, which
makes the rig its own reference instead of a hardcoded target.

This is a declared photometric change to published pixels: the exponent is
written into the episode metadata, and an exponent of 1.0 is a no-op for
sessions that do not need it (the 2026-09-09 Arducam ones measure much closer
to the reference).
"""
import numpy as np
import pytest

from twm.wrist_tone import (
    GAMMA_LIMITS,
    apply_tone_curve,
    tone_curve_lut,
    tone_gamma_for,
)


def grey(value, shape=(16, 16)):
    return np.full((*shape, 3), value, np.uint8)


def median_grey(img):
    return float(np.median(np.asarray(img, np.float32).mean(axis=-1)))


def test_a_dark_stream_is_lifted_onto_the_reference_median():
    dark, reference = grey(60), grey(134)
    g = tone_gamma_for(dark, reference)
    assert g > 1.0
    assert median_grey(apply_tone_curve(dark, g)) == pytest.approx(134, abs=2)


def test_a_stream_that_already_matches_is_left_alone():
    g = tone_gamma_for(grey(134), grey(134))
    assert g == pytest.approx(1.0, abs=0.01)
    img = np.random.default_rng(0).integers(0, 256, (8, 8, 3), dtype=np.uint8)
    np.testing.assert_array_equal(apply_tone_curve(img, 1.0), img)


def test_black_stays_black_and_white_stays_white():
    """The whole point of a curve over a gain: the specular tail survives."""
    lut = tone_curve_lut(2.25)
    assert lut[0] == 0 and lut[255] == 255
    assert (np.diff(lut.astype(int)) >= 0).all(), "curve must be monotonic"


def test_the_specular_tail_is_not_crushed_into_saturation():
    """A gain would collapse everything above 255/g into one value. The curve
    merges at most the top pair (254 rounds to 255 at a strong exponent), so
    the bright tail the wrist cameras do have survives as structure."""
    lut = tone_curve_lut(GAMMA_LIMITS[1])
    assert int((lut == 255).sum()) <= 2, "more than the top pair saturates"
    bright = np.array([[[200, 220, 254]]], np.uint8)
    out = apply_tone_curve(bright, 2.25)
    assert (out >= bright).all(), "a lift that darkens pixels"
    assert len(np.unique(out)) == 3, "distinct highlights merged"


def test_the_exponent_is_capped_at_both_ends():
    """An episode whose wrist view is nearly black would otherwise ask for an
    exponent that turns sensor noise into visible structure."""
    lo, hi = GAMMA_LIMITS
    assert tone_gamma_for(grey(2), grey(200)) <= hi
    assert tone_gamma_for(grey(250), grey(30)) >= lo


def test_a_reference_that_is_itself_dark_does_not_drag_the_wrist_down():
    """The correction only ever brightens: if the RealSense view is darker than
    the wrist (a dim scene), the wrist is left as recorded rather than dimmed
    to match, which would destroy information instead of redistributing it."""
    assert tone_gamma_for(grey(120), grey(60)) == 1.0


def test_the_measured_wrist_and_reference_medians_ask_for_a_plausible_curve():
    """The real numbers from the session, end to end."""
    g = tone_gamma_for(grey(60), grey(134))
    assert 1.8 <= g <= 2.8
    assert median_grey(apply_tone_curve(grey(76), g)) > 76


def test_the_curve_is_applied_per_frame_not_per_block_statistic():
    """A block of frames must get the SAME curve, or brightness flickers with
    whatever happens to be in view."""
    g = 2.0
    block = np.stack([grey(40), grey(200)])
    out = apply_tone_curve(block, g)
    assert out.shape == block.shape
    np.testing.assert_array_equal(out[0], apply_tone_curve(grey(40), g))
    np.testing.assert_array_equal(out[1], apply_tone_curve(grey(200), g))


# ── the exponent is a property of the camera, not of the episode ────────────

def test_each_wrist_generation_gets_the_exponent_its_measurement_asked_for():
    from twm.wrist_tone import gamma_for_camera
    # Derived on the same surface (the wood table both views see) through its
    # upper quantiles: usb 1.45-1.57, arducam 1.14-1.27.
    assert 1.45 <= gamma_for_camera("usb") <= 1.57
    assert 1.14 <= gamma_for_camera("arducam") <= 1.27
    assert gamma_for_camera("usb") > gamma_for_camera("arducam"), \
        "the USB pair records the darker image of the two"


def test_a_session_recorded_before_the_wrist_cameras_publishes_unchanged():
    from twm.wrist_tone import gamma_for_camera
    assert gamma_for_camera(None) == 1.0
    assert gamma_for_camera("something-new") == 1.0


def _h5_with_wrist_config(path, cams):
    import h5py
    import json as _json
    with h5py.File(path, "w") as f:
        f.create_group("metadata").attrs["arducam_config"] = _json.dumps(cams)
    return path


def test_the_two_wrist_generations_are_told_apart_by_serial(tmp_path):
    """Both report a serial, so presence cannot discriminate — and the USB pair
    reports the SAME serial on both cameras, which is why identity is by port."""
    import h5py

    from twm.react_preprocess.pipeline import _wrist_camera_kind

    ard = _h5_with_wrist_config(tmp_path / "a.h5", [
        {"slot": "cam0", "serial": "TWML0001", "id_path": "x"},
        {"slot": "cam1", "serial": "TWMR0001", "id_path": "y"}])
    usb = _h5_with_wrist_config(tmp_path / "u.h5", [
        {"slot": "cam0", "serial": "200901010001", "id_path": "x"},
        {"slot": "cam1", "serial": "200901010001", "id_path": "y"}])
    none = tmp_path / "n.h5"
    with h5py.File(none, "w") as f:
        f.create_group("metadata")

    for path, expected in ((ard, "arducam"), (usb, "usb"), (none, None)):
        with h5py.File(path, "r") as f:
            assert _wrist_camera_kind(f) == expected


# ── the preview and the publish path must agree ─────────────────────────────

def test_the_recorder_preview_uses_the_published_exponent(monkeypatch):
    """An operator aims the wrist cameras against the preview. If the preview
    shows the raw frame while the dataset ships a lifted one — or the other way
    round — the framing is done against a picture nobody will ever see."""
    from twm.recorder.app import wrist_preview_gamma
    from twm.wrist_tone import gamma_for_camera

    class Cam:
        def __init__(self, serial):
            self.serial = serial

    usb = [Cam("200901010001"), Cam("200901010001")]
    ard = [Cam("TWML0001"), Cam("TWMR0001")]
    assert wrist_preview_gamma(usb, "motherboard") == gamma_for_camera("usb", "motherboard")
    assert wrist_preview_gamma(usb, "pushT") == gamma_for_camera("usb", "pushT")
    assert wrist_preview_gamma(ard, "motherboard") == gamma_for_camera("arducam", "motherboard")
    assert wrist_preview_gamma([], "pushT") == 1.0


# ── the curve must not repaint the scene ────────────────────────────────────

def _hsv(img):
    import cv2
    h = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_BGR2HSV)
    return h[..., 0].astype(float) * 2.0, h[..., 1].astype(float)


def _colours():
    """Colours spanning the wrist scenes: wood, PCB green, skin, a red LED."""
    return np.array([[[40, 90, 150], [30, 60, 25], [120, 150, 200],
                      [20, 20, 180], [80, 80, 80], [5, 4, 9]]], np.uint8)


def test_the_curve_preserves_hue_and_saturation():
    """A power curve applied per channel changes the RATIOS between channels,
    and the ratios are the hue and the saturation. Measured on a real wrist
    frame at gamma 2.4, per-channel shifted hue by 4 deg (p95 18) and cut
    saturation by 20 points (p5 78) — the wood went pale and skin went grey."""
    src = _colours()
    out = apply_tone_curve(src, 2.4)
    h0, s0 = _hsv(src)
    h1, s1 = _hsv(out)
    coloured = s0 > 30
    assert np.abs(((h1 - h0 + 180) % 360) - 180)[coloured].max() <= 3.0
    assert np.abs(s1 - s0)[coloured].max() <= 4.0


def test_the_channel_ratios_survive_the_lift():
    src = np.array([[[40, 90, 150]]], np.uint8)
    out = apply_tone_curve(src, 2.4).astype(float)[0, 0]
    ref = src.astype(float)[0, 0]
    np.testing.assert_allclose(out / out.max(), ref / ref.max(), atol=0.02)


def test_no_channel_is_driven_into_clipping():
    """The lift is capped by whichever channel is brightest, so a saturated
    highlight loses lift rather than losing its colour."""
    for gamma in (1.5, 2.4, 3.0):
        out = apply_tone_curve(_colours(), gamma)
        assert out.max() <= 255
    # The brightest channel lands exactly where the table puts it — that is
    # what caps the gain — and only an input of 255 comes out at 255.
    bright = np.array([[[200, 230, 250]]], np.uint8)
    assert apply_tone_curve(bright, 3.0).max() == int(tone_curve_lut(3.0)[250])
    assert apply_tone_curve(np.array([[[200, 230, 255]]], np.uint8), 3.0).max() == 255


def test_a_grey_image_still_follows_the_lookup_table_exactly():
    """The exponents were derived from grey statistics, so grey has to keep
    behaving exactly as the table says or their meaning changes."""
    lut = tone_curve_lut(2.4)
    for v in (0, 5, 40, 128, 200, 255):
        out = apply_tone_curve(grey(v), 2.4)
        assert abs(int(out[0, 0, 0]) - int(lut[v])) <= 1
