"""Colour arrays in the raw HDF5 are BGR, and the file says so.

The RealSense is opened as `rs.format.bgr8`; the GelSight and wrist frames
come from `cv2.imdecode`, also BGR. The recorder stores what it is handed, so
every colour dataset is BGR. The published mp4 is the opposite -- the encoder
is told `pix_fmt="bgr24"` and a standard decoder returns RGB -- and both are
correct for what they are.

Reading the raw file as RGB is what is not. It turns the blue pushT block
orange, which looks like a lighting oddity rather than a defect. The repo
already carries `fix_channel_order.py` from the last time a channel swap went
unnoticed across a dataset, so the fact is asserted here rather than left to
whoever next opens an episode.
"""
from pathlib import Path

import numpy as np
import pytest


def test_the_capture_path_requests_bgr():
    src = Path("camera_stream/realsense_stream.py").read_text()
    assert "rs.format.bgr8" in src
    assert "rs.format.rgb8" not in src


def test_the_schema_states_the_channel_order():
    """A reader must not have to open the driver to learn this."""
    doc = Path("twm/recorder/schema.py").read_text().split('"""')[1]
    assert "BGR" in doc, "the episode layout does not say which way the channels go"


def test_the_encoder_declares_bgr_input():
    """The published mp4 is RGB only because ffmpeg is told the input is BGR;
    silently flipping this to rgb24 would swap every published frame."""
    src = Path("twm/react_preprocess/encode.py").read_text()
    assert 'pix_fmt="bgr24"' in src


BLUE = Path("/media/yxma/Disk1/twm/data/pushT")


@pytest.mark.skipif(not BLUE.is_dir(), reason="rig recordings not on this machine")
def test_the_pushT_block_is_blue_when_read_as_bgr():
    """The physical ground truth: that block is blue. Read as BGR its first
    channel dominates; read as RGB the third would."""
    import h5py, hdf5plugin  # noqa: F401
    import cv2

    eps = sorted(BLUE.rglob("episode_*.h5"))
    if not eps:
        pytest.skip("no pushT episodes")
    with h5py.File(eps[-1], "r") as f:
        ds = f["realsense/cam2/color"]
        frame = np.asarray(ds[len(ds) // 2])
    # the saturated region of the scene is the block
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = (hsv[:, :, 1] > 120) & (hsv[:, :, 2] > 60)
    if mask.sum() < 500:
        pytest.skip("no saturated region in this frame")
    b, g, r = frame[mask].astype(float).mean(0)
    assert b > r, f"channel 0 ({b:.0f}) should dominate channel 2 ({r:.0f}) on a blue block"
