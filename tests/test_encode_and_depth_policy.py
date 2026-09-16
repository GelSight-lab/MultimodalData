"""Two operator decisions, measured before they were made.

ENCODER PRESET. Measured on 300 real frames of a published rope video, against
the raw pixels they decode to:

    medium    74 fps   1.21 MB   PSNR 48.14 dB   (what shipped until now)
    fast     105 fps   1.21 MB   PSNR 47.96 dB   -0.18 dB for 1.4x
    veryfast 218 fps   0.87 MB   PSNR 46.32 dB   -1.82 dB for 2.9x
    ultrafast 159 fps  3.67 MB                   slower AND 3x larger

`fast` is the free one: same file size, 0.18 dB, which is far below anything
visible or measurable downstream. `veryfast` was NOT taken — the gel gradients
are the force estimator's input, and there is no evidence 46 dB leaves them
intact. Speed that costs unquantified accuracy on a published dataset is not
a bargain.

DEPTH. `--with-depth` encodes a second lossless FFV1 stream per camera and is
the largest single cost in `build`. The operator's decision (2026-09-16) is
that it is not needed. The flag stays — it is how the ten-hour rebuild
happened, and a flag that cannot be asked for is a feature that is gone — but
the scheduler no longer passes it.
"""
import pytest

import twm.pipeline_stages as PS
from twm.react_preprocess.encode import VideoWriter


@pytest.fixture
def cmd(tmp_path):
    """Constructed normally: `_build` reads `self.path`, so bypassing
    __init__ tests a different object than the one that runs."""
    return VideoWriter(tmp_path / "v.mp4")._cmd


def test_the_colour_encoder_uses_the_measured_preset(cmd):
    assert "-preset" in cmd
    assert cmd[cmd.index("-preset") + 1] == "fast", (
        "medium costs 40 % more time for 0.18 dB; veryfast buys 2.9x for "
        "1.82 dB, which nothing has shown is safe for the gel gradients the "
        "force estimator reads")


def test_the_quality_floor_did_not_move(cmd):
    """The preset changes the search, not the target. CRF 18 and yuv444p are
    what make it visually lossless, and both stay."""
    assert cmd[cmd.index("-crf") + 1] == "18"
    assert "yuv444p" in cmd


def test_the_build_stage_no_longer_asks_for_depth():
    for task in PS.TASKS:
        for c in PS._build(task=task, date="2026-09-20"):
            assert "--with-depth" not in [str(x) for x in c], (
                "the scheduler still asks for the depth channel the operator "
                "turned off")


def test_the_depth_flag_still_exists_for_whoever_wants_it():
    """It stays available: a flag that cannot be asked for is a feature that
    is gone, and this one was missing once at a cost of ten hours."""
    import subprocess
    import sys
    r = subprocess.run([sys.executable, "-m", "twm.react_preprocess", "build",
                        "--help"], capture_output=True, text=True)
    assert "--with-depth" in r.stdout
