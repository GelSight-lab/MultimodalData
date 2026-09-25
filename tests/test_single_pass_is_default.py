"""One traversal is the default now, and here is the evidence it waited for.

`_encode_rgb_single_pass` has existed, tested, since the day someone measured
that encoding one stream at a time seeks across the recorder's stride —
190 s against 351 s on 600 frames. It was left off by default with a note:

    Not yet the default. It changes the core build path and the difference is
    invisible in the output, so it wants a full session's worth of evidence
    before it becomes what every build does.

Measured 2026-09-16 on rope/2026-09-14/episode_001, a real 8 GB recording of
1989 frames, both paths run end to end:

    two-pass     352 s
    single-pass  253 s      1.39x

    7 of 7 videos byte-for-byte identical
    14 of 14 parquet columns identical, schema metadata identical

`object_pose` looked different at first and is not: rope tracks no object, so
the column is all-NaN on both sides and `==` says nan != nan. Worth recording,
because the same mistake would make any future A/B look like a regression.
"""
import inspect

import pytest

from twm.react_preprocess import pipeline


def test_the_build_traverses_the_file_once_by_default():
    sig = inspect.signature(pipeline.build_episode)
    assert sig.parameters["single_pass"].default is True, (
        "the measured-faster path with byte-identical output is still off")


def test_the_two_pass_path_is_still_reachable():
    """Kept, not deleted: it is the reference the equivalence was measured
    against, and the only way to re-measure it on a future recording."""
    assert hasattr(pipeline, "_encode_cameras")
    assert hasattr(pipeline, "_encode_wrist")
    assert hasattr(pipeline, "_encode_rgb_single_pass")


def test_the_flag_can_still_turn_it_off():
    import subprocess
    import sys
    r = subprocess.run([sys.executable, "-m", "twm.react_preprocess", "build",
                        "--help"], capture_output=True, text=True)
    assert "--single-pass" in r.stdout or "--two-pass" in r.stdout, (
        "a default that cannot be turned off is not a default, it is a rewrite")
