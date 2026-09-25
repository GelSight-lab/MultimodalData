"""The preview must show the gel frame its force was computed FROM.

The GelSight runs at ~17.8 Hz against a 29.8 Hz write tick, so the published
stream is a resample of the raw one. `force_<side>_source_frame` records, per
row, which raw H5 gel frame that row's scalars came from — about N+2.

The renderer read `gelsight/<side>/frames[N]` straight out of the H5, so the
force disc it drew beside the tile came from a frame ~2 ahead of the tile.
Measured on motherboard/2026-09-11/episode_000: force leads the raw H5 image
by 2 frames (r=0.991 at -2 vs 0.980 at 0), and re-indexing by source_frame
collapses it to 0.

Per SIDE, not one shared value: the two GelSights are independent streams with
their own timestamps, and their source_frame columns differ.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "twm" / "scripts"))
import build_episode_previews as BEP  # noqa: E402


def test_uses_the_frame_the_force_came_from():
    src = {"left": np.array([2, 3, 3, 5, 5]), "right": np.array([2, 2, 4, 4, 6])}
    assert BEP.gel_frame(src, "left", row=0, h5_frame=0, n_gel=100) == 2
    assert BEP.gel_frame(src, "right", row=2, h5_frame=2, n_gel=100) == 4


def test_falls_back_to_the_tick_when_the_column_is_absent():
    """Older releases have no source_frame; the tick is the honest answer."""
    assert BEP.gel_frame({}, "left", row=7, h5_frame=7, n_gel=100) == 7


def test_falls_back_outside_the_published_rows():
    """A sampled tick before/after the published episode has no row."""
    src = {"left": np.array([2, 3, 3])}
    assert BEP.gel_frame(src, "left", row=None, h5_frame=9, n_gel=100) == 9


def test_never_indexes_past_the_recording():
    src = {"left": np.array([98, 99, 120])}
    assert BEP.gel_frame(src, "left", row=2, h5_frame=2, n_gel=100) == 99


def test_row_beyond_the_column_falls_back():
    src = {"left": np.array([2, 3])}
    assert BEP.gel_frame(src, "left", row=5, h5_frame=5, n_gel=100) == 5
