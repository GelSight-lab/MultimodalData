"""The PUBLISHER's column-loss gate must honour a declared withdrawal too.

There were two `check_no_column_loss` implementations: one in
`react_preprocess/publish.py` and one in `scripts/build_release_publish.py`.
Only the second is on the real publish path (`pipeline_stages` runs the
script), so teaching the first about declared withdrawals achieved nothing --
the force-only publish would still have been refused at the very last step.

Both now decide through one function, `publish.lost_columns`, so a column the
operator has approved withdrawing is approved in both places or neither.

The withdrawal stays per column. A blanket "allow shrinking" flag would
re-open the defect this gate exists for: a publisher that silently reverted
the force channel because two staging trees wrote to the same repo paths and
the last upload won.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "twm" / "scripts"))


def test_both_gates_decide_through_one_function():
    import build_release_publish as BRP
    from twm.react_preprocess.publish import lost_columns
    src = Path(BRP.__file__).read_text()
    assert "lost_columns" in src, (
        "build_release_publish still has its own column-loss arithmetic; a "
        "withdrawal declared for one gate would not be seen by the other")


def test_the_publisher_takes_the_withdrawal_from_the_command_line():
    import build_release_publish as BRP
    src = Path(BRP.__file__).read_text()
    assert "--withdraw-column" in src, (
        "no way to tell the publisher a column is being withdrawn on purpose")


def test_the_derived_force_columns_are_the_documented_withdrawal():
    """What the 2026-09-16 force-only publish withdraws, named in one place."""
    from twm.dataset_layout import FORCE_DERIVED
    assert set(FORCE_DERIVED) == {
        "force_left_penetration_mm", "force_left_target_pose",
        "force_right_penetration_mm", "force_right_target_pose"}


def test_an_undeclared_drop_is_still_refused_by_the_shared_function():
    from twm.react_preprocess.publish import lost_columns
    from twm.dataset_layout import FORCE_MEASURED, FORCE_DERIVED
    old = set(FORCE_MEASURED) | set(FORCE_DERIVED) | {"frame_idx"}
    new = set(FORCE_MEASURED)                      # also drops frame_idx!
    lost = lost_columns(old, new, allow_dropping=frozenset(FORCE_DERIVED))
    assert lost == ["frame_idx"], (
        f"declaring the force columns excused an unrelated drop: {lost}")
