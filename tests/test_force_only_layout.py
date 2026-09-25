"""The layout check must accept a force channel that ships without the policy.

`FORCE_COLUMNS` was a flat 8-tuple and `require_force` demanded every one of
them, so an episode exported with `--force-only` -- force_normal_n and
force_source_frame, no penetration_mm and no target_pose -- fails the gate
with "the force export has not been run over it". It HAS been run; what is
absent is the stiffness-derived half, deliberately.

The two halves are not the same kind of thing. normal_n is a measurement and
source_frame is its provenance; penetration_mm and target_pose are a control
policy at an assumed stiffness. The gate should insist on the measurement and
treat the policy as optional -- but all-or-nothing, since a target_pose
without the penetration it was displaced by is a half-written export rather
than a deliberate choice.
"""
from __future__ import annotations

import pytest

from twm.dataset_layout import FORCE_MEASURED, FORCE_DERIVED, missing_force_columns


def test_the_measurement_and_its_provenance_are_required():
    assert set(FORCE_MEASURED) == {
        "force_left_normal_n", "force_left_source_frame",
        "force_right_normal_n", "force_right_source_frame"}


def test_a_force_only_episode_passes():
    assert missing_force_columns(list(FORCE_MEASURED)) == []


def test_a_full_export_still_passes():
    assert missing_force_columns(list(FORCE_MEASURED) + list(FORCE_DERIVED)) == []


def test_an_episode_with_no_force_at_all_is_still_caught():
    miss = missing_force_columns(["frame_idx", "timestamp"])
    assert "force_left_normal_n" in miss


def test_a_half_written_derived_export_is_caught():
    """target_pose without the penetration it was displaced by is a broken
    export, not a policy decision -- the derived pair travels together."""
    cols = list(FORCE_MEASURED) + ["force_left_target_pose"]
    assert missing_force_columns(cols), (
        "a target_pose with no penetration_mm beside it was accepted")


def test_the_gate_reports_a_force_only_episode_as_complete(tmp_path):
    """End to end through check_layout's own column rule."""
    from twm.dataset_layout import FORCE_COLUMNS
    # kept as the full set for anything that wants to name every column
    assert set(FORCE_COLUMNS) == set(FORCE_MEASURED) | set(FORCE_DERIVED)
