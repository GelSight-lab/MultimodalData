"""Dropping a published column must be declared, not discovered.

`check_no_column_loss` refuses to overwrite a published parquet with one that
has fewer columns. It earned that: a rebuild once silently deleted the force
channel from a dataset whose README documented it, leaving files that looked
completely normal -- nothing about a 19-column parquet says a column is
missing.

On 2026-09-16 the operator chose to publish force WITHOUT the stiffness-derived
half, so the new rope parquet legitimately has four fewer columns than the
published one. That is a decision, not an accident, and the two must not look
alike to this gate.

So the gate takes an explicit set of columns the operator has approved
withdrawing. Naming them is the audit trail; everything unnamed still refuses.
A blanket "allow shrinking" flag would have let the original defect through.
"""
from __future__ import annotations

import pytest

from twm.react_preprocess.publish import lost_columns


PUBLISHED = {"frame_idx", "force_left_normal_n", "force_left_penetration_mm",
             "force_left_target_pose", "force_left_source_frame"}
FORCE_ONLY = {"frame_idx", "force_left_normal_n", "force_left_source_frame"}


def test_an_undeclared_drop_is_still_refused():
    assert lost_columns(PUBLISHED, FORCE_ONLY, allow_dropping=frozenset()) == [
        "force_left_penetration_mm", "force_left_target_pose"]


def test_a_declared_withdrawal_is_allowed():
    ok = frozenset({"force_left_penetration_mm", "force_left_target_pose"})
    assert lost_columns(PUBLISHED, FORCE_ONLY, allow_dropping=ok) == []


def test_declaring_one_column_does_not_excuse_another():
    """The whole point: the approval is per column, not a mode switch."""
    ok = frozenset({"force_left_penetration_mm"})
    assert lost_columns(PUBLISHED, FORCE_ONLY, allow_dropping=ok) == [
        "force_left_target_pose"]


def test_declaring_a_column_that_is_not_being_dropped_changes_nothing():
    ok = frozenset({"some_other_column"})
    assert lost_columns(PUBLISHED, PUBLISHED, allow_dropping=ok) == []


def test_adding_columns_was_never_a_problem():
    assert lost_columns(FORCE_ONLY, PUBLISHED, allow_dropping=frozenset()) == []
