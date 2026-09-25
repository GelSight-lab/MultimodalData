"""A session recorded after the current solve was recorded THROUGH it.

The module's rule was: a session declares its epoch and nothing infers it.
That rule exists because DATE ORDER does not decide the answer -- pushT's
2026-06-18 belongs to the June-26 solve, measured eight days LATER -- and a
task-level default is how a session once shipped through another session's
extrinsics with nobody noticing.

The operator's standing decision (2026-09-15) is that new sessions use the
current epoch by default. That is not the inference the rule forbids. The live
recorder resolves its own overlay through `current_epoch()`, so a recording
made on or after the solve date was made THROUGH that solve: it is a fact
about how the rig ran, not a guess from the calendar.

The dangerous half is kept: a session dated BEFORE the current solve still has
to be declared, because for those the calendar really does say nothing.
"""
import pytest

import twm.calib_epoch as CE


def test_a_session_after_the_current_solve_defaults_to_it(monkeypatch):
    monkeypatch.setattr(CE, "CURRENT_EPOCH", "2026-09-09")
    assert CE.session_epoch("rope", "2026-09-20") == "2026-09-09"
    assert CE.session_epoch("motherboard", "2026-12-01") == "2026-09-09"


def test_a_session_on_the_solve_date_itself_defaults_to_it(monkeypatch):
    monkeypatch.setattr(CE, "CURRENT_EPOCH", "2026-09-09")
    assert CE.session_epoch("rope", "2026-09-09") == "2026-09-09"


def test_a_session_before_the_current_solve_must_still_declare(monkeypatch):
    """The pushT 2026-06-18 case: it belongs to a solve measured eight days
    later, so nothing about the date decides it."""
    monkeypatch.setattr(CE, "CURRENT_EPOCH", "2026-09-09")
    with pytest.raises(KeyError, match="does not declare"):
        CE.session_epoch("rope", "2026-07-01")


def test_a_declaration_always_wins_over_the_default(monkeypatch):
    """A session recorded after the solve but calibrated differently -- the
    table is still the authority."""
    monkeypatch.setattr(CE, "CURRENT_EPOCH", "2026-09-09")
    monkeypatch.setitem(CE.CALIB_SESSIONS, ("rope", "2026-09-20"), "2026-06-26")
    assert CE.session_epoch("rope", "2026-09-20") == "2026-06-26"


def test_the_existing_declarations_are_unchanged(monkeypatch):
    for (task, date), epoch in CE.CALIB_SESSIONS.items():
        assert CE.session_epoch(task, date) == epoch
