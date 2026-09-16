"""No force channel means the virtual-target check does not apply — not fail.

The preview certifier verifies that the displayed virtual target labels the
tile beside it. The target is `observed_pose + (f / K) * n̂`: without a force
there is nothing to compute, so `ev_tgt` comes back empty and the check
reported

    the virtual target labels the tile beside it:
    UNVERIFIED: no target computed on any sample

as a FAILURE, blocking the publish of `toy` — a task deliberately built
without force while a new estimator is written.

"No evidence" is not "evidence of wrong". The same distinction the force
overlay gate already draws: a DECLARED absence passes, an undeclared one does
not. A check that cannot run has to say so and stand aside, or every
force-free task is unpublishable.

What must NOT change: samples that DO have a target are still judged, and a
task whose episodes declare force but produce no target is still a failure —
that is a broken overlay, not an absent channel.
"""
import pytest


def _run(ev_tgt, tgt_bad, force_declared):
    """The decision under test, extracted so it can be exercised without
    decoding a preview video."""
    from twm.scripts.test_preview_alignment import target_verdict
    return target_verdict(ev_tgt, tgt_bad, force_declared)


def test_no_target_and_no_force_is_not_a_failure():
    ok, evidence = _run([], [], force_declared=False)
    assert ok is True
    assert "no force channel" in evidence.lower()


def test_no_target_but_force_declared_is_still_a_failure():
    """The overlay is broken, or the estimator wrote nothing. Either way the
    task claims a channel it is not displaying."""
    ok, evidence = _run([], [], force_declared=True)
    assert ok is False


def test_a_bad_target_still_fails():
    ok, _ = _run(["ep0: lag +9"], [True], force_declared=True)
    assert ok is False


def test_a_good_target_still_passes():
    ok, evidence = _run(["ep0: lag 0"], [False], force_declared=True)
    assert ok is True and "lag 0" in evidence
