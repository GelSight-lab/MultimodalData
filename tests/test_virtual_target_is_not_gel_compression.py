"""The virtual displacement is a CONTROLLER quantity, not a gel compression.

`dexforce.STIFFNESS_N_PER_M` is the impedance stiffness of the arm -- the
module says so itself ("Impedance controller stiffness the targets are
computed for", "low-mid for Franka-class arms (~150-3000 N/m)"). So
`penetration_mm = F / k` is the virtual deflection the controller commands,
and the actual gel compression is a different quantity entirely, measured
separately by the LUT depth field.

`verify` nonetheless FAILED the export whenever that virtual deflection
exceeded `GEL_THICKNESS_MM`, and the constant's own comment gives the reason
away: "a penetration INTERPRETED AS GEL COMPRESSION may not exceed it". The
interpretation was wrong, so the gate was comparing a control quantity against
sensor geometry.

The consequence was not cosmetic. v8 measures to 15 N, at k = 2 N/mm that is a
7.5 mm virtual deflection -- entirely ordinary for a Franka-class arm -- and
the gate refused 11.96% of contact frames, which is why the force-informed
action columns were nearly dropped from the release.

What is still worth enforcing is the controller identity (k*|dp| == F), the
no-contact identity, and a sanity bound that would catch a k off by orders of
magnitude. None of those mention the gel.
"""
from __future__ import annotations

import pytest

from twm.force_recovery import export_force_columns as EX


def _report(**over):
    r = {"identity_pass": True, "alignment_pass": True, "alignment_rate": 1.0,
         "roundtrip_max_abs_err_n": 0.0, "force_only": False,
         "penetration_over_gel_thickness_frac": 0.0,
         "penetration_p50_p95_p99_max_mm": (0.1, 1.0, 2.0, 3.0)}
    r.update(over)
    return r


def test_a_virtual_deflection_past_the_gel_thickness_is_not_a_failure():
    """7.5 mm at k = 2 N/mm is what 15 N costs, and it is a normal command."""
    assert EX._gate(_report(penetration_over_gel_thickness_frac=0.1196)) == 0


def test_even_every_row_past_it_is_not_a_failure():
    assert EX._gate(_report(penetration_over_gel_thickness_frac=1.0)) == 0


def test_a_broken_controller_identity_still_fails():
    """k*|dp| == F is the one thing the column actually promises."""
    assert EX._gate(_report(roundtrip_max_abs_err_n=1e-3)) == 1


def test_a_moved_no_contact_pose_still_fails():
    assert EX._gate(_report(identity_pass=False)) == 1


def test_a_misalignment_still_fails():
    assert EX._gate(_report(alignment_pass=False, alignment_rate=0.5)) == 1


def test_an_absurd_displacement_fails():
    """A k off by orders of magnitude would command metres. That is worth
    catching, and it is a statement about the CONTROLLER, not the gel."""
    assert EX._gate(_report(
        penetration_p50_p95_p99_max_mm=(1.0, 5.0, 20.0, 500.0))) == 1


def test_the_sanity_bound_does_not_trip_on_ordinary_commands():
    assert EX._gate(_report(
        penetration_p50_p95_p99_max_mm=(0.5, 3.0, 6.0, 7.5))) == 0


def test_force_only_still_passes_with_nothing_derived():
    assert EX._gate(_report(force_only=True,
                            penetration_over_gel_thickness_frac=None,
                            penetration_p50_p95_p99_max_mm=None,
                            roundtrip_max_abs_err_n=None)) == 0
