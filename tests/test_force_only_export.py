"""Publish the MEASUREMENT without the policy built on top of it.

`force_*_penetration_mm` is `force / k` and `force_*_target_pose` is the
observed pose advanced along the pressing direction by that penetration. Both
are functions of `dexforce.STIFFNESS_N_PER_M`, which is a control-policy knob,
not a measured property of the gel -- the runbook is explicit that the virtual
target "does not measure physical gel indentation".

The exporter refuses a virtual displacement past the 4.25 mm gel thickness, so
at the shipped k = 2 N/mm the gate caps usable force at 8.5 N. v8 measures to
15 N and the data reaches 14.4 N with no ceiling saturation at all: on the
2026-09-16 pilot, 11.96% of contact frames commanded a target past the gate.
Raising k to clear the gate would be choosing a number nobody measured and
silently restating every published target pose.

So the operator's call on 2026-09-16 was to ship the force and withhold the
policy. `stiffness=None` says exactly that, and says it structurally: with no
k there is nothing to derive, so the derived columns cannot be written by
accident and there is no gate left to fail.

`source_frame` stays. It is provenance -- which tactile frame produced this
number -- not a function of k, and it is what makes the alignment checkable.
"""
from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from twm.force_recovery import export_force_columns as EX


def _table(n=20):
    rng = np.random.default_rng(0)
    pose = [[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0] for _ in range(n)]
    return pa.table({
        "frame_idx": np.arange(n, dtype=np.int32),
        "sensor_left_pose": pose, "sensor_right_pose": pose,
        "tactile_left_is_new": np.ones(n, bool),
        "tactile_right_is_new": np.ones(n, bool),
    })


@pytest.fixture
def npz(tmp_path, monkeypatch):
    """One episode's force files, at forces that BREAK the k=2 gate."""
    n = 20
    d = tmp_path / "force" / "rope" / "2026-09-16"
    d.mkdir(parents=True)
    # 14 N at k=2 is a 7 mm virtual displacement -- past the 4.25 mm gel.
    force = np.linspace(0.0, 14.0, n)
    for side in ("left", "right"):
        np.savez(d / f"episode_000_{side}.npz",
                 force_normal_n=force, max_depth_mm=force / 10,
                 source_frame=np.arange(n, dtype=np.int32),
                 pipeline_version=EX.MIN_PIPELINE_VERSION,
                 force_calibration="test")
    monkeypatch.setattr(EX, "FORCE_ROOT", tmp_path / "force")
    return tmp_path / "force"


def test_force_only_writes_the_measurement_and_its_provenance(npz):
    cols, _ = EX.build_side("rope", "2026-09-16", "episode_000", "left",
                            _table(), None)
    assert set(cols) == {"force_left_normal_n", "force_left_source_frame"}, (
        f"force-only export produced {sorted(cols)}")


def test_force_only_does_not_write_the_stiffness_derived_columns(npz):
    cols, _ = EX.build_side("rope", "2026-09-16", "episode_000", "left",
                            _table(), None)
    for banned in ("force_left_penetration_mm", "force_left_target_pose"):
        assert banned not in cols, (
            f"{banned} was written with no stiffness to derive it from")


def test_a_stiffness_still_writes_all_four(npz):
    """The existing behaviour is untouched when a k IS supplied."""
    cols, _ = EX.build_side("rope", "2026-09-16", "episode_000", "left",
                            _table(), 2.0)
    assert set(cols) == {"force_left_normal_n", "force_left_penetration_mm",
                         "force_left_target_pose", "force_left_source_frame"}


def test_the_force_values_are_identical_either_way(npz):
    """Withholding the policy must not perturb the measurement."""
    a, _ = EX.build_side("rope", "2026-09-16", "episode_000", "left",
                         _table(), None)
    b, _ = EX.build_side("rope", "2026-09-16", "episode_000", "left",
                         _table(), 2.0)
    assert a["force_left_normal_n"].equals(b["force_left_normal_n"])
    assert a["force_left_source_frame"].equals(b["force_left_source_frame"])


def test_force_only_metadata_does_not_claim_a_stiffness(npz):
    """A column stamped with a k it was not built from is a false provenance."""
    m = EX._field_meta("force_left_normal_n", None)
    assert "twm.stiffness_n_per_mm" not in m, (
        "force-only columns carry a stiffness they do not depend on")


def test_the_gate_has_nothing_to_check_without_a_stiffness(npz):
    """The 4.25 mm gate exists to police penetration. With no penetration
    column there is no gate -- it must not fail on absent data."""
    base = {"identity_pass": True, "alignment_pass": True,
            "alignment_rate": 1.0, "roundtrip_max_abs_err_n": 0.0}
    assert EX._gate({**base, "penetration_over_gel_thickness_frac": None}) == 0
    # and it must still refuse a run that DID derive targets past the gel
    assert EX._gate({**base, "penetration_over_gel_thickness_frac": 0.12}) == 1


def test_the_cli_exposes_force_only_and_it_means_no_stiffness(monkeypatch):
    """--force-only must reach run_export as an absent k, not as a flag the
    column builder has to interpret a second time."""
    import sys
    seen = {}
    monkeypatch.setattr(EX, "run_export",
                        lambda k, root: seen.setdefault("k", k) or
                        {"n_episodes": 0, "n_sensor_sides": 0, "total_rows": 0})
    monkeypatch.setattr(EX, "verify", lambda root: {
        "identity_pass": True, "alignment_pass": True, "alignment_rate": 1.0,
        "roundtrip_max_abs_err_n": 0.0,
        "penetration_over_gel_thickness_frac": None})
    monkeypatch.setattr(EX, "_print", lambda r: None)
    monkeypatch.setattr(sys, "argv", ["x", "export", "--force-only"])
    assert EX.main() == 0
    assert seen["k"] is None, f"CLI passed stiffness {seen['k']!r}"
