"""The scheduler must ship the force-column policy the operator chose.

`_export` emitted a bare `export_force_columns export`, which writes all eight
columns at the default stiffness. On 2026-09-16 the operator chose to publish
the measurement without the control policy, so the scheduled export has to
carry `--force-only` -- otherwise the automated chain quietly reverts the
decision, and the 4.25 mm gel gate then fails the run at 11.96% of contact
frames.

Declared as a constant next to the stage list rather than typed into the argv,
so "what force columns do we publish" has one answer in the place the pipeline
is defined, and a test can read it.
"""
from __future__ import annotations

import pytest

import twm.pipeline_stages as PS


def test_the_policy_is_declared():
    assert hasattr(PS, "FORCE_COLUMN_POLICY"), (
        "the force-column policy is not declared anywhere the scheduler reads")
    assert PS.FORCE_COLUMN_POLICY in ("force-only", "with-targets")


def test_the_scheduled_export_carries_the_policy():
    cmds = [[str(x) for x in c] for c in PS._export()]
    assert cmds, "the export stage produced no command"
    flat = " ".join(" ".join(c) for c in cmds)
    if PS.FORCE_COLUMN_POLICY == "force-only":
        assert "--force-only" in flat, (
            "the scheduled export would write the stiffness-derived columns "
            "the operator withheld")
    else:
        assert "--force-only" not in flat


def test_the_flag_the_exporter_actually_accepts():
    """A policy that names a flag the CLI rejects fails at the last step."""
    from pathlib import Path
    from twm.force_recovery import export_force_columns as EX
    src = Path(EX.__file__).read_text()
    assert '"--force-only"' in src
