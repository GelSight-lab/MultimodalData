"""Every stage must name its module the way the runner imports it.

`_build` carries this comment:

    `twm.react_preprocess`, not `react_preprocess`: the latter only imports
    with cwd=twm/, which is how the old run_stages invoked it. Every other
    stage here runs from the repo root, and a module path that resolves in one
    and not the other is how a stage dies on its first line.

`_force` did not get the fix. It emitted `python -m force_recovery.batch_worker`
and died with ModuleNotFoundError at 10:30 on 2026-09-15, after the build stage
had just finished four recordings. Nothing downstream ran for three hours.

A comment on one stage does not protect the others. This checks all of them.
"""
import importlib.util
import sys

import pytest

import twm.pipeline_stages as PS


def _module_args(argv):
    return [argv[i + 1] for i, a in enumerate(argv[:-1]) if a == "-m"]


def _all_argv():
    out = []
    for stage in PS.STAGES:
        for task in PS.TASKS:
            try:
                out += [[str(x) for x in c] for c in stage.commands(task=task)]
            except Exception:                            # noqa: BLE001
                pass
    return out


def test_every_module_a_stage_invokes_is_importable_from_the_repo_root():
    bad = []
    for argv in _all_argv():
        for mod in _module_args(argv):
            root = mod.split(".")[0]
            if importlib.util.find_spec(root) is None:
                bad.append(mod)
    assert not bad, (
        f"stage(s) name a module that does not import from the repo root: "
        f"{sorted(set(bad))} — this is the failure `_build`'s comment warns "
        f"about, and it kills the stage on its first line")


def test_the_force_stage_specifically():
    """Named on its own because it is the one that failed, and a regression
    here costs a whole run rather than a stage."""
    argv = [[str(x) for x in c] for c in PS._force(task="rope")]
    mods = [m for a in argv for m in _module_args(a)]
    assert mods, "the force stage invokes no module at all"
    for m in mods:
        assert m.startswith("twm."), f"{m} does not resolve from the repo root"
