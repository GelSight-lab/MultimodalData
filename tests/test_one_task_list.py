"""One list of tasks. Nine copies is how rope kept falling out of the pipeline.

`pipeline_stages.TASKS` is the list. It was copied, by hand, into at least nine
other modules, and four of those copies never learned about rope:

    force_recovery/batch_worker.py        ("pushT", "motherboard")
    force_recovery/update_dataset_readme.py
    scripts/eyeball_flags.py
    scripts/oneoff/*

The consequence measured on 2026-09-15: the force stage reported exit 0 for
rope while producing nothing for it, because rope was not in batch_worker's
own tuple. The stage that came next refused — 12 sensor-sides without an npz —
and the run died four stages in, having claimed success at the stage that
actually failed.

A stage that exits 0 without doing its work is worse than one that crashes.
"""
import ast
import pathlib

import pytest

import twm.pipeline_stages as PS

REPO = pathlib.Path(__file__).resolve().parents[1]
# The one-off scripts are kept as a record of what was run once, against the
# data as it was then. They are not on any path the scheduler walks.
EXEMPT = (
    "scripts/oneoff/",          # kept as a record of what was run once
    "dataset_stats_fig.py",     # a drawing ORDER, and it includes `validation`
    "eyeball_flags.py",         # an argparse `choices` for a hand-run tool
    "test_transform_audit.py",  # a script-local audit over two named tasks
)


def _task_tuples(path):
    """Every literal tuple/list whose members are exactly task names."""
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return []
    known = set(PS.TASKS) | {"validation"}
    out = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Tuple, ast.List)):
            vals = [e.value for e in node.elts
                    if isinstance(e, ast.Constant) and isinstance(e.value, str)]
            if vals and len(vals) == len(node.elts) and set(vals) <= known \
                    and set(vals) & set(PS.TASKS):
                out.append((node.lineno, tuple(vals)))
    return out


def test_no_module_on_the_pipeline_path_restates_the_task_list():
    bad = []
    for p in sorted(REPO.glob("twm/**/*.py")):
        rel = str(p.relative_to(REPO))
        if any(x in rel for x in EXEMPT) or rel.endswith("pipeline_stages.py"):
            continue
        for lineno, vals in _task_tuples(p):
            if set(vals) != set(PS.TASKS):
                bad.append(f"{rel}:{lineno} {vals}")
    assert not bad, (
        "these restate the task list and disagree with pipeline_stages.TASKS; "
        "a stage whose own tuple omits a task exits 0 having done nothing for "
        "it:\n  " + "\n  ".join(bad))
