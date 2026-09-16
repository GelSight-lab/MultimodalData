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
# Names that are a FACT about particular tasks, not a copy of the list:
# only the motherboard has a rigid body on the manipulated object, and a task
# whose extrinsics predate CALIB_SESSIONS keeps its historical entry.
EXEMPT_NAMES = ("OBJECT_BODY", "CALIB_DIRS", "EXPECTED_EPOCH", "WORLD_OFFSET",
                # Per-task CONFIG, not a copy of the list: which episodes a
                # one-off clip script uses, and the human-readable task string.
                "TASK_CFG", "TASK_STRINGS")

EXEMPT = (
    "scripts/oneoff/",          # kept as a record of what was run once
    "dataset_stats_fig.py",     # a drawing ORDER, and it includes `validation`
    "eyeball_flags.py",         # an argparse `choices` for a hand-run tool
    "test_transform_audit.py",  # a script-local audit over two named tasks
)


def _exempt_lines(tree):
    """Line numbers of literals assigned to a name that is a FACT about
    particular tasks rather than a copy of the list. Built in ONE pass: walking
    the whole tree per node was quadratic and hung the suite."""
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and n.value is not None:
            if any(isinstance(t, ast.Name) and t.id in EXEMPT_NAMES
                   for t in n.targets):
                for sub in ast.walk(n.value):
                    ln = getattr(sub, "lineno", None)
                    if ln is not None:
                        out.add(ln)
    return out


def _task_tuples(path):
    """Every literal tuple/list whose members are exactly task names."""
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return []
    known = set(PS.TASKS) | {"validation"}
    out = []
    skip = _exempt_lines(tree)
    for node in ast.walk(tree):
        if getattr(node, "lineno", -1) in skip:
            continue
        # Dict KEYS too. H5_ROOTS was a fourth copy of the task list and this
        # check walked straight past it, so `toy` reached the build CLI and was
        # rejected by name after everything ahead of it had been built.
        if isinstance(node, ast.Dict):
            keys = [k.value for k in node.keys
                    if isinstance(k, ast.Constant) and isinstance(k.value, str)]
            if keys and len(keys) == len(node.keys) and set(keys) <= known \
                    and set(keys) & set(PS.TASKS) and set(keys) != set(PS.TASKS):
                out.append((node.lineno, tuple(keys)))
            continue
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
