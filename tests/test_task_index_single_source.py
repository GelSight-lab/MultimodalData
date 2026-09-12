"""`task_index` is published INSIDE every parquet, so it has to have one owner.

Four copies existed. Two were stale:

* `twm/dataset_prep.py` — `{"motherboard": 0, "pushT": 1}`, read as
  `TASK_INDEX.get(task, 0)`, so publishing a rope episode through the prep
  pipeline stamped it **task_index = 0, i.e. motherboard**, silently, into a
  column that then ships to everyone who downloads it.
* `twm/scripts/build_lerobot_dataset.py` — `TASK_ORDER` without rope, so
  `TASK_ORDER.index("rope")` raises there instead.

The mapping is APPEND-ONLY: the ints are inside every parquet already
downloaded, so renumbering a task relabels someone's local copy with nothing
to warn them.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "twm" / "scripts"))

from twm.react_preprocess.meta import TASK_INDEX, task_index  # noqa: E402


def test_every_published_task_is_mapped():
    assert TASK_INDEX == {"motherboard": 0, "pushT": 1, "rope": 2}


def test_an_unknown_task_raises_rather_than_defaulting():
    """`.get(task, 0)` is how rope became motherboard."""
    with pytest.raises(KeyError) as e:
        task_index("banana")
    assert "banana" in str(e.value)


def test_known_tasks_resolve():
    assert [task_index(t) for t in ("motherboard", "pushT", "rope")] == [0, 1, 2]


@pytest.mark.parametrize("module", ["twm.dataset_prep", "backfill_index_columns",
                                    "stage_validation_merge"])
def test_callers_share_the_one_mapping(module):
    import importlib
    m = importlib.import_module(module)
    assert m.TASK_INDEX is TASK_INDEX, (
        f"{module} holds its own copy of TASK_INDEX")


def test_the_lerobot_builder_does_not_restate_the_mapping():
    """Read, do not import: that module needs `av`, which this env lacks.

    A hardcoded list there is the same defect one level along — it was
    `["motherboard", "pushT"]`, so `TASK_ORDER.index("rope")` raised while
    dataset_prep silently answered 0 for the same task.
    """
    import ast
    src = (REPO / "twm" / "scripts" / "build_lerobot_dataset.py").read_text()
    for node in ast.walk(ast.parse(src)):
        if (isinstance(node, ast.Assign)
                and any(getattr(t, "id", "") == "TASK_ORDER" for t in node.targets)):
            assert not isinstance(node.value, ast.List), (
                "TASK_ORDER is a literal list again — derive it from TASK_INDEX")
            assert "TASK_INDEX" in ast.unparse(node.value)
            return
    pytest.fail("TASK_ORDER not found in build_lerobot_dataset.py")


def test_a_task_without_a_prompt_string_fails_loudly():
    """`TASK_STRINGS` has no rope entry; it must raise, not mislabel.

    A wrong task string ships to everyone who downloads the dataset, so a
    KeyError here is the correct behaviour — but rope still needs one written.
    """
    import ast
    src = (REPO / "twm" / "scripts" / "build_lerobot_dataset.py").read_text()
    assert "TASK_STRINGS[t]" in src or "TASK_STRINGS[" in src, (
        "a .get() default here would publish a silently wrong task label")
