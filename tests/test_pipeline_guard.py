"""The guard must RUN, and the two checks that matter must be able to fail.

`twm/pipeline_guard.py` encodes 14 hard-won invariants and exits non-zero on a
violation -- but nothing in the repo invoked it, so its verdict was never read.
Worse, two of its checks were structurally unable to report the violations they
were written for, and it takes a green run to make that look like safety:

* `check_single_lag_definition` matched `SHIFT = 15` but not
  `trim, shift = int(z["trim"]), 15` -- lowercase, and unpacked from a tuple.
* `check_no_raw_gel_indexing` matched `f["gelsight/left/frames"][i]` but not
  the two-step `ds = f["gelsight/left/frames"]` / `ds[i]`.

A check that cannot fail is not a check, so each one is exercised here against
code that must be rejected AND against code that must be accepted -- the second
half matters as much: this guard has previously cried wolf on already-corrected
lines, and a guard people learn to ignore is off.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from twm import pipeline_guard as G  # noqa: E402


def _tree(tmp_path: Path, name: str, body: str) -> Path:
    (tmp_path / name).write_text(body)
    return tmp_path


@pytest.fixture
def guarded(tmp_path, monkeypatch):
    """Point the guard at a throwaway tree instead of the repo."""
    monkeypatch.setattr(G, "ROOT", tmp_path)
    return tmp_path


# ── the guard actually runs ───────────────────────────────────────────────────

def test_the_repo_passes_the_guard():
    r = subprocess.run([sys.executable, "-m", "twm.pipeline_guard"],
                       cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


# ── check_single_lag_definition ───────────────────────────────────────────────

def test_tuple_unpacked_lag_is_caught(guarded):
    """The form that shipped in anyforce_react.py for the guard's whole life."""
    _tree(guarded, "m.py", 'import numpy\ntrim, shift = int(z["trim"]), 15\n')
    assert any("redeclares" in b or "hard-codes" in b
               for b in G.check_single_lag_definition())


def test_lowercase_single_lag_is_caught(guarded):
    _tree(guarded, "m.py", "tactile_lag = 15\n")
    assert G.check_single_lag_definition()


def test_importing_the_constant_is_accepted(guarded):
    _tree(guarded, "m.py",
          "from twm.tactile_align import LEGACY_SHIFT\nshift = LEGACY_SHIFT\n")
    assert G.check_single_lag_definition() == []


def test_a_zero_shift_is_accepted(guarded):
    """0 is the honest 'no shift', not a fourth copy of the constant."""
    _tree(guarded, "m.py", "trim, shift = 0, 0\n")
    assert G.check_single_lag_definition() == []


def test_an_unrelated_tuple_assignment_is_accepted(guarded):
    _tree(guarded, "m.py", "w, h = 640, 480\ncrf, fps = 18, 30\n")
    assert G.check_single_lag_definition() == []


# ── check_no_raw_gel_indexing ─────────────────────────────────────────────────

TWO_STEP_BAD = '''
def go(f, trim, s, e):
    ds = f["gelsight/left/frames"]
    return ds[trim + s : trim + e]
'''

TWO_STEP_OK = '''
from twm.tactile_align import gel_index
def go(f, i):
    ds = f["gelsight/left/frames"]
    return ds[gel_index(f, i)]
'''


def test_two_step_raw_indexing_is_caught(guarded):
    """The form that shipped in build_video_release.py."""
    _tree(guarded, "m.py", TWO_STEP_BAD)
    assert G.check_no_raw_gel_indexing()


def test_two_step_corrected_indexing_is_accepted(guarded):
    _tree(guarded, "m.py", TWO_STEP_OK)
    assert G.check_no_raw_gel_indexing() == []


def test_inline_raw_indexing_is_still_caught(guarded):
    _tree(guarded, "m.py", 'x = f["gelsight/right/frames"][i]\n')
    assert G.check_no_raw_gel_indexing()


def test_the_opt_out_marker_is_honoured(guarded):
    _tree(guarded, "m.py",
          'ds = f["gelsight/left/frames"]  # tactile-lag-exempt\n'
          'x = ds[i]  # tactile-lag-exempt\n')
    assert G.check_no_raw_gel_indexing() == []


def test_an_unparseable_file_is_reported_not_fatal(guarded):
    """A guard that dies on one bad file reads as 'no run'."""
    _tree(guarded, "m.py", "def (:\n")
    out = G.check_no_raw_gel_indexing() + G.check_single_lag_definition()
    assert any("cannot parse" in b for b in out)


TWO_STEP_VIA_INDEX_MAP = '''
from twm.react_preprocess.h5io import TactileAlignment
def go(f, align, lo, hi):
    ds = f["gelsight/left/frames"]
    idx = align.index_map
    return ds[lo:hi + 1], idx
'''


def test_a_module_that_maps_through_the_owner_is_accepted(guarded):
    """tactile.py reads `ds[lo:hi+1]` in chunks and maps after — correct.

    Judging the index expression token-by-token flagged it, and run_episode's
    `frames[src(row)]` too. Both go through `index_map`.
    """
    _tree(guarded, "m.py", TWO_STEP_VIA_INDEX_MAP)
    assert G.check_no_raw_gel_indexing() == []


# ── check_single_task_index ───────────────────────────────────────────────────

def test_a_second_task_mapping_is_caught(guarded):
    _tree(guarded, "m.py", 'TASK_INDEX = {"motherboard": 0, "pushT": 1}\n')
    assert G.check_single_task_index()


def test_a_derived_task_order_is_accepted(guarded):
    _tree(guarded, "m.py",
          "TASK_ORDER = [t for t, _ in sorted(TASK_INDEX.items(),"
          " key=lambda kv: kv[1])]\n")
    assert G.check_single_task_index() == []
