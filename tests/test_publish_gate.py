"""The publish gate has to certify the thing it is about to publish.

`build_release_publish` takes `--src` because the Hub holds SEGMENTS: the
chain publishes `release_cut`, not the uncut `release`. But `gate()` invoked
`certify_release.py` with no arguments at all, so it certified the DEFAULT
tree over ALL dates while the upload beneath it carried a different tree and
a one-week scope.

Measured 2026-09-14: the `verify` stage certified `release_cut --since
2026-09-08` and passed both tasks with 0 problems; the publish gate then
spent 43 minutes and 211 GB of disk re-reading the uncut tree back to May.
Two runs of the same certifier, neither one checking what shipped.

This is the ninth defect in this pipeline with one shape — a fact about the
run re-derived at the point of use instead of passed in — so the fix makes
the omission unrepresentable: `gate` takes the tree and the scope as
required arguments.
"""
import subprocess
import sys

import pytest

import twm.scripts.build_release_publish as P
from twm.pipeline_stages import SCOPE_SINCE


def _captured_gate(monkeypatch, **kw):
    seen = []

    class _Done:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, **_):
        seen.append([str(c) for c in cmd])
        return _Done()

    monkeypatch.setattr(subprocess, "run", fake_run)
    P.gate(**kw)
    return [c for c in seen if any("certify_release" in a for a in c)]


def test_the_gate_certifies_the_tree_that_is_about_to_be_published(monkeypatch):
    cmd, = _captured_gate(monkeypatch, src="/media/yxma/Disk1/twm/release_cut",
                          since="2026-09-08")
    assert "--src" in cmd, "certify ran with no tree — it checked the default"
    assert cmd[cmd.index("--src") + 1] == "/media/yxma/Disk1/twm/release_cut"


def test_the_gate_certifies_at_the_scope_the_run_declares(monkeypatch):
    cmd, = _captured_gate(monkeypatch, src="/tmp/t", since="2026-09-08")
    assert "--since" in cmd, \
        "certify ran unscoped — it re-read every date back to May"
    assert cmd[cmd.index("--since") + 1] == "2026-09-08"


def test_the_scope_has_one_declaration(monkeypatch):
    """The default must BE the scheduler's constant, not a copy of it."""
    cmd, = _captured_gate(monkeypatch, src="/tmp/t")
    assert cmd[cmd.index("--since") + 1] == SCOPE_SINCE


def test_publishing_without_naming_a_tree_is_not_expressible(monkeypatch):
    """The defect was `gate()` — callable with nothing decided. Requiring the
    tree is what stops the next caller from re-introducing it.

    subprocess is stubbed because on the UNFIXED code this call does not
    raise: it shells out and re-runs the 43-minute certification that this
    fix exists to prevent. Writing this test without the stub launched it.
    """
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail(
        "gate() ran the certifier without being told which tree to certify"))
    with pytest.raises(TypeError):
        P.gate()
