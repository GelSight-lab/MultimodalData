"""The exporter's OUTPUT root must be redirectable by environment.

`STAGE_ROOT` and `FORCE_ROOT` are `Path(os.environ.get("REACT_...", default))`.
`EXPORT_ROOT` was a bare literal, and the export stage runs OUT OF PROCESS --
`pipeline_stages.BY_NAME["export"].commands()` is
`python -m twm.force_recovery.export_force_columns export`, with no `--root`.
So a monkeypatched root never reached it, and the end-to-end smoke test, whose
inputs all pointed into its sandbox, wrote its synthetic output into the real
`/media/yxma/Disk1/twm/release_force`.

That is not hypothetical. On 2026-09-22 the production export tree held
`rope/2026-09-20/episode_000` -- 120 synthetic rows whose sidecar named
`/tmp/pytest-of-yxma/pytest-321/twm0/release/...` as its source. It had also
overwritten the run manifest, so the published receipt described the entire
force export as that one fake episode.

`upload_force_columns` publishes from this root. A test must not be able to
put a file there.
"""
from __future__ import annotations

import importlib
import os
from pathlib import Path

import pytest


def _reload(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("REACT_FORCE_EXPORT_ROOT", raising=False)
    else:
        monkeypatch.setenv("REACT_FORCE_EXPORT_ROOT", str(value))
    import twm.force_recovery.export_force_columns as EX
    return importlib.reload(EX)


def test_the_env_var_redirects_the_export_root(monkeypatch, tmp_path):
    EX = _reload(monkeypatch, tmp_path / "sandbox_export")
    try:
        assert EX.EXPORT_ROOT == tmp_path / "sandbox_export"
    finally:
        _reload(monkeypatch, None)


def test_the_default_is_still_production(monkeypatch):
    EX = _reload(monkeypatch, None)
    assert EX.EXPORT_ROOT == Path("/media/yxma/Disk1/twm/release_force")


def test_the_export_stage_argv_can_be_sandboxed(monkeypatch, tmp_path):
    """The stage passes no --root, so env is the ONLY redirect it has."""
    import twm.pipeline_stages as PS
    argv = [str(x) for x in PS.BY_NAME["export"].commands(task="rope")[0]]
    assert "--root" not in argv, (
        "if the stage ever passes --root this test should check that instead")
    EX = _reload(monkeypatch, tmp_path / "sandbox_export")
    try:
        assert str(EX.EXPORT_ROOT).startswith(str(tmp_path))
    finally:
        _reload(monkeypatch, None)
