"""`export_force_columns` must be able to export a SUBSET of the release.

It imports STAGE_ROOT from `run_episode` and enumerates
``STAGE_ROOT/*/meta/*/*.parquet``, then refuses if any episode it found lacks
a force npz. While that path was hardcoded to the whole release, the export
could only run when every episode of every task and date had been estimated —
so publishing one task, or one wave of episodes, was impossible. These paths
are parameterised in `react_preprocess.config` for the same reason.
"""
from __future__ import annotations

import importlib
import os
from pathlib import Path

import pytest


def _reload(monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import twm.force_recovery.run_episode as m
    return importlib.reload(m)


def test_stage_root_follows_the_env(monkeypatch, tmp_path):
    m = _reload(monkeypatch, REACT_STAGE_ROOT=str(tmp_path / "wave"))
    assert m.STAGE_ROOT == tmp_path / "wave"


def test_the_defaults_are_unchanged(monkeypatch):
    for k in ("REACT_STAGE_ROOT", "REACT_DATA_ROOT", "REACT_FORCE_RECOVERY_ROOT"):
        monkeypatch.delenv(k, raising=False)
    import twm.force_recovery.run_episode as m
    m = importlib.reload(m)
    assert m.STAGE_ROOT == Path("/media/yxma/Disk1/twm/release")
    assert m.DATA_ROOT == Path("/media/yxma/Disk1/twm/data")
    assert m.OUT_ROOT == Path("/media/yxma/Disk1/twm/force_recovery")


def test_the_exporter_enumerates_the_overridden_root(monkeypatch, tmp_path):
    """The whole point: which episodes the export considers must follow the
    root it is given, or a per-wave export is impossible."""
    root = tmp_path / "wave"
    (root / "motherboard" / "meta" / "2026-09-11").mkdir(parents=True)
    (root / "motherboard" / "meta" / "2026-09-11" / "episode_000.parquet").write_text("x")
    monkeypatch.setenv("REACT_STAGE_ROOT", str(root))

    import twm.force_recovery.run_episode as re_mod
    importlib.reload(re_mod)
    import twm.force_recovery.export_force_columns as ex
    ex = importlib.reload(ex)

    assert ex._episodes() == [("motherboard", "2026-09-11", "episode_000")]
