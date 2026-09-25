"""A preview must draw the force values it LABELS.

`build_one_preview` takes `force_root` and falls back to the module-level
`FORCE_ROOT` -- the production tree -- when nobody passes one. The release
preview builder never passed one, so there was no way to render previews
against a candidate force tree: every run read production regardless of what
the force writer's environment said.

That matters during a force reprocessing run. The v8 runbook produces
candidate NPZ under its own run directory and says, in as many words, "Never
render v8 labels over v7 values" -- which is exactly what the default
fallback does, silently, because a preview whose numbers came from the wrong
tree looks like any other preview.

Changing the force WRITER's environment does not retarget the renderer: the
fallback is a module constant, read at call time, in a different process.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "twm" / "scripts"))


def test_the_release_builder_offers_a_force_root_flag():
    """Without the flag there is no way to ask for the candidate tree."""
    import build_release_previews as BRP
    parser_src = Path(BRP.__file__).read_text()
    assert "--force-root" in parser_src, (
        "build_release_previews.py cannot be pointed at a candidate force "
        "tree, so a reprocessing run can only render against production")


def test_the_renderer_passes_the_chosen_root_through(tmp_path, monkeypatch):
    """The flag has to reach `build_one_preview`, not just be accepted."""
    import build_episode_previews as BEP
    import build_release_previews as BRP

    seen = {}

    def fake_build_one_preview(h5, out, clip_s, speed, *a, **kw):
        seen["force_root"] = kw.get("force_root")

    monkeypatch.setattr(BEP, "build_one_preview", fake_build_one_preview)
    monkeypatch.setattr(BEP, "_load_proj_calibs",
                        lambda task, date=None: ([], None, None, None))
    monkeypatch.setattr(BEP, "_parquet_trim_and_rows",
                        lambda t, d, e, p=None: (7, 50))

    candidate = tmp_path / "candidate_force"
    render, _ = BRP.make_renderer("rope", 30.0, 1.0, force_root=candidate)
    render(dict(task="rope", date="2026-09-16", episode="episode_000",
                trim_offset=7, world_offset=(0.0, 0.0, 0.0),
                h5=tmp_path / "episode_000.h5", out=tmp_path / "o.mp4"))

    assert seen["force_root"] == candidate, (
        f"the renderer passed {seen['force_root']!r}; previews would carry "
        f"force values from whatever tree the module default names")


def test_omitting_it_still_means_production():
    """The flag is opt-in: an ordinary release run must not change behaviour."""
    import build_episode_previews as BEP
    import build_release_previews as BRP
    import inspect
    sig = inspect.signature(BRP.make_renderer)
    assert sig.parameters["force_root"].default is None
