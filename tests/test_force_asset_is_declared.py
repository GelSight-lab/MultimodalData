"""The force stage's fitted-features cache is an ASSET, not a pipeline product.

`force_recovery/feature_cache/glowtact_round_mm.json` holds the measurements
from an August calibration experiment — gel-indentation geometry against
newtons. No stage produces it, it is not in git, and it lives on the data
disk. Without it the force stage exits on its first episode.

Nobody noticed, because on this machine the file has been there since August.
A fresh checkout, another machine, or a deleted cache, and the chain dies four
stages in — after the build has spent hours — with:

    run `build` first (…/glowtact_round_mm.json missing)

which sends the reader to a stage that does not produce it.

Two things follow: the message must name what is actually missing and where it
comes from, and the scheduler must refuse BEFORE the build rather than after.
"""
from dataclasses import replace

import pytest

import twm.pipeline_stages as PS


def test_the_force_stage_declares_its_asset(tmp_path, monkeypatch):
    monkeypatch.setattr(PS, "FORCE_ROOT", tmp_path / "force_recovery")
    why = PS.blocked(PS.BY_NAME["force"], "rope")
    assert why and "glowtact" in why, (
        f"the force stage did not refuse a missing fitted-features cache: {why}")


def test_the_refusal_says_where_the_asset_comes_from(tmp_path, monkeypatch):
    monkeypatch.setattr(PS, "FORCE_ROOT", tmp_path / "force_recovery")
    why = PS.blocked(PS.BY_NAME["force"], "rope")
    assert "build" not in why.split("glowtact")[0].lower() or "not" in why.lower(), \
        f"the message still sends the reader to `build`, which does not produce it: {why}"


@pytest.mark.parametrize("build_ready", [True, False])
def test_a_present_asset_does_not_block(tmp_path, monkeypatch, build_ready):
    root = tmp_path / "force_recovery"
    (root / "feature_cache").mkdir(parents=True)
    (root / "feature_cache" / "glowtact_round_mm.json").write_text("[]")
    (root / "feature_cache" / "glowtact_round_8_15_di4.json").write_text("[]")
    (root / "lut_calibration").mkdir()
    (root / "lut_calibration" / "glowtact_lut.npz").write_bytes(b"")
    monkeypatch.setattr(PS, "FORCE_ROOT", root)
    # This test exercises the asset gate, not the build-completion contract.
    # Scope the prerequisite to this case so host recordings cannot affect it.
    monkeypatch.setitem(PS.BY_NAME, "build", replace(
        PS.BY_NAME["build"], produced=lambda task: build_ready))
    why = PS.blocked(PS.BY_NAME["force"], "rope")
    if build_ready:
        assert why is None
    else:
        assert why and "needs build" in why


@pytest.mark.parametrize('missing', [
    'feature_cache/glowtact_round_8_15_di4.json',
    'lut_calibration/glowtact_lut.npz',
])
def test_v8_force_stage_requires_tail_and_geometry_assets(tmp_path, monkeypatch, missing):
    for rel in ['feature_cache/glowtact_round_mm.json',
                'feature_cache/glowtact_round_8_15_di4.json',
                'lut_calibration/glowtact_lut.npz']:
        if rel != missing:
            path = tmp_path/rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'asset')
    monkeypatch.setattr(PS, 'FORCE_ROOT', tmp_path)
    why = PS.blocked(PS.BY_NAME['force'], 'rope')
    assert why and missing in why
