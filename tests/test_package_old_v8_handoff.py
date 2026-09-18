import hashlib
import json
from pathlib import Path

from twm.force_recovery.package_old_v8_handoff import build_package


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path):
    source = tmp_path / "source"
    (source / "twm/force_recovery/__pycache__").mkdir(parents=True)
    (source / "twm/react_preprocess").mkdir(parents=True)
    (source / "twm/scripts").mkdir(parents=True)
    (source / "twm/__init__.py").write_text("")
    (source / "twm/pipeline_stages.py").write_text("TASKS = ('motherboard',)\n")
    (source / "twm/force_recovery/worker.py").write_text("VERSION = 8\n")
    (source / "twm/force_recovery/RUNBOOK.md").write_text("runbook\n")
    (source / "twm/force_recovery/__pycache__/worker.pyc").write_bytes(b"cache")
    (source / "twm/react_preprocess/h5io.py").write_text("def open_episode(): pass\n")
    (source / "twm/scripts/publish_old_motherboard_update.py").write_text("# publish\n")

    force = tmp_path / "force"
    rels = (
        "feature_cache/glowtact_round_mm.json",
        "feature_cache/glowtact_round_8_15_di4.json",
        "lut_calibration/glowtact_lut.npz",
    )
    for index, rel in enumerate(rels):
        path = force / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"asset-{index}".encode())
    expected = {rel: _sha256(force / rel) for rel in rels}
    return source, force, expected


def test_package_contains_source_assets_instructions_and_valid_manifest(tmp_path):
    source, force, expected = _fixture(tmp_path)
    out = tmp_path / "package"

    manifest = build_package(source, force, out, asset_sha256=expected)

    required = {
        "README.md",
        "manifest.json",
        "requirements-force-v8.txt",
        "twm/__init__.py",
        "twm/pipeline_stages.py",
        "twm/force_recovery/worker.py",
        "twm/react_preprocess/h5io.py",
        "twm/scripts/publish_old_motherboard_update.py",
        "assets/feature_cache/glowtact_round_mm.json",
        "assets/feature_cache/glowtact_round_8_15_di4.json",
        "assets/lut_calibration/glowtact_lut.npz",
    }
    files = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()}
    assert required <= files
    assert not any("__pycache__" in path or path.endswith(".pyc") for path in files)
    stored = json.loads((out / "manifest.json").read_text())
    assert stored == manifest
    for entry in stored["files"]:
        if entry["path"] != "manifest.json":
            assert entry["sha256"] == _sha256(out / entry["path"])


def test_package_refuses_wrong_calibration_asset(tmp_path):
    source, force, expected = _fixture(tmp_path)
    expected["feature_cache/glowtact_round_mm.json"] = "0" * 64

    try:
        build_package(source, force, tmp_path / "package", asset_sha256=expected)
    except ValueError as exc:
        assert "asset digest" in str(exc)
    else:
        raise AssertionError("wrong calibration asset was accepted")


def test_readme_has_exact_remote_pc_contract(tmp_path):
    source, force, expected = _fixture(tmp_path)
    out = tmp_path / "package"
    build_package(source, force, out, asset_sha256=expected)

    text = (out / "README.md").read_text()
    assert "REACT_DATA_ROOT" in text
    assert "REACT_STAGE_ROOT" in text
    assert "REACT_FORCE_RECOVERY_ROOT" in text
    assert "batch_worker 0 1" in text
    assert "old_data/motherboard/meta" in text
    assert "raw H5" in text
    assert "MP4" in text
