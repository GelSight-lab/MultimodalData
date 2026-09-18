from pathlib import Path

import pytest

from twm.scripts.publish_old_motherboard_update import operation_specs


DATES = ("2026-05-10", "2026-05-11", "2026-05-19")


def _episodes(root: Path, count=32):
    left = count
    index = 0
    for date in DATES:
        n = min(left, 12 if date != "2026-05-19" else left)
        for _ in range(n):
            path = root / "meta" / date / f"episode_{index:03d}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"parquet")
            index += 1
        left -= n
    return index


def test_tool_operations_stay_under_v8_rebuild_tools(tmp_path):
    root = tmp_path / "tools"
    (root / "twm").mkdir(parents=True)
    (root / "README.md").write_text("instructions")
    (root / "manifest.json").write_text("{}")
    (root / "twm/mod.py").write_text("x = 1")

    specs = operation_specs("tools", root)

    assert {s.path_in_repo for s in specs} == {
        "old_data/motherboard/v8_rebuild_tools/README.md",
        "old_data/motherboard/v8_rebuild_tools/manifest.json",
        "old_data/motherboard/v8_rebuild_tools/twm/mod.py",
    }


def test_action_operations_replace_exactly_32_parquet_and_add_sidecars(tmp_path):
    root = tmp_path / "actions"
    assert _episodes(root) == 32
    (root / "action_update_manifest.json").write_text("{}")
    event = root / "action_repair_events/2026-05-10/episode_000.json"
    event.parent.mkdir(parents=True)
    event.write_text("{}")

    specs = operation_specs("actions", root)
    paths = {s.path_in_repo for s in specs}

    assert len([p for p in paths if p.endswith(".parquet")]) == 32
    assert "old_data/motherboard/meta/2026-05-10/episode_000.parquet" in paths
    assert "old_data/motherboard/action_repair_events/2026-05-10/episode_000.json" in paths
    assert "old_data/motherboard/action_update_manifest.json" in paths


def test_action_operations_refuse_partial_or_unexpected_date(tmp_path):
    partial = tmp_path / "partial"
    _episodes(partial, count=31)
    (partial / "action_update_manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="32"):
        operation_specs("actions", partial)

    wrong = tmp_path / "wrong"
    _episodes(wrong)
    extra = wrong / "meta/2026-06-01/episode_999.parquet"
    extra.parent.mkdir(parents=True)
    extra.write_bytes(b"parquet")
    (wrong / "action_update_manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="date"):
        operation_specs("actions", wrong)


def test_force_operations_publish_parquet_and_v8_sidecars(tmp_path):
    root = tmp_path / "force"
    _episodes(root)
    for parquet in root.glob("meta/*/*.parquet"):
        parquet.with_suffix(".force.json").write_text("{}")
    (root / "force_export_manifest.json").write_text("{}")
    (root / "force_export_verify.json").write_text("{}")

    specs = operation_specs("force", root)
    paths = {s.path_in_repo for s in specs}

    assert len([p for p in paths if p.endswith(".parquet")]) == 32
    assert len([p for p in paths if p.endswith(".force.json")]) == 32
    assert "old_data/motherboard/v8_force_export_manifest.json" in paths
    assert "old_data/motherboard/v8_force_export_verify.json" in paths
