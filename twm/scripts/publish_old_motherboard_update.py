"""Publish old motherboard V8 tools, repaired actions, or V8 force supersets."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.parquet as pq


REPO = "yxma/React"
DATES = ("2026-05-10", "2026-05-11", "2026-05-19")
EXPECTED_EPISODES = 32
BASE = "old_data/motherboard"


@dataclass(frozen=True)
class UploadSpec:
    local_path: Path
    path_in_repo: str


def _metadata_root(root: Path) -> Path:
    direct = root / "meta"
    nested = root / "motherboard" / "meta"
    if direct.is_dir():
        return direct
    if nested.is_dir():
        return nested
    return direct


def _episode_files(root: Path) -> list[Path]:
    meta = _metadata_root(root)
    files = sorted(meta.glob("*/*.parquet"))
    dates = {path.parent.name for path in files}
    unexpected = dates - set(DATES)
    if unexpected:
        raise ValueError(f"unexpected date(s): {sorted(unexpected)}")
    if len(files) != EXPECTED_EPISODES:
        raise ValueError(f"expected {EXPECTED_EPISODES} parquet, found {len(files)}")
    if dates != set(DATES):
        raise ValueError(f"expected dates {DATES}, found {sorted(dates)}")
    return files


def operation_specs(mode: str, root: Path) -> list[UploadSpec]:
    root = Path(root)
    if mode == "tools":
        files = sorted(path for path in root.rglob("*") if path.is_file())
        if not files or not (root / "README.md").is_file() or not (root / "manifest.json").is_file():
            raise ValueError("tool package must contain README.md and manifest.json")
        return [UploadSpec(path, f"{BASE}/v8_rebuild_tools/{path.relative_to(root).as_posix()}")
                for path in files]

    parquets = _episode_files(root)
    meta = _metadata_root(root)
    specs = [UploadSpec(path, f"{BASE}/meta/{path.parent.name}/{path.name}")
             for path in parquets]
    if mode == "actions":
        manifest = root / "action_update_manifest.json"
        if not manifest.is_file():
            raise ValueError("missing action_update_manifest.json")
        specs.append(UploadSpec(manifest, f"{BASE}/action_update_manifest.json"))
        events = sorted((root / "action_repair_events").glob("*/*.json"))
        specs.extend(UploadSpec(
            path, f"{BASE}/action_repair_events/{path.parent.name}/{path.name}")
            for path in events)
    elif mode == "force":
        sidecars = sorted(meta.glob("*/*.force.json"))
        if len(sidecars) != EXPECTED_EPISODES:
            raise ValueError(f"expected {EXPECTED_EPISODES} force sidecars, found {len(sidecars)}")
        specs.extend(UploadSpec(
            path, f"{BASE}/meta/{path.parent.name}/{path.name}")
            for path in sidecars)
        for name in ("force_export_manifest.json", "force_export_verify.json"):
            path = root / name
            if not path.is_file():
                raise ValueError(f"missing {name}")
            specs.append(UploadSpec(path, f"{BASE}/v8_{name}"))
    else:
        raise ValueError(f"unsupported publication mode: {mode}")
    return sorted(specs, key=lambda spec: spec.path_in_repo)


def _validate_force_sidecars(specs: Iterable[UploadSpec]) -> None:
    from twm.force_recovery.react_calib import CALIBRATION_NAME
    from twm.force_recovery.run_episode import PIPELINE_VERSION

    for spec in specs:
        if not spec.local_path.name.endswith(".force.json"):
            continue
        value = json.loads(spec.local_path.read_text())
        for side in value.get("sides", []):
            if int(side.get("pipeline_version", -1)) != int(PIPELINE_VERSION):
                raise ValueError(f"non-V8 force sidecar: {spec.local_path}")
            if str(side.get("force_calibration", "")) != CALIBRATION_NAME:
                raise ValueError(f"wrong force calibration: {spec.local_path}")


def publish(mode: str, root: Path, *, dry_run: bool = False) -> dict:
    specs = operation_specs(mode, root)
    if mode == "force":
        _validate_force_sidecars(specs)
    if dry_run:
        return {"dry_run": True, "operations": len(specs),
                "paths": [spec.path_in_repo for spec in specs]}
    from huggingface_hub import CommitOperationAdd, HfApi

    operations = [CommitOperationAdd(path_in_repo=spec.path_in_repo,
                                     path_or_fileobj=str(spec.local_path))
                  for spec in specs]
    result = HfApi().create_commit(
        repo_id=REPO, repo_type="dataset", operations=operations,
        commit_message={
            "tools": "Add portable V8 rebuild tools for old motherboard data",
            "actions": "Add repaired actions and lost-track labels to old motherboard data",
            "force": "Update old motherboard force to V8",
        }[mode])
    return {"dry_run": False, "operations": len(specs),
            "commit_url": str(result.commit_url), "oid": str(result.oid)}


def verify_remote() -> dict:
    from huggingface_hub import HfFileSystem, hf_hub_download

    fs = HfFileSystem()
    errors = []
    paths = []
    for date in DATES:
        folder = f"datasets/{REPO}/{BASE}/meta/{date}"
        paths.extend(path for path in fs.glob(f"{folder}/episode_*.parquet"))
    if len(paths) != EXPECTED_EPISODES:
        errors.append(f"remote episode count {len(paths)} != {EXPECTED_EPISODES}")
    required = {
        "sensor_left_pose_repaired", "sensor_right_pose_repaired", "action",
        "action_valid_left", "action_valid_right", "action_repaired_left",
        "action_repaired_right", "action_lost_track_left",
        "action_lost_track_right", "action_long_lost_track_left",
        "action_long_lost_track_right",
    }
    for remote in paths:
        try:
            with fs.open(remote, "rb") as stream:
                schema = pq.read_schema(stream)
            missing = required - set(schema.names)
            action_type = schema.field("action").type if "action" in schema.names else None
            if missing or not pa.types.is_fixed_size_list(action_type) or action_type.list_size != 14:
                errors.append(f"{remote}: missing={sorted(missing)} action={action_type}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{remote}: {type(exc).__name__}: {exc}")
    samples = []
    for date in DATES:
        same_date = sorted(path for path in paths if f"/meta/{date}/" in path)
        if not same_date:
            continue
        repo_path = same_date[0].split(f"datasets/{REPO}/", 1)[-1]
        local = hf_hub_download(REPO, repo_path, repo_type="dataset", force_download=True)
        table = pq.read_table(local, columns=["action_valid_left", "action_valid_right"])
        ok = (table["action_valid_left"][-1].as_py() is False
              and table["action_valid_right"][-1].as_py() is False)
        samples.append({"path": repo_path, "terminal_invalid": ok})
        if not ok:
            errors.append(f"{repo_path}: terminal action is valid")
    return {"episodes": len(paths), "samples": samples, "errors": errors}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("tools", "actions", "force", "verify-remote"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.mode == "verify-remote":
        report = verify_remote()
        print(json.dumps(report, indent=2))
        return 1 if report["errors"] else 0
    if args.root is None:
        parser.error("--root is required for publication")
    report = publish(args.mode, args.root, dry_run=args.dry_run)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
