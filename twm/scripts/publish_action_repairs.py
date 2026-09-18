"""Merge concurrently rebuilt force into staged actions, then publish explicitly.

The force base must be the verified export of the immediately preceding HF
commit. A parent-commit guard prevents replacing a newer remote update.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from twm.scripts.export_old_motherboard_actions import (
    _atomic_parquet, _columns_equal, _sha256)


def _derived(name: str) -> bool:
    return (name == 'action' or name.startswith(('action_', 'pose_left_', 'pose_right_'))
            or name in ('sensor_left_pose_repaired', 'sensor_right_pose_repaired'))


def merge_force_table(action: pa.Table, force: pa.Table) -> pa.Table:
    if action.num_rows != force.num_rows:
        raise ValueError('force/action row counts differ')
    # Check every raw field, not only row count; old action fields in the force
    # base are intentionally ignored, since these are what this update replaces.
    for name in force.column_names:
        if name.startswith('force_') or _derived(name):
            continue
        if name not in action.column_names or not _columns_equal(action[name], force[name]):
            raise ValueError(f'force/action source mismatch: {name}')
    result = action
    for name in force.column_names:
        if not name.startswith('force_'):
            continue
        field = force.schema.field(name)
        if name in result.column_names:
            result = result.set_column(result.schema.get_field_index(name), field, force[name])
        else:
            result = result.append_column(field, force[name])
    metadata = dict(action.schema.metadata or {})
    for key, value in (force.schema.metadata or {}).items():
        if key.startswith(b'twm.force'):
            metadata[key] = value
    return result.replace_schema_metadata(metadata)


def merge_force_tree(root: Path, force_root: Path, force_revision: str) -> dict:
    path = root / 'action_publication_manifest.json'
    manifest = json.loads(path.read_text())
    records = {entry['remote_path']: entry for entry in manifest['episodes']}
    sources = sorted(force_root.glob('*/meta/*/*.parquet'))
    if len(sources) != 36:
        raise ValueError(f'expected36 force episodes, found {len(sources)}')
    receipt = json.loads((force_root / 'force_commit.json').read_text())
    if receipt.get('revision') != force_revision:
        raise ValueError('force export receipt revision mismatch')
    expected_hashes = receipt.get('parquet_sha256', {})
    if len(expected_hashes) != 36:
        raise ValueError('force receipt must bind36 uploaded parquets')
    for source in sources:
        if expected_hashes.get(str(source.relative_to(force_root))) != _sha256(source):
            raise ValueError(f'force export differs from uploaded content: {source}')
    for source in sources:
        relative = Path('old_data') / source.relative_to(force_root)
        if str(relative) not in records:
            raise ValueError(f'missing staged action: {relative}')
        dest = root / relative
        merged = merge_force_table(pq.read_table(dest), pq.read_table(source))
        _atomic_parquet(merged, dest)
        records[str(relative)]['output_sha256'] = _sha256(dest)
    manifest['force_revision'] = force_revision
    manifest['force_merged_episodes'] = len(sources)
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def validate_publication_manifest(manifest: dict, expected_revision: str) -> None:
    if manifest.get('force_revision') != expected_revision:
        raise ValueError('force revision must equal the immediately preceding HF commit')
    if manifest.get('force_merged_episodes') != 36:
        raise ValueError('publication requires36 merged force episodes')


def publish(root: Path, expected_revision: str, repo: str = 'yxma/React'):
    from huggingface_hub import HfApi, CommitOperationAdd
    api = HfApi()
    actual = api.repo_info(repo, repo_type='dataset').sha
    if actual != expected_revision:
        raise ValueError(f'remote changed: expected {expected_revision}, got {actual}')
    manifest = json.loads((root / 'action_publication_manifest.json').read_text())
    validate_publication_manifest(manifest, expected_revision)
    for entry in manifest['episodes']:
        if _sha256(root / entry['remote_path']) != entry['output_sha256']:
            raise ValueError(f'staged parquet changed: {entry["remote_path"]}')
    files = sorted(p for p in root.rglob('*') if p.is_file())
    top_files = {'action_publication_manifest.json', 'action_sample25.json',
                 'action_sample25.html', 'action_review.html',
                 'ACTION_REPAIR_20260918.md', 'sample25_verification.json'}
    directories = {'data', 'old_data', 'action_repair_tools', 'repair_audit'}
    suffixes = {'.parquet', '.npz', '.json', '.html', '.md', '.py', '.txt', '.toml'}
    for file in files:
        relative = file.relative_to(root)
        if (file.suffix not in suffixes or
                (len(relative.parts) == 1 and relative.name not in top_files) or
                (len(relative.parts) > 1 and relative.parts[0] not in directories)):
            raise ValueError(f'unexpected publication artifact: {relative}')
    operations = [CommitOperationAdd(path_in_repo=str(p.relative_to(root)),
                                     path_or_fileobj=str(p)) for p in files]
    result = api.create_commit(repo, repo_type='dataset', operations=operations,
        commit_message='Repair mocap actions across all tasks, old_data and validation; retain V8 force',
        parent_commit=expected_revision, num_threads=8)
    print(result, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--force-root', type=Path)
    parser.add_argument('--parent', required=True)
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    if args.force_root:
        merge_force_tree(args.root, args.force_root, args.parent)
    if args.publish:
        publish(args.root, args.parent)


if __name__ == '__main__':
    main()
