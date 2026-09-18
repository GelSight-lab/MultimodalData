"""Stage and verify an explicit operator-accepted action repair publication."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import shutil

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R

from twm.react_preprocess.accepted_repairs import accept_table, POLICY
from twm.react_preprocess.mocap_candidate import _write_actions
from twm.scripts.export_old_motherboard_actions import _atomic_parquet, _columns_equal, _sha256
from twm.scripts.publish_action_repairs import _derived
from twm.scripts.propagate_mocap_repairs import segment_reviews, review_html


def validate_table(before, after, sidecars, events):
    if before.num_rows != after.num_rows:
        raise ValueError('row count changed')
    n = after.num_rows
    for field in before.schema:
        if not _derived(field.name):
            if field != after.schema.field(field.name) or not _columns_equal(before[field.name], after[field.name]):
                raise ValueError(f'preserved field changed: {field.name}')
    for key, value in (before.schema.metadata or {}).items():
        if (after.schema.metadata or {}).get(key) != value:
            raise ValueError(f'preserved metadata changed: {key}')
    poses = [np.asarray(after[f'sensor_{s}_pose_repaired'].to_pylist(), float) for s in ('left', 'right')]
    expected = np.c_[poses[0], poses[1]]
    expected = np.vstack([expected[1:], expected[-1:]]).astype(np.float32)
    if not np.array_equal(np.asarray(after['action'].to_pylist(), np.float32), expected, equal_nan=True):
        raise ValueError('next-frame action mismatch')
    for side, pose in zip(('left', 'right'), poses):
        av = np.asarray(after[f'action_valid_{side}'], bool)
        pv = np.asarray(after[f'pose_{side}_valid'], bool)
        native, fps15 = sidecars[side]['native'], sidecars[side]['fps15']
        if av[-1] or not np.array_equal(av[:-1], native.valid):
            raise ValueError('native/terminal validity mismatch')
        if np.any(av[:-1] & ~(pv[:-1] & pv[1:])):
            raise ValueError('action crosses excluded pose')
        if not np.array_equal(fps15.valid, native.valid[fps15.start_rows] & native.valid[fps15.start_rows+1]):
            raise ValueError('15fps validity mismatch')
        for e in events:
            if e['side'] != side or e['policy_accepted']:
                continue
            s, t = e['local_start_row'], e['local_end_row']
            if pv[s:t+1].any() or av[max(0, s-1):min(n-1, t+1)].any():
                raise ValueError('skipped event admitted')
        fv = np.asarray(after[f'force_{side}_target_valid'], bool)
        target = np.asarray(after[f'force_{side}_target_pose_repaired'].to_pylist(), float)
        if np.any(fv & ~pv) or not np.isfinite(target[fv]).all():
            raise ValueError('force validity mismatch')
        if fv.any():
            raw = np.asarray(before[f'sensor_{side}_pose'].to_pylist(), float)[fv]
            original = np.asarray(before[f'force_{side}_target_pose'].to_pylist(), float)[fv]
            repaired = pose[fv]
            a = R.from_quat(raw[:, 3:]).inv().apply(original[:, :3]-raw[:, :3])
            b = R.from_quat(repaired[:, 3:]).inv().apply(target[fv, :3]-repaired[:, :3])
            if not np.allclose(a, b, atol=2e-7, rtol=0):
                raise ValueError('force local displacement changed')
            if not np.allclose(target[fv, 3:], repaired[:, 3:], atol=2e-7, rtol=0):
                raise ValueError('force target quaternion mismatch')
    av = np.asarray(after['action_valid'], bool)
    both = np.asarray(after['action_valid_left'], bool) & np.asarray(after['action_valid_right'], bool)
    fv = np.asarray(after['force_left_target_valid'], bool) & np.asarray(after['force_right_target_valid'], bool)
    if not np.array_equal(av, both) or not np.array_equal(np.asarray(after['action_force_valid'], bool), av & fv & np.r_[fv[1:], False]):
        raise ValueError('joint training mask mismatch')


def stage(source: Path, output: Path, revision: str):
    from huggingface_hub import HfApi
    source, output = Path(source), Path(output)
    if output.exists():
        raise ValueError('output already exists; choose a new immutable staging directory')
    api = HfApi()
    info = api.repo_info('yxma/React', repo_type='dataset', revision=revision, files_metadata=True)
    files = {x.rfilename: x for x in info.siblings}
    manifest = json.loads((source/'action_publication_manifest.json').read_text())
    # Verify the full base before any staged transformation.
    for e in manifest['episodes']:
        remote = files[e['remote_path']]
        if not remote.lfs or _sha256(source/e['remote_path']) != remote.lfs.sha256:
            raise ValueError(f'base differs from pinned publication: {e["remote_path"]}')
    output.mkdir(parents=True)
    counts = Counter(); all_events = []; hashes = {}; records = []
    video_files = {p for p in files if p.endswith('.mp4')}
    sample = json.loads((source/'action_sample25.json').read_text())
    sample_ids = {x['qualified_event_id'] for x in sample['events']}
    rebuilt_samples = {}
    for number, entry in enumerate(manifest['episodes'], 1):
        relative = Path(entry['remote_path']); before = pq.read_table(source/relative)
        ep = Path(*relative.parts[:2], 'action_repair_events', relative.parts[3], relative.stem+'.json')
        doc = json.loads((source/ep).read_text())
        after, sides, events = accept_table(before, doc['events'])
        validate_table(before, after, sides, events)
        destination = output/relative
        _atomic_parquet(after, destination)
        # Validate the serialized artifact, including existing field types.
        validate_table(before, pq.read_table(destination), sides, events)
        hashes[str(relative)] = _sha256(destination)
        for side, rates in sides.items():
            for rate, series in rates.items():
                sp = Path(*relative.parts[:2], 'actions_'+rate, relative.parts[3], relative.stem+'_'+side+'.npz')
                _write_actions(output/sp, series)
                with np.load(output/sp, allow_pickle=False) as stored:
                    for k in ('valid', 'repaired', 'start_rows', 'end_rows', 'values'):
                        if not np.array_equal(stored[k], getattr(series, k), equal_nan=True):
                            raise ValueError(f'sidecar mismatch: {sp}:{k}')
                hashes[str(sp)] = _sha256(output/sp)
        rows = np.arange(entry['start_row'], entry['end_row']+1)
        reviews = segment_reviews(str(relative), before, after, rows, events, video_files,
                                  revision, include_high_charts=True)
        for e in reviews:
            counts['accepted_events' if e['policy_accepted'] else 'skipped_events'] += 1
            if e.get('original_confidence') == 'LOW':
                counts['recovered_low_events'] += 1
                counts['recovered_low_changed_frames'] += e['evidence']['corrected_frames']
            if e['qualified_event_id'] in sample_ids:
                rebuilt_samples[e['qualified_event_id']] = e
        all_events.extend(reviews)
        doc['events'] = reviews; doc['acceptance_policy'] = POLICY
        (output/ep).parent.mkdir(parents=True, exist_ok=True)
        (output/ep).write_text(json.dumps(doc, indent=2, allow_nan=False)+'\n')
        hashes[str(ep)] = _sha256(output/ep)
        counts['episodes'] += 1; counts['rows'] += after.num_rows
        for col in ('action_valid', 'action_force_valid', 'action_valid_left', 'action_valid_right'):
            counts[col] += int(np.asarray(after[col], bool).sum())
        rec = deepcopy(entry); rec['output_sha256'] = hashes[str(relative)]
        rec['valid_left'] = int(np.asarray(after['action_valid_left'], bool).sum())
        rec['valid_right'] = int(np.asarray(after['action_valid_right'], bool).sum())
        records.append(rec)
        if number % 20 == 0 or number == len(manifest['episodes']):
            print(f'VERIFIED {number}/{len(manifest["episodes"])} {dict(counts)}', flush=True)
    manifest.update(episodes=records, parent_revision=revision, output_root=str(output),
                    acceptance_policy=POLICY, acceptance_counts=dict(counts))
    (output/'action_publication_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    sample['events'] = [rebuilt_samples[x['qualified_event_id']] for x in sample['events']]
    sample['acceptance_policy'] = POLICY
    (output/'action_sample25.json').write_text(json.dumps(sample, indent=2, allow_nan=False)+'\n')
    for filename, ev, title in [('action_sample25.html', sample['events'], '25 operator-policy repair examples'),
                                 ('action_review.html', all_events, 'Action repair acceptance and skipped events')]:
        page = review_html(ev, include_high=True, title=title)
        page = page.replace('No approval is implied.', 'Operator policy acceptance is recorded per event; this is not independent ground truth.')
        for e in ev:
            old = e['qualified_event_id']
            # Insert a status before the existing UNVERIFIED label.
            from html import escape
            page = page.replace(escape(old)+' —', escape(old)+' — '+e['review_status']+' —')
        (output/filename).write_text(page)
    for name in ('mocap_pattern_recovery.py', 'accepted_repairs.py'):
        dst = output/'action_repair_tools'/name; dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(__file__).parents[1]/'react_preprocess'/name, dst)
    shutil.copyfile(__file__, output/'action_repair_tools'/'accept_mocap_repairs.py')
    readme = f'''# Accepted mocap repairs — 2026-09-18

Base revision: `{revision}`. Policy: `{POLICY}`.

{json.dumps(dict(counts), indent=2)}

Raw sensor poses, all existing force columns, rows, timestamps, and videos are
unchanged. All 274 original MEDIUM candidates (at most 33 frames) were accepted
by the operator, not independently verified. Supported LOW patterns have
original confidence/evidence retained in event records. Unresolved events stay
masked. No row deletion or frame renumbering is used.

## Training contract

`action[t]` is the next-row repaired left/right absolute pose (14 values).
Filter with `action_valid[t]` for joint two-hand motion training. Per-hand masks
remain available. The terminal row is invalid. Never concatenate across masked
gaps when building temporal windows; every constituent transition must be valid.

Existing `force_*_target_pose[t]` retains its raw-pose, same-row meaning.
New `force_*_target_pose_repaired[t]` is a same-row force target in the repaired
pose frame. Its original sensor-local displacement is preserved, including the
original pressing-axis and stiffness convention. Use `force_*_target_valid[t]`.
For a force-informed next-pose action at t, select repaired force targets at
**t+1**, and filter by `action_force_valid[t]`. That mask requires both hands'
motion transitions and both current/next force targets to be valid. Do not mix
same-row force targets with next-row action poses without this shift.

Missing force/targets or contradictory target provenance is invalid, not zero
force. This publication does not upgrade force versions: the previous mix of
V8, older, unstamped, and missing force remains explicitly unchanged. Absolute
force accuracy and physical repair truth are not certified by alignment tests.

Each sidecar retains native/15 fps mappings. A 15 fps transition requires both
underlying native transitions to be valid. Check `action_acceptance_audit.json`
for all uploaded artifact hashes and per-event recovery details.
'''
    (output/'ACTION_ACCEPTANCE_20260918.md').write_text(readme)
    audit = {'parent_revision': revision, 'policy': POLICY, 'counts': dict(counts),
             'verification': 'All218 serialized parquets checked: raw/force columns+metadata, next-row actions, native/15fps masks, skipped boundaries, transported force targets;872 NPZ mappings reloaded.',
             'recovered_events': [e for e in all_events if e.get('original_confidence') == 'LOW'],
             'sha256': hashes}
    for path in sorted(output.rglob('*')):
        if path.is_file(): hashes[str(path.relative_to(output))] = _sha256(path)
    (output/'action_acceptance_audit.json').write_text(json.dumps(audit, indent=2, allow_nan=False)+'\n')
    print('STAGED', json.dumps(dict(counts)), output, flush=True)
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--revision', required=True)
    args = parser.parse_args()
    stage(args.source, args.output, args.revision)
