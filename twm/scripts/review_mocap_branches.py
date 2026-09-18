"""Write separate, never-training-valid branch hypotheses for human review."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from twm.react_preprocess.mocap_branch import find_branch_candidates
from twm.scripts.build_mocap_repair_candidates import _atomic_json


def write_branch_review(candidate_root, source_root, output_root, *, manifest_digest):
    candidate_root, source_root, output_root = map(Path, (candidate_root, source_root, output_root))
    for protected in (candidate_root.resolve(), source_root.resolve()):
        if output_root.resolve().is_relative_to(protected) or protected.is_relative_to(output_root.resolve()):
            raise ValueError('review output must be separate from source and primary candidates')
    proposals = []
    unique_events = set()
    gaps = {}
    for event_path in sorted(candidate_root.glob('*/repair_events/*/*.json')):
        record = json.loads(event_path.read_text())
        unresolved = [e for e in record['events'] if e['confidence'] != 'HIGH']
        if not unresolved:
            continue
        task = record['task']
        if task not in gaps:
            path = source_root/task/'pose_gaps.json'
            gaps[task] = json.loads(path.read_text()) if path.exists() else {}
        source_path = source_root/record['source_path']
        columns = [f'sensor_{side}_pose' for side in sorted({e['side'] for e in unresolved})]
        table = pq.read_table(source_path, columns=columns, use_threads=False)
        for side in sorted({e['side'] for e in unresolved}):
            raw = np.asarray(table[f'sensor_{side}_pose'].to_pylist(), dtype=float)
            known = gaps[task].get(record['date']+'/'+record['episode'], {}).get(side, [])
            for proposal in find_branch_candidates(raw, known):
                events = [e for e in unresolved if e['side'] == side and
                          e['start'] <= proposal.end+1 and e['end'] >= proposal.start-1]
                if not events:
                    continue
                path = Path(task)/record['date']/f"{record['episode']}_{side}_{proposal.start}_{proposal.end}.json"
                overlap_ids = [e['event_id'] for e in events]
                unique_events.update((str(event_path), eid) for eid in overlap_ids)
                payload = {
                    'schema_version': 1, 'manifest_digest': manifest_digest,
                    'source_path': record['source_path'], 'source_sha256': record.get('source_sha256'),
                    'task': task, 'date': record['date'], 'episode': record['episode'], 'side': side,
                    'start': proposal.start, 'end': proposal.end,
                    'confidence': 'MEDIUM', 'training_valid': False,
                    'requires_independent_branch_identity_evidence': True,
                    'overlapping_event_ids': overlap_ids, 'evidence': proposal.evidence,
                    'branch_rotation_xyzw': proposal.branch_rotation_xyzw.tolist(),
                    'branch_translation_m': proposal.branch_translation_m.tolist(),
                    'proposed_pose': proposal.pose.tolist(),
                }
                _atomic_json(output_root/path, payload)
                proposals.append({'path': str(path), 'task': task, 'side': side,
                                  'start': proposal.start, 'end': proposal.end,
                                  'matching_return_pairs': proposal.evidence['matching_return_pairs']})
    report = {'schema_version': 1, 'manifest_digest': manifest_digest,
              'proposal_count': len(proposals), 'overlapping_unresolved_events': len(unique_events),
              'training_valid': False, 'requires_independent_branch_identity_evidence': True,
              'proposals': proposals}
    _atomic_json(output_root/'index.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.candidate_root/'input_manifest.json').read_text())
    report = write_branch_review(args.candidate_root, manifest['source_root'], args.output_root,
                                manifest_digest=manifest['digest'])
    print(json.dumps({k:v for k,v in report.items() if k != 'proposals'}), flush=True)


if __name__ == '__main__':
    main()
