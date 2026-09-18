"""Benchmark body-frame branch correction on frozen clean pose fragments."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from twm.react_preprocess.mocap_branch import find_branch_candidates
from twm.react_preprocess.mocap_candidate import load_manifest
from twm.scripts.benchmark_mocap_repair import (
    _candidate_order, _native_errors, _rotation_angle_deg, _select_disjoint, _summary,
)
from twm.scripts.build_mocap_repair_candidates import _atomic_json, _clean_fragments


def _inject(pose, start, length):
    observed = pose.copy()
    rotations = Rotation.from_quat(pose[start:start+length, 3:])
    observed[start:start+length, :3] += rotations.apply([0.029, 0, 0])
    observed[start:start+length, 3:] = (rotations*Rotation.from_euler('yx', [100, 8], degrees=True)).as_quat()
    return observed


def _negative_controls():
    pose = np.zeros((180, 7)); pose[:, 6] = 1
    fast = pose.copy()
    fast[:, 3:] = Rotation.from_euler('z', 35*np.arange(len(fast)), degrees=True).as_quat()
    out_back = pose.copy()
    angle = np.r_[np.zeros(30), np.linspace(0, 100, 60), np.linspace(100, 0, 60), np.zeros(30)]
    out_back[:, 3:] = Rotation.from_euler('z', angle, degrees=True).as_quat()
    abrupt = find_branch_candidates(_inject(pose, 40, 60))
    repeated = find_branch_candidates(_inject(_inject(pose, 20, 30), 100, 30))
    return {
        'continuous_fast_turn_candidates': len(find_branch_candidates(fast)),
        'continuous_out_back_candidates': len(find_branch_candidates(out_back)),
        'abrupt_out_back_candidates': len(abrupt),
        'abrupt_out_back_automatic_admissions': sum(c.confidence == 'HIGH' for c in abrupt),
        'repeated_abrupt_out_back_candidates': len(repeated),
        'repeated_abrupt_out_back_repeat_supported_candidates': sum(c.evidence['matching_return_pairs'] >= 2 for c in repeated),
        'caveat': 'Repeated abrupt genuine motion is observationally indistinguishable from a tracker branch; repetition alone is not independent ground truth.',
    }


def benchmark_branch_task(clean_episodes, *, task, max_intervals=200, seed=17):
    episodes = [np.asarray(p, dtype=float) for p in clean_episodes]
    lengths = (6, 20, 60, 120)
    selected = _select_disjoint(_candidate_order(
        episodes, lengths, np.random.default_rng(seed), pool_size=max(1000, max_intervals*50)), max_intervals)
    names = ('translation_mm', 'orientation_deg', 'native_translation_mm', 'native_rotation_deg')
    values = {name: [] for name in names}
    all_values = {name: [] for name in names}
    by_length = {length: {name: [] for name in names} for length in lengths}
    counts = {length: 0 for length in lengths}
    for interval in selected:
        truth = episodes[interval.episode][interval.start-15:interval.start+interval.length+15]
        observed = _inject(truth, 15, interval.length)
        candidates = find_branch_candidates(observed)
        matches = [c for c in candidates if c.start == 15 and c.end == 15+interval.length-1]
        result = observed.copy()
        if matches:
            result[15:15+interval.length] = matches[0].pose
            counts[interval.length] += 1
        rows = slice(15, 15+interval.length)
        trans = np.linalg.norm(result[rows, :3]-truth[rows, :3], axis=1)*1000
        rot = _rotation_angle_deg(result[rows, 3:], truth[rows, 3:])
        action_trans, action_rot = _native_errors(result, truth, np.arange(14, 15+interval.length))
        for name, error in zip(names, (trans, rot, action_trans, action_rot)):
            all_values[name].extend(error.tolist())
            if matches:
                values[name].extend(error.tolist())
                by_length[interval.length][name].extend(error.tolist())
    metrics = {name: asdict(_summary(v)) for name, v in values.items()}
    per_length = {str(length): {'qualifying_intervals': counts[length],
                              'metrics': {name: asdict(_summary(v)) for name, v in by_length[length].items()}}
                  for length in lengths}
    def passes(measured):
        caps = {'translation_mm': (2, 10), 'orientation_deg': (1, 5),
                'native_translation_mm': (2, 5), 'native_rotation_deg': (1, 3)}
        return all(measured[name]['count'] > 0 and measured[name]['median'] <= median and
                   measured[name]['p95'] <= p95 for name, (median, p95) in caps.items())
    negative = _negative_controls()
    passed = (sum(counts.values()) >= 100 and passes(metrics) and
              all(counts[length] >= 10 and passes(per_length[str(length)]['metrics']) for length in lengths) and
              negative['continuous_fast_turn_candidates'] == 0 and
              negative['continuous_out_back_candidates'] == 0)
    return {
        'schema_version': 1, 'task': task, 'seed': seed, 'method': 'paired_body_branch',
        'intervals': len(selected), 'qualifying_intervals': sum(counts.values()),
        'rejected_intervals': len(selected)-sum(counts.values()), 'anomaly_lengths': list(lengths),
        'metrics': metrics, 'metrics_all': {name: asdict(_summary(v)) for name, v in all_values.items()},
        'by_length': per_length, 'negative_controls': negative,
        'gate': {'high_confidence_enabled': False, 'reconstruction_accuracy_passed': bool(passed),
                 'validated_max_frames': max(lengths) if passed else 0,
                 'requires_independent_branch_identity_evidence': True},
        'scope': 'Reconstruction accuracy conditional on exact paired-boundary identification; does not validate anomaly classification or branch identity.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--max-intervals', type=int, default=200)
    parser.add_argument('--seed', type=int, default=17)
    args = parser.parse_args()
    manifest = load_manifest(args.candidate_root/'input_manifest.json')
    for task in manifest.tasks:
        fragments = _clean_fragments(manifest, task)
        report = benchmark_branch_task(fragments, task=task, max_intervals=args.max_intervals, seed=args.seed)
        report['manifest_digest'] = manifest.digest
        path = _atomic_json(args.output_root/f'{task}.json', report)
        print(task, report['qualifying_intervals'], report['gate'], path, flush=True)


if __name__ == '__main__':
    main()
