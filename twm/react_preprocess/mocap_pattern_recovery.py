"""Operator-approved recovery of flagged, observable return-branch patterns.

Not a loss detector: callers supply an already flagged interval. A correction
remains a hypothesis, never independently verified ground truth.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

from .mocap_branch import BranchConfig, find_branch_candidates


def steps(pose):
    r = R.from_quat(pose[:, 3:])
    return (np.degrees((r[:-1].inv()*r[1:]).magnitude()),
            np.linalg.norm(np.diff(pose[:, :3], axis=0), axis=1)*1000)


def recover_pattern(raw, start, end, references=()):
    """Return (interval poses, evidence) or None; never mutate the input."""
    raw = np.asarray(raw, float)
    if start < 5 or end+6 > len(raw) or start > end:
        return None
    context = raw[start-5:end+6]
    if not np.isfinite(context).all() or np.any(np.linalg.norm(context[:, 3:], axis=1) < 1e-12):
        return None
    # Repeated identical poses are held observations, not measured deltas.
    held = np.all(np.diff(raw[start:end+1], axis=0) == 0, axis=1)
    if len(held) >= 3 and np.any(np.convolve(held.astype(int), np.ones(3, int), 'valid') == 3):
        return None
    for outer in (raw[start-5:start], raw[end+1:end+6]):
        rot, trans = steps(outer)
        if rot.max() > 12 or trans.max() > 50:
            return None
    observed = raw[start:end+1]
    n = len(observed)
    pred = Slerp([start-1, end+1], R.from_quat(raw[[start-1, end+1], 3:]))(np.arange(start, end+1))
    rr = R.from_quat(observed[:, 3:])
    alternatives = []
    if n <= 33:
        for ref in references:
            if ref.evidence.get('matching_return_pairs', 0) < 2:
                continue
            # An interval cannot be its own independent calibration support.
            if ref.start <= end and ref.end >= start:
                continue
            cr = rr*R.from_quat(ref.branch_rotation_xyzw).inv()
            err_raw = np.degrees((pred.inv()*rr).magnitude())
            err_fixed = np.degrees((pred.inv()*cr).magnitude())
            changed = err_fixed+10 < err_raw
            if not changed.any():
                continue
            fixed = observed.copy()
            fixed[changed, :3] -= cr.apply(ref.branch_translation_m)[changed]
            fixed[changed, 3:] = cr.as_quat()[changed]
            rot, trans = steps(np.vstack([raw[start-1], fixed, raw[end+1]]))
            if rot.max() <= 3 and trans.max() <= 50:
                alternatives.append((fixed, {
                    'method': 'repeated_body_branch_flicker',
                    'corrected_frames': int(changed.sum()),
                    'reference_start': ref.start, 'reference_end': ref.end,
                    'matching_return_pairs': ref.evidence['matching_return_pairs'],
                    'max_step_rotation_deg': float(rot.max()),
                    'max_step_translation_mm': float(trans.max()),
                }))
    if alternatives:
        return min(alternatives, key=lambda x: x[1]['max_step_rotation_deg'])

    # Mixed gap labels may cover observed clean motion between two returns.
    # Recover all bad stretches or leave the whole parent interval masked.
    rot, _ = steps(raw[start-1:end+2])
    boundaries = np.flatnonzero(rot > 30)+start
    if len(boundaries) < 2 or len(boundaries) % 2:
        return None
    if boundaries[0] != start or boundaries[-1] != end+1:
        return None
    fixed = observed.copy()
    pieces = []
    cfg = BranchConfig(max_interval_frames=33, transform_translation_tolerance_mm=6)
    paired = {(c.start, c.end): c for c in find_branch_candidates(raw[start-5:end+6], config=cfg)}
    for s, stop in zip(boundaries[::2], boundaries[1::2]):
        e = int(stop-1); s = int(s)
        if e-s+1 > 33:
            return None
        # Every clean anchor must agree with the parent's clean branch.
        for row in (s-1, e+1):
            reference = Slerp([start-1, end+1], R.from_quat(raw[[start-1, end+1], 3:]))([row])
            if np.degrees((reference.inv()*R.from_quat(raw[row, 3:])).magnitude())[0] > 12:
                return None
        if e-s+1 <= 5:
            anchor_r = R.from_quat(raw[[s-1, e+1], 3:])
            if np.degrees((anchor_r[0].inv()*anchor_r[1]).magnitude()) > 20:
                return None
            alpha = np.arange(1, e-s+2)/(e-s+2)
            corrected = np.c_[raw[s-1, :3]+alpha[:, None]*(raw[e+1, :3]-raw[s-1, :3]),
                              Slerp([0, 1], anchor_r)(alpha).as_quat()]
            method = 'endpoint_se3'
        else:
            proposal = paired.get((s-start+5, e-start+5))
            if proposal is None:
                return None
            corrected = proposal.pose
            method = 'paired_body_branch'
        fixed[s-start:e-start+1] = corrected
        pieces.append({'start': s, 'end': e, 'method': method})
    rot, trans = steps(np.vstack([raw[start-1], fixed, raw[end+1]]))
    if rot.max() > 30 or trans.max() > 50:
        return None
    return fixed, {'method': 'split_return_branches', 'pieces': pieces,
                   'corrected_frames': sum(p['end']-p['start']+1 for p in pieces),
                   'max_step_rotation_deg': float(rot.max()),
                   'max_step_translation_mm': float(trans.max())}
