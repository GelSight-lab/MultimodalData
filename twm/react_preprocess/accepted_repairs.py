"""Apply explicit operator policy without altering raw or measured force data."""
from __future__ import annotations

from copy import deepcopy
import json
import numpy as np
import pyarrow as pa
from scipy.spatial.transform import Rotation as R

from .mocap_branch import find_branch_candidates
from .mocap_pattern_recovery import recover_pattern
from .repaired_actions import native_actions, actions_fps15


POLICY = 'operator_short33_and_supported_return_branches_v1'


def _array(table, name, dtype=None):
    return np.asarray(table[name].to_pylist(), dtype=dtype)


def _put(table, name, values):
    if isinstance(values, pa.Array):
        arr = values
    else:
        values = np.asarray(values)
        arr = pa.array(values.tolist() if values.ndim > 1 else values)
    if name in table.column_names:
        field = table.schema.field(name)
        arr = arr.cast(field.type)
        return table.set_column(table.schema.get_field_index(name), field, arr)
    return table.append_column(name, arr)


def _finite_pose(pose):
    return np.isfinite(pose).all(axis=1) & (np.linalg.norm(pose[:, 3:], axis=1) > 1e-12)


def transport_force_target(raw, repaired, target, penetration_mm):
    """Carry the verified existing sensor-local command through pose repair.

    This preserves each export's actual pressing-axis convention. Returns
    NaNs/False for targets inconsistent with the raw pose or penetration.
    """
    raw, repaired, target = [np.asarray(x, float) for x in (raw, repaired, target)]
    pen = np.asarray(penetration_mm, float)
    valid = _finite_pose(raw) & _finite_pose(repaired) & _finite_pose(target)
    valid &= np.isfinite(pen) & (pen >= 0)
    out = np.full(repaired.shape, np.nan)
    rows = np.flatnonzero(valid)
    if len(rows):
        rr, tr = R.from_quat(raw[rows, 3:]), R.from_quat(target[rows, 3:])
        displacement = target[rows, :3]-raw[rows, :3]
        consistent = np.degrees((rr.inv()*tr).magnitude()) < .01
        consistent &= np.abs(np.linalg.norm(displacement, axis=1)*1000-pen[rows]) < .02
        valid[rows[~consistent]] = False
    rows = np.flatnonzero(valid)
    if len(rows):
        rr, nr = R.from_quat(raw[rows, 3:]), R.from_quat(repaired[rows, 3:])
        local = rr.inv().apply(target[rows, :3]-raw[rows, :3])
        out[rows] = repaired[rows]
        out[rows, :3] += nr.apply(local)
        zero = rows[pen[rows] == 0]
        out[zero] = repaired[zero]
    return out, valid


def accept_table(table, events, *, recover=True):
    """Return augmented table, native/15fps sidecars, and updated event records."""
    n = table.num_rows
    out = table
    events = deepcopy(events)
    sidecars, poses, force_valid = {}, {}, {}
    for side in ('left', 'right'):
        raw = _array(table, f'sensor_{side}_pose', float)
        pose = _array(table, f'sensor_{side}_pose_repaired', float)
        valid = _array(table, f'pose_{side}_valid', bool)
        repaired = _array(table, f'pose_{side}_repaired', bool)
        conf = _array(table, f'pose_{side}_repair_confidence', np.uint8)
        ids = _array(table, f'pose_{side}_repair_event_id', object)
        methods = _array(table, f'pose_{side}_repair_method', object)
        approved = np.zeros(n, bool)
        blocked = np.zeros(n, bool)
        own = [e for e in events if e['side'] == side]
        gaps = [(e['local_start_row'], e['local_end_row']-e['local_start_row']+1)
                for e in own if 'gap' in e.get('kind', '')]
        refs = find_branch_candidates(raw, gaps) if recover and any(e['confidence'] == 'LOW' for e in own) else []
        for e in own:
            s, t = e['local_start_row'], e['local_end_row']
            rows = slice(s, t+1)
            duration = e['end']-e['start']+1
            accepted = e['confidence'] == 'HIGH' or (e['confidence'] == 'MEDIUM' and duration <= 33)
            if e['confidence'] == 'LOW' and recover:
                # Never repair a truncated source interval as if it were whole.
                recovery = recover_pattern(raw, s, t, refs) if t-s+1 == duration else None
                if recovery is not None:
                    fixed, evidence = recovery
                    changed = np.any(np.abs(fixed-raw[rows]) > 1e-10, axis=1)
                    pose[rows] = fixed
                    repaired[rows] = changed
                    conf[rows] = 2
                    ids[rows] = e['event_id']
                    methods[rows] = evidence['method']
                    e['original_confidence'] = e['confidence']
                    e['original_method'] = e['method']
                    e['original_evidence'] = e.get('evidence', {})
                    e['confidence'] = 'MEDIUM'
                    e['method'] = evidence['method']
                    e['evidence'] = evidence
                    offset = e['start']-s
                    e['replaced_frames'] = (np.flatnonzero(changed)+s+offset).tolist()
                    accepted = True
            accepted = bool(accepted and _finite_pose(pose[rows]).all())
            e['policy_accepted'] = accepted
            e['review_status'] = 'ACCEPTED_BY_OPERATOR_POLICY' if accepted else 'SKIP'
            e['acceptance_policy'] = POLICY
            e['independently_verified'] = False
            if accepted:
                valid[rows] = True
                approved[rows] = True
            else:
                blocked[rows] = True
        valid &= _finite_pose(pose) & ~blocked
        approved &= valid
        native = native_actions(pose, valid, repaired, ids)
        sidecars[side] = {'native': native, 'fps15': actions_fps15(pose, native)}
        poses[side] = pose
        overlap = np.zeros(n, bool)
        long = np.zeros(n, bool)
        for e in own:
            sl = slice(max(0, e['local_start_row']-1), min(n-1, e['local_end_row']+1))
            overlap[sl] = True
            if e['end']-e['start']+1 > 60:
                long[sl] = True
        av = np.r_[native.valid, False]
        ac = np.r_[np.maximum(conf[:-1], conf[1:]), np.uint8(0)]
        for name, value in {
            f'sensor_{side}_pose_repaired': pose,
            f'pose_{side}_valid': valid,
            f'pose_{side}_repaired': repaired,
            f'pose_{side}_repair_confidence': conf,
            f'pose_{side}_repair_event_id': ids,
            f'pose_{side}_repair_method': methods,
            f'pose_{side}_policy_accepted': approved,
            f'action_valid_{side}': av,
            f'action_repaired_{side}': np.r_[native.repaired, False],
            f'action_repair_confidence_{side}': ac,
            f'action_lost_track_{side}': ~av & overlap,
            f'action_long_lost_track_{side}': ~av & long,
        }.items():
            out = _put(out, name, value)
        fn = f'force_{side}_target_pose'
        pen = f'force_{side}_penetration_mm'
        force = f'force_{side}_normal_n'
        target, fv = np.full((n, 7), np.nan), np.zeros(n, bool)
        if all(k in table.column_names for k in (fn, pen, force)):
            target, fv = transport_force_target(raw, pose, _array(table, fn, float), _array(table, pen, float))
            f = _array(table, force, float)
            fv &= np.isfinite(f) & (f >= 0)
            fv &= (f != 0) | (_array(table, pen, float) == 0)
        fv &= valid
        target[~fv] = np.nan
        out = _put(out, f'force_{side}_target_pose_repaired', target)
        out = _put(out, f'force_{side}_target_valid', fv)
        force_valid[side] = fv
    action = np.c_[poses['left'], poses['right']]
    action = np.vstack([action[1:], action[-1:]]).astype(np.float32)
    out = _put(out, 'action', action)
    av = _array(out, 'action_valid_left', bool) & _array(out, 'action_valid_right', bool)
    fv = force_valid['left'] & force_valid['right']
    out = _put(out, 'action_valid', av)
    out = _put(out, 'action_force_valid', av & fv & np.r_[fv[1:], False])
    metadata = dict(out.schema.metadata or {})
    metadata[b'twm.action_acceptance'] = json.dumps({
        'policy': POLICY, 'max_interpolation_frames': 33, 'raw_columns_preserved': True,
        'action': 'next-frame repaired left7+right7; terminal invalid',
        'force_target_repaired': 'same-row repaired pose with verified original sensor-local target displacement',
        'action_force_valid': 'both hand action transitions AND both same-row/next-row force target validity',
        'force_version': 'unchanged; see original force metadata; absent force is unavailable',
    }).encode()
    return out.replace_schema_metadata(metadata), sidecars, events
