"""Extend review invalidity across plausible persistent alternate branches."""
from __future__ import annotations

from dataclasses import asdict, replace
import json

import numpy as np

from .mocap_branch import find_branch_candidates
from .mocap_repair import AnomalyBout, Confidence, PoseRepairResult, RepairEvent


def apply_loss_evidence_gate(result: PoseRepairResult, raw: np.ndarray,
                             known_gaps=()) -> PoseRepairResult:
    """Require declared loss or invalid raw observations for automatic HIGH."""
    raw = np.asarray(raw, dtype=float)
    loss = ~np.isfinite(raw).all(axis=1) | (np.linalg.norm(raw[:, 3:], axis=1) <= 1e-12)
    for start, count in known_gaps:
        if int(count) > 0:
            loss[max(0, int(start)):max(0, min(len(raw), int(start)+int(count)))] = True
    confidence, valid = result.confidence.copy(), result.valid.copy()
    events = []
    for event in result.events:
        if event.confidence == Confidence.HIGH:
            rows = np.asarray(event.replaced_frames, dtype=int)
            supported = bool(len(rows) and loss[rows].all())
            evidence = {**event.evidence, 'independent_loss_evidence': supported,
                        'review_required': not supported}
            if not supported:
                confidence[event.bout.start:event.bout.end+1] = Confidence.MEDIUM
                valid[event.bout.start:event.bout.end+1] = False
                event = replace(event, confidence=Confidence.MEDIUM, evidence=evidence)
            else:
                event = replace(event, evidence=evidence)
        events.append(event)
    return replace(result, confidence=confidence, valid=valid, events=tuple(events))


def apply_review_branches(result: PoseRepairResult, raw: np.ndarray, side: str,
                          known_gaps=()) -> PoseRepairResult:
    """Merge non-HIGH boundary events and invalidate the complete proposal.

    Existing HIGH events take precedence. Proposals supply candidate values,
    never training permission. Any larger overlapping original event remains
    invalid in full; its original evidence is retained in the merged event.
    """
    pose, repaired = result.pose.copy(), result.repaired.copy()
    confidence, valid = result.confidence.copy(), result.valid.copy()
    event_ids, events = result.event_id.copy(), list(result.events)
    for proposal in find_branch_candidates(raw, known_gaps):
        overlapping = [e for e in events if e.bout.side == side and
                       e.bout.start <= proposal.end+1 and e.bout.end >= proposal.start-1]
        if not overlapping or any(e.confidence == Confidence.HIGH for e in overlapping):
            continue
        lo = min(proposal.start, *(e.bout.start for e in overlapping))
        hi = max(proposal.end, *(e.bout.end for e in overlapping))
        # A merged interval cannot overlap an event that retains a different
        # row owner. Expand to a fixed point before checking HIGH protection.
        while True:
            merged = [e for e in events if e.bout.side == side and e.bout.start <= hi and e.bout.end >= lo]
            new_lo = min(lo, *(e.bout.start for e in merged))
            new_hi = max(hi, *(e.bout.end for e in merged))
            if (new_lo, new_hi) == (lo, hi):
                break
            lo, hi = new_lo, new_hi
        if any(e.confidence == Confidence.HIGH for e in merged):
            continue
        eid = f'{side}:{lo}-{hi}:branch_review'
        rows = np.arange(proposal.start, proposal.end+1)
        pose[rows] = proposal.pose
        repaired[rows] = True
        confidence[lo:hi+1] = Confidence.MEDIUM
        valid[lo:hi+1] = False
        event_ids[lo:hi+1] = eid
        seeds = tuple(sorted({row for e in merged for row in e.bout.seed_frames}))
        bout = AnomalyBout(side, lo, hi, 'branch_review', seeds,
                           lo-1 if lo else None, hi+1 if hi+1 < len(raw) else None)
        evidence = {**proposal.evidence,
                    'duration_frames': hi-lo+1,
                    'proposal_start': proposal.start, 'proposal_end': proposal.end,
                    'requires_independent_branch_identity_evidence': True,
                    'training_valid': False,
                    'superseded_events_json': json.dumps([asdict(e) for e in merged],
                                                         default=lambda value: value.item())}
        replacement = RepairEvent(eid, bout, Confidence.MEDIUM, 'paired_body_branch_review',
                                  tuple(np.flatnonzero(repaired[lo:hi+1])+lo), evidence)
        removed_ids = {e.event_id for e in merged}
        events = [e for e in events if e.event_id not in removed_ids] + [replacement]
    return PoseRepairResult(pose, repaired, confidence, valid, event_ids,
                            tuple(sorted(events, key=lambda e: e.bout.start)))
