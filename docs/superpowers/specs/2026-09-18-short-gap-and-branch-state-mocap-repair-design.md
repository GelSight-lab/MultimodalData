# Short-gap and Branch-state Mocap Repair Design

## Goal

Reduce unresolved OptiTrack lost-track events across motherboard, pushT, rope,
toy, old data, and validation from 510 toward 100--150 without making genuine
fast motions valid. Preserve raw poses and provenance; only benchmark-qualified
repairs become training-valid automatically.

## Evidence and root cause

The current candidate tree contains 510 unresolved events: 92 MEDIUM and 418
LOW. Of these, 433 fail because independent left/right velocity extrapolations
disagree, 75 because the proposed reconstruction is nonphysical, one lacks an
anchor, and one MEDIUM event has ambiguous branch evidence. There are 416
events lasting only one to five frames.

The velocity-disagreement gate is the dominant false-negative mechanism. A
held-out masking experiment over 1,000 clean intervals per task showed that
endpoint-linear translation plus quaternion SLERP for one-to-five-frame gaps
meets the existing automatic-repair gates on every task: pose translation p95
5.9--7.3 mm, pose rotation p95 1.9--2.6 degrees, native-action translation
p95 3.5--4.3 mm, and native-action rotation p95 1.4--1.7 degrees.

## Stage 1: endpoint-constrained short-gap repair

For anomaly bouts of one to five native frames, reconstruct translation by
linear interpolation between the immediate finite anchors and orientation by
shortest-path quaternion SLERP. Do not require the two context velocities to
agree.

An interval is HIGH only when:

- both immediate anchors and the full 15-frame contexts are finite;
- branch classification is unambiguous;
- at least one anomalous observation is replaced;
- the reconstructed transition sequence stays within 50 mm and 30 degrees per
  native step;
- the task-specific clean-data masking benchmark passes the existing pose and
  action p95 gates.

Otherwise it remains MEDIUM or LOW. Current data indicates that this can
recover approximately 318 events, leaving about 192 unresolved.

## Stage 2: persistent wrong-branch decoding

Large one-frame boundary jumps can mark entry to or exit from a wrong but
internally smooth OptiTrack symmetry branch. The existing ten-frame grouping
window splits long wrong-branch occupancy into separate boundary events.

For each task, rigid body, and recording epoch:

1. Collect relative SE(3) transforms at high-angle discontinuities.
2. Cluster mutually inverse transforms to identify repeatable marker-symmetry
   branches. Require support from multiple transitions or a reviewed seed.
3. Run a two-state sequence decoder over each episode. Its observation cost
   measures local translational/angular acceleration after applying either the
   identity or learned inverse branch transform. Its transition cost prevents
   frame-level chatter while still allowing the observed jump-out/jump-back
   pattern.
4. Merge paired boundaries into one persistent wrong-branch interval and map
   its observations back through the learned inverse transform.
5. Reject promotion unless both corrected boundaries are continuous, every
   native step is physical, the transform cluster is tight, and held-out
   synthetic branch-switch tests pass the current pose/action error gates.

This stage targets another 40--80 recoveries. It must not interpolate long
true dropouts; it corrects a repeatable alternate pose solution only.

## Confidence and training policy

- HIGH: all method-specific gates and held-out task benchmark pass. Valid for
  action_full and action_causal training.
- MEDIUM: plausible reconstruction that lacks one independent check. Candidate
  values are stored but remain invalid pending human acceptance.
- LOW: missing anchors, nonphysical result, no stable branch transform, long
  unobserved gap, or possible genuine fast motion. Raw values remain and the
  event is invalid.

Raw `sensor_*_pose` columns are never overwritten. Repaired pose, method,
confidence, event ID, and validity remain separate fields. Native 30 fps and
15 fps actions are regenerated from accepted repaired poses; 15 fps validity
is the conjunction of both native transitions.

## Scope and publication

Run against a frozen inventory covering all four tasks, both old-data tasks,
and all nine validation episodes. Repair unsegmented source episodes first,
then propagate by source-frame identity into segmented release parquets. Do
not infer segment offsets from filenames alone.

Publish a manifest with input hashes and counts by task, confidence, method,
duration, repaired frames, recovered native/15-fps actions, and unresolved
events. Render every remaining LOW event longer than 60 frames and a
deterministic sample of shorter LOW/MEDIUM events for review.

## Verification and stopping rule

Tests cover endpoint interpolation, quaternion sign invariance, episode
boundaries, duplicate seeds, branch-transform inversion, decoder merging,
segment propagation, and confidence/action validity. Before publication:

- rerun clean-trajectory masking for each task;
- inject both short flickers and persistent alternate branches;
- verify no benchmark metric exceeds the existing gates;
- verify raw columns and row counts are unchanged;
- verify every valid repaired action has complete provenance.

The numerical target is at most 150 unresolved events, but accuracy gates take
precedence. If more than 150 remain after both validated stages, they stay
invalid and are rendered for human review rather than promoted by weakening a
gate.
