# Cross-Task Mocap Action Repair Design

**Date:** 2026-09-17

## Goal

Recover as many OptiTrack-corrupted hand actions as the evidence supports
across `motherboard`, `pushT`, `rope`, and `toy`, without rewriting genuine
large motions or hiding invented poses. High-confidence repairs become usable
training actions. Medium-confidence repairs remain masked until an operator
accepts their before/after preview. Low-confidence events remain invalid.

The first production-like run writes a candidate tree and review package. It
does not overwrite `release`, `release_zup`, `release_cut`, or existing action
files.

## Current Failure Modes

The existing repair handles short returning excursions with linear translation
and quaternion SLERP, and handles minority-branch A/B flicker in a local
window. Its main limitations are:

- a fixed 15-frame automatic repair limit, even when longer intervals have
  reliable anchors and little net motion;
- local minority support fails when the wrong branch occupies close to half or
  more of the window;
- a fixed return-search window can split a longer tracking-loss bout into
  isolated `persistent_branch` transitions;
- interpolating every frame in a bout discards good observations that briefly
  return to the correct branch;
- a raw `>30 degree` action threshold cannot distinguish real fast motion from
  a tracker branch switch.

On the current motherboard audit, 35 known long gaps contain 1,239 frames.
Thirty-one gaps are no longer than two seconds. A deliberately simple endpoint
and bidirectional-velocity check already identifies four gaps (96 frames) as
strong repair candidates; more may become reviewable with the robust model
below. This is evidence that the fixed 15-frame limit is too conservative, not
evidence that every short gap is safe to fill.

## Considered Approaches

### 1. Raise the linear/SLERP frame limit

Allow the current endpoint interpolation for up to 60 frames. This is easy and
would recover the four obvious motherboard gaps, but duration alone is not a
confidence measure. It can turn a genuine curved reach into a straight path and
cannot reliably select the correct samples in an A/B flicker.

### 2. Robust SE(3) trajectory reconstruction (selected)

Group discontinuities into complete tracking-loss bouts, fit translation and
orientation from clean context on both sides, iteratively classify observations
as correct-branch inliers or wrong-branch outliers, and replace only the
outliers. Confidence comes from independent forward/backward predictions,
reconstruction uncertainty, and physical continuity. This recovers both
intermittent flicker and fully missing runs while preserving good measured
frames.

### 3. Image-assisted pose reconstruction

Use optical flow or learned hand-pose estimation from the wrist and scene
cameras. This can help adjudicate rare ambiguous events, but it introduces a
new calibrated estimator and a much larger validation burden. It is reserved
as optional evidence in manual review and is not an automatic source of poses
in this implementation.

## Scope and Data Flow

The repair begins from each task's uncut 30 fps episode parquet under
`release/<task>/meta/<date>/episode_*.parquet`. It covers both hands and every
available date for the four declared pipeline tasks. Repair occurs before Z-up
conversion, segmentation, and action construction so all downstream trees can
be derived from one provenance-preserving source.

At the beginning of an audit, the scanner writes an input manifest containing
every parquet path, row count, size, and SHA-256 digest. Files created after
that snapshot, including episodes still being built, are not silently added to
the run; a later audit creates a new manifest. The first candidate output root
is `/media/yxma/Disk1/twm/review/mocap_repair_2026-09-17/candidate_release`.

For each episode and side:

1. Compute sign-insensitive quaternion transition angles, translation steps,
   angular/linear velocity, and acceleration.
2. Seed candidates from large discontinuities, known pose gaps, and existing
   pose-interpolation metadata.
3. Merge nearby seeds into a complete anomaly bout. A bout ends only after the
   stream remains compatible with one smooth trajectory for 10 consecutive
   native frames; individual `>30 degree` edges are not treated as independent
   events.
4. Establish clean context on both sides. Context frames containing another
   candidate, a known gap, non-finite pose, or existing low-confidence repair
   are excluded. High confidence requires 15 usable native frames on each
   side; less context limits the event to medium or low confidence.
5. Fit the robust SE(3) trajectory, classify observations, and calculate
   confidence evidence.
6. Write repaired poses, side-specific masks, evidence, and review artifacts
   into a candidate tree.
7. Rebuild native 30 fps actions and derived 15 fps actions from candidate
   poses and propagate validity and provenance.

The candidate writer is idempotent and refuses to write into a production root.
Frames outside a selected repair mask remain bit-identical to the source.

## Robust SE(3) Repair

### Translation

Fit robust linear velocities to clean context before and after the bout. If the
two velocities are statistically indistinguishable from zero, use endpoint
linear interpolation. Otherwise use a cubic Hermite curve constrained by the
two anchor positions and robust boundary velocities. Clamp neither the curve
nor the observations silently; excessive speed or acceleration lowers the
confidence tier.

### Orientation

Align quaternion signs locally, map rotations into the tangent space of the
left anchor, and fit the same boundary-constrained curve to rotation vectors.
Map the result back with the SO(3) exponential. Near singular or poorly
conditioned cases fall back to shortest-arc SLERP and cannot receive high
confidence unless the independent prediction checks still pass.

### Branch selection

Initialize a trajectory from the two clean contexts. For every observation in
the bout, calculate translation and rotation residuals to the predicted path.
Use robust, episode-local residual scales with global physical caps to classify
inliers. Refit on anchors plus inliers for at most five iterations and stop
early when the inlier mask no longer changes.

This treats an A/B flicker as a state-selection problem rather than a local
majority vote:

- observations close to the continuous branch remain measured and unchanged;
- observations on the alternate branch are replaced;
- a fully corrupt contiguous run is reconstructed only from the two clean
  contexts;
- two equally plausible continuous branches make the event low confidence.

## Confidence Tiers

Thresholds are calibrated from clean data by masking real intervals with the
observed task-specific duration distribution and reconstructing them. The
implementation uses clean-data error quantiles plus hard physical caps; it does
not tune arbitrary thresholds separately until the results look favorable.

### High confidence: automatically trainable

All of the following must hold:

- trustworthy clean context exists on both sides;
- forward prediction from the left and backward prediction from the right
  differ by at most 10 mm and 5 degrees at every reconstructed frame;
- endpoint displacement and fitted boundary velocities are mutually
  compatible;
- the reconstructed trajectory introduces no invalid quaternion, discontinuity,
  or velocity/acceleration outside the task's clean 99.9th percentile, and no
  reconstructed native step exceeds the hard caps of 50 mm translation or
  30 degrees rotation;
- the selected branch is unambiguous;
- clean-data masking experiments demonstrate that this event's duration and
  motion regime meet the high-confidence reconstruction error target.

A two-second duration is the initial ceiling for automatic use, not a reason by
itself to accept an event. Longer events may be reconstructed for review but
cannot be high confidence in the first rollout.

### Medium confidence: repaired candidate, operator approval required

The event has two usable anchors and one clearly preferred trajectory, but is
longer, has moderate forward/backward disagreement, or lies outside the
well-validated clean-data regime. It receives a repaired candidate and a
before/after preview, but its action transitions remain invalid until the
operator explicitly accepts the event.

### Low confidence: invalid

Examples include a missing anchor at an episode edge, a persistent branch with
no reliable return, multiple equally plausible trajectories, a very long gap,
or reconstructed dynamics that violate physical limits. Original poses remain
available for audit; no candidate replacement becomes trainable.

## Pose and Action Contract

Validity and repair provenance are side-specific. The candidate episode stores
at least:

- `pose_<side>_repaired`: frame was replaced by the repair;
- `pose_<side>_repair_confidence`: unsigned byte enum `0=none`, `1=low`,
  `2=medium`, `3=high`;
- `pose_<side>_valid`: measured or accepted repaired pose is usable;
- repair event ID, method, source interval, anchor interval, and evidence
  residuals in a sidecar manifest.

Native action `k -> k+1` is recomputed from the candidate poses. It stores:

- `action_<side>_repaired`: either endpoint was repaired;
- `action_<side>_valid`: both endpoint poses are valid and the recomputed
  transition passes finite and physical checks;
- the source repair event IDs when applicable.

An untouched frame outside every anomaly has confidence `none` and remains
valid. A replaced high-confidence frame is repaired and valid. A medium
candidate is repaired in the candidate pose column but remains invalid until
an exported decision accepts it. A low-confidence frame is not replaced and
is invalid. High-confidence repaired actions are valid for training while
remaining explicitly labelled. Medium-confidence actions are invalid until an
exported review decision accepts their event. Low-confidence actions remain
invalid.

The 15 fps action spanning two native transitions is valid only when both
native transitions are valid, and its repaired flag is their logical OR.

`action_full` drops a window if any required input action for either hand is
invalid. `action_causal` retains the window but masks invalid action targets.
Neither consumer is allowed to infer validity from rotation magnitude alone.

## Review Package

Every medium- and low-confidence bout receives a canonical preview using the
existing released scene, wrist, and tactile videos. High-confidence events are
also sampled for audit in the first rollout.

The page displays:

- synchronized raw and repaired pose overlays;
- rotation, translation, velocity, acceleration, and residual plots;
- clean anchors, retained measured frames, replaced frames, and confidence
  evidence;
- `keep raw`, `accept repair`, `invalidate`, and `unsure` decisions;
- deterministic event IDs and an exported decisions JSON file.

Decisions are replayable. Applying the same decisions twice produces identical
candidate poses, masks, and actions.

## Validation

### Synthetic regression cases

Cover one-frame and multi-frame returns, wrong-branch minority and majority
flicker, repeated A/B toggling, a fully corrupt run, a genuine continuous large
turn, a fast reach-and-return, a persistent branch, episode-edge loss, and
quaternion sign changes. Assert exact repair masks, unchanged good frames,
confidence tiers, and idempotence.

### Clean-data masking benchmark

For every task, select clean intervals covering its duration and motion
distribution. Mask them with the lengths and patterns observed in real anomaly
bouts, reconstruct them, and compare against the held-out truth. Report
translation, orientation, native-action, and 15 fps action errors by task,
duration, and confidence tier. High-confidence gates are chosen before the
production candidate run. On a disjoint held-out set, high-confidence repairs
must meet all four targets: median translation error at most 2 mm, 95th
percentile translation error at most 10 mm, median orientation error at most
1 degree, and 95th percentile orientation error at most 5 degrees. Recomputed
native-action errors must have 95th percentiles at most 5 mm translation and
3 degrees rotation. Synthetic genuine-large-motion cases must have zero
repaired frames. If a task has too few examples to measure these targets with
at least 100 masked intervals, that task may emit medium/low candidates but no
automatic high-confidence repair.

### Real-data audit

Run metadata-only classification over all uncut episodes, then generate the
candidate tree and review package. Inspect representative events from every
task, including real large rotations that must remain unchanged. Verify:

- no production file changed;
- every replaced frame belongs to a declared event;
- every untouched frame is bit-identical;
- all output quaternions are finite and unit length;
- repaired actions contain no unexplained discontinuity;
- all medium/low transitions are masked until accepted;
- rebuilding is deterministic and idempotent;
- every review clip exists and decodes.

The previously repaired motherboard 2026-09-11 episode 004 flicker remains a
regression fixture and must not reappear as an unresolved anomaly.

## Rollout

1. Implement and test the pure detector, trajectory fitter, confidence scorer,
   and action-validity propagation.
2. Calibrate confidence thresholds with the clean-data masking benchmark for
   all four tasks.
3. Run a metadata-only audit and report projected recovery counts.
4. Build a separate candidate tree and complete review package.
5. Apply operator decisions and rerun the action builders in the candidate
   tree.
6. Verify candidate outputs end to end.
7. Only after explicit operator approval, promote repaired metadata/actions
   into production-derived trees; retain the original raw recordings and the
   complete provenance manifest.

Promotion, upload, and deletion of any original data are separate explicit
operations and are not authorized by this design.
