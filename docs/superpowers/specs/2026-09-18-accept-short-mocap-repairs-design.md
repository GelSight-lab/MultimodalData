# Accept short mocap repairs without breaking action/force alignment

## Approved repair policy

The operator accepted all 274 published MEDIUM-confidence event occurrences
whose original source interval contains at most 33 frames (1.1 seconds at
30 fps). Retain the existing single HIGH-confidence repair. Keep all 125 LOW
event intervals invalid. Preserve confidence classifications; policy acceptance
is not independent verification of tracking loss or recovered motion.

The source is the published yxma/React revision
`5b1fa8596be2f73348c19c51bb2526281f8f61ab`, subject to a fresh remote-head check.
Do not overwrite a newer remote revision silently.

## Data contract

- Preserve every row, timestamp, frame identifier, raw sensor pose, camera and
  tactile asset, existing force value, and force provenance column.
- Use the existing repaired pose candidates, not a new fitting algorithm.
- Accept only finite, nonzero-quaternion candidate poses within approved
  intervals. LOW exclusions take precedence over acceptance in any overlap.
- Recompute native and 15 fps action validity using existing action builders.
  Retain physical step limits, terminal invalidity, segment boundaries, and
  invalidity of transitions touching excluded poses. A 15 fps transition is
  valid only if both underlying native transitions are valid.
- Add explicit policy-acceptance provenance and a joint two-hand action mask.
  Record why each excluded event is skipped. Do not delete rows and concatenate
  across gaps. Training windows must not bridge skipped intervals.

## Force compatibility proposal for operator review

Existing `force_{side}_target_pose` means a controller target derived from the
raw `sensor_{side}_pose`. Changing that meaning silently would break existing
consumers. Preserve it and add `force_{side}_target_pose_repaired` where an
existing force target and its derivation can be verified. This new target uses
the repaired pose, the same row's existing force/penetration, and the verified
existing sensor-local pressing-direction convention and stiffness. It copies
the repaired quaternion and equals the repaired pose at zero force.

Do not assume one pressing axis for every scope: the published old pushT
recovery declares body -Y; other exports declare calibrated gel axes. Verify
the applicable convention against existing raw targets before deriving new
targets. Unknown or contradictory provenance must be reported and masked for
the repaired force-target use case, never guessed. Do not fabricate absent
force columns or imply all force estimates are V8.

Provide clearly documented same-row force-target validity and a separate
joint action/force training mask. Define each temporal index explicitly:
`action[t]` targets repaired pose `t+1`; a force target stored at row `t` belongs
to pose/force `t`, so a next-pose force-informed target must select row `t+1`.
The joint mask must check all rows and both hands needed by the documented
training example. Missing force is unavailable, not a zero-force observation.

Alternatives considered: overwriting existing force targets would change their
raw-pose contract; retaining only raw targets would leave no aligned repaired
force-target option. Additive repaired targets preserve both interpretations.

## Edge-proximity audit

Use the published Z-up calibration and canonical preview projection. Define
an image-edge band as the outer 10% of a native 640 x 480 image. Count an event
if its affected sensor's raw projected gel center enters the band in any of
three cameras during the event. Report off-image centers separately. Also
check immediately adjacent poses, since corrupted poses can distort the
projection itself. This is a geometric association, not causal evidence of
OptiTrack camera failure; these RGB views are not necessarily tracker views.

Read-only audit: 49/125 LOW events have an in-image edge hit; 58/125 have an
edge or off-image hit. Adjacent poses give 54/125 edge hits and 61/125 edge or
off-image hits. All four LOW examples in the selected 25 have an edge hit.

## Verification and publication

Stage in a new output directory. Require focused synthetic tests covering
33-frame acceptance, 34-frame refusal, LOW overlap precedence, segment-edge
events, invalid quaternions, transition masks, and 15 fps gap propagation.
Test raw preservation, force-target reconstruction, zero-force identity, and
time-index semantics. Recompute and inspect data-level counts across all 218
parquets and sidecars; compare every preserved column and its metadata.

Run a documented loader smoke test for ordinary, repaired, skipped, missing
force, and segment-boundary examples. Validate NPZ row mappings and shapes.
Publish data and dependent manifests/docs together in one parent-guarded HF
commit only after validation. Read back changed remote artifacts, compare
hashes, and repeat loader checks at the new pinned revision. Retain the prior
revision for rollback. Report coverage limitations separately from structural
compatibility; passing checks is not a guarantee of physical ground truth.

No main-data force recalibration or force-version upgrade is included in this
repair-policy update. A later force update must run the same alignment and
derived-target consistency checks.

## Approved extension: examples 24 and 25

The operator subsequently approved bounded, repeated body-offset correction
and requested the analogous pattern in example 25. Add a conservative recovery
pass over LOW events, retaining original confidence and rejection evidence.
Repeated transforms require two matching paired-return proposals from the same
recording and side, not unrelated tasks or bodies. Correct only clearly
separated branch states within 33 frames, with unchanged outer anchors and a
maximum corrected angular step of 3 degrees for this repeated-flicker path.

For mixed longer events, pair the ordered large jump boundaries into separate
out-and-return stretches; do not interpolate the full parent interval. Each
stretch must be at most 33 frames. A stretch of at most five frames may use the
existing endpoint interpolation and same-branch anchor checks; longer stretches
must pass paired body-transform checks. Allow at most 6 mm disagreement between
the two body-translation estimates (previous cutoff 5 mm), while retaining the
3-degree transform rotation check and the original boundary/physical checks.
Require both outer contexts, compatible clean-branch anchors throughout, finite
observations, and physical reconstructed transitions. Missing data, holds, odd
unpaired jumps, incompatible branches, or incomplete recovery remain masked.
No correction is automatic ground truth; provenance records operator-approved
pattern policy and the original LOW classification. Do not apply this rule to
unflagged trajectories. Re-render examples 24 and 25 for review.
