# Short-gap and Branch-state Mocap Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Safely reduce unresolved OptiTrack lost-track events from 510 toward 100--150 by validating endpoint interpolation for 1--5-frame gaps and decoding persistent alternate tracking branches.

**Architecture:** Extend the pure pose-repair layer with a benchmark-gated short-gap method and a separate branch-transform sequence decoder. Keep detection, reconstruction, confidence, candidate publication, and segmented-release propagation separate so raw poses stay immutable and every repaired action remains attributable to one method and event.

**Tech Stack:** Python 3.9+, NumPy, SciPy Rotation/Slerp, PyArrow, pytest, JSON/NPZ manifests.

---

## File map

- Modify `twm/react_preprocess/mocap_repair.py`: endpoint SE(3) interpolation and method selection.
- Create `twm/react_preprocess/mocap_branch.py`: transform clustering and two-state decoding for persistent branches.
- Modify `twm/scripts/benchmark_mocap_repair.py`: benchmark metrics by method and duration.
- Modify `twm/react_preprocess/mocap_candidate.py`: carry method provenance and decoded intervals into candidate parquets/actions.
- Modify `twm/scripts/build_mocap_repair_candidates.py`: frozen multi-root inventory and summary counts.
- Create `twm/scripts/propagate_mocap_repairs.py`: propagate source repairs into segmented release, old_data, and validation by source-frame identity.
- Modify `twm/scripts/build_pose_review.py`: render all long LOW and deterministic shorter samples.
- Create/modify tests named below.

### Task 1: Endpoint interpolation for one-to-five-frame events

**Files:**
- Modify: `twm/react_preprocess/mocap_repair.py`
- Modify: `tests/test_mocap_repair.py`
- Modify: `tests/test_mocap_benchmark.py`

- [ ] **Step 1: Write failing tests for a direction change around a short flicker**

Add a clean curved SE(3) trajectory, replace rows 40--44 by the known wrong
branch, and assert that `repair_pose_stream` returns `endpoint_se3` with HIGH
confidence when `TaskGate(True, 5)` is supplied. Assert both immediate anchors
stay byte-identical and reconstructed native steps are at most 50 mm/30 degrees.

```python
result = repair_pose_stream(observed, "left", task_gate=TaskGate(True, 5))
event = result.events[0]
assert event.method == "endpoint_se3"
assert event.confidence == Confidence.HIGH
assert np.all(result.valid[40:45])
np.testing.assert_array_equal(result.pose[[39, 45]], observed[[39, 45]])
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `python -m pytest -q tests/test_mocap_repair.py -k endpoint_se3`

Expected: FAIL because the method does not exist and the current velocity
agreement gate returns MEDIUM/LOW.

- [ ] **Step 3: Implement the endpoint method**

Add a pure helper with the exact interface:

```python
def endpoint_se3(pose: np.ndarray, bout: AnomalyBout) -> np.ndarray:
    """Linear xyz plus shortest-path quaternion SLERP between immediate anchors."""
```

Select it before `_predict_bout` only for `1 <= len(rows) <= 5`. Require full
finite context, `branch_unambiguous`, a physical reconstructed sequence, at
least one replaced observation, and `gate.validated_max_frames >= len(rows)`.
Do not consult `prediction_*_max` for this method. Record endpoint displacement
and reconstructed step maxima in evidence.

- [ ] **Step 4: Run repair and benchmark tests**

Run:

```bash
python -m pytest -q tests/test_mocap_repair.py tests/test_mocap_benchmark.py tests/test_repaired_actions.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add twm/react_preprocess/mocap_repair.py tests/test_mocap_repair.py tests/test_mocap_benchmark.py
git commit -m "feat: repair short mocap gaps from endpoint constraints"
```

### Task 2: Benchmark-gate the short-gap method on each task

**Files:**
- Modify: `twm/scripts/benchmark_mocap_repair.py`
- Modify: `tests/test_mocap_benchmark.py`

- [ ] **Step 1: Write a failing method-specific benchmark test**

Assert the report exposes per-method interval counts and p95 metrics, and that
the gate enables endpoint repair only after at least 100 held-out intervals
pass all existing thresholds.

```python
assert report.methods["endpoint_se3"].intervals >= 100
assert report.methods["endpoint_se3"].metrics["native_rotation_deg"].p95 <= 3.0
assert report.gate.endpoint_max_frames == 5
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest -q tests/test_mocap_benchmark.py -k method`

Expected: FAIL because `methods` and `endpoint_max_frames` are absent.

- [ ] **Step 3: Add explicit method gates**

Extend `TaskGate` with `endpoint_max_frames: int = 0`. Extend benchmark output
with method-keyed interval/error summaries. The endpoint gate passes only when
there are at least 100 intervals and these limits all hold: pose translation
median <=2 mm and p95 <=10 mm; pose orientation median <=1 degree and p95 <=5
degrees; action translation p95 <=5 mm; action rotation p95 <=3 degrees.

- [ ] **Step 4: Run deterministic real-task benchmarks**

Run:

```bash
python -m twm.scripts.build_mocap_repair_candidates --output /media/yxma/Disk1/twm/review/mocap_repair_stage2_2026-09-18 benchmark --max-intervals 500 --seed 17
```

Expected: every task records `endpoint_max_frames=5`; otherwise leave that
task disabled rather than weakening a threshold.

- [ ] **Step 5: Commit**

```bash
git add twm/scripts/benchmark_mocap_repair.py twm/react_preprocess/mocap_repair.py tests/test_mocap_benchmark.py
git commit -m "feat: calibrate endpoint mocap repair per task"
```

### Task 3: Learn repeatable alternate-branch transforms

**Files:**
- Create: `twm/react_preprocess/mocap_branch.py`
- Create: `tests/test_mocap_branch.py`

- [ ] **Step 1: Write failing transform-clustering tests**

Generate a smooth pose sequence and replace two intervals by
`pose @ fixed_body_transform`. Assert inverse jump pairs form one cluster;
unrelated real 35-degree motion and a single unsupported jump do not.

```python
model = learn_branch_model(pose, discontinuities)
assert len(model.clusters) == 1
assert model.clusters[0].support >= 4
assert model.clusters[0].rotation_spread_deg <= 3.0
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest -q tests/test_mocap_branch.py -k cluster`

Expected: import failure because `mocap_branch` is absent.

- [ ] **Step 3: Implement transform representation and clustering**

Define immutable `BranchTransform` and `BranchModel` dataclasses. Compute local
relative transforms at discontinuity boundaries, pair approximately inverse
transforms, and cluster with sign-insensitive rotation distance plus translation
distance. Accept a cluster only with support >=4, rotation spread <=3 degrees,
and translation spread <=5 mm. Store member event IDs in the model.

- [ ] **Step 4: Run clustering tests and commit**

Run: `python -m pytest -q tests/test_mocap_branch.py -k cluster`

Expected: PASS.

```bash
git add twm/react_preprocess/mocap_branch.py tests/test_mocap_branch.py
git commit -m "feat: learn persistent OptiTrack branch transforms"
```

### Task 4: Decode persistent wrong-branch intervals

**Files:**
- Modify: `twm/react_preprocess/mocap_branch.py`
- Modify: `tests/test_mocap_branch.py`
- Modify: `twm/react_preprocess/mocap_repair.py`

- [ ] **Step 1: Write failing sequence-decoder tests**

Test a wrong branch lasting 100 frames, repeated A/B chatter, a transform seen
only once, and genuine fast motion. Assert only supported alternate-branch
states are corrected and paired boundaries become one event.

```python
decoded = decode_branch_states(observed, model)
assert decoded.intervals == ((80, 179),)
np.testing.assert_allclose(decoded.pose[80:180], truth[80:180], atol=1e-6)
assert decode_branch_states(real_fast_motion, model).intervals == ()
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest -q tests/test_mocap_branch.py -k decode`

Expected: FAIL because the decoder is absent.

- [ ] **Step 3: Implement two-state dynamic programming**

For each frame score identity and inverse-branch hypotheses by robust local
translation/angular acceleration. Add a switch penalty calibrated from clean
context. Backtrack the minimum-cost state sequence, merge adjacent alternate
states, and require continuous corrected entry/exit plus physical native steps.
Return corrected poses and evidence; do not mutate the input.

- [ ] **Step 4: Integrate after detection and before per-bout interpolation**

Mark accepted intervals with method `branch_state_inverse`. HIGH requires a
supported tight transform cluster, full entry/exit anchors, physical corrected
steps, and an enabled task benchmark gate. Unsupported cases remain unchanged.

- [ ] **Step 5: Run tests and commit**

Run:

```bash
python -m pytest -q tests/test_mocap_branch.py tests/test_mocap_repair.py tests/test_repaired_actions.py
git add twm/react_preprocess/mocap_branch.py twm/react_preprocess/mocap_repair.py tests/test_mocap_branch.py
git commit -m "feat: decode persistent OptiTrack wrong branches"
```

### Task 5: Rebuild candidates and enforce the confidence target

**Files:**
- Modify: `twm/react_preprocess/mocap_candidate.py`
- Modify: `twm/scripts/build_mocap_repair_candidates.py`
- Modify: `tests/test_mocap_candidate.py`
- Modify: `tests/test_build_mocap_repair_candidates.py`

- [ ] **Step 1: Add failing provenance and summary tests**

Assert candidate parquets carry repair method per side, native and 15-fps
actions are recomputed, and summary rows separate endpoint, branch-state,
MEDIUM, LOW, long-event, and recovered-action counts.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest -q tests/test_mocap_candidate.py tests/test_build_mocap_repair_candidates.py`

Expected: FAIL on absent method columns/counts.

- [ ] **Step 3: Implement provenance columns and complete inventory inputs**

Add `pose_{side}_repair_method` and carry the method into event JSON and action
sidecars. Build a frozen inventory that includes main four-task source
episodes, `old_data/motherboard`, `old_data/pushT`, and all nine validation
episodes without duplicate identities.

- [ ] **Step 4: Rebuild in a new candidate root**

Run audit, benchmark, build, verify, and summary under
`/media/yxma/Disk1/twm/review/mocap_repair_stage2_2026-09-18`. Never overwrite
the 2026-09-17 candidate root. Expected verification errors: zero. Expected
unresolved events: <=150; if higher, preserve them as invalid and report the
actual count.

- [ ] **Step 5: Commit**

```bash
git add twm/react_preprocess/mocap_candidate.py twm/scripts/build_mocap_repair_candidates.py tests/test_mocap_candidate.py tests/test_build_mocap_repair_candidates.py
git commit -m "feat: rebuild mocap candidates with method provenance"
```

### Task 6: Propagate to release, old data, and validation

**Files:**
- Create: `twm/scripts/propagate_mocap_repairs.py`
- Create: `tests/test_propagate_mocap_repairs.py`
- Modify: `twm/scripts/build_pose_review.py`

- [ ] **Step 1: Write failing source-frame propagation tests**

Construct one source episode and two segments with nonzero source frame starts.
Assert repaired poses/actions map by source-frame identity, not filename or row
offset, and every original column is Arrow-equal.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest -q tests/test_propagate_mocap_repairs.py`

Expected: import failure because the propagator is absent.

- [ ] **Step 3: Implement propagation and review selection**

Resolve rows using `(task, date, source_episode, source_h5_frame)`. Append
repaired poses, method/confidence/validity, and regenerated actions. Render all
remaining LOW events over 60 frames and a deterministic sample of shorter
LOW/MEDIUM events. Refuse missing or duplicate source identities.

- [ ] **Step 4: Run focused and full verification**

Run:

```bash
python -m pytest -q tests/test_propagate_mocap_repairs.py tests/test_mocap_candidate.py tests/test_repaired_actions.py tests/test_mocap_benchmark.py tests/test_mocap_branch.py
python -m pytest -q
```

Expected: focused suite passes. Report unrelated pre-existing full-suite
failures separately; do not weaken or delete them.

- [ ] **Step 5: Commit**

```bash
git add twm/scripts/propagate_mocap_repairs.py twm/scripts/build_pose_review.py tests/test_propagate_mocap_repairs.py
git commit -m "feat: propagate repaired actions across React data scopes"
```

### Task 7: Publish and remotely verify

**Files:**
- Modify: `twm/scripts/publish_old_motherboard_update.py`
- Modify: `tests/test_publish_old_motherboard_update.py`

- [ ] **Step 1: Add failing all-scope operation tests**

Assert dry-run operations cover all expected main, old-data, and validation
parquets plus one immutable manifest and event sidecars, with no video or depth
uploads.

- [ ] **Step 2: Implement scoped commits and remote footer verification**

Upload one scope per atomic commit. Range-read every parquet footer and verify
row count, repaired/action schema, terminal invalidity, provenance columns,
and manifest membership.

- [ ] **Step 3: Run tests, publish, and verify**

Run publication dry-runs, inspect operation counts, publish scope by scope,
then run remote verification. Expected errors: zero.

- [ ] **Step 4: Commit**

```bash
git add twm/scripts/publish_old_motherboard_update.py tests/test_publish_old_motherboard_update.py
git commit -m "feat: publish repaired actions across React scopes"
```
