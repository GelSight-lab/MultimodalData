# Cross-Task Mocap Action Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a provenance-preserving candidate pipeline that repairs high-confidence OptiTrack branch glitches across motherboard, pushT, rope, and toy, keeps medium/low-confidence actions masked, and regenerates native 30 fps plus derived 15 fps actions without modifying production release trees.

**Architecture:** Pure NumPy/SciPy modules detect complete anomaly bouts, fit robust two-sided SE(3) trajectories, score confidence, and construct side-specific pose/action validity. A separate candidate-tree module snapshots immutable inputs, writes copied-and-augmented parquet files plus event sidecars, and replays review decisions idempotently. Existing preview rendering is extended to compare raw and candidate trajectories; one CLI performs manifest, audit, candidate, decision, and verification phases under a non-production root.

**Tech Stack:** Python 3.9, NumPy, SciPy `Rotation`, PyArrow/Parquet, OpenCV/ffmpeg, pytest.

---

## File map

- Create `twm/react_preprocess/mocap_repair.py`: event model, seed/bout detector, SE(3) trajectory reconstruction, branch selection, confidence scoring.
- Create `twm/react_preprocess/repaired_actions.py`: native 30 fps and two-step 15 fps 9-D actions with validity/provenance.
- Create `twm/react_preprocess/mocap_candidate.py`: input manifest, safe candidate parquet/sidecar writer, deterministic decision replay, candidate verifier.
- Create `twm/scripts/build_mocap_repair_candidates.py`: four-task CLI and metadata summary.
- Create `twm/scripts/benchmark_mocap_repair.py`: clean-interval masking benchmark and calibrated task gates.
- Modify `twm/scripts/build_pose_review.py`: render raw/repaired overlays and all medium/low plus sampled high events; use canonical four-decision vocabulary.
- Create `tests/test_mocap_repair.py`, `tests/test_repaired_actions.py`, `tests/test_mocap_candidate.py`, and `tests/test_mocap_benchmark.py`.
- Modify `tests/test_build_pose_review.py` for candidate-aware review artifacts.

### Task 1: Detect complete anomaly bouts

**Files:**
- Create: `twm/react_preprocess/mocap_repair.py`
- Create: `tests/test_mocap_repair.py`

- [x] **Step 1: Write failing synthetic detector tests**

```python
def test_repeated_ab_toggles_are_one_bout():
    clean = smooth_pose(100)
    bad = alternate_branch(clean, [40, 42, 44, 47, 49])
    bouts = detect_bouts(bad, "right")
    assert len(bouts) == 1
    assert bouts[0].start <= 40 and bouts[0].end >= 49
    assert set(bouts[0].seed_frames) >= {40, 42, 44, 47, 49}

def test_bout_closes_only_after_ten_stable_frames():
    clean = smooth_pose(120)
    bad = alternate_branch(clean, [40, 42, 44, 51])
    bout = detect_bouts(bad, "left")[0]
    assert bout.end >= 51
    assert bout.right_context_start >= 52

def test_genuine_continuous_large_turn_is_not_a_bout():
    assert detect_bouts(continuous_turn(90, degrees=120), "left") == []

def test_nonfinite_and_known_gap_rows_seed_one_bout():
    pose = smooth_pose(100); pose[40:43] = np.nan
    bouts = detect_bouts(pose, "right", known_gaps=[(40, 3)])
    assert [(b.start, b.end, b.kind) for b in bouts] == [(40, 42, "known_gap")]
```

- [x] **Step 2: Run tests and verify RED**

Run: `python -m pytest -q tests/test_mocap_repair.py -k 'bout or turn or nonfinite'`

Expected: collection fails because `twm.react_preprocess.mocap_repair` does not exist.

- [x] **Step 3: Implement detector data model and bout grouping**

Implement immutable `RepairConfig`, `AnomalyBout`, and `RepairEvidence` dataclasses. Compute sign-insensitive quaternion steps, translation steps in mm, finite masks, and robust episode-local scales. Seed discontinuities only when an edge is physically surprising and has return/alternate-branch evidence; merge seeds and known gaps until ten consecutive finite transitions are compatible with the same continuous branch. Store clean-context boundaries and deterministic event IDs.

- [x] **Step 4: Run detector tests and existing anomaly regressions**

Run: `python -m pytest -q tests/test_mocap_repair.py tests/test_pose_anomaly.py tests/test_pose_flicker.py`

Expected: all pass.

- [x] **Step 5: Commit**

```bash
git add twm/react_preprocess/mocap_repair.py tests/test_mocap_repair.py
git commit -m "feat: detect complete mocap anomaly bouts"
```

### Task 2: Fit robust two-sided SE(3) trajectories and score confidence

**Files:**
- Modify: `twm/react_preprocess/mocap_repair.py`
- Modify: `tests/test_mocap_repair.py`

- [x] **Step 1: Add failing reconstruction tests**

```python
def test_majority_wrong_branch_keeps_correct_observations():
    truth = curved_pose(120)
    observed = wrong_branch(truth, range(42, 58))
    observed[47] = truth[47]
    result = repair_pose_stream(observed, "right", task_gate=passing_gate())
    assert result.repaired[47] is False
    assert result.repaired[42:47].all() and result.repaired[48:58].all()
    assert pose_error(result.pose[42:58], truth[42:58]).rotation_p95_deg < 5

def test_two_sided_linear_motion_can_repair_sixty_frames():
    truth = smooth_pose(140)
    observed = wrong_branch(truth, range(40, 100))
    result = repair_pose_stream(observed, "left", task_gate=passing_gate())
    assert result.confidence[40:100].min() == Confidence.HIGH
    assert result.valid[40:100].all()

def test_medium_candidate_is_reconstructed_but_invalid():
    result = repair_pose_stream(ambiguous_curve(), "left", task_gate=passing_gate())
    mask = result.confidence == Confidence.MEDIUM
    assert mask.any() and result.repaired[mask].all()
    assert not result.valid[mask].any()

def test_missing_anchor_is_low_and_original_is_preserved():
    observed = wrong_branch(smooth_pose(80), range(0, 20))
    result = repair_pose_stream(observed, "right", task_gate=passing_gate())
    assert np.array_equal(result.pose[:20], observed[:20])
    assert (result.confidence[:20] == Confidence.LOW).all()
```

- [x] **Step 2: Run new tests and verify RED**

Run: `python -m pytest -q tests/test_mocap_repair.py -k 'majority or sixty or medium or missing_anchor'`

Expected: failures for missing trajectory and confidence APIs.

- [x] **Step 3: Implement robust trajectory reconstruction**

Use median/MAD boundary velocities from up to 15 clean frames per side. Use linear translation when both velocities are statistically zero, otherwise cubic Hermite. Align quaternion signs, map rotations to the left-anchor tangent space with `Rotation.as_rotvec`, fit Hermite there, and map back with `Rotation.from_rotvec`; use shortest-arc SLERP only for ill-conditioned fits. Iteratively classify measured bout rows with episode-local robust residual scales and 20 mm/12 degree hard inlier caps, refitting for at most five iterations. Preserve inlier rows bit-identically and replace only selected outliers.

- [x] **Step 4: Implement confidence gates and result contract**

Add `Confidence(IntEnum)` values NONE/LOW/MEDIUM/HIGH and `PoseRepairResult` arrays for pose, repaired, confidence, valid, event IDs, and event evidence. HIGH requires 15 clean frames per side, duration <=60, calibrated task gate, forward/backward disagreement <=10 mm/5 degrees, unique branch, unit finite quaternions, and reconstructed transitions <=50 mm/30 degrees. MEDIUM gets a candidate but remains invalid; LOW preserves raw poses and is invalid.

- [x] **Step 5: Verify all repair tests**

Run: `python -m pytest -q tests/test_mocap_repair.py tests/test_pose_repair.py tests/test_pose_flicker.py`

Expected: all pass, including unchanged genuine motion and idempotence.

- [x] **Step 6: Commit**

```bash
git add twm/react_preprocess/mocap_repair.py tests/test_mocap_repair.py
git commit -m "feat: reconstruct and score mocap repair candidates"
```

### Task 3: Recompute native and 15 fps actions

**Files:**
- Create: `twm/react_preprocess/repaired_actions.py`
- Create: `tests/test_repaired_actions.py`

- [x] **Step 1: Write failing action-contract tests**

```python
def test_native_action_is_translation_plus_relative_rot6d():
    actions = native_actions(poses, pose_valid, pose_repaired, pose_event_id)
    assert actions.values.shape == (len(poses) - 1, 9)
    assert_allclose(actions.values[:, :3], np.diff(poses[:, :3], axis=0))

def test_native_valid_requires_both_valid_endpoints_and_physical_step():
    pose_valid[4] = False
    actions = native_actions(poses, pose_valid, repaired, event_ids)
    assert not actions.valid[3] and not actions.valid[4]

def test_fps15_valid_is_and_and_repaired_is_or():
    native.valid[:] = [True, False, True, True]
    native.repaired[:] = [False, True, False, True]
    half = actions_fps15(poses, native)
    assert half.valid.tolist() == [False, True]
    assert half.repaired.tolist() == [True, True]
```

- [x] **Step 2: Run and verify RED**

Run: `python -m pytest -q tests/test_repaired_actions.py`

Expected: import failure for the new module.

- [x] **Step 3: Implement action builders**

Build world-frame relative rotations `R[k+1] * R[k].inv()` and encode the first two rotation-matrix columns as rot6d after the 3-D translation delta. Mark transitions invalid for non-finite endpoints, invalid pose endpoints, translation >50 mm, or rotation >30 degrees. Propagate repaired flags and sorted event IDs from either endpoint. Build 15 fps actions directly from pose `k -> k+2`; validity is native `k AND k+1`, repaired is native `k OR k+1`, and provenance is their union.

- [x] **Step 4: Run action tests plus toolbox regressions**

Run: `python -m pytest -q tests/test_repaired_actions.py twm/scripts/test_synth_actions.py`

Expected: all pass.

- [x] **Step 5: Commit**

```bash
git add twm/react_preprocess/repaired_actions.py tests/test_repaired_actions.py
git commit -m "feat: rebuild repair-aware 30 and 15 fps actions"
```

### Task 4: Snapshot inputs and write a safe candidate tree

**Files:**
- Create: `twm/react_preprocess/mocap_candidate.py`
- Create: `tests/test_mocap_candidate.py`

- [ ] **Step 1: Write failing manifest and safety tests**

```python
def test_manifest_is_sorted_and_contains_sha256(tmp_path):
    source = sample_release(tmp_path / "release")
    manifest = snapshot_inputs(source, ["motherboard", "pushT", "rope", "toy"])
    assert [x.path for x in manifest.files] == sorted(x.path for x in manifest.files)
    assert all(len(x.sha256) == 64 and x.rows > 0 for x in manifest.files)

def test_writer_refuses_production_or_nested_destination(tmp_path):
    source = sample_release(tmp_path / "release")
    with pytest.raises(ValueError, match="production"):
        CandidateWriter(source, source)
    with pytest.raises(ValueError, match="production"):
        CandidateWriter(source, source / "candidate")

def test_candidate_keeps_unselected_columns_and_good_poses_exact(tmp_path):
    source = sample_episode(tmp_path)
    out = write_candidate_episode(source, tmp_path / "review", gate=passing_gate())
    before, after = pq.read_table(source), pq.read_table(out.parquet)
    assert before["force_left_normal_n"].equals(after["force_left_normal_n"])
    assert np.array_equal(raw_pose[~out.repaired], candidate_pose[~out.repaired])
```

- [ ] **Step 2: Run and verify RED**

Run: `python -m pytest -q tests/test_mocap_candidate.py`

Expected: import failure for the new module.

- [ ] **Step 3: Implement immutable manifest and root guards**

Snapshot relative path, bytes, rows, mtime_ns, and streaming SHA-256 for every `release/<task>/meta/*/*.parquet`; persist schema version, source root, creation time, task set, and digest of the canonical manifest JSON. Reject output equal to, inside, or parent of any known production root (`release`, `release_zup`, `release_cut`) and reject an existing candidate whose manifest digest differs unless a new output root is supplied.

- [ ] **Step 4: Implement candidate parquet and sidecar writer**

Copy the source table and replace only candidate pose columns. Add per-side `pose_*_repaired`, `pose_*_repair_confidence`, and `pose_*_valid` columns to the `T`-row candidate parquet. Store the `T-1` native actions and the derived 15 fps actions in separate per-episode sidecars with value/valid/repaired/event-ID arrays. Store event evidence and event-ID dictionaries in deterministic JSON. Use temporary sibling files plus `os.replace`; verify source size/digest immediately before writing. Never mutate source files.

- [ ] **Step 5: Implement deterministic decision replay and verification**

Accept only `keep_raw`, `accept_repair`, `invalidate`, or `unsure`. Match every decision to exactly one event ID; reject duplicates and unknown IDs. Accepted medium events become valid; keep_raw restores the raw pose and applies physical validity; invalidate/unsure stay invalid. Applying the same canonical decisions twice must produce byte-identical logical tables and sidecars. Verify all quaternion norms, changed-frame declarations, untouched-frame equality, action contracts, and source digests.

- [ ] **Step 6: Run candidate tests**

Run: `python -m pytest -q tests/test_mocap_candidate.py tests/test_repaired_actions.py`

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add twm/react_preprocess/mocap_candidate.py tests/test_mocap_candidate.py
git commit -m "feat: write provenance-safe mocap candidate trees"
```

### Task 5: Calibrate confidence from clean-data masking

**Files:**
- Create: `twm/scripts/benchmark_mocap_repair.py`
- Create: `tests/test_mocap_benchmark.py`

- [ ] **Step 1: Write failing benchmark tests**

```python
def test_benchmark_reports_all_required_errors():
    report = benchmark_task(clean_episodes, anomaly_lengths=[1, 5, 20], seed=7)
    assert set(report.metrics) >= {"translation_mm", "orientation_deg",
                                   "native_translation_mm", "native_rotation_deg"}

def test_fewer_than_one_hundred_intervals_disables_high_confidence():
    report = benchmark_task(clean_episodes, max_intervals=99, seed=7)
    assert report.gate.high_confidence_enabled is False

def test_calibration_is_deterministic():
    assert benchmark_task(clean, seed=7).to_dict() == benchmark_task(clean, seed=7).to_dict()
```

- [ ] **Step 2: Run and verify RED**

Run: `python -m pytest -q tests/test_mocap_benchmark.py`

Expected: script module cannot be imported.

- [ ] **Step 3: Implement deterministic masking benchmark**

Select disjoint clean intervals by task, duration, and motion bin using a seeded generator. Replace intervals with observed wrong-branch and full-gap patterns, reconstruct without exposing truth, and report median/p95 pose plus native-action errors. Emit a versioned JSON task gate. Enable HIGH only for >=100 held-out intervals and thresholds <=2/10 mm pose, <=1/5 degree pose, <=5 mm native translation p95, and <=3 degree native rotation p95.

- [ ] **Step 4: Run benchmark tests**

Run: `python -m pytest -q tests/test_mocap_benchmark.py tests/test_mocap_repair.py`

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add twm/scripts/benchmark_mocap_repair.py tests/test_mocap_benchmark.py
git commit -m "feat: calibrate mocap repair confidence gates"
```

### Task 6: Extend review artifacts for raw-versus-repaired decisions

**Files:**
- Modify: `twm/scripts/build_pose_review.py`
- Modify: `tests/test_build_pose_review.py`

- [ ] **Step 1: Write failing candidate-aware review tests**

```python
def test_review_contains_every_medium_and_low_event_and_samples_high():
    selected = select_review_events(events, high_sample_rate=1.0, seed=4)
    assert {e["event_id"] for e in selected} == {e["event_id"] for e in events}

def test_html_uses_canonical_replayable_decisions(tmp_path):
    html = write_html(events, tmp_path / "index.html").read_text()
    for choice in ("keep_raw", "accept_repair", "invalidate", "unsure"):
        assert f'data-decision="{choice}"' in html

def test_clip_receives_raw_and_candidate_pose_arrays(monkeypatch, tmp_path):
    info = render_event_clip(event, source_root, output, candidate_parquet=candidate)
    assert info["frames"] == expected_frames
    assert overlay_observations == expected_raw_and_candidate_rows
```

- [ ] **Step 2: Run and verify RED**

Run: `python -m pytest -q tests/test_build_pose_review.py -k 'canonical or candidate or medium'`

Expected: new selection/API assertions fail.

- [ ] **Step 3: Implement review selection and overlays**

Render every MEDIUM/LOW event and deterministic HIGH samples. Load candidate poses alongside raw poses, plot raw/candidate rotation, translation, velocity, acceleration, residuals, anchors, retained measured rows, and replaced rows. Keep existing camera/tactile panel and gel-point projection. Export schema-versioned decisions with manifest digest and exact event IDs.

- [ ] **Step 4: Run all review tests**

Run: `python -m pytest -q tests/test_build_pose_review.py tests/test_preview_clip_window.py tests/test_preview_gel_frame_choice.py`

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add twm/scripts/build_pose_review.py tests/test_build_pose_review.py
git commit -m "feat: review raw and repaired mocap trajectories"
```

### Task 7: Add the four-task CLI and run the candidate rollout

**Files:**
- Create: `twm/scripts/build_mocap_repair_candidates.py`
- Create: `tests/test_build_mocap_repair_candidates.py`

- [ ] **Step 1: Write failing CLI integration tests**

```python
def test_audit_then_build_uses_exact_manifest_snapshot(tmp_path):
    source = sample_four_task_release(tmp_path)
    manifest = cli("audit", source=source, output=tmp_path / "review")
    add_episode_after_snapshot(source)
    result = cli("build", manifest=manifest, output=tmp_path / "review")
    assert result.episodes == manifest.episode_count

def test_verify_detects_any_source_mutation(tmp_path):
    result = build_sample_candidate(tmp_path)
    mutate_source(result.source_episode)
    with pytest.raises(VerificationError, match="digest"):
        verify_candidate(result.root)
```

- [ ] **Step 2: Run and verify RED**

Run: `python -m pytest -q tests/test_build_mocap_repair_candidates.py`

Expected: CLI module does not exist.

- [ ] **Step 3: Implement CLI phases**

Provide `audit`, `benchmark`, `build`, `review`, `apply-decisions`, and `verify`. Defaults are source `/media/yxma/Disk1/twm/release`, tasks motherboard/pushT/rope/toy, and output `/media/yxma/Disk1/twm/review/mocap_repair_2026-09-17/candidate_release`. Require an existing manifest and task gates for build. Emit machine-readable summaries by task/confidence/recovered native actions/recovered 15 fps actions and never promote or upload.

- [ ] **Step 4: Run focused and full regression suites**

Run: `python -m pytest -q tests/test_build_mocap_repair_candidates.py tests/test_mocap_candidate.py tests/test_mocap_repair.py tests/test_repaired_actions.py tests/test_mocap_benchmark.py tests/test_build_pose_review.py tests/test_pose_anomaly.py tests/test_pose_repair.py tests/test_pose_flicker.py tests/test_repair_release_poses.py`

Expected: all pass.

- [ ] **Step 5: Commit implementation**

```bash
git add twm/scripts/build_mocap_repair_candidates.py tests/test_build_mocap_repair_candidates.py
git commit -m "feat: orchestrate cross-task mocap repair candidates"
```

- [ ] **Step 6: Run metadata-only audit and benchmark on the immutable snapshot**

```bash
python twm/scripts/build_mocap_repair_candidates.py audit
python twm/scripts/build_mocap_repair_candidates.py benchmark
```

Expected: manifest covers all four tasks; benchmark report either enables HIGH per task with >=100 held-out intervals and all accuracy gates, or explicitly restricts that task to MEDIUM/LOW.

- [ ] **Step 7: Build and verify the separate candidate tree**

```bash
python twm/scripts/build_mocap_repair_candidates.py build
python twm/scripts/build_mocap_repair_candidates.py review --render
python twm/scripts/build_mocap_repair_candidates.py verify
```

Expected: candidate and review roots exist; production file digests match the input manifest; every candidate file passes quaternion, provenance, action-validity, determinism, and clip-decode checks.

- [ ] **Step 8: Report results without promotion**

Report per-task event counts, confidence tiers, repaired frames, recovered native/15 fps actions, unresolved events, review index path, manifest digest, verification results, and any task whose HIGH gate remained disabled. Do not change `release`, `release_zup`, `release_cut`, upload to Hugging Face, or delete any data.

## Plan self-review

- Spec coverage: detector/bouts, robust SE(3), branch selection, all confidence gates, side-specific pose/action contract, 15 fps propagation, immutable input manifest, candidate-root safety, replayable review, masking benchmark, four-task rollout, and non-promotion are each assigned to a task.
- Placeholder scan: no deferred or unspecified steps; implementation behavior and test commands are explicit.
- Type consistency: `Confidence`, `PoseRepairResult`, native/15 fps action records, task gates, manifest digest, canonical decisions, and event IDs are introduced before downstream use.
