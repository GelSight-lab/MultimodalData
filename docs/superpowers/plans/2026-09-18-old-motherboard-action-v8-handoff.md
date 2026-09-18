# Old Motherboard Action and V8 Handoff Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a portable, exact V8 force-rebuild package for another PC and update all 32 old motherboard parquet files with repaired LeRobot-style actions plus explicit invalid/long-lost-track labels from this PC.

**Architecture:** A pure exporter combines the immutable release parquet, the provenance-preserving mocap candidate parquet, native action validity sidecars, and repair event JSON into an action-augmented staging tree without changing original columns. A separate handoff packager freezes the V8 Python source and its three reviewed calibration assets, generates a manifest and concise remote-PC runbook, and stages the package for one Hugging Face commit. Publication uses two atomic commits: tools first, action data second.

**Tech Stack:** Python 3.9, NumPy, PyArrow/Parquet, Hugging Face Hub, pytest, SHA-256 manifests.

---

## File map

- Create `twm/scripts/export_old_motherboard_actions.py`: build and verify the 32 action-augmented parquet files and event sidecars.
- Create `tests/test_export_old_motherboard_actions.py`: action alignment, validity, long-event, and source-preservation tests.
- Create `twm/force_recovery/package_old_v8_handoff.py`: freeze source/assets, write portable instructions and package manifest.
- Create `tests/test_package_old_v8_handoff.py`: package completeness and hash tests.
- Create `twm/scripts/publish_old_motherboard_update.py`: atomic Hugging Face tool/action publication with remote metadata checks.
- Create `tests/test_publish_old_motherboard_update.py`: repository-path and operation-set tests without performing network writes.

### Task 1: Export per-frame repaired actions

**Files:**
- Create: `twm/scripts/export_old_motherboard_actions.py`
- Create: `tests/test_export_old_motherboard_actions.py`

- [ ] **Step 1: Write failing tests for the T-row action contract**

Create fixtures with a four-row raw parquet, candidate poses, pose validity,
native sidecars, and one repair event. Assert:

```python
out = build_action_episode(raw, candidate, native_left, native_right, events)
assert out.num_rows == 4
assert out.schema.field("action").type == pa.list_(pa.float32(), 14)
assert out["action"][0].as_py() == candidate_next_left + candidate_next_right
assert out["action_valid_left"].to_pylist() == [True, False, True, False]
assert out["action_valid_right"].to_pylist()[-1] is False
```

Also assert every original raw column is Arrow-equal in the output and repaired
poses are added under new names instead of replacing raw poses.

- [ ] **Step 2: Run the action tests and verify RED**

Run: `python -m pytest -q tests/test_export_old_motherboard_actions.py`

Expected: FAIL because `twm.scripts.export_old_motherboard_actions` does not exist.

- [ ] **Step 3: Implement episode assembly**

Implement:

```python
def build_action_episode(raw: pa.Table, candidate: pa.Table,
                         native_left: Mapping[str, np.ndarray],
                         native_right: Mapping[str, np.ndarray],
                         events: Sequence[Mapping[str, object]]) -> pa.Table:
    """Return raw columns unchanged plus repaired poses and T-row actions."""
```

Use candidate `sensor_{side}_pose` as repaired pose. Define row `i < T-1`
validity and repaired flags from the matching native sidecar entry. Set the
terminal row invalid and unrepaired. Define action confidence as the maximum
of endpoint pose confidence. Define lost-track for nonterminal invalid native
transitions and long-lost-track when the transition overlaps an event whose
inclusive duration is greater than 60 frames. Store `action` as fixed-size
float32[14].

- [ ] **Step 4: Add failing tests for unresolved and long events**

Test that MEDIUM/LOW candidate transitions remain invalid, a 61-frame event
marks every overlapping transition as long lost-track, a 60-frame event does
not, and the terminal invalid row is not mislabeled as lost-track.

- [ ] **Step 5: Run tests and verify RED for long-event behavior**

Run: `python -m pytest -q tests/test_export_old_motherboard_actions.py -k 'long or unresolved'`

Expected: FAIL until event-overlap labeling exists.

- [ ] **Step 6: Implement tree export and minimal verifier**

Implement CLI arguments `--release`, `--candidate`, `--output`, and fixed dates
`2026-05-10,2026-05-11,2026-05-19`. Refuse any episode-set other than the 32
matching release/candidate files. Copy repair-event JSON unchanged, write an
`action_update_manifest.json`, and verify only: 32 episodes, equal row counts,
unchanged original names/types/values, action shape, terminal invalidity, and
no unresolved/non-finite transition marked valid.

- [ ] **Step 7: Run focused tests and commit**

Run: `python -m pytest -q tests/test_export_old_motherboard_actions.py tests/test_repaired_actions.py tests/test_mocap_candidate.py`

Expected: all pass.

Commit:

```bash
git add twm/scripts/export_old_motherboard_actions.py tests/test_export_old_motherboard_actions.py
git commit -m "feat: export repaired actions for old motherboard data"
```

### Task 2: Build the portable V8 handoff package

**Files:**
- Create: `twm/force_recovery/package_old_v8_handoff.py`
- Create: `tests/test_package_old_v8_handoff.py`

- [ ] **Step 1: Write a failing package-completeness test**

Use temporary source and asset roots and assert the staged package contains:

```python
required = {
    "README.md", "manifest.json", "requirements-force-v8.txt",
    "twm/pipeline_stages.py",
    "assets/feature_cache/glowtact_round_mm.json",
    "assets/feature_cache/glowtact_round_8_15_di4.json",
    "assets/lut_calibration/glowtact_lut.npz",
}
assert required <= relative_files(package)
assert all(entry["sha256"] == sha256(package / entry["path"])
           for entry in manifest["files"])
```

Assert Python source under both `twm/force_recovery/` and
`twm/react_preprocess/` is present while caches and generated media are absent.

- [ ] **Step 2: Run package tests and verify RED**

Run: `python -m pytest -q tests/test_package_old_v8_handoff.py`

Expected: FAIL because the packager does not exist.

- [ ] **Step 3: Implement deterministic packaging**

Implement `build_package(source_root, force_root, output)` to copy `.py`, `.md`,
`.json`, and `.toml` source files needed by the V8 modules, excluding
`__pycache__`, tests, media, and runtime caches. Copy the three assets only
after matching the reviewed SHA-256 values from the V8 runbook. Generate a
sorted manifest with source Git commit, dirty diff SHA-256, pipeline version,
calibration name, and file hashes.

- [ ] **Step 4: Generate actionable remote-PC instructions**

The generated README must give exact commands to:

1. download `old_data/motherboard/v8_rebuild_tools/` and the 32 old parquet files;
2. create/activate Python 3.9 environment and install pinned dependencies;
3. set `REACT_DATA_ROOT`, `REACT_STAGE_ROOT`, and `REACT_FORCE_RECOVERY_ROOT` before importing the package;
4. place raw H5 at `REACT_DATA_ROOT/motherboard/<date>/episode_NNN.h5`;
5. copy packaged assets into the candidate force root;
6. run `python -m twm.force_recovery.batch_worker 0 1` and safely resume it;
7. run force export with an explicit `--root`;
8. publish the V8/action superset to `old_data/motherboard/meta/...`.

State that MP4 input is prohibited for exact V8 reconstruction and that the
expected current-machine runtime is approximately two hours.

- [ ] **Step 5: Run tests and commit**

Run: `python -m pytest -q tests/test_package_old_v8_handoff.py tests/test_force_asset_is_declared.py tests/test_export_scopes_to_a_task.py`

Expected: all pass.

Commit:

```bash
git add twm/force_recovery/package_old_v8_handoff.py tests/test_package_old_v8_handoff.py
git commit -m "feat: package portable v8 force rebuild"
```

### Task 3: Define atomic Hugging Face publication

**Files:**
- Create: `twm/scripts/publish_old_motherboard_update.py`
- Create: `tests/test_publish_old_motherboard_update.py`

- [ ] **Step 1: Write failing publication-operation tests**

Assert tool operations target only
`old_data/motherboard/v8_rebuild_tools/...`; action operations target the 32
`old_data/motherboard/meta/<date>/episode_NNN.parquet` files, event JSON under
`old_data/motherboard/action_repair_events/...`, and one manifest. Reject a
non-32 action tree and any unexpected date.

- [ ] **Step 2: Run publication tests and verify RED**

Run: `python -m pytest -q tests/test_publish_old_motherboard_update.py`

Expected: FAIL because the publication module does not exist.

- [ ] **Step 3: Implement operation construction and dry-run**

Create deterministic `CommitOperationAdd` lists and CLI subcommands
`tools`, `actions`, and `verify-remote`. `--dry-run` prints operation count and
paths without contacting a write API. Non-dry publication calls one
`HfApi.create_commit` per subcommand and prints the returned commit URL/OID.

- [ ] **Step 4: Implement minimal remote verification**

After the action commit, range-read all 32 parquet footers and check row count,
required action fields, fixed-size action type, and presence of the manifest.
Download one representative parquet from each date and verify terminal action
validity is false. Do not download every complete file.

- [ ] **Step 5: Run tests and commit**

Run: `python -m pytest -q tests/test_publish_old_motherboard_update.py`

Expected: all pass.

Commit:

```bash
git add twm/scripts/publish_old_motherboard_update.py tests/test_publish_old_motherboard_update.py
git commit -m "feat: publish old motherboard action and v8 tools"
```

### Task 4: Build and publish the V8 handoff

**Files:**
- Output only: `/media/yxma/Disk1/twm/old_motherboard_update_2026-09-18/v8_rebuild_tools/`

- [ ] **Step 1: Build the package**

Run:

```bash
python -m twm.force_recovery.package_old_v8_handoff \
  --force-root /media/yxma/Disk1/twm/force_v8_20260916_182159/force \
  --output /media/yxma/Disk1/twm/old_motherboard_update_2026-09-18/v8_rebuild_tools
```

Expected: package manifest reports all three reviewed asset hashes and pipeline version 8.

- [ ] **Step 2: Dry-run and publish tools**

Run:

```bash
python -m twm.scripts.publish_old_motherboard_update tools \
  --root /media/yxma/Disk1/twm/old_motherboard_update_2026-09-18/v8_rebuild_tools --dry-run
python -m twm.scripts.publish_old_motherboard_update tools \
  --root /media/yxma/Disk1/twm/old_motherboard_update_2026-09-18/v8_rebuild_tools
```

Expected: one Hugging Face commit under `old_data/motherboard/v8_rebuild_tools/`.

### Task 5: Build and publish repaired actions

**Files:**
- Output only: `/media/yxma/Disk1/twm/old_motherboard_update_2026-09-18/actions/`

- [ ] **Step 1: Build and minimally verify all action parquet files**

Run:

```bash
python -m twm.scripts.export_old_motherboard_actions \
  --release /media/yxma/Disk1/twm/release \
  --candidate /media/yxma/Disk1/twm/review/mocap_repair_2026-09-17/candidate_release \
  --output /media/yxma/Disk1/twm/old_motherboard_update_2026-09-18/actions
```

Expected: `episodes=32 errors=0` and a manifest summarizing valid, repaired,
unresolved, and long-lost-track transitions.

- [ ] **Step 2: Dry-run and publish actions**

Run:

```bash
python -m twm.scripts.publish_old_motherboard_update actions \
  --root /media/yxma/Disk1/twm/old_motherboard_update_2026-09-18/actions --dry-run
python -m twm.scripts.publish_old_motherboard_update actions \
  --root /media/yxma/Disk1/twm/old_motherboard_update_2026-09-18/actions
```

Expected: one commit replacing exactly 32 old parquet paths and adding event sidecars plus manifest.

- [ ] **Step 3: Verify remote metadata and representative files**

Run:

```bash
python -m twm.scripts.publish_old_motherboard_update verify-remote
```

Expected: `episodes=32 schema_errors=0 sample_errors=0`.

- [ ] **Step 4: Run final regression suite**

Run:

```bash
python -m pytest -q \
  tests/test_export_old_motherboard_actions.py \
  tests/test_package_old_v8_handoff.py \
  tests/test_publish_old_motherboard_update.py \
  tests/test_repaired_actions.py \
  tests/test_mocap_candidate.py
```

Expected: all pass.
