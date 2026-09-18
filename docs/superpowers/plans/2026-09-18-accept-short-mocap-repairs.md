# Short Repair Acceptance Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline, task by task.

**Goal:** Apply the operator-approved short-repair policy, recover supported
LOW branch patterns including examples 24/25, and publish aligned artifacts.

**Architecture:** Isolate LOW pattern recovery from policy admission and force
target transport. Reuse existing native/15 fps action builders. Stage a pinned
dataset update, verify preserved columns, then commit all dependent artifacts
atomically. Raw poses and existing force columns remain immutable.

**Tech Stack:** Python, NumPy, SciPy Rotation/Slerp, PyArrow, pytest, HF Hub.

## Task 1: Pattern recovery

Files: `twm/react_preprocess/mocap_pattern_recovery.py` and
`tests/test_mocap_pattern_recovery.py`.

- [ ] Write synthetic tests for repeated flicker, mixed short/long branch
  stretches, motion preservation, missing return, NaN, overlength, and no
  evidence. Run `pytest -q tests/test_mocap_pattern_recovery.py` and observe
  failures before adding implementation.
- [ ] Implement `recover_pattern(raw, start, end, references)` returning a
  corrected interval plus method/evidence, or None. Repeated references come
  from `find_branch_candidates`; body correction is `Robs * B.inv()` and
  `pobs - Rcorrected.apply(b)`. Keep original unaffected rows byte-identical.
- [ ] Validate on the two exact real intervals and scan all 125 LOW events.
  Do not lower thresholds to obtain a target recovery count.

## Task 2: Policy and force consistency

Files: `twm/react_preprocess/accepted_repairs.py` and
`tests/test_accepted_repairs.py`.

- [ ] Write tests before implementation: MEDIUM duration 33 accepted/34
  refused, LOW remains invalid unless fully recovered, raw preservation,
  next-frame mask and 15 fps mask, zero-force target identity, and rigid-frame
  transport of existing force targets.
- [ ] Implement per-table acceptance. Rebuild native action and 15 fps
  sidecars through `native_actions` and `actions_fps15`. Add operator acceptance
  and joint masks; never raise confidence merely because of approval.
- [ ] Transport existing raw force target displacement into its sensor-local
  frame and back through the repaired orientation. Validate finite inputs,
  quaternion correspondence, and recorded penetration magnitude before use.
  This preserves each source's verified direction convention without guessing
  one global pressing axis. Add repaired targets and same-row validity, retaining
  all original force fields and metadata. Missing force remains unavailable.

## Task 3: Dataset staging, verification, publication

File: `twm/scripts/accept_mocap_repairs.py`.

- [ ] Add CLI that reads only a verified pinned publication tree, creates a
  new output tree, writes updated parquets/events/NPZ and policy/audit docs.
- [ ] Verify all preserved columns, unchanged row identities, recomputed
  actions, physical gates, excluded transitions, sidecar masks, force target
  transport and joint training indexing for every episode. Record hashes.
- [ ] Run focused regressions and loader smoke tests. Check remote parent and
  file hashes before uploading. Publish only scoped changes in one commit.
- [ ] Read back remote hashes and load representative parquets/sidecars from
  the new pinned revision. Re-render examples 24/25 using existing Z-up preview
  APIs with the new data; update the Space review paths with clear labels.

## Completion evidence

Report original and newly recovered counts separately, remaining skipped
events, force-version limitations, exact revision, tested alignment contracts,
and preview URLs. Never equate structural compatibility with physical truth.
