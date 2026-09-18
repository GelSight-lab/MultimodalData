# Old Motherboard Action Update and Portable V8 Handoff

**Date:** 2026-09-18

## Goal

Update the 32 old motherboard episodes (`2026-05-10`, `2026-05-11`, and
`2026-05-19`) with per-frame LeRobot-style actions and explicit mocap repair
validity as soon as possible. In parallel, publish a self-contained V8 force
rebuild package and operator instructions so the exact force reconstruction can
run on another PC that has the original raw H5 recordings.

The action update and force update are separate Hugging Face commits. This lets
the action data become available within the one-hour target without replacing
exact V8 reconstruction with an inaccurate MP4-based shortcut.

## Action Update

The source is the immutable release parquet plus the existing mocap repair
candidate for the matching episode. Original published columns remain
unchanged. The update appends:

- `sensor_left_pose_repaired`, `sensor_right_pose_repaired`: selected candidate
  poses; raw `sensor_*_pose` remains unchanged;
- `action`: float32[14], the next-frame repaired left pose followed by the
  next-frame repaired right pose;
- `action_valid_left`, `action_valid_right`;
- `action_repaired_left`, `action_repaired_right`;
- `action_repair_confidence_left`, `action_repair_confidence_right`;
- `action_lost_track_left`, `action_lost_track_right`;
- `action_long_lost_track_left`, `action_long_lost_track_right`.

Only HIGH repairs are automatically valid. MEDIUM and LOW events remain
invalid until a decision explicitly accepts them. Events longer than 60 native
frames are marked as long lost-track and remain invalid. The final frame repeats
the repaired state to preserve LeRobot's T-row action convention but is marked
invalid for both hands because no observed next transition exists.

The upload also includes deterministic per-episode repair-event JSON files and
a run manifest explaining confidence, invalidity, and long-loss intervals.

## Portable V8 Force Package

Publish under `old_data/motherboard/v8_rebuild_tools/`:

- a frozen source snapshot containing the V8 force reconstruction and the
  minimum React preprocessing modules it imports;
- the exact three calibration assets consumed by V8, with SHA-256 hashes;
- a requirements file and environment-variable template;
- an inventory builder for the 32 release parquet files and corresponding raw
  H5 recordings;
- resumable sharded worker, export, and upload commands;
- `README.md` with setup, expected runtime, disk behavior, restart procedure,
  and the exact destination `old_data/motherboard/meta/...`;
- a manifest identifying source commit, pipeline version, calibration name,
  and every packaged file hash.

The other PC must have the original raw H5 files. Published tactile MP4 files
are not an accepted substitute because measured single-frame force differences
reached 6.48 N.

The upload command first writes a local staged superset. It never uploads V8
NPZ files directly as dataset metadata. It merges V8 force columns into the
already action-augmented parquet and replaces the 32 remote parquet files in
one commit, together with V8 sidecars and manifests.

## Minimal Verification Budget

To minimize elapsed time, omit full preview rendering, full statistical report
generation, repeated model evaluation, and re-downloading every complete
parquet. Retain only checks that prevent a corrupt public release:

1. exactly 32 expected episodes are present;
2. every local and remote replacement has the same row count;
3. all original remote columns remain present with unchanged Arrow types;
4. every force source is readable, stamped pipeline version 8, and uses the
   packaged calibration name.

For actions, additionally assert T rows, float32[14] action shape, terminal
invalidity, and that every non-finite or unresolved mocap interval is invalid.
These checks operate on parquet metadata or vectorized columns and should take
seconds to minutes, not materially affect the one-hour action target.

## Publication Order and Recovery

1. Publish the frozen V8 tool package and instructions.
2. Build and publish the action-augmented 32-parquet commit.
3. Run exact V8 on the other PC, resumably.
4. Export and publish the force/action superset as a second atomic commit.

Each commit records its parent revision and manifest. If a publication fails,
rerunning is idempotent; no deletion of the existing remote dataset is needed.
