# v8 force re-run: what happens after the NPZ land

Run directory: `/media/yxma/Disk1/twm/force_v8_20260916_182159`
Inventory: 56 episodes / 112 sensor-sides / 404.4 min, `FORCE_SINCE=2026-09-10`.

## What this run changes

The production force tree currently holds, for dates >= 2026-09-10:

    pipeline_version 5   112 NPZ    the calibration that scored rho 0.143 e2e
    pipeline_version 0    10 NPZ    unstamped
    pipeline_version 7     1 NPZ
    pipeline_version 8     1 NPZ
    toy                    0 NPZ    never had force
    rope/2026-09-16        0 NPZ    never had force

So the run does two different things at once: it REPLACES 112 stale v5
estimates, and it FILLS the 27 published segments that currently declare
`force: false` (toy's 10 and rope/2026-09-16's 17).

## Column schema: the operator's 2026-09-16 decision

Publish the measurement, withhold the control policy.

    force_<side>_normal_n       measured newtons            SHIPS
    force_<side>_source_frame   which tactile frame         SHIPS
    force_<side>_penetration_mm force / k                   WITHHELD
    force_<side>_target_pose    pose displaced by that      WITHHELD

Reason: v8 measures to 15 N and the data reaches 14.99 N with 0% ceiling
saturation, while the shipped k = 2 N/mm caps the exporter's 4.25 mm
gel-thickness gate at 8.5 N -- 11.96% of contact frames commanded a target
past it. Raising k means choosing a number nobody measured and restating every
published target pose. See `--force-only`.

**Consequence to announce**: published rope segments currently carry all eight
columns. After this they carry four. Anyone who downloaded earlier sees a
schema change. `publish.check_no_column_loss` will refuse this unless the two
withdrawn columns per side are named in `allow_dropping` -- that naming IS the
audit trail, so do not reach for a blanket flag.

## Order of operations

Stage graph (`pipeline_stages.STAGES`):

    build -> force -> \
    build -> curate -> (export) -> ...
                    -> zup -> segment -> index -> verify -> publish

`export` needs BOTH `force` and `curate`. Do not run `run_all()`: the runbook
is explicit that its fixed default roots and whole-tree enumeration are wrong
for this migration.

### 1. Finish and validate (Steps 4-5)

    python "$RUN/validate.py"        # the full 112, extracted verbatim from the runbook

Acceptance is every selected side, not a subset. The strongest check it makes
is re-solving four named frames per side and requiring agreement below 1e-6 N;
on the 20-side partial run the worst was 4.61e-07. That check proves alignment
and reproducibility together -- a row-offset would differ by newtons, not 1e-7.

### 2. Back up, then promote

Back up the production force tree BEFORE copying anything in. Keep the
candidate tree and `inventory.json` for rollback. `motherboard` already
contains a `_pre_realign_backup`; do not delete an existing backup to make a
command proceed.

Promotion is a copy of the candidate NPZ over dates >= 2026-09-10 only. Older
dates and the 2026-09-09 validation epoch are out of scope and must not move.

### 3. Export force columns, force-only

    python -m twm.force_recovery.export_force_columns export --force-only --root "$RUN/release_force"

The source must be an uncut master WITHOUT existing `force_*` columns -- the
exporter appends and is not a replace-columns migration. `$RUN/input` is that
frozen source. Rebuild from it on every run.

Step 6's stiffness decision does not block this: with no k, the columns that
needed one are simply not written, and the gel-thickness gate has nothing to
police (`penetration_over_gel_thickness_frac` is None, not 0.0).

### 4. Z-up, segment, index

`convert_release_zup.py --force-src <the force-export dir>`; validate its merge
did not fall back to stale columns. It rotates whichever POSE_COLS the table
actually has, so an absent `target_pose` is correctly never rotated.

Force is computed on uncut episodes and then sliced with exactly the rows used
for observations and poses. Never renumber `force_<side>_source_frame`.

### 5. Previews

    python twm/scripts/build_release_previews.py --task <t> \
        --stage-root "$RUN/release_cut" --force-root <promoted force root> \
        --overwrite --clip-s 30 --speed 1

`--force-root` is mandatory here. Without it `build_one_preview` falls back to
its module-level FORCE_ROOT, and a preview whose numbers came from the wrong
tree looks exactly like one that didn't -- the runbook's "never render v8
labels over v7 values". All four tasks are accepted now that `toy` is in
`CALIB_DIRS`.

### 6. Certify, publish, restate

Certification receipts key on parquet CONTENT hash, so every re-cut episode
re-certifies and untouched ones stay cheap.

Then: `episodes.jsonl` force flags flip to true for the 27, statistics and the
stats figures recompute, and the docs that still describe eight force columns
have to be corrected -- `twm/docs/USAGE.md`, `twm/force_recovery/
readme_force_section.md`, `readme_usage_section.md`. Write those against the
shipped output, not against this plan.

## Gates already taught to tell force-only from broken

| gate | was | now |
|---|---|---|
| `dataset_layout.require_force` | demanded all 8 | measured half required, derived half optional but all-or-nothing |
| `publish.check_no_column_loss` | refused any shrink | refuses any UNDECLARED shrink |
| `build_release_publish.force_overlay_plan` | subset test on all 8 | subset test on the measured half |
| `convert_release_zup` | - | needed nothing; rotates columns that exist |
| `curation.force_flag` | - | needed nothing; set intersection |
