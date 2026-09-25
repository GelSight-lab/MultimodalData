# Reprocessing React Tactile Force with v8

This runbook is for the data processor. Run commands from
`/home/yxma/MultimodalData` in the existing project environment. The first five
steps produce and validate candidate force NPZ files. They do not modify raw
HDF5, published parquet, RGB videos or Hugging Face repositories.

Full release publication is a separate step. In particular, resolve the
force-informed-action stiffness policy in Step 6 before exporting targets.

## 1. Freeze the Inputs and Code

- Use complete, closed recordings and their uncut master parquet. Finish any
  rebuild of selected episodes first. Do not include an actively recorded H5.
- Include all four tasks: `motherboard`, `pushT`, `rope`, `toy`.
- Select scope from master parquet, not old force files. New episodes may have
  no old force NPZ. The current main-branch scope starts at `2026-09-10`;
  the `2026-09-09` validation branch and older archives need separate inventories.
- Keep the H5 and alignment/calibration files available. Missing raw data is
  a blocker, not permission to estimate from recompressed preview videos.
- Archive the actual code checkout, including the current uncommitted v8 edits,
  along with the run. A git commit ID alone may not identify the running code.
- Keep one immutable input inventory for all workers. Do not add episodes to
  it during a run: batch-worker shard membership is based on sorted inputs.

Set up a new candidate run. On resume, reuse the SAME `RUN` and environment
values; do not create another timestamped directory.

```bash
cd /home/yxma/MultimodalData
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export RUN=/media/yxma/Disk1/twm/force_v8_$(date +%Y%m%d_%H%M%S)
export FORCE_BASE=/media/yxma/Disk1/twm/force_recovery
export MASTER_RELEASE=/media/yxma/Disk1/twm/release
export FORCE_SINCE=2026-09-10
export REACT_DATA_ROOT=/media/yxma/Disk1/twm/data
export REACT_STAGE_ROOT="$RUN/input"
export REACT_FORCE_RECOVERY_ROOT="$RUN/force"
export REACT_RELEASE="$RUN/input"
```

The three force-root variables are read at import time. Set them before starting
Python; changing them in an already-imported interpreter does not retarget the
modules. They apply to the force writer/exporter, not every legacy preview or
top-level scheduler path. `REACT_RELEASE` separately routes episode-index
lookups used by action export to the frozen input tree.

The initialization block below copies only selected parquet and calibration
assets. It does not copy videos or H5, and it does not run inference. Review
the selected dates before executing it. Missing selected H5 files abort the
inventory instead of being silently skipped.

```bash
python - <<'PY'
import hashlib, json, os, re, shutil
from pathlib import Path
import pyarrow.parquet as pq

run = Path(os.environ['RUN'])
master = Path(os.environ['MASTER_RELEASE'])
data = Path(os.environ['REACT_DATA_ROOT'])
stage = Path(os.environ['REACT_STAGE_ROOT'])
force = Path(os.environ['REACT_FORCE_RECOVERY_ROOT'])
base = Path(os.environ['FORCE_BASE'])
tasks = ('motherboard', 'pushT', 'rope', 'toy')
expected = {
    'feature_cache/glowtact_round_mm.json':
        '6ac589b93dd42e77ebf286245f8e354b782f3e748d313dbf1a508b8d62e8c7a3',
    'feature_cache/glowtact_round_8_15_di4.json':
        '9e49f047f5f5b15bc3ca540aa8705e0b660ca17733b00f250ede21da21267e1a',
    'lut_calibration/glowtact_lut.npz':
        '70d145661e8a2aaf26eda91a364b790ba9fd13aad0a469729cbf31856baed925',
}
assert not run.exists(), 'New run only; resume an existing run without reinitializing'
jobs = []
for task in tasks:
    for p in sorted((master/task/'meta').glob('*/*.parquet')):
        if p.parent.name < os.environ['FORCE_SINCE']:
            continue
        assert re.fullmatch(r'episode_\d+', p.stem), f'Not an uncut episode: {p}'
        raw = data/task/p.parent.name/(p.stem+'.h5')
        assert raw.is_file(), f'Missing raw recording: {raw}'
        schema = pq.read_schema(p)
        required = ['source_h5_frame', 'timestamp'] + [
            f'tactile_{s}_{k}' for s in ('left', 'right') for k in ('intensity', 'is_new')]
        assert all(k in schema.names for k in required), p
        rows = pq.read_metadata(p).num_rows
        assert rows > 0, p
        stat = raw.stat()
        jobs.append(dict(task=task, date=p.parent.name, episode=p.stem, rows=rows,
                         parquet_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                         raw_size=stat.st_size, raw_mtime_ns=stat.st_mtime_ns))
assert all(any(j['task'] == task for j in jobs) for task in tasks), 'Incomplete task scope'
for rel, digest in expected.items():
    assert hashlib.sha256((base/rel).read_bytes()).hexdigest() == digest, rel
run.mkdir(parents=True)
for j in jobs:
    rel = Path(j['task'])/'meta'/j['date']/(j['episode']+'.parquet')
    (stage/rel).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(master/rel, stage/rel)
for rel in expected:
    (force/rel).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(base/rel, force/rel)
index_sha256 = {}
for task in tasks:
    rel = Path(task)/'episodes.jsonl'
    if (master/rel).is_file():
        shutil.copy2(master/rel, stage/rel)
        index_sha256[str(rel)] = hashlib.sha256((stage/rel).read_bytes()).hexdigest()
(run/'inventory.json').write_text(json.dumps(
    dict(jobs=jobs, asset_sha256=expected, index_sha256=index_sha256,
         source_release=str(master)), indent=2))
print('Inventory:', len(jobs), 'episodes /', 2*len(jobs), 'sensor-sides')
for task in tasks:
    subset = [j for j in jobs if j['task'] == task]
    print(task, len(subset), 'episodes,', sum(j['rows'] for j in subset), 'rows')
PY
python -m pip freeze > "$RUN/environment.txt"
git rev-parse HEAD > "$RUN/git-head.txt"
git diff --binary > "$RUN/tracked-changes.patch"
git status --short > "$RUN/git-status.txt"
```

The copied `input/` tree contains metadata only. It is sufficient for force
reconstruction, but is not a complete release tree for curation or publishing.
The patch above excludes untracked files; retain the full source checkout too.

## 2. Check the Assets and Calibration

Required force assets, all outside git, were copied in Step 1:

| File under force root | Purpose |
|---|---|
| `feature_cache/glowtact_round_mm.json` | Frozen v7 low-range calibration, 478 samples |
| `feature_cache/glowtact_round_8_15_di4.json` | v8 high-load calibration, 399 samples |
| `lut_calibration/glowtact_lut.npz` | Geometry fields written alongside force; loaded by the reconstruction module |

Raw calibration JPGs are not needed for inference when these files exist.
Do not rebuild the base cache during production reprocessing. Rebuilding a
calibration is a separate experiment; it invalidates the reviewed hashes.
Restore the reviewed files if they are missing. `react_calib build-tail` can
reproduce the tail only when the original `glowtact/cnc_mini_26/round` images
are also available, and its output must be checked against the reviewed asset.

The tested environment was Python 3.9.18, NumPy 1.26.4, SciPy 1.13.1,
scikit-learn 1.1.1, OpenCV 4.5.5.64, h5py 3.14.0, hdf5plugin 6.0.0,
PyArrow 21.0.0, Pillow 9.1.1 and joblib 1.1.0. Prefer the existing environment
or a recorded environment snapshot. No GPU, neural-network checkpoint or
Open3D renderer is required for batch force reconstruction. FFmpeg is needed
for previews. Preserve the project's rig/alignment calibration configuration
when moving the run to another machine.

```bash
python -m twm.force_recovery.react_calib fit
python -m twm.force_recovery.react_calib range-report
```

Expected: 320 low-range fit / 158 low-range held-out samples; 274 high-range
train / 125 high-range held-out samples; 141 training samples above the tail
join; fitted endpoint 14.99 N. The combined held-out MAE is approximately
1.975 N. `feature_cache/react_range15_holdout.json` contains paired results.
It should report zero change on the original 158 low-range test predictions.
These numbers check calibration reproduction, not actual React force accuracy.

## 3. Run a Pilot

Process one selected episode per task before the full run. This writes into
the candidate force tree and can be resumed by the batch worker afterwards:

```bash
python - <<'PY'
import json, os
from pathlib import Path
from twm.force_recovery.run_episode import process_side

jobs = json.loads((Path(os.environ['RUN'])/'inventory.json').read_text())['jobs']
for task in ('pushT', 'motherboard', 'rope', 'toy'):
    j = next(j for j in jobs if j['task'] == task)
    for side in ('left', 'right'):
        result = process_side(task, j['date'], j['episode'], side)
        print(result['out'], result['force_max_n'], flush=True)
PY
```

Check both sensors, weak contacts, strong contacts, release/re-contact,
reference drift and crop-edge contacts. A value reaching 14.99 N is a range
warning, not evidence the real force is exactly 14.99 N. Do not optimize the
threshold per task to make all four force histograms look similar.

The existing [review page](https://yxma-react-force-recovery.static.hf.space/task-review-2026-09-16-range15/index.html)
is a reference for the approved behavior. Its window-only NPZs are marked
`review_only=True` and contain NaN outside the selected windows. They must
never be copied into the production force tree.

## 4. Process the Frozen Inventory

Use the current checkout, including the 2026-09-16 CPU optimizations, before
starting workers. They are enabled by default in the reconstruction functions;
`process_side` and `batch_worker` need no new arguments. Restart old Python
workers to load new code, but wait for an in-progress output to finish first.
Do not change calibration or restamp force files to enable the optimization.
The pipeline version remains 8 and the full NPZ schema is unchanged.

The paired [speed benchmark](../../docs/superpowers/specs/2026-09-16-force-speed-results.md)
measured 1.84x faster computation and 1.77x including warm H5 reads, with zero
force difference across 184 four-task frames. This is not a full-dataset ETA:
reference setup, disk contention and contact mix also matter. Both force and
LUT geometry are still computed; only provably redundant work is removed.

The supported batch worker discovers input parquet, including episodes that
have never had force files. With the frozen `REACT_STAGE_ROOT` from Step 1:

```bash
python -m twm.force_recovery.batch_worker 0 1 > "$RUN/force.log" 2>&1
```

The process returns nonzero if any side fails. On interruption, rerun the same
command with the same environment. It skips readable files stamped with the
current pipeline version. That skip is not a full integrity check; Step 5 is
required even when the worker says everything was skipped.

For two workers, use disjoint shards `0 2` and `1 2` with separate logs. Wait
for both and check both exit codes. Start with one worker while capture or
other preprocessing is using the disk. Do not use `REVERSE` to run overlapping
workers on the same files, or change the worker count while workers are live.

Keep `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1` as in Step 1. This host
has four physical cores, not eight: the eight logical CPUs share them. More
workers are not automatically faster, especially while capture/encoding is
active. The reported benchmark used one worker with OpenCV's default thread
count. A small in-memory PushT comparison did not show an advantage from
forcing OpenCV to one thread; no new global thread setting was imposed.
For an idle host, measure aggregate completed fresh frames/second with one,
two and four disjoint workers before selecting the bulk setting. Finish one
configuration before starting the next; do not run overlapping shards.

Do not use `reprocess_react run` as the full-inventory command: it discovers
only existing force NPZs and can miss newly collected episodes. `run_episode`
exposes Python functions, not a batch CLI; use `batch_worker` for the scoped
run. Do not run `stamp_calibration` to upgrade old estimates.

## 5. Validate Before Export

Run the focused regression suite from the repository root:

```bash
python -m pytest tests/test_force_*.py tests/test_bounded_force.py \
  tests/test_preview_*.py tests/test_previews_for_segments.py \
  tests/test_upload_previews_incremental.py tests/test_pipeline_stages.py -q
```

Then validate the actual candidate files. This block reads every force array
and recomputes selected named raw frames; it writes only a validation report.
For a pilot, set `jobs` to the four processed inventory entries explicitly.
For release acceptance, use the complete inventory as written below.

```bash
python - <<'PY'
import hashlib, json, os
from pathlib import Path
import h5py, hdf5plugin
import numpy as np
import pyarrow.parquet as pq
from twm.force_recovery import react_calib as rc
from twm.force_recovery.lut_calibration import crop
from twm.force_recovery.run_episode import (
    DATA_ROOT, STAGE_ROOT, OUT_ROOT, PIPELINE_VERSION, open_episode,
    reference_stack, reference_noise_area, _reference_rows,
)

run = Path(os.environ['RUN'])
inventory = json.loads((run/'inventory.json').read_text())
assert PIPELINE_VERSION == 8
for rel, digest in inventory['asset_sha256'].items():
    assert hashlib.sha256((OUT_ROOT/rel).read_bytes()).hexdigest() == digest, rel
predict = rc.fit(report=False)
records = []
for j in inventory['jobs']:
    task, date, ep = (j[k] for k in ('task', 'date', 'episode'))
    path = STAGE_ROOT/task/'meta'/date/(ep+'.parquet')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == j['parquet_sha256'], path
    table = pq.read_table(path)
    raw = DATA_ROOT/task/date/(ep+'.h5')
    stat = raw.stat()
    assert (stat.st_size, stat.st_mtime_ns) == (j['raw_size'], j['raw_mtime_ns']), raw
    episode = open_episode(raw, task)
    assert episode.T == j['rows'], raw
    assert np.array_equal(table['source_h5_frame'].to_numpy(),
                          episode.trim + np.arange(episode.T)), path
    assert np.array_equal(table['timestamp'].to_numpy(), episode.trimmed_cam_ts), path
    with h5py.File(raw, 'r') as h5:
        for side in ('left', 'right'):
            path = OUT_ROOT/task/date/f'{ep}_{side}.npz'
            with np.load(path, allow_pickle=False) as z:
                assert not bool(z.get('review_only', False)), path
                assert int(z['pipeline_version']) == 8, path
                assert str(z['force_calibration']) == rc.CALIBRATION_NAME, path
                assert str(z['force_reconstruction']) == 'calibfree', path
                assert float(z['valid_mask_dI']) == 4, path
                assert float(z['force_calibration_max_n']) == 15, path
                assert np.isclose(float(z['force_calibration_ceiling_n']), predict.force_ceiling_n), path
                assert not bool(z['absolute_force_validated_on_react']), path
                force = z['force_normal_n'].astype(float)
                source = z['source_frame']
                assert force.shape == source.shape == (j['rows'],), path
                for key in ('volume_mm3', 'contact_area_mm2', 'max_depth_mm'):
                    assert z[key].shape == force.shape and np.isfinite(z[key]).all(), (path, key)
                assert np.isfinite(force).all() and ((force >= 0) & (force <= 15)).all(), path
                assert np.issubdtype(source.dtype, np.integer), path
                fresh = table[f'tactile_{side}_is_new'].to_numpy().astype(bool)
                intensity = table[f'tactile_{side}_intensity'].to_numpy()
                index = np.asarray(episode.align[side].index_map, int)
                assert len(index) == len(force), path
                held = np.maximum.accumulate(np.where(fresh | (np.arange(len(force)) == 0),
                                                      np.arange(len(force)), 0))
                assert np.array_equal(source, index[held]), path
                assert np.array_equal(force[1:][~fresh[1:]], force[:-1][~fresh[1:]]), path
                frames = h5[f'gelsight/{side}/frames']
                assert source.min() >= 0 and source.max() < len(frames), path
                assert np.array_equal(z['reference_rows'], _reference_rows(intensity, fresh)), path
                refs = reference_stack(frames, index, intensity, fresh)
                refs = np.stack([crop(im).astype(np.float32) for im in refs])
                reference = np.median(refs, axis=0)
                noise = reference_noise_area(refs)
                assert abs(noise-float(z['reference_noise_area_mm2'])) < 1e-9, path
                pick = sorted({0, len(force)//2, len(force)-1, int(force.argmax())})
                errors = []
                for row in pick:
                    image = crop(frames[int(source[row])]).astype(np.float32)
                    again = predict(rc.force_stages(image, reference), noise_area_mm2=noise)
                    errors.append(abs(again-force[row]))
                assert max(errors) < 1e-6, (path, errors)
                active = force[fresh]
                assert len(active), path
                records.append(dict(task=task, date=date, episode=ep, side=side,
                    rows=len(force), fresh=len(active), max_n=float(force.max()),
                    p95_n=float(np.quantile(active, .95)),
                    nonzero_fraction=float(np.mean(active > .02)),
                    ceiling_fraction=float(np.mean(active >= predict.force_ceiling_n-1e-6)),
                    checked_frames=len(pick), max_recompute_error_n=max(errors)))
            print('OK', task, date, ep, side, flush=True)
assert len(records) == 2*len(inventory['jobs'])
(run/'validation.json').write_text(json.dumps(records, indent=2))
print('Validated', len(records), 'sensor-sides; this is not a force-accuracy test')
PY
```

Acceptance requires every selected side, not a subset of completed sides.
Inspect per-task and per-side distributions; review unexpectedly all-zero
episodes, large reference-noise floors, high ceiling fractions and abrupt
changes from previous estimates. Use fresh rows for distribution statistics
so held captures do not dominate them. There is no validated universal
contact-fraction or ceiling-fraction cutoff.

The standalone `test_force_names_its_frame.py` currently defaults to an old
May episode. It is not the all-task validator for this run. Likewise,
`task_review verify` validates only its seven curated review windows.

## 6. Decide the Force-Informed Action Policy

Force reconstruction and virtual-target generation are separate operations.
The latter uses `delta_mm = F_N / k_N_per_mm` along
`R(q_row) @ [0, -1, 0]`: `press_direction()` uses the `body_y` default in
`dexforce.gel_axis()`, not the dual-ball calibrated axis. It does not measure
physical gel indentation. LUT depth is a separate field. Some existing export
descriptions still name the dual-ball axis; correct that provenance to match
the approved implementation before publishing new action columns.

The current shared constant is `dexforce.STIFFNESS_N_PER_M = 2000`, or
2 N/mm. The existing exporter treats virtual displacement beyond the 4.25 mm
gel thickness as a failed gate. Therefore any estimate above 8.5 N fails that
gate; 15 N would produce a 7.5 mm virtual displacement. This was compatible
with the old 8 N range and is not compatible with the full v8 range.

**Stop action publication until the data owner approves a consistent policy.**
Do not clip force to 8.5 N, ignore a nonzero export exit code, or silently change
stiffness. If retaining the existing gate, covering the configured 15 N range
requires k >= 15/4.25 = 3.53 N/mm; 4 N/mm would give 3.75 mm at 15 N, but is
only an example, not an approved or measured stiffness. Alternatively, the
physical interpretation of the virtual-target gate can be reviewed separately.

An approved stiffness change must be shared by the exporter, `pipeline`
utilities and preview targets, recorded in metadata, and followed by new
target/round-trip tests and preview review. Passing `--stiffness 4` only to
the exporter while previews retain 2 N/mm would make their targets disagree.
This handoff leaves the shared constant at 2 N/mm.

Export also calls `world_frame.build_declaration()`, which obtains world
offsets from `<REACT_RELEASE>/<task>/episodes.jsonl`, not `REACT_STAGE_ROOT`.
Step 1 snapshots existing indices and sets both roots to the same input tree.
Before export, require an index for every task and a `<date>/<episode>` entry
with an explicit, verified `world_frame_offset` for every selected episode.
Verify the snapshot against `inventory.json`'s `index_sha256`. Missing or stale
indices must be repaired from authoritative preprocessing records in a new
snapshot; do not accept the reader's zero-offset fallback or point it at a
changing production tree. Retain the matching calibration files as well.

After these decisions and checks, the existing commands are:

```bash
# Only after agreeing the action policy and validating input pose conventions.
python -m twm.force_recovery.export_force_columns export --root "$RUN/release_force"
python -m twm.force_recovery.export_force_columns verify --root "$RUN/release_force"
```

The source must be an uncut master without existing `force_*` columns; the
exporter appends columns and is not a replace-columns migration tool. Rebuild
from that source on every run. It writes force, F/k displacement, 7D target
pose and source-frame index for each side (eight columns total), plus sidecars.

Action export also requires the correct per-session pose convention and
`T_gel_to_rigid_<side>.json` calibration. The current declaration builder
assumes Z-up release poses. Verify the input's declared convention and actual
transform before export; do not stamp an undeclared or Y-up pose as Z-up.
Several review episodes originally lacked explicit frame declarations.
Repair them from authoritative preprocessing/rig records, not by guessing.

## 7. Rebuild Downstream Artifacts and Publish Separately

The candidate force NPZs are not a release. After acceptance:

1. Back up the old force tree and every downstream artifact being replaced.
   Keep the candidate tree and inventory for rollback. Do not delete an
   existing `_pre_promote_backup` to make a command proceed.
2. Hand the validated NPZ tree and accepted force-export tree to the release
   processor. The legacy `reprocess_react promote` enumerates old NPZs only;
   it is not a complete promotion mechanism for this inventory.
3. Rebuild Z-up metadata and segments with the new force columns. Force is
   computed on uncut episodes first, then sliced using exactly the same rows
   as observations and poses. Never renumber `force_<side>_source_frame`.
4. Rebuild per-segment indices/splits and every force-dependent preview.
   `python twm/scripts/build_release_previews.py --task pushT --stage-root "$RUN/release_cut" --overwrite --clip-s 30 --speed 1`
   requests 30-second, normal-speed previews after explicitly wiring the
   accepted force/calibration roots. The CLI also accepts `motherboard` and
   `rope`, but currently rejects `toy`: its choices use `CALIB_DIRS`. Treat
   four-task release preview automation as a downstream integration blocker,
   not a reason to omit toy. `task_review.render()` demonstrates toy rendering
   through the canonical `build_one_preview()` API with per-session calibration;
   its fixed review windows are not a full release-preview replacement.
5. Certify the final cut tree, verify the force provenance survived conversion
   and segmentation, decode the videos, and inspect samples from all four tasks.
6. Upload through the current segmented-release publisher only after approval.
   Keep the previous HF commit/branch available for rollback.

Do not blindly invoke `pipeline_stages.run_all()` for this migration. It has
fixed default roots, and its force/export commands enumerate whole trees
rather than respecting a per-task subset. The operator must route the accepted
trees explicitly. `convert_release_zup.py --force-src` takes the task-specific
force-export directory; validate its merge did not fall back to stale columns.
The preview renderer also has a default force root: changing the force writer's
environment alone does not guarantee previews read candidate force files.
For isolated previews, `build_one_preview(..., force_root=candidate_force_root)`
is the existing explicit API. Never render v8 labels over v7 values.

Do not use the older `upload_force_columns` command to push uncut episodes into
the segmented main release. Do not rerun a force-disabled release path with
`skip={'force'}` and assume the new columns were included; that also skips export.

## Handoff Checklist

- Exact code snapshot and environment record, including uncommitted source.
- Frozen input inventory, exclusions, per-task counts and expected sensor-sides.
- Hashes of all three calibration assets and the reproduced calibration report.
- Successful worker exit codes and logs with no unhandled failed sides.
- Full validation report with named-frame checks and per-task/side diagnostics.
- Reviewed full-episode samples, especially PushT weak and strong contacts.
- Explicit action-stiffness and pose-frame decision before action export.
- New downstream force metadata and previews, not cached old output.
- Backups, target HF revision, publication approval and rollback location.

The v8 seven-window comparison preserved all light-contact proxy decisions and
removed the old ceiling from 160 fresh PushT frames and 9 toy frames. This is
the review baseline, not a guarantee that every newly processed episode will
have the same contact distribution or no range saturation.
