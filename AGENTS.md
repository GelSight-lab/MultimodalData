# Repository guide for coding agents

## Overview and map

This Python monorepo contains TWM handheld multimodal capture and the React
dataset pipeline, legacy Franka/Panda collection and behavior cloning, and a
separate tactile-only pretraining corpus pipeline. Do not confuse React
(`yxma/React`) with `yxma/gelsight-mini-pretrain`.

- `twm/recorder/`: capture lifecycle, single HDF5 writer, health and validation;
  `twm/data_collection.py` is the collection entry point.
- `twm/react_preprocess/`: HDF5 → videos/parquet, curation, segmentation,
  pose/action processing and publishing.
- `twm/react_toolbox/`: independently distributed dataset reader and utilities;
  keep it usable without rig hardware or the rest of TWM.
- `twm/force_recovery/`: force reconstruction, controller targets and exports.
  Read its `RUNBOOK.md` before reprocessing.
- `twm/viz.py`: projection geometry; `twm/visualize.py`: inspection CLI.
  Checkouts containing `twm/visualization/` use that shared compositor/exporter
  for collection, playback and inspection; see its README before extending it.
- `twm/calib_epoch.py`, `twm/world_frame.py`, `twm/tactile_align.py`,
  `twm/pipeline_stages.py`: shared calibration, coordinates, alignment and stages.
- `camera_stream/`, `optitrack/`, `ft_sensor/`: hardware adapters.
  `probing_panda/`: robot and legacy BC code; Hydra configs in `config/` there.
- Root `pipeline.py`, ingest scripts and `PIPELINE.md`: tactile pretraining corpus.
- `tests/` and tests within `twm/`: verification. Design/history is in
  `docs/superpowers/`; old reports are not evidence of current release status.

## Environment and setup

Run commands from the intended checkout root. First inspect `git status --short`
and `git worktree list`: worktrees can have different code and uncommitted work.

`pyproject.toml` declares setuptools packaging, Python >=3.8, NumPy, SciPy,
Hydra, OpenCV, Pillow and robot dependencies including FrankaPy. This declaration
is not a tested Python-version matrix. Prefer the existing project environment;
do not upgrade it or install robot dependencies globally as incidental cleanup.
Some checkouts declare a `twm` extra; inspect their own packaging metadata.

Offline TWM processing additionally uses h5py, hdf5plugin, PyArrow and Matplotlib;
tests need pytest and real FFmpeg/FFprobe executables. Register `hdf5plugin`
before reading BLOSC-compressed HDF5. The standalone reader's dependency manifest
is `twm/react_toolbox/requirements.txt`. Depth/BC paths have additional model,
PyTorch and checkpoint requirements; hardware SDKs are not required for pure
rendering/unit tests. GUI examples require a display.

## Common commands

Verified entry points; replace example paths with real, scoped inputs:

```bash
python -m pytest -q
python -m pytest tests/recorder -q
python -m twm.pipeline_guard
git diff --check

python -m twm.recorder validate /path/to/closed_episode.h5
python -m twm.recorder integrity /path/to/closed_episode.h5 --json
python -m twm.visualize --help
python -m twm.visualize /path/to/closed_episode.h5 --check
python -m twm.react_preprocess --help
PYTHONPATH=twm python -m react_toolbox.demo --help
```

The toolbox demo downloads Hub assets and writes a montage when actually run;
choose an episode present in its expected dataset layout rather than trusting
its historical default. Viewer `--save_video /path/to/new_preview.mp4` writes output.
Use `python -m twm.visualization --help` when that package exists.

There is no repository-wide configured formatter/linter; `pipeline_guard` checks
architectural invariants, not formatting. Preserve pytest's configured
`--import-mode=importlib`: duplicate test basenames otherwise collide.

Legacy training entry: `probing_panda/scripts/train_bc.py`, config
`probing_panda/config/bc.yaml`. Do not advertise it as turnkey:
`python -m probing_panda.scripts.train_bc --help` currently fails on
`bc_policy.py`'s beyond-top-level relative import. It also needs external FoTa,
data and checkpoints, can use GPUs, and enables W&B in its default config.
Resolve those prerequisites before training; a CLI help check is not a training test.

## Coding and architectural conventions

- Follow neighboring code: four-space indentation, snake_case functions/modules,
  descriptive names, small functions, and typed dataclasses for structured state.
  New TWM modules generally use postponed annotations and explicit package imports.
  Document array shapes, units, color order and coordinate frames.
- Reuse shared authorities instead of copying task lists, calibration epochs,
  tactile delays, stiffness, projection or index conversion rules. Keep hardware,
  model and heavy storage imports out of pure visualization helpers.
- Sensor poses are `[x,y,z,qx,qy,qz,qw]`: metres, scalar-last quaternions.
  Rotation delta is world-frame `q_next * inverse(q_current)`.
  Projection calibration translations and gel offsets use millimetres.
- Match pose and calibration world frames. Published Z-up poses require matching
  Z-up calibration; a calibration declaration alone does not establish a pose
  tree's frame. Never repeat origin shifts or Y-up→Z-up conversion.
- Scene cameras map cam0→right, cam1→left, cam2→middle; wrist slots have their own
  mapping. Raw scene/wrist images are BGR, tactile images RGB; toolbox decoding
  returns RGB. Shared rendering produces BGR copies, without changing inputs.
- Video frame i and parquet row i must correspond. Source HDF5 rows, segment-local
  rows and capture timestamps differ. Preserve provenance through slicing.
  Use canonical alignment maps (`open_episode(...).align[side].index_map`);
  timestamped tactile must not receive the legacy lag shift again.
- Short held tactile runs are normal at a slower sensor rate than the row clock.
  Missing data/force is not a measured zero. Preserve raw poses, repair provenance
  and validity masks; proposals or smoother plots do not imply training approval.
  Derived actions must not cross invalid endpoints or segment boundaries.
- Force estimation and display-only surface reconstruction are distinct. Keep
  cosmetic processing out of force inputs. Controller stiffness is defined in
  `twm/force_recovery/dexforce.py` (`STIFFNESS_N_PER_M`); use its normal-axis
  helpers, not a fixed world-axis displacement or a second stiffness literal.
- Segment only after force export and world-frame conversion. Preserve recorder
  bounded-queue/backpressure, single-writer and flush/drain guarantees.

## Safety constraints

- Defaults often point at production data under `/media/yxma/Disk1/twm/`.
  Inspect each command's paths and side effects before running it. For staged
  force work, follow the runbook's `REACT_DATA_ROOT`, `REACT_STAGE_ROOT`,
  `REACT_FORCE_RECOVERY_ROOT` and `REACT_RELEASE` configuration before imports;
  these variables do not redirect every legacy script or scheduler.
- Preprocessing `build` can repair/rewrite raw HDF5 automatically; use
  `--no-repair` for non-repair work and isolated output roots. `--force`,
  backfills, segmentation and publishing are mutations, not diagnostic checks.
  Root `pipeline.py`'s run method defaults to pushing its output.
- Never process an actively recorded HDF5, silently substitute lossy MP4 for
  missing raw measurements, or bypass calibration/validity gates. Some historical
  raw recordings no longer exist; retain explicit provenance for any fallback.
- Do not start robot motion, camera capture, USB resets, ROS/Motive connections,
  GPU jobs or system tuning just to inspect/test code. Those need the appropriate
  task scope and hardware environment.
- Keep credentials, Hub tokens, W&B secrets, recordings and large generated
  videos/checkpoints out of commits/logs. Do not publish as part of a refactor.
  For requested publication, pin the parent revision, validate staged assets and
  verify remote hashes; local updates are not proof of a completed upload.
- Preserve unrelated dirty/untracked files and other worktrees. Never use broad
  cleanup, force checkout/reset, or recursive deletion of dataset/workspace roots.
  Check free disk space before full tests or rendering: real-video tests are large.

## Verification and definition of done

Add focused regression tests for changed behavior; use synthetic fixtures and
temporary outputs rather than hardware or production data. Run relevant tests,
then the full suite and pipeline guard for shared pipeline changes. Verify media
frame content/alignment as well as counts, coordinate/units correctness, missing
modalities and unchanged raw inputs. Packaging changes also need the installed-
package tests; avoid tests that only pass because of checkout-specific paths.

Done means scoped changes reviewed, applicable checks run successfully, no
unrelated edits/data mutations, and a concise handoff stating what changed,
actual verification results and any untested hardware/network/training paths.
Report code completion, local data generation and remote publication separately.
