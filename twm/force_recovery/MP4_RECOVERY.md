# V8 recovery from released tactile MP4, September 18, 2026

The updated `old_data` force covers 36 episodes: 32 motherboard episodes from
May 10, 11 and 19, and four pushT episodes from June 18 (72 sensor sides).
The May raw H5 recordings were deleted, so this run uses the released tactile
H.264 videos with explicit user authorization. The V8 estimator and calibration
are unchanged. These estimates are **not equivalent to raw-H5 V8**: compression
can materially change the force, and a prior single-frame comparison reached
a 6.48 N difference. Absolute force accuracy has not been validated on React.

Every parquet force field, its `.force.json` sidecar and the run manifest name
`source_format=release_tactile_mp4_h264` and `lossy_input=true`. A source-frame
value is the zero-based decoded **MP4 frame index**, not an H5 frame index.
Video frame i corresponds to parquet row i. Frames marked `is_new=false` hold
the preceding evaluated estimate and source-frame index.

Motherboard receives normal force and source-frame columns for both sides.
The existing pushT penetration and target-pose columns are also recomputed from
the new force at their previously declared controller stiffness, 2 N/mm, with
the existing sensor-body -Y pressing direction. F/k is a controller's virtual
deflection, not measured gel indentation. All nonforce columns, including
repaired poses and actions, are preserved by the force exporter.

## Reproduce

Download `old_data/motherboard/v8_rebuild_tools/**` from the dataset. Its original
README describes the exact raw-H5 workflow; use these instructions for the MP4
recovery instead. Install the package's `requirements-force-v8.txt` and
`av==15.1.0`. The two additional modules shipped alongside that package are
`twm.force_recovery.run_episode_mp4` and
`twm.force_recovery.finalize_mp4_recovery`.

Arrange the 36 old_data parquets and 72 tactile videos below one input root,
preserving the structure after `old_data/`, for example
`input/motherboard/meta/2026-05-10/episode_000.parquet` and
`input/motherboard/videos/2026-05-10/episode_000/tactile_left.mp4`.
From the downloaded `v8_rebuild_tools` directory:

```bash
export REACT_STAGE_ROOT=/absolute/path/to/input
export REACT_FORCE_RECOVERY_ROOT=/absolute/path/to/run/force
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
mkdir -p "$REACT_FORCE_RECOVERY_ROOT"
cp -a assets/. "$REACT_FORCE_RECOVERY_ROOT/"
python -m twm.force_recovery.run_episode_mp4 0 1
```

For eight workers run indices 0 through 7, each with total workers 8, in separate
processes. Repeating a worker resumes completed outputs. Do not run the same
side concurrently. The runner checks decoded video length against parquet rows.

Download the latest remote metadata before exporting, preserving newer actions:

```bash
python -m twm.force_recovery.finalize_mp4_recovery \
  --base-root /absolute/path/to/run/latest_meta --download-base
python -m twm.force_recovery.finalize_mp4_recovery \
  --base-root /absolute/path/to/run/latest_meta \
  --force-root "$REACT_FORCE_RECOVERY_ROOT" \
  --alignment-root "$REACT_STAGE_ROOT" \
  --output /absolute/path/to/run/release_force
```

The finalizer validates all 72 artifacts, frame alignment, finite nonnegative
force and geometry, calibration identity/ceiling, held duplicate rows and
reference frames. It preserves nonforce columns and checks parquet round trips.
It does not upload. Publishing must guard the current HF revision and merge any
metadata updates made since the downloaded snapshot, so newer actions survive.
