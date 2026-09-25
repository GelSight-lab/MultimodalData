# TWM Recorder Soak Test and Episode Validator — Design

**Date:** 2026-09-06 · **Base:** `feature/twm-arducam` (after the `twm.recorder` merge) · **Branch:** `feature/twm-soak-test`

## Goal

Prove, on the attached rig, that the recorder records every stream in the
correct format, that per-stream timestamps are synchronized, and that a
10-minute episode has no dropped or late ticks. Make that proof repeatable
with two commands:

    python -m twm.recorder soak --task soak_test --duration 600 [rig flags]
    python -m twm.recorder validate <episode.h5> --expected-duration 600

## What the attached rig looks like (2026-09-06)

- RealSense: D415 `143322063538` (one of the three configured) and a D435i
  `134322071848` (not a rig camera). Both do 640x480 color+depth at 30 fps.
- GelSight Mini: `2DUPB53G` (left), `2BKRDTAD` (right).
- Arducam: `TWML0001` (left, cam0), `TWMR0001` (right, cam1).
- No ROS master, so OptiTrack is unavailable.
- Recording disk `/media/yxma/Disk1`: 239 GB free (94 % used).

## Changes

### Configuration (`config.py`, `rig.py`, `preflight.py`, `episode.py`, `schema.py`)

- `--realsense_serials A,B,...` overrides the RealSense list; the number of
  `realsense/cam{i}` groups equals the number of configured serials
  (`create_episode_file(..., n_realsense=)`, default 3 for legacy callers;
  `append_ticks` writes as many groups as exist).
- `--no_optitrack` sets `use_optitrack=False`: the rig installs
  `DummyOptitrack` (no poses, empty buffers) instead of calling the ROS
  driver, `active_sensors` becomes empty so the freshness preflight and the
  watchdog are inert, and the OptiTrack groups stay empty.

### Headless soak (`app.py::run_headless`, `__main__ soak`)

Reuses the exact production objects (`SensorRig.open`, `EpisodeWriter`,
`CaptureLoop`, `EpisodeStore`, `Recorder`) built by one shared
`build_session(config, rig)` helper that `run()` also uses. No cv2. Flow:
startup preflight → open rig → wait_ready → start episode (exit 2 on
refused preflight) → poll every 50 ms, log the health line every 10 s →
after `duration` seconds end the episode as `operator` → log the summary →
exit 0 if valid, 1 if the episode was auto-ended (the summary says why),
and the file path is printed either way.

### Validator (`validate.py`, `__main__ validate`)

`validate_episode(path, fps=30, expected_duration=None, max_tick_gap_s=0.5)
-> ValidationReport` with named checks, each `(ok, detail)`:

| check | passes when |
|---|---|
| `metadata` | `valid` is True, `ended_by` in valid endings, `frame_count == T`, `gap_count == 0` |
| `shapes` | every frame dataset has the expected dtype and (T, 480, 640[, 3]) shape; all per-tick datasets share T |
| `tick_rate` | timestamps strictly increasing; median dt within 10 % of 1/fps; max dt ≤ `max_tick_gap_s`; < 1 % of dt above 1.5/fps |
| `duration` | if expected: T ≥ 0.97 × expected × fps (warm-up drops 10) |
| `sensor_sync` | for each `gelsight/*` and `arducam/*` timestamp set: non-decreasing; lag `tick_ts − sensor_ts` in [−5 ms, 250 ms] for every tick; sensor updates at ≥ 10 distinct timestamps per second |
| `content` | 20 evenly spaced frames per stream: nonzero variance, depth has nonzero pixels, ≥ half of consecutive samples differ |
| `optitrack` | if samples exist: timestamps non-decreasing and within [first tick − 1 s, last tick + 1 s]; else "no samples" (ok when the recording ran with `--no_optitrack`) |
| `writer` | `queue_peak_fraction < 0.5`, `writer_mean_mb_s > 0` |

`ok` is the conjunction. The CLI prints the report as JSON and exits 0/1.

## Acceptance on the rig

1. `python -m twm.recorder bench --dir /media/yxma/Disk1/twm/data --seconds 5 --arducams 2` reports the sustained rate on the real disk.
2. A 600 s soak with both RealSense, both GelSights, both Arducams, `--no_optitrack`, writing to `/media/yxma/Disk1/twm/data/soak_test/`, exits 0.
3. `validate` on that file passes every check; the JSON report is saved next to the file and its numbers go in the final message.
4. The soak file is deleted afterwards (it is a test artifact on a nearly full disk); the report is kept.

## Tests

Config flags; `DummyOptitrack` in the rig; `n_realsense` in the schema;
`run_headless` with fake drivers for 0.5 s into `tmp_path` producing a valid
file that `validate_episode` accepts; validator negatives built from a
synthetic episode with an injected tick gap, a length mismatch, a frozen
stream, and a sensor lag beyond 250 ms.
