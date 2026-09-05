# TWM Recorder Library Design

**Date:** 2026-09-05
**Base branch:** `feature/twm-arducam` (the 1,005-line `twm/data_collection.py`)
**Implementation branch:** `refactor/twm-recorder`

## Problem

`twm/data_collection.py` holds configuration, hardware startup, the 30 Hz
capture thread, the HDF5 schema, the background writer, the GUI loop, episode
state and cleanup in one module. Three things are wrong with it beyond size:

1. **Silent data loss under load.** `HDF5Writer.enqueue` drops a whole tick
   when its 300-item queue is full and only prints every 30th drop. The
   queue holds ~2.5 GB (300 × 8.3 MB) before that happens, so a slow disk
   first eats memory and then quietly punches holes in the 30 Hz timeline.
2. **OptiTrack loses its oldest samples on long episodes.** Poses are only
   written at episode end from a 50,000-sample ring buffer per tracker;
   at ~120 Hz that is about 7 minutes.
3. **No preflight or runtime monitoring of the recording disk.**
   `/media/yxma/Disk1` is at 94 % with 239 GB free. An episode is ~15 GB
   per 5 minutes after compression; nothing checks this.

## Goals

- A package `twm/recorder/` whose modules each own one responsibility and
  can be tested without hardware.
- **Fail-fast overload policy.** The recorder never silently drops a tick.
  If the writer cannot keep up, the episode is finalized, marked invalid
  with diagnostics, recording stops, and the operator is told why.
- **Preflight** at startup (disk free, write-throughput self-test) and at
  every episode start (disk free, OptiTrack freshness, writer idle).
- **Runtime monitoring** visible in the preview: queue occupancy, write
  rate, disk free, estimated minutes of recording left, status level.
- OptiTrack samples travel with each tick and are written continuously.
- Existing callers of `twm.data_collection` keep working through a thin
  facade; the HDF5 layout stays readable by every downstream tool.

## Non-goals

- Changing sensor drivers (`camera_stream/`, `optitrack/`).
- Changing the dataset layout consumed by `react_preprocess`.
- A new GUI toolkit. The cv2 preview window stays.

## Package layout

```
twm/recorder/
  __init__.py     public re-exports
  config.py       RecorderConfig / WriterConfig / DiskConfig, parse_args()
  frames.py       Tick: one synchronized multimodal sample; nbytes()
  schema.py       create_episode_file, append_ticks, append_optitrack,
                  finalize_episode  (HDF5 layout lives here and nowhere else)
  writer.py       EpisodeWriter: byte-bounded queue, batched writes,
                  periodic flush, overload/fault detection, WriterStats
  rig.py          SensorRig: opens/closes hardware in order, grab() -> Tick
  capture.py      CaptureLoop: strict-rate thread, warm-up, gap detection,
                  publishes CaptureSnapshot, raises stop requests
  preflight.py    CheckResult checks: disk free, write bandwidth, OT fresh,
                  writer idle
  monitor.py      health_line(): one status string + level for the preview
  episode.py      EpisodeStore (paths, numbering, CSV log), EpisodeSummary
  app.py          Recorder controller (state machine) + run() cv2 loop + main
twm/data_collection.py   facade: constants, schema functions, HDF5Writer
                          adapter, make_preview, main -> twm.recorder.app.main
```

## Data flow

```
SensorRig.grab()  ──Tick──▶  CaptureLoop (30 Hz thread)
                                │  submit(f, tick)          raises WriterOverloaded
                                ▼                           / WriterFault
                          EpisodeWriter (thread)  ──batch──▶ schema.append_ticks ──▶ HDF5
                                │ stats()
                                ▼
GUI thread: Recorder.poll(snapshot) ──▶ start/end episode, watchdog, preview
```

A `Tick` is a frozen dataclass: timestamp, 3 color, 3 depth, 2 GelSight
frames + capture timestamps, 0 or 2 Arducam frames + timestamps, and the
OptiTrack samples drained since the previous tick. It replaces the
positional tuples `b[0]…b[6]`.

## Overload policy (fail-fast)

| Condition | Detector | Action |
|---|---|---|
| Queue cannot accept the next tick (bytes) | `EpisodeWriter.submit` raises `WriterOverloaded` | CaptureLoop stops enqueueing, publishes `stop_request(kind="overload")` |
| Queue > `overload_fraction` (0.5) for `overload_sustained_s` (3 s) | `EpisodeWriter.check()` | same, before the buffer is full |
| Write raises (disk full, I/O error) | writer thread records `fault` | `submit` raises `WriterFault`; `stop_request(kind="writer_fault")` |
| Disk free < `min_free_gb` (50 GB) | writer samples disk at every flush | `stop_request(kind="disk_low")` |
| Gap between consecutive ticks > `max_tick_gap_s` (0.5 s) while recording | CaptureLoop | `stop_request(kind="capture_stall")` |
| Sensor raises during grab | CaptureLoop | `fatal_error`; recorder finalizes then exits |

`Recorder.poll` sees the stop request on the GUI thread and calls
`end_episode(ended_by=kind, reason=detail)`. The episode is finalized like a
normal one, plus `metadata.attrs["valid"] = False` and
`metadata.attrs["invalid_reason"]`, and the CSV log row gets
`notes = "INVALID: <reason>"`. Nothing is ever dropped from the middle of a
timeline. Recording stops; the operator presses `s` to start a new episode.

Queue capacity is measured in **bytes**, default `queue_seconds = 3.0`
(90 ticks ≈ 746 MB), instead of 300 items ≈ 2.5 GB.

## Preflight

Startup (`run_startup_preflight`):
- disk free ≥ `min_free_gb`;
- write-bandwidth self-test: the real writer writes `bandwidth_test_s`
  (2 s) of synthetic ticks to a temp file in the data directory and must
  achieve ≥ `fps × min_bandwidth_margin` (30 × 1.5 = 45 ticks/s). Skipped
  with `--no_bandwidth_test`.

Episode start (`run_episode_preflight`):
- disk free ≥ `min_free_gb`;
- every active OptiTrack body has a sample newer than
  `ot_preflight_max_age_s` (2 s);
- writer queue empty and no fault.

A failed preflight prints every failing check and refuses to start.

## Runtime monitoring

`WriterStats` (taken under the writer lock): queue items/bytes/fraction,
peak fraction this episode, mean raw MB/s, last batch ms, flushes, file
MB/s, disk free GB, fault, overload-since. `monitor.health_line` renders
`[writer 12% | 96 MB/s | disk 239 GB (~48 min) | OK]` with level
`ok` / `warn` / `fail`; the app draws it on the preview and the capture
thread logs it every 60 ticks.

## Episode metadata added

`metadata.attrs`: `frame_count`, `duration_s`, `valid`, `invalid_reason`,
`ended_by` (`operator` | `quit` | `watchdog` | `overload` | `writer_fault`
| `disk_low` | `capture_stall` | `sensor_error`), `max_tick_gap_s`,
`gap_count`, `queue_peak_fraction`, `writer_mean_mb_s`, `ended_at`.

## Backward compatibility

`twm.data_collection` keeps exporting `REALSENSE_SERIALS`,
`GELSIGHT_SERIALS`, `DATA_DIR`, `FPS`, `create_episode_file`,
`append_camera_frame`, `append_camera_frames_batch`,
`flush_optitrack_to_hdf5`, `log_episode`, `next_episode_number`,
`make_preview`/`make_optitrack_panel`/`TRACKER_COLORS`, and an `HDF5Writer`
adapter with the old `enqueue(...)` keyword signature. The adapter never
drops: it raises `WriterOverloaded`, and `dropped_frames` is always 0.
`twm.sensor_camera.record_verification` and
`twm/scripts/test_crash_leaves_readable_h5.py` move to the new API.

## Testing

Unit tests per module with in-memory fakes (no hardware, no cv2):
writer overload/fault/flush timing with an injected slow sink; capture
loop warm-up, gap detection, overload stop request, fatal error; rig
startup ordering and reverse cleanup; preflight checks with injected
`disk_usage`; recorder state machine (start, end, auto-end, watchdog,
quit-while-recording). The existing crash test keeps proving that periodic
flush leaves a readable file. Full suite must stay green.
