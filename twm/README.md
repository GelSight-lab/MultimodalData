# TWM Data Collection

Tools for collecting and reviewing multimodal data for the Tactile World Model
(TWM) project: a 30 Hz recorder for the full rig (3 RealSense, 2 GelSight,
2 Arducam wrist cameras, OptiTrack), an unattended soak test, an episode
validator, a per-stream integrity check, calibration tools, and a viewer.

## Quick start

```bash
# terminal 1 — OptiTrack poses over VRPN (skip and pass --no_optitrack to record without poses)
roslaunch vrpn_client_ros sample.launch

# terminal 2 — record; run from a checkout that contains twm/recorder/
cd ~/MultimodalData
python -m twm.data_collection --task <task_name>
#   s = start episode   e = end episode   r = reset GelSight diff reference
#   p = toggle projection overlay   q = quit (saves an in-progress episode)

# afterwards — check any episode
python -m twm.recorder validate  /media/yxma/Disk1/twm/data/<task>/<date>/episode_000.h5 --expected-duration <seconds>
python -m twm.recorder integrity /media/yxma/Disk1/twm/data/<task>/<date>/episode_000.h5   # frame rate + lost frames per stream

# replay it (add --cam_calib motherboard|pushT when the task has no calibration of its own)
python -m twm.visualize /media/yxma/Disk1/twm/data/<task>/<date>/episode_000.h5 --check
```

If you ever see black GelSight images or no wrist cameras in the preview, you
are running an old checkout without `twm/recorder/` (its hard-coded serials
fall back to black dummy frames). Pull the branch that has it.

---

## Hardware

| Sensor | Serials / identity | Where configured |
|--------|--------------------|------------------|
| 3× Intel RealSense D415, 640×480 color + depth @ 30 Hz | `143322063538` (right, cam0), `104122062574` (left, cam1), `217222066989` (middle, cam2) | `REALSENSE_SERIALS` in `twm/recorder/config.py` |
| 2× GelSight Mini, 640×480 @ ~18 Hz | `2DUPB53G` = left, `2BKRDTAD` = right | `GELSIGHT_SERIALS` in `twm/recorder/config.py` |
| 2× Arducam B0578 sensor-mounted RGB, 640×480 MJPEG @ 30 Hz | `TWML0001` = left → `arducam/cam0`, `TWMR0001` = right → `arducam/cam1` | `twm/config/arducam.json` (selected by USB serial, any port) |
| OptiTrack rigid bodies | `motherboard`, `sensor_left`, `sensor_right` via VRPN/ROS | `OT_TRACKERS` in `twm/recorder/config.py` |

Every other threshold (queue size, fail-fast limits, disk minimums, warm-up
frames) lives in `RecorderConfig` in `twm/recorder/config.py`, and each has a
command-line override (`--data_dir`, `--queue_seconds`, `--min_free_gb`,
`--bandwidth_margin`, `--no_bandwidth_test`, `--realsense_serials`,
`--no_optitrack`, `--no_arducam`, `--arducam_config`, `--no_projection`).

### Two rig rules learned the hard way (2026-09-06)

1. **Do not put the Arducams on the USB 2.0 hub at port 1-12 with the
   GelSights.** There they alternate between "no frames" and
   `VIDIOC_STREAMON: Protocol error` at the driver level. On the PC's own
   ports (currently 1-1.3 and 1-2) they stream for hours.
2. **The recording disk is slower than the rig.** `/media/yxma/Disk1` (HDD)
   sustains ~100 MB/s of real writes; the full rig stores ~4.8 MB per tick
   ≈ 145 MB/s. The kernel's write cache bridges the gap for a whole episode
   only if it is allowed to grow. Set once, and keep across reboots:

   ```bash
   sudo sysctl -w vm.dirty_ratio=60 vm.dirty_background_ratio=5
   echo -e "vm.dirty_ratio=60\nvm.dirty_background_ratio=5" | sudo tee /etc/sysctl.d/90-twm-recorder.conf
   ```

   With that, a 600 s full-rig episode (85 GB) recorded with a 24 % writer-queue
   peak and 14 GB of dirty pages. Without it the default 20 % budget runs out
   around 8–9 minutes and the episode is ended (explicitly, see below). Between
   long episodes give the disk a few minutes to drain.

---

## Prerequisites

### OptiTrack stream (required for pose data)

OptiTrack poses are streamed over VRPN. Before launching the recorder, start
the VRPN client in a separate terminal:

```bash
roslaunch vrpn_client_ros sample.launch
```

This connects to the Motive server and publishes pose topics for the
`motherboard`, `sensor_left`, and `sensor_right` rigid bodies. Motive on the
OptiTrack PC must be open and tracking the bodies. Without a ROS master the
recorder refuses to start unless you pass `--no_optitrack`, which records
every camera and leaves the pose datasets empty.

The recorder will not start an episode while an active body has no fresh
pose (`--active_sensors both|left|right`, default both), and it auto-ends an
episode after 10 s of OptiTrack silence.

---

## Collecting Data

```bash
python -m twm.data_collection --task <task_name>
```

`--task` is required and controls where data is saved:

```
/media/yxma/Disk1/twm/data/<task_name>/<YYYY-MM-DD>/episode_000.h5
                                                    episode_001.h5
```

A row is appended to `<data_dir>/dataset_log.csv` after every episode.

### The preview window

Three rows: the RealSense color views with the GelSight projection overlay,
the two GelSight images with their contact-difference thumbnails, and the
two wrist cameras labelled by slot, serial and side.

The overlay marks each GelSight's surface centre and body axes on every
calibrated RealSense view, using the current rig calibration in
`twm/calibration/result/` (the pushT epoch, 2026-06-26). It is built for
low latency: the thumbnails are rebuilt when a new tick arrives (30 Hz), and
the overlay is redrawn on every GUI frame from the newest OptiTrack pose
rather than the pose sampled at the tick, so the dot trails the sensor by
one GUI frame (about 33 ms plus display) instead of tick + preview delay.
The overlay is preview-only; nothing recorded depends on it. Press `p` to
toggle it, or start with `--no_projection`.

**CPU is the rig's scarcest resource.** The PC has 4 physical cores (8 with
hyperthreading). A py-spy profile of a full-rig recording on 2026-09-09
(motherboard episode 003, overloaded at 13.7 s) puts the demand at roughly:
RealSense readers ~1 core (depth-to-color alignment plus copies, three
cameras), GelSight decoders ~0.9 (8 MP MJPEG each at 18.75 fps), Arducam
readers ~0.3, the writer's compression ~1, the window ~0.6 (`imshow` of a
1280x768 panel is a third of it), capture ~0.2; the desktop (Chrome, VS
Code, Remmina, gnome-shell) added ~1.2. That is more than the machine has,
and the writer is what loses: its compression ran at 183 MB/s raw against
the ~250 MB/s the full rig produces. The window now refreshes at 15 Hz
(`gui_fps`), which halves its cost; the overlay then trails the sensor by
about 66 ms. Before recording, close what you do not need, and watch the
`queue` percentage in the `[REC]` log lines: it must stay well under 50 %.
Recording without the window (`python -m twm.recorder soak`) removes the
window's share entirely, and `--raw_depth` removes the SDK's depth
alignment (see [RealSense depth](#realsense-depth-aligned-now-or-later)). If the dots sit off the sensors,
recalibrate (see [Camera Calibration](#camera-calibration)); do not record
through a stale calibration expecting to fix it later.
A black text strip under the images carries the status bar (episode state)
and, below it, the health line
`writer <queue %> | <MB/s> | disk <GB free> (~min left) | OK/WARN/FAIL`;
neither is drawn over a camera image. The `MB/s` figure counts raw
(uncompressed) tick bytes; the full rig is about 8.3 MB per tick, so the
writer must sustain about 250 MB/s raw to keep up at 30 Hz.
Watch the queue percentage: it is the one number that says whether the disk
is keeping up.

### Controls

| Key | Action |
|-----|--------|
| `s` | Start a new episode (refused, with reasons, if a preflight check fails) |
| `e` | End the episode and save |
| `r` | Reset the GelSight diff reference to the current frame |
| `p` | Toggle the GelSight→camera projection overlay |
| `q` | Quit (saves an in-progress episode first) |

### Typical workflow

1. Start the script. All sensors initialize; the preview opens once every camera
   has delivered frames and settled.
2. Position the setup. Press `r` so the GelSight difference thumbnails are grey.
3. Press `s`. The first 10 ticks are discarded (camera warm-up), then recording
   runs at a strict 30 Hz.
4. Perform the task.
5. Press `e`. The writer drains, OptiTrack poses are flushed, the file is
   stamped valid and closed, and the summary is logged.
6. Repeat from 2, or `q` to quit.

### Sensor stalls (GelSight "Restarting the camera")

A GelSight whose USB link hiccups stops delivering frames; its driver prints
`Frame is not updated ... Restarting the camera [left serial=...]` and reopens
it (about 3 s). The recorder handles this without stopping:

- the capture thread never waits on a GelSight or Arducam. While a stream is
  stalled the tick still runs at 30 Hz and records that sensor's **last frame
  with its old capture timestamp**, so the gap is visible in
  `gelsight/<side>/timestamps` and every other stream is untouched;
- a supervisor thread restarts the stalled stream in the background (the
  restart used to run on the capture thread and, because of a driver bug,
  froze the whole recorder after the first one);
- the health line shows `gelsight_left STALE 3.2s (restarting, 1 restarts)`
  and turns yellow; the episode file stays open and valid, and
  `metadata.attrs["sensor_restarts"]` records how many restarts happened
  during the episode;
- `validate` reports such gaps as sensor outages (`sensor_outage_count`,
  `sensor_outage_longest_s`) and still passes the episode unless outages
  cover more than 5 % of its ticks or the sensor never came back.

If the restarts are frequent, the USB link is the problem (see rule 1 above).

### What is guaranteed, and what happens when it cannot be

The recorder never drops a frame from the middle of an episode. Every tick
either reaches the file or ends the episode explicitly:

| Condition | Result |
|-----------|--------|
| Writer queue (3 s of ticks, ~750 MB) full, or above 50 % for 3 s | episode ended, `ended_by=overload` |
| HDF5 write error | `ended_by=writer_fault` |
| Free disk below 50 GB | `ended_by=disk_low` |
| Two ticks more than 0.5 s apart, or no tick published for 2 s | `ended_by=capture_stall` |
| A sensor raises while grabbing | `ended_by=sensor_error`, recorder exits |
| OptiTrack silent for 10 s | `ended_by=watchdog` (episode stays valid) |

An auto-ended episode is written to disk intact up to the last good tick, gets
`metadata.attrs["valid"] = False` and an `invalid_reason`, and its CSV row says
`notes = "INVALID: ..."`. Recording stops; press `s` for a new episode.

Before every episode the recorder checks free disk, OptiTrack freshness for the
active bodies, that the writer is idle, and that the capture thread is alive.
At startup it also pushes two seconds of synthetic frames through the real
pipeline. That bandwidth self-test is advisory: it measures the page cache and
under-reads a slow HDD, so a shortfall against `fps × 1.5 × arducam-scale`
ticks/s is logged as a warning and recording proceeds (`--bandwidth_margin`
tunes it); only low free disk refuses to start. Run it by hand with

```bash
python -m twm.recorder bench --dir /media/yxma/Disk1/twm/data --seconds 5
```

---

## Verifying the pipeline

### Soak test (unattended timed recording, no window)

```bash
python -m twm.recorder soak --task <task_name> --duration <seconds> \
  [--data_dir D] [--no_optitrack] [--no_arducam] \
  [--realsense_serials A,B,C] [--bandwidth_margin M] [--min_free_gb G]
```

Same preflight and recorder as the window; records for `--duration` seconds
(or until an auto-end), prints `episode file: <path>`, logs a health line
every 10 s, and exits `0` (valid episode), `1` (auto-ended or short), or `2`
(refused to start). Ctrl-C finalizes the episode as `quit` before exiting.

| Flag | Effect |
|------|--------|
| `--realsense_serials A,B,C` | Record only these RealSense cameras; the file gets that many `realsense/cam{i}` groups |
| `--no_optitrack` | No ROS needed; pose datasets stay empty; the OptiTrack watchdog and freshness check are inert |
| `--no_arducam` | Legacy collection without the wrist cameras |
| `--bandwidth_margin M` | Advisory self-test threshold `fps × M × arducam-scale` (default `M=1.5`) |
| `--min_free_gb G` | Refuse to start, and auto-end an episode, below `G` GB free (default 50) |

### Validate an episode

```bash
python -m twm.recorder validate <episode.h5> --expected-duration <seconds> \
  [--fps N] [--warmup-frames 10] [--max-tick-gap 0.5] [--report out.json]
```

`--expected-duration` is the length you meant to record; the episode's own
length is in the CSV log (`duration_s`). Leave it off to skip the `duration`
check and still run the other seven.

Prints a JSON report and exits `0` only if all eight checks pass:

| Check | Passes when |
|-------|-------------|
| `metadata` | `valid=True`, `ended_by` ∈ {operator, quit, watchdog}, `frame_count == T`, `gap_count == 0` |
| `shapes` | Every stream promised by the metadata (RealSense serials, GelSight serials, Arducam config) exists with the right shape and dtype for all `T` ticks |
| `tick_rate` | Timestamps strictly increasing; median interval within 10 % of `1/fps`; max gap ≤ `--max-tick-gap`; fewer than 1 % of ticks late (count reported) |
| `duration` | `T ≥ 0.97 × (expected_duration × fps − warmup_frames)` |
| `sensor_sync` | Each GelSight/Arducam timestamp stream is non-decreasing, is not a copy of the tick clock, updates at ≥ 10 Hz, lags the tick by at most 250 ms on 99 % of ticks and never more than 500 ms |
| `content` | Sampled frames are not blank, not frozen, and depth is not all zero |
| `optitrack` | Pose timestamps non-decreasing and within the episode's span (± 1 s); ok when recorded with `--no_optitrack` |
| `writer` | `queue_peak_fraction < 0.5` and `writer_mean_mb_s > 0` |

### Check integrity: frame rate and lost frames per stream

```bash
python -m twm.recorder integrity <episode.h5>            # table, exit 0 when nothing is lost
python -m twm.recorder integrity <episode.h5> --json     # same as JSON
python -m twm.visualize <episode.h5> --check             # the table, then playback
```

`validate` answers "does this episode meet the recorder's contract". This
answers the simpler question stream by stream: how many frames, at what
rate, and were any dropped.

```
stream                  frames  rate Hz  native Hz  median dt    max dt  lost  held  status
timestamps                1052    29.76          -    33.5 ms   45.7 ms     0     -  ok
realsense/cam0/color      1052    29.76          -          -         -     0     -  ok
gelsight/left             1052    29.76       16.9    53.4 ms  113.3 ms    67   456  LOST 67
arducam/cam0              1052    29.76       29.7    32.3 ms   67.8 ms     3     3  ok
optitrack/sensor_left     3535   100.03          -    10.0 ms   20.5 ms     0     -  ok
INTEGRITY FAILED: ...  (T=1052, 30 fps)
  problem: gelsight/left: 67 lost frame(s)
  warning: optitrack/motherboard: no samples (rigid body not broadcast)
```

How to read it:

- **frames** is the dataset length; every frame dataset must have exactly
  `T` entries (a short one is reported as `1000/1052`).
- **rate Hz** is samples per second of the stored stream. For the tick-sampled
  sensors that is the tick rate; **native Hz** is how often the sensor itself
  produced a new frame. A GelSight runs at about 17 Hz, so about every second
  tick stores the previous tactile frame again: that is **held**, not lost.
- **lost** is counted, not guessed from single gaps, because sensor clocks
  jitter (OptiTrack delivers packets in bursts; a late Arducam frame is
  followed by an early one). Expected = span / the stream's typical period
  (its declared fps when it has one, else the densest interval in its
  histogram), capped at one frame per tick for GelSight and Arducam; lost =
  expected minus delivered. The tick clock is the recorder's own scheduler and
  barely jitters, so a tick is lost when an interval reaches 1.5/fps.
- A stream passes when lost is at most 0.5 % of expected, or 2 frames for
  short clips. The count is always shown, so a tolerated hiccup stays visible.
- Warnings do not fail the check: an empty tracker (rigid body not
  broadcast), tracker data that does not cover the tick span, or a single
  long gap inside the budget.

On this rig (2026-09-08 test episodes): ticks, RealSense, Arducams and
OptiTrack are clean; the GelSights skip about 10 % of their own frames
(intervals of 100 to 150 ms between new frames with no short neighbour). That
is the sensor/driver, not the recorder; the tactile timestamps record exactly
which ticks carry a new frame.

### Measured on this rig (2026-09-06/07)

| Configuration | Result |
|---------------|--------|
| 3 RealSense + 2 GelSight, 600 s, default `vm.dirty_ratio=20` | 17,839 frames, 29 late ticks, queue peak 13 %, ~3.9 MB/tick, all checks green |
| Full rig incl. 2 Arducam, 600 s, `vm.dirty_ratio=60` | 17,839 frames, 31 late ticks, max gap 121 ms, queue peak 24 %, 85 GB, GelSight lag 18–183 ms @ 17 Hz, Arducam lag ≤ 46 ms @ 29.7 Hz, all checks green |
| 3 RealSense + 2 GelSight, 600 s, old byte-shuffle compression | overloaded at 521 s, ended INVALID with 15,641 intact frames (the fail-fast working as designed) |

Short benches read 250–900 MB/s because the page cache absorbs them; trust a
multi-minute soak's `queue_peak_fraction`, not a bench number.

### Wrist-camera tools

`python -m twm.sensor_camera identify` shows both Arducam feeds labelled by
slot, serial and side (`0`/`1` choose the left slot, `u` unknown, `s` save,
`q` quit). `python -m twm.sensor_camera verify --duration 5 --output
/tmp/twm_arducam_verification.h5 --force` records both Arducams through the
production writer and validates the file.

### Library layout

`twm/recorder/`: `config` → `rig` (hardware, ordered start/stop) → `capture`
(strict-rate thread) → `writer` (byte-bounded HDF5 thread) with `schema`
(the only place that names a dataset), `preflight`, `monitor`, `episode`
(paths + CSV log), `validate`, `integrity`, and `app` (episode state machine, window,
headless soak). `twm/data_collection.py` is a thin compatibility facade.

Before merging any change under `twm/`, run `python -m twm.pipeline_guard`
(must print `14 checks, 0 violation(s)`) and `python -m pytest tests -q`.

---

## Visualizing Episodes

```bash
python -m twm.visualize path/to/episode_000.h5                 # play at the recorded FPS
python -m twm.visualize path/to/episode_000.h5 --fps 15        # slower playback
python -m twm.visualize path/to/episode_000.h5 --save_video ep.mp4   # export instead of a window
python -m twm.visualize path/to/2026-09-08 --save_videos       # one mp4 next to every .h5 in a folder
```

The window has three rows: the RealSense color views with the OptiTrack
poses, the two GelSights (raw and difference against a reference), and, when
the episode has them, the two Arducam wrist cameras (`cam0 left`, `cam1
right`). Episodes recorded without wrist cameras get two rows. Exported mp4s
follow the same layout (1280x480 or 1280x720).

Add `--check` to print each stream's frame rate and lost-frame count before
playback (the same report as `python -m twm.recorder integrity`, below).

### Calibration: which overlay, and when it is needed

By default the viewer projects each GelSight's centre onto the RealSense views
(see [Camera Calibration](#camera-calibration)). That needs the extrinsics of
the **epoch the episode was recorded in**, which the viewer picks from the task
name in the path (`twm/calib_epoch.py`, `CALIB_DIRS`):

| task in the path | calibration folder | epoch |
|---|---|---|
| `motherboard` | `twm/calibration/result backup/` | 2026-05-12 |
| `pushT` | `twm/calibration/result/` | 2026-06-26 |

Any other task name (for example `test`) stops with
`cannot infer task from '...'`. It refuses on purpose: viewing through the
wrong epoch silently draws the dots in the wrong place. You then have three
choices:

1. **No overlay** (test recordings, quickest):

   ```bash
   python -m twm.visualize path/to/test/2026-09-08/episode_000.h5 --no_projection
   ```

2. **Pick the epoch by task name.** `--cam_calib` accepts a task name and
   then supplies all five files of that epoch:

   ```bash
   python -m twm.visualize path/to/test/2026-09-08/episode_000.h5 --cam_calib motherboard
   ```

   Those extrinsics are only right if the cameras and the OptiTrack origin
   have not moved since that epoch. Dots landing off the sensors mean the
   calibration is stale, not the recording. Individual files can still be
   overridden with explicit paths (`--cam_calib a.json b.json c.json`,
   `--gel_left`, `--gel_right`); anything not given comes from the same epoch.

3. **A new task**: record under its own task name, calibrate (below), and add
   the task to `CALIB_DIRS` so the viewer finds it automatically.

### Controls

| Key | Action |
|-----|--------|
| `space` | Pause / resume |
| `→` / `d` | Next frame (pauses) |
| `←` / `a` | Previous frame (pauses) |
| `1` … `6` | Playback speed 1x, 2x, 5x, 10x, 25x, 50x |
| `l` | Toggle looping at the end of the episode |
| `r` | Reset GelSight diff reference to current frame |
| `q` | Quit |
| Frame slider | Seek anywhere in the episode |

---

## Camera Calibration

The live preview (`data_collection.py`) and the viewer (`visualize.py`) can
overlay each GelSight sensor's projected position onto the overhead RealSense
views. This requires a per-camera **OptiTrack → camera** extrinsic calibration
(`T_mocap_to_cam`). The full projection chain is:

```
P_gel_rigid  = gel_center_in_rigid_mm                (GelSight → rigid-body calib)
P_gel_mocap  = T_rigid_to_mocap @ P_gel_rigid        (live OptiTrack pose × 1000)
P_gel_cam    = T_mocap_to_cam   @ P_gel_mocap        (camera-view calib, below)
(u, v)       = project(K, P_gel_cam)                 (camera intrinsics K)
```

Calibration files live in `twm/calibration/result/`:

| File | What it maps |
|------|--------------|
| `T_mocap_to_cam_left.json`   | OptiTrack → **left** camera (serial `104122062574`, cam idx 1) |
| `T_mocap_to_cam_middle.json` | OptiTrack → **middle** camera (serial `217222066989`, cam idx 2) |
| `T_mocap_to_cam_right.json`  | OptiTrack → **right** camera (serial `143322063538`, cam idx 0) |
| `T_gel_to_rigid_left.json`   | GelSight surface center in the left rigid-body frame |
| `T_gel_to_rigid_right.json`  | GelSight surface center in the right rigid-body frame |

Each `T_mocap_to_cam_*.json` also stores that camera's intrinsics (`K`), RMSE,
and the raw point pairs used to solve it. The viewer/preview load these
automatically by matching each file's `camera_serial` to `REALSENSE_SERIALS`.

### When to recalibrate

- **A camera moved** (bumped, remounted, or re-aimed) → recalibrate that view.
- **The OptiTrack origin / ground plane was re-set** in Motive → recalibrate all views.
- The GelSight `T_gel_to_rigid_*` only needs redoing if the sensor's mounting on
  its rigid body changes.

### How it works

Click-to-calibrate using a reflective ball visible to **both** OptiTrack and the
camera. For each point you click the ball in the camera image and provide the
ball's OptiTrack position; with ≥4 (ideally ≥6–8) non-coplanar points a transform
is solved per camera and saved as JSON + NPY.

Two solvers are available (`--method`):

| Method | How | Notes |
|--------|-----|-------|
| `pnp` (default) | **2D↔3D**: clicked pixel + mocap point + factory color intrinsics, via `solvePnPRansac` + LM refine | **Depth-free** — avoids RealSense depth noise (the dominant error in `svd`) and minimizes pixel reprojection error directly. RANSAC auto-rejects bad clicks. Reports RMSE in **px**. |
| `svd` | **3D↔3D**: deproject the click with depth, then Arun SVD | Original method; sensitive to depth noise. Reports RMSE in **mm**. Kept for comparison. |

> Requires the OptiTrack VRPN stream running (see [Prerequisites](#prerequisites))
> so you can read each ball position from Motive.

**Auto-capture the mocap position** (skip typing) with `--mocap_body NAME`: the
ball's position is grabbed automatically from the live VRPN stream
`/vrpn_client_node/NAME/pose` when you advance a point. Requires the ball to be
tracked as a rigid body named `NAME` in Motive.

**Marker detection** — by default (`--source ir`) markers are detected in the
RealSense **infrared** stream (emitter on). OptiTrack markers are retroreflective
and built for IR: they appear as the brightest spots, the background is
suppressed, and — unlike RGB — the IR response is uniform across all three
cameras. Detection keys on **local contrast** (a white top-hat), not absolute
brightness or color, so it's robust to lighting gradients across the workspace.
The camera windows show the **RGB image** with each detected marker overlaid at
its mapped color pixel. Because the IR and color imagers are physically offset
(~few-cm baseline), each IR detection is transformed IR-pixel → color-pixel via
depth + factory IR→color extrinsics before being drawn/used — without this the
markers would sit ~15 px off in RGB. So detection is robust (IR) while display and
calibration stay in the color frame. Use `--source color` to detect directly in
RGB instead (far less reliable for these markers — verified on the rig: RGB found
2–5 of 8, IR is stable per camera).

> Detection is stable but each camera only sees the markers not occluded by the
> tracker bodies from its angle (e.g. one view may see all 8, another 5–7). The
> per-camera-optional-click workflow handles this — each camera just needs ≥4.
> Any marker the detector misses can still be hand-clicked (snap refines locally).

**Sub-pixel snapping** (on by default): detected markers are ringed in cyan; a
click **snaps** to the nearest one and refines to its intensity-weighted sub-pixel
center. Each window has a **`thresh` slider** (top-hat contrast), seeded per
camera (IR≈25 fixed; RGB adaptive). Drag it until markers are ringed; press **`m`**
to view the threshold mask, or **`-`/`=`** to nudge all sliders. Other knobs:
`--snap_radius`, `--blob_min_area`, `--blob_max_area`, `--no_snap`, `--blob_thresh`.

### Calibrate all cameras at once (recommended)

Place the ball once, click it in whichever camera views can see it, and type its
OptiTrack position **once** — shared across all those cameras. Each camera
accumulates its own points (≥4 each), so a ball out of frame for one camera just
doesn't contribute to it.

```bash
python -m twm.calibration.mocap_to_cam_multi --num_points 8                 # PnP (default)
python -m twm.calibration.mocap_to_cam_multi --num_points 10 \
    --mocap_body calib_ball                                                 # PnP + auto mocap
python -m twm.calibration.mocap_to_cam_multi --method svd                   # old depth-based solver
```

One live window opens per camera. Per-point controls (focus any window):

| Key | Action |
|-----|--------|
| left-click | Mark the ball in that camera view (re-click to move it) |
| `n` / `space` | Done with this point → enter its OptiTrack position once |
| `r` | Clear this point's clicks and re-click |
| `u` | Undo the last recorded point (all cameras) |
| `q` | Finish: solve + save with the points collected so far |

Solves and writes all three `T_mocap_to_cam_*.json` (+ `.npy`) at once, printing
per-camera RMSE and per-point residuals (✅ <5, ⚠️ <10, ❌ ≥10; in **px** for
`pnp`, **mm** for `svd`).

### Calibrate a single camera

```bash
python -m twm.calibration.mocap_to_cam --serial 217222066989 --num_points 6 \
    --output twm/calibration/result/T_mocap_to_cam_middle.json
```

Same click-then-type flow for one camera; controls: `y` accept · `r` redo ·
`q` quit. Use the output filename matching the camera's view from the table above.

### Using the overlay

Once calibrated, the projection overlay is **on by default**:

```bash
python -m twm.data_collection --task <task_name>   # press 'p' to toggle; --no_projection to start off
python -m twm.visualize path/to/episode_000.h5     # --no_projection to disable
python -m twm.visualize path/to/episode_000.h5 --cam_calib pushT   # pick the epoch when the path names no known task
```

The live recorder always uses `twm/calibration/result/` (the current rig);
the viewer picks the epoch the episode was recorded in, see
[Visualizing Episodes](#visualizing-episodes).

A colored dot + XYZ axes is drawn on each calibrated camera view at the
projected GelSight surface center. Accuracy is bounded by the calibration RMSE
(~5 mm ≈ ~2 px at the working distance).

---

## RealSense depth: aligned now, or later

A D415's depth comes from its infrared pair and its color from a different
lens, so the two images do not line up. `rs.align` reprojects each depth
frame onto the color grid. It is the single most expensive thing the
recorder does per camera:

| | one D415 reader thread |
|---|---|
| default (`rs.align` on) | 39.3 % of a core |
| `--raw_depth` | 9.0 % of a core |

Three cameras, so about 0.9 of a core is at stake — on a 4-core machine
whose writer is already short of CPU.

```bash
python -m twm.data_collection --task <task> --raw_depth   # store depth as the sensor sees it
python -m twm.realsense_align apply episode_000.h5 episode_000_aligned.h5
```

Alignment is a deterministic reprojection from the factory calibration, so
doing it afterwards gives the same pixels. `twm/realsense_align.py`
reimplements librealsense's `align_z_to_other`; on live frames from all
three cameras it reproduces `rs.align` for **99.998 %** of pixels (63 to 101
pixels of 4.6 million per camera, at depth discontinuities). Re-run that
comparison any time with `python -m twm.realsense_align verify`.

Every episode records `metadata.attrs["depth_aligned"]`. Episodes recorded
before the flag existed have aligned depth and no attribute, which reads as
`True`.

### The calibration

`twm/calibration/realsense/<serial>.json` holds each camera's depth and
color intrinsics, the depth→color extrinsics and the depth scale, read from
the camera at 640x480:

```bash
python -m twm.realsense_align export     # rewrite them from the attached cameras
```

Re-export after swapping a camera or changing the recording resolution:
intrinsics belong to a stream profile, and aligning with another camera's
extrinsics produces a depth map that looks plausible and is wrong
everywhere. These are also the intrinsics `K` that pose estimators such as
FoundationPose need, which no episode file carries yet.

---

## HDF5 File Format

Each episode is one `.h5` file. Structure:

```
episode_NNN.h5
├── metadata/               (attrs: fps, task, created_at, realsense_serials, gelsight_serials, arducam_config,
│                            and at finalize: valid, invalid_reason, ended_by, frame_count, duration_s,
│                            max_tick_gap_s, gap_count, queue_peak_fraction, writer_mean_mb_s,
│                            sensor_restarts (JSON, per stream), depth_aligned, ended_at)
├── timestamps              float64 [T]           — Unix time per frame
├── realsense/
│   ├── cam0/
│   │   ├── color           uint8  [T, 480, 640, 3]   — BGR
│   │   └── depth           uint16 [T, 480, 640]      — millimetres
│   ├── cam1/  (same)
│   └── cam2/  (same)
├── gelsight/
│   ├── left/
│   │   ├── frames          uint8   [T, 480, 640, 3]  — raw RGB
│   │   └── timestamps      float64 [T]               — capture time (sensor runs ~18 Hz)
│   └── right/  (same)
├── arducam/
│   ├── cam0/
│   │   ├── frames          uint8   [T, 480, 640, 3]  — BGR
│   │   └── timestamps      float64 [T]               — capture time
│   └── cam1/  (same)
└── optitrack/
    ├── motherboard/
    │   ├── timestamps      float64 [N]           — Unix time per pose sample
    │   └── pose            float64 [N, 7]        — [x, y, z, qx, qy, qz, qw] (metres)
    ├── sensor_left/  (same)
    └── sensor_right/ (same)
```

**Notes:**
- `T` = number of recorder ticks (same across all camera streams within an episode). GelSight and
  Arducam frames each carry their own capture `timestamps`; align by nearest time, not by index.
- A tick stores the sensor's newest frame. When a GelSight (~17 Hz) or a stalled
  sensor had no new frame, the previous frame is stored again with its previous
  timestamp, so a repeated `timestamps` value marks a held frame; use
  `np.unique` on that stream's timestamps to get the sensor's own frames.
  `python -m twm.recorder integrity` reports held and lost frames per stream.
- `N` = number of OptiTrack samples, recorded at the motion capture system rate (typically higher than camera FPS). Use `timestamps` to align with camera frames.
- Camera image data is BLOSC-LZ4 (bitshuffle) compressed and chunked per frame for fast
  random access. Import `hdf5plugin` before reading with h5py so the filter is
  registered.
- Each `arducam/cam*` group stores `usb_path`, `serial`, `device_at_recording`,
  `reported_serial`, `position`, `width`, `height`, `fps`, and `pixel_format`
  attributes. `position=unknown` is valid until physical mapping is completed.
- Depth values are in **millimetres** (uint16, range 0–65535). They are on the
  color camera's grid unless `metadata.attrs["depth_aligned"]` is False, in
  which case they are still on the depth camera's own grid and need
  `twm.realsense_align` (see above).
- GelSight frames are raw; compute contact difference offline: `diff = frame - ref + 128` (clipped to uint8), where `ref` is a no-contact reference frame.

### Reading an episode

```python
import h5py
import hdf5plugin  # registers the BLOSC filter used by recorded image datasets
import numpy as np

with h5py.File("episode_000.h5", "r") as f:
    timestamps   = f["timestamps"][:]               # (T,)
    color_cam0   = f["realsense/cam0/color"][:]     # (T, 480, 640, 3)
    depth_cam0   = f["realsense/cam0/depth"][:]     # (T, 480, 640)
    gs_left      = f["gelsight/left/frames"][:]     # (T, 480, 640, 3)
    wrist_cam0   = f["arducam/cam0/frames"][:]      # (T, 480, 640, 3)
    wrist_cam0_t = f["arducam/cam0/timestamps"][:]  # (T,)
    ot_poses     = f["optitrack/sensor_left/pose"][:] # (N, 7)
    ot_ts        = f["optitrack/sensor_left/timestamps"][:] # (N,)

    # Align OptiTrack to camera frame i
    i = 42
    cam_t = timestamps[i]
    nearest = np.argmin(np.abs(ot_ts - cam_t))
    pose_at_frame_i = ot_poses[nearest]  # [x, y, z, qx, qy, qz, qw]
```

---

## Force recovery

The recordings hold pose and GelSight images but **no applied force**. The
[`force_recovery/`](force_recovery/) package estimates normal force from the
tactile images alone, adds it to the dataset as an observation, and derives a
force-informed position target from it.

- **Live results:** https://huggingface.co/spaces/yxma/react-force-recovery
- **Package README:** [`force_recovery/README.md`](force_recovery/README.md) —
  method, the five validation datasets, and the measured limits
- **Module map:** [`force_recovery/ARCHITECTURE.md`](force_recovery/ARCHITECTURE.md)
- **Public API:** `force_recovery/pipeline.py`

```python
from force_recovery.pipeline import reconstruct, virtual_target, STIFFNESS_N_PER_MM

st     = reconstruct(img, ref)                 # dI → RGB LUT → Poisson → depth
target = virtual_target(pose, force_n, n_hat)  # pose + (F/k)·n̂ , k = 1 N/mm
```

Validated on five public force-labelled GelSight datasets with zero training
frames from this rig (ρ 0.946–0.986 on the four markerless sets, each beside a
within-group shuffle control). Two limits to read before using the numbers:
accuracy is depth-dependent (11 µm at a 0.3 mm press, 281 µm at 2.25 mm), and
the calibration is per-group, so ρ is a rank correlation rather than a
transferable absolute-newton scale. Details and negative results in the
package README.
