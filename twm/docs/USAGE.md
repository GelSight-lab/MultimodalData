# React — loading, preprocessing, and the tools that post-process it

Everything a consumer touches, in one place: what the columns mean, what the
loader does, what was done to the data before you got it, and what is known to
be wrong with it.

Companion docs: [`test_sets/probes_v1/README.md`](../test_sets/probes_v1/README.md)
for the action-following probe set, `toolbox/quickstart.md` for a five-minute
tour.

---

## Start here

Two bimanual tasks recorded with three cameras, two touch sensors and motion
capture. Each episode is a parquet table (one row per camera frame) plus five
videos.

```python
from react_video_dataset import ReactVideoDataset
ds  = ReactVideoDataset("data/motherboard", split="train", window_length=16)
cal = ds.calibration()          # poses and cameras in the SAME convention
```

Six things that bite, each explained below:

| | |
|---|---|
| **Up is +z** | recorded Y-up, published Z-up. Take poses and cameras from the same place, or every overlay moves by up to 165 px. §1 |
| **Quaternions are xyzw** | scalar-last, as `scipy` wants. §1 |
| **Touch updates on ~29 % of rows** | the rest repeat the previous value. Check `tactile_*_is_new` before averaging. §2 |
| **Force is a magnitude** | direction comes from the sensor's own frame, not from world "down". §2 |
| **Held-out data is intervals, not episodes** | a training window that starts just before one still sees it; the loader raises instead of leaking. §3 |
| **Frame rate is not 30 Hz everywhere** | read `timestamp`, never assume. §5 |

## 0. Terms

Plain definitions for the words used throughout.

| term | what it means here |
|---|---|
| **motion capture / OptiTrack** | cameras that track reflective markers and report where an object is, at ~120 Hz |
| **rigid body** | a marker cluster the system treats as one object. Its local axes are fixed when the body is created, and are not a property of the hardware |
| **pose** | position + orientation: `[x, y, z, qx, qy, qz, qw]`, metres and a unit quaternion |
| **scalar-last (xyzw)** | the `w` of the quaternion comes last. `wxyz` is the other common order and silently gives wrong rotations |
| **world frame** | the shared coordinate system all poses are in — here, OptiTrack's, with the 2026-05-10 origin |
| **up axis** | which axis points away from gravity. This release: `+z` |
| **extrinsics** (`T_mocap_to_cam`) | where a camera sits in the world frame, as a 4×4 matrix |
| **intrinsics** | a camera's focal lengths and optical centre, which turn a 3-D point into a pixel |
| **GelSight Mini** | a touch sensor: a soft gel pad filmed from inside, so contact shows up as an image |
| **gel** | that soft pad. **Gel centre** is its middle, **gel normal** the direction perpendicular to its surface |
| **normal force** | how hard the gel is pressed, in newtons. A single number, no direction attached |
| **penetration** | how far the gel would be pushed in: `force / k`, with stiffness `k = 2 N/mm` |
| **virtual target** | the pose a stiffness controller would have been commanded to reach: the observed pose pushed `penetration` further along the gel normal |
| **held-out interval** | a stretch of frames inside an episode reserved for testing |
| **guard** | frames next to a held-out interval that training must also avoid, because a window starting there would contain held-out frames |
| **segment** | a stretch curated as clean; `mode="segment"` uses only these |
| **bad frame** | a frame flagged as corrupt: a sensor spike, a pose jump, a tracking dropout |
| **calibration epoch** | one camera-calibration session. Sessions recorded under different epochs need different extrinsics |
| **probe set** | synthetic single-axis motions used to ask whether a model follows a commanded action, as opposed to predicting a recording |
| **reprojection error** | how far a known 3-D point lands from where the calibration says it should, in pixels or millimetres — the noise floor for any overlay |
| **fingerprint** | a few stored pixel coordinates you can recompute from your own poses to confirm you are in the same frame as the release |
| **trim** | the offset between a published row and its frame in the original recording (`source_h5_frame`) |

---

## 1. Conventions, stated once

| | |
|---|---|
| Pose layout | `[x, y, z, qx, qy, qz, qw]` |
| Position units | **metres** (millimetres everywhere inside the toolbox) |
| Quaternion order | **scalar-last (xyzw)** — `scipy...Rotation.from_quat` |
| Rotation deltas | **world-frame**: `dq = q[i+1] · q[i]⁻¹`, integrate `q[i+1] = dq · q[i]` |
| World frame | OptiTrack, **2026-05-10 reference** |
| **Up axis** | **+z**, right-handed. Recorded Y-up, published Z-up. See below. |
| Images | 640×480, three colour views (`left`, `middle`, `right`) + two tactile |

### Up is +z — converted, not recorded

OptiTrack records **Y-up**. This release is published **Z-up**, because most
robotics code assumes Z-up.

Why it matters: under Y-up, `pose[2]` is not height — it is a horizontal
coordinate. The numbers still look reasonable and the plots still look fine.
The only symptom is a model that never learns which way gravity points.

So the conversion was done once, in the data:

```python
ds  = ReactVideoDataset(root)                # up_axis="z" is the default
cal = ds.calibration()                       # extrinsics in the SAME convention
```

Measured on the published tree, the table normal is **[-0.015, -0.029, 0.999]**
— +z, 1.9° off — and every rotation has determinant +1. Pass `up_axis="y"` to
get the raw OptiTrack convention back; the calibration comes back with it.

**Never take the two halves from different places.** The conversion is a
rotation of the world frame, so it applies to the poses *and* to
`T_mocap_to_cam`. Applied to one only it moves every projection by up to
**165 px** and raises nothing. That is not hypothetical: the probe test set
drew poses from the converted release and calibration from an unconverted tree,
and every overlay was a median **153 px** off while all of its self-consistency
checks stayed green — the same wrong matrix was used to draw and to re-verify.

Two things now prevent it. Each camera calibration declares its convention:

```json
{"T_mocap_to_cam": [...], "up_axis": "z"}
```

and `toolbox/frames.py` exposes `require_up_axis(cal)`, which raises rather
than letting a mismatched pair through. A file with **no** `up_axis` key is
treated as the pre-conversion Y-up it was, not waved past.

`scripts/test_frames.py` asserts both directions: converting poses and cameras
together leaves projections identical to 8.5e-14 px, and converting one alone
moves them 165 px. The second check is what makes the first mean anything.

One field is deliberately **not** Z-up. Each parquet's `twm.world_frame`
declaration carries `raw_h5_offset_m`, whose job is to be added to a pose read
straight out of the source HDF5 — and that file is Y-up, as recorded. It ships
with `raw_h5_offset_up_axis: "y"` saying so. Everything else in that blob
describes the published poses and is Z-up.

The rotation is `R_x(-90°)`: `(x, y, z) → (x, -z, y)`. There are two
right-handed candidates; the other one leaves the world upside down with
`det = +1` and no handedness check would notice. `toolbox/frames.py` exposes
`convert_poses`, `convert_calibration` and `to_zup(poses, cal)` — prefer the
last, since the whole point is that they move as one piece.

The gel centre is in the sensor's own rigid frame and is untouched.

`toolbox/actions.py` once documented "quat wxyz" while its code was scalar-last.
The code was right; a reader who trusted the text would have swapped `w` into
`x`. If a docstring and the data disagree, trust a round-trip test, not prose.

## 2. Parquet columns

Per episode, one row per **camera** frame.

| group | columns |
|---|---|
| index | `frame_idx`, `timestamp`, `source_h5_frame`, `episode`, `episode_index`, `frame_index`, `task`, `task_index` |
| poses | `sensor_left_pose`, `sensor_right_pose`, `object_pose` — 7-vectors |
| tactile | `tactile_{side}_{intensity,area,mixed}`, `tactile_{side}_is_new` |
| force | `force_{side}_normal_n`, `force_{side}_penetration_mm`, `force_{side}_target_pose`, `force_{side}_source_frame` |

**`tactile_*_is_new` is not decoration.** Tactile — and therefore force — updates
on only **~29 %** of rows: the GelSight stream ran slower than the cameras. A
row where `is_new` is false repeats the previous tactile frame. Averaging force
over all rows silently weights repeats.

**`penetration_mm = force_normal_n / k`**, with `k` = `dexforce.STIFFNESS_N_PER_M / 1000`
= **2.0 N/mm**. Import it; do not retype it. A hard-coded `k = 1.0` in a test
went stale against the data and failed silently on 12 episode-sides.

**`force_{side}_source_frame`** names the tactile frame a force came from, so an
alignment claim is checkable rather than asserted.

### Which direction the force acts along

`force_normal_n` is a **magnitude**. It acts along the gel normal expressed in
the **sensor's own body frame** — not along world vertical, which is a median
7.7° away on motherboard and **23.3° on pushT, where 70 % of contact frames
exceed 15°**. `force_{side}_target_pose` already has the direction applied:

```python
from force_recovery.dexforce import gel_axis, STIFFNESS_N_PER_M
n_hat = R_from_quat(pose[3:7]) @ gel_axis(task, side)   # world unit vector
target_xyz = pose[:3] + (force_n / STIFFNESS_N_PER_M) * n_hat
```

**The default is local `-y`**: the GelSight Mini's sensing face is normal to
the body's y axis, so a compression acts along `-y`.

The calibration files also carry `gel_axis_in_rigid`, reachable as
`gel_axis(task, side, source="dual_ball")`. It is
`normalize(gelball_centre - refball_centre)` — the line between two calibration
ball centres 57 mm apart, from **three** poses. It never measured the gel
surface, and equals the normal only if the fixture held both balls along it.
It sits 21.2° (left) and 22.4° (right) from `-y`.

Which is right is **not settled**, and the two sensors do not agree. Measured:

| test | left | right |
|---|---|---|
| angle from board normal, pressing >6 N on a level board (38 k frames) | dual_ball **7.1°**, -y 25.6° | dual_ball 18.1°, -y **7.7°** |
| corr(dF, v·n̂) — no world-frame or table assumption (31 episodes) | dual_ball **+0.085**, -y +0.053; -y better on only 3 % of episodes | +0.116 vs +0.116, a tie |

So the **left** sensor's dual-ball axis looks right by both tests, and the
**right** sensor's looks wrong by one and indifferent by the other. The right
calibration also carries `depth_offset_mm = 0.0` where the left carries `-5.0`,
i.e. its ball centre was never backed off by a ball radius to reach the gel
surface — a second sign of trouble in the same file.

Kinematics cannot arbitrate: over contact frames `sum(R_i)` has singular-value
ratio 1.09 (left) and 1.04 (right), so the axis is unidentifiable from motion
alone, and the two candidates' concentration scores differ by 0.013.

Choosing `-y` moves `force_*_target_pose` by a median 0.57 mm (max 1.53 mm),
because `F/k` is itself only a few mm. Pass `source="dual_ball"` to reproduce
the earlier published values.

## 3. The loader

```python
from react_video_dataset import ReactVideoDataset

ds = ReactVideoDataset(
    "data/motherboard",
    window_length=16, stride=1, window_step=16,
    streams=("view_middle", "tactile_left", "tactile_right"),
    split="train",            # "train" | "test" | "all"
    skip_bad=True,
    tactile_latency=0,
)
```

| argument | what it does |
|---|---|
| `window_length`, `stride`, `window_step` | window shape and how far the index advances |
| `mode` | `"segment"` (default, clean intervals from `segments.json`) or `"window"` (whole episode) |
| `streams` | any of the three views, two tactile, plus depth if `load_depth=True` |
| `skip_bad` | drop windows touching `bad_frames.json` |
| `tactile_latency` | pairs `view[i]` with `tactile[i+lat]`; see §5 |
| `split` | reads `splits.json`; **raises** if your window is longer than the guard |

### The split, and the part that leaks

Held-out data is carved as **intervals from inside episodes**, not whole
episodes. There are 32 motherboard episodes; spending them on episode-level
held-out data buys independence a short-horizon world model does not need —
what it must generalise over is dynamics within a scene, not scenes.

Measured with a 64-frame training window: **test 12.1 %, guard 9.4 %,
train 78.5 %**, over 147 intervals plus 2 wholly-held-out episodes.

A window of length `S` that starts shortly **before** a held-out interval still
contains part of it. So the rejected range is `[a-(S-1), b]`, not just `[a, b]`
— that is what the guard is for. `splits.json` stores
`guard_frames = max_train_window - 1`, and the loader **raises** rather than
leak if your window is longer:

```
ValueError: window spans 128 frames but the split has guard_frames=63 …
            Rebuild with max_train_window >= 128.
```

That failure mode leaves no trace in any metric until the numbers are
suspiciously good. Enumerated: 159,890 admissible training windows, none touch
a held-out frame; drop the guard and 1,827 leak in the first six episodes alone.

Rebuild for a longer horizon:

```
python scripts/build_splits.py --max-train-window 128
```

## 4. Two evaluation sets, two questions

| | question | data |
|---|---|---|
| `split="test"` | can the model predict what actually happened? | real frames, real actions, real futures |
| **probe set** | does it *follow the action it is given*? | commanded motions nobody performed; ground truth is geometric |

The held-out split cannot test action-following on its own: in a recording, the
action is whatever the person happened to do, so you cannot ask "what if it had
moved 5 mm the other way".

The probes command one axis at a time, so a failure names a direction. Their
start frames come **only from held-out intervals**.

## 5. What was done to the data before you got it

1. **Trim.** `source_h5_frame` maps a row back to its raw HDF5 frame; row `r` is
   camera frame `trim + r`. Do not add any other lag term here — a fifth inline
   copy of a `+15` shift put the force disc half a second from its tile in every
   published preview.
2. **Bad-frame flagging.** `bad_frames.json` marks intensity spikes, pose
   teleports and OptiTrack dropouts. `skip_bad=True` honours it.
3. **Segments.** `segments.json` holds 81 clean intervals; `mode="segment"` uses
   only those.
4. **Rest-gel reference.** Per-episode fuzzy-mode background, falling back to a
   per-session reference only for episodes that are ~100 % contact.
   (`data/<task>/reference/validation.md`.)
5. **Force.** Estimated from tactile, then `penetration = F/k` and a DexForce
   virtual target pose. Only ~29 % of rows carry a fresh estimate.
6. **World frame.** 2026-05-19's OptiTrack origin was redefined mid-collection;
   the release bakes in a translation correction. See §7.

**Tactile latency.** Recordings before 2026-06-27 have a V4L2 buffer bug: the
tactile stream was captured *before* the view at the same index. Pass
`tactile_latency=15` to pair `view[i]` with `tactile[i+15]` if your task needs
tight tactile-visual sync. All motherboard episodes predate the fix.

**Frame rate is not 30 Hz everywhere.** 2026-05-10 and 2026-05-11 run at
29.9 Hz; **2026-05-19 runs at 11.7–23.5 Hz**, and varies between its own
episodes. Anything that converts frames to seconds must read `timestamp`, not
assume a rate.

## 6. Post-processing tools

| module | what it is for |
|---|---|
| `toolbox/calibration.py` | load calibration; project the gel centre or its full frame into a camera |
| `toolbox/viz.py` | draw a projection, a sensor triad, a collision circle, a force disc |
| `toolbox/actions.py` | derive actions from recorded poses (`delta_pose_action` / `integrate_delta`) |
| `toolbox/synth_actions.py` | generate the synthetic single-axis probes |
| `toolbox/probe_eval.py` | project ground truth, overlay it, score a rollout |
| `toolbox/world_frame.py` | declare and verify which world frame a pose array is in |
| `toolbox/conformance.py` | check **your** poses against this release's conventions — see §6b |
| `toolbox/splits.py` | build / read the held-out interval split |
| `toolbox/calib_epoch.py` | which calibration epoch and world transform a session uses |

### Overlaying ground truth on an image

```python
from react_toolbox.calibration import load_calibration
from react_toolbox.probe_eval import overlay_gt, rollout_error

cal = load_calibration(root)               # the calibration IN the package
vis = overlay_gt(frame, gt_poses, cal["gel_left"], cal["cams"]["middle"],
                 held_pose7=held, held_gel_mm=cal["gel_right"])
vis = overlay_gt(vis, my_rollout, cal["gel_left"], cal["cams"]["middle"],
                 color=(255, 90, 90))
err = rollout_error(my_rollout, gt_poses, cal["gel_left"], cal["cams"]["middle"])
```

All projection goes through `calibration.project_gel_to_pixel`, the same
function the previews and the release fingerprint use, so an overlay you draw
cannot disagree with a stored one.

**What "correct" means.** Two error sources set the floor. Camera reprojection
rmse is 4.7 / 5.3 / 7.5 mm for left / middle / right,
which is **3.6 / 4.0 / 5.7 px** at 800 mm. The gel centre is good to ~5 mm,
another ~3.8 px.

So **agreement within about 6 px is as good as this rig can tell.** Do not read
a 3 px difference as a result. `rollout_error` gives millimetres *and* pixels,
because the same millimetre is more pixels up close.

## 6b. Checking your own usage

Every other test here validates the dataset. This one validates *your* use of
it, which is the half that goes wrong. The data can be perfect and still be
read in millimetres, or with `w` first, or paired with extrinsics from the
other up-axis. **None of those raise.** They shift every projection and leave
your own self-consistency checks green, because the same wrong assumption both
draws and re-verifies.

```
python -m react_toolbox.conformance --release data/motherboard \
    --task motherboard --episode 2026-05-10/episode_000
```

```
conformance: PASS
  ok    quaternion norm deviates by at most 2.22e-16 from 1
  ok    median |position| = 0.435; expected metres, not millimetres
  ok    6890 poses given, episode has 6890 valid rows
  ok    worst-camera fingerprint error 0.00 px (tolerance 6)
        would read: other up-axis 364 px, quaternion as wxyz 55 px
```

Exit code 0 on pass, 1 on failure, so it drops into CI. Pass `--poses my.npy`
to check an array your own pipeline produced, or call `check_poses(...)` and
read `Report.failures`.

**A passing report still prints what each mistake would have read.** A
validator that only ever says "ok" tells you nothing about whether it *can*
say anything else.

### Is there a standard process for this?

Yes, and it has a name: a **conformance suite** over a **golden fixture**. The
same shape appears in
[BIDS](https://bids-standard.github.io/bids-validator/) for neuroimaging,
[Frictionless](https://framework.frictionlessdata.io/) for tabular data, and
[Croissant](https://mlcommons.org/croissant/) for ML dataset metadata. Four
parts, and where each one lives here:

| part | here |
|---|---|
| **1. Machine-readable declaration** of every convention | `up_axis` in each calibration JSON and in each parquet's `twm.world_frame` |
| **2. Golden values** the consumer recomputes | the projection fingerprint in that same blob — stored pixels you reproduce from your own poses |
| **3. A validator the consumer runs on their own code** | `python -m react_toolbox.conformance` |
| **4. Negative controls** — it must fail when you are wrong | printed on every report, and asserted by `scripts/test_conformance.py` |

Part 4 is the one usually skipped, and it is the one that matters. This
project shipped a self-consistency check that returned `0.0` for every input,
and an overlay test that was 153 px wrong with all its checks green. A check
nobody has watched fail is not evidence.

So `test_conformance.py` feeds the checker each classic mistake and requires
it to object, by name: millimetres, `wxyz`, the recorded Y-up convention,
unnormalised quaternions, and a row subset. Measured, those read 11,732 px,
55 px, 364 px, a norm error of 0.4, and 10.4 px respectively — against 0.00 px
for correct input.

That last one is not a mistake in your data, it is a mistake in how you *call*
the checker: the fingerprint is a median over the whole episode, so a subset
shifts it by real motion. The checker refuses the subset instead of reporting
a frame error that is not there. Pass `rows=` with your indices and it
compares row-wise against the release instead.

## 7. Known problems, stated rather than hidden

**2026-05-19's world frame.** Its OptiTrack calibration was re-run
mid-collection. The release applies a translation-only correction,
(230, 0, 175) mm. What remains unmeasured:

* rotation about the table normal (yaw). An estimate from board-outline
  matching gave +2.4°, but the reference date — zero by construction —
  scattered −1.8° to +3.3° over the same settings, so the method has no power
  and the number was withdrawn.
* Coupling mocap to the depth camera puts 05-19 **20–26 mm** from the reference
  frame, against a reference-to-reference floor of **8.7–11 mm**: about twice
  the instrument's own noise, so it is recorded, not corrected.

A tilt about an in-plane axis is **not** possible: the OptiTrack ground plane is
set with an L-bracket laid on the table, so two calibrations differ only by yaw
and in-plane translation. A 3.38° tilt measured here once was an artefact of a
non-planar contact cloud and has been retracted.

`calib_epoch.world_residual("motherboard", date)` returns all of this
programmatically. Use it to bound your own error rather than assuming zero.

## 7b. Running the scripts

Everything the sections above tell you to run ships under
[`scripts/`](scripts/). They read their roots from the environment, so:

```
REACT_RELEASE=/path/to/react/data python scripts/build_splits.py
REACT_RELEASE=... python scripts/test_splits.py
```

| variable | what it points at | published? |
|---|---|---|
| `REACT_RELEASE` | the release tree — `episodes.jsonl`, `splits.json`, `meta/`, `videos/` | **yes**, this dataset |
| `REACT_FORCE` | a release whose `meta/` has the force columns; defaults to `REACT_RELEASE` | yes |
| `REACT_TESTSET` | the probe package | yes, `test_sets/probes_v1` |
| `REACT_OUT` | where build scripts write | — |
| `REACT_RAW` | the original HDF5 capture tree | **no**, ~1 TB |

**What needs `REACT_RAW`, and therefore cannot be reproduced from the release
alone:** `build_probe_testset.py` and `render_probe_overlays.py` read original
camera frames for the probe context images, and `test_frame_consistency.py`
reads the depth stream. Everything else — the split, its tests, the probe
package's own tests, the pages — runs from what is published.

## 8. Adding a session or a task

The pieces that must be told about a new session, in order:

1. **`calib_epoch.CALIB_DIRS`** — which camera-extrinsics epoch the task uses.
   `calib_dir()` raises on an unknown task rather than falling back; a wrong
   epoch does not look wrong, it looks like a slightly miscalibrated rig, which
   is how it shipped unnoticed once.
2. **`episodes.jsonl`** — one record per episode, including
   `world_frame_offset`. Read by `calib_epoch.world_offset_m`; never retype the
   offset in code.
3. **`calib_epoch.WORLD_TRANSFORM` / `WORLD_RESIDUAL`** — only if the world
   frame moved. Record what you could not measure as `None` **with a reason**.
4. **`bad_frames.json`, `segments.json`** — from the curation pass.
5. **`splits.json`** — `python scripts/build_splits.py`.
6. **Validate**:

   ```
   python scripts/check_session_ready.py --task <task>   # is it registered?
   python scripts/validate_all.py                        # 22 checks
   python scripts/test_frame_consistency.py              # same world frame?
   python scripts/test_site.py                           # if pages were rebuilt
   ```

   `check_session_ready` answers steps 1–5 mechanically rather than leaving
   them to this list — a prose checklist gets skipped, and each omission has a
   silent failure mode: a wrong calibration epoch looks like a slightly
   miscalibrated rig (it shipped that way once, 35–73 px off), a missing
   `splits.json` entry puts the whole session in train.

If a new session's world frame moved and you cannot measure the change, say so
in `WORLD_RESIDUAL` and keep the session. Dropping data to hide a bounded,
declared error is the wrong trade — that mistake was made here once and undone.
