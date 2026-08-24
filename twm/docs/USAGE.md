# React — loading, preprocessing, and the tools that post-process it

Everything a consumer touches, in one place: what the columns mean, what the
loader does, what was done to the data before you got it, and what is known to
be wrong with it.

Companion docs: [`test_sets/probes_v1/README.md`](../test_sets/probes_v1/README.md)
for the action-following probe set, `toolbox/quickstart.md` for a five-minute
tour.

---

## 1. Conventions, stated once

| | |
|---|---|
| Pose layout | `[x, y, z, qx, qy, qz, qw]` |
| Position units | **metres** (millimetres everywhere inside the toolbox) |
| Quaternion order | **scalar-last (xyzw)** — `scipy...Rotation.from_quat` |
| Rotation deltas | **world-frame**: `dq = q[i+1] · q[i]⁻¹`, integrate `q[i+1] = dq · q[i]` |
| World frame | OptiTrack, **2026-05-10 reference** |
| Images | 640×480, three colour views (`left`, `middle`, `right`) + two tactile |

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

A training window starting shortly **before** a held-out interval still contains
its frames, so starts in `[a-(S-1), b]` must be rejected, not just `[a, b]`.
`splits.json` records `guard_frames = max_train_window - 1`, and the loader
**raises** on a longer window rather than leaking:

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

The held-out split cannot isolate action-following, because the action in a
recording is whatever the human happened to do. The probes command motions one
axis at a time, so a failure names a direction. Probe start frames are drawn
**only from held-out intervals**.

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

**What "correct" means.** Camera reprojection rmse is 4.7 / 5.3 / 7.5 mm for
left / middle / right → **3.6 / 4.0 / 5.7 px** at 800 mm; the gel centre in the
rigid frame is good to ~5 mm → ~3.8 px. **Agreement within about 6 px is at the
noise floor.** `rollout_error` reports millimetres *and* pixels because they
differ by depth.

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
6. **Validate**: `python scripts/validate_all.py`, then
   `scripts/test_frame_consistency.py` to confirm the new session shares the
   world frame, and `scripts/test_site.py` if you republished pages.

If a new session's world frame moved and you cannot measure the change, say so
in `WORLD_RESIDUAL` and keep the session. Dropping data to hide a bounded,
declared error is the wrong trade — that mistake was made here once and undone.
