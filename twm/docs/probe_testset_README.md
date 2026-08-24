# React probe test set — controlled actions for world-model rollout evaluation

72 commanded action sequences over 6 start frames, for scoring a tactile world
model's rollouts against ground truth that is **geometric, not photometric**.

Format `react-probe-testset/1.0`. Task: `motherboard`.

---

## Why this exists

A world model rolled out from a real episode can be scored against the frames
that were actually recorded. That measures interpolation of behaviour the model
has seen. It does not tell you whether the model has learned *how a commanded
motion moves the sensor* — because the action in a recording is whatever the
human happened to do.

These probes are **axis-aligned, controlled actions**: six pure translations
along ±x, ±y, ±z and six pure rotations about the same axes, from each start
frame. Nobody performed them, so **there is no ground-truth future image**.

What *is* ground truth is where the sensor **would be** if the action were
executed exactly — a pose sequence, and its projection into each camera. A
rollout is judged by comparing the sensor it draws against that projection.
Axis-aligned on purpose: when a probe fails you can say *which direction*
failed. A random direction gives you a number and no handle.

![overlay example](overlay_example.jpg)

*Yellow: commanded ground truth. Red: a deliberately wrong rollout, offset
25 mm in world x — it reads 18–19 px, three times the ~6 px noise floor.
Dimmed: the hand that must stay still. `overlays/` holds one such still for
every probe.*

## What a probe is

| | |
|---|---|
| Probes | **72** — 12 per start frame (6 translations, 6 rotations) |
| Start frames | **6**, each 4 consecutive context frames × 3 camera views |
| Translation amplitude | **0.113 – 0.391 m** |
| Rotation amplitude | **18.6 – 88.7°** |
| Horizon | **1.50 – 4.27 s** at 30 Hz |
| Speed | dataset percentile **p33 – p77** |
| Moving hand | 36 left / 36 right; one hand moves, the other holds |
| Closest approach between gels | **0.131 m** (rule: ≥ 0.12 m) |

**One hand moves per probe.** The other holds its pose for the whole horizon,
so a rollout must keep it still — a model that drifts both hands is visibly
wrong even when the moving one is right.

**Rotations pivot on the gel**, not on the OptiTrack marker cluster. The marker
cluster sits 65.7 mm from the gel, so rotating about it would swing the contact
point through an arc of up to 53 mm and a "pure rotation" would translate
across the screen.

## Layout

```
manifest.json                      format, conventions, error budget, residuals
calibration/                       T_mocap_to_cam_{left,middle,right}.json
                                   T_gel_to_rigid_{left,right}.json
probes/runN/
  meta.json                        episode, context rows, moving/held hand
  context/ctx{0..3}_{view}.jpg     the frames a model conditions on
  {trans,rot}{±x,±y,±z}.npz        one probe
overlays/runN_<probe>.jpg          ground truth drawn on the last context frame
overlay_example.jpg                the figure above
```

Each `.npz` holds:

| key | shape | meaning |
|---|---|---|
| `poses` | (T+1, 7) | commanded ground-truth pose of the moving sensor |
| `held_pose` | (7,) | the stationary hand, constant over the horizon |
| `context_poses_moving` / `_held` | (4, 7) | poses at the context frames |
| `gel_pos_m` | (T+1, 3) | the gel centre — what the action is measured at |
| `delta_gel_pos_m` | (T, 3) | **the action**: per-step translation, world axes, at the gel |
| `delta_gel_rotvec_rad` | (T, 3) | **the action**: per-step rotation, world axes |
| `action_scalar` | (T,) | the same action as one number: signed step along `action_axis` |
| `action_axis` / `action_sign` | scalar | 0/1/2 for x/y/z, and ±1 |
| `delta_rigid_pos_m` / `delta_rigid_rotvec_rad` | (T, 3) | the marker cluster's motion instead |
| `gt_px_{left,middle,right}` | (T+1, 2) | ground-truth gel-centre pixels |

Poses are `[x, y, z, qx, qy, qz, qw]`, position in **metres**, quaternion in
**xyzw** order (`scipy.spatial.transform.Rotation.from_quat`), in the OptiTrack
world frame with **2026-05-10** as reference.

### One action, one direction — and where you have to measure it

Every probe moves along **exactly one axis**: a translation probe has zero
rotation, a rotation probe has zero translation, and the off-axis components are
zero to machine precision. All of that is true **at the gel**, and false at the
marker cluster.

The pose 7-vec is the OptiTrack marker cluster's, and rotations pivot on the gel
65.7 mm away — so in rigid-body coordinates a "pure rotation" carries up to
**91 mm** of translation. A model fed `delta_rigid_*` for `rot+x` reads
"translate 91 mm *and* rotate 79°" for something labelled a pure rotation. Hence
`delta_gel_*` is the primary action; `delta_rigid_*` ships alongside for a model
that predicts the marker-cluster pose, under a name that cannot be confused.

Rotation deltas are **world-frame**, i.e. pre-multiplied: `dq = q[i+1] · q[i]⁻¹`,
integrate as `q[i+1] = dq · q[i]`. The probes rotate about world axes, so the
world-frame increment lies exactly along the named axis; the body-frame
increment `q[i]⁻¹ · q[i+1]` is the same rotation seen from the moving hand and
sits 7.1e-3 rad off it.

Both deltas integrate back to their own trajectory exactly — the rigid one to
`poses`, the gel one to `gel_pos_m` — asserted to 1e-9 m and 1e-6 deg.

## Usage

```python
import json, numpy as np, cv2
from react_toolbox.calibration import load_calibration
from react_toolbox.probe_eval import overlay_gt, rollout_error

root = "react_probe_testset"
cal  = load_calibration(root)              # the calibration IN the package
run  = json.load(open(f"{root}/probes/run0/meta.json"))
d    = np.load(f"{root}/probes/run0/trans+x.npz")

gel  = cal[f"gel_{run['moving_side']}"]
cam  = cal["cams"]["middle"]

# --- the model input: 4 context frames, and the action
ctx    = [cv2.imread(f"{root}/probes/run0/context/ctx{i}_middle.jpg")[:, :, ::-1]
          for i in range(4)]
action = np.concatenate([d["delta_gel_pos_m"],
                         d["delta_gel_rotvec_rad"]], axis=1)   # (T, 6), at the gel
# or, since each probe is one-directional, the same thing as one number:
#   d["action_scalar"], along axis "xyz"[int(d["action_axis"])]

pred = my_world_model.rollout(ctx, action)      # -> (T+1, 7) poses

# --- score it
err = rollout_error(pred, d["poses"], gel, cam)
print(err["pos_mm_final"], err["rot_deg_final"], err["px_final"])

# --- and look at it
vis = overlay_gt(ctx[-1], d["poses"], gel, cam,
                 held_pose7=d["held_pose"], held_gel_mm=cal[f"gel_{run['held_side']}"])
vis = overlay_gt(vis, pred, gel, cam, color=(255, 90, 90))   # your rollout, in red
```

### The overlay

`overlay_gt` draws, on a context frame:

* the **held hand**, dimmed, with a stem back to its marker cluster
* the **start** and **end** sensor frames as perspective triads — a dot cannot
  show a rotation probe, where the gel centre does not move at all
* the commanded **path** as a polyline, start a white dot, end a ring

Triads go down first and the path markers on top; the other order puts the start
triad's centre dot exactly over the start marker and hides it.

All projection goes through `calibration.project_gel_to_pixel`, the same
function the dataset previews and the release fingerprint use, so an overlay you
draw cannot disagree with the stored `gt_px_*`.

### What "correct" means — the overlay's error bar

Projected ground truth is not exact:

| source | at 800 mm depth |
|---|---|
| camera reprojection rmse (left / middle / right: 4.7 / 5.3 / 7.5 mm) | **3.6 / 4.0 / 5.7 px** |
| gel centre in the rigid frame (≤ ~5 mm) | **≈ 3.8 px** |

**Agreement within about 6 px is at the noise floor** and should be read as
correct. `rollout_error` reports millimetres *and* pixels because they differ by
depth, and neither substitutes for the other.

## How the probes were generated

1. **Sample a start frame.** A run of 4 consecutive rows with both sensors
   tracked, from a session whose world frame is pinned (below). Start frames and
   actions are sampled **independently**: a frame is accepted or rejected
   against the actions, never adjusted to fit them — nudging a trajectory to
   keep it on screen would make two probes named `+x` mean different things.

2. **Generate 12 actions** from the moving hand's start pose: six translations
   along the signed world axes, six rotations about them. Rotations pre-multiply
   in the world frame ("turn the hand this way in the room"), which is what a
   viewer can judge from a camera image.

3. **Pace them against the dataset.** Speed is drawn uniformly in *percentile*
   of the measured per-step distribution (p45–p85, capped by the horizon), not
   uniformly in mm/step — the distribution spans a decade between p25 and p90,
   so a uniform draw in value would put most probes in a tail the data barely
   occupies. Every probe records the percentile it actually lands on.

   The 1.5 s horizon caps speed at `amplitude / 45`, which bites at small
   amplitudes: 0.1 m over 45 steps is p44, 18° is p32. Those probes are the
   slowest in the set and cannot be faster without breaking the horizon.

4. **Reject, never adjust.** A probe is discarded, and the start frame with it,
   if the two gel centres come within 0.12 m — hands do not pass through each
   other, and a probe that says they do tests whether the model will hallucinate
   rather than whether it can predict. The projected path must also stay
   **40 px** clear of the image border: in frame is not enough, because a
   rollout that overshoots a path ending 15 px from the edge leaves the image
   and cannot be scored at all.

### Sessions

Start frames come only from **2026-05-10** and **2026-05-11**.

**2026-05-19 is excluded.** Its OptiTrack world was redefined mid-collection.
The release corrects that with a translation only; the yaw about the table
normal and the in-plane translation remain unmeasured — attempts scatter over
±2.3°, which is **16 px** at the workspace, three times the noise floor above.
Projected ground truth is the whole point of this test set, so a session whose
projection carries an unstated bias does not belong in it. `manifest.json`
records this under `excluded_sessions`, and `world_residual` for the included
sessions.

## Reproducing

```
python scripts/build_probe_testset.py --runs 6 --seed 0
python scripts/test_probe_testset.py
```

The test asserts the package is self-contained, that the stored ground-truth
pixels recompute from the calibration **inside the package**, that the deltas
integrate back to the poses, that the ground truth keeps its scoring margin, and
that the scorer reads zero on the ground truth and exactly 10 mm on an injected
10 mm error.
