### estimated contact force (`motherboard` + `pushT`, 36 episodes)

There is **no force/torque sensor on this rig** — the demonstrator's hand holds
the sensor, so demonstrated pose equals achieved pose and the usual
"position error × stiffness" force channel does not exist. These columns are
**estimated from the GelSight images alone** by photometric reconstruction
(difference image → per-sensor RGB lookup table → Poisson integration → depth),
then mapped to newtons by a calibration fitted on sphere presses of known load.

| Column | Type | Meaning |
|---|---|---|
| `force_{left,right}_normal_n` | float32 | estimated normal force [N], ≥ 0, exactly `0.0` on no-contact rows |
| `force_{left,right}_penetration_mm` | float32 | `F / k` — how far a stiffness-`k` environment would be pushed in |
| `force_{left,right}_target_pose` | list[7] | that sensor's pose displaced `F/k` along the contact normal (quaternion carried through unchanged) |

```python
import numpy as np, pyarrow.parquet as pq
t = pq.read_table("data/motherboard/meta/2026-05-10/episode_000.parquet")
f   = t["force_left_normal_n"].to_numpy()                  # (T,)   newtons
obs = np.array(t["sensor_left_pose"].to_pylist())          # (T, 7) xyz + quat
tgt = np.array(t["force_left_target_pose"].to_pylist())    # (T, 7) the action
```

#### What "force-informed action" means, and how to train on it

A policy trained to output `sensor_*_pose` learns **where to go**. It cannot
learn **how hard to press**, because in this data the two are the same signal:
a human hand reached a pose, and whatever force resulted was never recorded as
a separate command. Regressing that pose and replaying it on a compliant robot
reproduces the trajectory and not the interaction — the same motion against a
stiffer or differently-placed object produces a different force, and nothing
in the demonstration says which force was intended.

`force_*_target_pose` is that missing command, written in the units a robot
already accepts:

```
target = observed + (F / k) · n̂        n̂ = press direction of that sensor,
                                            R(q_row) @ gel_axis_in_rigid
```

It is the pose a **stiffness-`k` impedance controller** would have to be
commanded in order to generate the estimated force `F` against a surface at the
observed pose. Train the policy to output `target_pose`, deploy it as the
setpoint of an impedance/admittance controller with the same `k`, and the
controller produces both the reach and the press. This is the standard trick
behind position-based force control; the only new part is that `F` came from
the tactile images rather than from a load cell.

```python
action      = tgt                      # what the policy predicts
observation = obs                      # where the sensor actually was
# free space: byte-identical, so this is a strict addition to the old target
assert np.array_equal(action[f == 0], observation[f == 0])
```

That identity is not a claim — it is checked element-wise over all **301,727**
free-space rows of the release, maximum deviation `0.0`, quaternions included.
Nothing changes where nothing is touched, so a model trained on `target_pose`
degenerates to the pose-only model in free space and differs only in contact.

#### Choosing `k` — it is your controller's number, not ours

`k = 1.0 N/mm` is a **declared assumption**, recorded in the parquet field
metadata (`twm.stiffness_n_per_mm`) and in each `<episode>.force.json`, so a
target pose is never uninterpretable. It is deliberately soft, and at that
value the implied penetrations are larger than the gel is thick:

| | penetration at `k = 1` | inside the 4.25 mm gel? |
|---|---|---|
| p95 over all rows | 5.78 mm | no |
| p95 over **contact** rows | 6.86 mm | no |
| maximum | 7.285 N → 7.285 mm | no |

**8.84%** of all rows exceed the gel thickness at `k = 1`. To keep penetration
physically plausible you need a stiffer environment model:

* `k ≥ 1.37 N/mm` — p95 over all rows inside the gel. *This is the weakest of
  the three and the least useful:* 62.8% of rows are free space, so a
  percentile over all rows is mostly a percentile of zeros.
* `k ≥ 1.62 N/mm` — p95 over **contact** rows inside the gel. Use this one.
* `k ≥ 1.72 N/mm` — even the hardest press inside the gel.

Recompute rather than rescale the shipped column, since the direction matters:

```python
K = 1.62                                             # your controller's stiffness
n_hat  = (tgt[:, :3] - obs[:, :3])                   # F/k · n̂ at the shipped k=1
n_hat /= np.linalg.norm(n_hat, axis=1, keepdims=True) + 1e-12
my_target = obs.copy()
my_target[:, :3] = obs[:, :3] + (f / K)[:, None] * n_hat
```

#### Read this before using the numbers

- **Accuracy is rank-order within a group, not a certified absolute scale.**
  Held out by press position the estimator scores ρ = 0.739 / MAE 1.23 N on its
  own calibration objects. On five public force-labelled datasets the same
  pipeline reaches ρ 0.775–0.986. It is reliable for *how hard, relative to
  other frames*; it is not a load cell. Do not report absolute newtons from
  this dataset as ground truth.
- **Forces saturate at 7.285 N.** The calibration's isotonic stage clips at the
  hardest press it was fitted on, so 0.90% of samples sit exactly at that value.
  Treat the maximum as a floor, not a measurement, and consider masking rows at
  the ceiling out of a regression loss.
- **Duplicate tactile rows repeat the previous estimate.** The GelSight stream
  is slower than 30 Hz; rows with `tactile_{side}_is_new == False` carry the
  previous frame's force unchanged (forward fill, asserted exact). Filter on
  `is_new` if you need independent samples — and note that a force *derivative*
  computed without that filter is zero on ~72% of rows by construction.
- **Row alignment is verified, not assumed.** Every one of the **72/72**
  sensor-sides was checked row-for-row against the release parquet it was
  exported from.
- **The direction `n̂` comes from calibration, not from the image.** It is the
  sensor's gel axis rotated by the row's own quaternion. Two sensor-sides of 72
  lack a usable gel-to-rigid transform and carry force with no displacement;
  they are identified in `data/force_export_manifest.json`.
