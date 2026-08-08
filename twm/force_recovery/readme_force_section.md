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
import pyarrow.parquet as pq
t = pq.read_table("data/motherboard/meta/2026-05-10/episode_000.parquet")
f = t["force_left_normal_n"].to_numpy()          # N, per frame
tgt = np.array(t["force_left_target_pose"].to_pylist())   # (T, 7)
```

**`target_pose` is the force-informed action.** Training on the observed pose
teaches "go here"; training on the target teaches "go here *and push this
hard*", which an impedance controller reproduces at deployment. In free space
the two are byte-identical (`F = 0 → target == observed`, verified
element-wise on 301,727 rows), so nothing changes where nothing is touched.

#### Read this before using the numbers

- **`k = 1.0 N/mm` is a declared assumption, not a measurement.** It is written
  into the parquet field metadata (`twm.stiffness_n_per_mm`) and the
  `<episode>.force.json` sidecar so a target pose is never uninterpretable.
  To use a different stiffness, recompute from `force_*_normal_n` directly —
  `penetration = F / k`, `target = observed + (F/k)·n̂`.
- **1 N/mm is on the soft side.** p95 penetration is 5.78 mm and 8.84% of rows
  exceed the 4.25 mm gel thickness. `k ≈ 1.4` keeps p95 inside the gel,
  `k ≈ 1.7` the maximum.
- **Forces saturate at 7.285 N** — the calibration's isotonic stage clips at
  the hardest press it was fitted on, so 0.90% of samples sit exactly at that
  value. Treat the maximum as a floor, not a measurement.
- **Accuracy is rank-order within a group, not a certified absolute scale.**
  Held out by press position the estimator scores ρ = 0.739 / MAE 1.23 N on
  its own calibration objects. On five public force-labelled datasets the same
  pipeline reaches ρ 0.775–0.986. It is reliable for *how hard, relative to
  other frames*; it is not a load cell.
- **Duplicate tactile rows repeat the previous estimate.** The GelSight stream
  is slower than 30 Hz; rows with `tactile_{side}_is_new == False` carry the
  previous frame's force unchanged (forward fill, asserted exact). Filter on
  `is_new` if you need independent samples.
- **Contact must be visible.** A contact clipped by the sensor border is
  systematically under-measured.

Provenance travels with the data: `data/force_export_manifest.json` and
`data/force_export_verify.json` record which calibration produced these
newtons and every check it passed (row alignment 72/72, free-space identity,
round-trip `k·‖target−observed‖ = F` closing to 5.6e-14 N).

Method, validation against five public datasets, and the failure cases:
**https://huggingface.co/spaces/yxma/react-force-recovery**

The preview clips under `data/motherboard/previews/` show the force directly —
a semi-transparent disc on each camera view, centred on that sensor's projected
position, whose **area** is linear in newtons (legend in frame).
