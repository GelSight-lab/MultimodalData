# Data processing workflow

The order below is not a preference. Each step exists because skipping it
produced a defect that reached the published dataset or site at least once.

Run the two gates before anything is published:

```bash
python -m twm.pipeline_guard              # pipeline invariants
python -m twm.force_recovery.design_guard # figure layout invariants
python -m twm.force_recovery.test_units
```

---

## 0. Before touching a new recording: establish its alignment

**Never index `gelsight/<side>/frames` with a camera index.** The GelSight
capture lagged the camera by 15 frames (0.5 s at 30 fps) on every recording up
to 2026-06-18 — a V4L2 buffer that was never flushed, fixed in the rig on
2026-06-27. The published release has this baked out already
(`status: CORRECTED_IN_DATA`); **the raw H5 does not.**

```python
from twm.tactile_align import gel_index, gel_lag_frames, describe
gs = f[f"gelsight/{side}/frames"][gel_index(f, cam_i, side)]
status_line += describe(f)        # shows "gel+15f" so the viewer sees it
```

`gel_lag_frames` decides **per file** (does this recording carry per-sensor
timestamps?), because applying the constant to an already-aligned recording is
the mirror-image bug. `pipeline_guard` fails the build if any module
redeclares the constant or indexes frames without it.

> This is the defect that shipped: the preview builder paired `camera[i]` with
> `gelsight[i]` and 40 clips went to the dataset repo with the tactile tiles
> half a second ahead of the video beside them.

## 1. Preprocess → release

`react_preprocess` writes `release/<task>/meta/<date>/<episode>.parquet` plus
videos. Tactile scalars are shifted here, so **release rows are already on the
camera timeline**; force estimation reading raw H5 still applies the shift.

## 2. Force estimation

```bash
python -m force_recovery.run_episode          # → force_recovery/<task>/<date>/<ep>_<side>.npz
```

Reconstruction is `stages()` (dI → per-sensor RGB LUT → valid mask →
`fast_poisson`). Two rules the guard enforces:

- **Cosmetic steps never touch force features.** Marker inpainting and
  halo-pedestal removal are figure-only; measured on FEATS they *cost* accuracy
  (ρ 0.775 → 0.737). They may appear in the force module only behind an
  explicit flag, as a reported control.
- **Calibrate per sensor.** The LUT is self-calibrated from sphere presses via
  `a² = d(2R−d)`. Using another sensor's table is a documented failure: on
  Sparsh the borrowed table's gradients sat 93.3° from truth (chance is 90°)
  while bin coverage still read 90–97%. **Coverage does not detect
  out-of-domain use; the gradient angle does.**
  *Open issue: React itself has no sphere presses, so its depth currently uses
  the GlowTact table — 18% of contact pixels fall in uncalibrated bins vs 6% on
  the host sensor.*

## 3. Export force as observation + action

```bash
python -m force_recovery.export_force_columns export
python -m force_recovery.export_force_columns verify
```

Writes `force_{side}_{normal_n,penetration_mm,target_pose}` to a parallel
`release_force/` tree. Stiffness has **one** definition
(`dexforce.STIFFNESS_N_PER_M`) and is written into the parquet field metadata
and a sidecar — a target pose whose stiffness lives only in code is unreadable.

`verify` must show: row-count alignment 100%, no-contact rows identical to the
observed pose element-wise, and the round trip `k·‖target−observed‖ = F`.

## 4. Previews

```bash
python scripts/build_episode_previews.py --task <task> --date <date>
python -m force_recovery.upload_previews --dry-run   # then without
```

The force dot's **area** is linear in newtons (`radius ∝ √F`) — a
radius-proportional dot exaggerates large forces quadratically to the eye. The
legend uses the same `radius_px`, so picture and number cannot drift. The
manifest records, per clip, whether a dot is present, so "no dot" reads as
*no force data for that date* rather than a rendering failure.

## 5. Evaluation — the part that is easy to fake

```bash
python -m force_recovery.force_eval_all
```

Every headline number ships beside a **within-group label shuffle**. A global
shuffle proves nothing: on FeelAnyForce the pooled ρ was 0.455 and survived
global shuffling at 0.442, which is how we caught that its frame join had
never been demonstrated.

Also standard here, each earned by a past mistake:

| check | why |
|---|---|
| in-view / clipped split | contacts clipped by the sensor border are systematically under-measured (cnc: ρ 0.11 all positions vs 0.94 in view) |
| a "dumber" baseline in the control set | a dimensionless contact law beat the 5-feature model until `F = g(volume)` beat *it* — the gain was fewer features, not the physics |
| report the press depth with any depth accuracy | 11 µm at 0.3 mm, 281 µm at 2.25 mm; a single number is meaningless |
| state that ρ is within-group | stiffness is absorbed into per-group weights; it is not a transferable newton scale |

## 6. Publish

```bash
python -m twm.pipeline_guard && python -m force_recovery.design_guard
python -m force_recovery.publish_space          # site
python -m force_recovery.upload_previews        # dataset previews
```

Regenerate figures whenever a constant they depend on changes — changing text
without regenerating leaves the page contradicting the data. Site copies are
md5-checked against their sources by `design_guard`, because "regenerated but
not copied to the site" has happened twice.

---

## Verification habits worth keeping

Collected from things that went wrong here, not from a style guide:

1. **A check that passes on half the frame is not a pass.** The force-overlay
   verifier scanned `x ∈ [0,480)` — the left tiles only — and reported "no
   overlay" for a clip whose dot was on the right sensor.
2. **`max` is not a distribution.** "Reference frames differ by up to 12 grey
   levels" was a handful of outlier pixels; p95 was 3.0.
3. **One episode is not a dataset.** "0% of penetrations exceed the gel" came
   from one sensor-side whose forces peaked at 2.3 N; the full 480,080 samples
   give 7.85%.
4. **A guard that cries wolf gets ignored.** The first raw-indexing check
   flagged four lines that were already corrected; it now inspects the index
   expression.
5. **Adding a branch means re-reading the guards above it.** A stale
   `if "vol" in row` check silently dropped 28 captures after a `med_*` branch
   was added below it.
