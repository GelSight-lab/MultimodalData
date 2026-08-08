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
- **The artifact must name its own calibration.** Every npz carries
  `force_calibration` (from `react_calib.CALIBRATION_NAME`, the single source)
  and `pipeline_version`; `export_force_columns` refuses anything below the
  current version instead of exporting it. Bump `batch_worker.PIPELINE_VERSION`
  whenever the newtons change, or the rerun silently keeps the old ones.

> Both halves of that rule are scar tissue. The old scale was a lone float
> `scale_n_per_mm3`, which could not describe an estimator that is a gain field
> plus a clipping correction plus an isotonic fit — and because nothing
> recorded *which* map ran, pixel-unit weights scored mm-unit features for
> weeks at end-to-end ρ 0.143. Separately, `np.savez` was silently dropping
> every str-valued field, so even the name that existed never reached disk.
> And the first rerun after the fix rewrote 1 of 72 files because the version
> had not been bumped.

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

The force disc's **area** is linear in newtons (`radius ∝ √F`) — a
radius-proportional dot exaggerates large forces quadratically to the eye. The
legend uses the same `radius_px`, so picture and number cannot drift. The
manifest records, per clip, whether a disc is present, so "no disc" reads as
*no force data for that date* rather than a rendering failure.

It is drawn **on the camera views, at the sensor's projected position**, inside
`viz.draw_projection_overlay` — not by the caller, and not on the tactile
tiles. Two reasons, both learned the hard way: a disc over the GelSight tile
hides the very signal it annotates, and a second, independent projection of the
sensor would eventually disagree with the pose axes, which would read as a
calibration error rather than the drawing bug it is.

```bash
python scripts/verify_force_overlay.py <path/to/episode.h5>   # measures, not eyeballs
```

That verifier asserts the disc's ink lies in row 1 and **nowhere** in row 2,
that it covers the projected sensor point, and that a sensor outside a
camera's frustum draws nothing in that view. It took three tries to make it
measure the right thing, and each failure is a general one:

| the check said | it was actually measuring |
|---|---|
| "22% coverage" on a correct render | hue, minus the pre-overlay orange — which erased every disc pixel landing on the reddish motherboard. An alpha blend is defined by *which pixels changed*, not which look orange. |
| "ink leaked into the tactile row" | the pose **axes**, which legitimately extend past a view. The fix is to diff two renders differing only in the force argument. |
| nothing at all | the real bug it then found: `project_gel_pose` returns out-of-frustum coordinates, so a disc clipped only to the canvas painted onto a *neighbouring* tile. Clip to the view the disc annotates. |

The rule underneath: **a differential measurement must differ in exactly one
thing.** Twice the control render varied in more than the quantity under test,
and both times the verifier's verdict was about the extra variable.

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
   give 8.84%.
   And **a prediction about how a number will move is not a measurement of it.**
   When the calibration fix cut peak force 2.5× (18.3 → 7.29 N) I wrote that
   the penetration verdict would "likely reverse". It did not: p95 went
   5.69 → 5.78 mm and the fraction past the gel 7.85% → 8.84%. The bad map had
   inflated a thin tail, not the distribution. Had I edited the page from the
   prediction instead of re-running `verify`, I would have replaced correct
   numbers with confident wrong ones.
4. **A guard that cries wolf gets ignored.** The first raw-indexing check
   flagged four lines that were already corrected; it now inspects the index
   expression.
5. **Adding a branch means re-reading the guards above it.** A stale
   `if "vol" in row` check silently dropped 28 captures after a `med_*` branch
   was added below it.
6. **Exit code 0 is not success — and a gate you never saw fail is not a
   gate.** The upload's decode check read ffmpeg's return code; ffmpeg prints
   "partial file" for a truncated mp4 and *still exits 0*, so the gate passed
   a clip that decoded 79 of its 900 frames. It only surfaced because a
   deliberately truncated copy was fed to it. Every gate here now gets that
   treatment: reintroduce the defect, watch it fail, then fix it back. The
   tactile-lag guard was proved the same way.
