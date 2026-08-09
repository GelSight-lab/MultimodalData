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

## 0b. If a recording will not open: the build already handles it

`python -m react_preprocess build` diagnoses and, for one specific signature,
repairs an unopenable recording on its own. You do not have to do anything.
What you **do** have to do is read the line it prints, because the honest
outcome is usually a refusal:

```
episode_004: RECOVERED-NOT-PUBLISHABLE — using existing recovery
  (episode_004.recovered.h5), but it cannot become an episode:
  missing ['timestamps'] — recovery cannot rebuild what was still in the
  metadata cache, and interpolating timestamps misplaces frames by 15-1431
[build] done (0 failed, 1 recovered but not publishable)
```

A refusal is not a failure and does not stop the build, but it does mean a
recording exists that the release does not contain. `--no-repair` skips the
recovery attempt (it rewrites the recording's worth of bytes, ~35 min for
79 GB).

**The failure.** `bad object header version number` means the recorder died
without `close()`. HDF5 streams raw chunks straight to disk but keeps object
headers, chunk B-trees and the superblock's EOF field in a metadata cache, so
every pixel is present and nothing can reach it. `repair.diagnose` confirms
this rather than assuming it — EOF stuck at its creation value, consistency
flag reading open-for-write, root header version not 1 — and reports anything
else as `unknown-damage` without touching it. To look yourself:

```
python scripts/h5_forensics.py <broken.h5> <a_healthy_sibling.h5>
```

**Prevention is the real fix.** `HDF5Writer.FLUSH_INTERVAL_S` (10 s) turns
"lose the recording" into "lose the last few seconds".
`scripts/test_crash_leaves_readable_h5.py` kills a writer with and without it
and requires the unflushed arm to reproduce the exact error above — if that
arm ever starts opening, the test fails loudly, because it is then measuring
nothing.

**What recovery gets back, and what it must never invent.** It returns what
HDF5 had already evicted. For pushT/2026-06-18/episode_004 that is all eight
image streams — 15,447 complete frames, byte-verified — but only 2 of 16
timestamp chunks and no usable OptiTrack poses. Do **not** interpolate the
missing timestamps: a line through the two surviving anchors, checked against
episodes whose timestamps are known, misplaces frames by 15.6, 24.5 and 1431,
and the tactile lag this pipeline exists to get right is 15.

`repair.release_eligibility` is the gate, its default answer is no, and
`scripts/test_repair_refuses.py` reintroduces each defect (no timestamps,
short timestamps, no poses, unfamiliar damage, a `.recovered.h5` leaking into
`discover`) to watch it refuse. **A recovered recording with no timestamps is
video, not an episode.**

Two rules recovery will not let you skip, both learned by getting them wrong:

* **Use the filter mask from the B-tree key, never the chunk size.** "Exactly
  921,600 bytes means unfiltered" holds for 10,635 chunks of one stream and is
  wrong for 37 more, which are blosc output that lands on that length.
* **Verify stream identity separately from byte fidelity.** Comparing the
  recovered file against the index it was written from proves only that the
  copy is faithful; both sides use the same index. `verify_stream_identity`
  reads each dataset *by name* and correlates it against a healthy sibling.
  Watch the weak margins: single frames separate the two GelSights only
  0.995 vs 0.908, while averaging 41 frames cancels contact and leaves the
  sensor's own fixed pattern at 0.9996 vs 0.929.

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

**Extrinsics are per task, world offset per episode — `calib_epoch` owns
both.** The cameras were recalibrated between tasks (May-12 for motherboard,
June-26 for pushT; |dT| 53–64 mm, 35–73 px of projection error between the
epochs) and the 2026-05-19 session redefined the world origin by
(0.23, 0, 0.175) m. Both facts are recorded in the dataset itself
(`data/<task>/calibration/`, `episodes.jsonl`) and were nonetheless hard-coded
to one value in the builder — every motherboard preview shipped with pushT's
extrinsics, and 05-19 carried both errors at once, which is why it looked
worst and got reported first. The builder now reads `calib_epoch`, refuses a
wrong epoch, and bakes `calib <date> world+(dx,dy,dz)` into the status bar so
a viewer can catch the wrong epoch without trusting the pipeline.

**The clip window starts at the RELEASE trim, read from the release parquet —
never from a sidecar with a zero fallback.** `_get_trim_offset` read a
processed/.pt file that pushT never had and "fell back to 0", so pushT
previews played the H5 pre-roll: up to 6.5 minutes *before* the episode,
sensors parked, action frozen — reported as frozen actions in clips 2–4. The
force rows were already using the release trim; the same fact had two
sources, one right and one silently wrong.

**Frames the release catalogues as bad are labelled ON the frame.** The
curation (same detectors and thresholds for every task) records OT freezes,
pose teleports and tactile intensity spikes in `bad_frames.json` and excludes
them from `segments.json` training spans — but previews play the raw stream.
An unlabelled magenta GelSight flicker or frozen pose reads as dirty data;
with the red `FLAGGED <detector>` banner it reads as what it is: a known,
indexed dropout that training never sees.

Video corruption (`cam_corruption` / `tactile_corruption`) is detected from
the published MP4s themselves — the sidecar scalars cannot see a torn frame.
Two signatures, both earned: **burst-bracket** (two hot frame-diff boundaries
≤ 15 frames apart bracket an anomalous run; the single-frame "differs from
both neighbours" test scores a 9-frame magenta flicker 0, because its
interior diffs are calm) and **row-band tear** on an RGB per-channel-max
diff (grayscale measured the magenta flicker at under half its amplitude;
a torn GelSight frame moves chroma, not luma). Validated: exact hit on the
known 2026-05-11/ep003 flicker [247,255], two previously unindexed partial
tears found in the same episode ([2080,2089], [8583,8593] — bottom-band
magenta), zero false positives from fast arm motion on the colour views
(the naive "band changed while rest static" tear test flagged exactly that
motion, three times, before eyeballing killed it).

The uploader enforces two more gates, both proven on the real defects:
**previews must mirror the release** (a clip for an unpublished episode is
refused — four such orphans sat on the dataset for months and were reported
as "not updated"; they could never be updated, there was nothing to update
them from), and **clips must post-date the renderer sources** (31 of 32
motherboard clips on disk were rendered 30 minutes before the epoch fix and
were indistinguishable from good output).

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
