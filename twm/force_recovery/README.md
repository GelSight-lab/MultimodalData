# React Force Reconstruction v8

Estimate normal force from the two GelSight Mini streams. The current model
outputs 0-15 N, with a fitted endpoint of 14.99 N. It retains v7's contact
mask, reference-noise gate and low-score calibration, and extends the high end
using measured 8-15 N calibration presses.

**For the data processor:** follow [RUNBOOK.md](RUNBOOK.md). It covers input
selection, required assets, isolated batch processing, resuming, validation
and the conditions for publishing. Do not run a whole-dataset migration from
the historical research commands.

**Controller convention:** virtual targets use the shared **2 N/mm (2000 N/m)**
stiffness and the sensor's local **−Y** normal, rotated into the pose's world
frame. `F/k` is a controller displacement, not measured gel compression:
15 N gives 7.5 mm and is not rejected merely for exceeding gel thickness.
The exporter retains identity, alignment, round-trip and 100 mm displacement
sanity checks; these are data checks, not a robot-safety certification.

The linked seven-clip v7/v8 comparison is a historical model review, not proof
of current dataset coverage or publication. Verify those against the specific
run's inventory, validation report and remote commit receipt.

- [Four-task v7/v8 video review](https://yxma-react-force-recovery.static.hf.space/task-review-2026-09-16-range15/index.html)
- [Measured results and limitations](../../docs/superpowers/specs/2026-09-16-force-range-extension-results.md)
- [CPU speed benchmark](../../docs/superpowers/specs/2026-09-16-force-speed-results.md):
  1.77x faster including warm H5 reads on 184 paired frames, with identical force values.
- [Module map](ARCHITECTURE.md)
- [Historical research notes](RESEARCH_NOTES.md)

## Algorithm

1. Use the release parquet's row count, tactile intensity and `is_new` flags.
   Get each side's raw capture indices from
   `open_episode(raw_h5, task).align[side].index_map`.
2. Select up to 15 low-intensity fresh rows, spaced at least 30 rows apart.
   Sort them in time and use the first 12. Crop each RGB frame with the shared
   `crop()` function, then take the median image as the reference.
3. Estimate the reference noise area from the 90th percentile of leave-one-out
   reference residual contact areas. This assumes mostly unloaded reference
   images; it does not establish ground-truth zero force.
4. Run calibration-free reconstruction with `VALID_DI=4`. It returns the
   contact mask and positive depth in arbitrary units. Compute the five
   features in `react_calib.feature_vector`, spatial gain and crop correction.
5. Apply the v7 linear score and preserved low-score isotonic curve. Above its
   final knot, use the anchored monotone tail fitted on measured high loads.
   Clip outside fitted support, then multiply by the continuous contact weight.
6. Reconstruct fresh captures; hold force, geometry and `source_frame` on
   duplicate rows. Compute the separate LUT geometry fields alongside force.

There is no per-episode force rescaling. A low-force rope sequence should not
be stretched to fill 15 N. A zero estimate means insufficient evidence above
the reference-noise gate, not a measured absence of force.

CPU optimizations are enabled by default: fixed-grid caches, contact-mask
reuse, no unused normal maps in force inference, and an exactly empty-mask
fast path. Weak nonempty contacts retain the same reconstruction. All geometry
fields remain present; this is not a reduced-output or approximate mode.

## Python API

Prefer the episode writer for dataset processing. It owns alignment,
references, duplicate handling and provenance:

```python
from pathlib import Path
from twm.force_recovery.run_episode import process_side

# Writes only to this explicit candidate destination. Does not upload.
metadata = process_side(
    "pushT", "2026-09-11", "episode_001", "left",
    out_dir=Path("/media/yxma/Disk1/twm/force_v8_smoke/pushT/2026-09-11"),
    keep_top_depths=3,
)
```

For individual frames, fit once and reuse the returned callable. The arrays
below are RGB, not OpenCV BGR, and retain the original 0-255 intensity scale:

```python
import numpy as np
from twm.force_recovery.react_calib import fit, force_stages
from twm.force_recovery.lut_calibration import crop
from twm.force_recovery.run_episode import reference_stack, reference_noise_area

predict = fit(report=False)

def prepare_reference(frames, index_map, intensity, is_new):
    raw_refs = reference_stack(frames, index_map, intensity, is_new)
    refs = np.stack([crop(im).astype(np.float32) for im in raw_refs])
    return np.median(refs, axis=0), reference_noise_area(refs)

def estimate(raw_rgb, reference, noise_area_mm2):
    image = crop(raw_rgb).astype(np.float32)
    return predict(force_stages(image, reference),
                   noise_area_mm2=noise_area_mm2)
```

Pass raw frames to `crop()` exactly once. Each sensor-side/episode needs its own
reference and noise estimate. For release rows, use `process_side` rather than
reimplementing its fresh-frame hold. Do not add a manual `+15` frame shift.

`pipeline.reconstruct()` is the LUT geometry API. Its output is not the input
to the deployed v8 force predictor; use `react_calib.force_stages()`.
`pipeline.force_from_depth()` remains a historical evaluation helper.

## Output Contract

One file per sensor-side:
`<force_root>/<task>/<date>/<episode>_<left|right>.npz`.
Array lengths equal the uncut input parquet's row count.

| Field | Meaning |
|---|---|
| `force_normal_n` | Estimated normal force, float32, finite and bounded to 0-15 N |
| `source_frame` | Integer index of the actual raw GelSight image used, held on duplicates |
| `volume_mm3` | LUT-derived geometry volume, separate from the force reconstruction |
| `contact_area_mm2` | LUT-derived geometry contact area, not the v8 force contact mask area |
| `max_depth_mm` | LUT-derived nominal depth; calibration-free depth is not in mm |
| `reference_rows` | Selected release-row indices; only the first 12 build the reference |
| `reference_noise_area_mm2` | Per-side/episode noise floor used in the contact gate |
| `pipeline_version` | `8` |
| `force_calibration` | Exact `react_calib.CALIBRATION_NAME`, also stored as `scale_source` |
| `force_reconstruction` | `calibfree` |
| `geometry_reconstruction` | `stages (LUT, millimetres)` |
| `valid_mask_dI` | `4.0` |
| `force_calibration_max_n` | Configured range limit, `15.0` |
| `force_calibration_ceiling_n` | Fitted endpoint, currently `14.99` |
| `absolute_force_validated_on_react` | `False` |

`task`, `date`, `episode`, `side`, `trim` and `tactile_timestamped` identify the
recording and alignment. Optional `depth_row_*` arrays are diagnostic samples,
not a dense depth stream. This process does not rewrite tactile RGB videos.

## Accuracy and Limits

Held out by calibration press position, with paired v7/v8 predictions:

| True force | Samples | v7 MAE | v8 MAE |
|---|---:|---:|---:|
| <=8 N | 158 | 1.044 N | 1.044 N |
| >8 to 15 N | 125 | 4.957 N | 3.153 N |
| Combined | 283 | 2.772 N | 1.975 N |

These are calibration-domain errors, not React errors. React has no force
ground truth. There is no 0.5 N accuracy guarantee or OOD guarantee; 57 of 125
high-force test samples score below the tail join and remain underestimated.
Changes in gel, illumination, reference state, cropping and edge contact can
change the force scale. A new sensor requires its own validation.

For an exact low-range comparison, `fit(extend_range=False)` reproduces v7.
Do not increase a cap, multiply the output by a constant, or stamp old files
as v8 instead of recomputing them.
