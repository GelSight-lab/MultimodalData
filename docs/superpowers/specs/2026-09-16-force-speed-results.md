# v8 CPU Inference Speed

## Scope

The 2026-09-16 optimization preserves the v8 model and NPZ contract. It does
not change contact thresholds, references, noise gates, calibration, precision,
image resolution, alignment, duplicate handling, geometry fields or stiffness.
Version remains 8: this is an implementation optimization, not recalibration.

Changes:

- Reuse the contact mask between CF reconstruction and its gradient calculation.
- Omit diagnostic normal maps only inside `react_calib.force_stages`; the
  general reconstruction API still returns them by default.
- Cache Neumann denominators and full trend-coordinate bases by shape.
  Both caches have at most four entries and contain read-only arrays. The
  contact-dependent least-squares fit is still evaluated on every frame.
- Replace generic three-channel maximum reductions with equivalent elementwise
  maxima in CF and LUT contact detection.
- Return zero depth directly when the contact mask is exactly empty. A small
  or weak nonempty contact still runs the original solver. No force threshold
  or task-specific shortcut was added.

## Paired Benchmark

Four tasks, both sensors, 23 unique fresh frames per side: **184 frames**.
Samples combine uniform temporal coverage and intensity-stratified rows,
including the lowest and highest intensity. Each arm/mode runs three times
in alternating order; results sum the per-side median times.

Host: Intel i7-6700K, four physical cores/eight logical CPUs, Python 3.9.18,
NumPy 1.26.4, OpenCV 4.5.5, SciPy 1.13.1. BLAS and OpenMP threads were set to
one; OpenCV retained its eight-thread default. Other capture/preprocessing
work was running. This is a single-worker benchmark, not a concurrency sweep.

| Mode | Before | After | Speedup |
|---|---:|---:|---:|
| In-memory reconstruction + force | 58.76 ms/frame | 31.87 ms/frame | 1.84x |
| Same work including H5 read/crop | 60.64 ms/frame | 34.27 ms/frame | 1.77x |

H5-inclusive speedup by task, combining both sensors:

| Task | Speedup |
|---|---:|
| PushT | 1.55x |
| Motherboard | 2.10x |
| Rope | 1.69x |
| Toy | 1.83x |

Every compared force value was identical: maximum absolute difference **0 N**.
CF and LUT intermediate dictionaries were checked field by field, including
contact masks and depth. Maximum LUT depth difference was **0 mm**. Contact
decisions at 0.02 N did not change. The default diagnostic reconstruction,
including its normal map, was also compared once per sensor-side.

The benchmark uses warm caches after correctness checks and excludes reference
setup, model fitting and NPZ writing. Initial sparse-read profiling was slower
(about 99 ms/frame); it is not the denominator for the paired speedup. Disk
contention, empty-contact fraction and worker count affect whole-run throughput.
This does not establish a globally optimal implementation or a new accuracy
claim. React still has no force ground truth.

## Verification

The focused force/preview/pipeline suite plus force geometry utilities passed:
157 tests, including 25 new speed/equivalence tests. The analytic Poisson edge
test reported zero problems. A read-only independent review found no issues
in the scoped changes.

Full `process_side` runs also compared the original and optimized writers:

| Episode | Sensors | Rows per sensor | Fields per NPZ |
|---|---:|---:|---:|
| `pushT/2026-09-12/episode_004` | 2 | 1693 | 26 |
| `rope/2026-09-14/episode_004` | 2 | 250 | 26 |

All 26 fields in all four sensor-side files were array-equal, including force,
geometry, held source indices, reference metadata and retained depth maps.
This covers 3886 output rows. These were isolated comparison writes, not
production replacement or a whole-dataset run. Full-writer elapsed times are
retained for diagnosis, but are not a repeated cold-disk throughput benchmark.

Benchmark artifacts are outside git:
`/media/yxma/Disk1/twm/force_recovery/speed_review_2026-09-16/`.
`benchmark.json` stores sampled rows, all repeated timings, maximum errors and
SHA256 hashes of baseline/current source files. `force_recovery/` is the
frozen pre-optimization package, including the uncommitted v8 calibration work.
Git HEAD alone would not reproduce that baseline.
`episode_checks.json` records the full-writer comparisons; the corresponding
NPZs are under `before/` and `after/`, never the production task directories.

## Reproduce

From the repository root, with the three reviewed force assets available:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m twm.force_recovery.benchmark_speed \
  --baseline-dir /media/yxma/Disk1/twm/force_recovery/speed_review_2026-09-16/force_recovery \
  --frames 16 --repeats 3 --out /tmp/force-speed-repeat.json
```

The output report path must not already exist. Keep the snapshot directory
named `force_recovery`: the old package contains historical absolute imports.
The command reads source data and writes only its JSON report, not force NPZs.
The source episodes are listed in `benchmark_speed.JOBS`; it does not select
live recordings or walk the entire dataset.

For full reprocessing use the [runbook](../../../twm/force_recovery/RUNBOOK.md).
The optimization is on by default; no `--fast` switch or asset rebuild is needed.
