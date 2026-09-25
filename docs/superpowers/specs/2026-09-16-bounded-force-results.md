# 0-15 N bounded force experiments

Bounded output is achievable. OOD immunity and 0.5 N accuracy across unknown
sensors/shapes are not established. The deployed force channel is unchanged.

## Data and protocol

The six existing calibration families contain 4,709 labeled frames from
0.07 to 15.00 N, including 2,261 above 8 N and 286 below 1 N. No measured
exact-zero examples occur in this set. Earlier 0-8 N caches are preserved;
low-force and previously omitted weak-contact frames are included when
building the expanded cache.

Distinct rounded press positions partition the data into 2,062 fitting,
1,044 support/error-calibration, and 1,603 test frames. Model selection uses
leave-one-shape-out CV within the fitting partition. The calibration set
sets a 95th-percentile neighbor-distance threshold and a 90th-percentile
absolute-error band; neither is claimed to give coverage under distribution
shift. The saved estimator is not refitted on calibration or test samples.

Twelve configurations compare nonnegative deformation regression, weighted
nearest-neighbor interpolation, extra trees, and ridge image scores with
bounded isotonic calibration. The nonnegative model is physics-inspired;
it is not a finite-element model with measured material parameters.

## Results

| Evaluation | MAE (N) | Accepted fraction |
| --- | ---: | ---: |
| Selected interpolation model, all position-test samples | 1.208 | 100% scored diagnostically |
| Selected interpolation model, accepted position-test samples | 1.135 | 91.0% |
| Bounded ridge alpha=10, position-test diagnostic | 0.558 | 100% scored diagnostically |
| Bounded ridge alpha=10, accepted position-test diagnostic | 0.547 | 90.8% |
| Nested unseen-shape selection, all samples | 2.273 | 72.0% pass support checks |

The selected model is distance-weighted 5-neighbor regression on combined
image/depth features. It wins the fitting-partition shape-CV comparison
(1.472 N versus 1.865 N for the best bounded ridge variant). The position-test
ridge diagnostics are descriptive and were not used to change this selection.
They show why a lower same-distribution error cannot choose a transfer model.

The first nine-configuration interpolation/physics experiment is preserved
as `interpolation_report.json` and `interpolation_guarded_model.joblib`.
Adding bounded ridge alternatives did not change the selected position-test
model. In the outer unseen-star fold, however, selection on the other shapes
chose bounded ridge and its MAE rose to 5.603 N. Even among the 6.5% of star
frames passing the support check, MAE was 5.006 N. Rejection can miss bad cases.

## Sensor transfer

Cross-sensor diagnostics deliberately bypass the domain restriction to test
whether the statistical support check alone would suffice:

| Target | Frames | MAE before rejection | MAE among accepted | Accepted fraction |
| --- | ---: | ---: | ---: | ---: |
| FoTa CNC Mini | 192 | 0.974 N | 0.974 N | 100% |
| FeelAnyForce, unloaded-reference captures | 192 | 2.678 N | 0.920 N | 28.1% |
| PushT | 835 | Unknown | Unknown | 63.1% |

The FoTa subset covers approximately 1-8 N; it does not validate 8-15 N
transfer. PushT has no force ground truth here. Its diagnostic bounded
predictions span 0.154-13.858 N, with zero outputs above 15 N. That is a
numerical property, not evidence of correct newtons.

Actual inference rejects all three unregistered domains. This is an explicit
deployment restriction, not a successful automatic OOD detector. The only
registered domain is the current CNC calibration dataset; its name is not
a verified hardware serial number. It must not be relabeled as a React sensor
without target-sensor calibration and validation.

## Inference contract

`bounded_force.predict_frame(bundle, image, reference, domain=...)` returns
`force_n`, `supported`, `reason`, and an empirical error band. Accepted forces
are within 0-15 N. Unknown domains, distant inputs, nonfinite values, and
unsupported regression scores produce NaN with an explicit reason. They are
not zero-force labels. Feature version and contact threshold are checked.

The bound and rejection rules cannot guarantee that every accepted input is
in distribution. The measured transfer failures above demonstrate that limit.
Actual React sensors need force-labeled, representative contact data to test
the desired 0.5 N accuracy over 0-15 N, including low-force/release states,
multiple contact geometries, pad locations, and recording sessions.

## Reproduce

Use `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` with:

```bash
python3 -m twm.force_recovery.bounded_force_eval --nested-shapes
python3 -m twm.force_recovery.bounded_force_eval --position-diagnostics
python3 -m twm.force_recovery.bounded_force_transfer
pytest tests/test_bounded_force.py tests/test_force_model_search.py tests/test_force_recalibration.py tests/test_force_frame_alignment.py tests/test_force_verify_stale.py -q
```

Artifacts and experimental model:
`/media/yxma/Disk1/twm/force_recovery/feature_cache/bounded_force_0_15_2026-09-16/`.

## Relevant references

[OOD detection learnability](https://proceedings.neurips.cc/paper_files/paper/2022/file/f0e91b1314fa5eabf1d7ef6d1561ecec-Paper-Conference.pdf)
studies the assumptions required for generalization guarantees.
[FeelAnyForce](https://www.prg.cs.umd.edu/FeelAnyForce) targets forces up to
15 N and provides a sensor-adaptation calibration procedure in its
[implementation](https://github.com/prgumd/FeelAnyForce#-calibration).
Neither provides evidence that our present PushT estimates have 0.5 N MAE.
