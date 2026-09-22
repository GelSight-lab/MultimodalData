# Force model search results

The 0.5 N target is achievable on the existing calibration distribution.
It is not established on PushT. The candidate models have not replaced the
production estimator.

## Measured results

| Evaluation | Samples | MAE (N) | Position-bootstrap 95% interval |
| --- | ---: | ---: | --- |
| Existing estimator, historical round holdout | 158 | 1.037 | See report.json |
| Image + depth ridge, same round holdout | 158 | 0.305 | 0.236-0.388 |
| Round, four-outer/four-inner position-grouped CV | 478 | 0.246 | 0.207-0.289 |
| Six-shape model, held-out positions | 805 | 0.288 | 0.261-0.317 |
| Nested leave-one-shape-out selection | 2437 | 1.268 | 1.198-1.337 |

The mixed-shape evaluation trains on all six shape types at other positions.
It does not measure unseen-shape or unseen-sensor performance. Per-shape MAE
on its held-out positions is 0.272 (round), 0.368 (quad), 0.224 (star),
0.213 (triangle), 0.343 (B), and 0.362 N (quad_small).
The four outer round folds score 0.228, 0.209, 0.344, and 0.203 N.

## What improved

The round search evaluated 130 configurations: ridge and polynomial models,
RBF support-vector and kernel regression, extra trees, histogram gradient
boosting, and PCA with ridge. Kernel regression uses a fixed-kernel Gaussian
process implementation because the installed scikit-learn KernelRidge calls
a removed SciPy argument.

The selected model uses 759 observation-derived features: signed and absolute
RGB difference grids, reference-normalized differences, channel quantiles,
and depth/contact descriptors. Scaling is fitted within each training fold.
The round model selected ridge alpha=0.1; the mixed-shape model selected
alpha=10 from a separate 15-configuration search. No commanded depth, stage
position, filename, shape label, or force label enters the regression inputs.

Round training uses 320 frames; its historical holdout contains 158 frames
at 83 disjoint rounded press positions. Selection uses four grouped folds
within training. A further four-outer/four-inner grouped evaluation is saved
in report.json. The historical split has been used in prior project work,
so it is a comparison benchmark, not a newly blinded test set.

The old baseline fits its spatial gain field using commanded depths from
the entire cache, including held-out positions. New models do not use that
gain field. The baseline number is retained for comparison and explicitly
marked with this limitation in the artifact.

## PushT transfer check

On the same 835 sampled PushT rows used for the contact-threshold comparison:

| Model | Nonzero fraction in light-contact proxy band | Maximum output (N) | Outputs above 8 N |
| --- | ---: | ---: | ---: |
| Existing estimator, dI=4 | 90.5% | 7.79 | 0% |
| Round image/depth candidate | 52.0% | 146.02 | 3.5% |
| Six-shape candidate | 90.5% | 92.25 | 31.0% |

The light-contact proxy is published tactile intensity in [3, 6), not a
manual contact annotation. PushT has no force ground truth in this evaluation;
these are output-distribution checks, not MAE or contact-recall measurements.
The large extrapolations and the round model's contact losses block deployment.
Clipping predictions to 8 N would conceal these symptoms, not establish accuracy.

For unseen shapes, the nested MAE is 0.534-1.136 N for five shapes, but 3.500 N
for star. The calibration gains therefore do not demonstrate universal contact
dynamics. Force-labeled data from the actual React sensors are needed to
measure whether the 0.5 N target transfers.

## Reproduction and artifacts

Scripts are in `twm/force_recovery/`. Run in this order, with
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`:

```bash
python3 -m twm.force_recovery.model_search --nested
python3 -m twm.force_recovery.multishape_search
python3 -m twm.force_recovery.model_search --transfer
python3 -m twm.force_recovery.evaluate_search_pusht
python3 -m twm.force_recovery.plot_search_results
```

Artifacts: `/media/yxma/Disk1/twm/force_recovery/feature_cache/model_search_2026-09-16/`.
This contains JSON scores and predictions, feature caches, train-only evaluation
models, all-data experimental candidates, and `search_results.png` / `.pdf`.
Reconstruction threshold is dI=4; labels cover 0.15-8 N. The model interface
checks feature version and reconstruction threshold before inference.

Environment: Python 3.9.18, NumPy 1.26.4, SciPy 1.13.1, scikit-learn 1.1.1.
Grouped selection and fold-local preprocessing follow the
[scikit-learn leakage guidance](https://github.com/scikit-learn/scikit-learn/blob/main/doc/common_pitfalls.rst).
