# force_recovery — module map

44 modules, ~12.6k lines. Most of them are studies that each produced a
published number; only a small core runs in the pipeline. This file says which
is which, so nobody has to read the import graph again.

**Start here:** `pipeline.py` — the stable public API (reconstruct → force →
action) and the measured caveats that go with any number it produces.

## Core (imported by many; changing these changes results)

| module | role | imported by |
|---|---|---|
| `debug_gallery.py` | **`stages()` — the reconstruction core**: dI → LUT → valid mask → `fast_poisson` → depth + features | 8 |
| `lut_calibration.py` | LUT definition, sphere self-calibration (`a² = d(2R−d)`), `crop`, `MM_PER_PIXEL` | 11 |
| `run_episode.py` | batch force estimation over release episodes; roots (`DATA_ROOT`, `STAGE_ROOT`, `OUT_ROOT`) | 24 |
| `marker_removal.py` | marker-dot inpainting for the **depth/3D path only** (never the force features) | 5 |
| `o3d_view.py` | Open3D mesh rendering, halo-pedestal removal, content crop | 4 |
| `dexforce.py` | force → virtual position target | 5 |
| `evaluate.py` | shared evaluation helpers | 5 |
| `pipeline.py` | public API façade over the above | — |

> **Naming debt, deliberately not fixed:** the reconstruction core lives in
> `debug_gallery.py` because that module was written first as a diagnostic.
> Eight modules import `stages` from there. `pipeline.reconstruct` re-exports
> it so new code need not know; moving the definition would be a wide,
> risk-only refactor.

## Per-dataset adapters

`sparsh_data.py` · `sparsh_lut.py` (sensor-general self-calibration) ·
`faf_extract.py` (range-extraction from the FeelAnyForce split zip) ·
`fota_cnc_fetch.py` (same trick on FoTa's 392 GB archive) · `feats_infer.py`

## Studies — each produced a number that is on the site

| module | what it settled |
|---|---|
| `mnist_validation.py` | first external per-pixel GT: 11 µm @0.3 mm → 281 µm @2.25 mm |
| `improvement_study.py` | 5 candidates, 2 adopted; PINN-lite rejected by a `g(volume)` control |
| `marker_study.py` | marker removal fixes geometry (dimple ×0.65, p=2.6e-19), not force |
| `recon_study.py` | 20-sample workbench, every stage and knob |
| `force_eval_all.py` | one protocol over all datasets, each with its shuffle control |
| `faf_validation.py` | FeelAnyForce admitted for 14/42 captures (ρ 0.961 vs 0.338 control) |
| `goal6_final_eval.py` | letter recognition 98.2%, per-indenter ρ ≥ 0.975 |
| `fota_cnc_validation.py`, `glowtact_validation.py`, `fota_validation.py`, `external_validation.py` | earlier per-dataset validations |

## Site builders

`site.py` (index) · `results_page.py` · `method_page.py` · `actions_page.py` ·
`debug_page.py` · `showcase.py` (galleries + clips) · `sparsh_figure.py` ·
`visualize.py` · `publish_space.py` · `finalize.py`

`design_guard.py` — layout regression gate; run it before publishing.
`test_units.py` — unit tests for the pure-function core.

## Attic — superseded, kept for provenance

| module | why it is dead |
|---|---|
| `depth_force.py` | the **v1 MLP pipeline**. Its depth amplitude was ~25% of truth, which capped every force model built on it. Superseded by the LUT path. Still imported by three older validation scripts, so it is not deleted — do not build anything new on it. |
| `fusion_recon.py` | LUT+MLP gradient-band fusion. Rejected: the MLP's "sharp edges" were `\|n\|≥1` saturation artefacts (rms 220 vs LUT 0.027); the best of 8 sign conventions correlated 0.049 with the LUT. |
| `optimize.py`, `normalize_scale.py`, `feature_cache.py` | v1-era feature/scale fitting, all on `depth_force`. |
| `anyforce_react.py`, `anyforce_offline.py` | FeelAnyForce baseline inference, kept for the results matrix. |

## Reproducing

```bash
# reconstruction workbench (needs a display for Open3D)
xvfb-run -a -s "-screen 0 1400x1000x24" python -m force_recovery.recon_study glowtact
python -m force_recovery.force_eval_all          # all datasets + shuffle controls
python -m force_recovery.improvement_study all   # the five improvement candidates
python -m force_recovery.design_guard            # layout gate, exits non-zero on regression
python -m force_recovery.test_units
```

Data roots are all in `run_episode.py`. Heavy artefacts live under
`/media/yxma/Disk1/twm/force_recovery/`, never in the repo.
