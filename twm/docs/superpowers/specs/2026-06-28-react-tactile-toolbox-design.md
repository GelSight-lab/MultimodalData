# React Tactile Toolbox — Design

**Date:** 2026-06-28
**Status:** Approved (user delegated autonomous build + self-verification)

## Goal
Ship a lightweight, zero-install toolbox alongside the React dataset so others
can easily load and derive standard VBTS signals. GelSight Mini, 640×480,
**markerless**, **no calibration** available.

## Form
- `toolbox/` folder in `yxma/React` HF repo (clone-and-use) + `demo.ipynb` + `quickstart.md`.
- Mirrored locally at `react_toolbox/`. Toolbox code license: MIT.

## Modules (markerless, no-calib scoped)
| Module | Provides |
|---|---|
| `io.py` | thin re-export of ReactVideoDataset; `load_frames()`, `load_meta()` |
| `reference.py` | `get_reference(mode=p01/first/running_avg)`; `subtract_reference()` (Sparsh signed diff `(img-bg)/255+0.5`) |
| `contact.py` | `contact_mask()` (diff→threshold→largest connected component); `contact_metrics()` (intensity/area/mixed, matches dataset scalars) |
| `depth.py` | OPTIONAL pretrained-`nnmini` depth: clean-room RGB2NormNet (5→64→64→64→2 ReLU) + DCT Poisson; weights downloaded on demand (GPL note); graceful skip if unavailable |
| `viz.py` | diff heatmap, contact overlay, reference-vs-frame, depth colormap + point cloud |
| `calibration.py` | load per-task extrinsics; project gel center → cam pixel (reuse twm.viz logic) |
| `actions.py` | `next_state_action()`, `delta_pose_action()` |

## Non-goals
- Marker flow / shear (no markers).
- Metric-calibrated depth (no calibration data; pretrained nnmini is approximate).
- Device streaming (this is a dataset toolbox, not a capture SDK).

## Verification experiments (self-designed, to run + iterate)
1. **reference**: subtract_reference at the p01 frame ≈ 0; sanity of diff range.
2. **contact_mask**: mask area monotonically tracks stored `area` scalar across a contact event; empty on no-contact frames.
3. **contact_metrics**: recomputed intensity/area/mixed match dataset parquet columns within tolerance.
4. **depth**: on a contact frame, depth map is smooth, finite, peaks under contact; point cloud renders; no-contact frame ≈ flat.
5. **calibration/projection**: projected gel center lands on the sensor in the cam image (visual + within-frame bounds).
6. **actions**: delta-pose integration recovers absolute pose (round-trip).
7. **end-to-end**: download one HF sample → every toolbox fn runs without error → demo notebook executes top-to-bottom.

## Iteration loop
Build → run experiments 1–7 → fix failures → re-run until all pass → render
demo outputs → publish toolbox/ + demo to HF.
