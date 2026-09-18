# Unified TWM visualization and maintenance

Approved direction: incremental refactor, shared rendering with live/HDF5/release
adapters, compatible old entry points, explicit frame and timing conventions.

## Scope and sequence

1. Inventory TWM modules, scripts and tests; identify duplicated rendering,
   unsafe imports, resource ownership and expensive repeated operations.
2. Deliver a small `twm.visualization` library with named modality tiles,
   explicit RGB/BGR/depth conversion, composable tile overlays, layout metadata,
   and a legacy panel adapter. Reuse existing projection/calibration mathematics.
3. Migrate collection/replay/preview rendering to the shared API without changing
   native timestamp alignment, fixed legacy layout, or repair/force semantics.
4. Clean touched tests: scoped fixtures, behavioral assertions, parametrization;
   no deletion of existing safety checks. Keep historical CLI verifiers usable.
5. Add a single visualization CLI, documentation and reproducible CPU benchmarks.

## Contracts

- Rendering never mutates source arrays. Color conversion occurs explicitly at
  the input boundary. Camera depth and tactile height are distinct modalities.
- Tile overlays draw in native tile coordinates before resize; images are clipped
  to their own tile. Missing modalities are explicit, not fabricated observations.
- Existing `twm.viz` API and its pixel outputs remain compatible. Geometry remains
  authoritative there during migration; no second pose projection implementation.
- Calibration and pose frames must agree. Z-up data must not receive a second
  Y-up-to-Z-up transform. Source indices are not segment-local indices.
- Live preview retains independent base-image/pose update rates. File rendering
  uses recorded alignment, not the live latest-pose policy.
- No hardware capture, network uploads or dataset rewriting in this refactor.

## Validation

Characterize legacy output with synthetic RGB/tactile/wrist data. Check pixel
equality across adapter migration, input ownership, overlay clipping/order,
missing channels and finite-depth behavior. Run recorder, viewer, calibration,
action/force alignment regressions and pipeline guard. Benchmark identical
inputs after warmup; report measured results without universal speed claims.

## Boundaries

Force model research and historical one-off publishing scripts are audited, not
rewritten wholesale. New modality inference (depth/force networks), GUI redesign
and changes to physical calibration are outside this first implementation.
