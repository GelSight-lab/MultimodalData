# Shared TWM visualization

Run from the repository checkout or an installed package (see
[installation](../README.md#development-and-installation)):

```bash
python -m twm.visualization --help
python -m twm.visualization /path/to/episode.h5
python -m twm.visualization /path/to/episode.parquet
python -m twm.visualization /path/to/episode.h5 --save_video /path/to/preview.mp4
```

The CLI delegates to the existing interactive viewer, including its playback,
calibration, force overlay and export options. `python -m twm.visualize` remains
supported. Export through that viewer still uses its existing writer; verified
streaming batch export is available through `write_video` below. The CLI retains
the canonical camera/tactile/wrist layout; arbitrary layouts use the Python API.

## Custom modalities and overlays

The renderer imports no hardware drivers, ROS, storage readers or inference
models. Feed it already-aligned arrays from a live snapshot, recording, or release.

```python
import cv2
from twm.visualization import Tile, Renderer

def draw_contact(bgr):
    # Pixel coordinates are in this source's native resolution, before resize.
    cv2.circle(bgr, (100, 80), 12, (0, 255, 0), 2)

result = Renderer(columns=3, tile_width=320, tile_height=240).render([
    Tile("middle", camera_rgb, color_format="RGB", overlays=(draw_contact,)),
    Tile("depth", depth_mm, modality="depth", color_format="depth",
         depth_range=(200, 1500)),
    Tile("wrist left", wrist_bgr, modality="wrist"),
    Tile("tactile left", tactile_rgb, modality="tactile", color_format="RGB"),
    Tile("height", height_mm, modality="tactile_height", color_format="depth",
         depth_range=(-1, 4)),
    Tile("wrist right", None, modality="wrist"),
])
cv2.imshow("inspection", result.image)  # always uint8 BGR
```

- Color inputs are H×W×3 uint8 with explicit RGB/BGR order. `modality` is a
  descriptive label, not a decoder; RGB tactile images still need `RGB`.
- Depth/height inputs are real H×W arrays. Depth ignores nonpositive/nonfinite
  samples; tactile height permits signed finite samples. Invalid pixels are black.
  Supply a fixed range for comparable brightness over time; automatic min/max is
  per image. Constant auto-scaled images are black.
- Overlays run in tuple order on a contiguous native-resolution BGR **copy**.
  Draw poses, masks, contacts, force arrows, or annotations in callbacks; capture
  the frame's aligned context in closures. Inputs are never modified. Callbacks
  are not called for missing tiles, which have a visible missing-data label.
- `result.layouts` gives each tile's name, modality, rectangle and missing flag.
  Tiles are stretched to their configured size, with nearest-neighbor resizing.
- The compositor does not synchronize sensors, infer force, reconstruct tactile
  surfaces, or repair poses. Those remain source/model responsibilities.

## Canonical preview and mocap projection

`render_preview` accepts the same arguments as `twm.viz.build_preview_panel`.
It preserves the old layout, colors, status strip and projection mathematics.
The live recorder, viewer, and batch episode preview script use this adapter.
Legacy functions remain available for downstream scripts and compatibility.

```python
from twm.visualization import Projection, render_preview, draw_preview_overlay

projection = Projection(
    project_cams, gel_center_left, gel_center_right,
    pose_world_frame="z", calibration_world_frame="z",
    forces_n={"left": 2.0}, targets_7=targets, press_axis=press_axes,
)
panel = render_preview(colors, gels, references, poses, False, row, elapsed,
                       projection=projection)
```

The repository spells world up axes `"y"` and `"z"`. When declared, pose and
calibration frames must both be supplied and match. This validation does **not**
inspect calibration files or convert coordinates. Do not relabel Y-up calibration
as Z-up. Use the existing calibration loader and conversion utilities; published
Z-up poses must use the corresponding shipped Z-up calibration. Legacy callers
may omit both declarations, retaining their existing loader policy.

Pose translations are metres, quaternions are xyzw; calibration translations and
gel-center offsets follow the existing millimetre convention. Projection geometry
remains solely in `twm.viz`; no second projection implementation was introduced.

For live use, build/cache a plain preview at the camera preview rate, copy it,
then call `draw_preview_overlay(panel, latest_poses, projection)` at the display
rate. This preserves the recorder's independent latest-pose cadence.

For playback, retain the source adapter's timestamp/row alignment and tactile
latency. Release rows, raw HDF5 indices and capture timestamps are not interchangeable.
Pass missing/invalid modalities as missing, not fabricated zero measurements.
Keep action/force validity masks and repair provenance in the source context:
drawing a repaired trace does not approve it for training. This refactor changes
no dataset values, masks, time bases, repair policy or calibration files.

## Streaming batch export

```python
from twm.visualization.export import write_video

def frames():
    # Open and close source readers here. Recreate this exact sequence on retry.
    for snapshot in aligned_snapshots():
        yield renderer.render(tiles_for(snapshot)).image

count = write_video("preview.mp4", frames, fps=30)
```

FFmpeg must be installed. The encoder retains only current frames, checks fixed
uint8 BGR shape, then decodes the temporary output to verify its frame count.
Decode-count failures retry using the replayable factory (three attempts by
default). Producer and encoder failures propagate, release resources, and leave
existing output untouched. Only verified output atomically replaces the target.
Choose an `.mp4` destination. Verification checks decodability/count, not semantic
alignment; that is the source adapter's responsibility.

`twm.scripts.build_episode_previews.iter_preview_panels` exposes the existing
aligned HDF5 preview generator. Its `build_one_preview` wrapper uses this writer.
Frame transforms must be deterministic because a verification retry re-renders.

## Force-result inspection

`twm.force_recovery.visualize.overlay_clip` uses the same compositor and verified
writer. Its HDF5 readers reopen on retry and close on early encoder failure.
Tactile frames and depth-panel reference images use
`open_episode(...).align[side].index_map`, matching the estimator's source adapter
for both timestamped and legacy recordings. A saved NPZ `source_frame` records
held force estimates and is not substituted for the display's capture map.

Pure `diff_rgb`, `diff_caption` and `ForceOverlay` live in
`twm.visualization.force`. The first two remain importable from
`twm.force_recovery.visualize` for compatibility, without loading Arrow,
Matplotlib or the estimator. `ForceOverlay` pre-renders its timeline once, then
copies the background and draws the current value/cursor per frame. Force values
are supplied by the caller; rendering does not regenerate or smooth targets.

Existing force-result figure/clip APIs keep their signatures. `overlay_clip`
still treats `out_fps` as export playback rate, not a request to resample capture
timestamps. Its default force trace retains the existing median-on-fresh-frames
display policy; pass `force=` to display a supplied estimate unchanged.

## Checks and benchmark

```bash
PYTHONPATH=. python -m pytest tests/visualization tests/test_visualize.py -q
python -m twm.scripts.benchmark_visualization --iterations 100
python -m twm.pipeline_guard
```

The benchmark asserts old/new preview pixel equality and reports median/p95
render times; it intentionally has no machine-dependent speed assertion.
