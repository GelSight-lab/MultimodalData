# React Tactile Toolbox — Quickstart

Zero-install utilities for the [React dataset](https://huggingface.co/datasets/yxma/React)
(GelSight Mini, markerless, 640×480). MIT licensed.

```bash
# get the toolbox (it lives in the dataset repo)
huggingface-cli download yxma/React toolbox/ --repo-type dataset --local-dir react
cd react
pip install numpy pyarrow av opencv-python   # core
pip install torch scipy                       # optional: depth
```

```python
import react_toolbox as T
from huggingface_hub import hf_hub_download

# load tactile frames + per-frame metadata
vid  = hf_hub_download("yxma/React", "data/motherboard/videos/2026-05-10/episode_000/tactile_left.mp4", repo_type="dataset")
meta = T.load_meta(hf_hub_download("yxma/React", "data/motherboard/meta/2026-05-10/episode_000.parquet", repo_type="dataset"))
frames = T.load_video(vid, range(200))                       # (200, 480, 640, 3) RGB

# 1) reference (no-contact) frame + difference image
ref  = T.get_reference(frames, mode="p01", intensity=meta["tactile_left_intensity"][:200])
diff = T.difference(frames[100], ref)                        # Sparsh-style signed diff

# 2) contact detection (calibration-free)
mask    = T.contact_mask(frames[100], ref)                   # (480,640) bool
metrics = T.contact_metrics(frames[100], ref)               # {intensity, area, mixed} == dataset scalars

# 3) approximate depth / height map (pretrained nnmini, no calibration)
from react_toolbox import depth
h = depth.height_map(frames[100], ref)                       # (480,640) relative height

# 4) visualization (all return RGB uint8)
hm  = T.diff_heatmap(frames[100], ref)
ov  = T.contact_overlay(frames[100], ref)

# 5) camera projection (per-task extrinsics)
cal = T.load_calibration("data/motherboard")                # after snapshot_download of calibration/
uv  = T.project_gel_to_pixel(meta["sensor_left_pose"][100], cal["gel_left"], cal["cams"]["middle"])

# 6) derive actions from handheld poses
act   = T.next_state_action(meta["sensor_left_pose"])       # next-frame absolute state
delta = T.delta_pose_action(meta["sensor_left_pose"])       # frame-to-frame delta
```

Run the full demo (saves a montage):
```bash
python -m react_toolbox.demo --with_depth
```

## Functions

| Module | Function | Notes |
|---|---|---|
| `io` | `load_video`, `load_meta`, `episode_paths` | PyAV / OpenCV decode → RGB |
| `reference` | `get_reference`, `difference`, `l2_diff` | p01 / first / running-avg reference; Sparsh signed diff |
| `contact` | `contact_mask`, `contact_metrics`, `contact_centroid` | diff→threshold→largest component; reproduces dataset scalars |
| `depth` | `height_map`, `normals`, `poisson_integrate` | **approximate, uncalibrated** — pretrained markerless-Mini net (`nnmini.pt`, fetched on demand) + DCT Poisson |
| `viz` | `diff_heatmap`, `contact_overlay`, `reference_compare`, `depth_view` (grayscale by default = standard GelSight height map; cmap= for colormap), `contact_overlay`, `reference_compare`, `height_to_pointcloud` — all return RGB uint8 |
| `calibration` | `load_calibration`, `project_gel_to_pixel` | per-task extrinsics (motherboard=May-12, pushT=June-26) |
| `actions` | `next_state_action`, `delta_pose_action`, `integrate_delta` | handheld pose → IL/world-model targets |

## Notes & limits
- **No markers** on this sensor → no marker-flow / shear-field utilities (would need a markered gel).
- **Depth is approximate / relative**, not metric: it uses a network pretrained on a reference GelSight Mini (no per-unit calibration). Verified to track contact deformation locally (height elevation under contact correlates 0.74 with contact intensity), but the global height includes the gel's baseline curvature. For metric depth, collect a ball-indenter calibration.
- `nnmini.pt` weights are © GelSight Inc (GPL-3.0); only the weight file is fetched on demand — no GPL code is bundled. The toolbox code itself is MIT.
- All decoded frames are **RGB**. Contact scalars use L2 threshold `tau=8.0` (dataset default).
