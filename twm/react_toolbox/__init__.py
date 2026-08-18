"""React tactile toolbox — load + derive standard VBTS signals from the
React dataset (GelSight Mini, markerless, 640×480). MIT licensed.

    from react_toolbox import (
        load_video, load_meta, episode_paths,
        get_reference, difference, l2_diff,
        contact_mask, contact_metrics, contact_centroid,
        next_state_action, delta_pose_action,
    )
"""
from .io import load_video, load_meta, episode_paths
from .reference import get_reference, difference, l2_diff
from .contact import contact_mask, contact_metrics, contact_centroid, TAU
from .actions import next_state_action, delta_pose_action, integrate_delta
from .viz import diff_heatmap, contact_overlay, reference_compare, depth_view, height_to_pointcloud
from .calibration import load_calibration, project_gel_to_pixel

__all__ = [
    "load_video", "load_meta", "episode_paths",
    "get_reference", "difference", "l2_diff",
    "contact_mask", "contact_metrics", "contact_centroid", "TAU",
    "next_state_action", "delta_pose_action", "integrate_delta",
    "diff_heatmap", "contact_overlay", "reference_compare", "depth_view", "height_to_pointcloud",
    "load_calibration", "project_gel_to_pixel",
    "draw_projection", "force_radius_px",
    "read_world_frame", "verify_world_frame", "projection_fingerprint",
]
__version__ = "0.1.0"

# Projection debugging. `draw_projection` renders what `project_gel_to_pixel`
# computed, so a user can see whether it lands on the sensor; the world-frame
# helpers let them check the same thing without looking.
from .viz import draw_projection, force_radius_px          # noqa: F401
from .world_frame import (projection_fingerprint,           # noqa: F401
                          read_world_frame, verify_world_frame)
