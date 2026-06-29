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
]
__version__ = "0.1.0"
