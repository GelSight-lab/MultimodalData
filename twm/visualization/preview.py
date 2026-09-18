"""Canonical preview composition; all projection math remains in twm.viz."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from twm.viz import build_preview_panel, draw_projection_overlay


@dataclass(frozen=True)
class Projection:
    """Arguments for the existing preview projection, without pose conversion.

    Optional world-frame declarations ('y' or 'z', the repository's up-axis
    convention) must be supplied together and agree.
    Omit both for legacy callers whose frame convention is established by
    their calibration loader. Poses remain metres + xyzw quaternion and
    calibration transforms / gel centers remain millimetres, as in twm.viz.
    """

    project_cams: list[dict]
    gel_center_left: np.ndarray
    gel_center_right: np.ndarray
    frozen_side: Optional[str] = None
    forces_n: Optional[dict] = None
    targets_7: Optional[dict] = None
    press_axis: Optional[dict] = None
    axis_len_mm: float = 120.0
    pose_world_frame: Optional[str] = None
    calibration_world_frame: Optional[str] = None

    def __post_init__(self) -> None:
        frames = self.pose_world_frame, self.calibration_world_frame
        if frames == (None, None):
            return
        if any(frame not in ("y", "z") for frame in frames):
            raise ValueError("Both world frames must be declared as 'y' or 'z'")
        if frames[0] != frames[1]:
            raise ValueError("Pose and calibration world frames must match")


def draw_preview_overlay(panel: np.ndarray, optitrack_poses: dict,
                         projection: Projection) -> None:
    """Draw in place using the latest poses, independently of panel cadence."""
    draw_projection_overlay(
        panel, optitrack_poses, projection.project_cams,
        projection.gel_center_left, projection.gel_center_right,
        frozen_side=projection.frozen_side, forces_n=projection.forces_n,
        targets_7=projection.targets_7, press_axis=projection.press_axis,
        axis_len_mm=projection.axis_len_mm,
    )


def render_preview(*args, projection: Optional[Projection] = None, **kwargs) -> np.ndarray:
    """Build the legacy BGR panel, optionally drawing its projection overlay.

    All positional and keyword arguments retain ``build_preview_panel``'s
    meaning. The projection uses that same call's OptiTrack poses. For live
    callers with independent image / pose cadence, cache the plain result
    and call ``draw_preview_overlay`` on a copy with newer poses.
    """
    panel = build_preview_panel(*args, **kwargs)
    if projection is not None:
        poses = args[3] if len(args) > 3 else kwargs["optitrack_poses"]
        draw_preview_overlay(panel, poses, projection)
    return panel
