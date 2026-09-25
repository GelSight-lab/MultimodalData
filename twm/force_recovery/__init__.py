"""Recovering force-related actions for the React dataset.

The dataset records sensor pose (OptiTrack) and GelSight images, but no
applied force — demonstrated pose equals achieved pose, so the usual
"position error x stiffness" force channel of teleoperated data does not
exist here. Two methods are implemented:

1. ``feats_infer`` / ``evaluate_feats`` — pseudo force labels: FEATS
   (arXiv:2411.03315) maps GelSight Mini images to normal + shear force
   distributions; we run it offline over the recordings.
2. ``dexforce`` — force-informed position targets (arXiv:2501.10356): the
   estimated normal force becomes a virtual target displaced along the sensor
   normal, so the action stays a pose and composes with the existing 30 Hz
   pose actions.
"""

__version__ = "0.1.0"
