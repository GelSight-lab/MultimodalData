"""LUT geometry and force-informed virtual-target utilities.

For production React force reconstruction use ``run_episode.process_side``
or ``batch_worker``. The v8 predictor consumes ``react_calib.force_stages``
(calibration-free features), NOT this module's ``reconstruct`` LUT output.
``force_from_depth`` is retained for historical evaluation models only.

``reconstruct`` returns nominal millimetre geometry; its depth calibration
does not validate an absolute-newton scale on React. ``virtual_target`` uses
an assumed stiffness, not measured indentation. See README.md for the v8
model contract and RUNBOOK.md for batch processing and export restrictions.
"""
from __future__ import annotations

import numpy as np

# The reconstruction core. It lives in debug_gallery for historical reasons
# (that module was written first, as a diagnostic); 8 modules import it from
# there, so it is re-exported rather than moved.
from .debug_gallery import stages as reconstruct          # noqa: F401
from .lut_calibration import MM_PER_PIXEL, crop           # noqa: F401
from .marker_removal import stages_depth as reconstruct_marker_gel  # noqa: F401

# ---------------------------------------------------------------------------
# Stiffness for the force -> position-target conversion.
#
# DERIVED, not declared: `dexforce.STIFFNESS_N_PER_M` is the single source of
# truth. It is an ASSUMPTION about the environment, not a measured property of
# it, which is why it is exported as a named constant and written into the
# dataset sidecar rather than left at a call site — anyone reading a target
# pose must be able to see which stiffness produced it.
# ---------------------------------------------------------------------------
from .dexforce import STIFFNESS_N_PER_M

STIFFNESS_N_PER_MM = STIFFNESS_N_PER_M / 1000.0

FEATURES = ("vol", "vol2", "maxd", "area", "h1")


def feature_vector(st: dict) -> np.ndarray:
    """The five depth features the force models are fitted on."""
    f = st["feats"]
    return np.array([f[k] for k in FEATURES], dtype=float)


def force_from_depth(st: dict, model) -> float:
    """Historical evaluation helper, not the deployed React v8 predictor.

    `model` is a fitted (weights, isotonic) pair from one of the evaluation
    modules, calibrated per dataset and indenter group. For current force
    inference use `react_calib.fit` with `react_calib.force_stages` instead.
    """
    w, iso = model
    if st["feats"]["area"] < 1.0:            # no measurable contact
        return 0.0
    return float(max(0.0, iso.predict([float(feature_vector(st) @ w)])[0]))


def penetration_mm(force_n, k_n_per_mm: float = STIFFNESS_N_PER_MM):
    """How far past the surface a stiffness-k environment would be pushed.

    Zero force gives exactly zero penetration (not NaN), so a no-contact frame
    yields a target pose identical to the observed one.

    This is a virtual F/k displacement, not measured gel indentation. The
    current shared assumption is 2 N/mm: 15 N gives 7.5 mm. The export gate
    rejects displacement above 4.25 mm, so full-range v8 action export needs
    an explicit policy decision. This function does not clip displacement.
    """
    return np.asarray(force_n, dtype=float) / float(k_n_per_mm)


def virtual_target(pose_mm, force_n, normal_hat,
                   k_n_per_mm: float = STIFFNESS_N_PER_MM):
    """DexForce-style action: observed pose pushed along the contact normal.

        target = observed + (F / k) * n_hat

    Force reproduction depends on the controller and effective contact
    stiffness; the virtual target alone does not guarantee it.
    Free space is untouched: F = 0 -> target == observed, exactly.
    """
    pose = np.asarray(pose_mm, dtype=float)
    n = np.asarray(normal_hat, dtype=float)
    d = penetration_mm(force_n, k_n_per_mm)
    return pose + d[..., None] * n if pose.ndim > 1 else pose + d * n


__all__ = ["reconstruct", "reconstruct_marker_gel", "feature_vector",
           "force_from_depth", "penetration_mm", "virtual_target",
           "STIFFNESS_N_PER_MM", "FEATURES", "MM_PER_PIXEL", "crop"]
