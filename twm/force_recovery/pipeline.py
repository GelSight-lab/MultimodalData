"""The stable public API: GelSight frame -> depth -> normal force -> action.

Everything else in this package is either a study that produced a published
number, a site builder, or a dataset adapter. This module is the one place a
caller should need to import, and it is deliberately thin — it re-exports the
functions that are actually load-bearing and documents where they live, so
nobody has to rediscover that the reconstruction core sits in a module called
``debug_gallery``.

    from force_recovery.pipeline import reconstruct, force_from_depth, \
        virtual_target, STIFFNESS_N_PER_MM

    st = reconstruct(img, ref)          # dI -> LUT -> valid mask -> Poisson
    f  = force_from_depth(st, model)    # calibrated newtons
    p  = virtual_target(pose, f, n_hat) # DexForce-style position target

WHAT THE NUMBERS MEAN (measured, see task_plan.md for the evidence):

* The reconstruction is accurate for SHALLOW contact and degrades outside the
  calibrated slope range: against exact per-pixel ground truth, MAE is 11 um
  at a 0.3 mm press and 281 um at 2.25 mm, with the peak-depth ratio falling
  1.00 -> 0.55. Quote no accuracy figure without its press depth.
* Force interpolates well and extrapolates badly for the same reason: fitting
  on the low half of a force range and predicting the high half drops rho
  0.968 -> 0.552, because the isotonic stage clips outside its training range
  (the predicted range collapses to 5% of the true one).
* Force calibration is per-group. Stiffness/gel differences are absorbed into
  fitted weights, so a rho is a within-group rank correlation, NOT evidence of
  a transferable absolute-newton scale across sensors.

LAYOUT OF THE REST OF THE PACKAGE — see ARCHITECTURE.md for the full map.
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
# 1.0 N/mm is the project's declared starting point. It is an ASSUMPTION about
# the environment, not a measured property of it, which is why it is a named
# constant exported here and written into the dataset sidecar rather than
# buried at a call site: anyone reading a target pose must be able to see
# which stiffness produced it.
# ---------------------------------------------------------------------------
STIFFNESS_N_PER_MM = 1.0

FEATURES = ("vol", "vol2", "maxd", "area", "h1")


def feature_vector(st: dict) -> np.ndarray:
    """The five depth features the force models are fitted on."""
    f = st["feats"]
    return np.array([f[k] for k in FEATURES], dtype=float)


def force_from_depth(st: dict, model) -> float:
    """Calibrated normal force [N] for one reconstructed frame.

    `model` is a fitted (weights, isotonic) pair from one of the evaluation
    modules. Kept as an argument rather than a global because the calibration
    is per-dataset and per-indenter group — see the module docstring.
    """
    w, iso = model
    if st["feats"]["area"] < 1.0:            # no measurable contact
        return 0.0
    return float(max(0.0, iso.predict([float(feature_vector(st) @ w)])[0]))


def penetration_mm(force_n, k_n_per_mm: float = STIFFNESS_N_PER_MM):
    """How far past the surface a stiffness-k environment would be pushed.

    Zero force gives exactly zero penetration (not NaN), so a no-contact frame
    yields a target pose identical to the observed one.
    """
    return np.asarray(force_n, dtype=float) / float(k_n_per_mm)


def virtual_target(pose_mm, force_n, normal_hat,
                   k_n_per_mm: float = STIFFNESS_N_PER_MM):
    """DexForce-style action: observed pose pushed along the contact normal.

        target = observed + (F / k) * n_hat

    The action stays a pose, so it composes with the existing pose actions and
    an impedance controller reproduces the demonstrated force at deployment.
    Free space is untouched: F = 0 -> target == observed, exactly.
    """
    pose = np.asarray(pose_mm, dtype=float)
    n = np.asarray(normal_hat, dtype=float)
    d = penetration_mm(force_n, k_n_per_mm)
    return pose + d[..., None] * n if pose.ndim > 1 else pose + d * n


__all__ = ["reconstruct", "reconstruct_marker_gel", "feature_vector",
           "force_from_depth", "penetration_mm", "virtual_target",
           "STIFFNESS_N_PER_MM", "FEATURES", "MM_PER_PIXEL", "crop"]
