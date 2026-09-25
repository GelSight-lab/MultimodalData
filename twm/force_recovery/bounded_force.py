"""Experimental 0-15 N inference with explicit support and domain restrictions.

Bounding the output and rejecting distant inputs do not make a model OOD-free.
The residual band is empirical calibration information, not an OOD guarantee.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import nnls
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

FORCE_LIMIT_N = 15.0


class RidgeIsotonicRegressor(BaseEstimator, RegressorMixin):
    """Rich image score with a bounded monotone calibration table."""

    def __init__(self, alpha=10):
        self.alpha = alpha

    def fit(self, x, y):
        y = np.asarray(y, float)
        if not np.isfinite(y).all() or np.any((y < 0) | (y > FORCE_LIMIT_N)):
            raise ValueError("Force labels must be within 0-15 N")
        self.regressor_ = make_pipeline(StandardScaler(), Ridge(alpha=self.alpha, solver="lsqr")).fit(x, y)
        score = self.regressor_.predict(x)
        self.score_range_ = (float(score.min()), float(score.max()))
        self.calibrator_ = IsotonicRegression(y_min=0, y_max=FORCE_LIMIT_N, out_of_bounds="clip").fit(score, y)
        return self

    def predict(self, x):
        return self.calibrator_.predict(self.regressor_.predict(x))

    def in_score_support(self, x):
        score = self.regressor_.predict(x)
        return (score >= self.score_range_[0]) & (score <= self.score_range_[1])


class PositiveGeometryRegressor(BaseEstimator, RegressorMixin):
    """Nonnegative deformation-feature baseline, with no fitted intercept."""

    def fit(self, x, y):
        x = np.asarray(x, float)
        self.scale_ = np.maximum(np.std(x, axis=0), 1e-6)
        self.coef_ = nnls(x / self.scale_, np.asarray(y), maxiter=1000)[0]
        return self

    def predict(self, x):
        return (np.asarray(x) / self.scale_) @ self.coef_


class GuardedForceModel:
    def __init__(self, estimator, support_quantile=0.95, error_quantile=0.90):
        self.estimator = estimator
        self.support_quantile = support_quantile
        self.error_quantile = error_quantile

    def fit(self, x, y, calibration_x, calibration_y, *, domain):
        x, calibration_x = np.asarray(x, float), np.asarray(calibration_x, float)
        y, calibration_y = np.asarray(y, float), np.asarray(calibration_y, float)
        if (x.ndim != 2 or calibration_x.ndim != 2 or x.shape[1] != calibration_x.shape[1]
                or len(x) < 2 or len(calibration_x) < 2
                or y.shape != (len(x),) or calibration_y.shape != (len(calibration_x),)):
            raise ValueError("Incompatible fitting/calibration arrays")
        if not all(np.isfinite(a).all() for a in (x, y, calibration_x, calibration_y)):
            raise ValueError("Fitting/calibration data must be finite")
        if any(np.any((a < 0) | (a > FORCE_LIMIT_N)) for a in (y, calibration_y)):
            raise ValueError("Force labels must be within 0-15 N")
        if not domain or not 0 < self.support_quantile <= 1 or not 0 < self.error_quantile <= 1:
            raise ValueError("A calibration domain and valid quantiles are required")
        self.domain_ = domain
        self.model_ = clone(self.estimator).fit(x, y)
        self.scaler_ = StandardScaler().fit(x)
        self.neighbors_ = NearestNeighbors(n_neighbors=min(5, len(x))).fit(self.scaler_.transform(x))
        distances = self._distances(calibration_x)
        self.threshold_ = float(np.quantile(distances, self.support_quantile, method="higher"))
        errors = np.abs(self.predict_candidates(calibration_x) - calibration_y)
        self.error_radius_n_ = float(np.quantile(errors, self.error_quantile, method="higher"))
        return self

    def _distances(self, x):
        z = self.scaler_.transform(x)
        return self.neighbors_.kneighbors(z, return_distance=True)[0].mean(axis=1) / np.sqrt(z.shape[1])

    def predict_candidates(self, x):
        """Bounded diagnostic values; callers must not assume they are supported."""
        return np.clip(self.model_.predict(x), 0, FORCE_LIMIT_N)

    def predict_result(self, x, *, domain=None):
        x = np.asarray(x, float)
        if x.ndim != 2 or x.shape[1] != self.scaler_.n_features_in_:
            raise ValueError("Unexpected inference feature dimensions")
        n = len(x)
        result = {"force_n": np.full(n, np.nan), "lower_n": np.full(n, np.nan),
                  "upper_n": np.full(n, np.nan), "supported": np.zeros(n, bool),
                  "support_distance": np.full(n, np.nan),
                  "reason": np.full(n, "unregistered_domain", dtype="U32")}
        if domain != self.domain_:
            return result
        finite = np.isfinite(x).all(axis=1)
        result["reason"][:] = "nonfinite_input"
        if not finite.any():
            return result
        result["reason"][finite] = "outside_support"
        indices = np.flatnonzero(finite)
        with np.errstate(over="ignore", invalid="ignore"):
            normalized = self.scaler_.transform(x[finite])
        # Reject values whose squared Euclidean distance could overflow.
        safe_limit = np.sqrt(np.finfo(float).max / x.shape[1]) / 4
        computable = np.isfinite(normalized).all(axis=1) & (np.abs(normalized) < safe_limit).all(axis=1)
        indices = indices[computable]
        if not len(indices):
            return result
        distance = self.neighbors_.kneighbors(normalized[computable], return_distance=True)[0].mean(axis=1) / np.sqrt(x.shape[1])
        result["support_distance"][indices] = distance
        nearby = distance <= self.threshold_ + 1e-12
        indices = indices[nearby]
        if not len(indices):
            return result
        if hasattr(self.model_, "in_score_support"):
            result["reason"][indices] = "outside_score_support"
            indices = indices[self.model_.in_score_support(x[indices])]
            if not len(indices):
                return result
        raw = self.model_.predict(x[indices])
        result["reason"][indices] = "force_outside_calibration"
        keep = np.isfinite(raw) & (raw >= 0) & (raw <= FORCE_LIMIT_N)
        accepted = indices[keep]
        pred = raw[keep]
        result["supported"][accepted] = True
        result["reason"][accepted] = "within_calibrated_support"
        result["force_n"][accepted] = pred
        result["lower_n"][accepted] = np.maximum(0, pred - self.error_radius_n_)
        result["upper_n"][accepted] = np.minimum(FORCE_LIMIT_N, pred + self.error_radius_n_)
        return result


def predict_frame(bundle, img, reference, *, domain=None):
    """Return one experimental estimate; unknown domains remain explicit NaNs."""
    from .model_search import CF, FEATURE_VERSION, observation_features
    from . import react_calib as RC

    if bundle.get("valid_di") != CF.VALID_DI:
        raise ValueError("Model contact threshold differs from the active reconstruction")
    if bundle.get("feature_version") != FEATURE_VERSION:
        raise ValueError("Model feature version differs from the active extractor")
    model = bundle["model"]
    if domain != model.domain_:
        values = np.zeros((1, model.scaler_.n_features_in_))
    else:
        stage = RC.force_stages(img, reference)
        features = observation_features(img, reference, stage)
        features["physical"] = features["basic"][:5]
        values = features[bundle["feature"]][None]
    return {k: v[0].item() for k, v in model.predict_result(values, domain=domain).items()}
