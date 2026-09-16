import numpy as np
import pytest
from sklearn.neighbors import KNeighborsRegressor


def fitted_model():
    from twm.force_recovery.bounded_force import GuardedForceModel

    x = np.linspace(0, 1, 80)[:, None]
    return GuardedForceModel(KNeighborsRegressor(3, weights="distance")).fit(
        x[::2], 15 * x[::2, 0], x[1::2], 15 * x[1::2, 0], domain="calibration")


def test_unregistered_domain_never_silently_returns_zero():
    result = fitted_model().predict_result(np.array([[0.5]]), domain="pushT_left")
    assert not result["supported"][0]
    assert np.isnan(result["force_n"][0])
    assert result["reason"][0] == "unregistered_domain"


def test_far_and_nonfinite_inputs_are_rejected():
    result = fitted_model().predict_result(np.array([[0.5], [1000.0], [np.nan]]), domain="calibration")
    assert result["supported"].tolist() == [True, False, False]
    assert np.isnan(result["force_n"][1:]).all()
    assert result["reason"][1:].tolist() == ["outside_support", "nonfinite_input"]
    assert 0 <= result["force_n"][0] <= 15
    assert 0 <= result["lower_n"][0] <= result["upper_n"][0] <= 15


def test_force_labels_outside_declared_range_are_rejected():
    from twm.force_recovery.bounded_force import GuardedForceModel

    with pytest.raises(ValueError, match="0-15"):
        GuardedForceModel(KNeighborsRegressor(1)).fit(
            np.ones((10, 2)), np.full(10, 16.0), np.ones((3, 2)), np.ones(3), domain="test")


def test_extreme_candidates_are_bounded_but_not_accepted():
    from sklearn.linear_model import LinearRegression
    from twm.force_recovery.bounded_force import GuardedForceModel

    x = np.linspace(0, 1, 80)[:, None]
    model = GuardedForceModel(LinearRegression()).fit(
        x[::2], 15 * x[::2, 0], x[1::2], 15 * x[1::2, 0], domain="test")
    assert model.predict_candidates(np.array([[-1000], [1000]])).tolist() == [0, 15]
    result = model.predict_result(np.array([[-1000], [1000]]), domain="test")
    assert not result["supported"].any()
    assert np.isnan(result["force_n"]).all()


def test_finite_input_that_overflows_scaling_is_rejected():
    with np.errstate(over="ignore", invalid="ignore"):
        result = fitted_model().predict_result(np.array([[1e308]]), domain="calibration")
    assert not result["supported"].any()
    assert np.isnan(result["force_n"]).all()


def test_frame_inference_rejects_a_different_contact_threshold():
    from twm.force_recovery.bounded_force import predict_frame
    from twm.force_recovery.model_search import CF, FEATURE_VERSION

    frame = np.zeros((240, 320, 3), np.float32)
    with pytest.raises(ValueError, match="threshold"):
        predict_frame({"valid_di": CF.VALID_DI + 1, "feature_version": FEATURE_VERSION}, frame, frame)


def test_monotone_regressor_bounds_predictions_and_flags_score_extrapolation():
    from twm.force_recovery.bounded_force import RidgeIsotonicRegressor

    x = np.linspace(0, 1, 40)[:, None]
    model = RidgeIsotonicRegressor(alpha=0.1).fit(x, 15 * x[:, 0])
    queries = np.array([[-100], [0.5], [100]])
    assert model.in_score_support(queries).tolist() == [False, True, False]
    pred = model.predict(queries)
    assert ((pred >= 0) & (pred <= 15)).all()


def test_guard_rejects_score_extrapolation_even_inside_distance_threshold():
    from twm.force_recovery.bounded_force import GuardedForceModel, RidgeIsotonicRegressor

    x = np.linspace(0, 1, 80)[:, None]
    model = GuardedForceModel(RidgeIsotonicRegressor(0.1)).fit(
        x[::2], 15 * x[::2, 0], x[1::2], 15 * x[1::2, 0], domain="test")
    model.threshold_ = np.inf
    result = model.predict_result(np.array([[10.0]]), domain="test")
    assert not result["supported"].any()
    assert result["reason"][0] == "outside_score_support"
