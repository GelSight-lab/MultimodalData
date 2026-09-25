import numpy as np


def test_position_split_keeps_repeated_presses_together():
    from twm.force_recovery.model_search import position_groups, split_positions

    rows = [{"x": float(i), "y": float(i % 3)} for i in range(12) for _ in range(3)]
    groups = position_groups(rows)
    train, test = split_positions(groups)
    assert not set(groups[train]) & set(groups[test])
    assert train.sum() + test.sum() == len(rows)
    assert test.any() and train.any()


def test_features_depend_only_on_observations_and_are_finite_for_blank():
    from twm.force_recovery.model_search import observation_features

    ref = np.full((240, 320, 3), 100, dtype=np.float32)
    stage = {"depth": np.zeros((240, 320)), "contact": np.zeros((240, 320), bool),
             "feats": {k: 0.0 for k in ("vol", "vol2", "maxd", "area", "h1")}}
    first = observation_features(ref, ref, stage)
    stage.update(f=7.0, x=10.0, y=20.0, z=3.0)
    second = observation_features(ref, ref, stage)
    assert set(first) == {"basic", "geometry", "image", "combined"}
    for key in first:
        assert np.isfinite(first[key]).all()
        np.testing.assert_array_equal(first[key], second[key])


def test_selection_does_not_receive_outer_holdout():
    from sklearn.dummy import DummyRegressor
    from twm.force_recovery.model_search import select_model

    x = np.zeros((24, 2))
    y = np.repeat([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 4)
    groups = np.repeat(np.arange(6), 4)
    configs = [("mean", "basic", DummyRegressor(strategy="mean"))]
    chosen, scores = select_model({"basic": x}, y, groups, configs, folds=3)
    assert chosen[0] == "mean"
    assert len(scores) == 1 and scores[0]["cv_mae_n"] > 0


def test_inference_rejects_a_different_reconstruction_threshold():
    import pytest
    from twm.force_recovery.model_search import predict_images, FEATURE_VERSION, CF

    frame = np.full((240, 320, 3), 100, dtype=np.float32)
    with pytest.raises(ValueError, match="threshold"):
        predict_images({"feature_version": FEATURE_VERSION, "valid_di": CF.VALID_DI + 4}, frame, frame)
