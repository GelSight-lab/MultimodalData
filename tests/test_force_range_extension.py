import numpy as np
import pytest
from sklearn.isotonic import IsotonicRegression


def test_tail_preserves_base_knots_and_is_continuous_and_bounded():
    from twm.force_recovery.react_calib import extend_isotonic

    base = IsotonicRegression(out_of_bounds='clip').fit([0, 2, 5, 9], [.1, 1, 4, 7.8])
    extended = extend_isotonic(base, [10, 12, 14, 18], [9, 11, 13, 14.9])
    x = np.linspace(-1, 9, 500)
    np.testing.assert_allclose(extended.predict(x), base.predict(x), atol=1e-14)
    assert abs(extended.predict([9-1e-8])[0]-extended.predict([9+1e-8])[0]) < 1e-6
    y = extended.predict(np.linspace(-100, 100, 1000))
    assert (np.diff(y) >= -1e-12).all()
    assert y.max() == pytest.approx(14.9)
    assert extended.predict([12])[0] == pytest.approx(11)


def test_tail_requires_real_high_force_support_above_join():
    from twm.force_recovery.react_calib import extend_isotonic

    base = IsotonicRegression(out_of_bounds='clip').fit([0, 9], [0, 7.8])
    for scores, labels in [([2, 3], [12, 15]), ([10, 12], [9, 10]),
                           ([10, np.nan], [12, 15]), ([10, 12], [14, 16])]:
        with pytest.raises(ValueError):
            extend_isotonic(base, scores, labels)


def test_tail_split_never_trains_on_old_heldout_positions():
    from twm.force_recovery.react_calib import tail_holdout_mask

    base_keys = np.arange(9)
    tail_keys = np.repeat(np.arange(15), 3)
    mask = tail_holdout_mask(tail_keys, base_keys, {1, 5, 8})
    for key in np.unique(tail_keys):
        assert len(np.unique(mask[tail_keys == key])) == 1
    assert mask[np.isin(tail_keys, [1, 5, 8])].all()
    assert not mask[np.isin(tail_keys, [0, 2, 3, 4, 6, 7])].any()
    assert mask[tail_keys >= 9].sum() == 6


def test_force_overlay_does_not_saturate_at_eight_newtons():
    from twm.force_overlay import radius_px, R_MAX_PX

    assert radius_px(8) < radius_px(12) < radius_px(15)
    assert radius_px(15) == R_MAX_PX


def test_extended_fit_excludes_high_force_test_labels_and_preserves_low_fit(tmp_path, monkeypatch):
    import json
    from twm.force_recovery import react_calib as rc

    rng = np.random.default_rng(7)
    def row(i, force):
        return dict(vol=force, vol2=force**2, maxd=force/2, area=2+force,
                    h1=force*np.sqrt(2+force)/2, cx=rng.uniform(70, 230),
                    cy=rng.uniform(60, 180), x=i, y=0, z=force/2, f=force)
    base = [row(i, rng.uniform(1, 8)) for i in range(90)]
    high = [row(i, rng.uniform(9, 15)) for i in range(90)]
    monkeypatch.setattr(rc, '_load', lambda recon: (base, lambda k: np.array([r[k] for r in base])))
    cache = tmp_path / 'tail.json'
    cache.write_text(json.dumps(high))
    monkeypatch.setattr(rc, 'TAIL_CACHE', cache)
    _, low = rc.fit(report=False, holdout=True, extend_range=False)
    model, original = rc.fit(report=False, holdout=True)
    np.testing.assert_array_equal(original['pred'][:30], low['pred'])
    assert model.force_ceiling_n > 14
    held_positions = np.random.default_rng(0).permutation(90)[:30]
    for i in held_positions:
        high[i]['f'] = 8.01
        high[i]['z'] *= 100
    cache.write_text(json.dumps(high))
    _, changed = rc.fit(report=False, holdout=True)
    np.testing.assert_array_equal(changed['pred'], original['pred'])


def test_review_preservation_check_accounts_for_contact_weight():
    from twm.force_recovery import react_calib as rc
    from twm.force_recovery.task_review import extension_metrics

    areas = np.array([.2, 2, 2])
    weight = rc.contact_weight(areas[0])
    v7 = np.array([weight*7.79, 4, 7.79])
    v8 = np.array([weight*12, 4, 12])
    result = extension_metrics(v7, v8, areas, np.ones(3, bool), 0, 7.79)
    assert result['n_below_old_score_ceiling'] == 1
    assert result['max_change_below_old_score_ceiling_n'] == 0
    assert result['nonzero_decision_changes'] == 0


def test_historical_search_cache_keeps_its_eight_newton_range(tmp_path, monkeypatch):
    from PIL import Image
    from twm.force_recovery import model_search as search

    folder = tmp_path / 'raw' / 'round'
    folder.mkdir(parents=True)
    for name in ['initial.jpg', 'round|1pos|1.0, 1.0, -1.0 f|1.0.jpg',
                 'round|2pos|1.0, 1.0, -2.0 f|12.0.jpg']:
        Image.fromarray(np.full((64, 64, 3), 100, np.uint8)).save(folder/name)
    monkeypatch.setattr(search, 'ROOT', tmp_path/'cache')
    monkeypatch.setattr(search, 'CNC_MINI_26', tmp_path/'raw')
    monkeypatch.setattr(search.RC, 'force_stages', lambda *args: {
        'contact': np.ones((8, 8), bool), 'feats': {'area': 2}})
    monkeypatch.setattr(search, 'observation_features', lambda *args: {
        k: np.ones(2) for k in ('basic', 'geometry', 'image', 'combined')})
    rows, _ = search.load_features('round')
    assert [r['f'] for r in rows] == [1.0]
