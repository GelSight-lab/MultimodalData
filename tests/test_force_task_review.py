import numpy as np
import pytest


def test_contact_weight_keeps_small_contacts_and_is_continuous():
    from twm.force_recovery.react_calib import contact_weight

    assert contact_weight(0) == 0
    assert 0 < contact_weight(0.3) < contact_weight(0.8) < 1
    assert contact_weight(1.0) == 1
    assert abs(contact_weight(0.9999) - contact_weight(1.0001)) < 0.001
    assert contact_weight(0.8, noise_area_mm2=0.9) == 0
    assert contact_weight(1.5, noise_area_mm2=0.9) == 1
    with pytest.raises(ValueError):
        contact_weight(float('nan'))


def test_reference_stack_uses_capture_map_and_rejects_empty_pool():
    from twm.force_recovery.run_episode import reference_stack

    frames = np.arange(100)[:, None, None, None] * np.ones((100, 2, 2, 3))
    index_map = np.arange(65) + 7
    intensity = np.ones(65)
    intensity[[0, 32, 64]] = 0
    fresh = np.ones(65, bool)
    stack = reference_stack(frames, index_map, intensity, fresh)
    np.testing.assert_array_equal(stack[:, 0, 0, 0], [7, 39, 71])
    with pytest.raises(ValueError, match='reference'):
        reference_stack(frames, index_map, intensity, np.zeros(65, bool))
    with pytest.raises(ValueError, match='alignment'):
        reference_stack(frames, index_map[:-1], intensity, fresh)


def test_reference_noise_floor_is_zero_for_identical_unloaded_frames():
    from twm.force_recovery.run_episode import reference_noise_area

    images = np.full((5, 240, 320, 3), 120, np.float32)
    assert reference_noise_area(images) == 0
    with pytest.raises(ValueError, match='two'):
        reference_noise_area(images[:1])


def test_heldout_gain_fit_does_not_see_heldout_depths(monkeypatch):
    from twm.force_recovery import react_calib as rc

    rng = np.random.default_rng(7)
    rows = []
    for i in range(90):
        v = rng.uniform(1, 8)
        rows.append(dict(vol=v, vol2=v*v, maxd=v/2, area=2+v,
                         h1=v*np.sqrt(2+v)/2, cx=rng.uniform(70, 230),
                         cy=rng.uniform(60, 180), x=i, y=0, z=v/2, f=v))
    monkeypatch.setattr(rc, '_load', lambda recon: (rows, lambda k: np.array([r[k] for r in rows])))
    solve = np.linalg.lstsq
    sizes = []
    def record(a, b, **kwargs):
        sizes.append(len(a))
        return solve(a, b, **kwargs)
    monkeypatch.setattr(np.linalg, 'lstsq', record)
    _, before = rc.fit(report=False, holdout=True, extend_range=False)
    assert sizes[0] == 60
    held = set(np.random.default_rng(0).permutation(90)[:30])
    for i in held:
        rows[i]['z'] *= 100
    _, after = rc.fit(report=False, holdout=True, extend_range=False)
    np.testing.assert_allclose(before['pred'], after['pred'], atol=1e-10)


def test_review_cache_is_published_only_after_write_completes(tmp_path, monkeypatch):
    from twm.force_recovery import task_review as review

    dest = tmp_path / 'cache.joblib'
    dest.write_bytes(b'previous complete cache')
    def dump(payload, stream, **kwargs):
        stream.write(b'new complete cache')
        assert dest.read_bytes() == b'previous complete cache'
    monkeypatch.setattr(review.joblib, 'dump', dump)
    review.save_cache({}, dest)
    assert dest.read_bytes() == b'new complete cache'
