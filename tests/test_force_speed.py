import numpy as np
import pytest


def images(kind='contact'):
    y, x = np.mgrid[:64, :80]
    ref = np.full((64, 80, 3), 100, np.float32)
    bump = np.exp(-((x-40)**2+(y-30)**2)/100)
    img = ref + (bump[..., None]*[20, -15, 30]).astype(np.float32)
    return (ref.copy() if kind == 'empty' else img), ref


def test_force_path_does_not_compute_unused_normals(monkeypatch):
    from twm.force_recovery import calib_free as cf, react_calib as rc

    def unused(*args, **kwargs):
        pytest.fail('Force inference computed an unused normal map')
    monkeypatch.setattr(cf, 'normals', unused)
    st = rc.force_stages(*images())
    assert np.isfinite(st['depth']).all()


def test_reconstruction_reuses_its_contact_mask(monkeypatch):
    from twm.force_recovery import calib_free as cf

    original = cf.contact_mask
    calls = []
    def counted(di):
        calls.append(1)
        return original(di)
    monkeypatch.setattr(cf, 'contact_mask', counted)
    cf.reconstruct(*images())
    assert len(calls) == 1


def test_optional_normals_preserves_every_other_result():
    from twm.force_recovery import calib_free as cf

    img, ref = images()
    full = cf.reconstruct(img, ref)
    lean = cf.reconstruct(img, ref, include_normals=False)
    assert 'normals' in full and 'normals' not in lean
    assert lean.keys() == full.keys()-{'normals'}
    for key in lean:
        np.testing.assert_array_equal(lean[key], full[key])


def test_explicit_gradient_mask_matches_internal_detection():
    from twm.force_recovery import calib_free as cf

    img, ref = images()
    di = img-ref
    expected = cf.gradients(di, ref_img=ref)
    actual = cf.gradients(di, ref_img=ref, valid=cf.contact_mask(di))
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('module_name', ['calib_free', 'debug_gallery'])
def test_empty_contact_does_not_run_integrator(monkeypatch, module_name):
    from twm.force_recovery import calib_free, debug_gallery, poisson

    def unused(*args, **kwargs):
        pytest.fail('An exactly empty contact mask should not run Poisson')
    monkeypatch.setattr(poisson, 'integrate', unused)
    img, ref = images('empty')
    result = (calib_free.reconstruct(img, ref) if module_name == 'calib_free'
              else debug_gallery.stages(img, ref))
    assert not result['depth'].any()
    assert not result['valid'].any()


def test_neumann_shape_cache_is_bounded_and_read_only():
    from twm.force_recovery import poisson

    assert hasattr(poisson, '_neumann_denominator')
    fn = poisson._neumann_denominator
    fn.cache_clear()
    first = fn((20, 30))
    assert fn((20, 30)) is first
    assert not first.flags.writeable
    y, x = np.meshgrid(np.arange(20), np.arange(30), indexing='ij')
    expected = (2*np.cos(np.pi*x/30)-2)+(2*np.cos(np.pi*y/20)-2)
    expected[0, 0] = 1
    np.testing.assert_array_equal(first, expected)
    for n in range(10, 20):
        fn((n, n))
    assert fn.cache_info().currsize <= 4


@pytest.mark.parametrize('shape', [(32, 40), (65, 79)])
def test_cached_neumann_solver_matches_original_arithmetic(shape):
    from scipy.fft import dctn, idctn
    from twm.force_recovery.poisson import divergence, poisson_neumann

    rng = np.random.default_rng(8)
    gx, gy = rng.normal(size=(2, *shape))
    f = divergence(gx, gy)
    m, n = shape
    y, x = np.meshgrid(np.arange(m), np.arange(n), indexing='ij')
    denominator = (2*np.cos(np.pi*x/n)-2)+(2*np.cos(np.pi*y/m)-2)
    denominator[0, 0] = 1
    transformed = dctn(f, type=2, norm='ortho')/denominator
    transformed[0, 0] = 0
    z = idctn(transformed, type=2, norm='ortho')
    np.testing.assert_array_equal(poisson_neumann(gx, gy), z-z.mean())


def test_weak_nonempty_contact_still_runs_force_integrator(monkeypatch):
    from twm.force_recovery import calib_free as cf, react_calib as rc, poisson

    img, ref = images()
    img = ref + (img-ref)*.22
    valid = cf.contact_mask(img-ref)
    assert valid.any() and np.max(np.abs(img-ref)) < 8
    original = poisson.integrate
    calls = []
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(poisson, 'integrate', counted)
    result = rc.force_stages(img, ref)
    assert len(calls) == 1
    np.testing.assert_array_equal(result['contact'], valid)


def test_trend_basis_cache_is_bounded_read_only_and_exact():
    from twm.force_recovery import poisson

    assert hasattr(poisson, '_trend_basis')
    fn = poisson._trend_basis
    fn.cache_clear()
    basis = fn((20, 30), 2)
    assert fn((20, 30), 2) is basis
    y, x = np.mgrid[:20, :30]
    x, y = x/30, y/20
    for actual, expected in zip(basis, [np.ones_like(x), x, y, x*x, y*y, x*y]):
        np.testing.assert_array_equal(actual, expected)
        assert not actual.flags.writeable
    for n in range(10, 20):
        fn((n, n), 2)
    assert fn.cache_info().currsize <= 4


@pytest.mark.parametrize('level', [0, 3.99, 4.01, 8, 40])
def test_contact_mask_matches_original_rgb_reduction(level):
    import cv2
    from twm.force_recovery import calib_free as cf

    di = np.random.default_rng(10).normal(0, level, (64, 80, 3)).astype(np.float32)
    magnitude = cv2.GaussianBlur(np.abs(di).max(2), (5, 5), 1.5)
    expected = cv2.morphologyEx((magnitude > cf.VALID_DI).astype(np.uint8),
                               cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
    np.testing.assert_array_equal(cf.contact_mask(di), expected)


@pytest.mark.parametrize('solver', ['auto', 'neumann', 'dirichlet'])
@pytest.mark.parametrize('marker', [False, True])
def test_empty_cf_retains_solver_label_and_diagnostic_maps(monkeypatch, solver, marker):
    from twm.force_recovery import calib_free as cf, poisson

    monkeypatch.setattr(poisson, 'free_boundary_ok', lambda ref: not marker)
    result = cf.reconstruct(*images('empty'), solver=solver)
    expected = ('dirichlet-marker-gel' if marker else 'neumann-detrended') if solver == 'auto' else solver
    assert result['solver'] == expected
    assert not result['gx'].any() and not result['gy'].any()
    np.testing.assert_array_equal(result['normals'][..., 2], 1)
    assert not result['normals'][..., :2].any()


@pytest.mark.parametrize('order', [1, 2])
def test_cached_trend_matches_original_arithmetic(order):
    from twm.force_recovery.poisson import detrend_flat

    rng = np.random.default_rng(7)
    z = rng.normal(size=(64, 80))
    flat = rng.random(z.shape) > .3
    ys, xs = np.nonzero(flat)
    x, y = xs/z.shape[1], ys/z.shape[0]
    cols = [np.ones_like(x), x, y]
    if order >= 2:
        cols += [x*x, y*y, x*y]
    c, *_ = np.linalg.lstsq(np.stack(cols, axis=1), z[flat], rcond=None)
    gy, gx = np.mgrid[:64, :80]
    x, y = gx/80, gy/64
    full = [np.ones_like(x), x, y]
    if order >= 2:
        full += [x*x, y*y, x*y]
    expected = z-sum(ci*fi for ci, fi in zip(c, full))
    np.testing.assert_array_equal(detrend_flat(z, flat, order), expected)


def test_benchmark_comparison_rejects_changed_fields_and_masks():
    import importlib.util

    assert importlib.util.find_spec('twm.force_recovery.benchmark_speed') is not None
    from twm.force_recovery.benchmark_speed import assert_equivalent

    original = {'contact': np.array([True, False]), 'depth': np.array([0., 1.]),
                'feats': {'area': 3.}, 'recon': 'calibfree'}
    assert_equivalent(original, original)
    for changed in [dict(original, contact=np.array([False, False])),
                    dict(original, depth=np.array([0., 1.01])),
                    dict(original, recon='lut'), {'depth': original['depth']}]:
        with pytest.raises(AssertionError):
            assert_equivalent(original, changed)
