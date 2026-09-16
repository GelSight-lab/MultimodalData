"""React's newton scale, calibrated on features from the CURRENT pipeline.

The bug this replaces
---------------------
`showcase._glowtact_calib` fitted its weights on `lut_full.json`, a cache whose
`vol`/`area` are in PIXEL units and which predates the current reconstruction,
then applied those weights to `debug_gallery.stages()` output, which is in mm.
Measured mismatch: area 16874 px² vs 22 mm² (~580x), vol ~1000x — while `maxd`
is mm on both sides, so it is a PARTIAL mismatch, the kind that produces a
plausible-looking number instead of an obvious blow-up.

End-to-end on its own calibration objects it scored **rho 0.143, MAE 1.23 N**,
predicting 0.31-1.86 N for a true 0.19-4.75 N. Every React newton on the site,
in the clips, and in the exported dataset columns came from that map.

The fix is not a unit conversion factor. It is to compute the calibration
features with the SAME function that runs at inference, so the two cannot drift
again: `stages()` in, `stages()` out. The cache is rebuilt from the raw GlowTact
frames once and stored in mm.

Run:
    python -m force_recovery.react_calib build     # rebuild the mm-unit cache
    python -m force_recovery.react_calib fit       # fit + held-out report
"""
from __future__ import annotations

import json
import sys

import numpy as np

from .lut_calibration import GLOWTACT, MM_PER_PIXEL, PAT, crop
from .run_episode import OUT_ROOT

# THE name of this calibration, imported by everything that has to say which
# map produced a newton (npz metadata, the export sidecar, the site's action
# trace). A copy in each of those places is how the site went on advertising
# "LUT v2, GlowTact-calibrated" after that map had been replaced.
CALIBRATION_NAME = ("react_calib (calibration-free recon + gain field + "
                    "clip correction + continuous contact + anchored 0-15 N tail v8)")

# WHICH RECONSTRUCTION THE FORCE CHANNEL IS COMPUTED FROM
#
# Decided by measurement, not preference. Same 478 GlowTact `round` presses,
# held out by press position, the identical split and the identical fitting
# code below — only the reconstruction swapped
# (`scripts/react_calib_recon_ab.py`):
#
#     recon              held-out rho   MAE      in view    clipped   shuffle
#     LUT                    0.763    1.113 N   0.932 (14)   0.725    +0.056
#     calibration-free       0.812    1.024 N   0.996 (20)   0.762    -0.015
#
# The calibration-free solve wins on every split and its shuffle control is
# the cleaner one. Its lack of a millimetre scale is irrelevant HERE: the
# least squares below determines one global factor, which is exactly the
# unknown.
#
# It is NOT irrelevant elsewhere. Anything that needs depth in millimetres —
# the exported penetration fraction, the gel-thickness bound, the 3D figures —
# keeps using the LUT, which is calibrated in mm. So the dataset carries a
# force from one reconstruction and a depth from the other, on purpose, and
# both say which.
FORCE_RECONSTRUCTION = "calibfree"


def force_stages(img, ref, recon: str | None = None) -> dict:
    """The reconstruction the force channel is built on — ONE definition.

    Returned in the shape `predict` and `build_cache` both expect, so the
    calibration and the inference cannot drift onto different reconstructions
    (that exact drift is what the module docstring above is about).
    """
    from .debug_gallery import stages
    recon = recon or FORCE_RECONSTRUCTION
    if recon == "calibfree":
        from . import calib_free as CF
        from .lut_calibration import MM_PER_PIXEL
        r = CF.reconstruct(img, ref, include_normals=False)
        d = np.clip(r["depth"], 0, None)
        # THE CONTACT MASK, NOT A FRACTION OF THE PEAK.
        #
        # A relative floor was used here because this depth has no millimetre
        # scale, so the LUT's absolute 0.05 mm would mean something else for
        # it. But "is this pixel in contact" never needed a depth scale: it is
        # decided on the raw difference image by `contact_mask`, which both
        # reconstructions already share.
        #
        # The relative floor cannot say "no contact" — every frame has a peak,
        # so 5% of it always selects something. Measured on React
        # episode_000, frames where the published channel reads 0 N and the
        # new one read over 0.5 N: the true contact covers 0.01-6.2% of the
        # frame while the relative floor marked 1.8-40.6%, over-counting by
        # 3-30x, and the inflated area drove the force to the isotonic's lower
        # clip (a suspicious number of frames at exactly 1.59 N).
        m = r["valid"]
        return {"depth": d, "feats": feature_vector(d, m), "contact": m,
                "recon": recon}
    st = stages(img, ref)
    return {"depth": st["depth"], "feats": st["feats"],
            "contact": st["depth"] > 0.05, "recon": recon}

CACHE = OUT_ROOT / "feature_cache" / "glowtact_round_mm.json"
TAIL_CACHE = OUT_ROOT / "feature_cache" / "glowtact_round_8_15_di4.json"


def cache_for(recon: str):
    """One cache per reconstruction, so both can be fitted by the SAME code.

    `verify_force_channel` used to carry its own copy of the fit in order to
    have a second arm. After the force channel moved to the calibration-free
    solve that copy became the same arm twice, and the LUT arm was left
    feeding `stages()` features into a model fitted on calibration-free ones —
    caught at runtime by the check in `predict`, which is what it is for.
    """
    return (CACHE if recon == FORCE_RECONSTRUCTION
            else CACHE.with_name(f"glowtact_round_{recon}.json"))
FEATURES = ("vol", "vol2", "maxd", "area", "h1")


def feature_vector(depth, mask) -> dict:
    """THE five features. One definition, one place.

    Both arguments are supplied by the caller because the two reconstructions
    decide contact differently and only they know how: the LUT is in
    millimetres and thresholds its own depth at the production 0.05 mm; the
    calibration-free solve has no millimetre scale and takes the contact mask
    that `calib_free.reconstruct` already computed from the difference image.
    What must NOT differ is everything after that — the summation, the units,
    the 99.8th percentile, the order of the five numbers.

    This function exists because it had forked. `force_recon_matrix._feats`
    (which feeds the site's results table, the cross-dataset matrix, the
    prediction scatter and the error analysis) kept a relative floor of 5% of
    the frame's peak for the calibration-free arm after the deployed estimator
    in `force_stages` had moved to the contact mask — so the site was
    evaluating a model the deployment no longer used. Measured on React
    episode_000 the relative floor over-counts contact area by 3-30x.
    """
    d = np.clip(np.asarray(depth, np.float64), 0, None)
    m = np.asarray(mask, bool)
    px = MM_PER_PIXEL ** 2
    area = float(m.sum() * px)
    maxd = float(np.percentile(d, 99.8))
    return {"vol": float(d[m].sum() * px), "vol2": float((d[m] ** 2).sum() * px),
            "maxd": maxd, "area": area, "h1": float(np.sqrt(area) * maxd)}
# Preserve the approved low-force fit; only the high-score tail uses >8 N labels.
BASE_MAX_N = 8.0
F_MAX_N = 15.0


def contact_weight(area_mm2: float, noise_area_mm2: float = 0.0) -> float:
    """Continuous evidence ramp; not a calibrated contact probability."""
    if not np.isfinite([area_mm2, noise_area_mm2]).all():
        raise ValueError("Contact areas must be finite")
    floor = max(30 * MM_PER_PIXEL ** 2, noise_area_mm2)
    full = max(1.0, floor + 0.5)
    return float(np.clip((area_mm2 - floor) / (full - floor), 0, 1))


def _basis(x, y):
    return np.column_stack([np.ones_like(x), x, y, x * x, y * y, x * y])


def _clip_fraction(area_mm2, cx, cy):
    """How much of the contact disc falls outside the usable crop, 0 when in."""
    r_px = np.sqrt(np.clip(area_mm2, 0, None) / np.pi) / MM_PER_PIXEL
    margin = np.minimum.reduce([cx - r_px - 24, 296 - (cx + r_px),
                                cy - r_px - 20, 220 - (cy + r_px)])
    return np.clip(-margin, 0, None) / np.maximum(r_px, 1e-6)


def _with_clip(X, area_mm2, cx, cy):
    c = _clip_fraction(np.asarray(area_mm2), np.asarray(cx), np.asarray(cy))
    return np.column_stack([X, X[:, 0] * c, c])


def build_cache(recon: str | None = None, *, tail: bool = False) -> None:
    """Recompute GlowTact `round` features for one reconstruction."""
    from PIL import Image
    from . import calib_free as CF

    if tail and ((recon or FORCE_RECONSTRUCTION) != 'calibfree' or CF.VALID_DI != 4):
        raise ValueError('The v8 tail cache requires calibration-free reconstruction with dI=4')

    ref = crop(np.asarray(Image.open(GLOWTACT / "round" / "initial.jpg")
                          .convert("RGB"))).astype(np.float32)
    rows = []
    files = sorted((GLOWTACT / "round").glob("*.jpg"))
    for i, p in enumerate(files):
        m = PAT.search(p.name)
        if not m:
            continue
        f = float(m["f"])
        low, high = (BASE_MAX_N, F_MAX_N) if tail else (0.15, BASE_MAX_N)
        if not (low < f <= high):
            continue
        img = crop(np.asarray(Image.open(p).convert("RGB"))).astype(np.float32)
        st = force_stages(img, ref, recon)
        d = st["depth"]
        mm = st["contact"]
        if mm.sum() < 30:
            continue
        yy, xx = np.nonzero(mm)
        w = d[mm]
        rows.append({**st["feats"], "f": f,
                     "x": float(m["x"]), "y": float(m["y"]),
                     "z": -float(m["z"]),
                     "cx": float((xx * w).sum() / w.sum()),
                     "cy": float((yy * w).sum() / w.sum())})
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(files)} -> {len(rows)} kept", flush=True)
    out = TAIL_CACHE if tail else cache_for(recon or FORCE_RECONSTRUCTION)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows))
    print(f"{len(rows)} frames -> {out}")


def _load(recon: str | None = None):
    c = cache_for(recon or FORCE_RECONSTRUCTION)
    if not c.exists():
        raise SystemExit(
            f"{c} is missing. This is the fitted-features cache — the "
            f"gel-indentation geometry measured against newtons in the "
            f"calibration experiment — NOT a pipeline product: no stage builds "
            f"it, and it is not in git. Restore it from the data disk or "
            f"re-run the calibration fit. (The old message said 'run build "
            f"first', which produces nothing of the kind.)")
    rows = json.loads(c.read_text())
    a = lambda k: np.array([r[k] for r in rows])          # noqa: E731
    return rows, a


def tail_holdout_mask(keys, base_keys, base_holdout):
    """Extend the original position split without leaking any old test position."""
    unseen = np.setdiff1d(np.unique(keys), np.unique(base_keys))
    additional = unseen[np.random.default_rng(0).permutation(len(unseen))[:len(unseen)//3]]
    return np.isin(keys, list(base_holdout) + additional.tolist())


def extend_isotonic(base, scores, forces):
    """Append measured high-load knots without changing the approved low curve."""
    from sklearn.isotonic import IsotonicRegression

    scores, forces = np.asarray(scores, float), np.asarray(forces, float)
    if (scores.shape != forces.shape or scores.ndim != 1 or
            not np.isfinite(scores).all() or not np.isfinite(forces).all() or
            np.any((forces <= BASE_MAX_N) | (forces > F_MAX_N))):
        raise ValueError('Tail requires finite measured forces in (8, 15] N')
    selected = scores > base.X_thresholds_[-1]
    if np.unique(scores[selected]).size < 2 or forces[selected].max() < 14:
        raise ValueError('Insufficient high-force support above the old score ceiling')
    tail = IsotonicRegression(y_min=base.y_thresholds_[-1], y_max=F_MAX_N,
                             out_of_bounds='clip').fit(scores[selected], forces[selected])
    return IsotonicRegression(out_of_bounds='clip').fit(
        np.r_[base.X_thresholds_, tail.X_thresholds_],
        np.r_[base.y_thresholds_, tail.y_thresholds_])


def fit(report: bool = True, holdout: bool = False,
        recon: str | None = None, *, legacy: bool = False, extend_range: bool = True):
    """Fit the newton scale; returns predict(stages_dict) -> N.

    Held out by PRESS POSITION, not at random: neighbouring frames of one press
    are near-duplicates, so a random split would score its own training data.
    `extend_range=False` reproduces v7. Legacy and LUT fits never append the
    calibration-free tail. Extended holdouts include all high-load test cases,
    including those below the join, and paired v7 values in `before_pred`.
    """
    from scipy.stats import spearmanr
    from sklearn.isotonic import IsotonicRegression

    recon = recon or FORCE_RECONSTRUCTION
    rows, a = _load(recon)
    f, cx, cy, z = a("f"), a("cx"), a("cy"), a("z")
    X0 = np.column_stack([a(k) for k in FEATURES])

    key = np.round(a("x"), 1) * 1000 + np.round(a("y"), 1)
    uniq = np.unique(key)
    rng = np.random.default_rng(0)
    hold = set(uniq[rng.permutation(len(uniq))[:max(len(uniq) // 3, 1)]])
    te = np.array([k in hold for k in key])
    tr = ~te

    # spatial gain field u(x,y): the LED falloff makes the same press read
    # differently across the pad. Fitted on depth vs commanded z, as before.
    PHI = _basis(cx / 100, cy / 100)
    gain_rows = np.ones(len(f), bool) if legacy else tr
    w, *_ = np.linalg.lstsq(
        np.hstack([PHI * z[:, None], -PHI])[gain_rows],
        a("maxd")[gain_rows], rcond=None)
    gain = w[:6]

    def u_at(px, py):
        return 1.0 / np.clip(_basis(np.atleast_1d(px / 100),
                                    np.atleast_1d(py / 100)) @ gain, 0.15, 3.0)

    u = u_at(cx, cy)
    # Clipping correction instead of an in-view FILTER. Only 10% of these
    # 0-8 N presses are fully in view (median border margin -31 px), so
    # filtering leaves 48 frames. Feeding the clipped fraction as a feature
    # keeps every sample and measurably beats both alternatives on the
    # held-out split: 0.607 (plain) -> 0.712 (margin) -> 0.739 (clip frac),
    # MAE 1.73 -> 1.23 N, and it restores the low end (1.32 -> 0.44 N).
    X = _with_clip(np.column_stack([X0[:, 0] * u, X0[:, 1] * u ** 2,
                                    X0[:, 2] * u, X0[:, 3], X0[:, 4] * u]),
                   a("area"), cx, cy)

    wl, *_ = np.linalg.lstsq(X[tr], f[tr], rcond=None)
    iso = IsotonicRegression(out_of_bounds="clip").fit(X[tr] @ wl, f[tr])
    base_iso = iso
    tail_info = None
    high_held = None
    if extend_range and not legacy and recon == FORCE_RECONSTRUCTION:
        if not TAIL_CACHE.exists():
            raise FileNotFoundError(f'{TAIL_CACHE}: run python -m twm.force_recovery.react_calib build-tail')
        high = json.loads(TAIL_CACHE.read_text())
        h = lambda k: np.array([r[k] for r in high])
        hu = u_at(h('cx'), h('cy'))
        HX = _with_clip(np.column_stack([h('vol')*hu, h('vol2')*hu**2,
                         h('maxd')*hu, h('area'), h('h1')*hu]), h('area'), h('cx'), h('cy'))
        hs = HX @ wl
        ht = tail_holdout_mask(np.round(h('x'), 1)*1000+np.round(h('y'), 1), key, hold)
        iso = extend_isotonic(base_iso, hs[~ht], h('f')[~ht])
        hw = np.array([contact_weight(v) for v in h('area')[ht]])
        high_held = {'pred': iso.predict(hs[ht])*hw, 'f': h('f')[ht],
                     'clip': _clip_fraction(h('area'), h('cx'), h('cy'))[ht],
                     'cx': h('cx')[ht], 'cy': h('cy')[ht],
                     'before_pred': base_iso.predict(hs[ht])*hw, 'score': hs[ht]}
        tail_info = {'n_samples': len(high), 'n_train': int((~ht).sum()),
                     'n_heldout': int(ht.sum()),
                     'n_tail_fit': int(np.sum((~ht) & (hs > base_iso.X_thresholds_[-1]))),
                     'join_score': float(base_iso.X_thresholds_[-1]),
                     'join_force_n': float(base_iso.y_thresholds_[-1]),
                     'support_max_score': float(iso.X_thresholds_[-1]),
                     'training_label_max_n': float(h('f')[~ht].max())}
    # The held-out arrays are returned on request so a diagnostic can slice
    # them (in-view vs clipped, say) WITHOUT re-implementing the fit. A
    # diagnostic that rebuilds the model it is diagnosing measures its own
    # copy.
    contact_weights = (np.ones(te.sum()) if legacy else
                       np.array([contact_weight(v) for v in a("area")[te]]))
    held = {"pred": iso.predict(X[te] @ wl) * contact_weights, "f": f[te],
            "clip": _clip_fraction(a("area"), cx, cy)[te],
            "cx": cx[te], "cy": cy[te], 'score': X[te] @ wl,
            'before_pred': base_iso.predict(X[te] @ wl) * contact_weights}
    if high_held is not None:
        held = {k: np.r_[v, high_held[k]] for k, v in held.items()}
    if report:
        p = held["pred"]
        rho = spearmanr(p, held['f']).statistic
        sh = spearmanr(p, rng.permutation(held['f'])).statistic
        print(f"  base n={len(f)} ({tr.sum()} fit / {te.sum()} held-out); tail={tail_info}")
        print(f"  held-out rho={rho:.3f}  MAE={np.abs(p - held['f']).mean():.3f} N"
              f"  shuffled control={sh:+.3f}")
        print(f"  predicts {p.min():.2f}-{p.max():.2f} N "
              f"for a true {held['f'].min():.2f}-{held['f'].max():.2f} N")

    def predict(st: dict, *, noise_area_mm2: float = 0.0) -> float:
        """`st` must come from `force_stages`, not from `stages`.

        Checked at runtime rather than by convention. The weights below were
        fitted on ONE reconstruction; handing them another one's features is
        silent and produces plausible newtons, which is precisely the failure
        this module was created to undo (a pixel-unit weight vector scored
        mm-unit features for weeks and read as rho 0.143).
        """
        if st.get("recon") != recon:
            raise TypeError(
                f"force prediction fed a {st.get('recon') or 'plain stages()'} "
                f"reconstruction, but this calibration was fitted on "
                f"{recon!r} — call react_calib.force_stages(recon=...)")
        ft = st["feats"]
        if not np.isfinite([ft[k] for k in FEATURES]).all():
            raise ValueError("Nonfinite force features")
        weight = (float(ft["area"] >= 1.0) if legacy else
                  contact_weight(ft["area"], noise_area_mm2))
        if weight == 0:
            return 0.0
        d = st["depth"]
        mm = st.get("contact")
        if mm is None:
            mm = d > 0.05
        if mm.sum() < 30:
            return 0.0
        yy, xx = np.nonzero(mm)
        ww = d[mm]
        if not np.isfinite(ww).all():
            raise ValueError("Nonfinite contact depth")
        if ww.sum() <= 1e-12:
            return 0.0
        uu = float(u_at(float((xx * ww).sum() / ww.sum()),
                        float((yy * ww).sum() / ww.sum()))[0])
        # (centroid reused below for the clipping correction)
        pcx = float((xx * ww).sum() / ww.sum())
        pcy = float((yy * ww).sum() / ww.sum())
        v = np.array([ft["vol"] * uu, ft["vol2"] * uu ** 2, ft["maxd"] * uu,
                      ft["area"], ft["h1"] * uu])
        v = _with_clip(v[None, :], [ft["area"]], [pcx], [pcy])[0]
        return float(np.clip(iso.predict([float(v @ wl)])[0], 0, F_MAX_N) * weight)

    predict.force_ceiling_n = float(iso.y_thresholds_[-1])
    predict.tail_info = tail_info

    return (predict, held) if holdout else predict


HOLDOUT_JSON = OUT_ROOT / "feature_cache" / "react_holdout.json"


def range_report() -> dict:
    """Paired v7/v8 errors, including high loads whose scores never reach the tail."""
    import hashlib

    model, held = fit(report=False, holdout=True)
    result = {'calibration': CALIBRATION_NAME, 'output_range_n': [0, F_MAX_N],
              'fitted_ceiling_n': model.force_ceiling_n, 'tail': model.tail_info,
              'react_force_ground_truth': False, 'measured_zero_force_labels': False,
              'position_split': 'Original low-range split retained; one third of new positions held out',
              'low_range_max_prediction_change_n': float(np.max(np.abs(
                  held['pred'][held['f'] <= 8]-held['before_pred'][held['f'] <= 8]))),
              'high_force_below_join_count': int(np.sum((held['f'] > 8) &
                  (held['score'] <= model.tail_info['join_score']))), 'bands': []}
    for label, selected in [('0-1', held['f'] < 1),
                            ('1-4', (held['f'] >= 1) & (held['f'] < 4)),
                            ('4-8', (held['f'] >= 4) & (held['f'] <= 8)),
                            ('8-12', (held['f'] > 8) & (held['f'] < 12)),
                            ('12-15', held['f'] >= 12),
                            ('0-8', held['f'] <= 8), ('8-15', held['f'] > 8),
                            ('all', np.ones(len(held['f']), bool))]:
        result['bands'].append({'range_n': label, 'n': int(selected.sum()),
            'v7_mae_n': float(np.mean(np.abs(held['before_pred'][selected]-held['f'][selected]))),
            'v8_mae_n': float(np.mean(np.abs(held['pred'][selected]-held['f'][selected])))})
    result['heldout'] = {k: v.tolist() for k, v in held.items()}
    result['cache_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (CACHE, TAIL_CACHE)}
    path = OUT_ROOT/'feature_cache'/'react_range15_holdout.json'
    path.write_text(json.dumps(result, indent=2))
    print(path)
    return result


def holdout_report() -> dict:
    """Both arms held out by press position, WRITTEN DOWN.

    The results page used to carry "rho 0.812 against the LUT's 0.763, MAE
    1.024 against 1.113 N" as literal text in the HTML. When the feature step
    moved to the contact mask those became 0.781 and 1.072 and the page kept
    saying 0.812 — a typed number cannot go stale loudly. Every number on the
    site is read from an artifact; this is the artifact for this sentence.

    IT ALSO CARRIES ITS OWN UNCERTAINTY, because the margin does not survive
    it. "Calibration-free wins, rho 0.781 against the LUT's 0.763" reads like a
    decision; a paired bootstrap over the same 158 held-out presses puts that
    +0.018 at 95% CI [-0.081, +0.120], with calibration-free ahead in only 61%
    of resamples. MAE agrees: +0.040 N, CI [-0.149, +0.238], 66%. This holdout
    CANNOT separate the two arms, and `winner` on its own invited exactly the
    reading it does not support.

    The reason to deploy calibration-free is the external datasets, where the
    labels are real newtons and n is 605 to 2,000 — it leads on all five
    (`force_recon_matrix.json`). React's own 158 presses are corroboration at
    best, and this artifact now says so in a field rather than in a comment.
    """
    from scipy.stats import spearmanr

    out, held = {}, {}
    for arm in ("calibfree", "lut"):
        _predict, h = fit(report=False, holdout=True, recon=arm, extend_range=False)
        held[arm] = (np.asarray(h["pred"]), np.asarray(h["f"]))
        out[arm] = {"rho": float(spearmanr(h["pred"], h["f"]).statistic),
                    "mae_n": float(np.abs(h["pred"] - h["f"]).mean()),
                    "n_heldout": int(len(h["f"]))}
    out["higher_rho"] = max(("calibfree", "lut"), key=lambda k: out[k]["rho"])

    # Paired: one resample of FRAMES scores both arms, so their correlation is
    # preserved. Bootstrapping the arms separately would compare two
    # independent draws and inflate the spread.
    ta, tb = held["calibfree"][1], held["lut"][1]
    if not np.array_equal(ta, tb):
        raise AssertionError("arms held out on different frames — the "
                             "comparison would be meaningless")
    rng = np.random.default_rng(0)
    drho, dmae = [], []
    for _ in range(4000):
        i = rng.integers(0, len(ta), len(ta))
        if len(np.unique(ta[i])) < 5:
            continue
        drho.append(spearmanr(held["calibfree"][0][i], ta[i]).statistic
                    - spearmanr(held["lut"][0][i], tb[i]).statistic)
        dmae.append(np.abs(held["lut"][0][i] - tb[i]).mean()
                    - np.abs(held["calibfree"][0][i] - ta[i]).mean())
    drho, dmae = np.array(drho), np.array(dmae)
    out["paired_bootstrap"] = {
        "n_resamples": int(len(drho)),
        "d_rho_mean": float(drho.mean()),
        "d_rho_ci95": [float(np.percentile(drho, 2.5)),
                       float(np.percentile(drho, 97.5))],
        "d_rho_frac_calibfree_ahead": float((drho > 0).mean()),
        "d_mae_mean_n": float(dmae.mean()),
        "d_mae_ci95_n": [float(np.percentile(dmae, 2.5)),
                         float(np.percentile(dmae, 97.5))],
        "separates_the_arms": bool(np.percentile(drho, 2.5) > 0)}
    # Named `higher_rho`, not `winner`. The field was read as a verdict and
    # printed on the site as one; it is only ever an argmax over two numbers
    # whose difference this same artifact reports as indistinguishable from
    # zero. `verdict` states what the data supports, so a consumer that quotes
    # the obvious field cannot overstate it.
    out["verdict"] = (f"{out['higher_rho']} has the higher rho, but this "
                      f"holdout does not separate the arms"
                      if not out["paired_bootstrap"]["separates_the_arms"]
                      else f"{out['higher_rho']} is ahead beyond the "
                           f"bootstrap CI")
    HOLDOUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    HOLDOUT_JSON.write_text(json.dumps(out, indent=1))
    for arm in ("calibfree", "lut"):
        print(f"  {arm:10s} rho {out[arm]['rho']:.3f}  "
              f"MAE {out[arm]['mae_n']:.3f} N  n={out[arm]['n_heldout']}")
    b = out["paired_bootstrap"]
    print(f"  paired d_rho {b['d_rho_mean']:+.4f}  "
          f"95% CI [{b['d_rho_ci95'][0]:+.4f}, {b['d_rho_ci95'][1]:+.4f}]  "
          f"calibfree ahead in {b['d_rho_frac_calibfree_ahead']*100:.0f}% "
          f"-> separates the arms: {b['separates_the_arms']}")
    print(f"-> {HOLDOUT_JSON}")
    return out


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "fit"
    if cmd == "build":
        build_cache()
    elif cmd == "build-tail":
        build_cache(tail=True)
    elif cmd == 'range-report':
        range_report()
    elif cmd == "holdout":
        holdout_report()
    else:
        fit()
