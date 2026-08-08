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
CALIBRATION_NAME = "react_calib (stages + gain field + clip correction)"

CACHE = OUT_ROOT / "feature_cache" / "glowtact_round_mm.json"
FEATURES = ("vol", "vol2", "maxd", "area", "h1")
# The React clips span roughly 0-8 N; calibrating past that would fit the
# isotonic tail on presses the deployment never sees.
F_MAX_N = 8.0


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


def build_cache() -> None:
    """Recompute GlowTact `round` features with the CURRENT stages()."""
    from PIL import Image
    from .debug_gallery import stages

    ref = crop(np.asarray(Image.open(GLOWTACT / "round" / "initial.jpg")
                          .convert("RGB"))).astype(np.float32)
    rows = []
    files = sorted((GLOWTACT / "round").glob("*.jpg"))
    for i, p in enumerate(files):
        m = PAT.search(p.name)
        if not m:
            continue
        f = float(m["f"])
        if not (0.15 < f <= F_MAX_N):
            continue
        img = crop(np.asarray(Image.open(p).convert("RGB"))).astype(np.float32)
        st = stages(img, ref)
        d = st["depth"]
        mm = d > 0.05
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
    CACHE.write_text(json.dumps(rows))
    print(f"{len(rows)} frames -> {CACHE}")


def _load():
    if not CACHE.exists():
        raise SystemExit("run `build` first")
    rows = json.loads(CACHE.read_text())
    a = lambda k: np.array([r[k] for r in rows])          # noqa: E731
    return rows, a


def fit(report: bool = True):
    """Fit the newton scale; returns predict(stages_dict) -> N.

    Held out by PRESS POSITION, not at random: neighbouring frames of one press
    are near-duplicates, so a random split would score its own training data.
    """
    from scipy.stats import spearmanr
    from sklearn.isotonic import IsotonicRegression

    rows, a = _load()
    f, cx, cy, z = a("f"), a("cx"), a("cy"), a("z")
    X0 = np.column_stack([a(k) for k in FEATURES])

    # spatial gain field u(x,y): the LED falloff makes the same press read
    # differently across the pad. Fitted on depth vs commanded z, as before.
    PHI = _basis(cx / 100, cy / 100)
    w, *_ = np.linalg.lstsq(np.hstack([PHI * z[:, None], -PHI]), a("maxd"),
                            rcond=None)
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

    key = np.round(a("x"), 1) * 1000 + np.round(a("y"), 1)   # press position
    uniq = np.unique(key)
    rng = np.random.default_rng(0)
    hold = set(uniq[rng.permutation(len(uniq))[:max(len(uniq) // 3, 1)]])
    te = np.array([k in hold for k in key])
    tr = ~te

    wl, *_ = np.linalg.lstsq(X[tr], f[tr], rcond=None)
    iso = IsotonicRegression(out_of_bounds="clip").fit(X[tr] @ wl, f[tr])
    if report:
        p = iso.predict(X[te] @ wl)
        rho = spearmanr(p, f[te]).statistic
        sh = spearmanr(p, rng.permutation(f[te])).statistic
        print(f"  n={len(f)} ({tr.sum()} fit / {te.sum()} held-out by position)")
        print(f"  held-out rho={rho:.3f}  MAE={np.abs(p - f[te]).mean():.3f} N"
              f"  shuffled control={sh:+.3f}")
        print(f"  predicts {p.min():.2f}-{p.max():.2f} N "
              f"for a true {f[te].min():.2f}-{f[te].max():.2f} N")

    def predict(st: dict) -> float:
        ft = st["feats"]
        if ft["area"] < 1.0:
            return 0.0
        d = st["depth"]
        mm = d > 0.05
        if mm.sum() < 30:
            return 0.0
        yy, xx = np.nonzero(mm)
        ww = d[mm]
        uu = float(u_at(float((xx * ww).sum() / ww.sum()),
                        float((yy * ww).sum() / ww.sum()))[0])
        # (centroid reused below for the clipping correction)
        pcx = float((xx * ww).sum() / ww.sum())
        pcy = float((yy * ww).sum() / ww.sum())
        v = np.array([ft["vol"] * uu, ft["vol2"] * uu ** 2, ft["maxd"] * uu,
                      ft["area"], ft["h1"] * uu])
        v = _with_clip(v[None, :], [ft["area"]], [pcx], [pcy])[0]
        return float(max(0.0, iso.predict([float(v @ wl)])[0]))

    return predict


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "fit"
    if cmd == "build":
        build_cache()
    else:
        fit()
