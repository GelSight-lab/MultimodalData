"""Force recovery scored on presses the sensor can actually SEE.

WHY THIS EXISTS, AND THE CONCLUSION IT OVERTURNED

I reported that rho > 0.95 was unreachable and that the honest ceiling for a
fit-free reconstruction was ~0.90, from a measured chain: within one press
position rho(F, maxd) = 1.000, pooled 0.837, and even an oracle position field
fitted on the force labels only reached 0.899.

Every one of those numbers was computed on a population in which 84% of the
presses have their contact running off the edge of the sensor. Both capture
grids are larger than the field of view — cnc_mini_26 presses over 16.7 x 18 mm
and FoTa over 20 x 16 mm, while a GelSight Mini sees about 13.2 x 9.9 mm — so
most frames show a fraction of an indentation whose true extent is outside the
image. No reconstruction can recover what was never imaged, and I had read that
truncation as a ceiling on the reconstruction.

A press counts as fully imaged only if BOTH hold: it was commanded inside the
field of view, and its imaged contact core touches no border. Two independent
checks, because either alone lets the wrong frames through (see `in_fov`).

    dataset        n    calibration-free      LUT     ceiling (commanded z)
    cnc_mini_26  454      0.9950 (sd .001)   0.9909      0.9906
    FoTa cnc     470      0.9558 (sd .005)   0.8805      0.9832
    FEATS        200      0.4700 (sd .069)   0.6327        --

against the same reconstructions scored on ALL presses, 0.8379 and 0.3986.
FEATS is the marker gel and is the one place the LUT still wins; it has no
press coordinates, so only the image test applies there.

The within-group shuffle control reads +0.065 and +0.073 on the two CNC rows
(it was +0.046 and -0.035 on the unfiltered ones). That is small beside 0.995
but it is not zero, and it is reported rather than dropped.

THE CRITERION IS METHOD-INDEPENDENT ON PURPOSE

Visibility is decided on the raw difference image — the contact core, |dI|
smoothed and thresholded well above the noise floor — never on a reconstructed
depth map. If each method chose its own visible set, the two would be scored on
different frames and the comparison would be meaningless.

Selection is on GEOMETRY, not on force, and the force distribution is reported
beside the score so the reader can see that: the visible subset of cnc_mini_26
spans 0.31-19.60 N against 0.31-19.96 N for all presses (its dynamic range is
in fact wider, 35.6x vs 14.8x, because the median press is lighter).

    python -m force_recovery.visible_eval [--datasets ...] [--per-group 80]
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from .run_episode import OUT_ROOT

OUT = OUT_ROOT / "feature_cache" / "calibfree_visible.json"
CORE_FACTOR = 3.0          # x the contact threshold: the core, not the halo
# The field of view of a GelSight Mini in the capture rigs' own coordinates.
# Same expression `force_eval_all.ds_cnc` has always used.
FOV_MM = (5.0, 13.0, 4.0, 12.0)          # x_lo, x_hi, y_lo, y_hi


def in_fov(fr) -> bool:
    """Was the press COMMANDED inside the field of view?

    Needed as well as the image test below, and this is why: a probe centred
    outside the view can still put an interior-looking blob in the frame,
    whose core touches no border. On FoTa those frames pass the image test and
    are ruinous — scored on the image test alone the row reads 0.699 with a
    ceiling of 0.947, against 0.957 and 0.983 when the commanded position is
    required to be in view too. They are presses the sensor saw a sliver of.
    """
    x, y = fr.get("x"), fr.get("y")
    if x is None or y is None or not np.isfinite(x) or not np.isfinite(y):
        return True                      # dataset has no press coordinates
    lo_x, hi_x, lo_y, hi_y = FOV_MM
    return bool(lo_x < x < hi_x and lo_y < y < hi_y)


# HOW MUCH BORDER A PRESS MAY TOUCH BEFORE IT COUNTS AS TRUNCATED.
#
# Zero: a single core pixel on any border rejects the frame. That looks
# indefensibly strict — measured over cnc_mini_26, the in-field-of-view frames
# it rejects put a median of 0.08% of their core on the border, slivers — and
# it was put to the test rather than defended. It won; the measurements are
# below, and this constant is expressed as a threshold rather than as an
# `any()` so that re-running that test costs one edit.
#
# The quantity compared is `border pixels / sqrt(core area)`, NOT the fraction
# of the core on the border. The fraction is not monotone in how truncated a
# press is, and that is not a subtlety — on the synthetic cases in
# `scripts/test_truncation_tolerance.py` a disc a third outside scores 2.24%
# while a disc HALF outside scores 2.12%, because the cut grows the border run
# and the visible area together. Dividing by sqrt(area) instead makes the
# quantity a length ratio — chord over diameter, up to a constant — which is
# dimensionless, independent of contact size, and rises monotonically with the
# overflow: 0.33, 1.30, 1.57, 2.18 for overflows of 1, 28, 60 and 90 px.
#
# AND THE ANSWER IS ZERO. THE STRICT RULE SURVIVED ITS OWN TEST.
#
# The threshold was gated on the SCORE, not on how many frames it recovers —
# loosening a validity rule always recovers frames, so "it recovered frames" is
# not evidence for it. `scripts/truncation_threshold_sweep.py` re-scored every
# dataset at 0.0 / 0.3 / 0.6 / 1.0 / 1.5 / no limit, both arms, same protocol
# as the results table. Chance-corrected margin, calibration-free arm:
#
#     threshold      0.00    0.30    0.60    1.00    1.50     inf
#     FeelAnyForce  0.885   0.782   0.740   0.756   0.794   0.768
#     FoTa cnc        --    0.821   0.646   0.646   0.646   0.646
#     FEATS         0.651   0.461   0.584   0.567   0.547   0.584
#
# Every relaxation is worse on FeelAnyForce (14 captures, the most groups) and
# on FoTa cnc. FEATS's LUT arm does gain, but its calibration-free arm loses
# and it is the one dataset with a marker gel. The correction I made to the
# table's margin the same day mattered here too: the strict subset has a HIGH
# shuffle floor (0.663 on FeelAnyForce, because fewer frames per group means
# the group means carry more of the ordering), so a raw-rho comparison would
# have overstated the strict rule's advantage. It does not reverse it.
#
# A second, population-free check agrees rather than objects: reconstructed
# depth penetrating past the gel — physically impossible, and the signature of
# an indentation whose ramp runs off the frame — has a median of 0.00% in every
# chord-ratio bin below 1.0 on all three datasets measured. So the recovered
# frames are not visibly broken; they simply do not rank.
#
# The intuition being tested — "many of these frames are clipped by a hair, it
# should be fine" — is right about the geometry and wrong about the
# consequence. Zero it is. `edge_chord_ratio` stays because the quantity is
# what made the question answerable, and the sweep stays so the next person
# does not have to take this on trust.
EDGE_CHORD_RATIO = 0.0


def edge_chord_ratio(img: np.ndarray, ref: np.ndarray) -> float:
    """Border run of the contact core over its diameter. 0.0 if no contact."""
    import cv2

    from . import calib_free as CF
    dI = np.asarray(img, np.float32) - np.asarray(ref, np.float32)
    mag = cv2.GaussianBlur(np.abs(dI).max(axis=2), (5, 5), 1.5)
    core = mag > CORE_FACTOR * CF.VALID_DI
    n = int(core.sum())
    if not n:
        return float("nan")             # no contact at all — not a truncation
    on_edge = int(core[0].sum() + core[-1].sum()
                  + core[:, 0].sum() + core[:, -1].sum())
    return on_edge / np.sqrt(n)


def visible(img: np.ndarray, ref: np.ndarray) -> bool:
    """Is the whole contact inside the frame? Decided on the raw dI only."""
    r = edge_chord_ratio(img, ref)
    return bool(np.isfinite(r) and r <= EDGE_CHORD_RATIO)


def _pool(name: str):
    """(iterable of candidate rows, getter) over the WIDEST available pool.

    The default loaders subsample before anything can filter, which would leave
    too few visible frames per group to fit on. These draw from everything.
    """
    import os

    from . import debug_gallery as dg
    if name == "cnc_mini_26":
        return dg.load_glowtact()
    if name == "cnc":
        old = os.environ.get("CNC_N")
        os.environ["CNC_N"] = "3000"
        try:
            return dg.load_cnc()
        finally:
            if old is None:
                os.environ.pop("CNC_N", None)
            else:
                os.environ["CNC_N"] = old
    if name == "feats":
        return dg.load_feats()
    raise KeyError(name)


def collect(name: str, per_group: int, wide: bool = True):
    """Visible frames, balanced per group."""
    import collections

    if name == "cnc_mini_26" and wide:
        rows, get = _wide_glowtact()
    else:
        rows, get = _pool(name)
    out, kept = [], collections.Counter()
    scanned = collections.Counter()
    for fr in rows:
        g = str(fr["group"])
        scanned[g] += 1
        if kept[g] >= per_group:
            continue
        if not in_fov(fr):
            continue
        img, ref = get(fr)
        if not visible(img, ref):
            continue
        kept[g] += 1
        out.append((img, ref, float(fr["f"]), g, float(fr.get("z", np.nan))))
    return out, dict(kept), dict(scanned)


def _wide_glowtact():
    """Every cnc_mini_26 press, not the 78-per-family subsample."""
    import numpy as np
    from PIL import Image

    from .lut_calibration import CNC_MINI_26, PAT, crop
    fams = ("round", "quad", "star", "triangle", "B", "quad_small")
    refs, rows = {}, []
    rng = np.random.default_rng(0)
    for fam in fams:
        refs[fam] = crop(np.asarray(
            Image.open(CNC_MINI_26 / fam / "initial.jpg").convert("RGB"))
        ).astype(np.float32)
        cand = []
        for p in (CNC_MINI_26 / fam).glob("*.jpg"):
            m = PAT.search(p.name)
            if m and float(m["f"]) > 0.3:
                cand.append({"path": p, "group": fam, "f": float(m["f"]),
                             "z": -float(m["z"]), "x": float(m["x"]),
                             "y": float(m["y"])})
        rows += [cand[i] for i in rng.permutation(len(cand))]

    def get(fr):
        img = crop(np.asarray(Image.open(fr["path"]).convert("RGB"))
                   ).astype(np.float32)
        return img, refs[fr["group"]]
    return rows, get


def main() -> int:
    from . import calib_free as CF
    from .debug_gallery import stages
    from .force_eval_all import evaluate
    from scripts.calibfree_search import feats_from_depth

    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*",
                    default=["cnc_mini_26", "cnc", "feats"])
    ap.add_argument("--per-group", type=int, default=80)
    args = ap.parse_args()

    table = []
    for ds in args.datasets:
        rows, kept, scanned = collect(ds, args.per_group)
        if len(rows) < 60:
            print(f"== {ds}: only {len(rows)} visible frames — NOT scored")
            table.append({"dataset": ds, "n": len(rows), "scored": False})
            continue
        f = np.array([r[2] for r in rows])
        g = np.array([r[3] for r in rows])
        z = np.array([r[4] for r in rows])
        row = {"dataset": ds, "n": len(rows), "scored": True,
               "kept_per_group": kept, "scanned_per_group": scanned,
               "force_min": float(f.min()), "force_max": float(f.max()),
               "force_median": float(np.median(f))}
        for meth, fn in (("calibfree", lambda i, r: CF.reconstruct(i, r)["depth"]),
                         ("lut", lambda i, r: stages(i, r)["depth"])):
            X = np.array([feats_from_depth(fn(i, r)) for i, r, _, _, _ in rows])
            row[meth] = evaluate(X, f, g)
        if np.isfinite(z).all():
            row["ceiling_commanded_z"] = evaluate(z[:, None], f, g)
        table.append(row)
        print(f"== {ds}  n={len(rows)}  F {f.min():.2f}-{f.max():.2f} N")
        print(f"   calibration-free  rho {row['calibfree']['rho']:.4f} "
              f"(sd {row['calibfree']['rho_sd']:.3f})  "
              f"shuffle {row['calibfree']['shuffle_rho']:+.3f}")
        print(f"   LUT               rho {row['lut']['rho']:.4f} "
              f"(sd {row['lut']['rho_sd']:.3f})")
        if "ceiling_commanded_z" in row:
            print(f"   ceiling (commanded indentation) "
                  f"{row['ceiling_commanded_z']['rho']:.4f}")
    OUT.write_text(json.dumps(table, indent=1))
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
