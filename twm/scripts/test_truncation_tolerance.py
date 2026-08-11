"""The truncation rule, and the relaxation that was measured and rejected.

`visible()` rejects a frame if ANY pixel of the contact core touches ANY
border. That was challenged as too strict — most of what it throws away is
clipped by a hair, not running off the edge — and the challenge is right about
the geometry: on cnc_mini_26 the in-field-of-view frames it rejects put a
median of 0.08% of their core on the border.

It is wrong about the consequence. `scripts/truncation_threshold_sweep.py`
re-scored all five datasets at six thresholds, both reconstructions, on the
protocol the results table uses, and every relaxation was worse on the two
datasets with the most groups. `visible_eval.EDGE_CHORD_RATIO` carries the
table. This file locks in what that settled:

  1  the case geometry is what it claims (a checker, because it caught me)
  2  the rule's verdicts on known geometry
  3  the threshold form is EXACTLY the old any-border-pixel rule at 0.0 —
     the refactor that made the question askable must not have changed a
     single frame's verdict, and this checks that on real frames

    python scripts/test_truncation_tolerance.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                                              # noqa: E402

from force_recovery import calib_free as CF                     # noqa: E402
from force_recovery.visible_eval import (CORE_FACTOR,           # noqa: E402
                                         EDGE_CHORD_RATIO, visible)

H, W = 240, 320
LEVEL = 4.0 * CF.VALID_DI          # comfortably above CORE_FACTOR * VALID_DI


def _frame(cy: int, cx: int, r: int) -> tuple[np.ndarray, np.ndarray]:
    """A reference and an image differing by one filled disc."""
    ref = np.zeros((H, W, 3), np.float32)
    img = ref.copy()
    yy, xx = np.mgrid[0:H, 0:W]
    img[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = LEVEL
    return img, ref


# GEOMETRY, CHECKED — not eyeballed. My first attempt at a "grazing" case was
# a disc of radius 161 centred at (120, 160), which in a 240x320 frame runs 41
# px past the top AND the bottom: the case named "grazing" was the most
# truncated frame in the file. It failed against a candidate threshold and the
# threshold took the blame. Every case states its overflow in pixels and
# `test_case_geometry` recomputes it, so a case cannot silently stop meaning
# what its name says.
#
#   overflow = r - (distance from the centre to the nearest border)
#
# `want` is the SHIPPED rule's verdict. Everything that reaches a border is
# rejected, including the 1-px case — that is the decision the sweep made, not
# an oversight, and the file is named for the tolerance it declined to grant.
CASES = [
    # name, centre, radius, overflow px, expected visible()
    ("interior blob is whole",           (120, 160), 30, -89, True),
    ("blob grazing the border is cut",   (120,  32), 33,   1, False),
    ("blob a third outside is cut",      (120,  12), 40,  28, False),
    ("blob half outside is cut",         (120,   0), 60,  60, False),
    ("blob mostly outside is cut",       (120, -30), 60,  90, False),
]


def test_case_geometry() -> int:
    """Each case overflows the border by the number of pixels it claims."""
    bad = 0
    for name, (cy, cx), r, want_over, _ in CASES:
        got = r - min(cx, W - 1 - cx, cy, H - 1 - cy)
        ok = got == want_over
        bad += not ok
        print(f"  [{'ok' if ok else 'FAIL'}] {name:36s} overflow={got:+4d}px "
              f"claimed={want_over:+4d}px")
    return bad


def _old_rule(img, ref) -> bool:
    """The rule as it stood before the threshold form. Verbatim."""
    import cv2
    dI = np.asarray(img, np.float32) - np.asarray(ref, np.float32)
    mag = cv2.GaussianBlur(np.abs(dI).max(axis=2), (5, 5), 1.5)
    core = mag > CORE_FACTOR * CF.VALID_DI
    if not core.any():
        return False
    return not (core[0].any() or core[-1].any()
                or core[:, 0].any() or core[:, -1].any())


def test_refactor_changed_nothing(n: int = 120) -> int:
    """On REAL frames, the new form agrees with the old rule everywhere.

    The sweep concluded the threshold should stay at zero, which makes the
    refactor a pure no-op — and an unverified "pure no-op" that silently moved
    frames would invalidate every number the results table carries. Real
    frames, not synthetic discs: the synthetic cases cannot exercise multiple
    disconnected core blobs, sub-threshold speckle on a border, or a frame
    whose core is empty.
    """
    if EDGE_CHORD_RATIO != 0.0:
        print(f"  [--] threshold is {EDGE_CHORD_RATIO}, not 0.0 — the rule is "
              f"deliberately not the old one; identity NOT expected")
        return 0
    from force_recovery.force_recon_matrix import _rows
    bad = 0
    for ds in ("cnc_mini_26", "faf"):
        rows, get = _rows(ds)
        agree = same = 0
        for fr in rows[:n]:
            img, ref = get(fr)
            a, b = visible(img, ref), _old_rule(img, ref)
            agree += a == b
            same += 1
        ok = agree == same
        bad += not ok
        print(f"  [{'ok' if ok else 'FAIL'}] {ds:36s} {agree}/{same} frames "
              f"agree with the pre-refactor rule")
    return bad


def main() -> int:
    print("case geometry")
    bad = test_case_geometry()
    print("\nvisible()")
    for name, (cy, cx), r, _over, want in CASES:
        img, ref = _frame(cy, cx, r)
        got = visible(img, ref)
        ok = got == want
        bad += not ok
        print(f"  [{'ok' if ok else 'FAIL'}] {name:36s} visible={got} want={want}")
    print(f"\nthe refactor moved no frame (threshold {EDGE_CHORD_RATIO})")
    bad += test_refactor_changed_nothing()
    print(f"\ntruncation rule: {bad} failing")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
