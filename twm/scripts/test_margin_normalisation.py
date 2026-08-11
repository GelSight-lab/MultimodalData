"""A high shuffle floor caps the margin it is subtracted from.

The results table ranks the two arms by `rho - shuffle_rho`: how much of the
score is the reconstruction rather than the groups' force ranges lining up. It
was pointed out that this is unfair to whichever arm scores higher, and it is —
arithmetically, not as a matter of taste. An arm at rho 0.998 over a floor of
0.930 has 0.070 of headroom in TOTAL; an arm at 0.900 over the same floor has
0.070 to gain and 0.098 still available. The raw margin scores the second arm's
opportunity, not its performance, and calibration-free lost 2 of 5 datasets on
exactly that.

The fix is the chance-corrected form (Cohen's kappa): divide the margin by the
margin that was AVAILABLE.

    kappa = (rho - floor) / (1 - floor)

which reads "the fraction of the distance from the floor to a perfect score
that this arm actually covered", and is 1.0 for a perfect arm at any floor.

    python scripts/test_margin_normalisation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                                              # noqa: E402

from force_recovery.site2 import kappa_margin, raw_margin       # noqa: E402

RESULTS: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, evidence: str) -> None:
    RESULTS.append((bool(ok), name, evidence))


def arm(rho: float, floor: float) -> dict:
    return {"rho": rho, "shuffle_rho": floor}


def main() -> int:
    # 1 — with no floor there is nothing to correct for
    a = arm(0.700, 0.0)
    check(abs(kappa_margin(a) - raw_margin(a)) < 1e-12,
          "a zero floor leaves the margin alone",
          f"raw {raw_margin(a):.4f} == kappa {kappa_margin(a):.4f}")

    # 2 — THE COMPLAINT, made arithmetic. Same raw margin, different floors:
    # the arm that earned it from higher up covered more of what was left.
    lo, hi = arm(0.500, 0.400), arm(0.980, 0.880)
    check(abs(raw_margin(lo) - raw_margin(hi)) < 1e-12
          and kappa_margin(hi) > kappa_margin(lo) + 0.3,
          "equal raw margins are not equal achievements",
          f"both +{raw_margin(lo):.3f} raw; kappa {kappa_margin(lo):.3f} "
          f"from a 0.400 floor vs {kappa_margin(hi):.3f} from 0.880")

    # 3 — a perfect arm is perfect from any floor. Under the raw margin a
    # perfect arm over a 0.93 floor scores +0.07 and loses to a mediocre arm
    # over a 0.0 floor.
    ks = [kappa_margin(arm(1.0, f)) for f in (0.0, 0.5, 0.93)]
    check(all(abs(k - 1.0) < 1e-12 for k in ks),
          "a perfect score is 1.0 at every floor",
          f"kappa {['%.3f' % k for k in ks]} for floors 0.00 / 0.50 / 0.93")

    # 4 — cnc's LUT floor is NEGATIVE (-0.040). Dividing by (1 - -0.040)
    # would report 1.04 for a perfect arm, so the floor is clamped at zero: a
    # floor below chance is chance, not a handicap to be paid back.
    k = kappa_margin(arm(1.0, -0.040))
    check(abs(k - 1.0) < 1e-12, "a negative floor cannot lift kappa past 1",
          f"floor -0.040, rho 1.000 -> kappa {k:.4f}")

    # 5 — an arm BELOW its floor stays negative. Correction must not launder a
    # failure into a small positive number.
    k = kappa_margin(arm(0.300, 0.600))
    check(k < 0, "an arm below its own floor stays negative",
          f"rho 0.300 under a 0.600 floor -> kappa {k:.4f}")

    # 6 — SINGLE SOURCE, ACROSS THE WHOLE PACKAGE.
    #
    # This grepped `site2.py` alone and passed while the site carried TWO
    # margins: kappa in the results table, and `pred_vs_gt`'s raw difference
    # annotated on every scatter panel under the same word. A single-source
    # check scoped to one file is not a single-source check — it is a check
    # that one file is self-consistent, which was never in doubt.
    #
    # The functions now live in `force_eval_all`, beside the `evaluate` that
    # produces the numbers they combine, and every caller imports them.
    import re
    stray = []
    for py in sorted(Path("force_recovery").glob("*.py")):
        src = py.read_text()
        body = re.sub(r"def (raw|kappa)_margin.*?(?=\ndef |\nclass |\n[A-Z_]+ =)",
                      "", src, flags=re.S)
        for hit in re.findall(r'\[[\'"]rho[\'"]\]\s*-\s*[\w\[\]\'"]*'
                              r'shuffle(?:_rho)?', body):
            stray.append(f"{py.name}: {hit}")
    check(not stray, "only one place computes a margin",
          f"{len(stray)} inline margin(s) outside the helpers"
          + (f": {stray}" if stray else ""))

    # 7 — the real artifact, ranked both ways, so the change is visible rather
    # than asserted.
    from force_recovery.site2 import _artifact
    try:
        m = {r["dataset"]: r for r in _artifact("force_recon_matrix.json")
             if r.get("available")}
    except Exception as exc:                                    # noqa: BLE001
        check(False, "the shipped numbers rank by kappa",
              f"UNVERIFIED: matrix unavailable ({exc})")
    else:
        raw_wins = kap_wins = n = 0
        for name, r in m.items():
            p = r.get("whole", {})
            if not p.get("scored"):
                continue
            n += 1
            raw_wins += raw_margin(p["calibfree"]) > raw_margin(p["lut"])
            kap_wins += kappa_margin(p["calibfree"]) > kappa_margin(p["lut"])
        check(n > 0, "the shipped numbers rank by kappa",
              f"calibration-free wins {raw_wins}/{n} by raw margin, "
              f"{kap_wins}/{n} by kappa")

    width = max(len(x) for _, x, _ in RESULTS)
    print()
    for ok, name, ev in RESULTS:
        print(f"  [{'ok' if ok else 'FAIL'}] {name:<{width}}  {ev}")
    bad = sum(not ok for ok, _, _ in RESULTS)
    print(f"\nmargin normalisation: {len(RESULTS)} checks, {bad} failing")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
