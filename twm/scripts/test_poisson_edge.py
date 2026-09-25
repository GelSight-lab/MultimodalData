"""Does the boundary condition corrupt contacts that reach the frame edge?

Not an opinion about a figure: surfaces whose height is known analytically,
differentiated exactly, integrated by both solvers, scored against the truth.

Three cases, all the same bump, only its position moves:

    centre   — the flat-border assumption is TRUE
    edge     — bump centred on the right border, gel genuinely depressed there
    corner   — bump in the corner, two borders depressed

Both solvers recover height only up to an additive constant, so each result is
compared after removing its own median over the flat region — otherwise the
test would measure the datum, not the shape.

    python -m scripts.test_poisson_edge
"""
from __future__ import annotations

import numpy as np
import sys as _sys
from pathlib import Path as _Path
# repo root, so `force_recovery` / `twm` / `react_toolbox` import however
# this file is invoked. Six scripts lacked this and failed at import; all
# six sat in validate_all's "slow" skip list, so nothing ran them.
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))


from force_recovery.poisson import poisson_dirichlet, poisson_neumann

H, W = 240, 320
SIGMA = 34.0
AMP = 2.0                     # mm, a typical React press
TOL_CENTRE = 0.02             # of peak: both must be near-exact here
TOL_EDGE = 0.10               # of peak: what a usable solver should manage


def surface(cx: float, cy: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gaussian bump and its EXACT analytic gradient."""
    y, x = np.mgrid[0:H, 0:W].astype(np.float64)
    r2 = (x - cx) ** 2 + (y - cy) ** 2
    z = AMP * np.exp(-r2 / (2 * SIGMA ** 2))
    gx = z * (-(x - cx) / SIGMA ** 2)
    gy = z * (-(y - cy) / SIGMA ** 2)
    return z, gx, gy


def _err(rec: np.ndarray, truth: np.ndarray) -> float:
    """RMSE over the whole frame, after matching the datum, as a fraction of peak."""
    flat = truth < 0.02 * truth.max()
    off = np.median(rec[flat]) - np.median(truth[flat]) if flat.any() else 0.0
    return float(np.sqrt(np.mean((rec - off - truth) ** 2)) / truth.max())


def main() -> int:
    cases = {"centre": (W / 2, H / 2),
             "edge":   (W - 1.0, H / 2),
             "corner": (W - 1.0, H - 1.0)}
    bad, rows = [], []
    for name, (cx, cy) in cases.items():
        z, gx, gy = surface(cx, cy)
        d = _err(poisson_dirichlet(gx, gy), z)
        n = _err(poisson_neumann(gx, gy), z)
        rows.append((name, d, n))
        print(f"  {name:7s} border truth {z[:, -1].max():.2f} mm   "
              f"DST(Dirichlet) {d*100:6.1f}%   DCT(Neumann) {n*100:6.1f}%")
        tol = TOL_CENTRE if name == "centre" else TOL_EDGE
        if n > tol:
            bad.append(f"{name}: Neumann error {n*100:.1f}% of peak "
                       f"(> {tol*100:.0f}%)")
    centre = dict((r[0], r) for r in rows)["centre"]
    if centre[1] > TOL_CENTRE:
        bad.append(f"centre: the ORIGINAL solver fails a case it should pass "
                   f"({centre[1]*100:.1f}%) — the harness is wrong, not it")
    edge = dict((r[0], r) for r in rows)["edge"]
    if not edge[1] > 3 * edge[2]:
        bad.append(f"edge: Dirichlet {edge[1]*100:.1f}% vs Neumann "
                   f"{edge[2]*100:.1f}% — the bug this test documents is not "
                   f"reproducing; do not ship the fix on a stale claim")
    for b in bad:
        print(f"  FAIL: {b}")
    print(f"poisson-edge: {len(bad)} problem(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
