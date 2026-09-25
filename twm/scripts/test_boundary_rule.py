"""The boundary-condition rule is stable within a sensor, and it is data-driven.

Two properties, both of which were violated by an earlier version of the rule
and neither of which a figure would have shown:

  1. STABLE. The choice must not vary frame to frame inside one dataset. The
     first rule decided from the contact's anchor region, so on FEATS it chose
     the free boundary on 20% of frames and the clamped one on 80% — two
     height conventions inside one feature table, worth -0.044 rho on the LUT.
     It is now decided from the REFERENCE frame, which is constant per sensor.

  2. DATA-DRIVEN. It must actually be reading the gel, not a dataset name.
     Marker gel -> clamped, markerless -> free, asserted per dataset.

    python -m scripts.test_boundary_rule
"""
from __future__ import annotations

import numpy as np
import sys as _sys
from pathlib import Path as _Path
# repo root, so `force_recovery` / `twm` / `react_toolbox` import however
# this file is invoked. Six scripts lacked this and failed at import; all
# six sat in validate_all's "slow" skip list, so nothing ran them.
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))


from force_recovery import debug_gallery as dg
from force_recovery.poisson import free_boundary_ok

# (loader, expect_free_boundary). FEATS is the marker gel.
DATASETS = (("cnc_mini_26", "load_glowtact", True),
            ("cnc", "load_cnc", True),
            ("feats", "load_feats", False))
N = 12


def main() -> int:
    bad = []
    for name, loader, expect in DATASETS:
        try:
            rows, get = getattr(dg, loader)()
        except Exception as exc:                            # noqa: BLE001
            bad.append(f"{name}: could not load ({exc}) — NOT reporting a pass")
            continue
        rng = np.random.default_rng(0)
        sel = [rows[i] for i in rng.permutation(len(rows))[:N]]
        got = {free_boundary_ok(get(fr)[1]) for fr in sel}
        print(f"  {name:12s} free boundary: {sorted(got)}  (expected {expect})")
        if len(got) != 1:
            bad.append(f"{name}: the rule flips inside one dataset {got} — "
                       f"two height conventions in one feature table")
        elif got.pop() is not expect:
            bad.append(f"{name}: expected free={expect}")
    for b in bad:
        print(f"  FAIL: {b}")
    print(f"boundary-rule: {len(bad)} problem(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
