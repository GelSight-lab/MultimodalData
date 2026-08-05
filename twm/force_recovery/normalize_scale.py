"""Re-derive force from stored volume with the canonical calibration.

A mid-batch validation run briefly overwrote the calibration file, so npz
written in that window carry a different N/mm^3 scale. Volume is stored
alongside force, so the fix is exact arithmetic, not reprocessing:
``force = volume * canonical_scale`` for every npz, stamping the scale used.

Run after every batch:  python -m force_recovery.normalize_scale
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .run_episode import OUT_ROOT


def main() -> int:
    canon = json.loads(
        (OUT_ROOT / "scale_calibration.json").read_text())
    assert canon["split"] == "val", canon["split"]
    scale = float(canon["scale_n_per_mm3"])

    fixed = ok = 0
    for p in sorted(Path(OUT_ROOT).rglob("episode_*_*.npz")):
        with np.load(p) as z:
            data = dict(z)
        old = float(data.get("scale_n_per_mm3", 0.0))
        if abs(old - scale) < 1e-9:
            ok += 1
            continue
        data["force_normal_n"] = (data["volume_mm3"] * scale).astype(np.float32)
        data["scale_n_per_mm3"] = np.float64(scale)
        np.savez_compressed(p, **data)
        fixed += 1
        print(f"fixed {p.name}: scale {old:.3f} -> {scale:.3f}")
    print(f"[normalize_scale] {fixed} fixed, {ok} already canonical")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
