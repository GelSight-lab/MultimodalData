"""Which datasets use a marker gel — COUNTED on their reference frames.

WHY THIS IS MEASURED AND NOT WRITTEN DOWN

The results table has one persistent outlier: FEATS scores far below every
other dataset on both reconstructions. FEATS is also the only marker gel in
the set, and saying so is worth doing — a printed dot lattice displaces gel and
occludes the very surface the photometric solve integrates, which is a
mechanism, not a coincidence.

It is only worth saying if the marker/markerless split is a measurement. Twice
it was not, and twice the wrong answer was believed:

  * FEATS ships a `markered` column. It reads True for every row, including
    gel_5's, whose `gel_variant` is `black_dot`. Second-hand and wrong.
  * A blob detector thresholding at `mean - 2*std` reported ~34 dots on a
    smooth gel_5 reference. On a smooth image that threshold sits inside the
    noise and the detector invents blobs, so it "confirmed" the wrong answer.

The count here comes from `marker_removal.detect_markers` — the detector the
depth path already trusts to decide whether to inpaint, so it is not a fourth
opinion — and its separation is checked rather than assumed
(`scripts/test_gel_type_measured.py`): every markerless dataset must come back
at zero before any number here is quoted anywhere.

    python -m force_recovery.gel_type
"""
from __future__ import annotations

import json

import numpy as np

from .run_episode import OUT_ROOT

OUT = OUT_ROOT / "feature_cache" / "gel_type.json"

# Every dataset the results table scores, in its order.
DATASETS = ("cnc_mini_26", "cnc", "feats", "sparsh", "faf")
N_FRAMES = 300              # scanned per dataset; distinct references are few


def _refs(name: str, n: int):
    """Distinct reference frames of a dataset, with the group they belong to.

    Keyed on the reference's own bytes: several datasets hand out one shared
    reference per capture, so scanning frames and keying on content is both
    cheaper and more honest than trusting a per-frame field.
    """
    from .force_recon_matrix import _rows
    rows, get = _rows(name)
    seen, out = set(), []
    for fr in rows[:n]:
        _img, ref = get(fr)
        key = hash(np.asarray(ref, np.float32).tobytes())
        if key in seen:
            continue
        seen.add(key)
        out.append((str(fr.get("group", "?")), ref))
    return out


def measure(datasets=DATASETS, n: int = N_FRAMES) -> dict:
    from .marker_removal import marker_info

    table = []
    for ds in datasets:
        try:
            refs = _refs(ds, n)
        except Exception as exc:                                # noqa: BLE001
            table.append({"dataset": ds, "available": False,
                          "reason": str(exc)})
            print(f"  {ds}: UNAVAILABLE — {exc}", flush=True)
            continue
        counts = [int(marker_info(r)["n"]) for _g, r in refs]
        c = np.array(counts or [0])
        row = {"dataset": ds, "available": True, "n_refs": len(refs),
               "n_dots_median": int(np.median(c)), "n_dots_max": int(c.max()),
               "n_dots_min": int(c.min())}
        table.append(row)
        print(f"  {ds:14s} {len(refs):3d} references  "
              f"dots median {row['n_dots_median']:4d}  "
              f"min {row['n_dots_min']:4d}  max {row['n_dots_max']:4d}",
              flush=True)
    out = {"detector": "marker_removal.detect_markers",
           "n_frames_scanned": n, "datasets": table}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(f"-> {OUT}")
    return out


def main() -> int:
    """One writer at a time — see `artifact_lock` for why."""
    from .artifact_lock import one_writer
    with one_writer(OUT):
        measure()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
