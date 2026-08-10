"""How many frames each dataset actually holds — counted, not assumed.

This exists because the site published a loader ceiling as if it were a
dataset size. Every loader in `debug_gallery` was written to draw FIGURES and
carries a cap sized for that; when they started feeding the evaluation the
caps were never revisited, so "FEATS has 390 frames" meant "our loader reads
the val split and then stops at 390", and the page said, in so many words,
that a 2,000-frame evaluation was impossible on any of them. It is possible on
three of the five.

Counted here WITHOUT decoding an image or reconstructing anything, so it is
cheap enough to re-run whenever a loader changes:

    python -m force_recovery.dataset_sizes

`n_frames` is what exists on disk after each dataset's own validity rule
(a parseable force label, |F| above the noise floor, an image that resolves).
It is NOT the number of usable presses — roughly a fifth survive the
fully-imaged filter, and that number lives in `force_recon_matrix.json`.
"""
from __future__ import annotations

import json

from .run_episode import OUT_ROOT

OUT = OUT_ROOT / "feature_cache" / "dataset_sizes.json"


def _cnc_mini_26() -> tuple[int, str]:
    from .lut_calibration import CNC_MINI_26, PAT
    n = 0
    for fam in ("round", "quad", "star", "triangle", "B", "quad_small"):
        for p in (CNC_MINI_26 / fam).glob("*.jpg"):
            m = PAT.search(p.name)
            n += bool(m and float(m["f"]) > 0.3)
    return n, "every press on disk with a parseable label and F > 0.3 N"


def _cnc() -> tuple[int, str]:
    import re
    import tarfile

    from .debug_gallery import CNC, _num
    pat = re.compile(r"(Mini_[A-F])\|(\d+)pos\|([-\d]+), ([-\d]+), "
                     r"([-\d]+) f\|([-\d]+)\.jpg$")
    n = 0
    for split in ("train", "val"):
        with tarfile.open(CNC / split / "data-000000.tar") as tf:
            for name in tf.getnames():
                m = pat.search(name)
                n += bool(m and _num(m[6]) > 0.3)
    return n, "both webdataset shards; the archive is not sharded further"


def _feats() -> tuple[int, str]:
    import pyarrow.parquet as pq

    from .debug_gallery import FEATS_PQ, FEATS_SAME_SENSOR
    n = sum(pq.ParquetFile(FEATS_PQ / s).metadata.num_rows
            for s in FEATS_SAME_SENSOR)
    return n, ("train+val+test+unknown_indenters; the two test_diff_sensor "
               "splits (693 rows) are a different sensor and gel and are "
               "excluded on purpose")


def _sparsh() -> tuple[int, str]:
    import numpy as np

    from .sparsh_data import BATCHES, label_table
    n = 0
    for probe, b in BATCHES:
        n += int(np.asarray(label_table(probe, b)["in_contact"]).sum())
    return n, "frames flagged in_contact across all 10 batches"


def _faf() -> tuple[int, str]:
    import collections

    from .faf_extract import IMG_DIR, ZERO_N, load_labels
    by = collections.defaultdict(list)
    for r in load_labels():
        if (IMG_DIR / r["capture"] / f"{r['ts']}.png").exists():
            by[r["capture"]].append(abs(float(r["fz"])))
    a = sum(len(v) for v in by.values() if min(v) < ZERO_N)
    return a, (f"tier-A captures only (a contact-free reference exists); "
               f"{sum(len(v) for v in by.values()) - a} tier-B frames on disk "
               f"are excluded. 58,630 tier-A frames exist in the archive — "
               f"this counts what has been extracted from the 81 GB zip")


SOURCES = {
    "cnc_mini_26": ("GelSight Mini CNC", _cnc_mini_26),
    "cnc": ("FoTa cnc_Mini", _cnc),
    "feats": ("FEATS (marker)", _feats),
    "sparsh": ("Sparsh / Meta", _sparsh),
    "faf": ("FeelAnyForce", _faf),
}


def main() -> int:
    out = []
    for name, (label, fn) in SOURCES.items():
        try:
            n, how = fn()
        except Exception as exc:                               # noqa: BLE001
            print(f"  {name:12s} UNAVAILABLE — {exc}", flush=True)
            out.append({"dataset": name, "label": label, "available": False,
                        "reason": str(exc)})
            continue
        print(f"  {name:12s} {n:8,d}  {how}", flush=True)
        out.append({"dataset": name, "label": label, "available": True,
                    "n_frames": int(n), "counted": how})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
