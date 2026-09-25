"""Producers for the two feature caches the results table reads.

`feature_cache/lut_full.json` and `site_assets/debug_gallery/features_cnc_full.json`
are what `force_eval_all` scores, and NEITHER had a producer in the repository.
They were files on a disk that no code could regenerate — the same shape of
problem as the figure that shipped for weeks with a caption the code had
stopped applying. When the reconstruction changed, `force_eval_all`'s own
spotcheck refused to run (`stages() changed: max feature delta 90.9`) and there
was nothing to re-run.

The schemas are dictated by their consumers and are mirrored here exactly:

  lut_full.json         one row per cnc_mini_26 press
                        fam, x, y, z, f, vol, vol2, vol15, area, maxd, cx, cy
                        area in PIXELS (ds_cnc_mini_26 compares sqrt(area/pi)
                        against the 24/296 px frame margins), depth floored at
                        an absolute 0.05 mm — the LUT's own convention.
  features_cnc_full.json  one row per FoTa cnc_Mini frame
                        group, f, x, y, z + vol, vol2, maxd, area, h1 from
                        stages()['feats'], area in mm^2 there.

    python -m force_recovery.feature_caches lut     # ~6.2k presses
    python -m force_recovery.feature_caches cnc     # ~3.4k frames
    python -m force_recovery.feature_caches all
"""
from __future__ import annotations

import json
import sys

import numpy as np

from .run_episode import OUT_ROOT

CACHE = OUT_ROOT / "feature_cache"
DG = OUT_ROOT / "site_assets" / "debug_gallery"
DEPTH_FLOOR_MM = 0.05
FAMILIES = ("round", "quad", "star", "triangle", "B", "quad_small")


def _lut_feats(depth: np.ndarray, dI: np.ndarray) -> dict:
    """The LUT path's feature block, identical to calibfree_full's."""
    from .lut_calibration import MM_PER_PIXEL, detect_circle
    m = depth > DEPTH_FLOOR_MM
    px = MM_PER_PIXEL ** 2
    d = depth
    out = {"vol": float(d[m].sum() * px), "vol2": float((d[m] ** 2).sum() * px),
           "vol15": float((d[m] ** 1.5).sum() * px),
           "area": float(m.sum()), "maxd": float(np.percentile(d, 99.8))}
    det = detect_circle(dI)
    out["cx"], out["cy"] = ((float(det[0]), float(det[1])) if det
                            else (float("nan"), float("nan")))
    return out


def build_lut_full(out=None) -> int:
    from PIL import Image

    from .debug_gallery import stages
    from .lut_calibration import CNC_MINI_26, PAT, crop

    out = out or (CACHE / "lut_full.json")
    rows, refs = [], {}
    jobs = []
    for fam in FAMILIES:
        d = CNC_MINI_26 / fam
        if not d.is_dir():
            continue
        refs[fam] = crop(np.asarray(
            Image.open(d / "initial.jpg").convert("RGB"))).astype(np.float32)
        for p in sorted(d.glob("*.jpg")):
            m = PAT.search(p.name)
            if m:
                jobs.append((fam, p, float(m["x"]), float(m["y"]),
                             -float(m["z"]), float(m["f"])))
    print(f"[lut_full] {len(jobs)} presses over {len(refs)} families",
          flush=True)
    for i, (fam, p, x, y, z, f) in enumerate(jobs):
        img = crop(np.asarray(Image.open(p).convert("RGB"))).astype(np.float32)
        st = stages(img, refs[fam])
        rows.append({"fam": fam, "x": x, "y": y, "z": z, "f": f,
                     **_lut_feats(st["depth"], st["dI"])})
        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{len(jobs)}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows))
    print(f"-> {out}  ({len(rows)} rows)")
    return 0


def build_cnc_full(out=None) -> int:
    import os

    os.environ["CNC_N"] = "100000"          # every frame, not a subsample
    from .debug_gallery import load_cnc, stages

    out = out or (DG / "features_cnc_full.json")
    frames, get = load_cnc()
    print(f"[features_cnc_full] {len(frames)} frames", flush=True)
    rows = []
    for i, fr in enumerate(frames):
        img, ref = get(fr)
        rows.append({"group": fr["group"], "f": fr["f"], "x": fr["x"],
                     "y": fr["y"], "z": fr["z"], **stages(img, ref)["feats"]})
        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{len(frames)}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows))
    print(f"-> {out}  ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    rc = 0
    if cmd in ("lut", "all"):
        rc |= build_lut_full()
    if cmd in ("cnc", "all"):
        rc |= build_cnc_full()
    raise SystemExit(rc)
