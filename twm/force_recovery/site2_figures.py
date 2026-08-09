"""Panels for the rebuilt site: LUT vs calibration-free, per dataset.

One figure per dataset, one row per sample, the site's one column convention
(`eval_panel`): raw, signed colour difference, LUT depth, calibration-free
depth, LUT mesh, calibration-free mesh.

Two things this file will not do quietly:

* The calibration-free depth and mesh are drawn RELATIVE, and say so in the
  title. Its scale is not recovered (`calib_free.RETURNS_MILLIMETRES`), and
  the mesh renderer applies a fixed z exaggeration — drawn raw, correct
  geometry renders as a tower. That is not a display preference, it is the
  difference between a figure that shows the shape and one that libels it.
* A dataset whose frames cannot be loaded is REPORTED, not skipped. A missing
  row in a comparison reads as "worse"; a missing dataset reads as "we did
  not have it", and only one of those is true.

    python -m force_recovery.site2_figures [--datasets ...] [--samples 8]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .run_episode import OUT_ROOT

ASSETS = OUT_ROOT / "site2" / "assets"
MANIFEST = OUT_ROOT / "site2" / "figure_manifest.json"

# Every force-labelled dataset, with how its frames are reached. `loader` is a
# name resolved at call time so an unavailable dataset fails loudly here rather
# than at import.
DATASETS = {
    "cnc_mini_26": dict(label="cnc_mini_26 — GelSight Mini, CNC presses, 0–20 N",
                        loader="load_glowtact", gel="markerless"),
    "fota_cnc":    dict(label="FoTa cnc_Mini — GelSight Mini, CNC probes",
                        loader="load_cnc", gel="markerless"),
    "feats":       dict(label="FEATS — marker gel, FEA force labels",
                        loader="load_feats", gel="MARKER"),
    "sparsh":      dict(label="Sparsh / Meta — 10 gel pads",
                        loader="load_sparsh", gel="markerless"),
    "faf":         dict(label="FeelAnyForce — 42 captures",
                        loader="load_faf", gel="markerless"),
}


def _load(name: str):
    from . import debug_gallery as dg
    spec = DATASETS[name]
    fn = getattr(dg, spec["loader"], None)
    if fn is None:
        raise LookupError(f"{name}: debug_gallery has no {spec['loader']}()")
    return fn()


def figure(name: str, n: int = 8, seed: int = 0) -> dict:
    """One dataset's LUT-vs-calibration-free panel. Returns a manifest entry."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from . import calib_free as CF
    from . import eval_panel as EP
    from .debug_gallery import stages

    spec = DATASETS[name]
    try:
        rows, get = _load(name)
    except Exception as exc:                                # noqa: BLE001
        print(f"  {name}: UNAVAILABLE — {exc}", flush=True)
        return {"dataset": name, "label": spec["label"], "available": False,
                "reason": str(exc)}

    rng = np.random.default_rng(seed)
    sel = [rows[i] for i in rng.permutation(len(rows))[:n]]
    fig, ax = plt.subplots(len(sel), 6, figsize=(19, 2.55 * len(sel)))
    ax = np.atleast_2d(ax)
    for i, fr in enumerate(sel):
        img, ref = get(fr)
        lut = stages(img, ref)["depth"]
        cf = CF.reconstruct(img, ref)["depth"]
        cfn = cf / max(float(cf.max()), 1e-12)
        f_n = fr.get("f")
        tag = f"F={f_n:.1f} N  " if f_n is not None else ""
        cells = [
            (np.clip(img, 0, 255).astype(np.uint8), None,
             f"raw  {tag}[{fr.get('group', '?')}]"),
            (EP.diff_rgb(img, ref), None, EP.diff_caption()),
            (lut, "inferno", f"LUT depth  max {lut.max():.2f} mm"),
            (cfn, "inferno", "calibration-free depth (relative)"),
            (EP.mesh(lut), None, "LUT mesh"),
            (EP.mesh(cf, relative=True), None,
             "calibration-free mesh (relative z)"),
        ]
        for a, (d, cm, t) in zip(ax[i], cells):
            a.imshow(d, cmap=cm)
            a.set_title(t, fontsize=8)
            a.axis("off")
    fig.suptitle(f"{spec['label']}  ·  {spec['gel']}  ·  {len(sel)} samples",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    ASSETS.mkdir(parents=True, exist_ok=True)
    p = ASSETS / f"panel_{name}.png"
    fig.savefig(p, dpi=100)
    plt.close(fig)
    print(f"  {name}: {len(sel)} samples -> {p.name} "
          f"({p.stat().st_size/1e6:.1f} MB)", flush=True)
    return {"dataset": name, "label": spec["label"], "gel": spec["gel"],
            "available": True, "n": len(sel), "asset": p.name}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    ap.add_argument("--samples", type=int, default=8)
    args = ap.parse_args()

    out = [figure(d, args.samples) for d in args.datasets]
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(out, indent=1))
    ok = sum(r["available"] for r in out)
    print(f"\n[site2] {ok}/{len(out)} datasets rendered -> {MANIFEST}")
    for r in out:
        if not r["available"]:
            print(f"  MISSING {r['dataset']}: {r['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
