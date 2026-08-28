"""Render the test set's overlay stills and the four-panel example figure.

Written out as a script because it had been inlined three times, and an
inlined renderer drifts from the package it illustrates — which is the same
failure the clip page had.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from react_paths import testset_root   # noqa: E402

import cv2                                                     # noqa: E402
import numpy as np                                             # noqa: E402

from react_toolbox.calibration import load_calibration          # noqa: E402
from react_toolbox.frames import require_up_axis           # noqa: E402
from react_toolbox.probe_eval import overlay_gt, rollout_error  # noqa: E402
from react_toolbox.viz import draw_world_gizmo                  # noqa: E402

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _hud(img, text, h=16, scale=0.36):
    img = np.ascontiguousarray(img)
    cv2.rectangle(img, (0, 0), (img.shape[1], h + 1), (0, 0, 0), -1)
    cv2.putText(img, text, (4, h - 4), FONT, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(testset_root()))
    args = ap.parse_args()
    root = Path(args.root)
    cal = load_calibration(root)
    require_up_axis(cal, where=f"{root}/calibration")
    man = json.loads((root / "manifest.json").read_text())
    runs = [json.loads((root / p["meta"]).read_text()) for p in man["probes"]]
    (root / "overlays").mkdir(exist_ok=True)
    cam = cal["cams"]["middle"]

    for m in runs:
        img = cv2.imread(str(root / f"probes/run{m['run']}/context/ctx3_view_middle.jpg"))[:, :, ::-1]
        gel_m, gel_o = cal[f"gel_{m['moving_side']}"], cal[f"gel_{m['held_side']}"]
        for q in m["probes"]:
            d = np.load(root / q["file"])
            vis = overlay_gt(img, d["poses"], gel_m, cam, held_pose7=d["held_pose"],
                             held_gel_mm=gel_o, label=m["moving_side"][0].upper())
            vis = draw_world_gizmo(vis, cam, corner="tl", margin=26, title="world (z-up)")
            unit = "m" if q["amplitude_unit"] == "m" else "deg"
            vis = _hud(vis, f"{q['name']}  {q['amplitude']:g}{unit}  "
                            f"{q['horizon_s']:.2f}s  p{q['speed_percentile']:.0f}")
            cv2.imwrite(str(root / "overlays" / f"run{m['run']}_{q['name']}.jpg"),
                        vis[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])

    tiles = []
    for run, name in ((0, "trans+x"), (0, "rot+y"), (3, "trans-z"), (3, "rot-x")):
        m = runs[run]
        d = np.load(root / f"probes/run{run}/{name}.npz")
        img = cv2.imread(str(root / f"probes/run{run}/context/ctx3_view_middle.jpg"))[:, :, ::-1]
        gel_m, gel_o = cal[f"gel_{m['moving_side']}"], cal[f"gel_{m['held_side']}"]
        vis = overlay_gt(img, d["poses"], gel_m, cam, held_pose7=d["held_pose"],
                         held_gel_mm=gel_o, label=m["moving_side"][0].upper())
        bad = d["poses"].copy()
        bad[:, 0] += 0.025                       # a deliberately wrong rollout
        vis = overlay_gt(vis, bad, gel_m, cam, color=(255, 90, 90))
        vis = draw_world_gizmo(vis, cam, corner="tl", margin=26, title="world (z-up)")
        e = rollout_error(bad, d["poses"], gel_m, cam)
        q = next(p for p in m["probes"] if p["name"] == name)
        unit = "m" if q["amplitude_unit"] == "m" else "deg"
        tiles.append(_hud(vis, f"run{run} {name}  {q['amplitude']:g}{unit}  "
                               f"p{q['speed_percentile']:.0f}   yellow=GT  "
                               f"red=+25mm rollout ({e['px_final']:.0f}px)",
                          h=17, scale=0.34)[:, :, ::-1])
    cv2.imwrite(str(root / "overlay_example.jpg"),
                np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])]),
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    n = sum(len(m["probes"]) for m in runs)
    print(f"{n} overlays + overlay_example.jpg -> {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
