"""How far does a GelSight Mini's no-contact appearance move when you change
the gel pad, versus when you change the whole unit?

FEATS ships 3000 no-contact frames in a fully crossed design: 5 sensor units x
6 gel pads x 100 repeats, every cell exactly 100 frames. That is the rare case
where the two axes can be separated without confounding, because the same gel
indices appear under every sensor index.

THE STATISTIC IS FIXED, NOT CHOSEN. RMS pixel distance in uint8 between mean
no-contact frames over the sensing region, plus per-channel medians. It is
specified up front precisely so it cannot be tuned after seeing the answer.

WHY THE NOISE FLOOR IS COMPUTED FIRST. A distance between two mean images is
meaningless until you know what distance two means of the SAME cell produce.
The 100 repeats are split in half and the two half-means compared; that is the
floor every other number is read against. Without it "the gel axis is 3.2 RMS"
is a number with no unit of meaning.

WHY HALVES, NOT THE FULL CELL. Comparing a cell to itself gives 0 by
construction. Comparing two disjoint 50-frame halves estimates the sampling
noise of a 50-frame mean, which is the same order as the 100-frame means being
compared elsewhere -- slightly conservative (a 50-frame mean is noisier than a
100-frame mean), and stated as such rather than silently.

    python analysis/feats_appearance_axes.py
"""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/media/yxma/Disk1/yuxiang/mini_data/markered/FEATS/v24_labels_24_32")
OUT = Path(__file__).resolve().parent.parent / "plan" / "feats_appearance_axes.json"

# `<repeat>_nocontact_sensor_<s>_gel_<g>.npy`. The repeat index runs 1..100 and
# every repeat covers all 30 sensor x gel cells, so the design is crossed.
PAT = re.compile(r"^(?P<rep>\d+)_nocontact_sensor_(?P<s>\d+)_gel_(?P<g>\d+)\.npy$")

# The sensing region. GelSight Mini frames here are 240x320 with the gel filling
# the frame, but the outermost pixels carry lens vignetting and sensor-mount
# shadow that vary with how the pad was seated -- a mechanical difference, not
# an illumination one. A 12 px inset is reported alongside the full frame; the
# conclusion has to survive both or it is a statement about the border.
INSET = 12


def load_cells() -> dict[tuple[int, int], np.ndarray]:
    """Stack every no-contact frame per (sensor, gel) cell."""
    buckets: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
    for split in sorted(os.listdir(ROOT)):
        d = ROOT / split
        if not d.is_dir():
            continue
        for fn in sorted(os.listdir(d)):
            m = PAT.match(fn)
            if not m:
                continue
            rec = np.load(d / fn, allow_pickle=True).item()
            buckets[(int(m["s"]), int(m["g"]))].append(rec["gs_img"])
    return {k: np.stack(v).astype(np.float64) for k, v in sorted(buckets.items())}


def rms(a: np.ndarray, b: np.ndarray, inset: int) -> float:
    """RMS pixel distance in uint8 units over the sensing region."""
    if inset:
        a = a[inset:-inset, inset:-inset]
        b = b[inset:-inset, inset:-inset]
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main() -> int:
    cells = load_cells()
    sensors = sorted({s for s, _ in cells})
    gels = sorted({g for _, g in cells})
    counts = {f"s{s}g{g}": int(v.shape[0]) for (s, g), v in cells.items()}
    print(f"cells: {len(cells)}  sensors: {sensors}  gels: {gels}")
    print(f"frames/cell: min={min(counts.values())} max={max(counts.values())}")

    means = {k: v.mean(axis=0) for k, v in cells.items()}
    # Disjoint halves -> the sampling-noise floor.
    half_a = {k: v[: v.shape[0] // 2].mean(axis=0) for k, v in cells.items()}
    half_b = {k: v[v.shape[0] // 2:].mean(axis=0) for k, v in cells.items()}

    res: dict[str, dict] = {}
    for tag, inset in (("full_frame", 0), (f"inset_{INSET}px", INSET)):
        floor = [rms(half_a[k], half_b[k], inset) for k in cells]
        # Same unit, different pad: the gel axis, uncontaminated by unit identity.
        gel = [rms(means[(s, g1)], means[(s, g2)], inset)
               for s in sensors for i, g1 in enumerate(gels) for g2 in gels[i + 1:]]
        # Different unit. Held at the same gel index so the pair differs in unit
        # and nothing else that the filename exposes.
        unit = [rms(means[(s1, g)], means[(s2, g)], inset)
                for g in gels for i, s1 in enumerate(sensors) for s2 in sensors[i + 1:]]

        def stat(x: list[float]) -> dict:
            a = np.asarray(x)
            return {"n": int(a.size), "median": float(np.median(a)),
                    "mean": float(a.mean()), "min": float(a.min()),
                    "max": float(a.max()), "p95": float(np.percentile(a, 95))}

        # Is a gel index the SAME physical pad across units, or does each unit
        # carry its own six pads? The filenames cannot say, and it decides how
        # the unit number may be read: if pads are per-unit, "different unit"
        # also means "different pad" and the unit axis is an upper bound.
        #
        # It is still testable. If gel_g were one pad moved between units, then
        # holding g fixed across units would remove a real source of variation
        # and score BELOW pairs that also differ in g. If the index is only a
        # slot number, the two are the same measurement and must agree.
        unit_diff_g = [rms(means[(s1, g1)], means[(s2, g2)], inset)
                       for i, s1 in enumerate(sensors) for s2 in sensors[i + 1:]
                       for g1 in gels for g2 in gels if g1 != g2]

        res[tag] = {"noise_floor_half_vs_half": stat(floor),
                    "gel_axis_same_unit": stat(gel),
                    "unit_axis_same_gel_index": stat(unit),
                    "unit_axis_diff_gel_index": stat(unit_diff_g)}
        print(f"\n--- {tag} (RMS uint8 over sensing region)")
        for k, v in res[tag].items():
            print(f"  {k:26s} n={v['n']:4d} median={v['median']:7.3f} "
                  f"[{v['min']:.3f}, {v['max']:.3f}]")

    # Per-channel medians of each cell mean, as specified.
    chan = {f"s{s}g{g}": [float(np.median(means[(s, g)][..., c])) for c in range(3)]
            for (s, g) in cells}
    per_sensor = {f"sensor_{s}": [float(np.median([chan[f"s{s}g{g}"][c] for g in gels]))
                                  for c in range(3)] for s in sensors}
    print("\nper-channel median (R,G,B) by unit, median over its 6 gels:")
    for k, v in per_sensor.items():
        print(f"  {k}: R={v[0]:6.2f} G={v[1]:6.2f} B={v[2]:6.2f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(
        {"source": str(ROOT), "frames_per_cell": counts, "sensors": sensors,
         "gels": gels, "statistic": "RMS pixel distance, uint8, mean no-contact frames",
         "results": res, "per_cell_channel_median_rgb": chan,
         "per_sensor_channel_median_rgb": per_sensor}, indent=2))
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
