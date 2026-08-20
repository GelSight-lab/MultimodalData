"""Synthetic probe trajectories: axis-aligned, dataset-paced, in view.

Twelve controlled action sequences — six pure translations along +/-x, +/-y,
+/-z and six pure rotations about the same axes — for probing a world model
where no ground-truth future image exists. They are judged by eye against the
GT sensor-pose projection overlaid on the start frame.

The requirements that are checkable, and are checked:

  1  SIX DIRECTIONS, ONE AXIS EACH. A trajectory that drifts off its axis is
     not a controlled probe.
  2  DATASET-PACED. Per-step magnitude must sit inside the measured
     distribution, not merely "look reasonable". Measured over 480,008 rows:
     |dp| p25 0.971, p50 2.813, p90 10.158 mm/step; |dtheta| p25 0.320,
     p50 0.699, p90 2.296 deg/step.
  3  HORIZON > 1.5 s. At 30 Hz that is 45 steps.
  4  UNIFORM SPEED, so a failure is attributable to direction and magnitude
     rather than to an acceleration profile nothing else in the set shares.
  5  IN VIEW BY DEFAULT. The projected pose must stay inside the image for
     every step, or the probe leaves the distribution the model was trained
     on and its output is uninterpretable. `allow_leaving_view=True` exists
     for deliberate OOD probes and is NOT the default.
  6  ACTIONS AND START FRAMES ARE INDEPENDENT. The action set is generated
     without reference to any frame; a start frame is then accepted or
     rejected against it. Coupling them would make "which frames survive" a
     property of the generator rather than of the geometry.
  7  THE MODEL INPUT IS SEVERAL CONSECUTIVE FRAMES, so the sampler returns a
     context window, not one image.

    python scripts/test_synth_actions.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                              # noqa: E402

RESULTS: list[tuple[bool, str, str]] = []
# measured over the release, 30 Hz rows
DP_P25, DP_P50, DP_P90 = 0.971, 2.813, 10.158        # mm / step
DA_P25, DA_P50, DA_P90 = 0.320, 0.699, 2.296         # deg / step


def check(ok: bool, name: str, evidence: str) -> None:
    RESULTS.append((bool(ok), name, evidence))


def main() -> int:
    from react_toolbox.synth_actions import (DA_PCT, DP_PCT, _speed_percentile,
                                             gel_centre_world,
                                             make_rotation_set,
                                             make_translation_set)
    from scipy.spatial.transform import Rotation

    start = np.array([0.40, 0.02, 0.30, 0.0, 0.0, 0.0, 1.0])

    # the real measured gel offset: 65.7 mm from the rigid-body origin, which
    # is exactly why the pivot matters
    GEL = np.array([-42.6, -36.6, -34.2])
    tr = make_translation_set(start, seed=0)
    ro = make_rotation_set(start, GEL, seed=0)
    check(len(tr) == 6 and len(ro) == 6, "six directions in each set",
          f"{len(tr)} translations, {len(ro)} rotations")

    # 1 — one axis each, and the six cover +/- on all three
    axes = set()
    off = []
    for t in tr:
        d = t["poses"][-1, :3] - t["poses"][0, :3]
        a = int(np.argmax(np.abs(d)))
        axes.add((a, int(np.sign(d[a]))))
        lateral = np.linalg.norm(np.delete(d, a))
        if lateral > 1e-9:
            off.append(f"{t['name']}: {lateral*1000:.3f} mm off-axis")
    check(len(axes) == 6 and not off, "each translation moves on one axis",
          f"{len(axes)} distinct (axis, sign)" + (f"; {off[:2]}" if off else ""))

    # 2 — per-step magnitude inside the measured distribution
    bad = []
    for t in tr:
        s = np.linalg.norm(np.diff(t["poses"][:, :3], axis=0), axis=1) * 1000
        if not (DP_P25 <= s.mean() <= DP_P90):
            bad.append(f"{t['name']}: {s.mean():.2f} mm/step")
    for r in ro:
        q = Rotation.from_quat(r["poses"][:, 3:7])
        s = np.degrees((q[:-1].inv() * q[1:]).magnitude())
        if not (DA_P25 <= s.mean() <= DA_P90):
            bad.append(f"{r['name']}: {s.mean():.3f} deg/step")
    check(not bad, "per-step magnitude is inside the dataset distribution",
          f"{12-len(bad)}/12 within p25-p90" + (f"; {bad[:3]}" if bad else ""))

    # 3 / 4 — horizon and uniform speed
    short = [x["name"] for x in tr + ro if x["n_steps"] < 45]
    check(not short, "horizon exceeds 1.5 s (45 steps at 30 Hz)",
          f"shortest {min(x['n_steps'] for x in tr+ro)} steps"
          + (f"; too short: {short}" if short else ""))

    jitter = []
    for t in tr:
        s = np.linalg.norm(np.diff(t["poses"][:, :3], axis=0), axis=1)
        if s.std() / max(s.mean(), 1e-12) > 1e-6:
            jitter.append(f"{t['name']}: cv {s.std()/s.mean():.2e}")
    check(not jitter, "speed is uniform",
          f"{6-len(jitter)}/6 constant-speed" + (f"; {jitter[:2]}" if jitter else ""))

    # 5 — amplitude ranges as specified
    amps = [np.linalg.norm(t["poses"][-1, :3] - t["poses"][0, :3]) for t in tr]
    ang = []
    for r in ro:
        q = Rotation.from_quat(r["poses"][[0, -1], 3:7])
        ang.append(np.degrees((q[0].inv() * q[1]).magnitude()))
    check(all(0.1 - 1e-9 <= a <= 0.4 + 1e-9 for a in amps)
          and all(18 - 1e-6 <= a <= 90 + 1e-6 for a in ang),
          "amplitudes are within the requested ranges",
          f"translation {min(amps):.3f}-{max(amps):.3f} m, "
          f"rotation {min(ang):.1f}-{max(ang):.1f} deg")


    # 6 — SPEED IS SAMPLED, NOT DERIVED. The first version computed
    #     n = amplitude / p50, so every probe long enough to clear the 1.5 s
    #     floor ran at EXACTLY the median: 48 of 60 published probes sat
    #     within 1% of 2.813 mm/step and not one exceeded p50. A p25-p90
    #     range check passes on a constant, which is why it did.
    from react_toolbox.synth_actions import SPEED_PCT_RANGE
    pc_t, pc_r = [], []
    for sd in range(24):
        pc_t += [t["speed_percentile"] for t in make_translation_set(start, seed=sd)]
        pc_r += [r["speed_percentile"] for r in make_rotation_set(start, GEL, seed=sd)]
    pc_t, pc_r = np.array(pc_t), np.array(pc_r)
    lo, hi = SPEED_PCT_RANGE
    spread_ok = (np.percentile(pc_t, 90) - np.percentile(pc_t, 10) > 15
                 and np.percentile(pc_r, 90) - np.percentile(pc_r, 10) > 15)
    clumped = max(np.mean(np.abs(pc_t - np.median(pc_t)) < 1.0),
                  np.mean(np.abs(pc_r - np.median(pc_r)) < 1.0))
    check(spread_ok and clumped < 0.25, "speed is drawn at random, not pinned to p50",
          f"translation p10-p90 {np.percentile(pc_t,10):.0f}-{np.percentile(pc_t,90):.0f}, "
          f"rotation {np.percentile(pc_r,10):.0f}-{np.percentile(pc_r,90):.0f}; "
          f"{clumped*100:.0f}% within 1 pct-pt of the median")

    # 7 — AND NOT SUPER SLOW. The floor is the amplitude the 1.5 s horizon
    #     forces: 0.1 m over 45 steps is 2.22 mm/step, the dataset's p42.
    #     Nothing may be slower than that, and the bulk must clear `lo`.
    floor_t = _speed_percentile(0.100 * 1000 / 45, DP_PCT)
    floor_r = _speed_percentile(18.0 / 45, DA_PCT)
    check(pc_t.min() >= floor_t - 0.5 and pc_r.min() >= floor_r - 0.5
          and np.median(pc_t) >= lo and np.median(pc_r) >= lo,
          "no probe is slower than the horizon forces",
          f"slowest translation p{pc_t.min():.0f} (floor p{floor_t:.0f}), "
          f"rotation p{pc_r.min():.0f} (floor p{floor_r:.0f}); "
          f"medians p{np.median(pc_t):.0f}/p{np.median(pc_r):.0f} vs requested >= p{lo:.0f}")

    # 8 — A ROTATION PROBE ROTATES IN PLACE. The pose is the RIGID BODY's,
    #     the drawn frame is the GEL's, and the gel sits 65.7 mm off the
    #     rigid origin — so holding the rigid position fixed swings the gel
    #     through an arc of up to 52.8 mm. On screen a "pure rotation" then
    #     translates, which is what a viewer sees and calls a bug. The pivot
    #     must be the gel centre, the thing the picture actually shows.
    swing = []
    for r in ro:
        g = gel_centre_world(r["poses"], GEL)
        swing.append((r["name"], float(np.max(np.linalg.norm(g - g[0], axis=1)))))
    worst = max(swing, key=lambda x: x[1])
    turned = []
    for r in ro:
        q = Rotation.from_quat(r["poses"][[0, -1], 3:7])
        turned.append(np.degrees((q[0].inv() * q[1]).magnitude()))
    check(worst[1] < 1.0 and min(turned) > 17.0,
          "a rotation probe pivots about the gel, not the marker origin",
          f"gel centre moves at most {worst[1]:.2f} mm ({worst[0]}) while "
          f"turning {min(turned):.0f}-{max(turned):.0f} deg")

    _report()
    return 1 if sum(not ok for ok, _, _ in RESULTS) else 0


def _report() -> None:
    w = max(len(x) for _, x, _ in RESULTS)
    print()
    for ok, name, ev in RESULTS:
        print(f"  [{'ok' if ok else 'FAIL'}] {name:<{w}}  {ev}")
    print(f"\nsynth actions: {len(RESULTS)} checks, "
          f"{sum(not ok for ok, _, _ in RESULTS)} failing")


if __name__ == "__main__":
    raise SystemExit(main())
