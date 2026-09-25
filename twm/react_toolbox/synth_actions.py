"""Synthetic probe trajectories for a world model, paced by the real data.

WHY THESE EXIST

Twelve controlled action sequences — six pure translations along +/-x, +/-y,
+/-z and six pure rotations about the same axes. There is no ground-truth
future image for any of them, so they are judged two ways: by eye, and against
the GT sensor-pose projection drawn on the start frame. A model that moves the
hand the wrong way, or the right way by the wrong amount, is visible against
that overlay in a way it never is against a scalar loss.

Axis-aligned on purpose. When a probe fails you want to be able to say WHICH
direction failed; a random direction gives you a number and no handle.

PACED BY THE DATASET, AND THE PACING IS A MEASUREMENT

"Roughly like the dataset" is not checkable, so the distribution is stated.
Over 480,008 published rows at 30 Hz, per step:

    |dp|      p25 0.971   p50 2.813   p90 10.158  mm
    |dtheta|  p25 0.320   p50 0.699   p90  2.296  deg

SPEED IS DRAWN, AND THAT IS A FIX

The first version set `n = amplitude / p50`, so the speed was not chosen at
all — it was whatever the amplitude implied. Every probe long enough to clear
the 1.5 s floor therefore ran at EXACTLY the median: of the 60 published
probes, 48 sat within 1% of 2.813 mm/step and not one was faster than p50.
The test that was supposed to cover this asserted `p25 <= v <= p90`, which a
constant passes.

Now `_pace` draws the speed uniformly in PERCENTILE over SPEED_PCT_RANGE
(p45-p85). Uniform in percentile rather than in mm/step because the
distribution spans a decade between p25 and p90, and a uniform draw in value
would put most probes in a tail the dataset barely occupies.

THE CONSTRAINT THAT DOES NOT FIT, AND WHAT GIVES

A 1.5 s horizon caps the speed at `amplitude / 45`. At the small end of the
requested amplitudes that bites: 0.1 m over 45 steps is 2.22 mm/step (p44)
and 18 degrees is 0.40 deg/step (p32). Those probes cannot be faster without
breaking the horizon, so they are the slowest in the set, and the draw is
over [p45, min(p85, feasible)].

The horizon wins over the speed band because it is the harder requirement — a
model asked to roll out only 35 steps is being asked an easier question. The
amplitude range wins too, because it was specified explicitly; sampling the
speed FIRST and letting it dictate the amplitude would have made 0.1 m
unreachable. Every trajectory records the percentile its speed actually lands
on, so "does this look like the dataset" is a number you can read rather than
a claim in a docstring.

SPEED IS UNIFORM WITHIN A TRAJECTORY

Constant velocity from start to end — not because that is realistic, it is
not, but because a failure should be attributable to direction and magnitude.
An acceleration profile shared by nothing else in the set would be a third
variable. This is unrelated to the paragraph above: the speed varies BETWEEN
probes and is constant WITHIN one.

WHICH POINT A ROTATION TURNS ABOUT

The gel centre, not the marker cluster. See `make_rotation_set` — getting
this wrong made every "pure rotation" clip translate across the screen.
"""
from __future__ import annotations

import numpy as np

FPS = 30.0
MIN_HORIZON_S = 1.5
MIN_STEPS = int(np.ceil(MIN_HORIZON_S * FPS))          # 45

# Measured over the published release, 30 Hz rows, both sensors, both tasks.
# Stated here so a caller can see what "dataset-paced" was checked against;
# `scripts/test_synth_actions.py` asserts the generated speeds land inside it.
DP_PCT = {25: 0.971, 50: 2.813, 90: 10.158}            # mm / step
DA_PCT = {25: 0.320, 50: 0.699, 90: 2.296}             # deg / step

AXES = ((0, +1, "+x"), (0, -1, "-x"),
        (1, +1, "+y"), (1, -1, "-y"),
        (2, +1, "+z"), (2, -1, "-z"))

# Each gel carries an exclusion circle of this DIAMETER, so the two centres
# must stay at least this far apart at every step. Measured over 194,445
# published rows the centres sit 0.225 m apart at the median and closer than
# 0.12 m on 1.67% of frames — so this forbids the physically impossible and
# rejects a little, rather than rejecting most of the data.
COLLISION_DIAMETER_M = 0.12

# The band a probe's speed is drawn from, in percentiles of the measured
# per-step distribution. The floor is p45 rather than p25 because a probe that
# crawls tests nothing a model finds hard; the ceiling is p85 because the 1.5 s
# horizon and the 0.4 m amplitude cap between them make p86 the fastest
# trajectory this set can contain (0.40 m / 45 steps = 8.89 mm/step).
SPEED_PCT_RANGE = (45.0, 85.0)

TRANSLATION_M = (0.10, 0.40)          # requested amplitude range
ROTATION_DEG = (18.0, 90.0)           # pi/10 .. pi/2


def _speed_percentile(v: float, table: dict) -> float:
    """Where `v` sits in the measured distribution, by log interpolation.

    Log, because the distribution spans a decade between p25 and p90 and a
    linear reading would call almost everything "below p50".
    """
    ks = sorted(table)
    xs = np.log([table[k] for k in ks])
    return float(np.interp(np.log(max(v, 1e-9)), xs, ks))


def _speed_at_percentile(pct: float, table: dict) -> float:
    """Inverse of `_speed_percentile`: the per-step magnitude at `pct`."""
    ks = sorted(table)
    return float(np.exp(np.interp(pct, ks, np.log([table[k] for k in ks]))))


def _pace(rng, amplitude: float, table: dict) -> tuple[int, float, float]:
    """Steps, per-step magnitude and its percentile, at a RANDOM speed.

    The speed used to be `amplitude / p50`, which is not a speed choice at
    all: every probe long enough to clear the 1.5 s floor came out at exactly
    the median. 48 of the 60 published probes sat within 1% of 2.813 mm/step
    and none was faster than p50. The `p25 <= v <= p90` test passed the whole
    time, because a range check cannot see a constant.

    So the speed is drawn, uniformly in PERCENTILE (not in value): the
    distribution spans a decade between p25 and p90, and drawing uniformly in
    mm/step would put most samples in the fast tail where the dataset has
    almost no mass.

    THE HORIZON CAPS THE SPEED, and that is not negotiable. Covering
    `amplitude` in at least MIN_STEPS bounds v <= amplitude / MIN_STEPS: at
    the smallest requested amplitude (0.1 m, 18 deg) the fastest legal probe
    is 2.22 mm/step and 0.40 deg/step — the dataset's p44 and p32. So the
    draw is over [lo, min(hi, feasible)], and where that interval is empty the
    fastest feasible speed is used. The alternative — sampling the speed
    first and letting it dictate the amplitude — would have made 0.1 m
    unreachable, and the amplitude range was requested explicitly.

    Returns the REALISED numbers, after the integer-step rounding, so the
    recorded percentile is the one the trajectory actually has.
    """
    lo, hi = SPEED_PCT_RANGE
    feasible = _speed_percentile(amplitude / MIN_STEPS, table)
    hi = min(hi, feasible)
    lo = min(lo, hi)
    v = _speed_at_percentile(float(rng.uniform(lo, hi)), table)
    n = int(max(MIN_STEPS, round(amplitude / v)))
    step = amplitude / n
    return n, step, _speed_percentile(step, table)


def make_translation_set(start_pose7, seed: int = 0,
                         amp_range=TRANSLATION_M) -> list[dict]:
    """Six pure translations, one per signed axis. Orientation held fixed."""
    rng = np.random.default_rng(seed)
    p0 = np.asarray(start_pose7, float)
    out = []
    for axis, sign, name in AXES:
        amp = float(rng.uniform(*amp_range))
        n, step_mm, pct = _pace(rng, amp * 1000.0, DP_PCT)
        step = step_mm / 1000.0
        poses = np.repeat(p0[None, :], n + 1, axis=0)
        poses[:, axis] = p0[axis] + sign * step * np.arange(n + 1)
        out.append({
            "name": f"trans{name}",
            "kind": "translation", "axis": name,
            "poses": poses,                       # (n+1, 7) absolute, world
            "n_steps": n,
            "horizon_s": n / FPS,
            "amplitude_m": amp,
            "per_step_mm": step_mm,
            "speed_percentile": pct,
        })
    return out


def make_rotation_set(start_pose7, gel_center_mm, seed: int = 0,
                      amp_range=ROTATION_DEG) -> list[dict]:
    """Six pure rotations about the signed world axes, PIVOTING ON THE GEL.

    About WORLD axes, not the sensor's own: the probe is "turn the hand this
    way in the room", which is what a viewer can judge from a camera image.
    A body-frame rotation would be correct too but its meaning on screen
    changes with the starting orientation, so a failure would not be
    comparable across start frames.

    WHICH POINT STAYS STILL — the bug this argument exists to prevent.

    A 7-vec pose is the RIGID BODY's: the OptiTrack marker cluster. The gel
    centre sits 65.7 mm away from it (`gel_center_in_rigid_mm`), and the gel
    is what every picture in this project draws — the triad, the force disc,
    the collision circle, the projection fingerprint. Holding the rigid
    POSITION fixed and turning the quaternion therefore swings the gel
    through an arc: measured 30.7 to 52.8 mm across the six probes, plainly
    visible as translation in a clip labelled "pure rotation".

    That was not a rendering fault — the renderer drew the gel exactly where
    the poses put it. The trajectory was rotating about the wrong point.

    So the gel centre is what is held fixed, and the rigid position is solved
    for: `p_i = c_world - R_i @ gel_center_in_rigid`. `gel_center_mm` is
    REQUIRED, not defaulted to the origin, because a caller that omits it is
    not choosing "pivot on the marker" — it is unaware there is a choice.
    """
    from scipy.spatial.transform import Rotation

    rng = np.random.default_rng(seed + 1000)
    p0 = np.asarray(start_pose7, float)
    q0 = Rotation.from_quat(p0[3:7])
    c = np.asarray(gel_center_mm, float)
    pivot = p0[:3] * 1000.0 + q0.as_matrix() @ c      # gel centre, world mm
    out = []
    for axis, sign, name in AXES:
        amp = float(rng.uniform(*amp_range))
        n, step, pct = _pace(rng, amp, DA_PCT)
        poses = np.repeat(p0[None, :], n + 1, axis=0)
        vec = np.zeros(3)
        vec[axis] = sign
        for i in range(n + 1):
            dq = Rotation.from_rotvec(np.radians(step * i) * vec)
            qi = dq * q0                            # world-frame pre-multiply
            poses[i, 3:7] = qi.as_quat()
            # ...and put the rigid origin wherever it must be for the GEL to
            # stay on `pivot`. This is the whole fix.
            poses[i, :3] = (pivot - qi.as_matrix() @ c) / 1000.0
        out.append({
            "name": f"rot{name}",
            "kind": "rotation", "axis": name,
            "poses": poses,
            "n_steps": n,
            "horizon_s": n / FPS,
            "amplitude_deg": amp,
            "pivot_world_mm": pivot,
            "per_step_deg": step,
            "speed_percentile": pct,
        })
    return out


def make_probe_sets(start_pose7, gel_center_mm, seed: int = 0) -> list[dict]:
    """Both sets, twelve trajectories, in one call.

    `gel_center_mm` is the moving hand's — the rotations pivot on it, see
    `make_rotation_set`.
    """
    return make_translation_set(start_pose7, seed) + \
        make_rotation_set(start_pose7, gel_center_mm, seed)


# ── sampling a start, and keeping the probe in view ─────────────────────────

def poses_in_view(poses7, gel_center_mm, cams, margin_px: float = 0.0):
    """Per-step, per-camera: is the projected gel centre inside the image?

    Returns (all_in_view, per_camera_fraction). The margin lets a caller
    demand the marker stay clear of the border, where a viewer cannot judge
    direction because half the neighbourhood is missing.
    """
    from .calibration import project_gel_to_pixel

    frac = {}
    for name, cam in cams.items():
        K = cam["intrinsics"]
        w, h = K.get("width", 640), K.get("height", 480)
        ok = 0
        for q in np.asarray(poses7, float):
            uv = project_gel_to_pixel(q, gel_center_mm, cam)
            if uv is None:
                continue
            if (margin_px <= uv[0] < w - margin_px
                    and margin_px <= uv[1] < h - margin_px):
                ok += 1
        frac[name] = ok / max(len(poses7), 1)
    return (all(v == 1.0 for v in frac.values()) if frac else False), frac


def gel_centre_world(pose7, gel_center_mm):
    """Gel centre in world millimetres, for one pose or many."""
    from scipy.spatial.transform import Rotation

    p = np.atleast_2d(np.asarray(pose7, float))
    R = Rotation.from_quat(p[:, 3:7]).as_matrix()
    out = p[:, :3] * 1000.0 + np.einsum("nij,j->ni", R, np.asarray(gel_center_mm, float))
    return out[0] if np.ndim(pose7) == 1 else out


def min_separation_m(moving_poses, moving_gel, static_pose, static_gel):
    """Closest approach between the two gel centres over a trajectory."""
    a = gel_centre_world(moving_poses, moving_gel)
    b = gel_centre_world(static_pose, static_gel)
    return float(np.min(np.linalg.norm(a - b, axis=1)) / 1000.0)


def sample_probe(poses_by_side, cal, seed: int = 0, context: int = 4,
                 view: str = "middle", allow_leaving_view: bool = False,
                 margin_px: float = 8.0, max_tries: int = 200,
                 collision_m: float = COLLISION_DIAMETER_M,
                 moving_side: str | None = None):
    """Pick a context window, a moving hand, and twelve probes for it.

    ONE HAND MOVES. The other holds its pose for the whole horizon. Every
    probe in the first version moved the LEFT sensor, so half the rig was
    never exercised and a left-specific defect would have been invisible;
    `moving_side` is drawn at random unless pinned.

    THE MODEL INPUT IS SEVERAL FRAMES, so this returns `context_rows` — the
    consecutive rows a TWM conditions on — not a single index.

    ACTIONS AND START FRAMES ARE INDEPENDENT. The twelve trajectories come
    from the moving hand's start pose alone; a frame is accepted or rejected
    against them, never adjusted to fit. Nudging a trajectory to keep it on
    screen would make two probes named `+x` mean different things.

    TWO REJECTION RULES, both defaults:
      * the projected gel must stay in view — a marker that leaves the image
        asks about a configuration the training data never contains, and the
        answer is unreadable rather than wrong. `allow_leaving_view=True` is
        for deliberate OOD probing.
      * the two gel centres must stay `collision_m` apart — hands cannot pass
        through each other, and a probe that says they do is not a test of
        the model, it is a test of whether it will hallucinate.

    `poses_by_side` is {"left": (T,7), "right": (T,7)}; `cal` is a
    `load_calibration` result, which carries both gel centres and the cameras.
    """
    rng = np.random.default_rng(seed)
    P = {s: np.asarray(poses_by_side[s], float) for s in ("left", "right")}
    n = min(len(P["left"]), len(P["right"]))
    valid = np.ones(n, bool)
    for s in ("left", "right"):
        q = P[s][:n]
        valid &= np.isfinite(q).all(1) & (np.linalg.norm(q[:, 3:], axis=1) > 0.5)
    usable = np.flatnonzero(
        np.convolve(valid.astype(int), np.ones(context, int), "valid") == context)
    if len(usable) == 0:
        raise ValueError(f"no run of {context} rows with BOTH sensors tracked")

    rejected = {"view": 0, "collision": 0}
    for _ in range(max_tries):
        side = moving_side or ("left" if rng.random() < 0.5 else "right")
        other = "right" if side == "left" else "left"
        s0 = int(rng.choice(usable))
        rows = np.arange(s0, s0 + context)
        start = P[side][rows[-1]]
        held = P[other][rows[-1]]
        probes = make_probe_sets(start, cal[f"gel_{side}"],
                                 seed=int(rng.integers(1 << 30)))

        ok_all = True
        for p in probes:
            sep = min_separation_m(p["poses"], cal[f"gel_{side}"],
                                   held, cal[f"gel_{other}"])
            p["min_separation_m"] = sep
            in_view, frac = poses_in_view(p["poses"], cal[f"gel_{side}"],
                                          {view: cal["cams"][view]}, margin_px)
            p["in_view"], p["view_fraction"] = in_view, frac
            p["moving_side"], p["held_side"] = side, other
            if sep < collision_m:
                p["reject"] = "collision"
                ok_all = False
            elif not in_view and not allow_leaving_view:
                p["reject"] = "leaves view"
                ok_all = False
            else:
                p["reject"] = None
        if ok_all or (allow_leaving_view
                      and all(p["reject"] != "collision" for p in probes)):
            return {"context_rows": rows, "moving_side": side,
                    "held_side": other, "start_pose": start,
                    "held_pose": held, "probes": probes,
                    "in_view_enforced": not allow_leaving_view,
                    "collision_m": collision_m}
        for p in probes:
            if p["reject"] == "collision":
                rejected["collision"] += 1
            elif p["reject"]:
                rejected["view"] += 1

    raise ValueError(
        f"no start frame satisfies all 12 probes after {max_tries} tries "
        f"(rejections: {rejected['view']} for leaving the {view} view, "
        f"{rejected['collision']} for coming within {collision_m} m of the "
        f"other hand). Either shrink amp_range, or pass "
        f"allow_leaving_view=True to probe out of distribution deliberately "
        f"— the collision rule has no opt-out, because hands do not pass "
        f"through each other.")
