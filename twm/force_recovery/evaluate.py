"""Self-designed evaluation for both force-recovery methods.

There is no force ground truth in the dataset (that is the whole problem),
so each check tests a property that must hold if the method is right and
would break in a specific way if it is wrong:

Method 1 (depth -> Winkler force)
  E1.1 specificity      no-contact rows must read ~0 N; reported as the
                        ratio between median contact force and p95
                        no-contact force (an SNR — near 1 means the
                        estimator can't tell contact from noise)
  E1.2 correlation      Spearman rho against tactile intensity, an
                        independent contact proxy computed from the same
                        images a different way; ~0 would mean noise,
                        ~1 would mean force adds nothing over intensity
  E1.3 spikes           single-frame excursions that vanish the next fresh
                        frame are reconstruction noise, not physics
  E1.4 calibration      per-episode thresholds should agree per sensor;
                        a wild outlier flags a bad reference set

Method 2 (force-informed targets)
  E2.1 invariance       zero force must leave the action exactly the
                        observed pose (free-space behaviour untouched)
  E2.2 boundedness      penetration F/k must stay millimetre-scale;
                        a metre-scale target would be an unsafe action
  E2.3 roundtrip        k * ||target - pose|| must reproduce the input
                        force to machine precision (pure algebra check)
  E2.4 geometry         while pressing, motion should be mostly tangential
                        to the pressing direction (|v.n| / |v| small),
                        which checks the assumed sensor-normal axis against
                        how the demonstrator actually moved
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.stats import spearmanr

from .dexforce import force_informed_targets, gel_axis, roundtrip_force
from .run_episode import OUT_ROOT, STAGE_ROOT

# Rows above this are "in contact" for the SNR split. In FEATS-calibrated
# newtons: ~5x the typical no-contact noise, ~1/6 of a median press.
CONTACT_N = 0.1


def median3_fresh(force: np.ndarray, is_new: np.ndarray) -> np.ndarray:
    """3-tap median over *fresh* samples, re-expanded to all rows.

    Single-frame excursions in the estimate are reconstruction noise (a bad
    Poisson solve), not physics — real contact at these speeds spans several
    fresh frames. Filtering over rows would be wrong: duplicated rows repeat
    the previous estimate, so a row-wise median would see each noisy value
    three times and keep it.
    """
    fresh = np.where(is_new)[0]
    if len(fresh) < 3:
        return force.copy()
    f = force[fresh]
    med = f.copy()
    med[1:-1] = np.median(np.stack([f[:-2], f[1:-1], f[2:]]), axis=0)
    out = np.empty_like(force)
    idx = np.clip(np.searchsorted(fresh, np.arange(len(force)), "right") - 1,
                  0, len(fresh) - 1)
    out[:] = med[idx]
    return out


def _load(task: str, date: str, ep: str, side: str):
    npz = np.load(OUT_ROOT / task / date / f"{ep}_{side}.npz")
    table = pq.read_table(
        str(STAGE_ROOT / task / "meta" / date / f"{ep}.parquet"),
        columns=[f"tactile_{side}_intensity", f"tactile_{side}_is_new",
                 f"sensor_{side}_pose"])
    inten = np.asarray(table[f"tactile_{side}_intensity"].to_numpy())
    is_new = np.asarray(table[f"tactile_{side}_is_new"].to_numpy())
    pose = np.stack(table[f"sensor_{side}_pose"].to_numpy())
    return npz, inten, is_new, pose


def eval_method1(task: str, date: str, ep: str, side: str) -> dict:
    npz, inten, is_new, _ = _load(task, date, ep, side)
    raw = npz["force_normal_n"]
    force = median3_fresh(raw, is_new)
    fresh = np.where(is_new)[0]
    f, i = force[fresh], inten[fresh]
    raw_f = raw[fresh]

    # E1.1 — both sides of the split are defined WITHOUT the force under
    # test: no-contact rows are the estimator's reference rows, contact
    # rows the top intensity quartile (an independent image statistic).
    # Splitting by the force itself would make the SNR tautological.
    nc_rows = npz["reference_rows"]
    contact = i >= np.percentile(i, 75)
    p95_nc = float(np.percentile(force[nc_rows], 95))
    p50_contact = float(np.median(f[contact])) if contact.any() else 0.0

    # E1.2
    rho = float(spearmanr(f, i).statistic)

    # E1.3 — fresh-sample spikes: up then straight back down, before and
    # after the median filter (after should be ~0 if spikes are 1-frame)
    def spike_rate(x, jump=5.0 * CONTACT_N):
        dx = np.diff(x)
        if len(dx) < 2:
            return 0.0
        return float(((dx[:-1] > jump) & (dx[1:] < -jump)).mean())

    return {
        "episode": f"{task}/{date}/{ep}", "side": side,
        "rows": int(len(force)), "fresh": int(len(fresh)),
        "contact_fraction": float((f > CONTACT_N).mean()),
        "force_max_n": float(f.max()),
        "p50_contact_n": p50_contact,
        "p95_nocontact_n": p95_nc,
        "snr": p50_contact / max(p95_nc, 1e-6),
        "spearman_vs_intensity": rho,
        "spike_rate_raw": spike_rate(raw_f),
        "spike_rate_filtered": spike_rate(f),
        "threshold_um": float(npz["contact_threshold_mm"]) * 1000.0,
    }


def eval_method2(task: str, date: str, ep: str, side: str,
                 fps: float = 30.0) -> dict:
    npz, _, is_new, pose = _load(task, date, ep, side)
    force = median3_fresh(npz["force_normal_n"].astype(np.float64), is_new)
    actions = force_informed_targets(pose, force, gel_axis(task, side))

    # E2.1
    free = force <= 0.0
    invariance = float(np.abs(actions.offset[free]).max()) if free.any() else 0.0

    # E2.2
    pen_mm = actions.penetration_m * 1000.0

    # E2.3
    rt = roundtrip_force(actions, pose[:, :3])
    roundtrip_err = float(np.abs(rt - force).max())

    # E2.4 — signed alignment of motion with the calibrated pressing
    # direction. While force builds the mount moves toward the surface, so
    # the mean signed v-hat . n-hat should be higher when pressing than in
    # free motion; with a wrong axis the sign flips (observed for the
    # naive [0,0,1] guess: -0.067 at onsets vs +0.03..0.06 calibrated).
    v = np.diff(pose[:, :3], axis=0) * fps
    speed = np.linalg.norm(v, axis=1)
    n = actions.normal_world[:-1]
    moving = speed > 0.01                       # 1 cm/s
    pressing = (force[:-1] > CONTACT_N) & moving
    align = (v * n).sum(1) / np.maximum(speed, 1e-9)
    return {
        "episode": f"{task}/{date}/{ep}", "side": side,
        "invariance_max_offset_m": invariance,
        "penetration_p50_mm": float(np.median(pen_mm[force > CONTACT_N]))
            if (force > CONTACT_N).any() else 0.0,
        "penetration_max_mm": float(pen_mm.max()),
        "roundtrip_max_err_n": roundtrip_err,
        "align_pressing": float(align[pressing].mean())
            if pressing.any() else float("nan"),
        "align_free": float(align[~pressing & moving].mean())
            if (~pressing & moving).any() else float("nan"),
    }


def episodes_with_results() -> list[tuple[str, str, str, str]]:
    out = []
    for npz in sorted(Path(OUT_ROOT).rglob("episode_*_*.npz")):
        stem = npz.stem                            # episode_000_left
        ep, side = stem.rsplit("_", 1)
        out.append((npz.parent.parent.name, npz.parent.name, ep, side))
    return out


def run_all() -> tuple[list[dict], list[dict]]:
    m1, m2 = [], []
    for task, date, ep, side in episodes_with_results():
        m1.append(eval_method1(task, date, ep, side))
        m2.append(eval_method2(task, date, ep, side))
    return m1, m2
