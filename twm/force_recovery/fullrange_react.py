"""React deployment sample for the 0-15 N model search. No force ground truth.

Samples frames from real React recordings (all four tasks), recomputes the
production v8 force for each exactly as `run_episode.process_side` does, and
caches the 759-d observation features, so any candidate from
`fullrange_search` can be compared with v8 on the deployment domain without
re-reading H5 files.

Nothing here is an accuracy metric: React has no force labels. It answers
"does the candidate stay in-distribution and agree with v8 where v8 is
trustworthy", which is what decides whether a benchmark win can ship.

Run:
    OPENBLAS_NUM_THREADS=1 python -m twm.force_recovery.fullrange_react
"""
from __future__ import annotations

import glob
import os

import h5py
import hdf5plugin  # noqa: F401
import joblib
import numpy as np
import pyarrow.parquet as pq

from . import react_calib as RC
from .lut_calibration import crop
from .model_search import observation_features
from .run_episode import (DATA_ROOT, OUT_ROOT, STAGE_ROOT, open_episode,
                          reference_noise_area, reference_stack)

CACHE = RC.CACHE.parent / "fullrange_search_2026-09-24" / "react_sample.joblib"
TASKS = ("motherboard", "pushT", "rope", "toy")
PER_SIDE = 120


def pick_sides(rng):
    sides = []
    for task in TASKS:
        found = []
        for f in sorted(glob.glob(str(OUT_ROOT / task / "2026-09-1*" / "*.npz"))):
            z = np.load(f)
            h5 = DATA_ROOT / task / str(z["date"]) / f'{z["episode"]}.h5'
            if int(z["pipeline_version"]) == 8 and h5.is_file():
                found.append((float(z["force_normal_n"].max()), f))
        found.sort(reverse=True)
        chosen = {found[0][1], found[len(found) // 2][1]}
        rest = [f for _, f in found if f not in chosen]
        chosen |= set(rng.choice(rest, 2, replace=False))
        sides += [(task, f) for f in sorted(chosen)]
    return sides


def sample_rows(force, fresh, rng):
    """Half the budget spread over v8 force quantiles, half uniform over contact."""
    idx = np.flatnonzero(fresh)
    contact = idx[force[idx] > 0]
    zero = idx[force[idx] == 0]
    order = contact[np.argsort(force[contact])]
    strat = order[np.linspace(0, len(order) - 1, min(PER_SIDE // 2, len(order))).astype(int)] \
        if len(order) else np.array([], int)
    uni = rng.choice(contact, min(PER_SIDE // 3, len(contact)), replace=False) if len(contact) else np.array([], int)
    zz = rng.choice(zero, min(PER_SIDE // 6, len(zero)), replace=False) if len(zero) else np.array([], int)
    return np.unique(np.r_[strat, uni, zz]).astype(int)


def build():
    rng = np.random.default_rng(0)
    predict = RC.fit(report=False)
    rows, feats = [], {k: [] for k in ("basic", "geometry", "image", "combined")}
    for task, npz in pick_sides(rng):
        z = np.load(npz)
        date, ep, side = str(z["date"]), str(z["episode"]), str(z["side"])
        table = pq.read_table(STAGE_ROOT / task / "meta" / date / f"{ep}.parquet")
        inten = table[f"tactile_{side}_intensity"].to_numpy()
        fresh = table[f"tactile_{side}_is_new"].to_numpy().astype(bool)
        v8_npz = z["force_normal_n"]
        if len(v8_npz) != len(inten):
            print(f"skip {npz}: length mismatch", flush=True)
            continue
        h5 = DATA_ROOT / task / date / f"{ep}.h5"
        idx_map = np.asarray(open_episode(h5, task).align[side].index_map, np.int64)
        with h5py.File(h5, "r") as f:
            frames = f[f"gelsight/{side}/frames"]
            refs = np.stack([crop(im).astype(np.float32) for im in
                             reference_stack(frames, idx_map, inten, fresh)])
            ref = np.median(refs, 0)
            noise = reference_noise_area(refs)
            for r in sample_rows(v8_npz, fresh, rng):
                fi = min(int(idx_map[r]), len(frames) - 1)
                img = crop(frames[fi]).astype(np.float32)
                st = RC.force_stages(img, ref)
                v8 = predict(st, noise_area_mm2=noise)
                of = observation_features(img, ref, st)
                rows.append({"task": task, "date": date, "episode": ep, "side": side,
                             "row": int(r), "frame": fi, "v8_n": float(v8),
                             "v8_npz_n": float(v8_npz[r]), "area": st["feats"]["area"],
                             "contact_px": int(st["contact"].sum()),
                             "weight": RC.contact_weight(st["feats"]["area"], noise),
                             "noise_area": noise, "intensity": float(inten[r])})
                for k in feats:
                    feats[k].append(of[k])
        print(f"{task}/{date}/{ep}/{side}: {len(rows)} rows so far", flush=True)
    data = {k: np.asarray(v) for k, v in feats.items()}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump((rows, data), CACHE)
    v = np.array([[r["v8_n"], r["v8_npz_n"]] for r in rows])
    print(f"{len(rows)} frames -> {CACHE}; v8 recompute vs published npz: "
          f"max |diff| {np.abs(v[:, 0] - v[:, 1]).max():.4f} N", flush=True)


if __name__ == "__main__":
    build()
