"""Drop the OT-uninitialized prefix from each .pt episode.

For each .pt, find the first cam frame at which *all active sensors* have at
least one OT sample available, then slice every per-frame array in the .pt
from that index onwards. Records the trim offset in `_contact_meta.trim_offset`.

Episodes without a prefix issue are left untouched (no-op trim).
"""
import gc
import json
import sys
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch

PT_ROOT = Path("/media/yxma/Disk1/twm/processed/mode1_v1/motherboard")
H5_ROOT = Path("/media/yxma/Disk1/twm/data/motherboard")
TASKS_JSON = Path("/tmp/tasks_local.json")
BACKUP_DIR = Path("/media/yxma/Disk1/twm/processed/mode1_v1_pretrim_backup/motherboard")


def active_for(tasks, task, date):
    n = tasks.get("tasks", {}).get(task, {}).get("per_date_notes", {}).get(date, {})
    return tuple(n.get("active_sensors") or ("left", "right"))


def find_first_valid(cam_ts, sl_ts, sr_ts, active):
    """First cam frame index where every active sensor has ≥ 1 OT sample at
    or before that camera timestamp."""
    thresholds = []
    if "left" in active and len(sl_ts) > 0:
        thresholds.append(float(sl_ts[0]))
    if "right" in active and len(sr_ts) > 0:
        thresholds.append(float(sr_ts[0]))
    if not thresholds:
        return 0
    t_threshold = max(thresholds)
    return int(np.searchsorted(cam_ts, t_threshold))


def trim_one(pt_path, h5_path, active, dry_run=False):
    with h5py.File(h5_path, "r") as f:
        cam_ts = f["timestamps"][:]
        sl_ts = f["optitrack/sensor_left/timestamps"][:]
        sr_ts = f["optitrack/sensor_right/timestamps"][:]
    first_valid = find_first_valid(cam_ts, sl_ts, sr_ts, active)
    if first_valid <= 0:
        return ("ok", 0, len(cam_ts), len(cam_ts))

    if dry_run:
        return ("trim", first_valid, len(cam_ts), len(cam_ts) - first_valid)

    # Back up before modifying
    rel = pt_path.relative_to(PT_ROOT)
    backup = BACKUP_DIR / rel
    if not backup.exists():
        backup.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(pt_path, backup)

    ep = torch.load(pt_path, weights_only=False)
    T = ep["view"].shape[0]
    new_ep = {}
    for k, v in ep.items():
        if hasattr(v, "shape") and len(v.shape) >= 1 and v.shape[0] == T:
            new_ep[k] = v[first_valid:].clone()
        else:
            new_ep[k] = v
    meta = dict(new_ep.get("_contact_meta", {}))
    meta["trim_offset"] = first_valid
    meta["trim_reason"] = "OT not yet streaming at recording start"
    meta["pre_trim_n_frames"] = T
    new_ep["_contact_meta"] = meta
    torch.save(new_ep, pt_path)
    new_T = T - first_valid
    del ep, new_ep
    gc.collect()
    return ("trim", first_valid, T, new_T)


def main():
    dry_run = "--dry-run" in sys.argv
    tasks = json.loads(TASKS_JSON.read_text())
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Backup dir: {BACKUP_DIR}\n")

    n_trim = 0
    for pt in sorted(PT_ROOT.rglob("episode_*.pt")):
        date = pt.parent.name
        if date == "2026-03-23":
            continue
        ep_stem = pt.stem
        h5 = H5_ROOT / date / f"{ep_stem}.h5"
        if not h5.exists():
            print(f"SKIP {date}/{ep_stem}: no H5")
            continue
        active = active_for(tasks, "motherboard", date)
        try:
            status, off, old_T, new_T = trim_one(pt, h5, active, dry_run=dry_run)
            if status == "trim":
                n_trim += 1
                print(f"TRIM {date}/{ep_stem}: drop {off} frames "
                      f"({off / 30:.1f}s)  →  {old_T} → {new_T}")
            else:
                print(f"OK   {date}/{ep_stem}: no trim needed (T={old_T})")
        except Exception as exc:
            print(f"FAIL {date}/{ep_stem}: {exc}")

    print(f"\n{'Dry-run: would trim' if dry_run else 'Trimmed'} {n_trim} episodes.")


if __name__ == "__main__":
    main()
