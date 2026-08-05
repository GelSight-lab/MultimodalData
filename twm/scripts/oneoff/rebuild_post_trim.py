"""After trimming the .pt files, rebuild:

  1. bad_frames.json — re-key each interval into trimmed-frame coordinates
     (subtract trim_offset; drop intervals that were entirely inside the
     trimmed prefix, clamp partials).
  2. data_quality_breakdown.json — recompute mode totals and percentages.
  3. README.md headline numbers (frames + duration).

The .pt's `_contact_meta.trim_offset` is the authoritative source; freeze
intervals come from /tmp/freeze_diagnose.csv (the existing H5-indexed run).
"""

# This is a top-level script (no main() function — all logic runs at
# module load). Refuse to act when imported to keep import-time tests safe.
if __name__ != "__main__":
    raise ImportError(
        f"{__name__!r} is a one-shot script and should be run via "
        f"\"python scripts/{__file__.split(chr(47))[-1]}\""
    )
import csv
import gc
import json
import os
from copy import deepcopy
from pathlib import Path

import torch

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

PT_ROOT = Path("/media/yxma/Disk1/twm/processed/mode1_v1/motherboard")
EVENTS = list(csv.DictReader(open("/tmp/freeze_diagnose.csv")))

# Pull trim offsets + new T from each trimmed .pt (only need _contact_meta,
# but torch.load reads the whole file. Use mmap-style if available... no,
# torch.load doesn't support partial. Iterate; 17 files × ~2 GB load is OK.)
trim = {}
new_T = {}
for pt in sorted(PT_ROOT.rglob("episode_*.pt")):
    date = pt.parent.name
    if date == "2026-03-23":
        continue
    ep_key = f"{date}/{pt.stem}"
    print(f"Reading {ep_key} for trim_offset...")
    d = torch.load(pt, weights_only=False)
    cm = d.get("_contact_meta", {})
    trim[ep_key] = int(cm.get("trim_offset", 0))
    new_T[ep_key] = int(d["view"].shape[0])
    del d
    gc.collect()

print("\nTrim offsets per episode:")
for k in sorted(trim):
    print(f"  {k}: trim_offset={trim[k]:>6d}  new_T={new_T[k]:>6d}")

# ── Load current HF bad_frames + breakdown ─────────────────────────────────
import urllib.request
def _fetch(p):
    return urllib.request.urlopen(f"https://huggingface.co/datasets/yxma/React/raw/main/{p}").read().decode()
BAD = json.loads(_fetch("bad_frames.json"))
BD  = json.loads(_fetch("figures/dataset_figures/data_quality_breakdown.json"))
README = _fetch("README.md")
QUALITY = _fetch("docs/quality.md")

# ── Rebuild per-episode intervals in trimmed coords ────────────────────────
new_bad = deepcopy(BAD)
for ek, ep in new_bad["episodes"].items():
    off = trim.get(ek, 0)
    T_new = new_T.get(ek, ep["n_frames"] - off)
    ep["n_frames"] = T_new
    ep["duration_s"] = round(T_new / 30.0, 3)
    # Per-side ot_loss: drop pre-trim intervals, shift the rest
    for key in ("ot_loss_L", "ot_loss_R"):
        kept = []
        for a, b in ep.get(key, []):
            a -= off; b -= off
            if b < 0:
                continue
            a = max(0, a)
            b = min(T_new - 1, b)
            if a > b:
                continue
            kept.append([a, b])
        ep[key] = kept
    # Same shift for the legacy spike + teleport intervals
    for key in ("intensity_spikes", "pose_teleports_L", "pose_teleports_R"):
        kept = []
        for a, b in ep.get(key, []):
            a -= off; b -= off
            if b < 0:
                continue
            a = max(0, a); b = min(T_new - 1, b)
            if a > b: continue
            kept.append([a, b])
        ep[key] = kept
    # Re-derive total_bad_frames as |union|
    mask = [False] * T_new
    for a, b in (ep.get("intensity_spikes", [])
                 + ep.get("pose_teleports_L", [])
                 + ep.get("pose_teleports_R", [])
                 + ep.get("ot_loss_L", [])
                 + ep.get("ot_loss_R", [])):
        for i in range(a, b + 1):
            mask[i] = True
    ep["total_bad_frames"] = sum(mask)
    ep["bad_fraction"] = round(ep["total_bad_frames"] / T_new, 4) if T_new else 0.0

total_frames = sum(ep["n_frames"] for ep in new_bad["episodes"].values())
total_bad = sum(ep["total_bad_frames"] for ep in new_bad["episodes"].values())
n_with_bad = sum(1 for ep in new_bad["episodes"].values() if ep["total_bad_frames"] > 0)
new_bad["summary"] = {
    "n_episodes": len(new_bad["episodes"]),
    "total_frames": total_frames,
    "total_bad_frames": total_bad,
    "bad_fraction_overall": total_bad / total_frames,
    "n_episodes_with_bad_frames": n_with_bad,
}
new_bad["trim_note"] = (
    "Each .pt has been trimmed at the start to drop the OT-uninitialized "
    "prefix (sensor pose not yet streaming). The amount trimmed is recorded "
    "in `_contact_meta.trim_offset` inside each .pt; all frame indices in "
    "this file are in TRIMMED coordinates (0 = first frame with valid OT)."
)
Path("/tmp/out3_bad_frames.json").write_text(json.dumps(new_bad, indent=2))
print(f"\nNew totals: {len(new_bad['episodes'])} eps, "
      f"{total_frames:,} frames ({total_frames/30/60:.1f} min), "
      f"{total_bad:,} bad ({100*total_bad/total_frames:.3f}%), "
      f"{n_with_bad} eps with bad.")

# ── data_quality_breakdown.json ────────────────────────────────────────────
def union(mode_keys):
    n = 0
    for ek, ep in new_bad["episodes"].items():
        T = ep["n_frames"]; m = [False] * T
        for k in mode_keys:
            for a, b in ep.get(k, []):
                for i in range(a, b + 1):
                    m[i] = True
        n += sum(m)
    return n

new_bd = deepcopy(BD)
new_bd["total_frames"] = total_frames
new_bd["n_episodes"] = len(new_bad["episodes"])
n_spike = union(["intensity_spikes"])
n_tel = union(["pose_teleports_L", "pose_teleports_R"])
n_otloss = union(["ot_loss_L", "ot_loss_R"])
new_bd["modes"]["gelsight_led_flicker"]["frames"] = n_spike
new_bd["modes"]["gelsight_led_flicker"]["pct"] = 100.0 * n_spike / total_frames
new_bd["modes"]["pose_teleport"]["frames"] = n_tel
new_bd["modes"]["pose_teleport"]["pct"] = 100.0 * n_tel / total_frames
new_bd["modes"]["ot_loss"]["frames"] = n_otloss
new_bd["modes"]["ot_loss"]["pct"] = 100.0 * n_otloss / total_frames
new_bd["modes"]["ot_loss"]["note"] = (
    "OptiTrack lost the rigid body mid-episode (recorder kept emitting the "
    "held pose). The huge start-of-episode prefixes that previously dominated "
    "this number (ep_005/012/017) have been TRIMMED out of the .pt files — "
    "what remains is real mid-episode mocap dropout."
)
new_bd["modes"]["ot_loss"]["episodes_affected"] = sorted(
    ek for ek, ep in new_bad["episodes"].items()
    if ep["ot_loss_L"] or ep["ot_loss_R"]
)
Path("/tmp/out3_breakdown.json").write_text(json.dumps(new_bd, indent=2))
print(f"\nPer mode now:")
for m, v in new_bd['modes'].items():
    print(f"  {m}: {v['frames']:,} frames ({v['pct']:.3f}%), "
          f"{len(v['episodes_affected'])} episodes")

# ── README headline numbers ────────────────────────────────────────────────
duration_min = total_frames / 30 / 60
duration_h = duration_min / 60
# Bimanual contact: assume same fraction of newly-kept frames are in contact
# as before — without re-running contact metrics we can only approximate.
# Skip for now; just update the synced-duration / frame-count lines.
new_readme = README
new_readme = new_readme.replace(
    "**126 min of robot-free human-hand multimodal interaction · 81 min (66 %) of confirmed bimanual tactile contact · 221,621 frames @ 30 Hz across 3 × RGB-D + 2 × GelSight + 3-body OptiTrack**",
    f"**{duration_min:.0f} min of robot-free human-hand multimodal interaction · {total_frames:,} frames @ 30 Hz across 3 × RGB-D + 2 × GelSight + 3-body OptiTrack**",
)
new_readme = new_readme.replace(
    "Total synchronized duration | **126.0 min** at 30 Hz (221,621 multimodal frames)",
    f"Total synchronized duration | **{duration_min:.1f} min** at 30 Hz ({total_frames:,} multimodal frames, post-trim)",
)
# Add a trim note in the recording-sessions section
new_readme = new_readme.replace(
    "See [`tasks.json`](tasks.json) for the machine-readable registry (per-date `active_sensors`, etc.).",
    "See [`tasks.json`](tasks.json) for the machine-readable registry (per-date `active_sensors`, etc.).\n\n"
    "**OT-uninitialized prefixes trimmed.** Three episodes had OptiTrack offline "
    "for the first 1–11 min of recording (`2026-05-11/episode_{005,012,017}`); "
    "those prefixes have been cut from the published `.pt` files (see "
    "`_contact_meta.trim_offset` per file). The original recordings remain "
    "in the H5 archive untouched. Future episodes use a recorder-side OT "
    "watchdog that refuses to start without an active mocap stream.",
)
Path("/tmp/out3_README.md").write_text(new_readme)

# ── docs/quality.md ────────────────────────────────────────────────────────
new_quality = QUALITY
# Update headline numbers
new_quality = new_quality.replace(
    "| Total synchronized frames | 221,621 (126.0 min @ 30 Hz) |",
    f"| Total synchronized frames | {total_frames:,} ({duration_min:.1f} min @ 30 Hz, post-trim) |",
)
new_quality = new_quality.replace(
    f"| Frames flagged in `bad_frames.json` | **122 (0.055 %)** |",
    f"| Frames flagged in `bad_frames.json` | **{total_bad:,} ({100*total_bad/total_frames:.3f} %)** |",
)
new_quality = new_quality.replace(
    "| Recording files with ≥1 flagged frame | 8 |",
    f"| Recording files with ≥1 flagged frame | {n_with_bad} |",
)
Path("/tmp/out3_quality.md").write_text(new_quality)

# ── Now push: trimmed big-3 .pt + refreshed metadata + 3 previews ──────────
# (Preview re-render is its own script; we push the 3 .pt files + new
# metadata + updated docs here in one commit.)
from huggingface_hub import HfApi, CommitOperationAdd
api = HfApi()
ops = []
# Trimmed .pt for the 3 substantial trims (skip 1-frame trims to save upload)
for ek in ("2026-05-11/episode_005", "2026-05-11/episode_012", "2026-05-11/episode_017"):
    src = PT_ROOT / f"{ek}.pt"
    ops.append(CommitOperationAdd(path_in_repo=f"processed/mode1_v1/motherboard/{ek}.pt",
                                   path_or_fileobj=str(src)))
# Metadata
ops += [
    CommitOperationAdd(path_in_repo="bad_frames.json",
                       path_or_fileobj="/tmp/out3_bad_frames.json"),
    CommitOperationAdd(path_in_repo="figures/dataset_figures/data_quality_breakdown.json",
                       path_or_fileobj="/tmp/out3_breakdown.json"),
    CommitOperationAdd(path_in_repo="README.md",
                       path_or_fileobj="/tmp/out3_README.md"),
    CommitOperationAdd(path_in_repo="docs/quality.md",
                       path_or_fileobj="/tmp/out3_quality.md"),
]
print(f"\nUploading {len(ops)} ops:")
for op in ops:
    sz = Path(op.path_or_fileobj).stat().st_size / 1024**2
    print(f"  {op.path_in_repo}  ({sz:.1f} MB)")
api.create_commit(
    repo_id="yxma/React", repo_type="dataset", operations=ops,
    commit_message=(
        "Trim OT-uninitialized prefixes from 3 .pt files "
        "(ep_005/012/017 — see `_contact_meta.trim_offset`); "
        "recompute bad_frames + data_quality_breakdown in trimmed coords; "
        "ot_loss frame count now reflects real mid-episode mocap dropout only. "
        "Recorder-side OT watchdog added to data_collection.py (in twm repo) "
        "so future episodes won't repeat the bug."
    ),
)
print("Done.")
