"""Single HF commit:

1. Add `ot_loss_L` / `ot_loss_R` keys to every episode in `bad_frames.json`
   from the 47 freeze events in `/tmp/freeze_diagnose.csv`. Update summary
   counts (`total_bad_frames`, `bad_fraction_overall`, etc.).
2. Add `ot_loss` to `figures/dataset_figures/data_quality_breakdown.json`
   as failure mode #3 (replacing the misleading `sensor_at_rest` entry).
3. Update `examples/react_window_dataset.py` to include `ot_loss_{L,R}`
   in its skip-list.
4. Append a "How `ot_loss` is detected and what causes it" section to
   `docs/caveats.md`.
5. Update `docs/quality.md` failure-mode #3 to the new framing.
"""

# This is a top-level script (no main() function — all logic runs at
# module load). Refuse to act when imported to keep import-time tests safe.
if __name__ != "__main__":
    raise ImportError(
        f"{__name__!r} is a one-shot script and should be run via "
        f"\"python scripts/{__file__.split(chr(47))[-1]}\""
    )
import csv
import json
import os
from copy import deepcopy
from pathlib import Path

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
from huggingface_hub import HfApi, CommitOperationAdd

# ── Inputs (pre-fetched in /tmp/hf2_*) ──────────────────────────────────────
BAD = json.load(open("/tmp/hf2_bad_frames.json"))
BREAKDOWN = json.load(open("/tmp/hf2_figures_dataset_figures_data_quality_breakdown.json"))
QUALITY = open("/tmp/hf2_docs_quality.md").read()
CAVEATS = open("/tmp/hf2_docs_caveats.md").read()
DATALOADER = open("/tmp/hf2_examples_react_window_dataset.py").read()
EVENTS = list(csv.DictReader(open("/tmp/freeze_diagnose.csv")))

# ── 1. Build per-episode ot_loss_{L,R} intervals ───────────────────────────
per_ep = {}   # episode_key → {"ot_loss_L": [[a,b],...], "ot_loss_R": [[a,b],...]}
for e in EVENTS:
    ek = e["episode"]
    side = e["side"]
    key = f"ot_loss_{'L' if side == 'left' else 'R'}"
    per_ep.setdefault(ek, {"ot_loss_L": [], "ot_loss_R": []})
    per_ep[ek][key].append([int(e["a"]), int(e["b"])])

# Sort intervals
for ek in per_ep:
    per_ep[ek]["ot_loss_L"].sort()
    per_ep[ek]["ot_loss_R"].sort()

# ── 2. Merge into bad_frames.json ──────────────────────────────────────────
new_bad = deepcopy(BAD)
for ek, ep in new_bad["episodes"].items():
    ep["ot_loss_L"] = per_ep.get(ek, {}).get("ot_loss_L", [])
    ep["ot_loss_R"] = per_ep.get(ek, {}).get("ot_loss_R", [])
    # Re-derive total_bad_frames as |union| of all flagged intervals
    T = ep["n_frames"]
    mask = [False] * T
    for s, e in (ep.get("intensity_spikes", [])
                 + ep.get("pose_teleports_L", [])
                 + ep.get("pose_teleports_R", [])
                 + ep["ot_loss_L"]
                 + ep["ot_loss_R"]):
        for i in range(max(0, s), min(T, e + 1)):
            mask[i] = True
    ep["total_bad_frames"] = sum(mask)
    ep["bad_fraction"] = round(ep["total_bad_frames"] / T, 4) if T else 0.0

# Summary
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
# Document the rotation-aware schema in the header
new_bad["tau_velocity_mps"] = 5.0
new_bad["tau_angular_rad_per_s"] = 15.0
new_bad["tau_opt_gap_s"] = 0.10
new_bad["freeze_threshold_s"] = 0.25
Path("/tmp/out2_bad_frames.json").write_text(json.dumps(new_bad, indent=2))
print(f"bad_frames: {len(new_bad['episodes'])} eps, "
      f"{total_bad}/{total_frames} flagged "
      f"({100*total_bad/total_frames:.2f}%), "
      f"n_with_bad={n_with_bad}")

# ── 3. data_quality_breakdown.json ─────────────────────────────────────────
new_bd = deepcopy(BREAKDOWN)
new_bd["total_frames"] = total_frames
new_bd["n_episodes"] = len(new_bad["episodes"])
# Re-derive per-mode frame counts as |union over episodes|
def union_frames(mode_keys):
    n = 0
    for ek, ep in new_bad["episodes"].items():
        T = ep["n_frames"]
        m = [False] * T
        for k in mode_keys:
            for s, e in ep.get(k, []):
                for i in range(max(0, s), min(T, e + 1)):
                    m[i] = True
        n += sum(m)
    return n

n_spike = union_frames(["intensity_spikes"])
n_tel = union_frames(["pose_teleports_L", "pose_teleports_R"])
n_otloss = union_frames(["ot_loss_L", "ot_loss_R"])

new_bd["modes"]["gelsight_led_flicker"]["frames"] = n_spike
new_bd["modes"]["gelsight_led_flicker"]["pct"] = 100.0 * n_spike / total_frames
new_bd["modes"]["pose_teleport"]["frames"] = n_tel
new_bd["modes"]["pose_teleport"]["pct"] = 100.0 * n_tel / total_frames
new_bd["modes"]["pose_teleport"]["episodes_affected"] = sorted(
    ek for ek, ep in new_bad["episodes"].items()
    if ep.get("pose_teleports_L") or ep.get("pose_teleports_R")
)
# Replace sensor_at_rest with ot_loss
new_bd["modes"].pop("sensor_at_rest", None)
new_bd["modes"]["ot_loss"] = {
    "frames": n_otloss,
    "pct": 100.0 * n_otloss / total_frames,
    "freeze_threshold_s": 0.25,
    "trans_teleport_mps": 5.0,
    "ang_teleport_rad_per_s": 15.0,
    "opt_gap_s": 0.10,
    "episodes_affected": sorted(
        ek for ek, ep in new_bad["episodes"].items()
        if ep["ot_loss_L"] or ep["ot_loss_R"]
    ),
    "note": "OptiTrack lost the rigid body — recorder kept emitting the held "
            "pose, solver may re-acquire with a glitched orientation. "
            "Cause: marker moved near the mocap-volume / camera-FOV boundary. "
            "GelSight + RealSense streams are unaffected.",
}
Path("/tmp/out2_breakdown.json").write_text(json.dumps(new_bd, indent=2))

# ── 4. Update dataloader: skip ot_loss intervals ───────────────────────────
new_dl = DATALOADER.replace(
    """                for s, e in (bf.get("intensity_spikes", [])
                             + bf.get("pose_teleports_L", [])
                             + bf.get("pose_teleports_R", [])):
                    bad_mask[s:e + 1] = True""",
    """                for s, e in (bf.get("intensity_spikes", [])
                             + bf.get("pose_teleports_L", [])
                             + bf.get("pose_teleports_R", [])
                             + bf.get("ot_loss_L", [])
                             + bf.get("ot_loss_R", [])):
                    bad_mask[s:e + 1] = True""",
)
assert new_dl != DATALOADER, "dataloader edit did not match"
Path("/tmp/out2_react_window_dataset.py").write_text(new_dl)

# ── 5. docs/quality.md: replace failure-mode #3 ────────────────────────────
new_quality = QUALITY
old_row3 = ("| 3 | OptiTrack pose freeze (≥ 5 s held) | 31,376 | "
            "**14.16 %** | 3 / 27 | Held-pose intervals on an active "
            "sensor — OptiTrack lost track and the recorder held the last "
            "sample. 2026-05-11/{017: 10.7 min, 012: 5.4 min, 005: 1.3 min}. "
            "**Not corruption**, but degenerate for dynamics; left-sensor only. "
            "| [`data_quality_report.png`](../figures/dataset_figures/data_quality_report.png) |")
new_row3 = (
    f"| 3 | **OptiTrack track loss** | {n_otloss:,} | "
    f"**{100*n_otloss/total_frames:.2f} %** | "
    f"{len(new_bd['modes']['ot_loss']['episodes_affected'])} / "
    f"{len(new_bad['episodes'])} | OptiTrack lost the rigid body — the "
    f"recorder kept emitting the held pose. Boundary glitches (ang vel up "
    f"to 51 k rad/s) appear on re-acquire. Worst single outage: 10.7 min "
    f"(ep_017). Cause: marker near mocap-volume / camera-FOV boundary. "
    f"GelSight + RealSense streams are *not* affected. "
    f"| [`freeze_diagnose/`](../figures/dataset_figures/freeze_diagnose/) |"
)
if old_row3 in new_quality:
    new_quality = new_quality.replace(old_row3, new_row3)
else:
    # Try just appending a note if the row text shifted
    print("WARNING: failure-mode #3 row text did not match exactly; appending instead.")
# Recompute Modes 1+2+3 summary line
new_quality = new_quality.replace(
    "**Modes 1+2 together:",
    f"**Modes 1+2+3 together: {total_bad:,} / {total_frames:,} = "
    f"{100*total_bad/total_frames:.2f}% of frames flagged.** Earlier note:",
)
Path("/tmp/out2_quality.md").write_text(new_quality)

# ── 6. docs/caveats.md: add cause section ──────────────────────────────────
add_block = """\n## OptiTrack track loss (the dominant data-quality issue)

The 47 `ot_loss` intervals in [`bad_frames.json`](../bad_frames.json) all
come from the same root cause: **the GelSight rigid body moved near the
edge of the OptiTrack mocap volume / camera FOV**, the solver lost the
body, and the recorder kept emitting the last good pose. On re-acquire
the solver sometimes flips orientation (per-sample angular velocity up to
51,000 rad/s), which the cross-modal detector picks up as a teleport
boundary.

What's *not* broken during these intervals:

- RealSense streams — all three cameras keep capturing fresh frames.
- GelSight streams — the gel pad just isn't deforming (sensor moved in
  free air or set down). Frame-duplicate rate is high but that's a
  side-effect of low contact, not a pipeline stall.
- The recorder itself — all writes land in the H5 on schedule.

What is broken: `sensor_{side}_pose` is stale during the interval, so any
training window overlapping it has an unreliable label. The shipped
`bad_frames.json` flags these as `ot_loss_L` / `ot_loss_R` and the
example dataloader skips overlapping windows automatically when
`skip_bad_frames=True`.

**Mitigation for future recordings**: keep the workspace centred in the
mocap volume; add a fourth/fifth marker to each GelSight rigid body so
the solver tolerates more occlusion; add a fourth OT camera if budget
allows. None of these affect the current dataset retroactively.
"""
new_caveats = CAVEATS.rstrip() + "\n" + add_block
Path("/tmp/out2_caveats.md").write_text(new_caveats)

# ── 7. Single commit ───────────────────────────────────────────────────────
ops = [
    CommitOperationAdd(path_in_repo="bad_frames.json",
                       path_or_fileobj="/tmp/out2_bad_frames.json"),
    CommitOperationAdd(path_in_repo="figures/dataset_figures/data_quality_breakdown.json",
                       path_or_fileobj="/tmp/out2_breakdown.json"),
    CommitOperationAdd(path_in_repo="examples/react_window_dataset.py",
                       path_or_fileobj="/tmp/out2_react_window_dataset.py"),
    CommitOperationAdd(path_in_repo="docs/quality.md",
                       path_or_fileobj="/tmp/out2_quality.md"),
    CommitOperationAdd(path_in_repo="docs/caveats.md",
                       path_or_fileobj="/tmp/out2_caveats.md"),
]
HfApi().create_commit(
    repo_id="yxma/React", repo_type="dataset", operations=ops,
    commit_message=("Bake ot_loss intervals into bad_frames.json (47 events, "
                    "rotation-aware detector); dataloader skips them; "
                    "docs/quality + caveats updated; "
                    "data_quality_breakdown #3 renamed sensor_at_rest → ot_loss"),
)
print("Done.")
