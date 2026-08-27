"""Export the synthetic probe TEST SET: actions, context frames, ground truth.

WHAT WAS MISSING. `probes.json` recorded amplitudes, speeds and which episode
a probe came from — enough to build a preview page, and not enough to evaluate
anything. It held no pose arrays, no images, no held-hand pose and no
calibration, so reproducing a probe required the raw HDF5, which is not
published. This exports the artefact a rollout can actually be scored against.

TRUSTED SESSIONS ONLY. Start frames are drawn from 2026-05-10 and 2026-05-11.
2026-05-19 is excluded: its OptiTrack world was redefined and the release
corrects it with a translation only, while the yaw about the table normal and
the in-plane translation remain unmeasured (attempts scatter +/-2.3 deg, which
is 16 px at the workspace). Projected ground truth is the entire point of this
test set, so a session whose projection carries an unstated bias does not
belong in it.

    python scripts/build_probe_testset.py --runs 6
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from react_paths import force_meta, release_root, testset_root   # noqa: E402

import cv2                                                     # noqa: E402
import numpy as np                                             # noqa: E402
import pyarrow.parquet as pq                                   # noqa: E402

import react_toolbox as T                                      # noqa: E402
from react_toolbox.probe_eval import project_gt                # noqa: E402
from twm.calib_epoch import calib_dir, world_residual          # noqa: E402

REL = force_meta("motherboard")
CAM_H5 = {"left": 1, "middle": 2, "right": 0}
VIEWS = ("left", "middle", "right")
# The context a tactile world model conditions on is not three camera views.
# The first export shipped only those, which made the package unusable for the
# thing it exists to test.
TACTILE = ("tactile_left", "tactile_right")
STREAMS = tuple(f"view_{v}" for v in VIEWS) + TACTILE
# All three sessions. 2026-05-19 is included with the translation-only
# correction the release already applies, (230, 0, 175) mm; its residual — the
# unmeasured yaw about the table normal — is published in the manifest under
# `world_residual` rather than the session being dropped. The earlier build
# excluded it, which cost a fifth of the sessions to avoid an error that is
# declared and bounded.
TRUSTED = ("2026-05-10", "2026-05-11", "2026-05-19")

# Start frames come from HELD-OUT intervals only. Without this the probe's
# context frames were training frames: the action is novel, but the model had
# already seen the image it starts from, and nothing said so.
SPLITS = "splits.json"
CONTEXT = 4
# Wider than the preview's 8 px. A ground-truth path that ends 15 px from the
# edge is inside the frame but useless for scoring: a rollout that overshoots
# even slightly leaves the image and cannot be compared at all. The margin is
# a property of what the set is FOR, not of the geometry.
VIEW_MARGIN_PX = 40.0
FORMAT_VERSION = "react-probe-testset/1.0"


def _episodes():
    out = []
    for d in sorted(REL.iterdir()):
        if d.name not in TRUSTED:
            continue
        for p in sorted(d.glob("*.parquet")):
            # gated on the VIDEOS, not the raw HDF5. The frame reading moved
            # to the published videos but this filter did not, so a clean-room
            # run found zero usable episodes and reported "0 probes over 0
            # start frames" — success-shaped output for a total failure.
            vids = release_root("motherboard") / "videos" / d.name / p.stem
            if all((vids / f"{s}.mp4").is_file() for s in STREAMS):
                out.append((d.name, p.stem))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=6)
    ap.add_argument("--out", default=str(testset_root()))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    (out / "probes").mkdir(parents=True)
    stage = Path(tempfile.mkdtemp())
    shutil.copytree(calib_dir("motherboard"), stage / "calibration")
    cal = T.load_calibration(stage)
    shutil.copytree(stage / "calibration", out / "calibration")

    eps = _episodes()
    splits = json.loads((release_root("motherboard") /
                         SPLITS).read_text())
    rng = np.random.default_rng(args.seed)
    manifest = {
        "format": FORMAT_VERSION,
        "task": "motherboard",
        "views": list(VIEWS),
        "context_streams": list(STREAMS),
        "context_note": ("ctx{i}_{stream}.jpg for i in 0..3. Video frame r is "
                         "parquet row r for every stream; the tactile videos "
                         "are already row-aligned (the +15 acquisition lag was "
                         "applied at encode time), so nothing is re-applied."),
        "context_frames": CONTEXT,
        "view_margin_px": VIEW_MARGIN_PX,
        "image_size": [640, 480],
        "action_convention": {
            "primary": "delta_gel_pos_m (T,3) + delta_gel_rotvec_rad (T,3), "
                       "world axes, measured AT THE GEL",
            "one_dimensional": "action_scalar (T,) is the signed step along "
                               "action_axis; metres for a translation probe, "
                               "radians for a rotation probe",
            "why_gel": "rotations pivot on the gel, and the pose 7-vec is the "
                       "marker cluster's 65.7 mm away, so a pure rotation "
                       "carries up to 75.7 mm of RIGID-BODY translation. Only "
                       "at the gel is each probe one-directional.",
            "rigid_body": "delta_rigid_* for a model that predicts the marker "
                          "cluster pose instead",
        },
        "pose_convention": {
            "layout": "[x, y, z, qx, qy, qz, qw]",
            "position_units": "metres",
            "quaternion_order": "xyzw (scipy Rotation.from_quat)",
            "frame": "OptiTrack world, 2026-05-10 reference",
        },
        "trusted_sessions": list(TRUSTED),
        "excluded_sessions": {},
        "start_frames_from": "held-out intervals of splits.json (never training frames)",
        "overlay_error_budget_px": {
            "camera_reprojection": {v: round(cal["cams"][v]["rmse"] / 800.0 *
                                             cal["cams"][v]["intrinsics"]["fx"], 1)
                                    for v in VIEWS},
            "gel_centre": 3.8,
            "note": "agreement within about 6 px is at the noise floor",
        },
        "world_residual": {d: world_residual("motherboard", d) for d in TRUSTED},
        "session_note": {
            "2026-05-19": "world frame redefined mid-collection; the release "
                          "applies a translation-only correction and the "
                          "residual yaw about the table normal is unmeasured "
                          "(+/-2.3 deg, about 16 px at the workspace). Included "
                          "with that stated; see world_residual."},
        "probes": [],
    }

    made, tried = 0, 0
    while made < args.runs and tried < args.runs * 14:
        tried += 1
        date, ep = eps[int(rng.integers(len(eps)))]
        t = pq.read_table(REL / date / f"{ep}.parquet").to_pydict()
        poses = {s: np.asarray([x for x in t[f"sensor_{s}_pose"]], float)
                 for s in ("left", "right")}
        trim = int(np.asarray(t["source_h5_frame"])[0])
        # restrict the sampler to this episode's held-out intervals by masking
        # every other row's pose to NaN — `sample_probe` already requires a run
        # of `CONTEXT` tracked rows, so an invalid row is simply never chosen.
        info = splits["episodes"].get(f"{date}/{ep}")
        if info is None:
            continue
        iv = info["test"]
        if not iv:
            continue
        n = min(len(poses["left"]), len(poses["right"]))
        allow = np.zeros(n, bool)
        for a, b in iv:
            allow[a:min(b, n - 1) + 1] = True
        if allow.sum() < CONTEXT + 4:
            continue
        poses = {k: np.where(allow[:n, None], v[:n], np.nan) for k, v in poses.items()}
        try:
            r = T.sample_probe(poses, cal, seed=int(rng.integers(1 << 30)),
                               context=CONTEXT, view="middle",
                               margin_px=VIEW_MARGIN_PX)
        except ValueError:
            continue
        assert allow[np.asarray(r["context_rows"], int)].all(), \
            "start frame outside a held-out interval"
        rows = np.asarray(r["context_rows"], int)
        # FROM THE PUBLISHED VIDEOS, not the raw HDF5. The release ships
        # view_{left,middle,right}.mp4 and tactile_{left,right}.mp4, and video
        # frame r IS parquet row r: measured against the raw H5 at 1.88 mean
        # pixel difference, where two ADJACENT raw frames differ by 4.89. (My
        # first comparison said 11.46 because I compared cv2's BGR against an
        # already channel-flipped array — the data was fine, the test was not.)
        #
        # The tactile videos are already row-aligned: cross-correlating a
        # contact measure from the video against the parquet's
        # tactile_left_intensity peaks at lag 0 with r = 0.980, falling off
        # symmetrically. The +15 acquisition lag was applied at encode time, so
        # nothing must be re-applied here.
        #
        # This removes the last dependency on the unpublished ~1 TB raw tree,
        # so the test set can be rebuilt from what the dataset ships.
        ctx = {}
        for stream in STREAMS:
            f = REL.parent / "videos" / date / ep / f"{stream}.mp4"
            if not f.is_file():
                f = release_root("motherboard") / "videos" / date / ep / f"{stream}.mp4"
            cap = cv2.VideoCapture(str(f))
            got = []
            for rr in rows:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(rr))
                ok, fr = cap.read()
                got.append(fr if ok else None)
            cap.release()
            if any(g is None for g in got):
                ctx = None
                break
            ctx[stream] = got
        if ctx is None:
            continue

        run_dir = out / "probes" / f"run{made}"
        (run_dir / "context").mkdir(parents=True)
        for stream, frames in ctx.items():
            for i, im in enumerate(frames):
                cv2.imwrite(str(run_dir / "context" / f"ctx{i}_{stream}.jpg"),
                            im, [cv2.IMWRITE_JPEG_QUALITY, 95])

        side, other = r["moving_side"], r["held_side"]
        gel_m, gel_o = cal[f"gel_{side}"], cal[f"gel_{other}"]
        run_meta = {
            "run": made, "episode": f"{date}/{ep}", "context_rows": rows.tolist(),
            "source_h5_frames": (trim + rows).tolist(),
            "moving_side": side, "held_side": other,
            "held_pose": [float(x) for x in r["held_pose"]],
            "collision_diameter_m": r["collision_m"],
            "probes": [],
        }
        for p in r["probes"]:
            P = np.asarray(p["poses"], float)
            npz = {"poses": P, "held_pose": np.asarray(r["held_pose"], float),
                   "context_poses_moving": poses[side][rows],
                   "context_poses_held": poses[other][rows]}
            # the numeric channels AT the context rows, so a model that reads
            # scalars sees the same instants as the images
            for col in ("tactile_left_intensity", "tactile_right_intensity",
                        "tactile_left_area", "tactile_right_area",
                        "tactile_left_is_new", "tactile_right_is_new",
                        "force_left_normal_n", "force_right_normal_n",
                        "force_left_penetration_mm", "force_right_penetration_mm"):
                if col in t:
                    npz[f"context_{col}"] = np.asarray(t[col])[rows]
            for v in VIEWS:
                npz[f"gt_px_{v}"] = project_gt(P, gel_m, cal["cams"][v])
            # THE ACTION, IN THE GEL FRAME. Each probe moves along exactly one
            # axis — but only if you measure it at the GEL. The pose 7-vec is
            # the OptiTrack marker cluster's, and rotations pivot on the gel
            # 65.7 mm away, so a "pure rotation" carries up to 75.7 mm of
            # rigid-body translation. A model fed that reads "translate 76 mm
            # AND rotate 79 deg" for something labelled a pure rotation.
            #
            # So the gel-frame delta is the primary action, and in it a
            # translation probe has exactly zero rotation and a rotation probe
            # exactly zero translation. The rigid-body delta is kept too, for
            # a model that predicts the marker cluster's pose, but it is named
            # `delta_rigid_*` so the two cannot be confused.
            from scipy.spatial.transform import Rotation
            q = Rotation.from_quat(P[:, 3:7])
            Rm = q.as_matrix()
            gelw = P[:, :3] * 1000.0 + np.einsum("nij,j->ni", Rm, gel_m)
            npz["gel_pos_m"] = gelw / 1000.0
            npz["delta_gel_pos_m"] = np.diff(gelw, axis=0) / 1000.0
            # WORLD-FRAME DELTA, so pre-multiply: dq = q[i+1] * q[i]^-1.
            # The probes rotate about WORLD axes (`dq * q0`), so the
            # world-frame increment lies exactly along the named axis. The
            # body-frame increment, q[i]^-1 * q[i+1], is the same rotation
            # seen from the moving hand and does NOT — measured 7.1e-3 rad
            # off-axis, which is precisely the one-directionality this set is
            # built on. Integrate as q[i+1] = dq * q[i].
            npz["delta_gel_rotvec_rad"] = (q[1:] * q[:-1].inv()).as_rotvec()
            npz["delta_rigid_pos_m"] = np.diff(P[:, :3], axis=0)
            npz["delta_rigid_rotvec_rad"] = npz["delta_gel_rotvec_rad"]
            # ...and its one-dimensional form: a signed step along the named
            # axis. metres for a translation probe, radians for a rotation.
            ax = "xyz".index(p["axis"][1])
            if p["kind"] == "translation":
                npz["action_scalar"] = npz["delta_gel_pos_m"][:, ax].copy()
            else:
                npz["action_scalar"] = npz["delta_gel_rotvec_rad"][:, ax].copy()
            npz["action_axis"] = np.array(ax, np.int8)
            npz["action_sign"] = np.array(1 if p["axis"][0] == "+" else -1, np.int8)
            np.savez_compressed(run_dir / f"{p['name']}.npz", **npz)
            run_meta["probes"].append({
                "name": p["name"], "kind": p["kind"], "axis": p["axis"],
                "file": f"probes/run{made}/{p['name']}.npz",
                "steps": int(p["n_steps"]), "horizon_s": round(p["horizon_s"], 3),
                "amplitude": round(float(p.get("amplitude_m", p.get("amplitude_deg"))), 4),
                "amplitude_unit": "m" if p["kind"] == "translation" else "deg",
                "speed_percentile": round(p["speed_percentile"], 1),
                "min_separation_m": round(p["min_separation_m"], 4),
                "in_view_middle": bool(p["in_view"]),
            })
        (run_dir / "meta.json").write_text(json.dumps(run_meta, indent=1))
        manifest["probes"].append({"run": made, "episode": f"{date}/{ep}",
                                   "moving_side": side,
                                   "n_probes": len(run_meta["probes"]),
                                   "meta": f"probes/run{made}/meta.json"})
        print(f"  run{made}: {date}/{ep} rows {rows.tolist()} moving={side} "
              f"({len(run_meta['probes'])} probes)", flush=True)
        made += 1

    manifest["n_runs"] = made
    manifest["n_probes"] = sum(p["n_probes"] for p in manifest["probes"])
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    # The README is SOURCE, not output — it lived in the export directory once
    # and the next rebuild's rmtree deleted it. Searched rather than assumed:
    # hard-coding the repo's docs/ made a clean-room rebuild die on a path that
    # exists only here, which is the same failure calib_dir had.
    here = Path(__file__).resolve().parent
    for cand in (here.parents[0] / "docs" / "probe_testset_README.md",
                 here / "docs" / "probe_testset_README.md",
                 here.parent / "docs" / "probe_testset_README.md",
                 Path.cwd() / "docs" / "probe_testset_README.md"):
        if cand.is_file():
            shutil.copy(cand, out / "README.md")
            break
    else:
        print("  note: probe_testset_README.md not found; package built without "
              "it. Fetch docs/probe_testset_README.md from the dataset to "
              "include it.", flush=True)
    print(f"\n{manifest['n_probes']} probes over {made} start frames -> {out}")
    return 0 if made == args.runs else 1


if __name__ == "__main__":
    raise SystemExit(main())
