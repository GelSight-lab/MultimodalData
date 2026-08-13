"""Write the estimated normal force into the React dataset as observation
columns, plus DexForce-style force-informed target poses.

Reads
    force:  ``force_recovery/<task>/<date>/<episode>_<side>.npz``
            (written by :mod:`run_episode`, field ``force_normal_n``)
    meta:   ``release/<task>/meta/<date>/<episode>.parquet``

Writes a *parallel* tree — the source release is never touched:

    release_force/<task>/meta/<date>/<episode>.parquet   original 19 columns
                                                         + 6 new ones
    release_force/<task>/meta/<date>/<episode>.force.json  per-episode sidecar
    release_force/force_export_manifest.json               run-level sidecar

Every run rebuilds each output file from the two immutable inputs, so the
export is idempotent (verified byte-identical over two runs, see ``verify``).

New columns, per side ``left`` / ``right``
------------------------------------------
``force_<side>_normal_n``          float32, newtons, >= 0
``force_<side>_penetration_mm``    float32, millimetres, = force / k
``force_<side>_target_pose``       list<double>[7], same layout and units as
                                   ``sensor_<side>_pose`` (x, y, z in metres;
                                   qx, qy, qz, qw). Position is displaced by
                                   ``penetration_mm`` along the pressing
                                   direction; the quaternion is copied
                                   unchanged (DexForce displaces position
                                   only).

The stiffness ``k`` is an *assumption*, not a measurement.  Its value is not
declared here — it is read from ``pipeline.STIFFNESS_N_PER_MM`` (derived from
``dexforce.STIFFNESS_N_PER_M``, the single definition in the codebase), so the
dataset column and the site figures can never disagree about it.  It is then
recorded in three places so a reader can never be unaware of it: as per-field
parquet metadata on each new column, as schema-level metadata under the key
``twm.force_export``, and in the sidecar JSON next to the parquet.

Pressing direction (derived, not hard-coded)
--------------------------------------------
``n_hat = R(q_row) @ gel_axis_in_rigid`` where ``gel_axis_in_rigid`` comes
from the rig's dual-ball calibration (``calibration/*/T_gel_to_rigid_<side>``,
pose-to-pose consistency <= 1.07 deg) and ``R(q_row)`` is the *per-row*
world rotation of the sensor rigid body.  The direction therefore rotates
with the sensor and is never a fixed world axis.  In the rigid-body frame
that axis is dominated by -Y (left ``[-0.174, -0.932, -0.316]``, right
``[0.350, -0.925, 0.149]``): the naive "tool z axis" guess would be 71-108
deg off, which is why it is read from calibration rather than assumed.

The *sign* and the direction itself are validated against the motion data
alone by :func:`direction_agreement`, per sensor-side: during force rise the
mount must advance along ``n_hat`` and during release it must retreat.  The
per-side numbers land in the sidecar; the aggregate is printed by ``verify``.

Edge cases, all explicit
------------------------
* **No contact.**  ``run_episode`` emits force exactly ``0.0`` when the
  estimator finds no contact volume.  Those rows get ``penetration_mm = 0``
  and ``target_pose`` set by *copy* of the observed pose, so the identity is
  enforced rather than left to floating point.  Never NaN.
* **Duplicate tactile frames.**  ``tactile_<side>_is_new == False`` rows carry
  the previous frame's pixels, and ``run_episode`` already forward-fills the
  estimate onto them (``if is_new[row] or last is None``).  This export keeps
  that forward fill and asserts it: on every duplicate row the force must
  equal the previous row's exactly.  Consequence worth knowing downstream:
  force is a ~8 Hz signal held on a 30 Hz pose stream, so on duplicate rows
  the target pose combines a *fresh* pose with a *stale* force.
* **Row-count mismatch.**  Raises :class:`RowCountMismatch` and aborts.  No
  truncation, no padding, no partial write.
* **Missing npz for a side.**  Raises; the run reports which episodes.

Usage
-----
    python -m force_recovery.export_force_columns export [--stiffness 1.0]
    python -m force_recovery.export_force_columns verify
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .dexforce import gel_axis, quat_to_matrix
from .pipeline import STIFFNESS_N_PER_MM, penetration_mm
from .run_episode import OUT_ROOT as FORCE_ROOT
from .run_episode import STAGE_ROOT

EXPORT_ROOT = Path("/media/yxma/Disk1/twm/release_force")

# Stiffness for ``penetration = F / k``.  NOT redeclared here: it comes from
# ``pipeline.STIFFNESS_N_PER_MM``, itself derived from the single definition in
# ``dexforce.STIFFNESS_N_PER_M``, where the value and the measurement that
# settled it are written down.  A stiffness that disagreed between this dataset
# column and the rest of the codebase would be a silent lie about what the
# action means -- which is why the value is not repeated here.  It used to be,
# as "(1000 N/m = 1 N/mm, the project's declared starting point)" alongside a
# quoted gel stiffness of "median 15.4 N/mm"; both were stale copies by the
# time anyone read them.  ``verify`` prints the current numbers every run.
#
# It is an ASSUMPTION about the deployment controller, not a measured gel
# property.  What ``verify`` does enforce is the one thing that is not a matter
# of taste: a commanded target may not sit further past the surface than the
# gel can be compressed.

# GelSight Mini elastomer thickness; the hard physical ceiling a penetration
# interpreted as gel compression may not exceed. Defined in lut_calibration,
# beside the other sensor geometry, and imported so the reconstruction and the
# exporter cannot disagree about how thick the gel is.
from .lut_calibration import GEL_THICKNESS_MM  # noqa: E402

SIDES = ("left", "right")
POSE_DIM = 7

# Imported, not re-declared: the exporter must reject exactly the npz the
# batch considers stale. A second literal here would silently let a
# superseded calibration into the published columns the next time one side
# was bumped and the other forgotten — which is how 36 sidecars ended up
# advertising a scale that no longer existed.
from .batch_worker import PIPELINE_VERSION as MIN_PIPELINE_VERSION  # noqa: E402


class RowCountMismatch(RuntimeError):
    """npz row count != parquet row count. Never aligned by truncation."""


class MissingForceFile(RuntimeError):
    """A release episode has no force npz for one of its sides."""


# --------------------------------------------------------------------------
# geometry


def press_direction(task: str, side: str, pose: np.ndarray) -> np.ndarray:
    """Per-row unit pressing direction in the world frame, ``(T, 3)``.

    ``pose`` is ``(T, 7)`` = x, y, z, qx, qy, qz, qw.  See the module
    docstring for why the local axis comes from calibration.
    """
    axis = gel_axis(task, side)
    n = quat_to_matrix(np.asarray(pose, np.float64)[:, 3:POSE_DIM]) @ axis
    return n / np.linalg.norm(n, axis=1, keepdims=True)


def direction_agreement(pose_xyz_mm: np.ndarray, n_hat: np.ndarray,
                        force: np.ndarray, win: int = 5,
                        force_floor: float = 0.5,
                        df_floor: float = 0.5) -> dict:
    """Does the mount actually advance along ``n_hat`` when force rises?

    Uses only pose + force, no calibration, so it is an independent check on
    the direction taken from calibration.  Returns the mean along-normal
    velocity during force rise (should be > 0) and during release (< 0), and
    the correlation between dF and the along-normal velocity.
    """
    T = len(force)
    if T < 4 * win + 2:
        return {"usable": False}
    v = np.zeros_like(pose_xyz_mm)
    v[win:-win] = pose_xyz_mm[2 * win:] - pose_xyz_mm[:-2 * win]
    v_n = np.einsum("ij,ij->i", v, n_hat)
    d_f = np.zeros(T)
    d_f[win:-win] = force[2 * win:] - force[:-2 * win]

    active = force > force_floor
    rise = active & (d_f > df_floor)
    fall = active & (d_f < -df_floor)
    if rise.sum() < 20 or fall.sum() < 20:
        return {"usable": False}
    corr = float(np.corrcoef(d_f[active], v_n[active])[0, 1]) \
        if active.sum() > 2 and np.std(v_n[active]) > 0 else float("nan")
    return {
        "usable": True,
        "v_dot_n_rise_mm": float(np.mean(v_n[rise])),
        "v_dot_n_fall_mm": float(np.mean(v_n[fall])),
        "corr_dforce_vdotn": corr,
        "n_rise": int(rise.sum()), "n_fall": int(fall.sum()),
    }


# --------------------------------------------------------------------------
# export


def _episodes() -> list[tuple[str, str, str]]:
    """(task, date, episode) for every release meta parquet, sorted."""
    out = []
    for parquet in sorted(STAGE_ROOT.glob("*/meta/*/*.parquet")):
        out.append((parquet.parts[-4], parquet.parts[-2], parquet.stem))
    return out


def _force_npz(task: str, date: str, ep: str, side: str) -> Path:
    return FORCE_ROOT / task / date / f"{ep}_{side}.npz"


def _list_column(values: np.ndarray) -> pa.ListArray:
    """(T, D) float64 -> list<double> column, matching ``sensor_*_pose``."""
    T, D = values.shape
    offsets = pa.array(np.arange(T + 1, dtype=np.int32) * D, type=pa.int32())
    return pa.ListArray.from_arrays(
        offsets, pa.array(np.ascontiguousarray(values, np.float64).ravel(),
                          type=pa.float64()))


def build_side(task: str, date: str, ep: str, side: str, table: pa.Table,
               stiffness: float) -> tuple[dict, dict]:
    """Columns + diagnostics for one sensor-side.  Raises on any mismatch."""
    npz_path = _force_npz(task, date, ep, side)
    if not npz_path.exists():
        raise MissingForceFile(f"{npz_path} missing for {task}/{date}/{ep}")
    with np.load(npz_path) as d:
        force = np.asarray(d["force_normal_n"], np.float64)
        max_depth = np.asarray(d["max_depth_mm"], np.float64)
        # THE FRAME EACH NUMBER CAME FROM. Recorded by `run_episode` at the
        # read, not recomputed here — recomputing it would only prove that
        # two copies of one formula agree, which is precisely the arrangement
        # that produced three alignment defects in a day.
        if "source_frame" not in d:
            raise ValueError(
                f"{npz_path}: no source_frame array. Reprocess with the "
                f"current `run_episode`; a force value that cannot name its "
                f"tactile frame is not exportable.")
        source_frame = np.asarray(d["source_frame"], np.int64)
        # WHICH calibration produced these newtons, carried in the artifact.
        # The old field was a single float `scale_n_per_mm3`, which could only
        # describe a one-number map; the current estimator is a gain field plus
        # a clipping correction plus an isotonic fit, so a float cannot name
        # it. Refuse rather than guess: an unlabelled npz is exactly the state
        # in which the pixel/mm mismatch went unnoticed.
        version = int(d["pipeline_version"]) if "pipeline_version" in d else 0
        if "force_calibration" not in d:
            raise ValueError(
                f"{npz_path}: no force_calibration field (pipeline_version "
                f"{version}) -- run `python -m force_recovery.stamp_calibration`")
        calibration = str(d["force_calibration"])
    if version < MIN_PIPELINE_VERSION:
        raise ValueError(
            f"{npz_path}: pipeline_version {version} < {MIN_PIPELINE_VERSION}. "
            f"Versions below {MIN_PIPELINE_VERSION} come from the calibration "
            f"that scored rho 0.143 end to end; re-run run_episode.")
    T = table.num_rows
    if len(force) != T:
        raise RowCountMismatch(
            f"{task}/{date}/{ep}_{side}: npz has {len(force)} rows, "
            f"parquet has {T} -- refusing to align by truncation")
    if not np.isfinite(force).all():
        raise ValueError(f"{npz_path}: non-finite force")
    if (force < 0).any():
        raise ValueError(f"{npz_path}: negative force")

    pose = np.array(table[f"sensor_{side}_pose"].to_pylist(), np.float64)
    if pose.shape != (T, POSE_DIM):
        raise ValueError(f"sensor_{side}_pose has shape {pose.shape}")
    is_new = np.asarray(table[f"tactile_{side}_is_new"].to_numpy(), bool)

    # forward-fill contract from run_episode: duplicated tactile rows must
    # carry the previous row's estimate, exactly.
    dup = np.zeros(T, bool)
    dup[1:] = ~is_new[1:]
    held = dup.copy()
    held[1:] &= force[1:] == force[:-1]
    ffill_violations = int(dup.sum() - held.sum())
    if ffill_violations:
        raise ValueError(
            f"{task}/{date}/{ep}_{side}: {ffill_violations} duplicate rows "
            f"whose force differs from the previous row -- the forward fill "
            f"assumed by this export does not hold")

    penetration = penetration_mm(force, stiffness)        # N / (N/mm) = mm
    n_hat = press_direction(task, side, pose)
    target = pose.copy()
    contact = force > 0.0
    target[contact, :3] += (penetration[contact, None] / 1000.0
                            ) * n_hat[contact]            # mm -> m
    # no-contact rows are left as a byte copy of the observed pose

    diag = {
        "side": side,
        "rows": T,
        "npz": str(npz_path),
        "force_calibration": calibration,
        "pipeline_version": version,
        "n_contact_rows": int(contact.sum()),
        "no_contact_frac": float(1.0 - contact.mean()),
        "is_new_frac": float(is_new.mean()),
        "duplicate_rows_forward_filled": int(dup.sum()),
        "force_p50_n": float(np.percentile(force, 50)),
        "force_p95_n": float(np.percentile(force, 95)),
        "force_max_n": float(force.max()),
        "force_p50_contact_n": float(np.percentile(force[contact], 50))
        if contact.any() else 0.0,
        "penetration_p50_mm": float(np.percentile(penetration, 50)),
        "penetration_p95_mm": float(np.percentile(penetration, 95)),
        "penetration_max_mm": float(penetration.max()),
        "gel_max_depth_p95_mm": float(np.percentile(max_depth, 95)),
        "gel_max_depth_max_mm": float(max_depth.max()),
        "identity_max_abs_dev": float(
            np.abs(target[~contact] - pose[~contact]).max())
        if (~contact).any() else 0.0,
        "direction": direction_agreement(pose[:, :3] * 1000.0, n_hat, force),
    }
    columns = {
        f"force_{side}_normal_n": pa.array(force.astype(np.float32)),
        f"force_{side}_penetration_mm": pa.array(
            penetration.astype(np.float32)),
        f"force_{side}_target_pose": _list_column(target),
        f"force_{side}_source_frame": pa.array(source_frame.astype(np.int32)),
    }
    return columns, diag


def _field_meta(name: str, stiffness: float) -> dict:
    common = {
        "twm.source": "force_recovery/<task>/<date>/<episode>_<side>.npz "
                      "(force_recovery.run_episode)",
        "twm.stiffness_n_per_mm": repr(stiffness),
    }
    if name.endswith("_normal_n"):
        common |= {"twm.units": "N",
                   "twm.desc": "estimated normal force, >= 0; exactly 0 on "
                               "no-contact rows; held on duplicate tactile "
                               "rows (tactile_<side>_is_new == False)"}
    elif name.endswith("_penetration_mm"):
        common |= {"twm.units": "mm",
                   "twm.desc": "force / stiffness; ASSUMED stiffness, see "
                               "twm.stiffness_n_per_mm"}
    elif name.endswith("_source_frame"):
        common |= {"twm.units": "index",
                   "twm.desc": "index into gelsight/<side>/frames of the "
                               "SOURCE H5 that this row's force was computed "
                               "from, recorded at the read. Held rows repeat "
                               "it, so rows sharing one tactile image are "
                               "explicit rather than inferred from is_new. "
                               "Re-reading that frame reproduces the force to "
                               "the bit (scripts/test_force_names_its_frame)"}
    else:
        common |= {"twm.units": "m,m,m,quat(xyzw)",
                   "twm.desc": "DexForce virtual target: observed pose with "
                               "position advanced by penetration_mm along the "
                               "per-row pressing direction "
                               "R(q) @ gel_axis_in_rigid; quaternion copied; "
                               "identical to the observed pose when force==0"}
    return common


def export_episode(task: str, date: str, ep: str, stiffness: float,
                   root: Path = EXPORT_ROOT) -> dict:
    src = STAGE_ROOT / task / "meta" / date / f"{ep}.parquet"
    table = pq.read_table(str(src))

    diags = []
    fields = list(table.schema)
    arrays = list(table.columns)
    for side in SIDES:
        columns, diag = build_side(task, date, ep, side, table, stiffness)
        diags.append(diag)
        for name, arr in columns.items():
            fields.append(pa.field(name, arr.type,
                                   metadata=_field_meta(name, stiffness)))
            arrays.append(arr)

    header = {
        "generator": "twm.force_recovery.export_force_columns",
        "stiffness_n_per_mm": stiffness,
        "penetration": "penetration_mm = force_normal_n / "
                       "stiffness_n_per_mm  (ASSUMED stiffness)",
        "press_direction": "R(q_row) @ gel_axis_in_rigid from "
                           "calibration/*/T_gel_to_rigid_<side>.json",
        "no_contact_rule": "force == 0 -> penetration 0 and target_pose is a "
                           "copy of sensor_<side>_pose",
        "duplicate_rows": "tactile_<side>_is_new == False rows reuse the "
                          "previous estimate (forward fill, asserted exact)",
        "source_release": str(src),
        "force_calibration": sorted({d["force_calibration"] for d in diags}),
        "supersedes": "Exports before 2026-08-07 used a calibration that "
                      "applied pixel-unit weights to mm-unit features "
                      "(end-to-end rho 0.143, peak force 18.3 N). Those "
                      "newton values are void; these come from react_calib "
                      "(held-out rho 0.739, MAE 1.23 N, split by press "
                      "position).",
    }
    meta = dict(table.schema.metadata or {})
    meta[b"twm.force_export"] = json.dumps(header).encode()
    out_table = pa.Table.from_arrays(arrays, schema=pa.schema(fields, meta))

    dst = root / task / "meta" / date / f"{ep}.parquet"
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out_table, str(dst), compression="zstd")

    sidecar = {**header, "task": task, "date": date, "episode": ep,
               "rows": table.num_rows, "sides": diags,
               "columns": [f"force_{s}_{c}" for s in SIDES
                           for c in ("normal_n", "penetration_mm",
                                     "target_pose")]}
    (dst.parent / f"{ep}.force.json").write_text(json.dumps(sidecar, indent=2))
    return sidecar


def run_export(stiffness: float = STIFFNESS_N_PER_MM,
               root: Path = EXPORT_ROOT) -> dict:
    episodes = _episodes()
    have = {p.stem for p in FORCE_ROOT.glob("*/*/*.npz")}
    missing = [(t, d, e, s) for t, d, e in episodes for s in SIDES
               if f"{e}_{s}" not in have
               or not _force_npz(t, d, e, s).exists()]
    if missing:
        raise MissingForceFile(f"{len(missing)} sensor-sides without an npz: "
                               f"{missing[:5]}")
    out = []
    for task, date, ep in episodes:
        out.append(export_episode(task, date, ep, stiffness, root))
        print(f"  {task}/{date}/{ep}  rows={out[-1]['rows']}")
    manifest = {
        "generator": "twm.force_recovery.export_force_columns",
        "stiffness_n_per_mm": stiffness,
        "gel_thickness_mm": GEL_THICKNESS_MM,
        "n_episodes": len(out),
        "n_sensor_sides": 2 * len(out),
        "total_rows": sum(o["rows"] for o in out),
        "export_root": str(root),
        "episodes": [{"task": o["task"], "date": o["date"],
                      "episode": o["episode"], "rows": o["rows"],
                      "sides": o["sides"]} for o in out],
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "force_export_manifest.json").write_text(
        json.dumps(manifest, indent=2))
    return manifest


# --------------------------------------------------------------------------
# verification


def _pct(a: np.ndarray, q) -> tuple:
    return tuple(float(x) for x in np.percentile(a, q))


def verify(root: Path = EXPORT_ROOT) -> dict:
    """Re-read the exported parquets and check every claim, with numbers."""
    manifest = json.loads((root / "force_export_manifest.json").read_text())
    k = manifest["stiffness_n_per_mm"]
    files = sorted(root.glob("*/meta/*/*.parquet"))

    n_sides = aligned = 0
    forces, penets, gel_depths = [], [], []
    ep_stats, ident_dev, ident_rows = [], [], 0
    dup_rows = dup_held = 0
    quat_dev, roundtrip = [], []
    dirs = []
    for f in files:
        task, date, ep = f.parts[-4], f.parts[-2], f.stem
        t = pq.read_table(str(f))
        src = pq.read_table(str(STAGE_ROOT / task / "meta" / date /
                                f"{ep}.parquet"))
        assert t.num_rows == src.num_rows
        for side in SIDES:
            n_sides += 1
            with np.load(_force_npz(task, date, ep, side)) as d:
                raw = np.asarray(d["force_normal_n"], np.float64)
                gel_depths.append(np.asarray(d["max_depth_mm"], np.float64))
            col = np.asarray(t[f"force_{side}_normal_n"].to_numpy(), np.float64)
            if len(raw) == t.num_rows and np.array_equal(
                    raw.astype(np.float32), col.astype(np.float32)):
                aligned += 1
            pen = np.asarray(t[f"force_{side}_penetration_mm"].to_numpy(),
                             np.float64)
            pose = np.array(t[f"sensor_{side}_pose"].to_pylist(), np.float64)
            tgt = np.array(t[f"force_{side}_target_pose"].to_pylist(),
                           np.float64)
            is_new = np.asarray(t[f"tactile_{side}_is_new"].to_numpy(), bool)

            # DexForce consistency: an impedance controller at k sitting on
            # the observed pose with this target exerts exactly the force.
            roundtrip.append(float(np.abs(
                k * np.linalg.norm(tgt[:, :3] - pose[:, :3], axis=1) * 1000.0
                - col).max()))

            free = col == 0.0
            ident_rows += int(free.sum())
            if free.any():
                ident_dev.append(float(np.abs(tgt[free] - pose[free]).max()))
            quat_dev.append(float(np.abs(tgt[:, 3:] - pose[:, 3:]).max()))
            dup = np.zeros(len(col), bool)
            dup[1:] = ~is_new[1:]
            dup_rows += int(dup.sum())
            dup_held += int((dup[1:] & (col[1:] == col[:-1])).sum())

            forces.append(col)
            penets.append(pen)
            ep_stats.append({
                "id": f"{task}/{date}/{ep}_{side}",
                "p50": float(np.percentile(col, 50)),
                "p95": float(np.percentile(col, 95)),
                "max": float(col.max()),
                "no_contact_frac": float(free.mean()),
                "pen_p95_mm": float(np.percentile(pen, 95)),
                "pen_max_mm": float(pen.max()),
            })
            n_hat = press_direction(task, side, pose)
            dirs.append(direction_agreement(pose[:, :3] * 1000.0, n_hat, col))

    force = np.concatenate(forces)
    pen = np.concatenate(penets)
    depth = np.concatenate(gel_depths)
    strong = force > 0.5
    k_gel = force[strong] / np.maximum(depth[strong], 1e-6)

    used = [d for d in dirs if d.get("usable")]
    rise = np.array([d["v_dot_n_rise_mm"] for d in used])
    fall = np.array([d["v_dot_n_fall_mm"] for d in used])
    corr = np.array([d["corr_dforce_vdotn"] for d in used])

    report = {
        "stiffness_n_per_mm": k,
        "n_sensor_sides": n_sides,
        "alignment_pass": aligned,
        "alignment_rate": aligned / max(n_sides, 1),
        "total_force_samples": int(len(force)),   # rows x 2 sensor-sides
        "force_p50_p95_p99_max": _pct(force, [50, 95, 99, 100]),
        "no_contact_frac_global": float((force == 0.0).mean()),
        "no_contact_frac_per_side_p5_p50_p95": _pct(
            np.array([e["no_contact_frac"] for e in ep_stats]), [5, 50, 95]),
        "per_side_p50_min_med_max": _pct(
            np.array([e["p50"] for e in ep_stats]), [0, 50, 100]),
        "per_side_p95_min_med_max": _pct(
            np.array([e["p95"] for e in ep_stats]), [0, 50, 100]),
        "per_side_max_min_med_max": _pct(
            np.array([e["max"] for e in ep_stats]), [0, 50, 100]),
        "penetration_p50_p95_p99_max_mm": _pct(pen, [50, 95, 99, 100]),
        "penetration_over_gel_thickness_frac": float(
            (pen > GEL_THICKNESS_MM).mean()),
        "sides_with_p95_over_gel": int(sum(
            e["pen_p95_mm"] > GEL_THICKNESS_MM for e in ep_stats)),
        "gel_stiffness_implied_n_per_mm_p5_p25_p50_p75_p95": _pct(
            k_gel, [5, 25, 50, 75, 95]),
        # Minimum stiffness that keeps a penetration inside the gel. These are
        # LOWER BOUNDS, so they must be quoted rounded UP — the README used to
        # carry k_for_max_within_gel = 1.7141 as "k ~ 1.7", and 7.285/1.7 =
        # 4.285 mm, which is outside the 4.25 mm gel. Rounding a threshold to
        # the nearest value inverts what it asserts.
        #
        # The p95 over ALL rows is a poor number to choose a stiffness from:
        # most rows are free space, so it is mostly a percentile of zeros. The
        # contact-only figure is the one a user needs, and it is larger. The
        # two values were quoted here as "6.86 mm against 5.78 mm" until the
        # channel moved underneath them; `verify` prints both every run.
        "k_for_p95_within_gel": float(np.percentile(force, 95)
                                      / GEL_THICKNESS_MM),
        "k_for_contact_p95_within_gel": float(
            np.percentile(force[force > 0], 95) / GEL_THICKNESS_MM),
        # In MILLIMETRES, i.e. divided by the shipped stiffness. This held
        # `np.percentile(force, 95)` — newtons — and was only ever right
        # because k was 1.0 N/mm and the two were numerically equal. The
        # README quoted it as a penetration.
        "contact_p95_penetration_mm": float(
            np.percentile(force[force > 0], 95) / k),
        "contact_frac": float((force > 0).mean()),
        "k_for_max_within_gel": float(force.max() / GEL_THICKNESS_MM),
        # How much of the data sits ON the estimator's upper limit. The
        # isotonic stage clips at the hardest press in its calibration set,
        # so anything harder is recorded AT that value, not above it — the
        # max is a floor, and this fraction says how often it is one.
        # Computed here rather than quoted, because quoting it from a partial
        # batch is exactly how "0.27%" (48 of 72 sides) reached a draft when
        # the full set says 0.90%.
        "force_ceiling_n": float(force.max()),
        "force_at_ceiling_frac": float((force >= force.max() - 0.01).mean()),
        "identity_rows_checked": ident_rows,
        "identity_max_abs_dev": float(max(ident_dev)) if ident_dev else 0.0,
        "identity_pass": bool(ident_dev and max(ident_dev) == 0.0),
        "quaternion_max_abs_dev": float(max(quat_dev)),
        "roundtrip_max_abs_err_n": float(max(roundtrip)),
        "duplicate_rows": dup_rows,
        "duplicate_rows_held": dup_held,
        "duplicate_forward_fill_rate": dup_held / max(dup_rows, 1),
        "direction_sides_usable": len(used),
        "direction_rise_positive_frac": float((rise > 0).mean()),
        "direction_fall_negative_frac": float((fall < 0).mean()),
        "direction_rise_gt_fall_frac": float((rise > fall).mean()),
        "direction_corr_p25_p50_p75": _pct(corr, [25, 50, 75]),
        "direction_corr_positive_frac": float((corr > 0).mean()),
        "penetration_sweep": {
            str(kk): {
                "p95_mm": float(np.percentile(force, 95) / kk),
                "max_mm": float(force.max() / kk),
                "frac_over_gel": float((force / kk > GEL_THICKNESS_MM).mean()),
            } for kk in sorted({0.5, 1.0, 1.5, 3.0, 5.6, 15.4, k})},
    }
    (root / "force_export_verify.json").write_text(json.dumps(
        {**report, "per_side": ep_stats}, indent=2))
    return report


def _print(report: dict) -> None:
    k = report["stiffness_n_per_mm"]
    print(f"\n== export verification (k = {k} N/mm) ==")
    print(f"sensor-sides            : {report['n_sensor_sides']}")
    print(f"row alignment           : {report['alignment_pass']}/"
          f"{report['n_sensor_sides']} "
          f"({100 * report['alignment_rate']:.1f}%)")
    print(f"force samples           : {report['total_force_samples']} "
          f"(parquet rows x 2 sensor-sides)")
    print("force  p50/p95/p99/max  : %.4f / %.3f / %.3f / %.3f N"
          % report["force_p50_p95_p99_max"])
    print("per-side p50 min/med/max: %.4f / %.4f / %.3f N"
          % report["per_side_p50_min_med_max"])
    print("per-side p95 min/med/max: %.3f / %.3f / %.3f N"
          % report["per_side_p95_min_med_max"])
    print("per-side max min/med/max: %.3f / %.3f / %.3f N"
          % report["per_side_max_min_med_max"])
    nc = tuple(100 * x for x in report["no_contact_frac_per_side_p5_p50_p95"])
    print("no-contact rows         : %.2f%% (per-side p5/p50/p95 "
          "%.1f/%.1f/%.1f%%)"
          % (100 * report["no_contact_frac_global"], *nc))
    print("penetration p50/p95/p99/max: %.4f / %.3f / %.3f / %.3f mm"
          % report["penetration_p50_p95_p99_max_mm"])
    print(f"  rows over gel thickness {GEL_THICKNESS_MM} mm: "
          f"{100 * report['penetration_over_gel_thickness_frac']:.2f}%  "
          f"sides with p95 over it: {report['sides_with_p95_over_gel']}"
          f"/{report['n_sensor_sides']}")
    print("  gel stiffness implied by the estimator's own indentation "
          "(F / max_depth_mm, F>0.5 N):")
    print("    p5/p25/p50/p75/p95 = %.1f / %.1f / %.1f / %.1f / %.1f N/mm"
          % report["gel_stiffness_implied_n_per_mm_p5_p25_p50_p75_p95"])
    print(f"  k so that p95 penetration <= gel thickness: "
          f"{report['k_for_p95_within_gel']:.2f} N/mm")
    print(f"  k so that max penetration <= gel thickness: "
          f"{report['k_for_max_within_gel']:.2f} N/mm")
    for kk, v in report["penetration_sweep"].items():
        print(f"    k={kk:>5} N/mm -> p95 {v['p95_mm']:6.3f} mm, "
              f"max {v['max_mm']:7.3f} mm, "
              f"{100 * v['frac_over_gel']:5.2f}% over gel")
    print(f"identity (no-contact)   : {report['identity_rows_checked']} rows, "
          f"max |target - observed| = {report['identity_max_abs_dev']:.1e} -> "
          f"{'PASS' if report['identity_pass'] else 'FAIL'}")
    print(f"quaternion untouched    : max dev "
          f"{report['quaternion_max_abs_dev']:.1e}")
    print(f"roundtrip k*|dp| == F   : max err "
          f"{report['roundtrip_max_abs_err_n']:.2e} N")
    print(f"duplicate-frame ffill   : {report['duplicate_rows_held']}/"
          f"{report['duplicate_rows']} "
          f"({100 * report['duplicate_forward_fill_rate']:.2f}%)")
    print(f"press direction         : {report['direction_sides_usable']} "
          f"sides usable; v.n>0 on rise "
          f"{100 * report['direction_rise_positive_frac']:.1f}%, "
          f"v.n<0 on release "
          f"{100 * report['direction_fall_negative_frac']:.1f}%, "
          f"rise>fall {100 * report['direction_rise_gt_fall_frac']:.1f}%")
    print("  corr(dF, v.n) p25/p50/p75 = %.3f / %.3f / %.3f, positive on "
          "%.1f%% of sides" % (*report["direction_corr_p25_p50_p75"],
                               100 * report["direction_corr_positive_frac"]))


def digest(root: Path = EXPORT_ROOT) -> str:
    """Content hash of every exported parquet -- idempotency check."""
    h = hashlib.sha256()
    for f in sorted(root.glob("*/meta/*/*.parquet")):
        h.update(f.read_bytes())
    return h.hexdigest()


def _gate(report: dict) -> int:
    """Turn the verification report into an EXIT CODE.

    `verify` computed all of this already and `_print` showed it, but `main`
    returned None either way — so a run with `identity_pass` False, or with a
    commanded target displaced further past the surface than the gel can be
    compressed, exited 0 and read as a clean export. The penetration case was
    not hypothetical: at k = 1 N/mm it covered 14.98% of rows and 49 of 72
    sides, printed on every run, and nothing downstream of the print cared.
    """
    fails = []
    if not report.get("identity_pass", False):
        fails.append("no-contact rows do not leave the pose identical")
    if not report.get("alignment_pass", False):
        fails.append(f"row alignment {report.get('alignment_rate', 0)*100:.1f}%")
    frac = report.get("penetration_over_gel_thickness_frac", 0.0)
    if frac > 0:
        fails.append(
            f"{frac*100:.2f}% of rows command a target more than "
            f"{GEL_THICKNESS_MM} mm past the surface — the gel cannot be "
            f"compressed that far, so raise the stiffness "
            f"(dexforce.STIFFNESS_N_PER_M)")
    err = report.get("roundtrip_max_abs_err_n", 0.0)
    if err > 1e-6:
        fails.append(f"k*|dp| != F, max err {err:.2e} N")
    for f in fails:
        print(f"  FAIL: {f}")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["export", "verify", "digest"])
    ap.add_argument("--stiffness", type=float, default=STIFFNESS_N_PER_MM,
                    help="N/mm used for penetration = force / k")
    ap.add_argument("--root", type=Path, default=EXPORT_ROOT)
    args = ap.parse_args()
    if args.command == "export":
        m = run_export(args.stiffness, args.root)
        print(f"\nwrote {m['n_episodes']} episodes / "
              f"{m['n_sensor_sides']} sensor-sides / {m['total_rows']} rows "
              f"to {args.root}")
        rep = verify(args.root)
        _print(rep)
        return _gate(rep)
    elif args.command == "verify":
        rep = verify(args.root)
        _print(rep)
        return _gate(rep)
    print(digest(args.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
