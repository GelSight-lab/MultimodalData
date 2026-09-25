"""Check that YOUR poses are in the frame this release publishes.

Every other test in this project validates the dataset. This one validates
your use of it, which is the half that actually goes wrong: the data can be
perfect and still be read in millimetres, or with `w` first, or paired with
extrinsics from the wrong convention. None of those raise. They shift every
projection by tens to hundreds of pixels and leave every self-consistency
check green, because the same wrong assumption draws and re-verifies.

    from react_toolbox.conformance import check_poses
    r = check_poses(my_poses, task="motherboard", date="2026-05-10",
                    episode="episode_000", side="left", release=ROOT)
    print(r)
    assert r.ok

or, without writing any code:

    python -m react_toolbox.conformance --release /path/to/data \\
        --task motherboard --episode 2026-05-10/episode_000

A passing report also prints what each mistake WOULD have read. A validator
that only ever says "ok" teaches you nothing about whether it can say
anything else.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Distances are in pixels of a 640x480 view unless stated. The floor is set by
# the rig: camera reprojection rmse is 3.6-5.7 px and the gel centre adds
# ~3.8 px, so agreement inside ~6 px is as good as this hardware can tell.
FINGERPRINT_TOL_PX = 6.0
_M_PER_MM = 1e-3


@dataclass
class Report:
    ok: bool = True
    n_checks: int = 0
    failures: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    detail: dict = field(default_factory=dict)

    def _add(self, ok: bool, key: str, note: str):
        self.n_checks += 1
        self.notes.append(("ok  " if ok else "FAIL") + "  " + note)
        if not ok:
            self.ok = False
            self.failures.append(key)

    def __str__(self):
        head = "conformance: PASS" if self.ok else \
            f"conformance: FAIL ({', '.join(self.failures)})"
        return head + "\n" + "\n".join("  " + n for n in self.notes)


def _decl(release: Path, task: str, date: str, episode: str) -> dict:
    """The episode's world-frame declaration, from its parquet metadata."""
    import pyarrow.parquet as pq
    for base in (Path(release) / "meta", Path(release) / task / "meta"):
        f = base / date / f"{episode}.parquet"
        if f.exists():
            md = pq.read_schema(str(f)).metadata or {}
            raw = md.get(b"twm.world_frame")
            if raw:
                return json.loads(raw.decode())
    raise FileNotFoundError(
        f"no world-frame declaration for {task}/{date}/{episode} under "
        f"{release}. It lives in the parquet's `twm.world_frame` metadata; "
        f"the force export carries it.")


def _episode_poses(release: Path, task: str, date: str, episode: str,
                   side: str):
    import pyarrow.parquet as pq
    for base in (Path(release) / "meta", Path(release) / task / "meta"):
        f = base / date / f"{episode}.parquet"
        if f.exists():
            t = pq.read_table(str(f),
                              columns=[f"sensor_{side}_pose"]).to_pydict()
            return np.asarray([x for x in t[f"sensor_{side}_pose"]], float)
    return None


def check_poses(poses7, *, task: str, date: str, episode: str,
                side: str = "left", release, tol_px: float = FINGERPRINT_TOL_PX,
                rows=None) -> Report:
    """Battery of checks on a user-supplied (N, 7) pose array.

    Pass the episode's WHOLE pose array. The stored fingerprint is a median
    over its rows, so a subset moves the median by real motion and reads as a
    frame error: measured on one episode, the first 400 rows give 10.4 px and
    a random 400 give 3.9, against 0.000 for all 6,890. That is a false alarm
    big enough to send someone hunting a bug that is not there.

    If you only have some rows, pass their indices as `rows=`; the check then
    compares your poses to the release's own for those same rows, which is
    stronger than the fingerprint and works on any subset.
    """
    from .world_frame import verify_world_frame
    from .frames import convert_poses

    r = Report()
    P = np.asarray(poses7, float)
    if P.ndim != 2 or P.shape[1] != 7:
        r._add(False, "shape", f"expected (N, 7), got {P.shape}")
        return r
    root = Path(release)
    task_root = root if (root / "calibration").is_dir() else root / task

    # --- quaternion norm ------------------------------------------------
    n = np.linalg.norm(P[:, 3:7], axis=1)
    good = np.isfinite(n) & (n > 0.5)
    worst = float(np.abs(n[good] - 1).max()) if good.any() else np.inf
    r._add(worst < 1e-3, "quaternion-norm",
           f"quaternion norm deviates by at most {worst:.2e} from 1 "
           f"(a scaled or unnormalised quaternion silently scales the "
           f"rotation)")

    # --- units ----------------------------------------------------------
    # The rig's working volume is a table: metres give |p| of order 1, and
    # millimetres of order 1000. Nothing here is 1000 m from the origin.
    med = float(np.median(np.linalg.norm(P[good, :3], axis=1)))
    r._add(med < 20.0, "units",
           f"median |position| = {med:.3f}; expected metres (order 1), not "
           f"millimetres (order 1000)")
    r.detail["median_abs_position"] = med

    # --- row coverage ---------------------------------------------------
    ref_all = _episode_poses(root, task, date, episode, side)
    if ref_all is not None and rows is None:
        valid = np.isfinite(ref_all).all(1) & (
            np.linalg.norm(ref_all[:, 3:7], axis=1) > 0.5)
        want = int(valid.sum())
        r._add(len(P) == want, "row-coverage",
               f"{len(P)} poses given, episode has {want} valid rows. The "
               f"fingerprint is a MEDIAN over the episode; a subset shifts it "
               f"by real motion (measured: 10.4 px from the first 400 rows, "
               f"0.000 from all of them). Pass rows= to check a subset.")

    # --- row-wise comparison when the caller named their rows -----------
    if rows is not None and ref_all is not None:
        idx = np.asarray(rows, int)
        if len(idx) != len(P):
            r._add(False, "rows", f"{len(idx)} indices for {len(P)} poses")
        else:
            d = np.abs(P[:, :3] - ref_all[idx][:, :3]).max()
            r.detail["max_position_diff_m"] = float(d)
            r._add(d < 1e-9, "rows-match",
                   f"worst position difference from the release's own rows "
                   f"{d:.2e} m")

    # --- the frame itself, against the release's own fingerprint --------
    # Skipped when the caller named their rows: the row-wise comparison above
    # already settles it, and the episode-level fingerprint would fail on any
    # subset for reasons that have nothing to do with the frame.
    decl = _decl(root, task, date, episode)
    if rows is not None and "rows-match" not in r.failures and \
            "rows" not in r.failures:
        r.notes.append("      fingerprint skipped: rows were named, so the "
                       "row-wise comparison above is the stronger check")
        r.detail["negative_controls"] = {}
        return r
    try:
        px = float(verify_world_frame(P[good], side, task_root, decl))
    except Exception as ex:
        r._add(False, "frame", f"could not verify the frame: "
                               f"{type(ex).__name__}: {ex}")
        return r
    r.detail["fingerprint_px"] = px
    r._add(px <= tol_px, "frame",
           f"worst-camera fingerprint error {px:.2f} px "
           f"(tolerance {tol_px:g}; the rig's own floor is ~6)")

    # --- negative controls ----------------------------------------------
    # Printed on a PASS too. A reader should be able to see the check can
    # fail without having to make it fail themselves.
    ctrl = {}
    try:
        ctrl["other up-axis"] = float(
            verify_world_frame(convert_poses(P[good], to_zup=False), side,
                               task_root, decl))
    except Exception:
        pass
    try:
        Q = P[good].copy()
        Q[:, 3:7] = P[good][:, [6, 3, 4, 5]]
        ctrl["quaternion as wxyz"] = float(
            verify_world_frame(Q, side, task_root, decl))
    except Exception:
        pass
    r.detail["negative_controls"] = ctrl
    if ctrl:
        r.notes.append("      would read: " + ", ".join(
            f"{k} {v:.0f} px" for k, v in ctrl.items()))
    return r


def _main(argv=None) -> int:
    import argparse
    import pyarrow.parquet as pq

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--release", required=True,
                    help="the release tree (holds episodes.jsonl / meta / calibration)")
    ap.add_argument("--task", default="motherboard")
    ap.add_argument("--episode", required=True, help="<date>/<episode>")
    ap.add_argument("--side", default="left", choices=("left", "right"))
    ap.add_argument("--poses", default=None,
                    help=".npy of your own (N, 7) poses; omit to check the "
                         "release's own, which is the self-test")
    a = ap.parse_args(argv)
    date, ep = a.episode.split("/")
    root = Path(a.release)

    if a.poses:
        P = np.load(a.poses)
    else:
        base = root / "meta" if (root / "meta").is_dir() else root / a.task / "meta"
        t = pq.read_table(str(base / date / f"{ep}.parquet"),
                          columns=[f"sensor_{a.side}_pose"]).to_pydict()
        P = np.asarray([x for x in t[f"sensor_{a.side}_pose"]], float)
        P = P[np.isfinite(P).all(1) & (np.linalg.norm(P[:, 3:], axis=1) > .5)]

    r = check_poses(P, task=a.task, date=date, episode=ep, side=a.side,
                    release=root)
    print(r)
    return 0 if r.ok else 1


if __name__ == "__main__":
    raise SystemExit(_main())
