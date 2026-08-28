"""Every task, every episode, every file: +z is up. Measured, then declared.

The release rotated from OptiTrack's Y-up to Z-up. That conversion was run on
`motherboard` and NOT on `pushT`, so the published dataset shipped two
conventions at once with nothing saying so: pushT's calibration and
episodes.jsonl simply had no `up_axis` key, which reads as "old file" only if
you already know to look.

Both halves are checked here, in this order:

  MEASURED FIRST. The sensors press into the table, so `R(q) @ gel_axis` --
  the pressing direction in world coordinates -- points DOWN whenever there is
  real force on the gel. That is physics; it does not care what any file says.

  DECLARED SECOND, and only then compared. A file that agrees with itself
  proves nothing: for a whole day this project had poses in one convention and
  extrinsics in the other, every self-consistency check green, and every
  overlay 153 px wrong.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                             # noqa: E402
import pyarrow.parquet as pq                                   # noqa: E402

from react_paths import force_meta, release_root, testset_root  # noqa: E402
from react_toolbox.frames import UP_AXIS_RECORDED               # noqa: E402

RESULTS: list[tuple[bool, str, str]] = []
UP = "xyz".index(UP_AXIS_RECORDED)


def check(ok: bool, name: str, evidence: str) -> None:
    RESULTS.append((bool(ok), name, evidence))


def _tasks() -> list[str]:
    return sorted(p.name for p in release_root().iterdir()
                  if (p / "episodes.jsonl").exists())


def main() -> int:
    argparse.ArgumentParser().parse_args()
    from force_recovery.dexforce import gel_axis
    from scipy.spatial.transform import Rotation

    tasks = _tasks()

    # 1 — MEASURED. Which way is down, according to the sensors themselves?
    measured, weak = {}, []
    for task in tasks:
        files = sorted(glob.glob(str(force_meta(task) / "*" / "*.parquet")))
        cols = pq.read_schema(files[0]).names if files else []
        if "force_left_normal_n" not in cols:
            weak.append(task)
            continue
        press = []
        for f in files:
            t = pq.read_table(f, columns=["sensor_left_pose",
                                          "force_left_normal_n"]).to_pydict()
            P = np.asarray([x for x in t["sensor_left_pose"]], float)
            F = np.asarray(t["force_left_normal_n"], float)
            m = (np.isfinite(P).all(1) & (np.linalg.norm(P[:, 3:], axis=1) > .5)
                 & (F > 3))
            if m.sum() < 30:
                continue
            R = Rotation.from_quat(P[m, 3:7]).as_matrix()
            press.append(np.median(np.einsum("nij,j->ni", R,
                                             gel_axis(task, "left")), 0))
        v = np.median(np.asarray(press), 0)
        measured[task] = v / np.linalg.norm(v)
    bad = {t: np.round(v, 3).tolist() for t, v in measured.items()
           if int(np.argmax(abs(v))) != UP or v[UP] >= 0}
    check(measured and not bad and not weak,
          f"the sensors press along -{UP_AXIS_RECORDED} in every task",
          "; ".join(f"{t}: {np.round(v, 3).tolist()}"
                    for t, v in measured.items())
          + (f"; NO FORCE CHANNEL, unmeasured: {weak}" if weak else ""))

    # 2 — DECLARED: calibration
    cbad = []
    for task in tasks:
        for f in sorted((release_root(task) / "calibration")
                        .glob("T_mocap_to_cam_*.json")):
            if (json.loads(f.read_text()).get("up_axis") or "y") != UP_AXIS_RECORDED:
                cbad.append(f"{task}/{f.name}")
    check(not cbad, "every camera calibration declares the convention",
          f"{len(tasks)} tasks, all cameras say up_axis={UP_AXIS_RECORDED!r}"
          if not cbad else f"missing or wrong: {cbad[:6]}")

    # 3 — DECLARED: episodes.jsonl
    ebad, n_ep = [], 0
    for task in tasks:
        for line in (release_root(task) / "episodes.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            n_ep += 1
            r = json.loads(line)
            if (r.get("up_axis") or "y") != UP_AXIS_RECORDED:
                ebad.append(f"{task}/{r.get('episode', '?')}")
    check(not ebad, "every episode record declares the convention",
          f"{n_ep} episode records across {len(tasks)} tasks"
          if not ebad else f"{len(ebad)} without it, e.g. {ebad[:4]}")

    # 4 — DECLARED: the parquet metadata, in BOTH published trees
    pbad, n_pq = [], 0
    # force_meta() falls back to the plain release for an unknown task, so
    # asking it about a made-up name collapsed both roots into one and this
    # check counted zero declarations -- and said so, which is the only
    # reason it was caught.
    roots = {release_root()}
    for t in tasks:
        roots.add(force_meta(t).parent.parent)
    for root in sorted(roots):
        for task in tasks:
            base = root / task / "meta"
            for f in sorted(glob.glob(str(base / "*" / "*.parquet"))):
                md = pq.read_schema(f).metadata or {}
                raw = md.get(b"twm.world_frame")
                if raw is None:
                    continue
                n_pq += 1
                if (json.loads(raw.decode()).get("up_axis") or "y") != UP_AXIS_RECORDED:
                    pbad.append(str(Path(f).relative_to(root)))
    check(n_pq > 0 and not pbad,
          "every parquet's world-frame declaration says the same",
          f"{n_pq} declarations across both published trees"
          if not pbad else f"{len(pbad)} disagree, e.g. {pbad[:4]}")

    # 5 — the declaration is the one the data actually has
    mism = [t for t, v in measured.items()
            if (json.loads((release_root(t) / "calibration"
                            / "T_mocap_to_cam_middle.json").read_text())
                .get("up_axis") or "y") != "xyz"[int(np.argmax(abs(v)))]]
    check(not mism, "what the files say is what the sensors measured",
          f"{len(measured)} tasks agree"
          if not mism else f"declared != measured for {mism}")

    # 6 — the artefacts built from it
    ts = testset_root() / "calibration" / "T_mocap_to_cam_middle.json"
    sim = release_root().parent / "sim" / "sim.json"
    extra = []
    if ts.exists() and (json.loads(ts.read_text()).get("up_axis")
                        != UP_AXIS_RECORDED):
        extra.append("probe test set")
    if sim.exists() and json.loads(sim.read_text()).get("up_axis") != UP_AXIS_RECORDED:
        extra.append("simulator")
    check(not extra, "the derived artefacts carry the convention too",
          f"probe test set and simulator both {UP_AXIS_RECORDED}-up"
          if not extra else f"still on the old convention: {extra}")

    w = max(len(x) for _, x, _ in RESULTS)
    print()
    for ok, name, ev in RESULTS:
        print(f"  [{'ok' if ok else 'FAIL'}] {name:<{w}}  {ev}")
    n = sum(not x for x, _, _ in RESULTS)
    print(f"\nup axis everywhere: {len(RESULTS)} checks, {n} failing")
    return 1 if n else 0


if __name__ == "__main__":
    raise SystemExit(main())
