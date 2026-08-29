"""The conformance checker must fail on every way a user gets this wrong.

A validator that only passes is worthless. This asserts the opposite half:
feed it each classic mistake and it has to say so, by name, with a number.

The mistakes are not hypothetical. Every one of them happened in this repo:
poses in one up-axis convention with extrinsics in the other (153 px, all
self-checks green); a doc that said "quat wxyz" over scalar-last code;
millimetres handed to a function documented in metres.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                             # noqa: E402

from react_paths import release_root                           # noqa: E402

RESULTS: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, evidence: str) -> None:
    RESULTS.append((bool(ok), name, evidence))


def main() -> int:
    try:
        from react_toolbox import conformance
    except Exception as ex:
        check(False, "react_toolbox.conformance exists",
              f"{type(ex).__name__}: {ex}")
        return _report()

    import json
    import pyarrow.parquet as pq
    from react_toolbox.frames import convert_poses

    # Stage the layout a USER downloads: calibration and the declaring meta/
    # in one tree. Locally they live apart (calibration in the release,
    # twm.world_frame in the force export), and checking against either half
    # alone would be testing a shape nobody receives.
    import shutil
    from react_toolbox.staging import staging_dir
    from react_paths import force_meta
    src = release_root("motherboard")
    root = staging_dir() / "motherboard"
    (root).mkdir(parents=True)
    shutil.copytree(src / "calibration", root / "calibration")
    shutil.copytree(force_meta("motherboard"), root / "meta")
    shutil.copy(src / "episodes.jsonl", root / "episodes.jsonl")
    key = json.loads(
        [l for l in (root / "episodes.jsonl").read_text().splitlines()
         if l.strip()][0])["episode"]
    date, ep = key.split("/")
    t = pq.read_table(str(root / "meta" / date / f"{ep}.parquet"),
                      columns=["sensor_left_pose"]).to_pydict()
    P = np.asarray([x for x in t["sensor_left_pose"]], float)
    P = P[np.isfinite(P).all(1) & (np.linalg.norm(P[:, 3:], axis=1) > 0.5)]

    def run(poses, **kw):
        return conformance.check_poses(poses, task="motherboard", date=date,
                                       episode=ep, side="left",
                                       release=root, **kw)

    # 1 — the published poses, used correctly, must pass
    r = run(P)
    check(r.ok and r.n_checks >= 4,
          "the published poses pass their own conformance check",
          f"{r.n_checks} checks, {len(r.failures)} failing; "
          f"worst fingerprint error {r.detail.get('fingerprint_px', float('nan')):.3f} px")

    # 2 — millimetres instead of metres
    r = run(P * np.array([1000, 1000, 1000, 1, 1, 1, 1.0]))
    check(not r.ok and any("unit" in f for f in r.failures),
          "positions in millimetres are rejected",
          f"failures: {r.failures}")

    # 3 — quaternion read as wxyz
    Q = P.copy()
    Q[:, 3:7] = P[:, [6, 3, 4, 5]]
    r = run(Q)
    check(not r.ok, "a wxyz quaternion is rejected",
          f"failures: {r.failures}; fingerprint "
          f"{r.detail.get('fingerprint_px', float('nan')):.1f} px")

    # 4 — the OTHER up-axis convention
    r = run(convert_poses(P, to_zup=False))
    check(not r.ok and any("frame" in f or "up" in f for f in r.failures),
          "poses in the recorded Y-up convention are rejected",
          f"failures: {r.failures}; fingerprint "
          f"{r.detail.get('fingerprint_px', float('nan')):.1f} px")

    # 5 — unnormalised quaternions
    U = P.copy(); U[:, 3:7] *= 1.4
    r = run(U)
    check(not r.ok and any("quaternion" in f for f in r.failures),
          "unnormalised quaternions are rejected",
          f"failures: {r.failures}")

    # 6 — the report names what each mistake would have read, so a reader can
    #     see the check has teeth without running it themselves
    r = run(P)
    ctrl = r.detail.get("negative_controls") or {}
    check(isinstance(ctrl, dict) and len(ctrl) >= 2
          and all(v > 20 for v in ctrl.values()),
          "a passing report still shows what each mistake would read",
          f"negative controls (px): "
          f"{ {k: round(v, 1) for k, v in ctrl.items()} }")

    # 7 — a subset must be REFUSED, not silently reported as a frame error.
    #     Measured: the first 400 rows of this episode read 10.4 px against
    #     the episode fingerprint, which is above tolerance and looks exactly
    #     like a broken world frame.
    r = run(P[:400])
    check(not r.ok and "row-coverage" in r.failures,
          "a subset of rows is refused rather than misread as a frame error",
          f"failures: {r.failures}; fingerprint would have read "
          f"{r.detail.get('fingerprint_px', float('nan')):.1f} px")

    # 8 — but a subset the caller identifies is checked row-wise, and passes
    idx = np.arange(400)
    r = run(P[idx], rows=idx)
    check(r.ok, "an identified subset is checked row-wise and passes",
          f"worst position difference "
          f"{r.detail.get('max_position_diff_m', float('nan')):.2e} m")

    return _report()


def _report() -> int:
    w = max(len(x) for _, x, _ in RESULTS)
    print()
    for ok, name, ev in RESULTS:
        print(f"  [{'ok' if ok else 'FAIL'}] {name:<{w}}  {ev}")
    n = sum(not x for x, _, _ in RESULTS)
    print(f"\nconformance: {len(RESULTS)} checks, {n} failing")
    return 1 if n else 0


if __name__ == "__main__":
    raise SystemExit(main())
