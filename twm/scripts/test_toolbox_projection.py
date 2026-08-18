"""The published toolbox projects the gel centre, and says so when it cannot.

`toolbox/calibration.py` is what a user of yxma/React runs; `quickstart.md`
shows exactly this call. It returned the RIGID-BODY ORIGIN instead of the gel
centre on every episode of both tasks, because `load_calibration` looked for
three keys — `T_gel_to_rigid`, `T`, `gel_center_mm` — none of which exist in
the published files. The real key is `gel_center_in_rigid_mm`. All three `.get`
calls missed and the last one fell back to `[0, 0, 0]`.

A default of zero is the worst possible one here: it is a VALID-LOOKING offset
(it means "the gel centre is the rigid-body origin") so nothing downstream can
tell it apart from a real answer. Measured against the correct centre on
motherboard/2026-05-11/episode_003, median over the episode:

    left camera    35.8 px      (p90 44.5)
    middle camera  20.8 px      (p90 36.5)
    right camera   28.0 px      (p90 37.6)

against a calibration rmse of 4.75 mm — about 3 px at this depth. Seven to
twelve times the rig's own error, and still shaped like a slightly
miscalibrated rig rather than like a bug. That is the third time this project
has shipped a defect with that shape.

    python scripts/test_toolbox_projection.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                              # noqa: E402

RESULTS: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, evidence: str) -> None:
    RESULTS.append((bool(ok), name, evidence))


def main() -> int:
    import json
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "react_toolbox"))
    import react_toolbox.calibration as C

    # The release tree assembles `calibration/` only at publish time, so point
    # at the epoch directory `calib_epoch` owns. My first version used the
    # release path, found no files, and reported the bug's own symptom
    # ([0,0,0]) for an entirely different reason — a test that fails for the
    # wrong reason is indistinguishable from one that fails for the right one.
    cdir = Path("/home/yxma/MultimodalData/twm/calibration/result backup")
    root = cdir.parent
    task = cdir.name

    # 1 — the gel centre must be the measured one, not a zero that looks valid
    # load_calibration expects <root>/calibration/, so hand it the parent of
    # the epoch dir under a temporary name that matches that layout.
    import tempfile, shutil as _sh
    stage = Path(tempfile.mkdtemp())
    _sh.copytree(cdir, stage / "calibration")
    cal = C.load_calibration(stage)
    truth = {s: np.asarray(json.loads(
        (cdir / f"T_gel_to_rigid_{s}.json").read_text())["gel_center_in_rigid_mm"],
        float) for s in ("left", "right")}
    bad = []
    for s in ("left", "right"):
        got = np.asarray(cal.get(f"gel_{s}", [0, 0, 0]), float)
        if not np.allclose(got, truth[s], atol=1e-6):
            bad.append(f"gel_{s}: {np.round(got,1)} != {np.round(truth[s],1)}")
    check(not bad, "load_calibration returns the measured gel centre",
          f"{2-len(bad)}/2 sides correct" + (f"; {bad}" if bad else ""))

    # 2 — AND IT REFUSES rather than substituting a plausible zero. A missing
    # key must raise; the whole defect was a fallback nobody could see.
    import tempfile, shutil
    tmp = Path(tempfile.mkdtemp()) / "calibration"
    shutil.copytree(cdir, tmp)
    d = json.loads((tmp / "T_gel_to_rigid_left.json").read_text())
    d.pop("gel_center_in_rigid_mm")
    (tmp / "T_gel_to_rigid_left.json").write_text(json.dumps(d))
    try:
        C.load_calibration_from(tmp) if hasattr(C, "load_calibration_from") \
            else C.load_calibration(tmp.parent)
        check(False, "a missing gel centre raises",
              "returned a value for a file with no gel centre")
    except Exception as exc:                                    # noqa: BLE001
        check("gel_center_in_rigid_mm" in str(exc) or "gel" in str(exc).lower(),
              "a missing gel centre raises",
              f"{type(exc).__name__}: {str(exc)[:70]}")

    width = max(len(x) for _, x, _ in RESULTS)
    print()
    for ok, name, ev in RESULTS:
        print(f"  [{'ok' if ok else 'FAIL'}] {name:<{width}}  {ev}")
    n = sum(not ok for ok, _, _ in RESULTS)
    print(f"\ntoolbox projection: {len(RESULTS)} checks, {n} failing")
    return 1 if n else 0


if __name__ == "__main__":
    raise SystemExit(main())
