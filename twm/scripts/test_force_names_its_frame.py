"""Every force value says which tactile frame it came from, and it is checkable.

Force and the tactile columns are on the same frame today — measured, 24
side-episodes, cross-correlation lag 0 in every one. But nothing in the data
RECORDS that. The agreement comes from two modules independently evaluating
`trim + row + LEGACY_SHIFT` and happening to get the same number.

That is the exact arrangement that failed three times in one session:

  * `row_for_h5_frame` subtracted the lag once too often, so the published
    previews put the force disc half a second from its own tactile tile
  * `verify_force_overlay` hard-coded 15 inline and rendered frames half a
    second from the ones it named — and passed, because its two errors
    cancelled
  * the unit test asserted the wrong mapping, with a docstring naming the
    exact failure it was locking in

Each time the index was re-derived at the point of use. A reader of the
dataset cannot check any of it without reading our code.

`force_<side>_source_frame` makes the pairing a property OF THE DATA. The
checks below are deliberately not "does the formula agree with itself":

  1  the column exists, is integer, and repeats exactly where the tactile
     frame repeats (so "these four rows share one image" is stated, not
     inferred from is_new)
  2  END TO END: re-read gelsight[source_frame] out of the source H5,
     recompute the force through the deployed estimator, and require the
     stored value back to the bit. A formula cannot satisfy this by agreeing
     with another copy of itself; only the actually-read frame can.

    python scripts/test_force_names_its_frame.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                                              # noqa: E402

RESULTS: list[tuple[bool, str, str]] = []
TASK, DATE, EP, SIDE = "motherboard", "2026-05-10", "episode_004", "left"


def check(ok: bool, name: str, evidence: str) -> None:
    RESULTS.append((bool(ok), name, evidence))


def main() -> int:
    import h5py

    from force_recovery.run_episode import DATA_ROOT, OUT_ROOT

    npz_path = OUT_ROOT / TASK / DATE / f"{EP}_{SIDE}.npz"
    if not npz_path.exists():
        check(False, "the npz names its source frame",
              f"UNVERIFIED: {npz_path} absent")
        return 1
    z = np.load(npz_path, allow_pickle=True)

    key = "source_frame"
    if key not in z:
        check(False, "the npz names its source frame",
              f"no `{key}` array — the pairing is still implicit")
        check(False, "the named frame reproduces the stored force",
              "not attempted: no source frame to read")
        _report()
        return 1

    src = np.asarray(z[key])
    force = np.asarray(z["force_normal_n"], float)
    check(np.issubdtype(src.dtype, np.integer) and len(src) == len(force),
          "the npz names its source frame",
          f"{len(src)} rows, dtype {src.dtype}, "
          f"range {int(src.min())}..{int(src.max())}")

    # 1 — ONE DIRECTION ONLY. Same image must give the same number; the
    # converse is false and I wrote it that way first. Distinct frames
    # routinely produce identical forces — every no-contact frame gives
    # exactly 0.0 — so `force repeats <=> frame repeats` fails on 399 of 2405
    # rows of a perfectly correct episode. The invariant worth asserting is
    # that a held row is held from the frame it names.
    same_src = src[1:] == src[:-1]
    same_f = force[1:] == force[:-1]
    viol = int((same_src & ~same_f).sum())
    check(viol == 0, "a repeated frame gives a repeated force",
          f"{int(same_src.sum())} repeated frames, {viol} of them changed the "
          f"force (must be 0); {int((~same_src & same_f).sum())} distinct "
          f"frames share a force, which is allowed — 0 N in free space")

    # 2 — THE ONE THAT CANNOT BE FAKED BY A FORMULA.
    from force_recovery.lut_calibration import crop
    from force_recovery.react_calib import fit, force_stages

    predict = fit(report=False)
    ref_rows = np.asarray(z["reference_rows"])
    with h5py.File(str(DATA_ROOT / TASK / DATE / f"{EP}.h5"), "r") as f:
        frames = f[f"gelsight/{SIDE}/frames"]        # tactile-lag-exempt
        ref = np.median(np.stack(
            [crop(frames[int(src[int(r)])]).astype(np.float32)
             for r in ref_rows[:12]]), 0)
        rng = np.random.default_rng(0)
        pick = rng.choice(np.flatnonzero(force > 0.05), size=8, replace=False)
        bad = []
        for row in pick:
            img = crop(frames[int(src[int(row)])]).astype(np.float32)
            again = float(predict(force_stages(img, ref)))
            if abs(again - float(force[int(row)])) > 1e-6:
                bad.append((int(row), again, float(force[int(row)])))
    check(not bad, "the named frame reproduces the stored force",
          f"{len(pick) - len(bad)}/{len(pick)} rows reproduce to 1e-6"
          + (f"; first mismatch row {bad[0][0]}: {bad[0][1]:.6f} vs "
             f"{bad[0][2]:.6f}" if bad else ""))

    _report()
    return 1 if sum(not ok for ok, _, _ in RESULTS) else 0


def _report() -> None:
    width = max(len(x) for _, x, _ in RESULTS)
    print()
    for ok, name, ev in RESULTS:
        print(f"  [{'ok' if ok else 'FAIL'}] {name:<{width}}  {ev}")
    print(f"\nforce names its frame: {len(RESULTS)} checks, "
          f"{sum(not ok for ok, _, _ in RESULTS)} failing")


if __name__ == "__main__":
    raise SystemExit(main())
