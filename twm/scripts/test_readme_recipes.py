"""Run the dataset README's own code against the published data.

A README is the one file nobody re-derives, and example code in it is read as
a promise. This executes the recipes the force section documents — loading the
columns, the free-space identity, and recomputing target poses at a different
stiffness — against real parquet from `yxma/React`, and checks the claims made
around them.

It reads the published files over HfFileSystem rather than the local staging
tree, because the README describes what a user downloads, and those two have
been different before: the local release parquet has 19 columns and the
published one has 25.

    python scripts/test_readme_recipes.py [--episodes N]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# The repo root, so `force_recovery` (the single source of the stiffness
# constant) is importable however this script is invoked.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import pyarrow.parquet as pq

REPO = "yxma/React"
GEL_MM = 4.25


def _published(limit: int):
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    files = sorted(fs.glob(f"datasets/{REPO}/data/*/meta/*/*.parquet"))
    if limit:
        step = max(1, len(files) // limit)
        files = files[::step][:limit]
    for f in files:
        with fs.open(f, "rb") as fh:
            yield f, pq.read_table(fh)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=6,
                    help="how many published episodes to sample (0 = all)")
    args = ap.parse_args()

    problems: list[str] = []
    n_free = n_contact = 0
    pen_contact: list[np.ndarray] = []
    n_files = 0

    for name, t in _published(args.episodes):
        n_files += 1
        for side in ("left", "right"):
            # ── the README's loading recipe, verbatim ────────────────────
            f = t[f"force_{side}_normal_n"].to_numpy()
            obs = np.array(t[f"sensor_{side}_pose"].to_pylist())
            tgt = np.array(t[f"force_{side}_target_pose"].to_pylist())
            pen = t[f"force_{side}_penetration_mm"].to_numpy()

            if obs.shape != tgt.shape or obs.shape[1] != 7:
                problems.append(f"{name} {side}: pose columns are not (T, 7)")
                continue

            # ── claim: free space is byte-identical ─────────────────────
            free = f == 0
            n_free += int(free.sum())
            if not np.array_equal(tgt[free], obs[free]):
                bad = int((~np.all(tgt[free] == obs[free], axis=1)).sum())
                problems.append(f"{name} {side}: {bad} free-space rows where "
                                f"target_pose != sensor_pose — the README "
                                f"promises they are identical")

            # ── claim: penetration_mm == F / k with the declared k ──────
            #
            # IMPORTED, not retyped. This line said `k = 1.0` while the data
            # ships k = 2.0 N/mm, so the check failed on 12 episode-sides for a
            # year — the data was right and the test was stale. `pipeline_guard`
            # already warns that "STIFFNESS HAS ONE SOURCE ... a second literal
            # would let the two drift"; this was that second literal, living in
            # the file whose job was to catch drift.
            from force_recovery.dexforce import STIFFNESS_N_PER_M
            k = STIFFNESS_N_PER_M / 1000.0          # N/m -> N/mm
            if not np.allclose(pen, f / k, atol=1e-5):
                problems.append(f"{name} {side}: penetration_mm is not "
                                f"force / {k} N/mm")

            # ── claim: the displacement is F/k along a UNIT direction ───
            con = f > 0
            n_contact += int(con.sum())
            if con.any():
                d = tgt[con, :3] - obs[con, :3]
                dist = np.linalg.norm(d, axis=1)
                moved = dist > 1e-9
                if moved.any():
                    # displacement magnitude must equal penetration in metres
                    want = pen[con][moved] / 1000.0
                    if not np.allclose(dist[moved], want, rtol=1e-3, atol=1e-6):
                        worst = np.abs(dist[moved] - want).max()
                        problems.append(
                            f"{name} {side}: |target - observed| does not "
                            f"equal penetration_mm/1000 (worst {worst:.2e} m)")
                    # ── the README's re-stiffening recipe ───────────────
                    K = 1.62
                    n_hat = d[moved] / dist[moved][:, None]
                    my = obs[con][moved].copy()
                    my[:, :3] = (obs[con][moved][:, :3]
                                 + (f[con][moved] / K)[:, None] / 1000.0 * n_hat)
                    got = np.linalg.norm(my[:, :3] - obs[con][moved][:, :3],
                                         axis=1) * 1000.0
                    if not np.allclose(got, f[con][moved] / K, rtol=1e-3):
                        problems.append(f"{name} {side}: the README's "
                                        f"recompute-at-k recipe does not "
                                        f"produce F/K penetration")
                # quaternion must be carried through unchanged
                if not np.array_equal(tgt[con, 3:], obs[con, 3:]):
                    problems.append(f"{name} {side}: target_pose changed the "
                                    f"quaternion — the README says it is "
                                    f"carried through unchanged")
                pen_contact.append(pen[con])

    # ── claim: k >= 1.62 puts the contact p95 inside the gel ───────────
    if pen_contact:
        allpen = np.concatenate(pen_contact)
        p95 = float(np.percentile(allpen, 95))
        for K, label in ((1.62, "contact p95"), (1.72, "maximum")):
            val = p95 / K if label == "contact p95" else allpen.max() / K
            print(f"[recipe] {label} penetration at k={K}: {val:.3f} mm "
                  f"(gel {GEL_MM} mm) {'OK' if val <= GEL_MM else 'OUTSIDE'}")
            if val > GEL_MM:
                problems.append(f"the README's k >= {K} does not keep the "
                                f"{label} inside the gel ({val:.3f} mm) — "
                                f"on this sample")

    print(f"[recipe] {n_files} published episodes, "
          f"{n_free:,} free-space rows, {n_contact:,} contact rows")
    for p in problems[:20]:
        print(f"  FAIL: {p}")
    print(f"readme recipes: {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
