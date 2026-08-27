"""Every number in USAGE.md is recomputed from the data it describes.

Documentation rots silently. This session found `toolbox/actions.py` claiming
"quat wxyz" over scalar-last code, and a test asserting `penetration = F/1.0`
against data shipping `F/2.0`. Both read as authoritative and both were wrong.

So the doc's checkable claims are extracted and recomputed. A claim that cannot
be recomputed does not belong in a table.

    python scripts/test_usage_doc.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from react_paths import force_meta, release_root   # noqa: E402

import numpy as np                                             # noqa: E402
import pyarrow.parquet as pq                                   # noqa: E402

RESULTS: list[tuple[bool, str, str]] = []
DOC = Path(__file__).resolve().parents[1] / "docs" / "USAGE.md"
REL = release_root("motherboard")
RELF = force_meta("motherboard")


def check(ok: bool, name: str, evidence: str) -> None:
    RESULTS.append((bool(ok), name, evidence))


def main() -> int:
    text = DOC.read_text()

    # 1 — the stiffness the doc quotes is the one the code defines
    from force_recovery.dexforce import STIFFNESS_N_PER_M
    k = STIFFNESS_N_PER_M / 1000.0
    m = re.search(r"=\s*\*\*([\d.]+) N/mm\*\*", text)
    check(m and abs(float(m.group(1)) - k) < 1e-9,
          "the quoted stiffness is dexforce's",
          f"doc says {m.group(1) if m else '?'} N/mm, code defines {k:g}")

    # ...and the data obeys it
    t = pq.read_table(sorted(RELF.glob("*/*.parquet"))[0]).to_pydict()
    ratios = []
    for s in ("left", "right"):
        F = np.asarray(t[f"force_{s}_normal_n"], float)
        P = np.asarray(t[f"force_{s}_penetration_mm"], float)
        sel = F > 0.05
        if sel.any():
            ratios.append(float(np.median(P[sel] / F[sel])))
    check(all(abs(r - 1.0 / k) < 1e-6 for r in ratios),
          "the published data obeys that stiffness",
          f"penetration/force = {np.round(ratios,4).tolist()}, expected {1/k:g}")

    # 2 — the tactile refresh fraction
    fr = []
    for p in sorted(RELF.glob("*/*.parquet"))[:8]:
        tt = pq.read_table(p, columns=["tactile_left_is_new", "tactile_right_is_new"]).to_pydict()
        for c in tt:
            fr.append(float(np.mean(np.asarray(tt[c], bool))))
    got = 100 * float(np.mean(fr))
    m = re.search(r"only \*\*~(\d+) %\*\* of rows", text)
    check(m and abs(float(m.group(1)) - got) < 4.0,
          "the quoted tactile refresh rate matches the data",
          f"doc ~{m.group(1) if m else '?'} %, measured {got:.1f} % over "
          f"{len(fr)} episode-sides")

    # 3 — the split numbers
    sp = json.loads((REL / "splits.json").read_text())
    st = sp["stats"]
    want = {"test": st["test_fraction"] * 100, "guard": st["guard_fraction"] * 100,
            "train": (1 - st["test_fraction"] - st["guard_fraction"]) * 100}
    m = re.search(r"\*\*test ([\d.]+) %, guard ([\d.]+) %,\s*\n?train ([\d.]+) %\*\*", text)
    ok3 = m and all(abs(float(m.group(i + 1)) - v) < 0.15
                    for i, v in enumerate((want["test"], want["guard"], want["train"])))
    n_iv = re.search(r"over (\d[\d,]*) intervals plus (\d+) wholly-held-out", text)
    ok3 = ok3 and n_iv and int(n_iv.group(1).replace(",", "")) == st["n_test_intervals"] \
        and int(n_iv.group(2)) == st["n_whole_test_episodes"]
    check(bool(ok3), "the quoted split fractions come from splits.json",
          f"doc {[m.group(i+1) for i in range(3)] if m else '?'} vs "
          f"{want['test']:.1f}/{want['guard']:.1f}/{want['train']:.1f} %, "
          f"{st['n_test_intervals']} intervals + {st['n_whole_test_episodes']} whole")

    # 4 — the guard claim, recomputed by enumeration
    from twm.splits import forbidden_starts
    eps = [json.loads(l) for l in (REL / "episodes.jsonl").read_text().splitlines() if l.strip()]
    W = sp["max_train_window"]
    n_ok, leaks_noguard = 0, 0
    for e in eps:
        key, N = e["episode"], e["n_frames"]
        info = sp["episodes"][key]
        test = np.zeros(N, bool)
        for a, b in info["test"]:
            test[a:b + 1] = True
        forb = forbidden_starts(sp, key, W)
        for s in range(0, N - W + 1):
            if any(lo <= s <= hi for lo, hi in forb):
                continue
            n_ok += 1
            if test[s:s + W].any():
                leaks_noguard = -1
    m = re.search(r"Enumerated: ([\d,]+) admissible training windows", text)
    check(m and int(m.group(1).replace(",", "")) == n_ok and leaks_noguard == 0,
          "the enumerated window count is the one the doc quotes",
          f"doc {m.group(1) if m else '?'}, recomputed {n_ok:,}, leaks 0")

    # 5 — the reprojection error budget
    import shutil, tempfile
    from react_toolbox.calibration import load_calibration
    from twm.calib_epoch import calib_dir
    stage = Path(tempfile.mkdtemp())
    shutil.copytree(calib_dir("motherboard"), stage / "calibration")
    cal = load_calibration(stage)
    px = {v: cal["cams"][v]["rmse"] / 800.0 * cal["cams"][v]["intrinsics"]["fx"]
          for v in ("left", "middle", "right")}
    m = re.search(r"\*\*([\d.]+) / ([\d.]+) / ([\d.]+) px\*\* at 800 mm", text)
    ok5 = m and all(abs(float(m.group(i + 1)) - px[v]) < 0.2
                    for i, v in enumerate(("left", "middle", "right")))
    mr = re.search(r"rmse is ([\d.]+) / ([\d.]+) / ([\d.]+) mm", text)
    ok5 = ok5 and mr and all(
        abs(float(mr.group(i + 1)) - cal["cams"][v]["rmse"]) < 0.1
        for i, v in enumerate(("left", "middle", "right")))
    check(bool(ok5), "the quoted reprojection budget is the calibration's",
          f"doc {[m.group(i+1) for i in range(3)] if m else '?'} px, "
          f"computed {[round(px[v],1) for v in ('left','middle','right')]}")

    # 6 — frame rates
    hz = {}
    for d in sorted({p.parent.name for p in RELF.glob("*/*.parquet")}):
        v = []
        for p in sorted((RELF / d).glob("*.parquet")):
            ts = np.asarray(pq.read_table(p, columns=["timestamp"]).to_pydict()["timestamp"], float)
            if len(ts) > 10:
                v.append(1 / np.median(np.diff(ts)))
        hz[d] = (min(v), max(v))
    m = re.search(r"\*\*2026-05-19 runs at ([\d.]+)–([\d.]+) Hz\*\*", text)
    ok6 = m and abs(float(m.group(1)) - hz["2026-05-19"][0]) < 0.3 \
        and abs(float(m.group(2)) - hz["2026-05-19"][1]) < 0.3
    check(bool(ok6), "the quoted frame rates are measured from timestamps",
          f"doc {m.group(1)}–{m.group(2)} Hz for 05-19; measured "
          f"{hz['2026-05-19'][0]:.1f}–{hz['2026-05-19'][1]:.1f}; other dates "
          f"{hz['2026-05-10'][0]:.1f}/{hz['2026-05-11'][0]:.1f}" if m else "no match")

    # 7 — every module the doc's table names is published
    from huggingface_hub import HfApi
    pub = set(HfApi().list_repo_files("yxma/React", repo_type="dataset"))
    named = re.findall(r"`toolbox/([a-z_]+\.py)`", text)
    missing = [n for n in named if f"toolbox/{n}" not in pub]
    check(not missing and len(named) >= 8,
          "every toolbox module the doc names is actually published",
          f"{len(named)} named, all present on the Hub"
          + (f"; MISSING {missing}" if missing else ""))

    w = max(len(x) for _, x, _ in RESULTS)
    print()
    for ok, name, ev in RESULTS:
        print(f"  [{'ok' if ok else 'FAIL'}] {name:<{w}}  {ev}")
    n = sum(not ok for ok, _, _ in RESULTS)
    print(f"\nusage doc: {len(RESULTS)} checks, {n} failing")
    return 1 if n else 0


if __name__ == "__main__":
    raise SystemExit(main())
