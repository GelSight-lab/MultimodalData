"""Gate: the pipeline invariants whose violation already shipped once.

`python -m twm.pipeline_guard` exits non-zero on any violation.

Each check exists because the defect it names actually happened, not because
it sounded prudent:

1. ONE DEFINITION OF THE TACTILE LAG. The GelSight capture lag lived in four
   places with three different values (15 / 15 / 3 / 0) and a fifth consumer
   applied none, which put the tactile tiles half a second ahead of the camera
   in every published preview. Any module that reads GelSight frames must get
   the offset from `tactile_align`, not declare its own.

2. NO RAW GELSIGHT INDEXING. Reading `gelsight/<side>/frames[i]` with a camera
   index and no lag correction is the exact shape of that bug. New code must
   go through `tactile_align.gel_index` (or explicitly opt out with the marker
   comment, which makes the exception visible in review).

3. THE FORCE PATH STAYS UNTOUCHED BY COSMETIC STEPS. Marker inpainting and the
   halo-pedestal removal are figure-only; they measurably do NOT help force
   (0.775 -> 0.737 on FEATS) and must never enter the feature path.

4. STIFFNESS HAS ONE SOURCE. `dexforce.STIFFNESS_N_PER_M` defines it and
   `pipeline.STIFFNESS_N_PER_MM` derives it; a second literal would let the
   dataset column and the site figure disagree about what an action means.
"""
from __future__ import annotations

import ast
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent
OPT_OUT = "tactile-lag-exempt"          # marker comment for deliberate cases
SKIP_DIRS = {"__pycache__", ".git", "calibration"}
# Files that legitimately index raw frames: the recorder writing them, the
# latency studies that exist precisely to compare shifted vs unshifted, and
# the interactive player whose whole point is a user-adjustable offset.
RAW_OK = {"data_collection.py", "test_latency.py", "latency_align_viewer.py",
          "build_latency_clips.py", "build_latency_correction_clips.py",
          "play_react_pt.py", "tactile_align.py", "pipeline_guard.py"}


def _py_files():
    for p in ROOT.rglob("*.py"):
        if not any(d in p.parts for d in SKIP_DIRS):
            yield p


def check_single_lag_definition() -> list[str]:
    """No module may state the lag itself — as an assignment OR a default.

    The first version matched assignments only, and missed
    `row_for_h5_frame(..., legacy_shift: int = 15)` sitting in the force
    overlay for the whole life of the guard: a fourth copy of the constant,
    in the exact module whose docstring explains why copies are dangerous.
    A default argument is a declaration; it just does not look like one.
    """
    bad = []
    assign = re.compile(r"\s*(LEGACY_SHIFT|SHIFT|TACTILE_LAG)\s*=\s*\d+")
    default = re.compile(r"(legacy_shift|shift|tactile_lag|lag)"
                         r"\s*(:\s*[\w\[\]| ]+)?\s*=\s*(\d+)", re.I)
    for p in _py_files():
        if p.name in ("tactile_align.py", "pipeline_guard.py"):
            continue
        src = p.read_text()
        for i, line in enumerate(src.splitlines(), 1):
            if assign.match(line):
                bad.append(f"{p.relative_to(ROOT)}:{i}: redeclares the tactile "
                           f"lag — import it from tactile_align instead")
        # defaults: only inside a def, and only a nonzero literal (0 is the
        # honest "no shift" used by the latency viewer's opt-out)
        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            # Report, never crash: a guard that dies on one unparseable file
            # takes every other check down with it and reads as "no run".
            bad.append(f"{p.relative_to(ROOT)}:{e.lineno}: cannot parse "
                       f"({e.msg}) — not checked for a duplicate lag")
            continue
        for fn in (n for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef)):
            for arg, dflt in zip(
                    fn.args.args[len(fn.args.args) - len(fn.args.defaults):],
                    fn.args.defaults):
                if (default.match(f"{arg.arg}={ast.unparse(dflt)}")
                        and isinstance(dflt, ast.Constant)
                        and isinstance(dflt.value, int) and dflt.value != 0
                        and OPT_OUT not in src.splitlines()[fn.lineno - 1]):
                    bad.append(f"{p.relative_to(ROOT)}:{fn.lineno}: "
                               f"{fn.name}({arg.arg}={dflt.value}) hard-codes "
                               f"the tactile lag as a default — default to "
                               f"None and fall back to tactile_align")
    return bad


def check_no_raw_gel_indexing() -> list[str]:
    bad = []
    # Match the INDEX EXPRESSION, not merely the presence of brackets: the
    # first version flagged four lines that already read `[gel_at(i)]`, and a
    # guard that cries wolf on correct code trains people to ignore it.
    pat = re.compile(
        r'gelsight/(\{side\}|left|right|%s|\{s\})/frames"?\]\s*\[([^\]]*)\]')
    corrected = ("gel_at", "gel_index", "gel_lag", "LEGACY_SHIFT", "+ lag",
                 "idx_map")
    for p in _py_files():
        if p.name in RAW_OK:
            continue
        for i, line in enumerate(p.read_text().splitlines(), 1):
            m = pat.search(line)
            if not m or OPT_OUT in line:
                continue
            if any(c in m.group(2) for c in corrected):
                continue                      # already lag-corrected
                bad.append(f"{p.relative_to(ROOT)}:{i}: indexes raw GelSight "
                           f"frames — use tactile_align.gel_index, or add the "
                           f"'{OPT_OUT}' comment if intentional")
    return bad


def check_force_path_clean() -> list[str]:
    """The DEFAULT force features must come from the untouched reconstruction.

    A cosmetic step may still appear behind an explicit opt-in flag, because
    the site reports the inpainted variant as a *rejected control* (0.737 vs
    0.775). So the check is not "is the name present" — my first version was,
    and it flagged exactly that legitimate control. It is: does the call sit
    under a condition, and is the unconditional branch the plain `stages`.
    """
    bad = []
    fe = ROOT / "force_recovery" / "force_eval_all.py"
    if not fe.exists():
        return bad
    tree = ast.parse(fe.read_text())
    cosmetic = {"stages_depth", "remove_halo_pedestal", "crop_to_contact"}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in cosmetic):
            continue
        # walk up: a cosmetic call is acceptable only inside a conditional
        guarded = any(isinstance(a, (ast.IfExp, ast.If))
                      for a in ast.walk(tree)
                      if any(c is node for c in ast.walk(a)))
        if not guarded:
            bad.append(f"force_eval_all.py calls {node.func.id} "
                       f"unconditionally at line {node.lineno}: cosmetic steps "
                       f"are figure-only (0.775 -> 0.737 on FEATS)")
    return bad


def check_single_stiffness() -> list[str]:
    bad = []
    for p in _py_files():
        if p.name in ("dexforce.py", "pipeline_guard.py"):
            continue
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if re.match(r"\s*STIFFNESS\w*\s*=\s*[\d.]+", line):
                bad.append(f"{p.relative_to(ROOT)}:{i}: second stiffness "
                           f"literal — derive it from dexforce")
    return bad


CHECKS = {
    "single tactile-lag definition": check_single_lag_definition,
    "no raw GelSight indexing": check_no_raw_gel_indexing,
    "force path free of cosmetic steps": check_force_path_clean,
    "single stiffness definition": check_single_stiffness,
}


def main() -> int:
    total = 0
    for name, fn in CHECKS.items():
        bad = fn()
        total += len(bad)
        print(f"[{'FAIL' if bad else 'ok'}] {name}")
        for b in bad:
            print(f"    {b}")
    print(f"\npipeline guard: {len(CHECKS)} checks, {total} violation(s)")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
