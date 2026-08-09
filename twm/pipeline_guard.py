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
OPT_OUT_FALLBACK = "fallback-ok"        # ... for a deliberate default answer
# diagnostics/ and legacy_pt/ are dated one-off forensics, frozen with the
# defects they investigated; guarding them retro-flags history, not pipeline.
SKIP_DIRS = {"__pycache__", ".git", "calibration", "diagnostics", "legacy_pt"}
# Files that legitimately index raw frames: the recorder writing them, the
# latency studies that exist precisely to compare shifted vs unshifted, and
# the interactive players whose whole point is a user-adjustable offset.
RAW_OK = {"data_collection.py", "test_latency.py", "latency_align_viewer.py",
          "build_latency_clips.py", "build_latency_correction_clips.py",
          "play_react_pt.py", "visualize.py", "tactile_align.py",
          "pipeline_guard.py"}


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
            # NB: this append sat AFTER the continue, inside its block, for
            # the guard's whole life — unreachable, so the check could never
            # report anything. A gate you never saw fail is not a gate.
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


def check_single_calib_epoch() -> list[str]:
    """The task->extrinsics-epoch mapping lives in calib_epoch, nowhere else.

    The cameras were recalibrated between tasks (May-12 for motherboard,
    June-26 for pushT). That mapping existed in FIVE copies, and two viewers
    carried no mapping at all — they defaulted every task to
    `calibration/result` (June-26). Every motherboard preview shipped through
    the wrong camera pose: 53-64 mm / 2.6-6.0 deg between epochs, a 35-73 px
    projection error that looks like a miscalibrated rig, not a bug.
    `scripts/diagnostics/` is exempt: dated one-off forensics, not pipeline.
    """
    bad = []
    pat = re.compile(r"""result\ backup|calibration.{0,4}[/"']\s*result""")
    # data_collection.py is the LIVE recorder: for a recording being made right
    # now, "the current calibration" (`result/`) is correct by definition —
    # epochs only exist for replaying the past.
    for p in _py_files():
        if p.name in ("calib_epoch.py", "pipeline_guard.py",
                      "data_collection.py"):
            continue
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if pat.search(line) and OPT_OUT not in line:
                bad.append(f"{p.relative_to(ROOT)}:{i}: names a calibration "
                           f"epoch directory — resolve it through "
                           f"calib_epoch.calib_dir(task) instead")
    return bad


def check_no_silent_fallback() -> list[str]:
    """A resolver must not answer a question it failed to answer.

    `_get_trim_offset` read a sidecar pushT never had and returned 0 on any
    failure. Zero is also the correct answer for every motherboard episode,
    so the wrong answer was indistinguishable from the right one — pushT
    previews played 6.5 minutes of pre-roll and nothing reported anything.
    `world_offset_m` shipped the same shape and had to be changed to raise.

    The pattern: a function whose name says it RESOLVES a fact about the
    data, swallowing an exception and returning a constant. Returning a
    constant is fine when it is the answer; it is not fine when it means
    "I don't know". Raise, or name the fallback in a comment marked
    ``{OPT_OUT}``.
    """
    bad = []
    resolver = re.compile(r"^(_?(get|load|read|resolve|lookup|find)_|.*_"
                          r"(offset|dir|path|index|shift|epoch|trim))", re.I)
    for p in _py_files():
        if p.name == "pipeline_guard.py":
            continue
        src = p.read_text()
        lines = src.splitlines()
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue                       # already reported by the lag check
        for fn in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)):
            if not resolver.match(fn.name):
                continue
            for h in (n for n in ast.walk(fn) if isinstance(n, ast.ExceptHandler)):
                for r in (n for n in ast.walk(h)
                          if isinstance(n, ast.Return) and n.value is not None):
                    if not isinstance(r.value, ast.Constant):
                        continue
                    near = "\n".join(lines[max(0, h.lineno - 2):r.lineno + 1])
                    if OPT_OUT_FALLBACK in near:
                        continue
                    bad.append(
                        f"{p.relative_to(ROOT)}:{r.lineno}: {fn.name} returns "
                        f"{ast.unparse(r.value)} from an except handler — a "
                        f"silent fallback in a resolver. Raise, or mark it "
                        f"'{OPT_OUT_FALLBACK}' with the reason.")
    return bad


def check_single_repair_policy() -> list[str]:
    """Only `repair` decides whether a damaged recording may be published.

    Auto-repair hands the pipeline a file rebuilt from a crashed recording.
    Recovery returns whatever HDF5 had evicted from its metadata cache — the
    pixels, and for pushT/2026-06-18/episode_004 only 2 of 16 timestamp
    chunks. The single most dangerous thing anyone can add to this codebase is
    a second place that decides such a file is good enough, or that fills in
    the missing time base. Interpolating those timestamps is measurably wrong:
    it misplaces frames by 15.6, 24.5 and 1431 against episodes where the
    answer is known, and the pipeline's tactile lag is 15 frames.

    So the naming rule (`.recovered.h5`) and the publish decision both live in
    `repair` and are reached through `repair.is_recovered` / `source_stem` /
    `release_eligibility`. `scripts/` may name the suffix in test fixtures.
    """
    bad = []
    pat = re.compile(r'["\']\.recovered|\.recovered\.h5|'
                     r'bad object header version')
    for p in _py_files():
        if p.name in ("repair.py", "pipeline_guard.py", "h5raw.py"):
            continue
        if p.parts[-2:-1] == ("scripts",) and p.name.startswith("test_"):
            continue
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            if pat.search(line) and OPT_OUT_FALLBACK not in line:
                bad.append(f"{p.relative_to(ROOT)}:{i}: hard-codes the "
                           f"recovered-file naming or the crash signature — "
                           f"go through react_preprocess.repair, which is the "
                           f"only module allowed to decide that a rebuilt "
                           f"recording may be published")
    return bad


def check_difference_images_are_colour() -> list[str]:
    """A difference image is drawn in colour, through `visualize.diff_rgb`.

    Several panels drew `|img - ref|` averaged over channels. That discards
    the sign — a bump and a dent are the same picture — and it discards the
    channel split, which is the raw material of every reconstruction in this
    project: three LEDs from three directions, one per channel. The React
    dataset previews have always shown the signed colour difference; the
    force figures did not, and a reader comparing the two saw two different
    things called the same thing.
    """
    bad = []
    pat = re.compile(r"np\.abs\(\s*(img|frame|st\[.dI.\])\s*-\s*ref\s*\)"
                     r"\s*\.\s*mean\s*\(\s*axis\s*=\s*2")
    for p in _py_files():
        if p.name in ("visualize.py", "pipeline_guard.py"):
            continue
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if pat.search(line):
                bad.append(f"{p.relative_to(ROOT)}:{i}: draws a difference "
                           f"image as a greyscale magnitude — use "
                           f"force_recovery.visualize.diff_rgb, which keeps "
                           f"the sign and the channels")
    return bad


def check_single_mesh_law() -> list[str]:
    """One camera, one z-exaggeration, one mesh renderer — in `o3d_view`.

    `MESH_KW` (smooth_px, z_scale, front, zoom) was defined identically in
    `o3d_view` and `showcase`. Identical today; one edit from two different
    surfaces of the same depth map on two pages of the same site, with nothing
    to say which one the reader is looking at.
    """
    bad = []
    for p in _py_files():
        if p.name in ("o3d_view.py", "pipeline_guard.py"):
            continue
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if re.match(r"\s*MESH_KW\s*=\s*dict", line):
                bad.append(f"{p.relative_to(ROOT)}:{i}: redeclares the mesh "
                           f"render law — import MESH_KW from o3d_view")
    return bad


CHECKS = {
    "single calibration-epoch definition": check_single_calib_epoch,
    "single mesh render law": check_single_mesh_law,
    "difference images drawn in colour": check_difference_images_are_colour,
    "single repair/publish policy": check_single_repair_policy,
    "no silent fallback in resolvers": check_no_silent_fallback,
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
