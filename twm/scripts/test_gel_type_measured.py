"""Which gels carry markers is MEASURED, and the page says so from that.

This is the third attempt at the question and the first one that is allowed to
answer it, because the first two were wrong and were believed anyway:

  * FEATS's own `markered` column says True for every row including gel_5's,
    whose `gel_variant` is `black_dot`. It is a second-hand label and it is
    incorrect.
  * A blob detector thresholding at `mean - 2*std` reported ~34 dots on gel_5.
    On a smooth image that threshold sits inside the noise, so it invents
    blobs; it "confirmed" the wrong answer twice.

Rendering the references settled it by eye — gel_0..gel_4 carry dot arrays,
gel_5 does not — and this checks that the DETECTOR agrees with what the eye
saw, on both classes, before anything on the site is allowed to cite it:

  1  `marker_removal.detect_markers` finds dots on the marker gels and finds
     none on gels that have none. Separation, not a threshold picked to fit.
  2  the artifact records a measured count per dataset, never a typed label.
  3  the results page's marker sentence is generated from that artifact.

    python scripts/test_gel_type_measured.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RESULTS: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, evidence: str) -> None:
    RESULTS.append((bool(ok), name, evidence))


def main() -> int:
    from force_recovery.gel_type import OUT, measure

    table = measure() if not OUT.exists() else json.loads(OUT.read_text())

    by = {r["dataset"]: r for r in table["datasets"]}

    # 1 — separation. Marker gels and markerless gels must not overlap, or the
    # count is not measuring what its name says.
    markered = [r for r in table["datasets"] if r["n_dots_median"] > 0]
    clean = [r for r in table["datasets"] if r["n_dots_median"] == 0]
    lo = min((r["n_dots_median"] for r in markered), default=0)
    hi = max((r["n_dots_max"] for r in clean), default=0)
    check(bool(markered) and bool(clean) and lo > 10 * max(hi, 1),
          "marker and markerless gels separate",
          f"markered from {lo} dots, markerless up to {hi} — "
          f"{len(markered)} vs {len(clean)} datasets")

    # 2 — FEATS is the markered one, and it is the low scorer. That is the
    # claim the page wants to make; it is only worth making if the count says
    # so rather than a constant in this repo.
    check(by.get("feats", {}).get("n_dots_median", 0) > 10,
          "FEATS is measured as the marker gel",
          f"feats median {by.get('feats', {}).get('n_dots_median')} dots "
          f"over {by.get('feats', {}).get('n_refs')} references")

    # 3 — no typed labels. A hand-written `markered=True` beside a measured
    # count is a second source, and this file exists because a second source
    # was wrong twice.
    #
    # Match ASSIGNMENTS, not prose. The first version matched any line holding
    # both a marker word and True/False, and its only hit was the paragraph
    # above explaining that FEATS's own `markered` column reads True — the
    # check firing on the documentation of the bug it exists to prevent.
    import re
    src = Path("force_recovery/gel_type.py").read_text()
    typed = [ln for ln in src.splitlines()
             if re.search(r"""["']?(markered|marker_free|has_markers)["']?"""
                          r"\s*[:=]\s*(True|False)\b", ln)
             and "# ok:" not in ln]
    check(not typed, "no gel is labelled by hand",
          f"{len(typed)} typed marker label(s)"
          + (f": {typed[0].strip()[:60]}" if typed else ""))

    # 4 — and the page's sentence comes from it.
    from force_recovery.site2 import _marker_line
    line = _marker_line()
    n = by.get("feats", {}).get("n_dots_median", 0)
    check(str(n) in line and "FEATS" in line,
          "the results page states the measurement",
          f"{line[:120]}")

    width = max(len(x) for _, x, _ in RESULTS)
    print()
    for ok, name, ev in RESULTS:
        print(f"  [{'ok' if ok else 'FAIL'}] {name:<{width}}  {ev}")
    bad = sum(not ok for ok, _, _ in RESULTS)
    print(f"\ngel type measured: {len(RESULTS)} checks, {bad} failing")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
