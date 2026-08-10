"""Every artifact's declared laws must cover what actually computes it.

`site2.NUMBER_LAWS` says which modules a number depends on, and the build
refuses a number older than those modules. The declaration is hand-written, so
it can drift two ways and both are silent:

  * DECLARE TOO LITTLE and the gate stops catching real staleness — the page
    pairs a fresh figure with a number computed by code that has since moved.
  * DECLARE TOO MUCH and the gate condemns work it cannot justify. That is
    what happened: three artifacts inherited the default union, which named
    `force_recon_matrix.py`, and were failed by a change that only added a
    file lock to it. A gate that orders a two-hour recompute for no reason is
    a gate people learn to bypass.

This walks the real import graph from each artifact's producer and asserts the
declaration covers it. Producers are named here because nothing in the code
links an artifact to the module that writes it — that mapping is itself
checked, by grepping for the write.

    python scripts/test_number_laws.py
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
PKG = ROOT / "force_recovery"

from force_recovery.site2 import (DEPTH_LAWS, FORCE_LAWS,  # noqa: E402
                                  NUMBER_ARTIFACTS, NUMBER_LAWS)

# artifact -> module that writes it. Verified below against the source.
PRODUCER = {
    "force_recon_matrix.json": "force_recon_matrix",
    "error_analysis.json": "error_analysis",
    "depth_eval.json": "depth_eval",
    "cross_dataset.json": "cross_dataset",
    "dataset_sizes.json": "dataset_sizes",
    "react_holdout.json": "react_calib",
    "truncation.json": "truncation_figure",
    "force_agreement.json": "verify_force_channel",
    "results_metrics.json": "results_page",
}
# Not checked here: artifacts whose producer lives outside this package or
# which describe the release rather than a computation over it.
SKIP = {"force_matrix.json", "calibfree_vs_lut.json", "react_weight_ab.json",
        "release_channel.json"}


def imports(mod: str, seen: set[str] | None = None) -> set[str]:
    """Transitive intra-package imports of `mod`, including deferred ones."""
    seen = set() if seen is None else seen
    if mod in seen:
        return seen
    seen.add(mod)
    f = PKG / f"{mod}.py"
    if not f.exists():
        return seen
    tree = ast.parse(f.read_text())
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.ImportFrom) and node.level == 1:
            # `from .x import y` and `from . import x`
            names = [node.module] if node.module else [
                a.name for a in node.names]
        elif isinstance(node, ast.Import):
            names = [a.name.split("force_recovery.")[-1]
                     for a in node.names if a.name.startswith("force_recovery.")]
        for n in names:
            if n and (PKG / f"{n}.py").exists():
                imports(n, seen)
    return seen


def main() -> int:
    bad = []
    for art in NUMBER_ARTIFACTS:
        if art in SKIP:
            continue
        mod = PRODUCER.get(art)
        if not mod:
            bad.append(f"{art}: no producer declared in this test")
            continue
        src = (PKG / f"{mod}.py").read_text()
        if art not in src:
            bad.append(f"{art}: {mod}.py does not mention it — "
                       f"PRODUCER is wrong")
            continue
        declared = set(NUMBER_LAWS.get(art, FORCE_LAWS))
        if not declared:
            continue                        # deliberately law-free, documented
        actual = {f"{m}.py" for m in imports(mod)} - {f"{mod}.py"}
        # Only laws matter — a producer imports plenty that cannot change a
        # number (json, paths). The law vocabulary is what the gate can name.
        vocab = set(DEPTH_LAWS) | set(FORCE_LAWS) | {f"{mod}.py"}
        missing = (actual & vocab) - declared
        if missing:
            bad.append(f"{art}: declares {sorted(declared)} but {mod}.py "
                       f"imports {sorted(missing)} — the gate would miss a "
                       f"change in those")
        unused = declared - actual - {f"{mod}.py"}
        if unused:
            bad.append(f"{art}: declares {sorted(unused)}, which {mod}.py "
                       f"never imports — the gate would condemn this number "
                       f"for an unrelated change")
    for b in bad:
        print(f"  [FAIL] {b}")
    print(f"\nnumber laws: {len(NUMBER_ARTIFACTS) - len(SKIP)} artifacts "
          f"checked, {len(bad)} bad declaration(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
