"""The README builder must be a fixed point, and its heading scan fence-aware.

`update_dataset_readme` replaces managed sections in a published README. If
replacement is not idempotent, the second run appends instead of replacing and
the dataset's front page grows a duplicate of itself — which is what happened:
the section remover treated a `# comment` line inside a fenced code block as a
level-1 heading, cut the usage section in half, left the tail, and appended a
fresh copy below it.

    python scripts/test_readme_builder.py
"""
from __future__ import annotations

import collections
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from twm.force_recovery.update_dataset_readme import build  # noqa: E402

FIXTURE = """# Fixture

## Format
text

### depth (optional, `data/<task>/depth/`)
depth text

## Downloading — depth is optional
The dataset splits into a **lightweight core** (RGB + tactile + poses, ~4.4 GB)
and an **optional depth tree**, whole dataset ≈ 4.3 GB vs raw.

```bash
# Core only — RGB + tactile + parquet, NO depth (~4.4 GB)
huggingface-cli download yxma/React
# Everything including depth (~37 GB)
```

## Notes
- stale note

## License
CC-BY-4.0
"""


def main() -> int:
    problems: list[str] = []

    one = build(FIXTURE)
    two = build(one)
    three = build(two)

    print(f"[builder] fixture {len(FIXTURE.splitlines())} lines -> "
          f"{len(one.splitlines())} -> {len(two.splitlines())} -> "
          f"{len(three.splitlines())}")

    if one != two:
        problems.append("build() is not idempotent: the second pass differs "
                        "from the first, so re-running it edits the published "
                        "README instead of leaving it alone")
    if two != three:
        problems.append("build() has not converged by the third pass")

    heads = re.findall(r"^#{1,4} .+$", two, re.M)
    dupes = [h for h, c in collections.Counter(heads).items() if c > 1]
    if dupes:
        problems.append(f"duplicated headings after a second pass: {dupes[:4]}")

    # a comment inside a fence must not be mistaken for a heading
    if "# Core only" not in one:
        problems.append("the fenced download snippet was eaten — the section "
                        "scanner is treating code comments as headings")

    for stale in ("~4.4 GB", "~37 GB"):
        if re.search(rf"(?<![\d.]){re.escape(stale)}", one):
            problems.append(f"stale download size {stale!r} survived the "
                            f"rewrite")

    for want in ("## How to use this dataset", "## Data quality",
                 "### estimated contact force"):
        if one.count(want) != 1:
            problems.append(f"{want!r} appears {one.count(want)} times, "
                            f"expected exactly 1")

    for p in problems:
        print(f"  FAIL: {p}")
    print(f"readme builder: {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
