"""Insert the force-columns section into the `yxma/React` README.

The dataset README described 19 parquet columns and mentioned force zero times
in 214 lines, while the force estimation had its own package, its own results
Space, and four separate docs — all of them in places a dataset consumer never
looks. Documentation that lives only where the author works is not
documentation.

Before writing, every number in the section is re-checked against the
artifacts that produced it (`force_export_verify.json`) and against the
uploaded parquet themselves. A README is the one file nobody re-derives, so a
stale number here outlives every other copy.

    python -m force_recovery.update_dataset_readme --dry-run
    python -m force_recovery.update_dataset_readme
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

SECTION = Path(__file__).with_name("readme_force_section.md")
VERIFY = Path("/media/yxma/Disk1/twm/release_force/force_export_verify.json")
REPO = "yxma/React"
ANCHOR = "### depth (optional, `data/<task>/depth/`)"
MARK = "### estimated contact force"


def check_numbers(text: str) -> list[str]:
    """Every quoted statistic must match the verify artifact."""
    v = json.loads(VERIFY.read_text())
    sweep = v["penetration_sweep"]["1.0"]
    want = {
        f"{sweep['p95_mm']:.2f} mm": "p95 penetration",
        f"{sweep['frac_over_gel']:.2%}": "fraction past the gel",
        f"{v['identity_rows_checked']:,}": "free-space identity rows",
        f"{v['force_ceiling_n']:.3f} N": "isotonic ceiling",
        f"{v['force_at_ceiling_frac']:.2%}": "samples at the ceiling",
        f"{v['k_for_p95_within_gel']:.1f}": "k for p95 within gel",
        f"{v['k_for_max_within_gel']:.1f}": "k for max within gel",
        f"{v['alignment_pass']}/{v['n_sensor_sides']}": "row alignment",
    }
    return [f"README does not contain {label} = {s!r}"
            for s, label in want.items() if s not in text]


def build() -> str:
    from huggingface_hub import hf_hub_download

    readme = Path(hf_hub_download(REPO, "README.md", repo_type="dataset")).read_text()
    section = SECTION.read_text().rstrip() + "\n\n"
    if MARK in readme:                       # idempotent: replace, never stack
        start = readme.index(MARK)
        nxt = readme.find("\n### ", start + 1)
        end = readme.find("\n## ", start + 1)
        cut = min(x for x in (nxt, end, len(readme)) if x > 0)
        readme = readme[:start] + readme[cut + 1:]
    if ANCHOR not in readme:
        raise SystemExit(f"anchor not found in README: {ANCHOR!r}")
    return readme.replace(ANCHOR, section + ANCHOR, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    new = build()
    bad = check_numbers(new)
    if bad:
        print(f"REFUSING: {len(bad)} number(s) not backed by {VERIFY.name}")
        for b in bad:
            print("   ", b)
        return 1
    print(f"all quoted statistics match {VERIFY.name}")
    print(f"README: {len(new.splitlines())} lines "
          f"({new.count('force')} mentions of 'force')")
    if args.dry_run:
        i = new.index(MARK)
        print("\n--- inserted section (first 20 lines) ---")
        print("\n".join(new[i:].splitlines()[:20]))
        return 0

    from huggingface_hub import HfApi
    HfApi().upload_file(
        path_or_fileobj=new.encode(), path_in_repo="README.md",
        repo_id=REPO, repo_type="dataset",
        commit_message="README: document the estimated contact-force columns "
                       "(force_*_normal_n / penetration_mm / target_pose), the "
                       "1 N/mm stiffness assumption, and the limits — the data "
                       "shipped without a word about it")
    print("uploaded README.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
