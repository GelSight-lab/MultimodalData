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
USAGE = Path(__file__).with_name("readme_usage_section.md")
USAGE_MARK = "## How to use this dataset"
VERIFY = Path("/media/yxma/Disk1/twm/release_force/force_export_verify.json")
REPO = "yxma/React"
ANCHOR = "### depth (optional, `data/<task>/depth/`)"
MARK = "### estimated contact force"


def _ceil2(x: float) -> str:
    """Two decimals, rounded UP — for a lower bound, never down."""
    import math
    return f"{math.ceil(x * 100) / 100:.2f}"


def check_numbers(text: str) -> list[str]:
    """Every quoted statistic must match the verify artifact."""
    v = json.loads(VERIFY.read_text())
    # The SHIPPED stiffness, not a hard-coded "1.0". Pinning the check to a
    # literal k meant that when the shipped value moved to 2.0 the gate went on
    # validating the README against a configuration the dataset no longer used
    # — it would have passed a README describing the wrong column.
    sweep = v["penetration_sweep"][str(v["stiffness_n_per_mm"])]
    want = {
        f"{sweep['p95_mm']:.2f} mm": "p95 penetration",
        f"{sweep['frac_over_gel']:.2%}": "fraction past the gel",
        f"{v['identity_rows_checked']:,}": "free-space identity rows",
        f"{v['force_ceiling_n']:.3f} N": "isotonic ceiling",
        f"{v['force_at_ceiling_frac']:.2%}": "samples at the ceiling",
        # Lower bounds, so they are quoted rounded UP. `:.1f` on 1.7141 gives
        # "1.7", and k = 1.7 puts the hardest press 4.285 mm into a 4.25 mm
        # gel — the README asserted a threshold that does not hold, and this
        # gate confirmed it because the gate rounded the same way.
        f"{_ceil2(v['k_for_contact_p95_within_gel'])}": "k for contact p95 "
                                                        "within gel",
        f"{_ceil2(v['k_for_max_within_gel'])}": "k for max within gel",
        f"{v['contact_p95_penetration_mm']:.2f} mm": "contact-only p95",
        f"{v['alignment_pass']}/{v['n_sensor_sides']}": "row alignment",
    }
    return [f"README does not contain {label} = {s!r}"
            for s, label in want.items() if s not in text]


QUALITY_MARK = "## Data quality"
NOTES_MARK = "## Notes"
STAGE = Path("/media/yxma/Disk1/twm/release")


def quality_section() -> str:
    """Render the data-quality block FROM the artifacts, never by hand.

    Every number here changes when the curator is re-run, and the release was
    re-curated twice in one day. A hand-written percentage in the one file
    nobody re-derives is a number that will be wrong and look right.
    """
    import collections

    rows = []
    for task in ("motherboard", "pushT"):
        bf = json.loads((STAGE / task / "bad_frames.json").read_text())
        sg = json.loads((STAGE / task / "segments.json").read_text())
        fam = collections.Counter()
        for ep in bf["episodes"].values():
            for key, ivs in ep.items():
                if isinstance(ivs, list) and ivs and isinstance(ivs[0], list):
                    fam[key] += sum(b - a + 1 for a, b in ivs)
        rows.append((task, fam, bf["summary"], sg))

    # Every family the detector can emit, so a family that found nothing shows
    # as 0 rather than vanishing — "the detector ran and was quiet" and "the
    # detector is not there" must not look the same.
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from react_preprocess.curation import BAD_KEYS
    families = sorted(set(BAD_KEYS) | {k for _, fam, _, _ in rows for k in fam})
    head = "| flag | " + " | ".join(t for t, _, _, _ in rows) + " |"
    sep = "|---|" + "---|" * len(rows)
    # Per-family counts OVERLAP — one frame can trip two detectors — so they
    # are shown as they are and never summed. The union is what `summary`
    # carries, and it is the only number that reconciles with the segments.
    body = "\n".join(
        f"| `{k}` | " + " | ".join(f"{fam.get(k, 0):,}" for _, fam, _, _ in rows)
        + " |" for k in families)
    totals = ("| **flagged (union)** | "
              + " | ".join(f"**{s['total_bad_frames']:,} / {s['total_frames']:,}"
                           f" ({100*s['total_bad_frames']/s['total_frames']:.2f}%)**"
                           for _, _, s, _ in rows) + " |")
    clean = ("| **clean segments** | "
             + " | ".join(f"{sg['n_segments']} spans, {sg['total_frames']:,} "
                          f"frames ({sg['total_duration_min']:.1f} min)"
                          for _, _, _, sg in rows) + " |")
    slivers = ("| **dropped, clean but < 16 frames** | "
               + " | ".join(f"{s['total_frames'] - s['total_bad_frames'] - sg['total_frames']:,}"
                            for _, _, s, sg in rows) + " |")

    return f"""{QUALITY_MARK}

Per-task `bad_frames.json` marks intervals that should not be trained on, and
`segments.json` is their complement — contiguous clean spans, already excluding
every flag below. **Use `segments.json` and you never have to think about
this table.**

{head}
{sep}
{body}
{totals}
{clean}
{slivers}

The three rows above reconcile exactly: flagged + clean + dropped = total, for
both tasks. Per-flag counts do **not** sum to the flagged total, because one
frame can trip two detectors; the union is what `summary` reports and what the
segments complement.

`ot_loss_*` is OptiTrack track loss (a run of bit-identical poses, i.e. frozen
action), `pose_teleports_*` an implausible jump in translation *and* rotation
in one frame, `intensity_spikes` a GelSight reading above anything contact
produces. `tactile_corruption` and `cam_corruption` are **video** defects —
torn frames the sidecar scalars cannot see. They are found by looking for
off-illumination magenta laid out in scanlines: a GelSight is lit by three
coloured LEDs, magenta is outside that gamut, and a corrupt row is written
edge to edge while an object pressed into the gel is not. Every flagged
interval in this release was also inspected by eye.

**Runt episodes.** Two motherboard recordings are far too short to be complete
demonstrations and are best filtered out: `2026-05-19/episode_003` (4.0 s) and
`2026-05-19/episode_004` (7.0 s). Median episode length is 213 s; these two are
together 0.8 % of the release. They are shipped rather than deleted so episode
numbering stays stable.

**A missing pushT episode.** `pushT/2026-06-18/episode_004` was recorded but is
not published. Its recorder died without closing the file, which loses HDF5's
metadata cache: 79 GB of intact pixels behind a root object header that was
never written. All eight image streams were recovered (15,447 frames,
byte-verified), but only 2 of 16 timestamp chunks survived and no usable
OptiTrack poses. Without timestamps there is no cross-modal alignment, and
reconstructing them by interpolation misplaces frames by 15–1431 — so it is
video, not an episode, and is deliberately absent rather than published
half-aligned. Episode numbering is unaffected: pushT publishes 000–003.
"""


def repo_sizes() -> dict:
    """Measured from the dataset repo. Sizes drift with every upload."""
    from huggingface_hub import HfApi
    info = HfApi().repo_info(REPO, repo_type="dataset", files_metadata=True)
    total = depth = 0
    for s in info.siblings:
        sz = s.size or 0
        total += sz
        if "/depth/" in s.rfilename:
            depth += sz
    return {"total_gb": total / 1e9, "depth_gb": depth / 1e9,
            "core_gb": (total - depth) / 1e9}


def notes_section() -> str:
    """Rewritten because both of its bullets had gone stale.

    It promised depth "in a later upload" while 108 depth files were already
    published and documented two sections above, and it described the missing
    pushT recording as simply "corrupt" — which is now known to be wrong in a
    way that matters: nothing about it was corrupt, its metadata was never
    flushed. Data quality carries the full account.
    """
    z = repo_sizes()
    return f"""{NOTES_MARK}
- **Depth is published**, under `data/<task>/depth/<date>/<episode>/depth_*.mkv`
  (16-bit millimetres, FFV1-in-Matroska, lossless). It is {z['depth_gb']:.1f} GB
  of the {z['total_gb']:.1f} GB repo, so the download recipes above let you skip
  it — everything else is {z['core_gb']:.1f} GB.
- The previous single-task `.pt` release (`episodes/`, `segments/`) is
  superseded by this video format.
- Preview clips under `data/<task>/previews/` are 30 s renders at 2x with the
  three camera views, the OptiTrack skeleton, both GelSight streams and the
  projected sensor position. They are for looking, not for training, and frames
  excluded by `bad_frames.json` are outlined and named in red.
"""


def build(readme: str | None = None) -> str:
    """Rewrite the managed sections of the README.

    `readme` is injectable so idempotency can be tested — feeding the output
    back in must be a fixed point, and it was not: see `_drop`.
    """
    if readme is None:
        from huggingface_hub import hf_hub_download
        readme = Path(hf_hub_download(REPO, "README.md",
                                      repo_type="dataset")).read_text()

    def _drop(text: str, mark: str) -> str:
        """Remove an existing section so this stays idempotent.

        Fence-aware. The naive `text.find("\\n# ", ...)` treats a shell or
        Python comment inside a ``` block as a level-1 heading, and the usage
        section is full of them. That cut the section in half, left its tail in
        place and appended a fresh copy below — a second run of this script
        duplicated `### 3. Train an action that includes *how hard*` and
        everything after it.
        """
        lines = text.splitlines(keepends=True)
        level = len(mark) - len(mark.lstrip("#"))
        start = None
        fence = False
        for i, ln in enumerate(lines):
            if ln.lstrip().startswith("```"):
                fence = not fence
                continue
            if fence:
                continue
            if start is None:
                if ln.startswith(mark):
                    start = i
                continue
            h = len(ln) - len(ln.lstrip("#"))
            if 0 < h <= level and ln[h:h + 1] == " ":
                return "".join(lines[:start] + lines[i:])
        return "".join(lines[:start]) if start is not None else text

    readme = _drop(readme, MARK)
    section = SECTION.read_text().rstrip() + "\n\n"
    if ANCHOR not in readme:
        raise SystemExit(f"anchor not found in README: {ANCHOR!r}")
    readme = readme.replace(ANCHOR, section + ANCHOR, 1)

    # Usage goes before Data quality: a reader wants the recipes first and the
    # caveats where they will be re-read.
    readme = _drop(readme, USAGE_MARK)
    readme = _drop(readme, QUALITY_MARK)
    if "## Notes" not in readme:
        raise SystemExit("anchor not found in README: '## Notes'")
    z = repo_sizes()
    # Sizes appear in prose, in a tree comment and in two shell snippets. The
    # first pass replaced only the parenthesised forms and left "poses, ~4.4 GB"
    # behind — a replacement that reports success because it matched SOMEWHERE
    # is how a stale number survives an update. So: replace every occurrence,
    # then refuse if any known-stale figure is still present anywhere.
    for old, new in (("~4.4 GB", f"~{z['core_gb']:.1f} GB"),
                     ("~37 GB", f"~{z['total_gb']:.0f} GB"),
                     ("\u2248 4.3 GB",
                      f"\u2248 {z['core_gb']:.1f} GB without depth")):
        readme = readme.replace(old, new)
    # Word-boundary, not substring: "4.3 GB" occurs inside the correct
    # "34.3 GB" depth figure, and a gate that fires on its own output is a
    # gate that gets switched off.
    stale = [x for x in ("4.4 GB", "4.3 GB", "37 GB")
             if re.search(rf"(?<![\d.]){re.escape(x)}", readme)]
    if stale:
        raise SystemExit(f"stale download sizes still in the README: {stale}")

    readme = _drop(readme, NOTES_MARK)
    if "## License" not in readme:
        raise SystemExit("anchor not found in README: '## License'")
    tail = (USAGE.read_text().rstrip() + "\n\n"
            + quality_section().rstrip() + "\n\n"
            + notes_section().rstrip() + "\n\n## License")
    return readme.replace("## License", tail, 1)


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
        commit_message="README: force columns recomputed from the "
                       "calibration-free reconstruction; stiffness raised to "
                       "2 N/mm so no commanded target sits deeper than the gel")
    print("uploaded README.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
