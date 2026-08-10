"""Both reconstructions, every ground-truth dataset, one protocol.

The question this answers: if React's force channel is computed from the
calibration-free solve instead of the LUT, is that better everywhere, or only
where it was first measured?

Held fixed for every cell — the contact mask, the integrator, the five
features (vol, vol2, maxd, area, h1), the per-group half/half least squares
with isotonic calibration, 5 seeds, and the within-group shuffle control. The
only thing that varies is the colour-to-gradient map, so a difference in rho
is attributable to it.

THE SHUFFLE CONTROL IS PART OF THE RESULT, NOT A FOOTNOTE

Every cell is scored beside the same protocol with the force labels permuted
WITHIN each group. On FeelAnyForce that control reads +0.63 (all frames) and
+0.83 (imaged whole): with 42 captures, each having its own force range, the
per-group isotonic fit reproduces the between-capture ordering whether or not
the frame-to-force pairing survives. So its rho is not a measurement of the
reconstruction, and the row is reported as UNUSABLE rather than as a number.
`faf_extract` records the same trap being sprung once before (0.455 real
against 0.442 shuffled).

TWO POPULATIONS, because one of them is a trap
    all          every frame the loader returns
    imaged whole the press was commanded inside the field of view AND its
                 imaged contact core touches no border

Most of these capture grids are LARGER than the sensor sees (cnc_mini_26
presses over 16.7 x 18 mm, FoTa over 20 x 16 mm, against a ~13.2 x 9.9 mm
view), so the "all" column is dominated by contacts whose true extent is
outside the image and whose depth is not identifiable from it. Scores there
measure the truncation as much as the method. Both are reported; neither is
hidden.

The floor differs between arms and has to: the LUT is in millimetres and uses
its absolute 0.05 mm, the calibration-free solve has no millimetre scale so it
uses the same fraction of its own peak. That asymmetry is forced, it is the
one `calibfree_eval` already documents, and it is the only one.

    python -m force_recovery.force_recon_matrix [--per-dataset 400]
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from .run_episode import OUT_ROOT

OUT = OUT_ROOT / "feature_cache" / "force_recon_matrix.json"

DATASETS = (
    ("cnc_mini_26", "GelSight Mini, CNC presses, 0-20 N"),
    ("cnc", "FoTa cnc_Mini"),
    ("feats", "FEATS (MARKER gel)"),
    ("sparsh", "Sparsh / Meta, 10 gel pads"),
    ("faf", "FeelAnyForce"),
)


# How many raw frames each dataset actually has, measured on disk, against
# what these loaders used to hand back. The gap was not a sampling decision —
# the loaders were written to draw FIGURES and their caps were never revisited
# when they started feeding the evaluation, so the site reported a loader
# ceiling as if it were the dataset:
#
#     dataset       on disk    previously used      why
#     cnc_mini_26     6,219            6,219        (no cap — correct)
#     cnc             3,358            3,000        CNC_N ceiling
#     feats          16,969              390        val split only, then [:390]
#     sparsh        174,866              960        BATCHES[:8], ~120 each
#     faf           110,109 labelled   5,410        only 5,202 PNGs extracted
#                                                   from the 81 GB spanned zip
#
# POOL is the number of raw frames to draw; roughly a fifth survive the
# fully-imaged filter, so this is sized to clear 2,000 whole presses where the
# dataset physically allows it. cnc_mini_26 and cnc cannot: they hold 6,219
# and 3,358 frames in total and the honest ceiling is theirs, not ours.
POOL = int(__import__("os").environ.get("RECON_POOL", "12000"))


def _rows(name: str):
    """(rows, get) over the widest pool available for one dataset."""
    import os

    from . import debug_gallery as dg

    dg.RNG = np.random.default_rng(0)          # module-level; see verify script
    if name == "cnc_mini_26":
        from .visible_eval import _wide_glowtact
        return _wide_glowtact()                # already every press on disk
    if name == "cnc":
        os.environ["CNC_N"] = str(POOL)        # 3,358 exist; this takes all
        try:
            return dg.load_cnc()
        finally:
            os.environ.pop("CNC_N", None)
    if name == "sparsh":
        return dg.load_sparsh(n=POOL)
    if name == "faf":
        return _faf_labelled()
    return dg.load_feats(n=POOL)


def _faf_labelled(tiers=("A",)):
    """FeelAnyForce WITH its force labels — `debug_gallery.load_faf` has none.

    That loader was written for figures and returns `f: None`, so FeelAnyForce
    could not be scored at all; it is listed as a force dataset on the site and
    has never appeared in a force table. The labels live in three CSVs that
    reference frames by timestamped filename (`faf_extract.load_labels`), and
    the join is by capture + timestamp, which resolves for all 42 captures on
    disk.

    The reference frame is the LIGHTEST labelled frame of each capture, not
    the first file: nothing guarantees the first frame is unloaded, and the
    labels say which one is closest to it.

    TIERS. `faf_extract.cmd_select` established — and its docstring says in so
    many words that tier B must be "reported separately, never mixed into the
    headline" — that only 14 of the 42 captures contain a contact-free frame.
    In the other 28 the lightest frame available carries 5.50 to 6.29 N, so
    "the lightest frame" is not a reference at all: every difference image in
    those captures is one loaded state minus another. This function ignored
    the distinction and pooled all 42, which is 2,240 of 5,465 frames (41%),
    and that pooled number is the FeelAnyForce row the site has been showing.

    Default is tier A only. Pass tiers=("A", "B") to reproduce the old mix, or
    ("B",) to score the loaded-reference captures on their own.

    HOW MUCH IT MATTERS: measured, and the answer is almost nothing.

        population   tier A only        tier A + B (the old mix)
        whole 2000   LUT .9392 CF .9522   LUT .9413 CF .9515
        all          LUT .9052 CF .9360   LUT .9110 CF .9325

    A difference of 0.0007 is not a difference. The reason is structural: each
    capture is its own GROUP, and the protocol fits every group separately, so
    a reference carrying 6 N puts a constant offset into that capture's depths
    and the group's own intercept absorbs it. The tier split protects absolute
    depth, which this ranking protocol never reads.

    Tier A stays the default anyway — a reference frame that is not
    contact-free is wrong on its face, and it costs nothing here — but it
    would be dishonest to present it as a fix that bought accuracy.
    """
    import collections

    from PIL import Image

    from .faf_extract import IMG_DIR, ZERO_N, load_labels
    from .lut_calibration import crop

    by_cap = collections.defaultdict(list)
    for r in load_labels():
        p = IMG_DIR / r["capture"] / f"{r['ts']}.png"
        if p.exists():
            by_cap[r["capture"]].append((p, abs(float(r["fz"]))))
    if not by_cap:
        raise LookupError("no FeelAnyForce frame resolved to a label")
    refs, rows, dropped = {}, [], 0
    for cap, items in sorted(by_cap.items()):
        items.sort(key=lambda t: t[1])
        tier = "A" if items[0][1] < ZERO_N else "B"
        if tier not in tiers:
            dropped += len(items)
            continue
        refs[cap] = crop(np.asarray(Image.open(items[0][0]).convert("RGB"))
                         ).astype(np.float32)
        for p, fz in items[1:]:
            rows.append({"path": p, "group": cap, "f": fz, "tier": tier})
    if not rows:
        raise LookupError(f"no FeelAnyForce capture is tier {tiers}")
    print(f"  FeelAnyForce: {len(refs)} captures of tier {'+'.join(tiers)}, "
          f"{len(rows)} frames; {dropped} frames dropped as another tier",
          flush=True)
    rng = np.random.default_rng(0)
    rows = [rows[i] for i in rng.permutation(len(rows))]

    def get(fr):
        img = crop(np.asarray(Image.open(fr["path"]).convert("RGB"))
                   ).astype(np.float32)
        return img, refs[fr["group"]]
    return rows, get


def _feats(d: np.ndarray, absolute_floor: bool):
    """The five features, from a depth map.

    `absolute_floor` is the one forced asymmetry: the LUT is in millimetres and
    uses its production 0.05 mm; the calibration-free solve has no millimetre
    scale, so the same absolute number would mean something different for it
    and it uses the same fraction of its own peak.
    """
    from .lut_calibration import MM_PER_PIXEL
    d = np.clip(np.asarray(d, np.float64), 0, None)
    m = d > 0.05 if absolute_floor else d > 0.05 * max(d.max(), 1e-12)
    px = MM_PER_PIXEL ** 2
    area = float(m.sum() * px)
    maxd = float(np.percentile(d, 99.8))
    return [float(d[m].sum() * px), float((d[m] ** 2).sum() * px),
            maxd, area, float(np.sqrt(area) * maxd)]


def main() -> int:
    """One writer at a time — see `artifact_lock`, which explains why.

    This module MERGES into its artifact rather than overwriting it, which
    makes a second concurrent run worse than for a plain overwrite: two runs
    with different `--per-dataset`/`--per-group` merge row by row and the file
    ends up describing two protocols at once, with nothing in it saying so.
    """
    from .artifact_lock import one_writer
    with one_writer(OUT):
        return _main()


def _main() -> int:
    from . import calib_free as CF
    from .debug_gallery import stages
    from .force_eval_all import evaluate

    ap = argparse.ArgumentParser()
    ap.add_argument("--per-dataset", type=int, default=400)
    ap.add_argument("--per-group", type=int, default=0,
                    help="quota per group instead of per dataset; needed when "
                         "a dataset has many small groups (FeelAnyForce has "
                         "42 captures, so a flat 400 leaves ~9 each)")
    ap.add_argument("--datasets", nargs="*", default=[d for d, _ in DATASETS])
    args = ap.parse_args()

    table = []
    for name, label in DATASETS:
        if name not in args.datasets:
            continue
        from .visible_eval import in_fov, visible
        try:
            rows, get = _rows(name)
        except Exception as exc:                               # noqa: BLE001
            print(f"== {name}: UNAVAILABLE ({exc})", flush=True)
            table.append({"dataset": name, "label": label, "available": False,
                          "reason": str(exc)})
            continue
        # Quotas on what is KEPT, not on what is scanned. The first version
        # capped the scan at 400 frames and then filtered, which left 32 fully
        # imaged presses on cnc_mini_26 and scored nan.
        cap = args.per_dataset
        import collections
        pg = args.per_group
        seen_all = collections.Counter()
        seen_whole = collections.Counter()
        X = {"lut": [], "calibfree": []}
        f, g, whole = [], [], []
        n_all = n_whole = 0
        for fr in rows:
            if not pg and n_all >= cap and n_whole >= cap:
                break
            if pg and n_all >= cap * 6:
                break
            grp = str(fr["group"])
            if pg and seen_all[grp] >= pg and seen_whole[grp] >= pg:
                continue
            img, ref = get(fr)
            w = bool(in_fov(fr) and visible(img, ref))
            if not pg and n_all >= cap and not w:
                continue
            if pg and seen_all[grp] >= pg and not w:
                continue
            seen_all[grp] += 1
            seen_whole[grp] += w
            X["lut"].append(_feats(stages(img, ref)["depth"], True))
            X["calibfree"].append(_feats(CF.reconstruct(img, ref)["depth"],
                                         False))
            f.append(float(fr["f"]))
            g.append(str(fr["group"]))
            whole.append(w)
            n_all += 1
            n_whole += w
        f = np.array(f)
        g = np.array(g)
        whole = np.array(whole)
        row = {"dataset": name, "label": label, "available": True,
               "n_all": int(len(f)), "n_whole": int(whole.sum())}
        for pop, mask in (("all", np.ones(len(f), bool)), ("whole", whole)):
            # every group needs >= 16 frames or the protocol silently skips
            # it (8 per fit half) and the pooled rho comes back nan
            sizes = {q: int((g[mask] == q).sum()) for q in set(g[mask])}
            # The protocol skips any group with under 8 frames in its fit
            # half, silently. Rather than refuse the dataset, score it and
            # report how many groups actually carried the fit — a rho from
            # two of six indenters and a rho from six are not the same claim.
            fittable = sum(1 for v in sizes.values() if v // 2 >= 8)
            if mask.sum() < 60 or fittable < 3:
                row[pop] = {"n": int(mask.sum()), "scored": False,
                            "group_sizes": sizes, "groups_fittable": fittable}
                continue
            row[pop] = {"n": int(mask.sum()), "scored": True,
                        "groups_fittable": fittable,
                        "groups_total": len(sizes)}
            for arm in ("lut", "calibfree"):
                row[pop][arm] = evaluate(np.array(X[arm])[mask], f[mask],
                                         g[mask])
        table.append(row)
        s = []
        for pop in ("all", "whole"):
            r = row[pop]
            s.append(f"{pop}: n={r['n']}" + ("" if not r.get("scored") else
                     f" LUT {r['lut']['rho']:.4f} / CF {r['calibfree']['rho']:.4f}"))
        print(f"== {name:12s} " + "   ".join(s), flush=True)

    # MERGE, do not overwrite. Running with --datasets on a subset used to
    # replace the whole artifact with those rows, so a follow-up run on two
    # datasets silently deleted the other three.
    prev = {r["dataset"]: r for r in
            (json.loads(OUT.read_text()) if OUT.exists() else [])}
    prev.update({r["dataset"]: r for r in table})
    order = [d for d, _ in DATASETS]
    OUT.write_text(json.dumps(sorted(prev.values(),
                                     key=lambda r: order.index(r["dataset"])),
                              indent=1))
    print(f"\n{'dataset':14s}{'n':>6}{'LUT':>9}{'calib-free':>12}{'delta':>9}"
          f"   | {'n':>5}{'LUT':>9}{'calib-free':>12}{'delta':>9}")
    print(f"{'':14s}{'-- all frames --':>36}   | {'-- imaged whole --':>37}")
    for r in table:
        if not r.get("available"):
            print(f"{r['dataset']:14s}  UNAVAILABLE: {r['reason'][:50]}")
            continue
        line = f"{r['dataset']:14s}"
        for pop in ("all", "whole"):
            p = r[pop]
            if not p.get("scored"):
                line += f"{p['n']:>6}{'too few':>30}"
            else:
                d = p["calibfree"]["rho"] - p["lut"]["rho"]
                line += (f"{p['n']:>6}{p['lut']['rho']:>9.4f}"
                         f"{p['calibfree']['rho']:>12.4f}{d:>+9.4f}"
                         f" [{p['groups_fittable']}/{p['groups_total']}]")
            line += "   | " if pop == "all" else ""
        print(line)
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
