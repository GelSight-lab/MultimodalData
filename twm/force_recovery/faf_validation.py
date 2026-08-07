"""FeelAnyForce force validation on the ORIGINAL, exactly-joined frames.

This is the re-test of a dataset we had excluded. The exclusion was earned:
with the labels re-attached by an INFERRED per-capture frame index, the
pooled rho was 0.455 while the same protocol with labels shuffled WITHIN each
capture scored 0.442, and per-capture rho was ~0.09 -- i.e. the frame->force
mapping carried essentially nothing and only between-capture structure was
being scored. `faf_extract.py` replaces that guess with the archive's own
timestamped filenames, which is the key the label CSVs actually use, so the
join is exact by construction (110,109/110,109 rows resolve to an entry).

Nothing else changes: same `debug_gallery.stages()` force path, same
per-group half/half + isotonic protocol from `force_eval_all.evaluate`, same
5 seeds, and the within-capture shuffle control is printed beside every
headline number. The verdict rule is fixed in advance -- if the real join
does not clear its own shuffle control by a wide margin, FeelAnyForce stays
excluded.

Image convention: the archive's tactile PNGs are natively 320x240, which is
already our (W, H), so frames are used as-is with no crop and no resize. The
GlowTact border crop is not applied because these are not GlowTact frames;
this matches how FEATS frames are handled.

Two tiers, decided by the labels and not by the results (see
`faf_extract.cmd_select`): tier A = the 14 captures that contain contact-free
frames (|Fz| < 0.1 N) and therefore have an honest per-session reference;
tier B = the 28 captures whose MINIMUM |Fz| is 4.9-6.0 N, which have no
unloaded frame at all and can only use a median-image reference. Tier A is
also run with the median-image reference so the cost of a contaminated
reference is measured on identical frames instead of argued.

Run:
  python -m force_recovery.faf_validation features   # pipeline -> feature cache
  python -m force_recovery.faf_validation eval       # tables + metrics json
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from .faf_extract import IMG_DIR, PLAN_JSON, ROOT, local_path
from .force_eval_all import FE, evaluate
from .lut_calibration import H, W, detect_circle

CACHE = ROOT / "feature_cache"
OUT_JSON = CACHE / "faf_metrics.json"
MARGIN = 10           # px clearance from the frame border, as on Sparsh/cnc
N_MEDIAN_REF = 60     # frames averaged for the median-image reference


def _img(key: str) -> np.ndarray:
    a = np.asarray(Image.open(local_path(key)).convert("RGB"))
    if a.shape[:2] != (H, W):
        raise ValueError(f"{key}: expected {H}x{W}, got {a.shape[:2]}")
    return a.astype(np.float32)


# ------------------------------------------------------------------ features
def _rows_for_capture(cap: str, evs: list[dict], refs: list[dict],
                      tier: str) -> dict:
    """Run the force pipeline over one capture, both reference variants."""
    from .debug_gallery import stages

    # zero-force reference, per session: median of the N_REF lightest frames
    zero_ref = {}
    by_ses = defaultdict(list)
    for r in refs:
        by_ses[r["session"]].append(r)
    for s, rs in by_ses.items():
        zero_ref[s] = np.median(np.stack([_img(r["key"]) for r in rs]), 0)

    # median-image reference: the FEATS-style fallback, the only thing tier B
    # can have. Built from the sampled frames themselves (unsupervised, no
    # labels touched), so it costs no extra download.
    rng = np.random.default_rng(0)
    sel = [evs[i] for i in rng.permutation(len(evs))[:N_MEDIAN_REF]]
    med_ref = np.median(np.stack([_img(r["key"]) for r in sel]), 0)

    out = []
    for e in evs:
        img = _img(e["key"])
        ref = zero_ref.get(e["session"]) if tier == "A" else None
        rec = {"key": e["key"], "ts": e["ts"], "session": e["session"],
               "split": e["split"], "fz": e["fz"], "f": abs(e["fz"]),
               "has_zero_ref": ref is not None}
        for tag, rf in (("", ref), ("med_", med_ref)):
            if rf is None:
                continue
            st = stages(img, rf)
            rec.update({tag + k: v for k, v in st["feats"].items()})
            rec[tag + "cov"] = st["lut_coverage"]
            # shape-agnostic visibility, from the pipeline's own valid mask:
            # `detect_circle` only fires on round contacts, and most
            # FeelAnyForce indenters are cylinders / crosses / rings / cube
            # edges, so a disc-only criterion would call 3/4 of the frames
            # "no contact". The mask border test asks the question that
            # actually matters -- is the contact clipped by the crop.
            v = st["valid"]
            rec[tag + "mask_px"] = int(v.sum())
            rec[tag + "edge_px"] = int(v[:MARGIN].sum() + v[-MARGIN:].sum()
                                       + v[:, :MARGIN].sum()
                                       + v[:, -MARGIN:].sum())
            det = detect_circle(img - rf)
            if det is None:
                rec[tag + "cx"] = rec[tag + "cy"] = rec[tag + "a_px"] = None
                rec[tag + "inview"] = False
            else:
                cx, cy, a = det
                rec[tag + "cx"], rec[tag + "cy"], rec[tag + "a_px"] = cx, cy, a
                rec[tag + "inview"] = bool(
                    MARGIN + a < cx < W - MARGIN - a
                    and MARGIN + a < cy < H - MARGIN - a)
        out.append(rec)
    ref_rgb = ((zero_ref[sorted(zero_ref)[0]] if zero_ref else med_ref)
               .mean((0, 1)))
    return {"capture": cap, "tier": tier,
            "n_sessions_with_ref": len(zero_ref),
            "ref_rgb": [float(v) for v in ref_rgb],
            "med_ref_rgb": [float(v) for v in med_ref.mean((0, 1))],
            "rows": out}


def _job(args):
    return _rows_for_capture(*args)


def cmd_features() -> None:
    """Run `debug_gallery.stages` over every cached frame, cache the features.

    Written per capture as `feature_cache/faf_<capture>.json` in the same
    {meta..., "rows": [...]} shape as the Sparsh caches, so the site consumes
    it with the loader it already has.
    """
    import multiprocessing as mp

    plan = json.loads(PLAN_JSON.read_text())
    tier = {r["capture"]: r["tier"] for r in plan["per_capture"]}
    ev, rf = defaultdict(list), defaultdict(list)
    for e in plan["eval"]:
        ev[e["capture"]].append(e)
    for e in plan["ref"]:
        rf[e["capture"]].append(e)
    missing = [e for e in plan["eval"] + plan["ref"]
               if not local_path(e["key"]).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} planned frames are not cached — "
                         f"run `faf_extract fetch` first")
    CACHE.mkdir(parents=True, exist_ok=True)
    jobs = [(c, ev[c], rf[c], tier[c]) for c in sorted(ev)]
    with mp.Pool(8) as pool:
        for i, res in enumerate(pool.imap_unordered(_job, jobs)):
            (CACHE / f"faf_{res['capture']}.json").write_text(json.dumps(res))
            print(f"  [{i+1}/{len(jobs)}] {res['capture']:22s} tier "
                  f"{res['tier']} n={len(res['rows'])} "
                  f"refs={res['n_sessions_with_ref']}", flush=True)
    print(f"-> {CACHE}/faf_*.json")


def load_features() -> list[dict]:
    out = []
    for p in sorted(CACHE.glob("faf_*.json")):
        if p.name == OUT_JSON.name:
            continue
        d = json.loads(p.read_text())
        for r in d["rows"]:
            r["capture"], r["tier"] = d["capture"], d["tier"]
        out += d["rows"]
    return out


# ---------------------------------------------------------------- evaluation
def _X(rows, pre=""):
    return np.array([[r[pre + k] for k in FE] for r in rows], float)


def _run(rows, pre="", label=""):
    if len(rows) < 40:
        return None
    res = evaluate(_X(rows, pre), np.array([r["f"] for r in rows]),
                   np.array([r["capture"] for r in rows]))
    res["label"] = label
    return res


def _temporal(rows, pre=""):
    """Control: fit on each capture's EARLIEST half, evaluate on the LATEST.

    The headline split is random within a capture, and the sampled frames come
    from long recordings, so neighbouring frames can land on opposite sides of
    it. The model is a 5-term linear fit plus isotonic (6 effective degrees of
    freedom) and cannot memorise individual frames, but a time-blocked split
    removes the objection outright: no evaluation frame has a temporal
    neighbour in the fit set, and the gel has drifted in between.
    """
    from scipy.stats import spearmanr
    from sklearn.isotonic import IsotonicRegression

    T, P = [], []
    for cap in sorted({r["capture"] for r in rows}):
        s = sorted([r for r in rows if r["capture"] == cap],
                   key=lambda r: int(r["ts"]))
        h = len(s) // 2
        if h < 8:
            continue
        Xf, Xe = _X(s[:h], pre), _X(s[h:], pre)
        ff = np.array([r["f"] for r in s[:h]])
        fe = np.array([r["f"] for r in s[h:]])
        w, *_ = np.linalg.lstsq(Xf, ff, rcond=None)
        iso = IsotonicRegression(out_of_bounds="clip").fit(Xf @ w, ff)
        P.append(iso.predict(Xe @ w))
        T.append(fe)
    t, p = np.concatenate(T), np.concatenate(P)
    return {"rho": float(spearmanr(p, t).statistic),
            "mae": float(np.abs(p - t).mean()), "n_eval": int(len(t))}


def _shift_labels(rows, k):
    """Re-label each frame with the force recorded k frames later.

    This is the control aimed straight at the thing that killed the first
    attempt. A shuffle asks "do the features carry force at all"; this asks
    "is THIS frame's force the one we attached", by re-running the identical
    protocol on a join that is wrong by a fixed, small offset while keeping
    the capture, the ordering and the marginal force distribution intact.
    The offset is taken in the capture's own full timeline (all 101,883
    labelled frames), not in our subsample, so k=1 really is the neighbouring
    exposure. If a 1-frame slip is already visible, the exact filename join
    is doing the work and is not decorative.
    """
    from .faf_extract import joined, sessions

    by_cap = defaultdict(list)
    for r in joined():
        by_cap[r["capture"]].append(r)
    order, pos = {}, {}
    for cap, rs in by_cap.items():
        rs = sessions(rs)
        order[cap] = rs
        for i, r in enumerate(rs):
            pos[r["key"]] = i
    out = []
    for r in rows:
        seq = order[r["capture"]]
        i = pos[r["key"]]
        j = min(i + k, len(seq) - 1) if i + k < len(seq) else i - k
        q = dict(r)
        q["f"] = abs(seq[j]["fz"])
        out.append(q)
    return out


def _line(name, res):
    if res is None:
        return f"{name:46s} {'(too few frames)':>40s}"
    # a shifted-label capture can go constant, which makes its rho undefined;
    # drop those instead of poisoning the summary with nan
    pg = np.array([v for v in res["per_group_rho"].values() if np.isfinite(v)])
    return (f"{name:46s} {res['n_eval']:6d} {res['n_groups']:4d} "
            f"{res['rho']:7.3f} {res['shuffle_rho']:9.3f} {res['mae']:8.2f} "
            f"{np.median(pg):8.3f} [{pg.min():.2f},{pg.max():.2f}] "
            f"n={len(pg)}")


def cmd_eval() -> None:
    rows = load_features()
    A = [r for r in rows if r["tier"] == "A"]
    B = [r for r in rows if r["tier"] == "B"]
    res = {}

    print("=" * 118)
    print("FeelAnyForce, exact filename join. Protocol: per-capture half/half "
          "least-squares on [vol, vol2, maxd,")
    print("area, sqrt(area)*maxd] + isotonic, 5 seeds, median. 'shuffle' = "
          "SAME protocol, labels permuted WITHIN")
    print("each capture. The inferred-index join this replaces scored "
          "0.455 vs 0.442 shuffled.")
    print("=" * 118)
    print(f"{'subset':46s} {'n':>6s} {'cap':>4s} {'rho':>7s} {'shuffle':>9s} "
          f"{'MAE[N]':>8s} {'per-capture rho med [min,max]':>26s}")

    res["tierA_zero_ref"] = _run(A, "", "tier A, contact-free reference")
    print(_line("A  clean ref (14 captures, |Fz|min<0.1N)",
                res["tierA_zero_ref"]))
    res["tierA_median_ref"] = _run(A, "med_", "tier A, median-image reference")
    print(_line("A  median-image ref, SAME frames",
                res["tierA_median_ref"]))
    res["tierB_median_ref"] = _run(B, "med_", "tier B, median-image reference")
    print(_line("B  median-image ref (28 caps, |Fz|min>4.8N)",
                res["tierB_median_ref"]))
    res["all_median_ref"] = _run(A + B, "med_", "all 42, median-image ref")
    print(_line("A+B median-image ref (all 42 captures)",
                res["all_median_ref"]))

    # ---- join-perturbation control
    print("\nJoin-perturbation control (tier A, clean ref): re-label each "
          "frame with the force k frames later")
    print(f"{'offset':46s} {'n':>6s} {'cap':>4s} {'rho':>7s} {'shuffle':>9s} "
          f"{'MAE[N]':>8s} {'per-capture rho med [min,max]':>26s}")
    for k in (1, 5, 25):
        rk = _run(_shift_labels(A, k), "", f"tier A, labels off by {k} frames")
        res[f"tierA_shift{k}"] = rk
        print(_line(f"A  labels shifted by {k} frame(s)", rk))

    # ---- temporal control (no fit/eval neighbours)
    print("\nTemporal control (fit = each capture's earliest half, "
          "eval = latest half):")
    for nm, sub, pre in (("A clean ref", A, ""), ("A median ref", A, "med_"),
                         ("B median ref", B, "med_")):
        t = _temporal(sub, pre)
        res[f"temporal_{nm.replace(' ', '_')}"] = t
        print(f"  {nm:16s} n={t['n_eval']:5d}  rho={t['rho']:.3f}  "
              f"MAE={t['mae']:.2f} N")

    # ---- in-view split (the effect already seen on cnc, Sparsh, Tactile MNIST)
    print(f"\nContact visibility, tier A, clean ref ({MARGIN}px border):")
    det = [r for r in A if r.get("a_px") is not None]
    ivd = [r for r in A if r.get("inview")]
    print(f"  disc detector: fires on {len(det)}/{len(A)} ({len(det)/len(A):.1%})"
          f", of which {len(ivd)} in view. Only sphere3 / sphere28 / lemon "
          f"press a round contact, so this criterion is shape-limited here.")
    con = [r for r in A if r["mask_px"] > 200]
    iv = [r for r in con if r["edge_px"] <= 20]
    cl = [r for r in con if r["edge_px"] > 20]
    nc = [r for r in A if r["mask_px"] <= 200]
    print(f"  contact mask: contact on {len(con)}/{len(A)} "
          f"({len(con)/len(A):.1%}); fully inside the crop {len(iv)} "
          f"({len(iv)/len(A):.1%}); touching the border {len(cl)} "
          f"({len(cl)/len(A):.1%}); no contact {len(nc)} "
          f"({len(nc)/len(A):.1%})")
    for nm, sub in (("median |Fz| in view", iv), ("touching border", cl),
                    ("no contact", nc)):
        if sub:
            print(f"    {nm:22s} {np.median([r['f'] for r in sub]):6.2f} N")
    res["tierA_inview"] = _run(iv, "", "tier A, contact fully inside the crop")
    res["tierA_clipped"] = _run(cl, "", "tier A, contact touching the border")
    res["tierA_inview_disc"] = _run(ivd, "", "tier A, in-view disc detector")
    print(f"{'subset':46s} {'n':>6s} {'cap':>4s} {'rho':>7s} {'shuffle':>9s} "
          f"{'MAE[N]':>8s} {'per-capture rho med [min,max]':>26s}")
    print(_line("A  contact fully inside the crop", res["tierA_inview"]))
    print(_line("A  contact touching the border", res["tierA_clipped"]))
    print(_line("A  in-view disc (round indenters only)",
                res["tierA_inview_disc"]))

    # ---- per-capture detail
    print("\nPer-capture (tier A, clean ref):")
    print(f"{'capture':22s} {'n':>5s} {'rho':>7s} {'rhoMed':>7s} "
          f"{'min|Fz|':>8s} {'max|Fz|':>8s} {'inview%':>8s} {'disc%':>7s}")
    pg = res["tierA_zero_ref"]["per_group_rho"]
    pgm = res["tierA_median_ref"]["per_group_rho"]
    for cap in sorted({r["capture"] for r in A}):
        s = [r for r in A if r["capture"] == cap]
        f = np.array([r["f"] for r in s])
        print(f"{cap:22s} {len(s):5d} {pg.get(cap, float('nan')):7.3f} "
              f"{pgm.get(cap, float('nan')):7.3f} "
              f"{f.min():8.3f} {f.max():8.2f} "
              f"{np.mean([r['mask_px'] > 200 and r['edge_px'] <= 20 for r in s]):8.1%}"
              f"{np.mean([bool(r.get('inview')) for r in s]):7.1%}")
    print("\nPer-capture (tier B, median ref — no contact-free frame exists):")
    pgb = res["tierB_median_ref"]["per_group_rho"]
    for cap in sorted({r["capture"] for r in B}):
        s = [r for r in B if r["capture"] == cap]
        f = np.array([r["f"] for r in s])
        print(f"{cap:22s} {len(s):5d} {pgb.get(cap, float('nan')):7.3f} "
              f"{'':7s} {f.min():8.3f} {f.max():8.2f}")

    res["counts"] = {"tierA_frames": len(A), "tierB_frames": len(B),
                     "tierA_captures": len({r["capture"] for r in A}),
                     "tierB_captures": len({r["capture"] for r in B}),
                     "inview_frac_tierA": len(iv) / len(A),
                     "contact_frac_tierA": len(con) / len(A),
                     "disc_detected_frac_tierA": len(det) / len(A)}
    res["protocol"] = ("per-capture half/half least squares on "
                       "[vol, vol2, maxd, area, sqrt(area)*maxd] + isotonic, "
                       "5 seeds, median; control = labels permuted WITHIN "
                       "each capture; frames 320x240 native, no crop/resize")
    OUT_JSON.write_text(json.dumps(res, indent=1))
    print(f"\n-> {OUT_JSON}")


CMDS = {"features": cmd_features, "eval": cmd_eval}

if __name__ == "__main__":
    CMDS[sys.argv[1] if len(sys.argv) > 1 else "eval"]()
