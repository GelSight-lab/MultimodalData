"""Run the two supervised baselines on Sparsh, on OUR exact frame subset.

The results matrix had an empty Sparsh row ("not run — no published
predictions"). This module fills it. The only way that row is comparable to
the other three is if all three estimators see the *same frames*: the same
750-frame per-batch sample our features were cached on, filtered by the same
in-view contact-disc mask (`sparsh_probe/eval_circles.json`). So the frame
list is not re-derived here — it is read straight out of the cached feature
files, index by index.

Pipeline

    export   decode those frames once, rebuild the SAME local rest reference
             the physics path uses (median of the 6 nearest contact-free
             anchors), and dump FeelAnyForce-ready uint8 tensors to Disk1.
             FeelAnyForce lives in its own conda env (`anyforce`) that has no
             scipy/h5py/pyarrow, so the hand-off is a plain .npz.
    feats    FEATS runs in base — no export needed, it takes the raw frame.
    eval     one protocol for all three estimators: per-batch half/half +
             isotonic, 5 seeds, Spearman rho, plus the within-batch label
             shuffle control (`force_eval_all.evaluate`).

Both baselines are also scored WITHOUT any fitting, because both emit
newtons directly and Sparsh's labels are newtons — a pretrained model that
has to be rank-calibrated per gel pad before it correlates is a different
claim from one that reads out force.

    python -m force_recovery.sparsh_baselines sweep    # preprocessing A/B
    python -m force_recovery.sparsh_baselines export
    conda run -n anyforce python force_recovery/anyforce_sparsh.py
    python -m force_recovery.sparsh_baselines feats
    python -m force_recovery.sparsh_baselines eval
"""
from __future__ import annotations

import io
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import sparsh_data as sd
from .lut_calibration import crop
from .run_episode import OUT_ROOT

CACHE = OUT_ROOT / "feature_cache"
CACHE_SP = OUT_ROOT / "feature_cache_sparshlut"
PROBE = OUT_ROOT / "sparsh_probe"
EXPORT = OUT_ROOT / "sparsh_baseline_frames"
SWEEP = OUT_ROOT / "sparsh_baseline_sweep"


# ------------------------------------------------------------------ frames
def wanted(name: str) -> tuple[list[int], np.ndarray]:
    """(global frame indices, in-view flags) for one batch — from the cache.

    Reading the indices back out of the cached features rather than re-running
    `sparsh_data._stratified` guarantees the baselines are scored on the exact
    rows the physics numbers were computed on, even if the sampler is ever
    reseeded.
    """
    rows = json.loads((CACHE_SP / f"sparsh_{name}.json").read_text())["rows"]
    circ = json.loads((PROBE / "eval_circles.json").read_text())
    c = circ.get(name.replace("_batch_", "_b"), {})
    idx = [int(r["index"]) for r in rows]
    iv = np.array([bool(c.get(str(i)) and c[str(i)]["inview"]) for i in idx])
    return idx, iv


def decode_batch(probe: str, b: int, idx: list[int]):
    """Decode `idx` + the rest anchors -> (full-view, cropped, anchors, swap).

    Reproduces `sparsh_data.load_frames` exactly (rotation, the BGR batch, the
    local rest reference) but keeps the un-cropped 240x320 view as well, since
    FEATS wants the whole sensor image and does its own border crop.

    Frames stay uint8 and the 750 per-frame local references are NOT
    materialised: at float64 they are 1.8 MB each and holding two dicts of
    them pushed this machine into swap, which is what made the first pass 15
    min/batch instead of 2. `local_ref` rebuilds one on demand; the median of
    the uint8 anchors equals the median of their float32 copies, so the
    reference is bit-identical to the physics path's.
    """
    tab = sd.label_table(probe, b)
    ic = tab["in_contact"]
    anch = sd._anchors(ic)
    need = set(idx) | {int(a) for a in anch}

    full, off = {}, 0
    for f in sd.image_files(probe, b):
        with open(f, "rb") as fh:
            blob = pickle.load(fh)
        for j, buf in enumerate(blob):
            g = off + j
            if g in need:
                full[g] = np.ascontiguousarray(
                    np.rot90(np.asarray(Image.open(io.BytesIO(buf))
                                        .convert("RGB")), sd.ROT))
        off += len(blob)
        del blob
    if off != len(ic):
        raise RuntimeError(f"{probe}/batch_{b}: {off} frames vs {len(ic)} flags")

    cropped = {g: crop(a) for g, a in full.items()}
    step = max(1, len(anch) // sd.N_REST)
    ref_g = np.median(np.stack([cropped[int(a)] for a in anch[::step]])
                      .astype(np.float32), 0)
    # same BGR detector as the physics path (sharp/batch_2 is stored BGR)
    swap = bool(ref_g[..., 2].mean() > ref_g[..., 0].mean())
    if swap:
        cropped = {g: np.ascontiguousarray(a[..., ::-1])
                   for g, a in cropped.items()}
        full = {g: np.ascontiguousarray(a[..., ::-1]) for g, a in full.items()}
    return full, cropped, anch, swap


def local_ref(imgs: dict, anch: np.ndarray, g: int) -> np.ndarray:
    """Median of the 6 temporally nearest contact-free frames, as in the
    physics path (a single global rest median smears gel relaxation)."""
    near = anch[np.argsort(np.abs(anch - g))[:6]]
    return np.median(np.stack([imgs[int(a)] for a in near])
                     .astype(np.float32), 0)


def _bgsub(img: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """FeelAnyForce's background removal: clip(im - bg + 127)."""
    return np.clip(img.astype(np.float32) - ref + 127, 0, 255).astype(np.uint8)


def variants(full, cropped, anch, g, keys):
    """The four defensible FeelAnyForce inputs for a frame already at 320x240.

    Sparsh ships 320x240 portrait frames. Whether they are the raw sensor view
    (so gsdevice's 1/7 border crop still applies, as our physics path assumes)
    or already-cropped gsdevice output (so cropping again throws away a
    seventh of the pad) is not stated on the dataset card, and it changes what
    FeelAnyForce sees. Both are run; the baseline gets whichever is better.
    """
    out = {}
    for k in keys:
        if k == "crop_bg":
            out[k] = _bgsub(cropped[g], local_ref(cropped, anch, g))
        elif k == "crop_raw":
            out[k] = cropped[g]
        elif k == "full_bg":
            out[k] = _bgsub(full[g], local_ref(full, anch, g))
        elif k == "full_raw":
            out[k] = full[g]
        else:
            raise KeyError(k)
    return out


# ------------------------------------------------------------------ export
def cmd_export(names=None, keys=("crop_bg", "full_bg")) -> None:
    EXPORT.mkdir(parents=True, exist_ok=True)
    for probe, b in sd.BATCHES:
        name = f"{probe}_batch_{b}"
        if names and name not in names:
            continue
        out = EXPORT / f"{name}.npz"
        if out.exists():
            print(f"cached {out.name}")
            continue
        idx, iv = wanted(name)
        full, cropped, anch, swap = decode_batch(probe, b, idx)
        stacks = {k: [] for k in keys}
        for g in idx:
            v = variants(full, cropped, anch, g, keys)
            for k in keys:
                stacks[k].append(v[k])
        np.savez(out, index=np.array(idx), inview=iv,
                 **{k: np.stack(v) for k, v in stacks.items()})
        print(f"{name}: n={len(idx)} inview={int(iv.sum())} swap_rb={swap} "
              f"-> {out.name} ({out.stat().st_size/1e6:.0f} MB)", flush=True)


# ------------------------------------------------------------------- FEATS
def _feats_inputs(full, cropped, g):
    """FEATS candidate views. Their chain is resize(895,672) -> 1/7 crop ->
    drop last column -> resize(320,240), written for a full sensor frame."""
    from .feats_infer import preprocess
    return {"pre_full": preprocess(full[g]),
            "pre_crop": preprocess(cropped[g]),
            "asis_full": cv2.resize(full[g], (320, 240))}


FEATS_VARIANTS = ("pre_full", "pre_crop", "asis_full")
FEATS_MAIN = "pre_crop"          # the most generous of the three (see ledger)


def cmd_feats(main: str = FEATS_MAIN) -> None:
    """FEATS U-net on every sampled Sparsh frame -> feats_on_sparsh.json.

    All three input chains are stored, and `pred` is the best of them, so the
    reported failure cannot be blamed on having fed it the wrong crop.
    """
    from .feats_infer import FeatsPredictor

    pred = FeatsPredictor(device="cuda:0", shift=None)
    rows = []
    for probe, b in sd.BATCHES:
        name = f"{probe}_batch_{b}"
        idx, iv = wanted(name)
        full, cropped, _, _ = decode_batch(probe, b, idx)
        got = {}
        for v in FEATS_VARIANTS:
            ims = np.stack([_feats_inputs(full, cropped, g)[v] for g in idx])
            got[v] = pred.predict(ims, batch_size=32)
            del ims
        for j, (g, k) in enumerate(zip(idx, iv)):
            gr = got[main][j]
            rows.append({"group": name, "index": int(g), "inview": bool(k),
                         # their z is compression-negative; flip so that
                         # "pressing harder" is a larger number, as elsewhere
                         "pred": -gr.total_normal,
                         "shear": gr.total_shear,
                         "absmean": float(np.abs(gr.normal).mean()),
                         **{f"pred_{v}": -got[v][j].total_normal
                            for v in FEATS_VARIANTS}})
        print(f"{name}: {len(idx)} frames", flush=True)
    (CACHE / "feats_on_sparsh.json").write_text(json.dumps(rows))
    print(f"-> {CACHE / 'feats_on_sparsh.json'}  ({len(rows)} rows)")


def cmd_sweep() -> None:
    """Preprocessing A/B on one batch, before spending a full pass on either.

    A baseline reported under a preprocessing chain it was never trained for
    is not a result about the baseline. sphere_batch_1 is the cleanest pad
    (467 in-view frames, our rho 0.977), so if a variant carries signal it
    shows here.
    """
    from scipy.stats import spearmanr
    from .feats_infer import FeatsPredictor

    name, probe, b = "sphere_batch_1", "sphere", 1
    idx, iv = wanted(name)
    rows = json.loads((CACHE_SP / f"sparsh_{name}.json").read_text())["rows"]
    f = np.array([abs(r["fz"]) for r in rows])
    full, cropped, anch, _ = decode_batch(probe, b, idx)

    SWEEP.mkdir(parents=True, exist_ok=True)
    keys = ("crop_bg", "crop_raw", "full_bg", "full_raw")
    st = {k: [] for k in keys}
    for g in idx:
        v = variants(full, cropped, anch, g, keys)
        for k in keys:
            st[k].append(v[k])
    np.savez(SWEEP / f"{name}.npz", index=np.array(idx), inview=iv, f=f,
             **{k: np.stack(v) for k, v in st.items()})
    print(f"-> {SWEEP / (name + '.npz')} (run anyforce_sparsh.py --sweep)")

    pred = FeatsPredictor(device="cuda:0", shift=None)
    for k in ("pre_full", "pre_crop", "asis_full"):
        ims = np.stack([_feats_inputs(full, cropped, g)[k] for g in idx])
        p = np.array([-gr.total_normal for gr in pred.predict(ims, 32)])
        print(f"FEATS {k:10s} rho_all={spearmanr(p, f).statistic:+.3f} "
              f"rho_inview={spearmanr(p[iv], f[iv]).statistic:+.3f} "
              f"mean={p.mean():.3f} sd={p.std():.4f} "
              f"range=[{p.min():.3f},{p.max():.3f}]", flush=True)


# -------------------------------------------------------------- evaluation
FE = ("vol", "vol2", "maxd", "area", "h1")


def _load_pred(fname: str) -> dict:
    rows = json.loads((CACHE / fname).read_text())
    return {(r["group"], int(r["index"])): r for r in rows}


def _table(inview_only=True) -> dict:
    """Assemble truth / physics features / both baselines on identical rows."""
    out = {}
    anyf = _load_pred("anyforce_on_sparsh.json")
    fts = _load_pred("feats_on_sparsh.json")
    for probe, b in sd.BATCHES:
        name = f"{probe}_batch_{b}"
        rows = json.loads((CACHE_SP / f"sparsh_{name}.json").read_text())["rows"]
        _, iv = wanted(name)
        keep = iv if inview_only else np.ones(len(rows), bool)
        if keep.sum() < 40:
            continue
        rs = [r for r, k in zip(rows, keep) if k]
        ks = [(name, int(r["index"])) for r in rs]
        out[name] = {
            "probe": probe,
            "f": np.array([abs(r["fz"]) for r in rs]),
            "X": np.array([[r[k] for k in FE] for r in rs]),
            "anyforce": np.array([anyf[k]["pred"] for k in ks]),
            "feats": np.array([fts[k]["pred"] for k in ks]),
            "feats_absmean": np.array([fts[k]["absmean"] for k in ks]),
        }
    return out


def cmd_eval(inview_only: bool = True) -> dict:
    from scipy.stats import spearmanr
    from .force_eval_all import evaluate

    T = _table(inview_only)
    groups = np.array(sum([[k] * len(v["f"]) for k, v in T.items()], []))
    f = np.concatenate([v["f"] for v in T.values()])
    probes = np.array(sum([[v["probe"]] * len(v["f"]) for v in T.values()], []))
    cols = {
        "Ours (physics)": np.vstack([v["X"] for v in T.values()]),
        "FeelAnyForce": np.concatenate([v["anyforce"] for v in T.values()])[:, None],
        "FEATS U-net": np.concatenate([v["feats"] for v in T.values()])[:, None],
        # A pure-noise column under the IDENTICAL protocol. Needed because a
        # per-pad monotone calibration refit on 5 random half-splits gets to
        # pick its own sign per pad per seed, which lifts |rho| off zero for
        # free. Any calibrated rho at or below this line is not a result.
        "(random control)": np.random.default_rng(4).standard_normal(
            (sum(len(v["f"]) for v in T.values()), 1)),
    }
    rho = lambda a, b: float(spearmanr(a, b).statistic)          # noqa: E731

    rep = {"n": int(len(f)), "inview_only": inview_only,
           "groups": sorted(T), "calibrated": {}, "raw": {}, "per_probe": {}}

    print(f"\n=== calibrated (per-batch half/half + isotonic, 5 seeds) "
          f"n={len(f)} over {len(T)} pads ===")
    print(f"{'estimator':16s} {'rho':>7s} {'[min,max]':>16s} {'MAE_N':>8s} "
          f"{'shuffle':>8s} {'n_eval':>7s}")
    for k, X in cols.items():
        m = evaluate(X, f, groups)
        rep["calibrated"][k] = m
        print(f"{k:16s} {m['rho']:7.3f} "
              f"[{m['rho_min']:.3f},{m['rho_max']:.3f}]".ljust(41)
              + f"{m['mae']:8.3f} {m['shuffle_rho']:8.3f} {m['n_eval']:7d}")

    print("\n=== raw model output, NO fitting (baselines emit newtons) ===")
    print(f"{'estimator':16s} {'rho':>7s} {'MAE_N':>8s} {'mean':>8s} "
          f"{'sd':>8s} {'min':>8s} {'max':>8s}")
    for k in ("FeelAnyForce", "FEATS U-net"):
        p = cols[k][:, 0]
        m = {"rho": rho(p, f), "mae": float(np.abs(p - f).mean()),
             "mean": float(p.mean()), "sd": float(p.std()),
             "min": float(p.min()), "max": float(p.max()),
             "rho_vs_truth_sd_ratio": float(p.std() / f.std())}
        rep["raw"][k] = m
        print(f"{k:16s} {m['rho']:7.3f} {m['mae']:8.3f} {m['mean']:8.3f} "
              f"{m['sd']:8.4f} {m['min']:8.3f} {m['max']:8.3f}")
    print(f"{'(ground truth)':16s} {'':7s} {'':8s} {f.mean():8.3f} "
          f"{f.std():8.4f} {f.min():8.3f} {f.max():8.3f}")

    print("\n=== per indenter (calibrated rho / MAE | raw rho) ===")
    print(f"{'probe':8s} {'n':>5s} " + "".join(
        f"{k:>26s}" for k in cols))
    for pr in ("sphere", "flat", "sharp", "ALL"):
        m = np.ones(len(f), bool) if pr == "ALL" else (probes == pr)
        if m.sum() < 40:
            continue
        cells, rec = [], {}
        for k, X in cols.items():
            e = evaluate(X[m], f[m], groups[m])
            # "raw" only means anything for the pretrained baselines, which
            # emit newtons; the physics column is a feature vector.
            r0 = rho(X[m, 0], f[m]) if X.shape[1] == 1 else float("nan")
            rec[k] = dict(e, raw_rho=r0)
            cells.append(f"{e['rho']:.3f}/{e['mae']:.3f}|"
                         f"{'  n/a' if np.isnan(r0) else f'{r0:+.3f}'}")
        rep["per_probe"][pr] = rec
        print(f"{pr:8s} {int(m.sum()):5d} " + "".join(f"{c:>26s}" for c in cells))

    print("\n=== per gel pad: calibrated rho (raw rho) ===")
    rep["per_pad"] = {}
    for name in sorted(T):
        m = groups == name
        line, rec = f"{name:16s} n={int(m.sum()):4d}", {}
        for k, X in cols.items():
            e = evaluate(X[m], f[m], groups[m])
            r0 = rho(X[m, 0], f[m]) if X.shape[1] == 1 else float("nan")
            rec[k] = {"rho": e["rho"], "mae": e["mae"], "raw_rho": r0,
                      "shuffle_rho": e["shuffle_rho"]}
            line += f"  {k}={e['rho']:+.3f}/sh{e['shuffle_rho']:+.3f}"
            line += "     " if np.isnan(r0) else f"/raw{r0:+.3f}"
        rep["per_pad"][name] = rec
        print(line)

    # ---- is a "collapsed" baseline actually collapsed, or just miscalibrated?
    print("\n=== failure mechanism: is the output a constant? ===")
    print(f"{'estimator':16s} {'p50':>7s} {'IQR':>7s} {'IQR/GT':>7s} "
          f"{'frac within':>12s} {'MAE vs':>8s} {'MAE of':>8s}")
    print(f"{'':16s} {'':7s} {'':7s} {'':7s} {'+-20% of p50':>12s} "
          f"{'truth':>8s} {'best c':>8s}")
    q = lambda v: float(np.percentile(v, 75) - np.percentile(v, 25))  # noqa
    for k in ("FeelAnyForce", "FEATS U-net"):
        p = cols[k][:, 0]
        med = float(np.median(p))
        near = float(np.mean(np.abs(p - med) <= 0.2 * abs(med)))
        best_c = float(np.abs(np.median(f) - f).mean())   # best constant in MAE
        m = {"p50": med, "iqr": q(p), "iqr_ratio": q(p) / q(f),
             "frac_within_20pct_of_median": near,
             "mae": float(np.abs(p - f).mean()), "mae_best_constant": best_c}
        rep["raw"][k].update(m)
        print(f"{k:16s} {med:7.3f} {q(p):7.3f} {q(p)/q(f):7.2f} {near:12.2f} "
              f"{m['mae']:8.3f} {best_c:8.3f}")
    print(f"{'(ground truth)':16s} {np.median(f):7.3f} {q(f):7.3f} "
          f"{1.0:7.2f}")

    # ---- how many labelled frames does each estimator need on this sensor?
    # The per-pad refit above gives our pipeline 10 calibrations and gives the
    # pretrained baselines a free extra one. This block removes that: fit ONCE
    # on sphere_batch_1 and apply the frozen map everywhere, against the
    # baselines' untouched newtons (zero Sparsh labels).
    print("\n=== one calibration for the whole sensor (fit on sphere_batch_1, "
          "frozen elsewhere) ===")
    from .force_eval_all import _fit_apply
    fitm = groups == "sphere_batch_1"
    ev = ~fitm
    print(f"{'estimator':16s} {'labels used':>12s} {'rho':>7s} {'MAE_N':>8s}")
    rep["frozen"] = {}
    for k, X in cols.items():
        if k == "(random control)":
            continue
        p = _fit_apply(X[fitm], f[fitm], X[ev])
        rep["frozen"][k] = {"rho": rho(p, f[ev]),
                            "mae": float(np.abs(p - f[ev]).mean()),
                            "n_fit": int(fitm.sum()), "n_eval": int(ev.sum())}
        print(f"{k:16s} {int(fitm.sum()):12d} {rho(p, f[ev]):7.3f} "
              f"{np.abs(p - f[ev]).mean():8.3f}")
    for k in ("FeelAnyForce", "FEATS U-net"):
        p = cols[k][ev, 0]
        rep["frozen"][k + " (0 labels)"] = {
            "rho": rho(p, f[ev]), "mae": float(np.abs(p - f[ev]).mean()),
            "n_fit": 0, "n_eval": int(ev.sum())}
        print(f"{k + ' (raw)':16s} {0:12d} {rho(p, f[ev]):7.3f} "
              f"{np.abs(p - f[ev]).mean():8.3f}")

    tag = "" if inview_only else "_allframes"
    p = CACHE / f"sparsh_baselines{tag}.json"
    p.write_text(json.dumps(rep, indent=1, default=float))
    print(f"\n-> {p}")
    return rep


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "eval"
    if cmd == "sweep":
        cmd_sweep()
    elif cmd == "export":
        cmd_export()
    elif cmd == "feats":
        cmd_feats(sys.argv[2] if len(sys.argv) > 2 else "pre_full")
    elif cmd == "eval":
        cmd_eval(inview_only=("all" not in sys.argv[2:]))
    else:
        raise SystemExit(f"unknown command {cmd}")
