"""Sparsh / TacBench force dataset (facebook/gelsight-force-estimation).

Why this dataset: it is the first external force set we found whose
frame<->force join is *explicit*. Every trajectory ships `indexes`, the global
frame index into the concatenation of the per-batch image pickles, so there is
nothing to infer. Contrast FeelAnyForce, where we had to guess the frame order
and a within-group label shuffle reproduced the "signal" almost exactly
(0.455 real vs 0.442 shuffled) — the join there was never demonstrated.

Structure on HF:

    {sphere/batch_1..6, flat/batch_1..2, sharp/batch_1..2}/
        dataset_gelsight_00..03.pkl   list of PNG-encoded bytes, 5000 each
                                      (last file short; flat/batch_2 has only 3)
        dataset_slip_forces.pkl       {'in_contact': (Nframes,) flag per frame,
                                       'trajectories': {tid: {'indexes' (N,),
                                       'forces' (N,3) Fx,Fy,Fz [N], 'poses',
                                       'slip_label', 'coef_friction', ...}}}

    org_dataset_gelsight_*.pkl are the 1.6 GB uncompressed originals and are
    NOT downloaded — the non-org files are the same frames at 2.2 GB total.

Frames decode to (320, 240, 3) uint8, i.e. PORTRAIT. The GelSight Mini pad is
landscape, so every frame is rotated before it enters the pipeline; ROT is
picked empirically (see `cmd_orient`).

Reference (rest) frame: `in_contact == 0` marks contact-free frames — 22-39%
of every batch — so the rest frame is the pixelwise median of a spread sample
of those, per batch. No hand-picked "lightest press" heuristic needed.

Commands (run from twm/):
    python -m force_recovery.sparsh_data download
    python -m force_recovery.sparsh_data verify
    python -m force_recovery.sparsh_data orient
    python -m force_recovery.sparsh_data features
    python -m force_recovery.sparsh_data eval
"""
from __future__ import annotations

import io
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1]))
from force_recovery.debug_gallery import stages, feat_vec  # noqa: E402
from force_recovery.lut_calibration import crop  # noqa: E402
from force_recovery.run_episode import OUT_ROOT  # noqa: E402

REPO = "facebook/gelsight-force-estimation"
ROOT = OUT_ROOT / "fb_gs"
CACHE = OUT_ROOT / "feature_cache"
BATCHES = [("sphere", i) for i in range(1, 7)] + \
          [("flat", i) for i in range(1, 3)] + \
          [("sharp", i) for i in range(1, 3)]

# Portrait -> landscape. np.rot90 k; chosen by cmd_orient (see ledger).
ROT = 1
N_PER_BATCH = 750        # labelled frames sampled per batch for the feature pass
N_REST = 48              # contact-free frames averaged into the global reference
ANCHOR_STRIDE = 200      # one rest anchor every N frames -> local reference
SEED = 0


# ---------------------------------------------------------------- download
def download(probes=None) -> None:
    """Fetch only the non-org pickles (2.2 GB); org_* are 1.6 GB each."""
    from huggingface_hub import hf_hub_download
    ROOT.mkdir(parents=True, exist_ok=True)
    for probe, b in BATCHES:
        if probes and probe not in probes:
            continue
        d = f"{probe}/batch_{b}"
        want = [f"{d}/dataset_slip_forces.pkl"] + \
               [f"{d}/dataset_gelsight_{k:02d}.pkl" for k in range(4)]
        for f in want:
            try:
                hf_hub_download(REPO, f, repo_type="dataset",
                                local_dir=str(ROOT))
            except Exception as e:                       # flat/batch_2 has no _03
                print(f"  skip {f}: {type(e).__name__}")


def batch_dir(probe: str, b: int) -> Path:
    return ROOT / probe / f"batch_{b}"


def image_files(probe: str, b: int) -> list[Path]:
    return sorted(batch_dir(probe, b).glob("dataset_gelsight_*.pkl"))


def load_labels(probe: str, b: int) -> dict:
    with open(batch_dir(probe, b) / "dataset_slip_forces.pkl", "rb") as fh:
        return pickle.load(fh)


def label_table(probe: str, b: int) -> dict:
    """Flatten trajectories -> global index, force, trajectory id, slip flag.

    UPSTREAM DEFECT: in every flat/* and sharp/* trajectory (598/598),
    `indexes` is exactly 5 longer than `forces` (and `poses` exactly 10
    longer); all 911 sphere/* trajectories are consistent. Pairing must
    therefore happen INSIDE each trajectory. Concatenating `indexes` and
    `forces` separately and zipping afterwards accumulates a 5-row drift per
    trajectory — by trajectory ~30 the force is read from a different press
    entirely. That bug alone drove flat/sharp to rho ~ 0 (sharp/b1 -0.184);
    with per-trajectory pairing the same frames give per-trajectory
    rho(|dI|, |Fz|) medians of 0.93 / 0.90.

    Offset scanned empirically (cmd_offset): forces[k] <-> indexes[k] (the
    trailing 5 indexes are dropped) maximises per-trajectory rho on flat
    (0.898 vs 0.594 at offset 5) and is within noise of the best on sharp.

    `poses` is (N,7) = [x, y, z, rx, ry, rz, _] of the probe in robot mm/deg.
    Column 2 is the indentation axis (verified: pooled rho(z,|Fz|) = -0.888 and
    per-trajectory median -0.927 on sphere/batch_1; columns 0/1 are the 2 mm
    lateral slide of the protocol and correlate -0.16/-0.07). It is carried
    here because the LUT calibration needs a depth datum, not just a force.
    """
    lab = load_labels(probe, b)
    idx, F, tid, slip, P, dropped = [], [], [], [], [], 0
    for t, tr in lab["trajectories"].items():
        i = np.asarray(tr["indexes"]).astype(np.int64)
        f = np.asarray(tr["forces"], dtype=np.float64)
        p = np.asarray(tr["poses"], dtype=np.float64)
        n = min(len(i), len(f))
        dropped += len(i) - n
        idx.append(i[:n])
        F.append(f[:n])
        P.append(p[:n])
        tid.append(np.full(n, int(t)))
        slip.append(np.asarray(tr["slip_label"]).astype(np.int64)[:n])
    return {"index": np.concatenate(idx), "F": np.concatenate(F),
            "P": np.concatenate(P),
            "tid": np.concatenate(tid), "slip": np.concatenate(slip),
            "in_contact": np.asarray(lab["in_contact"]), "dropped": dropped}


# ---------------------------------------------------------------- loader
def _decode(buf: bytes) -> np.ndarray:
    """PNG bytes -> pipeline-native float32 (240, 320, 3)."""
    a = np.asarray(Image.open(io.BytesIO(buf)).convert("RGB"))
    return crop(np.ascontiguousarray(np.rot90(a, ROT))).astype(np.float32)


def _stratified(tab: dict, n: int, seed: int) -> np.ndarray:
    """Pick n labelled rows spread over the |Fz| range, not over time.

    Sampling uniformly in time would over-represent the low-force approach
    phase of every trajectory; stratifying on |Fz| deciles keeps the high end
    populated so the fit is not dominated by near-zero presses.
    """
    fz = np.abs(tab["F"][:, 2])
    rng = np.random.default_rng(seed)
    edges = np.quantile(fz, np.linspace(0, 1, 11))
    per = int(np.ceil(n / 10))
    take = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = np.where((fz >= lo) & (fz <= hi))[0]
        if len(m):
            take.append(rng.choice(m, min(per, len(m)), replace=False))
    out = np.unique(np.concatenate(take))
    return out if len(out) <= n else rng.choice(out, n, replace=False)


def _anchors(ic: np.ndarray) -> np.ndarray:
    """Contact-free frames spaced ~ANCHOR_STRIDE apart, for a LOCAL reference.

    A batch runs 15-19k frames (~10 min). A single global rest median smears
    gel relaxation and lamp drift into every dI. Taking the median of the 6
    temporally nearest rest anchors instead lifts sphere/b1 rho(vol,|Fz|)
    0.533 -> 0.564 and rho(|dI|,|Fz|) 0.460 -> 0.568 at no extra decode cost.
    """
    rest = np.where(ic == 0)[0]
    return np.array(sorted({int(rest[np.argmin(np.abs(rest - t))])
                            for t in range(0, len(ic), ANCHOR_STRIDE)}))


def load_frames(probe: str, b: int, n: int = N_PER_BATCH, seed: int = SEED,
                local_ref: bool = True, swap_rb: bool = None,
                select=None):
    """Return (rows, ref_global) for one batch.

    Each row: global index, Fx/Fy/Fz, |Fz|, |F|, shear, tid, slip, z, image, ref.
    One pass over the image pickles; the rest anchors and the sampled
    labelled frames are pulled from the same load.

    `select` overrides the default |Fz|-stratified sampling: a callable
    tab -> row indices. Used by the LUT calibration, which needs low-shear
    frames spread over indentation DEPTH, not over force.
    """
    tab = label_table(probe, b)
    ic = tab["in_contact"]
    sel = _stratified(tab, n, seed) if select is None else select(tab)
    want = {int(tab["index"][i]): i for i in sel}
    anch = _anchors(ic)
    if len(anch) < 8:
        raise RuntimeError(f"{probe}/batch_{b}: only {len(anch)} rest anchors")

    need = set(want) | set(int(a) for a in anch)
    imgs, off = {}, 0
    for f in image_files(probe, b):
        with open(f, "rb") as fh:
            blob = pickle.load(fh)
        for j, buf in enumerate(blob):
            g = off + j
            if g in need:
                imgs[g] = _decode(buf)
        off += len(blob)
        del blob
    if off != len(ic):
        raise RuntimeError(f"{probe}/batch_{b}: {off} frames vs "
                           f"{len(ic)} in_contact flags — join broken")

    step = max(1, len(anch) // N_REST)
    ref_g = np.median(np.stack([imgs[int(a)] for a in anch[::step]]), 0)

    # UPSTREAM DEFECT: sharp/batch_2 is stored BGR while the other nine
    # batches are RGB. Its rest reference is (66, 120, 134) against
    # (127, 116, 60) everywhere else — the separation is unambiguous, so
    # detect it rather than hard-code a batch name. Undoing the swap lifts
    # sharp/b2 rho(vol,|Fz|) 0.376 -> 0.486 and LUT coverage 0.895 -> 0.946;
    # the same swap applied to sharp/b1 as a control makes it worse
    # (0.413 -> 0.361), so this is the sensor's channel order, not a fit knob.
    if swap_rb is None:
        swap_rb = bool(ref_g[..., 2].mean() > ref_g[..., 0].mean())
    if swap_rb:
        ref_g = np.ascontiguousarray(ref_g[..., ::-1])
        imgs = {g: np.ascontiguousarray(a[..., ::-1]) for g, a in imgs.items()}

    rows = []
    for g, i in sorted(want.items()):
        F = tab["F"][i]
        if local_ref:
            near = anch[np.argsort(np.abs(anch - g))[:6]]
            ref = np.median(np.stack([imgs[int(a)] for a in near]), 0)
        else:
            ref = ref_g
        rows.append({"index": g, "fx": float(F[0]), "fy": float(F[1]),
                     "fz": float(F[2]), "f": float(abs(F[2])),
                     "fmag": float(np.linalg.norm(F)),
                     "shear": float(np.hypot(F[0], F[1])),
                     "tid": int(tab["tid"][i]), "slip": int(tab["slip"][i]),
                     "px": float(tab["P"][i, 0]), "py": float(tab["P"][i, 1]),
                     "pz": float(tab["P"][i, 2]),
                     "img": imgs[g], "ref": ref})
    return rows, ref_g


# ---------------------------------------------------------------- commands
def cmd_verify() -> None:
    """Frame counts must sum to len(in_contact) or the global index is a lie."""
    for probe, b in BATCHES:
        cnt = [len(pickle.load(open(f, "rb"))) for f in image_files(probe, b)]
        tab = label_table(probe, b)
        ic, idx, F = tab["in_contact"], tab["index"], tab["F"]
        tot = sum(cnt)
        print(f"{probe}/batch_{b}: files={len(cnt)} sum={tot} "
              f"len(in_contact)={len(ic)} match={tot == len(ic)} "
              f"rest_frac={1 - ic.mean():.3f} labelled={len(idx)} "
              f"dropped_unpaired={tab['dropped']} "
              f"unique={len(set(idx.tolist()))} in_range={bool((idx < tot).all())} "
              f"|Fz|=[{np.abs(F[:, 2]).min():.3f},{np.abs(F[:, 2]).max():.3f}] "
              f"shear_max={np.hypot(F[:, 0], F[:, 1]).max():.3f}")


def cmd_traj() -> None:
    """The join test that matters: does |dI| track |Fz| WITHIN a trajectory?

    Batch-level correlation can be manufactured by between-press heterogeneity
    (the FeelAnyForce trap). A per-trajectory correlation cannot — inside one
    press the only thing that changes is the indentation.
    """
    from scipy.stats import spearmanr
    for probe, b in [("sphere", 1), ("flat", 1), ("sharp", 1)]:
        lab = load_labels(probe, b)
        tr, ic = lab["trajectories"], np.asarray(lab["in_contact"])
        tids = sorted(tr)[:20]
        anch = _anchors(ic)
        need = set(int(a) for a in anch)
        for t in tids:
            n = min(len(np.asarray(tr[t]["indexes"])), len(np.asarray(tr[t]["forces"])))
            need |= set(int(i) for i in np.asarray(tr[t]["indexes"])[:n])
        imgs, off = {}, 0
        for f in image_files(probe, b):
            blob = pickle.load(open(f, "rb"))
            for j, buf in enumerate(blob):
                if off + j in need:
                    imgs[off + j] = _decode(buf)
            off += len(blob)
            del blob
        rs = []
        for t in tids:
            i = np.asarray(tr[t]["indexes"]).astype(int)
            F = np.asarray(tr[t]["forces"])
            i = i[:len(F)]
            near = anch[np.argsort(np.abs(anch - i[0]))[:6]]
            ref = np.median(np.stack([imgs[int(a)] for a in near]), 0)
            d = [float(np.abs(imgs[g] - ref).max(2).mean()) for g in i]
            rs.append(spearmanr(d, np.abs(F[:, 2])).statistic)
        rs = np.array(rs, float)
        print(f"{probe}/batch_{b}: per-trajectory rho(|dI|,|Fz|) "
              f"median={np.nanmedian(rs):.3f} frac>0.5={(rs > 0.5).mean():.2f} "
              f"(n={len(rs)} trajectories)")


def cmd_orient() -> None:
    """A/B the portrait->landscape rotation on one batch, empirically."""
    from scipy.stats import spearmanr
    global ROT
    for k in (1, 3):
        ROT = k
        rows, _ = load_frames("sphere", 1, n=200, seed=0)
        v, m, c, f = [], [], [], []
        for r in rows:
            st = stages(r["img"], r["ref"])
            v.append(st["feats"]["vol"])
            m.append(st["feats"]["maxd"])
            c.append(st["lut_coverage"])
            f.append(r["f"])
        print(f"ROT={k}: n={len(f)} rho(vol,|Fz|)={spearmanr(v, f).statistic:.3f} "
              f"rho(maxd,|Fz|)={spearmanr(m, f).statistic:.3f} "
              f"LUT-cov med={np.median(c):.3f}")


def cmd_features() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    from scipy.stats import spearmanr
    for probe, b in BATCHES:
        path = CACHE / f"sparsh_{probe}_batch_{b}.json"
        if path.exists():
            print(f"cached {path.name}")
            continue
        rows, ref = load_frames(probe, b)
        out = []
        for r in rows:
            st = stages(r["img"], r["ref"])
            out.append({k: r[k] for k in
                        ("index", "fx", "fy", "fz", "f", "fmag", "shear",
                         "tid", "slip")}
                       | st["feats"] | {"cov": st["lut_coverage"]})
        json.dump({"probe": probe, "batch": b, "rot": ROT,
                   "ref_rgb": [float(x) for x in ref.mean((0, 1))],
                   "rows": out}, open(path, "w"))
        f = np.array([r["f"] for r in out])
        vol = np.array([r["vol"] for r in out])
        print(f"{probe}/batch_{b}: n={len(out)} "
              f"rho(vol,|Fz|)={spearmanr(vol, f).statistic:.3f} "
              f"cov={np.median([r['cov'] for r in out]):.3f} -> {path.name}")


# ------------------------------------------------------------- evaluation
def _load_cache() -> dict:
    out = {}
    for probe, b in BATCHES:
        p = CACHE / f"sparsh_{probe}_batch_{b}.json"
        if p.exists():
            out[f"{probe}_b{b}"] = json.load(open(p))
    return out


def fit(rows, target="f"):
    """Least-squares on [vol, vol2, maxd, area, h1] + isotonic rectification."""
    from sklearn.isotonic import IsotonicRegression
    X = np.array([feat_vec(r) for r in rows])
    y = np.array([r[target] for r in rows])
    w, *_ = np.linalg.lstsq(X, y, rcond=None)
    iso = IsotonicRegression(out_of_bounds="clip").fit(X @ w, y)
    return lambda rs: iso.predict(np.array([feat_vec(r) for r in rs]) @ w)


def _split_eval(rows, seeds=5, target="f", shuffle=None):
    """Half/half within-batch, `seeds` repeats -> pooled predictions.

    shuffle=None   real labels
    shuffle='batch' WITHIN-batch label shuffle. A GLOBAL shuffle is not a
            control: it also destroys the between-batch force-range structure
            and so always collapses to 0, which tells you nothing. On
            FeelAnyForce the global shuffle "passed" (0.455 vs -0.003) while
            the within-group shuffle exposed the join as never demonstrated
            (0.455 vs 0.442).
    shuffle='traj'  WITHIN-trajectory shuffle — the strictest form: keeps
            which press each label belongs to, destroys only the frame-level
            pairing inside it.
    """
    rng = np.random.default_rng(0)
    y = np.array([r[target] for r in rows])
    tid = np.array([r["tid"] for r in rows])
    P, Y = [], []
    for s in range(seeds):
        perm = np.random.default_rng(s).permutation(len(rows))
        yy = y.copy()
        if shuffle == "batch":
            yy = rng.permutation(yy)
        elif shuffle == "traj":
            for t in np.unique(tid):
                m = np.where(tid == t)[0]
                yy[m] = rng.permutation(yy[m])
        rs = [dict(rows[i], **{target: yy[i]}) for i in perm]
        h = len(rs) // 2
        f = fit(rs[:h], target)
        P.append(f(rs[h:]))
        Y.append(np.array([r[target] for r in rs[h:]]))
    return np.concatenate(P), np.concatenate(Y)


def cmd_eval() -> None:
    from scipy.stats import spearmanr
    data = _load_cache()
    rho = lambda a, b: float(spearmanr(a, b).statistic)

    print("\n=== (a) raw feature correlations, per batch ===")
    print(f"{'batch':14s} {'n':>4s} {'rho(vol,|Fz|)':>14s} {'rho(maxd,|Fz|)':>15s}"
          f" {'rho(vol,|F|)':>13s} {'rho(vol,shear)':>15s} {'LUTcov':>7s}")
    pool = []
    for k, d in data.items():
        r = d["rows"]
        pool += [dict(x, batch=k) for x in r]
        v = [x["vol"] for x in r]
        print(f"{k:14s} {len(r):4d} {rho(v, [x['f'] for x in r]):14.3f} "
              f"{rho([x['maxd'] for x in r], [x['f'] for x in r]):15.3f} "
              f"{rho(v, [x['fmag'] for x in r]):13.3f} "
              f"{rho(v, [x['shear'] for x in r]):15.3f} "
              f"{np.median([x['cov'] for x in r]):7.3f}")
    v = [x["vol"] for x in pool]
    print(f"{'POOLED':14s} {len(pool):4d} {rho(v, [x['f'] for x in pool]):14.3f} "
          f"{rho([x['maxd'] for x in pool], [x['f'] for x in pool]):15.3f} "
          f"{rho(v, [x['fmag'] for x in pool]):13.3f} "
          f"{rho(v, [x['shear'] for x in pool]):15.3f}")

    print("\n=== (b,c) calibrated per batch + SHUFFLE CONTROLS ===")
    print(f"{'batch':14s} {'rho':>6s} {'MAE_N':>7s} | {'rho_shufB':>9s} "
          f"{'MAE_shufB':>9s} | {'rho_shufT':>9s} | {'rho(|F|)':>8s} "
          f"{'MAE(|F|)':>8s}")
    allp, ally, allps, allys, allpt, allyt = [], [], [], [], [], []
    for k, d in data.items():
        p, y = _split_eval(d["rows"])
        ps, ys = _split_eval(d["rows"], shuffle="batch")
        pt, yt = _split_eval(d["rows"], shuffle="traj")
        pm, ym = _split_eval(d["rows"], target="fmag")
        allp.append(p); ally.append(y); allps.append(ps); allys.append(ys)
        allpt.append(pt); allyt.append(yt)
        print(f"{k:14s} {rho(p, y):6.3f} {np.abs(p - y).mean():7.3f} | "
              f"{rho(ps, ys):9.3f} {np.abs(ps - ys).mean():9.3f} | "
              f"{rho(pt, yt):9.3f} | {rho(pm, ym):8.3f} "
              f"{np.abs(pm - ym).mean():8.3f}")
    p, y = np.concatenate(allp), np.concatenate(ally)
    ps, ys = np.concatenate(allps), np.concatenate(allys)
    pt, yt = np.concatenate(allpt), np.concatenate(allyt)
    print(f"{'POOLED':14s} {rho(p, y):6.3f} {np.abs(p - y).mean():7.3f} | "
          f"{rho(ps, ys):9.3f} {np.abs(ps - ys).mean():9.3f} | "
          f"{rho(pt, yt):9.3f}")
    print("  (pooled shuffled rho is NOT 0 because a within-batch shuffle "
          "preserves each batch's force range; the comparison that matters is "
          f"real {rho(p, y):.3f} vs shuffled {rho(ps, ys):.3f}.)")

    print("\n=== (d) cross-batch transfer: fit on row, apply to column ===")
    for probe in ("sphere", "flat", "sharp"):
        ks = [k for k in data if k.startswith(probe + "_")]
        if len(ks) < 2:
            continue
        print(f"\n-- {probe} -- rho (MAE N)")
        print(f"{'fit|eval':12s}" + "".join(f"{k.split('_')[1]:>16s}" for k in ks))
        for a in ks:
            fa = fit(data[a]["rows"])
            cells = []
            for b in ks:
                pr = fa(data[b]["rows"])
                gt = np.array([r["f"] for r in data[b]["rows"]])
                cells.append(f"{rho(pr, gt):.3f} ({np.abs(pr - gt).mean():.3f})")
            print(f"{a.split('_')[1]:12s}" + "".join(f"{c:>16s}" for c in cells))

    print("\n=== cross-PROBE transfer (fit sphere_b1) ===")
    fa = fit(data["sphere_b1"]["rows"])
    for k, d in data.items():
        pr = fa(d["rows"])
        gt = np.array([r["f"] for r in d["rows"]])
        print(f"  sphere_b1 -> {k:12s} rho={rho(pr, gt):.3f} "
              f"MAE={np.abs(pr - gt).mean():.3f} N")

    print("\n=== (4) shear: are high-shear frames outliers? ===")
    print(f"{'batch':14s} {'q90(sh)':>8s} {'max':>6s} | {'rho lo-sh':>10s} "
          f"{'rho hi-sh':>10s} | {'|resid| lo':>10s} {'|resid| hi':>10s} "
          f"| {'rho noslip':>10s} {'rho slip':>9s}")
    for k, d in data.items():
        r = d["rows"]
        sh = np.array([x["shear"] for x in r])
        sl = np.array([x["slip"] for x in r])
        v = np.array([x["vol"] for x in r])
        f = np.array([x["f"] for x in r])
        hi = sh > np.quantile(sh, 0.90)          # top shear decile
        pr = fit(r)(r)                           # in-sample, residual shape only
        res = np.abs(pr - f)
        print(f"{k:14s} {np.quantile(sh, .9):8.3f} {sh.max():6.3f} | "
              f"{rho(v[~hi], f[~hi]):10.3f} {rho(v[hi], f[hi]):10.3f} | "
              f"{res[~hi].mean():10.3f} {res[hi].mean():10.3f} | "
              f"{rho(v[sl == 0], f[sl == 0]):10.3f} "
              f"{rho(v[sl == 1], f[sl == 1]) if (sl == 1).sum() > 10 else float('nan'):9.3f}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "features"
    {"download": download, "verify": cmd_verify, "traj": cmd_traj,
     "orient": cmd_orient, "features": cmd_features, "eval": cmd_eval}[cmd]()
