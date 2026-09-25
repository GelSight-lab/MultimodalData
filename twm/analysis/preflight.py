"""Preflight: prove every resource the plan names actually loads, before any run.

A cross-validation study over 8 datasets and 6 encoders fails in exactly one
boring way -- eight hours in, on a path that was never checked. This file is the
gate. Every task in plan/cross-validation-plan.md has at least one check here,
and a task may not start until its checks are green.

Checks are INDEPENDENT and each is wrapped: one missing dataset reports FAIL and
the rest still run, so a single pass tells you the whole state of the world
rather than the first thing that broke.

Nothing here trains or evaluates anything. It answers only: does it load, is it
the shape and dtype the plan assumes, and is the label actually populated. That
last one matters -- several parquet subsets on this disk have force COLUMNS that
are entirely null, which looks like data until you check.

    python analysis/preflight.py          # all checks
    python analysis/preflight.py --task T2   # only checks gating T2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

FEATS_NPY = Path("/media/yxma/Disk1/yuxiang/mini_data/markered/FEATS/v24_labels_24_32")
FEATS_PQ = Path("/media/yxma/Disk1/yuxiang/mini_data_parquet/feats")
OUT_ROOT = Path("/media/yxma/Disk1/twm/force_recovery")
CACHE = OUT_ROOT / "feature_cache"
CNC26 = OUT_ROOT / "glowtact" / "cnc_mini_26"
GLOWTACT = OUT_ROOT / "glowtact" / "GlowTact_force_final_14716"
FAF_LABELS = OUT_ROOT / "faf_labels"
SPARSH = OUT_ROOT / "fb_gs"
TACQUAD = Path("/media/yxma/Disk1/yuxiang/mini_data/multi_sensor/TacQuad")
THREEDCAL = Path("/media/yxma/Disk1/yuxiang/mini_data/markerless/3DCal"
                 "/gsmini_calibration_data")

T3_DIR = Path("/media/yxma/Disk1/azhao/backup/FoundationTactile/TheProbe/ProbingPanda"
              "/checkpoints_supercloud/best_2024-04-25_20_23_03-step6_withstate"
              "_withpretrainedtactile_base0.7_abs")
FEATS_UNET = Path("/home/yxma/feats/src/feats/models/unet_09042025_124903_80.pt")
TORCH_HUB = Path("/home/yxma/.cache/torch/hub/checkpoints")

CHECKS: list = []


def check(cid: str, tasks: str):
    def deco(fn):
        CHECKS.append((cid, tasks, fn))
        return fn
    return deco


# ---------------------------------------------------------------- data


@check("D1-feats-grid", "T0,T1,T2,T3,T4")
def d1():
    """The 5x6x100 crossed grid Task 1 established. Everything on the gel axis
    rests on it being balanced, so assert balance rather than trusting it."""
    import re
    from collections import Counter
    pat = re.compile(r"^(\d+)_nocontact_sensor_(\d+)_gel_(\d+)\.npy$")
    grid = Counter()
    for sp in os.listdir(FEATS_NPY):
        d = FEATS_NPY / sp
        if d.is_dir():
            for f in os.listdir(d):
                m = pat.match(f)
                if m:
                    grid[(int(m.group(2)), int(m.group(3)))] += 1
    n = sorted(set(grid.values()))
    assert len(grid) == 30, f"expected 30 cells, got {len(grid)}"
    assert n == [100], f"unbalanced cells: {n}"
    return f"30 cells (5 sensor x 6 gel), exactly 100 frames each, n={sum(grid.values())}"


@check("D2-feats-npy", "T0,T2,T3")
def d2():
    """Array contract: gs_img uint8 (240,320,3); no-contact frames truly zero."""
    f = FEATS_NPY / "test" / "100_nocontact_sensor_0_gel_2.npy"
    r = np.load(f, allow_pickle=True).item()
    img = r["gs_img"]
    assert img.shape == (240, 320, 3) and img.dtype == np.uint8, (img.shape, img.dtype)
    assert float(np.abs(r["f_z"])) == 0.0, "no-contact frame carries nonzero f_z"
    return f"gs_img {img.shape} {img.dtype}, keys={sorted(r)[:4]}..., f_z=0 on no-contact"


@check("D3-feats-force", "T4")
def d3():
    """FEATS force labels must be POPULATED, not merely present as columns."""
    import pandas as pd
    pq = sorted(FEATS_PQ.glob("*.parquet"))
    assert pq, f"no parquet under {FEATS_PQ}"
    df = pd.read_parquet(pq[0], columns=["f_x", "f_y", "f_z"])
    nn = int(df["f_z"].notna().sum())
    assert nn > 0, "f_z entirely null"
    return (f"{pq[0].name}: {len(df)} rows, f_z non-null={nn}, "
            f"range [{df.f_z.min():.2f}, {df.f_z.max():.2f}] N")


@check("D4-cnc26-force", "T1,T3,T4")
def d4():
    """Force lives in the FILENAME here: '...|<x,y,z> f|<newtons>.jpg'."""
    subs = [p for p in CNC26.iterdir() if p.is_dir()]
    tot, ex = 0, None
    for s in subs:
        for f in s.iterdir():
            if f.suffix == ".jpg" and "f|" in f.name:
                tot += 1
                if ex is None:
                    ex = f.name
    assert tot > 0, "no force-tagged jpgs"
    val = float(ex.split("f|")[1].rsplit(".jpg", 1)[0])
    return f"{len(subs)} families, {tot} force-tagged jpgs, parsed f={val} N from {ex[:38]}..."


@check("D5-glowtact-force", "T5")
def d5():
    """Second sensor DESIGN with real force GT -- the cross-design arm."""
    tot, ex = 0, None
    for s in [p for p in GLOWTACT.iterdir() if p.is_dir()]:
        for f in s.iterdir():
            if f.suffix == ".jpg" and "f|" in f.name:
                tot += 1
                ex = ex or f.name
    assert tot > 0, "no force-tagged jpgs"
    return f"{tot} force-tagged jpgs, e.g. f={float(ex.split('f|')[1].rsplit('.jpg',1)[0])} N"


@check("D6-faf-labels", "T4")
def d6():
    """FeelAnyForce: the in-tree CSVs are git-LFS pointers; these are the real ones."""
    import pandas as pd
    csvs = sorted(FAF_LABELS.glob("*.csv"))
    assert csvs, f"no csv under {FAF_LABELS}"
    df = pd.read_csv(csvs[0], nrows=5)
    assert "FT" in df.columns, f"no FT column: {list(df.columns)}"
    head = open(csvs[0]).readline()
    assert not head.startswith("version https://git-lfs"), "LFS pointer, not data"
    return f"{len(csvs)} csv, cols={list(df.columns)}, FT[0]={str(df.FT.iloc[0])[:34]}..."


@check("D7-sparsh-pkl", "T0b,T4")
def d7():
    """Sparsh DATA (cc-by-nc). The brief bans the Sparsh ENCODER, not this force GT.
    Batches 1..6 on one probe are sequential -> the wear proxy for T0b."""
    pk = sorted(SPARSH.rglob("*.pkl"))
    assert pk, f"no pkl under {SPARSH}"
    # Probe is the PARENT-OF-PARENT dir (sphere/batch_1/dataset_*.pkl), never the
    # filename -- reading it off the filename silently yields zero batches.
    probes = sorted({p.parent.parent.name for p in pk})
    batches = sorted({(p.parent.parent.name, p.parent.name) for p in pk})
    nseq = len([b for b in batches if b[0] == "sphere"])
    assert nseq > 1, f"need >1 sequential sphere batch for the wear axis, got {nseq}"
    forces = [p for p in pk if "slip_forces" in p.name]
    return (f"{len(pk)} pkl, probes={probes}, {len(batches)} probe-batches "
            f"({nseq} sequential sphere batches for T0b), {len(forces)} force files")


@check("D8-tacquad", "T5")
def d8():
    """Four sensor DESIGNS -> puts the within-design unit spread in context."""
    ex = TACQUAD / "tacquad_extracted"
    assert ex.is_dir(), f"missing {ex}"
    subs = sorted(p.name for p in ex.iterdir() if p.is_dir())
    return f"extracted: {subs}"


@check("D9-3dcal", "T1")
def d9():
    """No force, but penetration_depth_mm is a monotone force proxy -- attenuated
    rank correlation, and the plan must say so wherever it is used."""
    import pandas as pd
    c = THREEDCAL / "annotations" / "annotations.csv"
    assert c.exists(), f"missing {c}"
    df = pd.read_csv(c, nrows=5)
    assert "penetration_depth_mm" in df.columns, list(df.columns)
    return f"cols={list(df.columns)}"


@check("D10-caches", "T4")
def d10():
    """Existing cross-dataset eval to be EXTENDED, not duplicated."""
    f = CACHE / "calibfree_vs_lut.json"
    assert f.exists(), f"missing {f}"
    rows = json.loads(f.read_text())
    fr = CACHE / "feats_rows.json"
    nfr = len(json.loads(fr.read_text())) if fr.exists() else 0
    return (f"calibfree_vs_lut: {[r['label'].split(' (')[0] for r in rows]}; "
            f"feats_rows n={nfr}")


# ---------------------------------------------------------------- code


@check("C1-protocol", "T1,T2,T3,T4,T5")
def c1():
    """The scoring protocol, exercised on synthetic data with a known answer.
    A monotone signal must score high and its shuffle must not."""
    from force_recovery.force_eval_all import evaluate
    rng = np.random.default_rng(0)
    n = 240
    f = rng.uniform(0, 20, n)
    X = np.c_[f + rng.normal(0, .4, n), f ** 2, f * .5, f * 2, f * 1.5]
    g = np.array(["a", "b", "c", "d"] * (n // 4))
    r = evaluate(X, f, g, seeds=2)
    assert r["rho"] > 0.9, f"monotone signal scored only {r['rho']}"
    assert abs(r["shuffle_rho"]) < 0.5, f"shuffle control leaked: {r['shuffle_rho']}"
    return f"rho={r['rho']:.4f} shuffle={r['shuffle_rho']:+.4f} n_eval={r['n_eval']}"


@check("C2-features", "T1,T3,T4")
def c2():
    """The canonical 5 features, in order. T3/T4 stack these into X."""
    from force_recovery import pipeline
    assert pipeline.FEATURES == ("vol", "vol2", "maxd", "area", "h1"), pipeline.FEATURES
    v = pipeline.feature_vector({"feats": dict(vol=1., vol2=2., maxd=3., area=4., h1=5.)})
    assert np.allclose(v, [1, 2, 3, 4, 5]), v
    return f"FEATURES={pipeline.FEATURES}, feature_vector -> {v.tolist()}"


@check("C3-lut-recon", "T3")
def c3():
    """LUT reconstruction on a real FEATS pair (marker gel is its hard case)."""
    from force_recovery.debug_gallery import stages
    d = FEATS_NPY / "test"
    ref = np.load(d / "100_nocontact_sensor_0_gel_2.npy", allow_pickle=True).item()["gs_img"]
    img = np.load(d / "100_1744012761409948994_sphere_15.npy", allow_pickle=True).item()["gs_img"]
    st = stages(img.astype(np.float32), ref.astype(np.float32))
    assert st["depth"].shape == (240, 320), st["depth"].shape
    assert set(("vol", "vol2", "maxd", "area", "h1")) <= set(st["feats"]), st["feats"].keys()
    return f"depth{st['depth'].shape} peak={st['feats']['maxd']:.4f}mm area={st['feats']['area']:.1f}"


@check("C4-calibfree", "T3")
def c4():
    """Calibration-free solve: scale-free, so features need the relative floor."""
    from force_recovery.calib_free import reconstruct
    d = FEATS_NPY / "test"
    ref = np.load(d / "100_nocontact_sensor_0_gel_2.npy", allow_pickle=True).item()["gs_img"]
    img = np.load(d / "100_1744012761409948994_sphere_15.npy", allow_pickle=True).item()["gs_img"]
    r = reconstruct(img.astype(np.float32), ref.astype(np.float32))
    assert r["depth"].shape == (240, 320), r["depth"].shape
    return f"depth{r['depth'].shape} range [{r['depth'].min():.4f}, {r['depth'].max():.4f}] (scale-free)"


@check("C5-loaders", "T4")
def c5():
    """The loader registry contract every new dataset arm must satisfy."""
    from force_recovery.debug_gallery import DATASETS
    assert set(DATASETS) >= {"glowtact", "cnc", "feats"}, list(DATASETS)
    return f"DATASETS={sorted(DATASETS)} (contract: () -> (rows, get))"


# ---------------------------------------------------------------- models


def _sd(p: Path):
    import torch
    o = torch.load(p, map_location="cpu", weights_only=False)
    for k in ("state_dict", "model", "model_state_dict"):
        if isinstance(o, dict) and k in o and isinstance(o[k], dict):
            o = o[k]
            break
    return {k: v for k, v in o.items() if hasattr(v, "numel")}


@check("M1-torch", "T2")
def m1():
    import torch
    return (f"torch {torch.__version__} cuda={torch.cuda.is_available()} "
            f"devices={torch.cuda.device_count()}")


@check("M2-t3-trunk", "T2")
def m2():
    """T3 shared trunk. NOTE: this is the downstream fine-tuned checkpoint, NOT
    the released mini.pth (absent from this machine). Any T3 number must say so."""
    p = T3_DIR / "trunk.pth"
    assert p.exists(), f"missing {p}"
    sd = _sd(p)
    n = sum(v.numel() for v in sd.values())
    return f"{n/1e6:.2f}M params, {len(sd)} tensors, e.g. {list(sd)[0]}"


@check("M3-t3-encoder", "T2")
def m3():
    p = T3_DIR / "encoders" / "wedge.pth"
    assert p.exists(), f"missing {p}"
    sd = _sd(p)
    n = sum(v.numel() for v in sd.values())
    assert any("patch_embed" in k for k in sd), "not a ViT encoder"
    return f"ViT {n/1e6:.2f}M params, {len(sd)} tensors"


@check("M4-dinov2", "T2")
def m4():
    out = []
    for nm in ("dinov2_vitb14_pretrain.pth", "dinov2_vits14_pretrain.pth"):
        p = TORCH_HUB / nm
        if p.exists():
            n = sum(v.numel() for v in _sd(p).values())
            out.append(f"{nm.split('_')[1]}={n/1e6:.1f}M")
    assert out, "no DINOv2 weights"
    return "DINOv2 " + ", ".join(out)


@check("M5-resnet", "T2")
def m5():
    out = []
    for nm in ("resnet50-19c8e357.pth", "resnet18-5c106cde.pth"):
        p = TORCH_HUB / nm
        if p.exists():
            n = sum(v.numel() for v in _sd(p).values())
            out.append(f"{nm.split('-')[0]}={n/1e6:.1f}M")
    assert out, "no ResNet weights"
    return "ImageNet " + ", ".join(out)


@check("M6-feats-unet", "T2,T4")
def m6():
    assert FEATS_UNET.exists(), f"missing {FEATS_UNET}"
    sd = _sd(FEATS_UNET)
    return f"FEATS U-Net {sum(v.numel() for v in sd.values())/1e6:.2f}M params"


@check("M7-banned-absent", "T2")
def m7():
    """The brief bans Sparsh and SITR as extractors because both cancel the
    per-unit illumination offset under measurement. Confirm they cannot be
    reached by accident."""
    import glob
    hits = []
    for pat in ("**/sparsh*.pth", "**/sparsh*.pt", "**/sitr*.pth", "**/sitr*.pt"):
        hits += glob.glob(str(Path("/home/yxma") / pat), recursive=True)
    assert not hits, f"banned encoder weights present: {hits[:3]}"
    return "Sparsh/SITR encoder weights absent -- ban cannot be violated by accident"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=None, help="only checks gating this task, e.g. T2")
    a = ap.parse_args()
    sel = [c for c in CHECKS if not a.task or a.task in c[1].split(",")]
    print(f"preflight: {len(sel)} checks" + (f" gating {a.task}" if a.task else ""))
    print("=" * 74)
    npass = 0
    fails = []
    for cid, tasks, fn in sel:
        try:
            detail = fn()
            npass += 1
            print(f"  PASS  {cid:18s} [{tasks:22s}] {detail}")
        except Exception as e:
            fails.append((cid, e))
            msg = str(e).replace("\n", " ")[:88] or type(e).__name__
            print(f"  FAIL  {cid:18s} [{tasks:22s}] {msg}")
            if os.environ.get("PREFLIGHT_TRACE"):
                traceback.print_exc()
    print("=" * 74)
    print(f"preflight: {npass}/{len(sel)} passed, {len(fails)} failed")
    if fails:
        print("blocked tasks:", sorted({t for cid, _ in fails
                                        for c in CHECKS if c[0] == cid
                                        for t in c[1].split(",")}))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
