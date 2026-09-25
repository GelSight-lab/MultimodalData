"""FeelAnyForce on Sparsh frames — runs inside the `anyforce` conda env.

That env has no scipy/h5py/pyarrow, so this script is deliberately standalone:
numpy + torch + PIL only, and it reads the frames from the .npz files that
`sparsh_baselines.export` writes (already gsdevice-cropped, already
background-subtracted with the same local rest reference the physics path
uses). Everything after that is `test_headless.py` verbatim: Pad[0,40,0,40],
Resize(crop_size), ToTensor, Normalize with the checkpoint's own dataset
stats, DINOv2 backbone -> regressor -> (Fx, Fy, Fz) in newtons.

    conda run -n anyforce python force_recovery/anyforce_sparsh.py
    conda run -n anyforce python force_recovery/anyforce_sparsh.py --sweep
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

FAF = os.path.expanduser("~/projects/TacForce/force_prediction/FeelAnyForce")
sys.path.insert(0, FAF)

ROOT = "/media/yxma/Disk1/twm/force_recovery"
FRAMES = os.path.join(ROOT, "sparsh_baseline_frames")
SWEEP = os.path.join(ROOT, "sparsh_baseline_sweep")
OUT = os.path.join(ROOT, "feature_cache", "anyforce_on_sparsh.json")


def build():
    import torch
    from torchvision import transforms as pth_transforms
    from composed_model import ComposedModel

    ckpt = torch.load(os.path.join(FAF, "ckpt", "checkpoint_v1.pth.tar"),
                      map_location="cpu")
    cfg = argparse.Namespace(**ckpt["config"])
    print(f"config: model={cfg.tactile_model} modality={cfg.input_modality} "
          f"labels={cfg.num_labels} mode={cfg.tactile_mode}", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ComposedModel(cfg)
    model.to(device).eval()
    model.load_state_dict(ckpt["state_dict"])

    sp = cfg.dataset_stats if os.path.isabs(cfg.dataset_stats) \
        else os.path.join(FAF, cfg.dataset_stats)
    with open(sp) as f:
        stats = json.load(f)
    mean_rgb, std_rgb = stats[cfg.labels_train][cfg.tactile_mode]
    tf = pth_transforms.Compose([
        pth_transforms.Pad([0, 40, 0, 40]),
        pth_transforms.Resize(cfg.crop_size),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize(mean_rgb, std_rgb),
    ])
    return torch, device, model, tf


def infer(torch, device, model, tf, ims, bs=32):
    out = []
    for i in range(0, len(ims), bs):
        x = torch.stack([tf(Image.fromarray(a)) for a in ims[i:i + bs]]).to(device)
        with torch.no_grad():
            y = model.regressor(model.tactile_backbone(x)).cpu().numpy()
        out.append(y)
        if (i // bs) % 20 == 0:
            print(f"    {i}/{len(ims)}", flush=True)
    return np.concatenate(out)


def rank_rho(a, b):
    """Spearman without scipy (no ties correction needed for float outputs)."""
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))


def main():
    sweep = "--sweep" in sys.argv
    torch, device, model, tf = build()
    if sweep:
        d = np.load(os.path.join(SWEEP, "sphere_batch_1.npz"))
        f, iv = d["f"], d["inview"]
        for k in ("crop_bg", "crop_raw", "full_bg", "full_raw"):
            if k not in d:
                continue
            y = infer(torch, device, model, tf, d[k])
            p = -y[:, 2]
            print(f"ANYFORCE {k:9s} rho_all={rank_rho(p, f):+.3f} "
                  f"rho_inview={rank_rho(p[iv], f[iv]):+.3f} "
                  f"mean={p.mean():.3f} sd={p.std():.4f} "
                  f"range=[{p.min():.3f},{p.max():.3f}]", flush=True)
        print("SWEEP DONE", flush=True)
        return

    MAIN = "full_bg"          # picked by --sweep, see ledger
    rows = []
    for fn in sorted(os.listdir(FRAMES)):
        if not fn.endswith(".npz"):
            continue
        name = fn[:-4]
        d = np.load(os.path.join(FRAMES, fn))
        keys = [k for k in ("crop_bg", "full_bg") if k in d]
        y = {k: infer(torch, device, model, tf, d[k]) for k in keys}
        for j, (g, k) in enumerate(zip(d["index"], d["inview"])):
            r = y[MAIN][j]
            rows.append({"group": name, "index": int(g), "inview": bool(k),
                         "fx": float(r[0]), "fy": float(r[1]),
                         "fz": float(r[2]), "pred": float(-r[2]),
                         "variant": MAIN,
                         **{f"pred_{v}": float(-y[v][j][2]) for v in keys}})
        print(f"{name}: {len(d['index'])} frames ({keys})", flush=True)
    with open(OUT, "w") as f:
        json.dump(rows, f)
    print(f"-> {OUT} ({len(rows)} rows)", flush=True)
    print("ANYFORCE DONE", flush=True)


if __name__ == "__main__":
    main()
