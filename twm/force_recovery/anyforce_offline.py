"""Offline FeelAnyForce inference on cnc_Mini frames (run in the anyforce env).

Mirrors test_headless.py exactly: gsdevice-style 1/7 border crop to 320x240,
background subtraction against a near-zero-force frame, Pad+Resize+Normalize,
DINOv2 backbone + regressor -> (Fx, Fy, Fz).

    conda run -n anyforce python anyforce_offline.py <out.json> [max_frames]
"""
import argparse
import io
import json
import os
import sys
import tarfile

import cv2
import numpy as np
from PIL import Image

FAF = os.path.expanduser("~/projects/TacForce/force_prediction/FeelAnyForce")
sys.path.insert(0, FAF)

CNC = "/media/yxma/Disk1/twm/force_recovery/fota_cnc/cnc/cnc_Mini"


def resize_crop_mini(img, imgw=320, imgh=240):
    bx, by = int(img.shape[0] / 7), int(np.floor(img.shape[1] / 7))
    img = img[bx + 2:img.shape[0] - bx, by:img.shape[1] - by]
    return cv2.resize(img, (imgw, imgh))


def build_index(split):
    idx = []
    tar = os.path.join(CNC, split, "data-000000.tar")
    with tarfile.open(tar) as tf:
        for m in tf:
            if m.name.endswith(".json"):
                meta = json.loads(tf.extractfile(m).read())
                idx.append({"tar": tar, "jpg": m.name[:-5] + ".jpg", **meta})
    return idx


def main():
    out_path = sys.argv[1]
    max_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 700

    import torch
    from torchvision import transforms as pth_transforms
    from composed_model import ComposedModel

    ckpt_path = os.path.join(FAF, "ckpt", "checkpoint_v1.pth.tar")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = argparse.Namespace(**ckpt["config"])
    print(f"config: model={cfg.tactile_model} modality={cfg.input_modality} "
          f"labels={cfg.num_labels} mode={cfg.tactile_mode}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ComposedModel(cfg)
    model.to(device).eval()
    model.load_state_dict(ckpt["state_dict"])

    stats_path = cfg.dataset_stats if os.path.isabs(cfg.dataset_stats) \
        else os.path.join(FAF, cfg.dataset_stats)
    with open(stats_path) as f:
        stats = json.load(f)
    mean_rgb, std_rgb = stats[cfg.labels_train][cfg.tactile_mode]
    transform_rgb = pth_transforms.Compose([
        pth_transforms.Pad([0, 40, 0, 40]),
        pth_transforms.Resize(cfg.crop_size),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize(mean_rgb, std_rgb),
    ])

    index = build_index("train")
    forces = np.array([e["force"] for e in index])
    order = np.argsort(forces)

    def load(p):
        with tarfile.open(p["tar"]) as tf:
            raw = np.asarray(Image.open(io.BytesIO(
                tf.extractfile(p["jpg"]).read())).convert("RGB"))
        return resize_crop_mini(raw)

    bg = load(index[order[0]]).astype(np.int16)      # the freest frame
    picks = [index[i] for i in
             order[np.linspace(0, len(order) - 1,
                               min(max_frames, len(order))).astype(int)]]

    rows = []
    with tarfile.open(picks[0]["tar"]) as tf:
        for i, p in enumerate(picks):
            raw = np.asarray(Image.open(io.BytesIO(
                tf.extractfile(p["jpg"]).read())).convert("RGB"))
            im = resize_crop_mini(raw)
            proc = np.clip(im.astype(np.int16) - bg + 127, 0, 255).astype(np.uint8)
            tensor = transform_rgb(Image.fromarray(proc)).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = model.tactile_backbone(tensor)
                out = model.regressor(feat).cpu().squeeze(0).numpy()
            rows.append({"force_true": float(p["force"]),
                         "x_mm": float(p["x_mm"]), "y_mm": float(p["y_mm"]),
                         "probe": str(p["obj_name"]),
                         "fx": float(out[0]), "fy": float(out[1]),
                         "fz": float(out[2])})
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(picks)}", flush=True)

    with open(out_path, "w") as f:
        json.dump(rows, f)
    try:
        from scipy.stats import spearmanr
        fz = np.array([r["fz"] for r in rows])
        ft = np.array([r["force_true"] for r in rows])
        print(f"n={len(rows)} rho(|fz|,F)={spearmanr(np.abs(fz), ft).statistic:.3f} "
              f"rho(-fz,F)={spearmanr(-fz, ft).statistic:.3f}", flush=True)
    except ImportError:
        print(f"n={len(rows)} (scipy unavailable in this env)", flush=True)
    print("ANYFORCE DONE", flush=True)


if __name__ == "__main__":
    main()
