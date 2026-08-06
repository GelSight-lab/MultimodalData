"""FeelAnyForce on React frames — labelling-fitness spot check (anyforce env).

React has no force GT, so the checks are: no-contact frames read ~0, and
rank agreement with (a) our physics estimate and (b) tactile intensity.

    conda run -n anyforce python anyforce_react.py <out.json>
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image

FAF = os.path.expanduser("~/projects/TacForce/force_prediction/FeelAnyForce")
sys.path.insert(0, FAF)

H5 = "/media/yxma/Disk1/twm/data/motherboard/2026-05-10/episode_000.h5"
NPZ = "/media/yxma/Disk1/twm/force_recovery/motherboard/2026-05-10/episode_000_left.npz"


def resize_crop_mini(img, imgw=320, imgh=240):
    bx, by = int(img.shape[0] / 7), int(np.floor(img.shape[1] / 7))
    img = img[bx + 2:img.shape[0] - bx, by:img.shape[1] - by]
    return cv2.resize(img, (imgw, imgh))


def main():
    out_path = sys.argv[1]
    import h5py
    import hdf5plugin  # noqa: F401
    import torch
    from torchvision import transforms as pth_transforms
    from composed_model import ComposedModel

    ckpt = torch.load(os.path.join(FAF, "ckpt", "checkpoint_v1.pth.tar"),
                      map_location="cpu")
    cfg = argparse.Namespace(**ckpt["config"])
    device = torch.device("cuda")
    model = ComposedModel(cfg)
    model.to(device).eval()
    model.load_state_dict(ckpt["state_dict"])
    stats = json.load(open(os.path.join(FAF, cfg.dataset_stats)
                           if not os.path.isabs(cfg.dataset_stats)
                           else cfg.dataset_stats))
    mean_rgb, std_rgb = stats[cfg.labels_train][cfg.tactile_mode]
    tf = pth_transforms.Compose([
        pth_transforms.Pad([0, 40, 0, 40]),
        pth_transforms.Resize(cfg.crop_size),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize(mean_rgb, std_rgb)])

    z = np.load(NPZ)
    trim, shift = int(z["trim"]), 15
    ref_rows = z["reference_rows"]
    ours = z["force_normal_n"]

    with h5py.File(H5, "r") as f:
        frames = f["gelsight/left/frames"]
        nmax = len(frames) - 1
        bg = resize_crop_mini(frames[min(trim + int(ref_rows[0]) + shift, nmax)]
                              ).astype(np.int16)
        rows_idx = np.linspace(0, len(ours) - 1, 250).astype(int)
        preds = []
        for row in rows_idx:
            img = resize_crop_mini(frames[min(trim + int(row) + shift, nmax)])
            proc = np.clip(img.astype(np.int16) - bg + 127, 0, 255).astype(np.uint8)
            t = tf(Image.fromarray(proc)).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model.regressor(model.tactile_backbone(t)).cpu().squeeze(0).numpy()
            preds.append({"row": int(row), "fz": float(out[2]),
                          "fx": float(out[0]), "fy": float(out[1]),
                          "ours": float(ours[row])})
            if len(preds) % 50 == 0:
                print(f"  {len(preds)}/250", flush=True)
    json.dump(preds, open(out_path, "w"))
    print("REACT ANYFORCE DONE", flush=True)


if __name__ == "__main__":
    main()
