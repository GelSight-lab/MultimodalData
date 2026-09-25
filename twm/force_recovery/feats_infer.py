"""FEATS inference on React GelSight frames.

FEATS (arXiv:2411.03315) is a U-net trained to reproduce FEA-computed force
distributions from GelSight Mini images: input is a 320x240 RGB view of the
gel, output a 24x32x3 grid of per-node forces in newtons — grid_x / grid_y
shear, grid_z normal (positive into the sensor). Summing a channel over the
grid gives the total force component.

Preprocessing must match their capture pipeline bit-for-bit, otherwise the
model sees the gel at the wrong scale: full frame -> resize (895, 672) ->
crop 1/7 border on each side -> drop last column -> resize (320, 240). Our
stored frames are the full sensor view at 640x480 RGB, so the same chain
applies directly; the color-channel flip in their code happens after the
geometry, so RGB input stays RGB.

Per-sensor calibration follows their ``calibrate_sensor.py``: the marker dots
of our sensor sit at a slightly different position than the training
sensor's, and a constant (dx, dy) translation — the mean nearest-dot
displacement against their reference no-contact image — aligns them.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

FEATS_ROOT = Path.home() / "feats" / "src" / "feats"
MODEL = FEATS_ROOT / "models" / "unet_09042025_124903_80.pt"
NORM = FEATS_ROOT / "data" / "labels" / "normalization_08042025_122519.npy"
NC_TRAIN = FEATS_ROOT / "calibration" / "nocontact_samples" / "nocontact_train_sensor.npy"

ENC_CHS = (3, 16, 32, 64, 128, 256)
DEC_CHS = (256, 128, 64, 32, 16)
OUTPUT_SIZE = (24, 32)


def preprocess(frame_rgb: np.ndarray) -> np.ndarray:
    """Full-view RGB frame (any 4:3 size) -> FEATS 320x240 RGB uint8."""
    img = cv2.resize(frame_rgb, (895, 672))
    bx, by = int(img.shape[0] / 7), int(np.floor(img.shape[1] / 7))
    img = img[bx:img.shape[0] - bx, by:img.shape[1] - by]
    img = img[:, :-1]
    return cv2.resize(img, (320, 240))


def _dot_centers(img_rgb: np.ndarray, dot_fraction: float = 0.056) -> np.ndarray:
    """Centers of the black marker dots.

    The reference implementation thresholds at gray 55, tuned to their
    sensor's illumination; our units' dots bottom out around 56, so that
    fixed cut detects nothing. The marker pattern covers a known fraction of
    the image (~5.6% on their reference), so the threshold is chosen as that
    percentile of the blurred image instead — invariant to overall
    brightness. Components far from the marker-dot size are rejected.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 2)
    thresh = np.percentile(blurred, dot_fraction * 100.0)
    _, black = cv2.threshold(blurred, thresh, 255, cv2.THRESH_BINARY_INV)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        black.astype(np.uint8))
    if n <= 1:
        return np.empty((0, 2))
    # marker dots on the 320x240 view are ~15-100 px blobs
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = (areas >= 8) & (areas <= 400)
    return centroids[1:][keep]


def calibrate(nocontact_rgb_320: np.ndarray) -> tuple[np.ndarray, dict]:
    """(dx, dy) aligning our sensor's dots to the training sensor's.

    Mean nearest-neighbour displacement, as in their calibrate_sensor.py.
    Returns the shift and diagnostics (dot counts and displacement spread —
    a large spread means the dot matching failed and the shift is garbage).
    """
    train = np.load(NC_TRAIN, allow_pickle=True).item()["gs_img"]
    centers_train = _dot_centers(train)
    centers_test = _dot_centers(nocontact_rgb_320)
    vecs = []
    for center in centers_test:
        d = np.linalg.norm(centers_train - center, axis=1)
        vecs.append(centers_train[np.argmin(d)] - center)
    vecs = np.asarray(vecs)
    shift = vecs.mean(axis=0)
    return shift, {
        "n_dots_train": len(centers_train),
        "n_dots_ours": len(centers_test),
        "shift_px": shift.tolist(),
        "spread_px": vecs.std(axis=0).tolist(),
    }


@dataclass
class ForceGrids:
    """One frame's predicted force distributions, in newtons per grid node."""
    shear_x: np.ndarray      # (24, 32)
    shear_y: np.ndarray      # (24, 32)
    normal: np.ndarray       # (24, 32), positive = pressing into the gel

    @property
    def total_normal(self) -> float:
        return float(self.normal.sum())

    @property
    def total_shear(self) -> float:
        return float(np.hypot(self.shear_x.sum(), self.shear_y.sum()))


class FeatsPredictor:
    """Batched FEATS inference with per-sensor dot calibration."""

    def __init__(self, device: str = "cuda:0", shift: np.ndarray | None = None):
        import torch

        sys.path.insert(0, str(FEATS_ROOT / "src"))
        from models.unet import UNet

        self.torch = torch
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = UNet(enc_chs=ENC_CHS, dec_chs=DEC_CHS,
                          out_sz=OUTPUT_SIZE).to(self.device)
        self.model.load_state_dict(torch.load(str(MODEL), map_location=self.device))
        self.model.eval()

        norm = np.load(NORM, allow_pickle=True).item()
        # x/y are scaled to (-1, 1), z to (0, 1)
        self._scale = {}
        for key, lo in (("grid_x", -1.0), ("grid_y", -1.0), ("grid_z", 0.0)):
            self._scale[key] = (norm[key]["min"], norm[key]["max"], lo)
        self.shift = shift

    def _unnormalize(self, out: np.ndarray, key: str, channel: int) -> np.ndarray:
        mn, mx, lo = self._scale[key]
        span = 1.0 - lo
        return (out[..., channel] - lo) / span * (mx - mn) + mn

    def predict(self, frames_320: np.ndarray, batch_size: int = 64) -> list[ForceGrids]:
        """frames_320: (N, 240, 320, 3) RGB uint8 -> per-frame force grids."""
        torch = self.torch
        frames_320 = np.asarray(frames_320)
        if frames_320.ndim == 3:
            frames_320 = frames_320[None]
        if self.shift is not None:
            M = np.float32([[1, 0, self.shift[0]], [0, 1, self.shift[1]]])
            frames_320 = np.stack([
                cv2.warpAffine(f, M, (320, 240), borderMode=cv2.BORDER_REPLICATE)
                for f in frames_320])

        results = []
        with torch.no_grad():
            for i in range(0, len(frames_320), batch_size):
                batch = frames_320[i:i + batch_size].astype(np.float32) / 255.0
                x = torch.from_numpy(batch).permute(0, 3, 1, 2).to(self.device)
                y = self.model(x).permute(0, 2, 3, 1).cpu().numpy()
                for out in y:
                    results.append(ForceGrids(
                        shear_x=self._unnormalize(out, "grid_x", 0),
                        shear_y=self._unnormalize(out, "grid_y", 1),
                        normal=self._unnormalize(out, "grid_z", 2)))
        return results
