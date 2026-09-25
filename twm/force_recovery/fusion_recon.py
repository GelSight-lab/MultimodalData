"""Band-fused photometric reconstruction: LUT amplitude + MLP structure.

Evidence behind the design (see task_plan Goal 6):
- The sphere-calibrated LUT recovers depth AMPLITUDE (unit-tested at
  104-128% at press centers) but blurs high-frequency structure — letter
  counters survive only faintly (sparse color bins + hole filling).
- The generic per-pixel MLP resolves edges sharply on ANY Mini (color
  CHANGES are in-domain even when absolute colors are not) but its
  low-frequency amplitude is garbage on foreign sensors (14x overshoot,
  rho=0 vs commanded depth on GlowTact).

So fuse in the gradient domain, one exact Poisson solve at the end:

    g_fused = lowpass(g_LUT) + s * highpass(g_MLP)

with the seam sigma splitting the bands and the scalar s self-calibrated
per frame by matching band energy in the mid-band both sources resolve.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from .lut_calibration import (BINS, DI_RANGE, CAL_OUT, MM_PER_PIXEL, W, H)

sys.path.insert(0, str(Path.home() / "gsrobotics"))
sys.path.insert(0, str(Path.home() / "gelsight_heightmap_reconstruction"
                       / "python_version"))

SEAM_SIGMA = 6.0          # px; below this scale trust the MLP, above the LUT


class FusionReconstructor:
    def __init__(self, lut_path: Path | None = None, use_gpu: bool = True):
        import torch
        from utilities.reconstruction import Reconstruction3D

        cal = np.load(lut_path or (CAL_OUT / "glowtact_lut.npz"))
        self.lut = cal["lut"]
        self.rec = Reconstruction3D(image_width=W, image_height=H,
                                    use_gpu=use_gpu)
        self.rec.load_nn(str(Path.home() / "gsrobotics/models/nnmini.pt"))
        self.torch = torch
        yy, xx = np.mgrid[0:H, 0:W]
        self._pos = np.stack([yy / H, xx / W], -1).reshape(-1, 2).astype(np.float32)

    # ---- sources -------------------------------------------------------
    def _lut_gradients(self, dI: np.ndarray) -> np.ndarray:
        q = np.clip((dI + DI_RANGE) / (2 * DI_RANGE) * (BINS - 1),
                    0, BINS - 1).astype(np.int32)
        g = self.lut[q[..., 0], q[..., 1], q[..., 2]].copy()
        mag = cv2.GaussianBlur(np.abs(dI).max(axis=2), (5, 5), 1.5)
        valid = mag > 8.0
        valid = cv2.morphologyEx(valid.astype(np.uint8), cv2.MORPH_OPEN,
                                 np.ones((3, 3), np.uint8)).astype(bool)
        g[~valid] = 0.0
        return g

    def _mlp_gradients(self, img: np.ndarray) -> np.ndarray:
        feat = np.concatenate([(img / 255.0).reshape(-1, 3).astype(np.float32),
                               self._pos], axis=1)
        with self.torch.no_grad():
            out = self.rec.net(self.torch.from_numpy(feat).float()
                               .to(self.rec.device)).cpu().numpy()
        nx = out[:, 0].reshape(H, W)
        ny = out[:, 1].reshape(H, W)
        # Out-of-domain colors push |n| >= 1, nz -> 0 and the slope
        # explodes (measured rms 220 vs the LUT's 0.027) — clamp nz and
        # the physical slope, then convert to mm-per-pixel so both
        # gradient sources share units.
        nz = np.sqrt(np.clip(1 - nx ** 2 - ny ** 2, 0.02, None))
        gx = np.clip(-nx / nz, -3.0, 3.0) * MM_PER_PIXEL
        gy = np.clip(-ny / nz, -3.0, 3.0) * MM_PER_PIXEL
        return np.stack([gx, gy], axis=-1)

    # ---- fusion --------------------------------------------------------
    def reconstruct(self, img: np.ndarray, ref: np.ndarray) -> np.ndarray:
        from .poisson import poisson_dirichlet as fast_poisson

        img = img.astype(np.float32)
        dI = img - ref
        gL = self._lut_gradients(dI)
        gM = self._mlp_gradients(img) - self._mlp_gradients(ref)

        def lp(a, s):
            return np.stack([cv2.GaussianBlur(a[..., c], (0, 0), s)
                             for c in range(2)], axis=-1)

        gL_lo = lp(gL, SEAM_SIGMA)
        gM_hi = gM - lp(gM, SEAM_SIGMA)
        # self-calibrate MLP band gain in the mid-band both sources resolve
        bandL = lp(gL, SEAM_SIGMA / 2) - lp(gL, SEAM_SIGMA * 2)
        bandM = lp(gM, SEAM_SIGMA / 2) - lp(gM, SEAM_SIGMA * 2)
        num = float(np.sum(bandL * bandM))
        den = float(np.sum(bandM * bandM)) + 1e-9
        s = np.clip(num / den, 0.0, 3.0)
        g = gL_lo + s * gM_hi
        return fast_poisson(g[..., 0], g[..., 1]), s
