"""Calibration-free depth / height-map reconstruction for GelSight Mini.

Uses the GelSight-Inc pretrained markerless-Mini network (`nnmini.pt`,
RGB+xy -> surface-normal regression) so NO per-sensor calibration is needed.
The network architecture is reimplemented clean-room from the published
state-dict (fc 5->64->64->64->2, ReLU); only the public weight file is
downloaded on demand. Height is recovered by fast DCT Poisson integration.

Result is an APPROXIMATE relative height map (the pretrained net was fit to a
reference Mini, not this exact unit) — good for visualization, point clouds,
and relative geometry; not a metric-calibrated measurement. For metric depth,
collect a ball-indenter calibration and retrain.

Optional dependency: torch. Weights: GelSight Inc (GPL-3.0) — only the .pt
file is fetched; no GPL code is vendored here.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

_WEIGHTS_URL = "https://raw.githubusercontent.com/gelsightinc/gsrobotics/main/models/nnmini.pt"
_CACHE = Path(os.path.expanduser("~/.cache/react_toolbox/nnmini.pt"))


def _ensure_weights():
    if _CACHE.exists():
        return _CACHE
    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    import urllib.request
    urllib.request.urlretrieve(_WEIGHTS_URL, str(_CACHE))
    return _CACHE


class _RGB2NormNet:
    """Clean-room MLP matching nnmini.pt: (R,G,B,x,y) -> (nx,ny), ReLU."""

    def __init__(self):
        import torch
        import torch.nn as nn
        self.torch = torch
        net = nn.Sequential(
            nn.Linear(5, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 2))
        ck = torch.load(str(_ensure_weights()), map_location="cpu", weights_only=False)
        sd = ck["state_dict"]
        mapping = {"fc1": 0, "fc2": 2, "fc3": 4, "fc4": 6}
        with torch.no_grad():
            for name, idx in mapping.items():
                net[idx].weight.copy_(sd[f"{name}.weight"])
                net[idx].bias.copy_(sd[f"{name}.bias"])
        net.eval()
        self.net = net


_NET = None


def _get_net():
    global _NET
    if _NET is None:
        _NET = _RGB2NormNet()
    return _NET


def normals(frame, reference, mask=None):
    """Predict per-pixel surface normals (nx, ny, nz) for a GelSight frame.

    Inputs are the difference image (frame-reference) plus normalized pixel
    coords, matching the gsrobotics convention. Returns (H, W, 3) float32.
    """
    net = _get_net(); torch = net.torch
    H, W = frame.shape[:2]
    # Background-subtracted RGB normalized to [-1,1]/255 scale + xy in [0,1].
    # (Verified: this keeps predicted nx,ny in valid range; feeding 0-255 diff
    # pushes the MLP out of distribution and nz->0 blows up the gradients.)
    diff = (frame.astype(np.float32) - reference.astype(np.float32)) / 255.0
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    xs /= (W - 1); ys /= (H - 1)
    feat = np.stack([diff[..., 0], diff[..., 1], diff[..., 2], xs, ys], axis=-1)
    feat = feat.reshape(-1, 5)
    with torch.no_grad():
        out = net.net(torch.from_numpy(feat)).numpy()              # (HW, 2) = nx,ny
    nx = out[:, 0].reshape(H, W); ny = out[:, 1].reshape(H, W)
    nz = np.sqrt(np.clip(1 - nx**2 - ny**2, 1e-6, 1.0))
    n = np.stack([nx, ny, nz], axis=-1).astype(np.float32)
    if mask is not None:
        n[~mask] = [0, 0, 1]
    return n


def poisson_integrate(gx, gy):
    """Fast Poisson solver (DCT, Neumann BC): integrate gradients -> surface."""
    from scipy.fftpack import dct, idct
    H, W = gx.shape
    gxx = np.zeros_like(gx); gyy = np.zeros_like(gy)
    gxx[:, 1:] = gx[:, 1:] - gx[:, :-1]
    gyy[1:, :] = gy[1:, :] - gy[:-1, :]
    f = gxx + gyy
    fcos = dct(dct(f, axis=0, norm="ortho"), axis=1, norm="ortho")
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    denom = (2 * np.cos(np.pi * x / W) - 2) + (2 * np.cos(np.pi * y / H) - 2)
    denom[0, 0] = 1.0
    z = fcos / denom; z[0, 0] = 0
    return idct(idct(z, axis=0, norm="ortho"), axis=1, norm="ortho")


def height_map(frame, reference, mask=None):
    """Reconstruct a relative height map (H, W) float32 from one frame.

    height>0 = pushed in (contact). Approximate (uncalibrated). Requires torch
    + scipy; raises a clear error if torch is unavailable.
    """
    n = normals(frame, reference, mask=mask)
    nx, ny, nz = n[..., 0], n[..., 1], n[..., 2]
    gx = -nx / nz; gy = -ny / nz
    h = poisson_integrate(gx, gy).astype(np.float32)
    return h - h.min()
