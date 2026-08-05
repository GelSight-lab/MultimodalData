"""Normal-force estimation for markerless GelSight Mini gel.

FEATS turned out not to transfer to our sensors: it is trained on
marker-dot gel, and on our markerless images it returns its no-contact
output regardless of contact (verified on the strongest-contact frame of
pushT episode_000). So the force estimate is built on what markerless gel
does support — photometric-stereo depth from the official GelSight SDK
(gsrobotics ``Reconstruction3D``: per-pixel MLP RGB->surface normal, Poisson
integration -> height map).

From indentation depth to force, the gel pad is modelled as a Winkler
elastic foundation — independent springs of stiffness E*/h per unit area:

    F_n = (E* / h) * sum(delta) * pixel_area,   E* = E / (1 - nu^2)

with gel thickness h and indentation map delta. This is the standard
thin-elastic-layer approximation; it is exact for a flat punch and within
tens of percent for curved indenters, so the *scale* carries model error
but the shape of the force signal over time does not. Constants below give
newtons under stated assumptions (E from the GelSight elastomer's reported
~0.1-0.2 MPa range; treat absolute values as estimates until a
scale-calibration against a known weight is done).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

GSROBOTICS = Path.home() / "gsrobotics"
NN_MINI = GSROBOTICS / "models" / "nnmini.pt"

# GelSight Mini imaging geometry: ~18.6 x 14.3 mm field of view
W, H = 320, 240
MM_PER_PIXEL = 18.6 / W                      # 0.058 mm/px
PIXEL_AREA_MM2 = MM_PER_PIXEL ** 2

# Winkler foundation constants (documented assumptions, not measurements)
GEL_E_MPA = 0.145                            # elastomer Young's modulus
GEL_NU = 0.48                                # near-incompressible
GEL_H_MM = 4.0                               # pad thickness
E_STAR_MPA = GEL_E_MPA / (1.0 - GEL_NU ** 2)
# F[N] = E*[MPa=N/mm^2] / h[mm] * sum(delta[mm]) * dA[mm^2]
FORCE_PER_MM3 = E_STAR_MPA / GEL_H_MM        # N per mm^3 of displaced volume

# The Poisson DCT solve is unreliable at the image boundary (Neumann
# conditions turn illumination drift into tall edge spikes — observed 1.1 mm
# phantom depth at col 319 on a no-contact frame), so a margin is excluded
# from force integration.
MARGIN_Y, MARGIN_X = 12, 16


@dataclass
class ForceEstimate:
    normal_n: float          # Winkler normal force, newtons
    volume_mm3: float        # displaced volume
    contact_area_mm2: float
    max_depth_mm: float
    depth_map: np.ndarray | None = None      # (H, W) mm, positive into the gel


class DepthForceEstimator:
    """Photometric-stereo depth + Winkler foundation -> normal force.

    Calibrated per sensor from a set of no-contact frames: their mean depth
    becomes the zero map (averaging down the per-pixel MLP noise), and the
    residual depth of the same frames sets the contact threshold — the p99.9
    no-contact residual with a 3-sigma floor. A fixed threshold cannot work
    here because reconstruction noise varies between sensors and lighting
    states (measured: 0.023 mm std on one sensor's interior).
    """

    def __init__(self, reference_frames: np.ndarray | list, use_gpu: bool = True):
        """reference_frames: no-contact full-view RGB frames (>=1)."""
        sys.path.insert(0, str(GSROBOTICS))
        from utilities.reconstruction import Reconstruction3D

        self.recon = Reconstruction3D(image_width=W, image_height=H,
                                      use_gpu=use_gpu)
        if self.recon.load_nn(str(NN_MINI)) is None:
            raise FileNotFoundError(NN_MINI)
        self.recon.depth_map_zero_counter = 51    # disable their live zeroing
        self.recon.depth_map_zero = 0.0

        refs = [reference_frames] if np.asarray(reference_frames).ndim == 3 \
            else list(reference_frames)
        raw = np.stack([self._raw_depth(self._resize(f)) for f in refs])
        # Median, not mean: the reference frames are chosen by an intensity
        # heuristic and occasionally include a lightly-touching or
        # not-yet-recovered-gel frame; a mean zero map absorbs that contact
        # and a std-based threshold explodes (observed 0.159 mm on pushT vs
        # 0.006 mm on motherboard from the same selection logic). Median +
        # MAD stay correct as long as fewer than half the frames are bad.
        self.recon.depth_map_zero = np.median(raw, axis=0)

        if len(refs) > 2:
            resid = (raw - self.recon.depth_map_zero) * MM_PER_PIXEL
            interior = resid[:, MARGIN_Y:-MARGIN_Y, MARGIN_X:-MARGIN_X]
            sigma = 1.4826 * float(np.median(np.abs(interior)))   # MAD -> std
            self.contact_threshold_mm = max(5.0 * sigma, 0.01)
        else:
            self.contact_threshold_mm = 0.07      # conservative default
        self.noise_std_mm = float(
            (raw - self.recon.depth_map_zero).std() * MM_PER_PIXEL)

    @staticmethod
    def _resize(frame_rgb: np.ndarray) -> np.ndarray:
        if frame_rgb.shape[:2] != (H, W):
            frame_rgb = cv2.resize(frame_rgb, (W, H))
        return frame_rgb

    def _raw_depth(self, frame_320: np.ndarray) -> np.ndarray:
        depth, *_ = self.recon.get_depthmap(frame_320)
        return depth

    def estimate(self, frame_rgb: np.ndarray,
                 keep_depth: bool = False) -> ForceEstimate:
        """Normal force for one full-view RGB frame."""
        depth_px = self._raw_depth(self._resize(frame_rgb))
        # positive = surface pressed toward the camera = into the gel
        indent_mm = depth_px * MM_PER_PIXEL
        indent_mm = cv2.GaussianBlur(indent_mm, (5, 5), 1.5)

        interior = np.zeros_like(indent_mm, dtype=bool)
        interior[MARGIN_Y:-MARGIN_Y, MARGIN_X:-MARGIN_X] = True
        contact = (indent_mm > self.contact_threshold_mm) & interior

        volume = float(indent_mm[contact].sum()) * PIXEL_AREA_MM2
        return ForceEstimate(
            normal_n=volume * FORCE_PER_MM3,
            volume_mm3=volume,
            contact_area_mm2=float(contact.sum()) * PIXEL_AREA_MM2,
            max_depth_mm=float(indent_mm[contact].max()) if contact.any() else 0.0,
            depth_map=indent_mm if keep_depth else None,
        )
