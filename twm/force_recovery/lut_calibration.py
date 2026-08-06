"""Per-sensor photometric calibration from spherical presses (classic LUT).

Reimplements the Dong/Yuan GelSight calibration without a GUI, supervised by
GlowTact's `round` family: a smooth spherical indenter pressed at logged
depths across the whole pad, with a per-family `initial.jpg` reference.

Why: the generic gsrobotics MLP recovers only ~25% of true indentation depth
and saturates with depth (peak-depth vs commanded-z rho = 0.39) — the depth
AMPLITUDE is broken, which caps any force model built on it. The classic
pipeline is per-sensor and works in difference space:

    dI = img - ref  →  3D RGB lookup table  →  surface gradient (gx, gy)
                        →  Poisson integration  →  height map [mm]

Geometry supervision needs the sphere radius R and true depth per frame.
Neither is given, but for a sphere the contact radius obeys a^2 = 2 R (z - z0)
in the shallow regime — regressing measured circle radii against commanded z
recovers both R and the unknown depth datum z0 in one line.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .run_episode import OUT_ROOT

GLOWTACT = OUT_ROOT / "glowtact" / "GelSight_force_final_14716"
CAL_OUT = OUT_ROOT / "lut_calibration"

# same view as the rest of the pipeline: 1/7 border crop -> 320x240
W, H = 320, 240
MM_PER_PIXEL = 18.6 * (1 - 2 / 7) / W

PAT = re.compile(r"\|(?P<idx>\d+)pos\|(?P<x>[-\d.]+), (?P<y>[-\d.]+), "
                 r"(?P<z>[-\d.]+) f\|(?P<f>[\d.]+)\.jpg$")

BINS = 90
DI_RANGE = 90.0          # dI clipped to [-90, 90] per channel


def crop(img: np.ndarray) -> np.ndarray:
    bx, by = int(img.shape[0] / 7), int(np.floor(img.shape[1] / 7))
    return cv2.resize(img[bx + 2:img.shape[0] - bx, by:img.shape[1] - by],
                      (W, H))


def load_family(family: str) -> tuple[np.ndarray, list[dict]]:
    from PIL import Image

    ref = crop(np.asarray(Image.open(GLOWTACT / family / "initial.jpg")
                          .convert("RGB"))).astype(np.float32)
    rows = []
    for p in (GLOWTACT / family).glob("*.jpg"):
        m = PAT.search(p.name)
        if m:
            rows.append({"path": p, "z": -float(m["z"]), "f": float(m["f"]),
                         "x": float(m["x"]), "y": float(m["y"])})
    return ref, rows


def f01_compensation(ref: np.ndarray) -> np.ndarray:
    """Yuan's vignette compensation: dI is scaled by 1+((mean/f0)-1)*2 so a
    given slope produces the same compensated color change anywhere on the
    pad. Fitted post-hoc, the spatial gain field explained 80% of the
    residual depth variance — this removes it at the optical source."""
    f0 = cv2.GaussianBlur(ref.sum(axis=2), (31, 31), 0)
    t = float(f0.mean())
    return (1.0 + ((t / np.maximum(f0, 1.0)) - 1.0) * 2.0)[..., None]


def detect_circle(dI: np.ndarray) -> tuple[float, float, float] | None:
    """Contact circle (cx, cy, a_px) from the difference image."""
    mag = np.abs(dI).max(axis=2)
    mag = cv2.GaussianBlur(mag, (7, 7), 2)
    thr = max(12.0, np.percentile(mag, 97) * 0.4)
    mask = (mag > thr).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 80:
        return None
    (cx, cy), a = cv2.minEnclosingCircle(c)
    circularity = area / (np.pi * a * a + 1e-9)
    if circularity < 0.55:                      # not a clean disc
        return None
    return float(cx), float(cy), float(a)


@dataclass
class SphereGeometry:
    R_mm: float
    z0_mm: float
    n_used: int
    fit_r2: float


def fit_sphere_geometry(ref, rows, margin: float = 18.0) -> tuple[SphereGeometry, list[dict]]:
    """Regress a_mm^2 = 2 R (z - z0) over interior, clean-circle presses."""
    from PIL import Image

    fits = []
    for r in rows:
        img = crop(np.asarray(Image.open(r["path"]).convert("RGB"))
                   ).astype(np.float32)
        det = detect_circle(img - ref)
        if det is None:
            continue
        cx, cy, a = det
        if not (margin + a < cx < W - margin - a
                and margin + a < cy < H - margin - a):
            continue                             # clipped by the border
        fits.append({**r, "cx": cx, "cy": cy, "a_px": a,
                     "a_mm": a * MM_PER_PIXEL})
    from scipy.optimize import least_squares

    z = np.array([f["z"] for f in fits])
    a2 = np.array([f["a_mm"] ** 2 for f in fits])

    # exact sphere: a^2 = d (2R - d) with d = clip(z - z0, 0, R)
    def model(params, zz):
        R, z0 = params
        d = np.clip(zz - z0, 1e-3, R)
        return d * (2 * R - d)

    sol = least_squares(lambda pr: a2 - model(pr, z), x0=[3.0, 0.0],
                        bounds=([0.5, -2], [15, 2]))
    resid = a2 - model(sol.x, z)
    keep = np.abs(resid) < 3 * np.std(resid)
    sol = least_squares(lambda pr: a2[keep] - model(pr, z[keep]), x0=sol.x,
                        bounds=([0.5, -2], [15, 2]))
    pred = model(sol.x, z[keep])
    r2 = 1 - np.sum((a2[keep] - pred) ** 2) / np.sum(
        (a2[keep] - a2[keep].mean()) ** 2)
    geom = SphereGeometry(R_mm=float(sol.x[0]), z0_mm=float(sol.x[1]),
                          n_used=int(keep.sum()), fit_r2=float(r2))
    fits = [f for f, kp in zip(fits, keep) if kp]
    return geom, fits


def build_lut(ref, fits, geom, inner_frac: float = 0.97) -> dict:
    """Accumulate (dI -> gradient) over all clean sphere presses."""
    from PIL import Image

    ssum = np.zeros((BINS, BINS, BINS, 2), np.float64)
    cnt = np.zeros((BINS, BINS, BINS), np.int64)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    f01 = f01_compensation(ref)

    for f in fits:
        img = crop(np.asarray(Image.open(f["path"]).convert("RGB"))
                   ).astype(np.float32)
        dI = img - ref
        depth = f["z"] - geom.z0_mm
        if not (0.15 <= depth <= 0.9 * geom.R_mm):
            continue                       # exact-cap regime only
        rx = (xx - f["cx"])
        ry = (yy - f["cy"])
        r_px = np.sqrt(rx ** 2 + ry ** 2)
        r_mm = r_px * MM_PER_PIXEL
        inside = (r_px < inner_frac * f["a_px"]) & (r_mm < 0.985 * geom.R_mm)
        # exact sphere surface: h(r) = depth - (R - sqrt(R^2 - r^2))
        # dh/dr = -r / sqrt(R^2 - r^2)   [mm/mm]; per-pixel: * MM_PER_PIXEL
        denom = np.sqrt(np.clip(geom.R_mm ** 2 - r_mm ** 2, 1e-6, None))
        slope = -(r_mm / denom) * MM_PER_PIXEL      # solver wants (-gx,-gy) for positive bumps
        with np.errstate(invalid="ignore", divide="ignore"):
            gx = np.where(r_px > 1e-6, slope * rx / np.maximum(r_px, 1e-6), 0.0)
            gy = np.where(r_px > 1e-6, slope * ry / np.maximum(r_px, 1e-6), 0.0)
        q = np.clip((dI[inside] + DI_RANGE) / (2 * DI_RANGE) * (BINS - 1),
                    0, BINS - 1).astype(np.int32)
        idx = (q[:, 0], q[:, 1], q[:, 2])
        np.add.at(ssum, idx + (0,), gx[inside])
        np.add.at(ssum, idx + (1,), gy[inside])
        np.add.at(cnt, idx, 1)

    have = cnt > 0
    lut = np.zeros((BINS, BINS, BINS, 2), np.float32)
    lut[have] = (ssum[have] / cnt[have, None]).astype(np.float32)
    return {"lut": lut, "count": cnt, "filled_frac": float(have.mean())}


def fill_lut_holes(lut: np.ndarray, cnt: np.ndarray,
                   min_count: int = 3) -> np.ndarray:
    """Nearest-neighbour fill of unobserved color bins."""
    from scipy.ndimage import distance_transform_edt

    have = cnt >= min_count
    out = lut.copy()
    _, idx = distance_transform_edt(~have, return_indices=True)
    for c in range(2):
        out[..., c] = lut[..., c][tuple(idx)]
    return out


def reconstruct(dI: np.ndarray, lut: np.ndarray,
                f01: np.ndarray | None = None) -> np.ndarray:
    """dI -> gradients via LUT -> Poisson -> height map [mm]."""
    import sys

    # NOT the gsrobotics DCT solver: benchmarked on an analytic sphere-cap
    # gradient field it returns only 39% of the true amplitude (a silent
    # attenuation that also capped the MLP pipeline); Dong's fast_poisson
    # reproduces the field at 100.7%.
    sys.path.insert(0, str(Path.home()
                           / "gelsight_heightmap_reconstruction"
                           / "python_version"))
    from fast_poisson import fast_poisson

    if f01 is not None:
        dI = dI * f01
    q = np.clip((dI + DI_RANGE) / (2 * DI_RANGE) * (BINS - 1),
                0, BINS - 1).astype(np.int32)
    g = lut[q[..., 0], q[..., 1], q[..., 2]]
    # validmask (Yuan): only trust the table where the color actually moved;
    # elsewhere the gel is flat and the gradient is zero by definition.
    # Without this, noise pixels land in sparse bins whose nearest-filled
    # values are steep rim slopes, and Poisson hallucinates spikes that a
    # percentile-peak then reads as depth (measured: depth rho collapsed to
    # ~0 while the amplitude at true contact centers stayed correct).
    mag = np.abs(dI).max(axis=2)
    mag = cv2.GaussianBlur(mag, (5, 5), 1.5)
    valid = mag > 8.0
    valid = cv2.morphologyEx(valid.astype(np.uint8), cv2.MORPH_OPEN,
                             np.ones((3, 3), np.uint8)).astype(bool)
    g[~valid] = 0.0
    return fast_poisson(g[..., 0], g[..., 1])


def calibrate(save: bool = True) -> dict:
    ref, rows = load_family("round")
    geom, fits = fit_sphere_geometry(ref, rows)
    print(f"sphere fit: R={geom.R_mm:.2f} mm, z0={geom.z0_mm:.2f} mm, "
          f"n={geom.n_used}, R^2={geom.fit_r2:.3f}", flush=True)
    raw = build_lut(ref, fits, geom)
    lut = fill_lut_holes(raw["lut"], raw["count"])
    print(f"LUT filled bins: {raw['filled_frac']*100:.1f}% observed, "
          f"holes nearest-filled", flush=True)
    if save:
        CAL_OUT.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(CAL_OUT / "glowtact_lut.npz",
                            lut=lut, count=raw["count"],
                            R_mm=geom.R_mm, z0_mm=geom.z0_mm,
                            bins=BINS, di_range=DI_RANGE)
        (CAL_OUT / "geometry.json").write_text(json.dumps(vars(geom), indent=2))
    return {"geom": geom, "lut": lut}


if __name__ == "__main__":
    calibrate()
