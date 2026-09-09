"""RealSense factory calibration, and depth→color alignment done offline.

The recorder can run the cameras with the SDK's `rs.align` switched off
(`--raw_depth`): aligning three cameras costs about 0.75 of a CPU core, and
on this rig the CPU is what the writer runs out of first. Alignment is a
deterministic reprojection — depth intrinsics, color intrinsics and the
depth→color extrinsics are all the SDK uses — so doing it here, later, off
the recording machine, reproduces the same pixels.

`align_depth_to_color` reimplements librealsense's `align_z_to_other`
(`src/proc/align.cpp`), including the parts that are easy to miss and that
make the result differ from a naive reprojection:

  * each depth pixel is mapped as a *quad*: its top-left corner
    (x-0.5, y-0.5) and its bottom-right corner (x+0.5, y+0.5) are both
    reprojected, and every color pixel in the rectangle between them is
    written. With this rig's near-identical intrinsics that dilates the
    depth by one pixel toward +x/+y — the SDK's output looks the same.
  * both corners are rounded with `ceil`, and the rectangle is clipped to
    the color image rather than dropped when it hangs over an edge.
  * where several depth pixels land on one color pixel the *nearest* wins.
  * the value written is the original depth reading, not its z in the color
    frame (verified: every value in an SDK-aligned frame also occurs in the
    raw frame).

The projection matches `rs2_deproject_pixel_to_point` →
`rs2_transform_point_to_point` → `rs2_project_point_to_pixel` to the last
bit (checked against pyrealsense2's own scalar functions, 0.000000 px over
3000 sampled pixels). The rounding and edge rules above were then fitted to
the SDK's output, because librealsense's C++ is not on this machine to read:
on a live D415 frame this reproduces `rs.align` for 99.998 % of pixels, the
remaining 6 pixels in 307 200 being float ties that differ by ≤ 12 mm.
`python -m twm.realsense_align verify` re-runs that comparison.

The rig's D415s report zero distortion coefficients at 640x480, so the
projection is a plain pinhole. Non-zero coefficients raise rather than being
ignored: a silently mis-projected depth map looks like a slightly
miscalibrated rig, not like a bug.

    python -m twm.realsense_align export            # read the cameras, write JSON
    python -m twm.realsense_align verify            # compare against rs.align live
    python -m twm.realsense_align apply in.h5 out.h5
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

CALIB_SUBDIR = Path(__file__).resolve().parent / "calibration" / "realsense"


@dataclass(frozen=True)
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    ppx: float
    ppy: float
    model: str
    coeffs: Tuple[float, ...]


@dataclass(frozen=True)
class DepthToColor:
    """Everything `rs.align` uses, for one camera at one resolution."""
    serial: str
    depth: Intrinsics
    color: Intrinsics
    rotation: Tuple[float, ...]      # 9 values, COLUMN-major, as rs2_extrinsics stores them
    translation: Tuple[float, ...]   # 3 values, metres
    depth_scale: float               # metres per stored depth unit (0.001 = millimetres)
    position: str = ""               # left / middle / right, for humans
    created_at: str = ""

    def rotation_matrix(self) -> np.ndarray:
        """Row-major 3x3. `rs2_transform_point_to_point` reads the flat array
        column-major; reading it row-major transposes the rotation and
        mis-aligns every frame by a few pixels — small enough to look like a
        calibration drift rather than a bug."""
        return np.asarray(self.rotation, np.float64).reshape(3, 3).T


# ── serialization ────────────────────────────────────────────────────────────

def _intrinsics_from_dict(d: Dict) -> Intrinsics:
    return Intrinsics(width=int(d["width"]), height=int(d["height"]),
                      fx=float(d["fx"]), fy=float(d["fy"]),
                      ppx=float(d["ppx"]), ppy=float(d["ppy"]),
                      model=str(d["model"]), coeffs=tuple(float(c) for c in d["coeffs"]))


def calibration_to_dict(calib: DepthToColor) -> Dict:
    d = asdict(calib)
    d["rotation"] = list(calib.rotation)
    d["translation"] = list(calib.translation)
    d["depth"]["coeffs"] = list(calib.depth.coeffs)
    d["color"]["coeffs"] = list(calib.color.coeffs)
    return d


def calibration_from_dict(d: Dict) -> DepthToColor:
    return DepthToColor(serial=str(d["serial"]),
                        depth=_intrinsics_from_dict(d["depth"]),
                        color=_intrinsics_from_dict(d["color"]),
                        rotation=tuple(float(v) for v in d["rotation"]),
                        translation=tuple(float(v) for v in d["translation"]),
                        depth_scale=float(d["depth_scale"]),
                        position=str(d.get("position", "")),
                        created_at=str(d.get("created_at", "")))


def save_calibration(calib: DepthToColor, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(calibration_to_dict(calib), indent=2, sort_keys=True))
    return path


def load_calibration(path) -> DepthToColor:
    return calibration_from_dict(json.loads(Path(path).read_text()))


def calibration_path(serial: str, directory=None) -> Path:
    return Path(directory or CALIB_SUBDIR) / f"{serial}.json"


def load_for_serial(serial: str, directory=None) -> DepthToColor:
    """The stored calibration for one camera.

    Raises rather than guessing: aligning with another camera's extrinsics
    produces a plausible-looking depth map that is wrong everywhere.
    """
    path = calibration_path(serial, directory)
    if not path.is_file():
        raise FileNotFoundError(
            f"no RealSense calibration for serial {serial} at {path}. Record it "
            f"with `python -m twm.realsense_align export` while the camera is "
            f"attached.")
    return load_calibration(path)


# ── alignment ────────────────────────────────────────────────────────────────

def _check_pinhole(intr: Intrinsics, which: str) -> None:
    if any(c != 0.0 for c in intr.coeffs):
        raise NotImplementedError(
            f"{which} camera reports distortion coefficients {tuple(intr.coeffs)}; "
            f"this aligner implements the pinhole case only (the rig's D415s report "
            f"zeros at 640x480). Implement the Brown-Conrady terms before using it.")


def _ceil_to_int(v: np.ndarray) -> np.ndarray:
    """The SDK's corner rounding, fitted to its output: plain `ceil`.

    Round-half-up (`static_cast<int>(x + 0.5f)`) agrees with rs.align on only
    61 % of pixels; `ceil` on 99.99 %.
    """
    return np.ceil(v).astype(np.int32)


def align_depth_to_color(depth: np.ndarray, calib: DepthToColor) -> np.ndarray:
    """Depth in the depth camera's frame → depth on the color camera's grid.

    Same shape convention as the recorder stores: uint16, one unit =
    `calib.depth_scale` metres. Pixels no depth pixel reaches stay 0.
    """
    d = np.asarray(depth)
    if d.ndim != 2 or d.shape != (calib.depth.height, calib.depth.width):
        raise ValueError(
            f"depth image is {tuple(d.shape)}; this calibration is for "
            f"{(calib.depth.height, calib.depth.width)}")
    _check_pinhole(calib.depth, "depth")
    _check_pinhole(calib.color, "color")

    out_h, out_w = calib.color.height, calib.color.width
    out = np.zeros((out_h, out_w), np.uint16)
    if not d.any():
        return out

    z = d.astype(np.float32) * np.float32(calib.depth_scale)      # metres
    R = calib.rotation_matrix().astype(np.float32)
    t = np.asarray(calib.translation, np.float32)
    xs = np.arange(calib.depth.width, dtype=np.float32)
    ys = np.arange(calib.depth.height, dtype=np.float32)

    def project_corner(offset: float):
        """Pixel coordinates in the color image of one corner of every depth
        pixel, at that pixel's own depth."""
        u = (xs + np.float32(offset) - np.float32(calib.depth.ppx)) / np.float32(calib.depth.fx)
        v = (ys + np.float32(offset) - np.float32(calib.depth.ppy)) / np.float32(calib.depth.fy)
        X = u[None, :] * z
        Y = v[:, None] * z
        Xc = R[0, 0] * X + R[0, 1] * Y + R[0, 2] * z + t[0]
        Yc = R[1, 0] * X + R[1, 1] * Y + R[1, 2] * z + t[1]
        Zc = R[2, 0] * X + R[2, 1] * Y + R[2, 2] * z + t[2]
        with np.errstate(divide="ignore", invalid="ignore"):
            px = Xc / Zc * np.float32(calib.color.fx) + np.float32(calib.color.ppx)
            py = Yc / Zc * np.float32(calib.color.fy) + np.float32(calib.color.ppy)
        ok = (Zc > 0) & np.isfinite(px) & np.isfinite(py)
        return _ceil_to_int(np.where(ok, px, 0)), _ceil_to_int(np.where(ok, py, 0)), ok

    x0, y0, ok0 = project_corner(-0.5)
    x1, y1, ok1 = project_corner(+0.5)
    # A quad hanging over an edge is clipped, not dropped: dropping it loses
    # the ~200 border pixels the SDK keeps.
    valid = (d != 0) & ok0 & ok1 & (x1 >= 0) & (y1 >= 0) & (x0 < out_w) & (y0 < out_h)
    if not valid.any():
        return out

    values = d[valid]
    x0v = np.clip(x0[valid], 0, out_w - 1)
    y0v = np.clip(y0[valid], 0, out_h - 1)
    span_x = np.clip(x1[valid], 0, out_w - 1) - x0v
    span_y = np.clip(y1[valid], 0, out_h - 1) - y0v
    # Writing in far-to-near order lets a plain scatter leave the nearest
    # value behind, which is the SDK's per-pixel min without ufunc.at.
    order = np.argsort(values, kind="stable")[::-1]
    values, x0v, y0v = values[order], x0v[order], y0v[order]
    span_x, span_y = span_x[order], span_y[order]

    for dy in range(int(span_y.max()) + 1):
        for dx in range(int(span_x.max()) + 1):
            m = (span_x >= dx) & (span_y >= dy)
            if not m.any():
                continue
            layer = np.zeros(out_h * out_w, np.uint16)
            layer[(y0v[m] + dy) * out_w + (x0v[m] + dx)] = values[m]
            layer = layer.reshape(out_h, out_w)
            # 0 means "nothing written", so it must not win a minimum().
            out = np.where(out == 0, layer,
                           np.where(layer == 0, out, np.minimum(out, layer)))
    return out


# ── reading the cameras ──────────────────────────────────────────────────────

def export_from_hardware(serials: Sequence[str], width: int = 640, height: int = 480,
                         fps: int = 30) -> List[DepthToColor]:
    """Start each camera briefly and read its factory calibration.

    Intrinsics depend on the stream profile, so this must run at the same
    resolution the recorder uses.
    """
    import time

    import pyrealsense2 as rs

    from twm.recorder.config import realsense_position

    out = []
    for serial in serials:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        profile = pipeline.start(config)
        try:
            dp = profile.get_stream(rs.stream.depth).as_video_stream_profile()
            cp = profile.get_stream(rs.stream.color).as_video_stream_profile()
            di, ci = dp.get_intrinsics(), cp.get_intrinsics()
            ex = dp.get_extrinsics_to(cp)
            scale = profile.get_device().first_depth_sensor().get_depth_scale()
        finally:
            pipeline.stop()
        out.append(DepthToColor(
            serial=serial,
            depth=Intrinsics(di.width, di.height, di.fx, di.fy, di.ppx, di.ppy,
                             str(di.model).split(".")[-1], tuple(di.coeffs)),
            color=Intrinsics(ci.width, ci.height, ci.fx, ci.fy, ci.ppx, ci.ppy,
                             str(ci.model).split(".")[-1], tuple(ci.coeffs)),
            rotation=tuple(ex.rotation), translation=tuple(ex.translation),
            depth_scale=float(scale), position=realsense_position(serial),
            created_at=time.strftime("%Y-%m-%dT%H:%M:%S")))
    return out


def compare_against_sdk(serial: str, frames: int = 20, width: int = 640,
                        height: int = 480, fps: int = 30) -> Dict[str, float]:
    """Align the same frames both ways and report how far apart they are.

    Both depth maps come from one frameset, so any difference is the
    aligner's, not the camera's.
    """
    import time

    import pyrealsense2 as rs

    calib = load_for_serial(serial)
    time.sleep(1.0)          # let the previous camera's pipeline release the USB bus
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    align = rs.align(rs.stream.color)
    pipeline.start(config)
    equal = total = mismatched = 0
    worst = 0
    try:
        for _ in range(5):                       # let auto-exposure settle
            pipeline.wait_for_frames(timeout_ms=15000)
        for _ in range(frames):
            fs = pipeline.wait_for_frames(timeout_ms=15000)
            raw = np.asanyarray(fs.get_depth_frame().get_data()).copy()
            sdk = np.asanyarray(align.process(fs).get_depth_frame().get_data()).copy()
            mine = align_depth_to_color(raw, calib)
            diff = mine.astype(np.int32) - sdk.astype(np.int32)
            equal += int((diff == 0).sum())
            total += diff.size
            mismatched += int((diff != 0).sum())
            worst = max(worst, int(np.abs(diff).max()))
    finally:
        pipeline.stop()
    return {"frames": frames, "pixels": total, "identical_fraction": equal / total,
            "mismatched_pixels": mismatched, "worst_abs_diff_mm": worst}


# ── applying it to a recorded episode ────────────────────────────────────────

def apply_to_episode(in_path, out_path, directory=None, progress=print) -> Path:
    """Copy an episode and replace its raw depth with aligned depth."""
    import shutil

    import h5py
    try:
        import hdf5plugin  # noqa: F401  (registers BLOSC)
    except ImportError:
        pass

    in_path, out_path = Path(in_path), Path(out_path)
    with h5py.File(in_path, "r") as f:
        if bool(f["metadata"].attrs.get("depth_aligned", True)):
            raise ValueError(f"{in_path} already holds aligned depth")
        serials = [s.decode() if isinstance(s, bytes) else str(s)
                   for s in f["metadata"].attrs["realsense_serials"]]
        calibs = [load_for_serial(s, directory) for s in serials]

    shutil.copy2(in_path, out_path)
    with h5py.File(out_path, "a") as f:
        for i, calib in enumerate(calibs):
            ds = f[f"realsense/cam{i}/depth"]
            for k in range(ds.shape[0]):
                ds[k] = align_depth_to_color(ds[k], calib)
                if progress and k % 200 == 0:
                    progress(f"  cam{i} ({calib.position}) {k + 1}/{ds.shape[0]}")
        f["metadata"].attrs["depth_aligned"] = True
        f["metadata"].attrs["depth_aligned_by"] = "twm.realsense_align"
    return out_path


# ── CLI ──────────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="python -m twm.realsense_align",
                                description="RealSense calibration and offline depth alignment.")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("export", help="read the attached cameras' factory calibration")
    e.add_argument("--serials", default=None, help="comma-separated (default: the rig's three)")
    e.add_argument("--dir", default=None, help=f"output directory (default: {CALIB_SUBDIR})")

    v = sub.add_parser("verify", help="compare this aligner against rs.align, live")
    v.add_argument("--serials", default=None)
    v.add_argument("--frames", type=int, default=20)

    a = sub.add_parser("apply", help="write an aligned copy of a raw-depth episode")
    a.add_argument("input")
    a.add_argument("output")
    a.add_argument("--dir", default=None, help="calibration directory")

    args = p.parse_args(argv)
    from twm.recorder.config import REALSENSE_SERIALS
    serials = ([s.strip() for s in args.serials.split(",") if s.strip()]
               if getattr(args, "serials", None) else list(REALSENSE_SERIALS))

    if args.cmd == "export":
        for calib in export_from_hardware(serials):
            path = save_calibration(calib, calibration_path(calib.serial, args.dir))
            print(f"{calib.serial} ({calib.position or 'unknown'}) -> {path}")
        return 0

    if args.cmd == "verify":
        ok = True
        for serial in serials:
            r = compare_against_sdk(serial, frames=args.frames)
            same = r["identical_fraction"]
            ok &= same == 1.0
            print(f"{serial}: {same * 100:.4f}% of pixels identical to rs.align "
                  f"({r['mismatched_pixels']} differ, worst {r['worst_abs_diff_mm']} mm, "
                  f"{r['frames']} frames)")
        return 0 if ok else 1

    apply_to_episode(args.input, args.output, args.dir)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
