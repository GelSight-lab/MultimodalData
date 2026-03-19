#!/usr/bin/env python3
"""
Calibrate OptiTrack → RealSense camera coordinate transformation.

Place reflective balls visible to both OptiTrack and the overhead RealSense camera.
For each ball: click in the camera image to get its 3D camera-frame position,
then type its OptiTrack 3D position in the terminal.

Usage:
    python -m twm.calibrate_mocap_to_cam --serial 217222066989 --num_points 6
    python -m twm.calibrate_mocap_to_cam --serial 217222066989 --num_points 6 \
        --output calibration/T_mocap_to_cam.json

Full projection chain (after calibration):
    P_gel_mocap  = T_rigid_to_mocap @ P_gel_in_rigid      (from gelsight calibration)
    P_gel_cam    = T_mocap_to_cam   @ P_gel_mocap          (this calibration)
    pixel (u, v) = project(K, P_gel_cam)                   (camera intrinsics)

All 3D coordinates are in millimetres.
OptiTrack input is in native units (typically metres) and scaled by --mocap_scale.
"""

import argparse
import json
import time
import numpy as np
import cv2
import pyrealsense2 as rs
from pathlib import Path
from scipy.spatial.transform import Rotation

# ── Mouse callback ────────────────────────────────────────────────────────────
_click_xy = None


def _on_mouse(event, x, y, flags, param):
    global _click_xy
    if event == cv2.EVENT_LBUTTONDOWN:
        _click_xy = (x, y)


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_depth_at(depth_frame, x, y, patch=7):
    """Median depth in a patch×patch window around (x, y), ignoring zeros."""
    h, w = depth_frame.shape
    half = patch // 2
    x0, x1 = max(0, x - half), min(w, x + half + 1)
    y0, y1 = max(0, y - half), min(h, y + half + 1)
    vals = depth_frame[y0:y1, x0:x1].flatten()
    valid = vals[vals > 0]
    return float(np.median(valid)) if len(valid) > 0 else 0.0


def deproject(intrinsics, px, py, depth_mm):
    """Pixel (px, py) + depth (mm) → 3D point in camera frame (mm)."""
    return np.array(
        rs.rs2_deproject_pixel_to_point(intrinsics, [float(px), float(py)], depth_mm),
        dtype=np.float64,
    )


def rigid_transform_svd(P, Q):
    """
    Compute 4×4 rigid transform T such that Q ≈ T @ [P; 1].
    P, Q: (N, 3) corresponding 3D point pairs.
    Arun's SVD method with reflection correction.
    """
    assert P.shape == Q.shape and P.shape[0] >= 3
    cP, cQ = P.mean(axis=0), Q.mean(axis=0)
    H = (P - cP).T @ (Q - cQ)
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    t = cQ - R @ cP
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def parse_vector(s):
    return np.array([float(x) for x in s.replace(',', ' ').split()], dtype=np.float64)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Calibrate OptiTrack → RealSense camera transform (click-to-calibrate)",
    )
    parser.add_argument("--serial", type=str, required=True,
                        help="RealSense serial number (the overhead camera)")
    parser.add_argument("--num_points", type=int, default=6,
                        help="Number of calibration points (≥4 required, ≥6 recommended)")
    parser.add_argument("--mocap_scale", type=float, default=1000.0,
                        help="OptiTrack input → mm (default 1000, i.e. input in metres)")
    parser.add_argument("--output", type=str, default="calibration/T_mocap_to_cam.json",
                        help="Output JSON path")
    parser.add_argument("--depth_patch", type=int, default=7,
                        help="Patch size for median depth reading (default 7)")
    args = parser.parse_args()

    if args.num_points < 4:
        print("⚠ Need at least 4 points. Setting num_points=4.")
        args.num_points = 4

    # ── Start RealSense ──────────────────────────────────────────────────────
    print(f"Starting RealSense (serial={args.serial}) ...")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(args.serial)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intrinsics = color_profile.get_intrinsics()
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    depth_to_mm = depth_scale * 1000.0  # raw_uint16 × depth_to_mm = mm

    print(f"  Intrinsics: fx={intrinsics.fx:.1f}  fy={intrinsics.fy:.1f}  "
          f"cx={intrinsics.ppx:.1f}  cy={intrinsics.ppy:.1f}")
    print(f"  Depth scale: {depth_scale}  (raw × {depth_to_mm:.3f} = mm)")
    print(f"  Collecting {args.num_points} point pairs.\n")

    # Warm-up
    for _ in range(30):
        pipeline.wait_for_frames()

    win = "Calibration — OptiTrack to Camera"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, _on_mouse)

    points_cam = []
    points_mocap = []
    click_pixels = []  # for drawing accepted points on live feed

    global _click_xy

    try:
        for idx in range(args.num_points):
            print(f"── Point {idx + 1}/{args.num_points} ──────────────────────────")
            print(f"   Click on the ball in the camera image ...")

            accepted = False
            while not accepted:
                _click_xy = None

                # ── Live feed until user clicks ──────────────────────────────
                while _click_xy is None:
                    frames = align.process(pipeline.wait_for_frames())
                    color = np.asanyarray(frames.get_color_frame().get_data())
                    depth = np.asanyarray(frames.get_depth_frame().get_data())

                    disp = color.copy()
                    # Draw previously accepted points
                    for i, (px, py) in enumerate(click_pixels):
                        cv2.circle(disp, (px, py), 6, (0, 255, 0), 2)
                        cv2.putText(disp, str(i + 1), (px + 8, py - 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

                    cv2.putText(disp,
                                f"Point {idx+1}/{args.num_points}: click on the ball",
                                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.imshow(win, disp)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("Aborted.")
                        return

                # ── Click received — compute 3D ──────────────────────────────
                cx, cy = _click_xy
                raw_d = get_depth_at(depth, cx, cy, args.depth_patch)
                depth_mm = raw_d * depth_to_mm

                if depth_mm <= 0:
                    print(f"   ⚠ No valid depth at ({cx}, {cy}). Try again.")
                    continue

                p_cam = deproject(intrinsics, cx, cy, depth_mm)

                # Show frozen frame with annotation
                disp = color.copy()
                for i, (px, py) in enumerate(click_pixels):
                    cv2.circle(disp, (px, py), 6, (0, 255, 0), 2)
                    cv2.putText(disp, str(i + 1), (px + 8, py - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                cv2.circle(disp, (cx, cy), 8, (0, 0, 255), 2)
                cv2.putText(disp,
                            f"[{p_cam[0]:.1f}, {p_cam[1]:.1f}, {p_cam[2]:.1f}] mm",
                            (cx + 12, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
                cv2.putText(disp,
                            f"depth={depth_mm:.1f}mm  pixel=({cx},{cy})",
                            (10, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
                cv2.putText(disp,
                            "[y] accept   [r] redo   [q] quit",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
                cv2.imshow(win, disp)

                print(f"   Pixel:  ({cx}, {cy})   Depth: {depth_mm:.1f} mm")
                print(f"   P_cam:  [{p_cam[0]:.2f}, {p_cam[1]:.2f}, {p_cam[2]:.2f}] mm")

                # Wait for accept / redo
                while True:
                    key = cv2.waitKey(50) & 0xFF
                    if key == ord('y'):
                        accepted = True
                        break
                    elif key == ord('r'):
                        print("   Redo — click again.\n")
                        break
                    elif key == ord('q'):
                        print("Aborted.")
                        return

            # ── Accepted — ask for OptiTrack position ────────────────────────
            while True:
                try:
                    line = input(f"   OptiTrack position (x y z): ")
                    p_mocap = parse_vector(line)[:3] * args.mocap_scale  # → mm
                    break
                except (ValueError, IndexError) as e:
                    print(f"   ⚠ Parse error: {e}. Try again.")

            points_cam.append(p_cam)
            points_mocap.append(p_mocap)
            click_pixels.append((cx, cy))

            print(f"   P_mocap: [{p_mocap[0]:.2f}, {p_mocap[1]:.2f}, {p_mocap[2]:.2f}] mm")
            print(f"   ✅ Point {idx + 1} recorded.\n")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    if len(points_cam) < 4:
        print(f"Only {len(points_cam)} points collected — need at least 4. Exiting.")
        return

    # ── Compute rigid transform ──────────────────────────────────────────────
    P_mocap = np.array(points_mocap)  # (N, 3) mm
    P_cam = np.array(points_cam)      # (N, 3) mm

    T = rigid_transform_svd(P_mocap, P_cam)
    R = T[:3, :3]
    t = T[:3, 3]

    # ── Evaluate ─────────────────────────────────────────────────────────────
    residuals = []
    for i in range(len(P_mocap)):
        pred = (T @ np.append(P_mocap[i], 1.0))[:3]
        residuals.append(np.linalg.norm(pred - P_cam[i]))
    rmse = np.sqrt(np.mean(np.array(residuals) ** 2))
    euler = Rotation.from_matrix(R).as_euler('xyz', degrees=True)

    print(f"\n{'='*60}")
    print(f"  T_mocap_to_cam  (all units: mm)")
    print(f"{'='*60}")
    for row in T:
        print(f"    [{row[0]:+10.6f} {row[1]:+10.6f} {row[2]:+10.6f} {row[3]:+10.4f}]")
    print(f"\n  Translation (mm):  [{t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f}]")
    print(f"  Rotation (XYZ°):   [{euler[0]:.2f}, {euler[1]:.2f}, {euler[2]:.2f}]")
    print(f"  det(R): {np.linalg.det(R):.6f}")
    print(f"\n  Per-point residuals:")
    for i, r in enumerate(residuals):
        status = "✅" if r < 5.0 else ("⚠️" if r < 10.0 else "❌")
        print(f"    Point {i+1}: {r:.3f} mm  {status}")
    print(f"\n  RMSE: {rmse:.3f} mm")

    # ── Save ─────────────────────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "description": "T_mocap_to_cam: maps OptiTrack (mm) -> camera frame (mm)",
        "T_mocap_to_cam": T.tolist(),
        "camera_serial": args.serial,
        "intrinsics": {
            "fx": intrinsics.fx, "fy": intrinsics.fy,
            "ppx": intrinsics.ppx, "ppy": intrinsics.ppy,
            "width": intrinsics.width, "height": intrinsics.height,
        },
        "mocap_scale": args.mocap_scale,
        "num_points": len(points_cam),
        "rmse_mm": float(rmse),
        "residuals_mm": [float(r) for r in residuals],
        "points_cam_mm": P_cam.tolist(),
        "points_mocap_mm": P_mocap.tolist(),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    with open(output_path, 'w') as fp:
        json.dump(result, fp, indent=2)

    npy_path = output_path.with_suffix('.npy')
    np.save(npy_path, T)

    print(f"\n✅ Saved:")
    print(f"   JSON: {output_path}")
    print(f"   NPY:  {npy_path}")
    print(f"\n💡 To use in visualizer, the full chain is:")
    print(f"   P_gel_rigid  = gel_center_in_rigid_mm              (from gelsight calib)")
    print(f"   P_gel_mocap  = T_rigid_to_mocap @ [P_gel_rigid; 1] (OptiTrack live pose × 1000)")
    print(f"   P_gel_cam    = T_mocap_to_cam   @ [P_gel_mocap; 1] (this calib)")
    print(f"   (u, v)       = project(K, P_gel_cam)               (intrinsics)")


if __name__ == "__main__":
    main()