#!/usr/bin/env python3
"""
Benchmark HDF5 write speed: no compression vs BLOSC/LZ4.

Simulates the exact data layout used in data_collection.py at a given fps.
"""

import time
import tempfile
import os
import numpy as np
import h5py

try:
    import hdf5plugin
    HAS_BLOSC = True
except ImportError:
    HAS_BLOSC = False
    print("hdf5plugin not installed — only testing uncompressed.\n")

N_FRAMES   = 60   # frames to write per test
N_CAMS     = 3
H, W       = 480, 640


def make_fake_frame():
    color  = [np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)  for _ in range(N_CAMS)]
    depth  = [np.random.randint(0, 5000, (H, W),   dtype=np.uint16) for _ in range(N_CAMS)]
    gs     = [np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)  for _ in range(2)]
    return color, depth, gs


def create_file(f, compression_kwargs):
    for i in range(N_CAMS):
        g = f.create_group(f"realsense/cam{i}")
        g.create_dataset("color", shape=(0, H, W, 3), maxshape=(None, H, W, 3),
                         dtype=np.uint8,  chunks=(1, H, W, 3), **compression_kwargs)
        g.create_dataset("depth", shape=(0, H, W),    maxshape=(None, H, W),
                         dtype=np.uint16, chunks=(1, H, W),    **compression_kwargs)
    for name in ["left", "right"]:
        g = f.create_group(f"gelsight/{name}")
        g.create_dataset("frames", shape=(0, H, W, 3), maxshape=(None, H, W, 3),
                         dtype=np.uint8, chunks=(1, H, W, 3), **compression_kwargs)


def write_frames(f, frames_list):
    for n, (color, depth, gs) in enumerate(frames_list):
        for i in range(N_CAMS):
            ds_c = f[f"realsense/cam{i}/color"]
            ds_d = f[f"realsense/cam{i}/depth"]
            ds_c.resize(n + 1, axis=0); ds_c[n] = color[i]
            ds_d.resize(n + 1, axis=0); ds_d[n] = depth[i]
        for j, name in enumerate(["left", "right"]):
            ds = f[f"gelsight/{name}/frames"]
            ds.resize(n + 1, axis=0); ds[n] = gs[j]


def run_benchmark(label, compression_kwargs):
    frames = [make_fake_frame() for _ in range(N_FRAMES)]

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        path = tmp.name

    try:
        with h5py.File(path, "w") as f:
            create_file(f, compression_kwargs)
            t0 = time.perf_counter()
            write_frames(f, frames)
            elapsed = time.perf_counter() - t0

        size_mb = os.path.getsize(path) / 1e6
        raw_mb  = N_FRAMES * (N_CAMS * H * W * 3 + N_CAMS * H * W * 2 + 2 * H * W * 3) / 1e6
        fps     = N_FRAMES / elapsed

        print(f"{label}")
        print(f"  Time      : {elapsed:.2f}s for {N_FRAMES} frames")
        print(f"  Throughput: {fps:.1f} fps  ({raw_mb/elapsed:.0f} MB/s raw in)")
        print(f"  File size : {size_mb:.1f} MB  (raw: {raw_mb:.1f} MB, ratio: {raw_mb/size_mb:.1f}x)")
        print()
    finally:
        os.remove(path)


if __name__ == "__main__":
    print(f"Writing {N_FRAMES} frames × {N_CAMS} RealSense + 2 GelSight at {H}×{W}\n")

    run_benchmark("No compression", {})

    if HAS_BLOSC:
        run_benchmark("BLOSC LZ4 (shuffle=BYTE)", hdf5plugin.Blosc(
            cname="lz4", clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE))
        run_benchmark("BLOSC LZ4 (shuffle=BIT)", hdf5plugin.Blosc(
            cname="lz4", clevel=5, shuffle=hdf5plugin.Blosc.BITSHUFFLE))
        run_benchmark("BLOSC LZ4HC clevel=4", hdf5plugin.Blosc(
            cname="lz4hc", clevel=4, shuffle=hdf5plugin.Blosc.BITSHUFFLE))
