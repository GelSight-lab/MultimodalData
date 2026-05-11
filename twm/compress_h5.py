#!/usr/bin/env python3
"""
Compress HDF5 episode files in-place using GZIP+shuffle.

Usage:
    python compress_h5.py episode_001.h5 [episode_002.h5 ...]
    python compress_h5.py data/motherboard/2026-03-22/
"""

import argparse
import os
import sys
import hdf5plugin  # registers BLOSC/etc. plugin path so compressed sources read correctly
import h5py
import numpy as np


def compress_file(src_path, gzip_level=4):
    tmp_path = src_path + ".compress_tmp"

    src_size = os.path.getsize(src_path)
    print(f"\n{src_path}  ({src_size / 1e9:.2f} GB)")

    try:
        with h5py.File(src_path, "r") as src, h5py.File(tmp_path, "w") as dst:
            # Copy attributes on root
            for k, v in src.attrs.items():
                dst.attrs[k] = v

            def copy_item(name, obj):
                if isinstance(obj, h5py.Dataset):
                    chunks = obj.chunks
                    if chunks is None and obj.ndim > 0:
                        # default chunk: one "row"
                        chunks = (1,) + obj.shape[1:]

                    if chunks is not None and obj.shape:
                        chunks = tuple(min(c, max(s, 1)) for c, s in zip(chunks, obj.shape))

                    kwargs = dict(
                        data=obj[()],
                        dtype=obj.dtype,
                        chunks=chunks,
                    )
                    if chunks is not None:
                        kwargs["compression"] = "gzip"
                        kwargs["compression_opts"] = gzip_level
                        kwargs["shuffle"] = True

                    dst.create_dataset(name, **kwargs)
                    for k, v in obj.attrs.items():
                        dst[name].attrs[k] = v

                elif isinstance(obj, h5py.Group):
                    grp = dst.require_group(name)
                    for k, v in obj.attrs.items():
                        grp.attrs[k] = v

            src.visititems(copy_item)

        dst_size = os.path.getsize(tmp_path)
        ratio = src_size / dst_size if dst_size > 0 else 0
        print(f"  {src_size / 1e9:.2f} GB -> {dst_size / 1e9:.2f} GB  ({ratio:.1f}x)")

        os.replace(tmp_path, src_path)
        print(f"  Done.")

    except Exception as e:
        print(f"  ERROR: {e}", file=sys.stderr)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def main():
    parser = argparse.ArgumentParser(description="Compress HDF5 episode files in-place.")
    parser.add_argument("paths", nargs="+", help="Files or directories to compress")
    parser.add_argument("--level", type=int, default=4, help="GZIP level 1-9 (default: 4)")
    args = parser.parse_args()

    h5_files = []
    for p in args.paths:
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                for f in sorted(files):
                    if f.endswith(".h5"):
                        h5_files.append(os.path.join(root, f))
        elif os.path.isfile(p):
            h5_files.append(p)
        else:
            print(f"Warning: {p} not found, skipping.", file=sys.stderr)

    if not h5_files:
        print("No .h5 files found.")
        sys.exit(1)

    print(f"Compressing {len(h5_files)} file(s) with GZIP level {args.level}...")
    for f in h5_files:
        compress_file(f, gzip_level=args.level)

    print("\nAll done.")


if __name__ == "__main__":
    main()
