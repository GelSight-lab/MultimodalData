"""Recover an HDF5 recording whose metadata was never flushed.

pushT/2026-06-18/episode_004.h5 is 79 GB that no HDF5 tool will open:

    OSError: Unable to synchronously open file (bad object header version number)

The recorder died without closing the file. What that costs is visible in the
superblock: `EOF` still reads 2048 (its value at creation) against 79 GB on
disk, the consistency flag still says "open for write", and the root group's
object header at offset 96 is 24 bytes of zeros — version 0, hence the error.
The library stops there and never reaches the data.

The data is all present. HDF5 writes raw chunks straight through and keeps
structure in a metadata cache, so a kill loses the cache, not the recording.
Recovery is therefore not carving: enough of the structure was evicted to disk
during the 19-minute write to rebuild an exact chunk index.

WHAT IS READ OFF THE DISK, AND WHAT IS INFERRED

Read, not inferred:
  * 2204 v1 B-tree nodes with a TREE signature. Parsed at rank 4 and rank 3 and
    scored on whether chunk sizes, trailing chunk dimensions and child
    addresses are plausible; every one resolves unambiguously.
  * The leaf nodes' left/right SIBLING pointers chain them into exactly 8
    components of 271-272 leaves each — the 8 image datasets, each covering
    frames 0..15503 or 0..15446 with no gaps. The split is structural; no
    heuristic decides which chunk belongs to which dataset.
  * Each leaf key carries the chunk's byte offset, stored size and FILTER MASK.

Inferred, and each inference is checked against a file where the answer is
known (episodes 000-003 from the same session and rig):
  * Which chain is which stream, by correlating decoded frames against the
    healthy episodes. All 8 assignments are each other's mutual best match.
    The two weak margins were closed with independent evidence:
      - depth cameras: within every one of the 15,447 complete frames, chains
        (0,1), (2,3) and (5,6) are written adjacently — 100% of frames. That
        pairs each depth with its colour structurally, which resolves the
        0.758-vs-0.719 correlation between cam0 and cam2 depth.
      - GelSight left vs right: single frames separate 0.995 vs 0.908, which is
        not enough. Averaging 41 frames cancels contact and leaves the sensor's
        own fixed pattern; against all three reference episodes that reads
        0.9996-0.9998 for the match and 0.928-0.931 for the alternative.

A rule that was measured and rejected: "a chunk stored at exactly 921,600 bytes
is unfiltered". Mostly true — 10,635 of chain 4's chunks are raw at that size —
but 37 chunks are blosc output that happens to land on exactly that length. The
filter mask in the B-tree key is authoritative and is what this script uses.
Inferring it from the size would have decoded those 37 frames as noise.

Also rejected, and worth recording because it looked convincing: the eight
streams appear to be written in a rigid 8-cycle. The first 60 chunks of
episode_003 show one exactly. Over whole files the cycle does not hold in any
of the four healthy episodes. It is not needed — the B-tree gives the frame
index directly — but it would have silently misassigned streams.

    python scripts/recover_h5_episode.py index  --src <broken.h5> --out <idx.json>
    python scripts/recover_h5_episode.py write  --src <broken.h5> --idx <idx.json> \
        --ref <healthy.h5> --out <recovered.h5>
    python scripts/recover_h5_episode.py verify --src <broken.h5> --idx <idx.json> \
        --out <recovered.h5> [--samples 200]
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np

UNDEF = 0xFFFFFFFFFFFFFFFF
SCAN_CHUNK = 64 << 20

# The eight image datasets, in the chain order this script resolves them to.
# Names come from the reference file; nothing here assumes an ordering.
IMAGE_STREAMS = [
    "realsense/cam0/color", "realsense/cam0/depth",
    "realsense/cam1/color", "realsense/cam1/depth",
    "gelsight/left/frames",
    "realsense/cam2/color", "realsense/cam2/depth",
    "gelsight/right/frames",
]


# ---------------------------------------------------------------- scanning

def scan_tree_nodes(path: Path) -> list[int]:
    """Byte offsets of every ``TREE`` signature in the file."""
    out, scanned, prev = [], 0, b""
    with open(path, "rb") as f:
        while True:
            b = f.read(SCAN_CHUNK)
            if not b:
                break
            buf = prev + b
            base = scanned - len(prev)
            start = 0
            while True:
                i = buf.find(b"TREE", start)
                if i < 0:
                    break
                out.append(base + i)
                start = i + 1
            prev = buf[-8:]
            scanned += len(b)
            print(f"  scanned {scanned/1e9:6.1f} GB  TREE hits {len(out)}",
                  flush=True)
    return out


def parse_btree(f, off: int, ndim: int, filesize: int) -> dict | None:
    """Parse a v1 B-tree node assuming a dataset of rank ``ndim``.

    Returns None if the bytes cannot be a node at that rank. The score says
    how well they fit, so the caller can pick between candidate ranks rather
    than assume one.
    """
    key_size = 8 + 8 * (ndim + 1)
    f.seek(off)
    hdr = f.read(24)
    if len(hdr) < 24 or hdr[:4] != b"TREE":
        return None
    node_type, level = hdr[4], hdr[5]
    nused = struct.unpack("<H", hdr[6:8])[0]
    left, right = struct.unpack("<QQ", hdr[8:24])
    if node_type != 1 or not 0 < nused <= 64:
        return None
    need = nused * (key_size + 8) + key_size
    f.seek(off + 24)
    body = f.read(need)
    if len(body) < need:
        return None

    keys, kids, p = [], [], 0
    for _ in range(nused):
        size, mask = struct.unpack("<II", body[p:p + 8])
        offs = struct.unpack(f"<{ndim + 1}Q", body[p + 8:p + 8 + 8 * (ndim + 1)])
        keys.append((size, mask, offs))
        p += key_size
        kids.append(struct.unpack("<Q", body[p:p + 8])[0])
        p += 8
    size, mask = struct.unpack("<II", body[p:p + 8])
    offs = struct.unpack(f"<{ndim + 1}Q", body[p + 8:p + 8 + 8 * (ndim + 1)])
    keys.append((size, mask, offs))

    score = sum(1 for k in keys if 1000 < k[0] < 2_000_000)
    score += sum(1 for k in keys if all(x == 0 for x in k[2][1:]))
    score += sum(1 for c in kids if 0 < c < filesize)
    frames = [k[2][0] for k in keys]
    if all(b > a for a, b in zip(frames, frames[1:])):
        score += 10
    return {"off": off, "level": level, "nused": nused, "left": left,
            "right": right, "keys": keys, "kids": kids, "score": score,
            "ndim": ndim}


def build_index(src: Path) -> dict:
    """Chain the B-tree leaves into per-dataset chunk indices."""
    filesize = os.path.getsize(src)
    print(f"[recover] scanning {src} ({filesize/1e9:.1f} GB) for B-tree nodes",
          flush=True)
    offsets = scan_tree_nodes(src)

    parsed = {}
    with open(src, "rb") as f:
        for off in offsets:
            best = None
            for ndim in (4, 3):
                r = parse_btree(f, off, ndim, filesize)
                if r and (best is None or r["score"] > best["score"]):
                    best = r
            if best and best["score"] >= 2 * best["nused"]:
                parsed[off] = best
    leaves = {o: n for o, n in parsed.items() if n["level"] == 0}
    print(f"[recover] {len(parsed)} B-tree nodes, {len(leaves)} leaves",
          flush=True)

    adj = collections.defaultdict(set)
    for o, n in leaves.items():
        for s in (n["left"], n["right"]):
            if s != UNDEF and s in leaves:
                adj[o].add(s)
                adj[s].add(o)
    seen, chains = set(), []
    for o in leaves:
        if o in seen:
            continue
        stack, comp = [o], []
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            comp.append(x)
            stack.extend(adj[x] - seen)
        chains.append(sorted(comp))
    chains.sort(key=len, reverse=True)
    if len(chains) != 8:
        raise SystemExit(f"expected 8 sibling chains, got {len(chains)} — "
                         f"the recording's structure is not what this script "
                         f"was validated against; stop and look at it")

    index = {}
    for ci, comp in enumerate(chains):
        entries, ndim = {}, None
        for off in comp:
            node = leaves[off]
            ndim = node["ndim"]
            for key, kid in zip(node["keys"], node["kids"]):
                size, mask, offs = key
                if 0 < kid < filesize:
                    entries[offs[0]] = (kid, size, mask)
        index[ci] = {"ndim": ndim, "n": len(entries),
                     "entries": {str(k): v for k, v in entries.items()}}
        frames = sorted(entries)
        print(f"  chain{ci}: rank={ndim} chunks={len(entries)} "
              f"frames {frames[0]}..{frames[-1]}", flush=True)
    return {"src": str(src), "filesize": filesize, "chains": index}


# ---------------------------------------------------------------- writing

def read_chunk(f, entry) -> tuple[bytes, int]:
    off, size, mask = entry
    f.seek(off)
    return f.read(size), mask


def write_recovered(src: Path, idx: dict, ref: Path, out: Path,
                    assign: list[str]) -> None:
    import h5py
    import hdf5plugin

    chains = idx["chains"]
    n_frames = min(v["n"] for v in chains.values())
    print(f"[recover] {n_frames} complete frames across all 8 streams",
          flush=True)

    with h5py.File(ref, "r") as r:
        spec = {}
        for name in IMAGE_STREAMS:
            d = r[name]
            spec[name] = {"dtype": d.dtype, "chunks": d.chunks,
                          "shape": (n_frames,) + d.shape[1:]}
        meta_attrs = {k: r["metadata"].attrs[k] for k in r["metadata"].attrs}

    with h5py.File(out, "w") as w:
        g = w.create_group("metadata")
        for k, v in meta_attrs.items():
            g.attrs[k] = v
        g.attrs["recovered_from"] = str(src)
        g.attrs["recovery_note"] = (
            "metadata rebuilt from on-disk B-tree leaves; the recorder never "
            "closed this file. Frame count is the largest index present in "
            "every stream. Timestamps and OptiTrack poses were NOT recovered "
            "— their B-trees were never flushed.")

        dsets = {}
        for ci, name in enumerate(assign):
            s = spec[name]
            typesize = s["dtype"].itemsize
            dsets[ci] = w.create_dataset(
                name, shape=s["shape"], dtype=s["dtype"], chunks=s["chunks"],
                **hdf5plugin.Blosc(cname="lz4", clevel=5,
                                   shuffle=hdf5plugin.Blosc.SHUFFLE))
            print(f"  {name}: {s['shape']} {s['dtype']} chunks={s['chunks']} "
                  f"(itemsize {typesize})", flush=True)

        with open(src, "rb") as f:
            for ci, name in enumerate(assign):
                d = dsets[ci]
                ent = chains[ci]["entries"]
                rank = len(d.shape)
                for fr in range(n_frames):
                    raw, mask = read_chunk(f, ent[str(fr)])
                    d.id.write_direct_chunk((fr,) + (0,) * (rank - 1), raw,
                                            filter_mask=mask)
                    if fr % 2000 == 0:
                        print(f"    {name} {fr}/{n_frames}", flush=True)
    print(f"[recover] wrote {out} ({os.path.getsize(out)/1e9:.1f} GB)")


# ---------------------------------------------------------------- verify

def verify(src: Path, idx: dict, out: Path, assign: list[str],
           samples: int) -> int:
    """Every sampled frame must equal the bytes carved from the broken file.

    The recovered file being readable proves nothing about whether it holds
    the right pixels, so this decodes each sample independently — straight
    from the original at the offset the B-tree gave — and compares arrays.
    """
    import h5py
    import hdf5plugin  # noqa: F401  (registers the blosc filter)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from blosc_raw import decompress

    chains = idx["chains"]
    rng = np.random.default_rng(0)
    problems = []
    with h5py.File(out, "r") as w, open(src, "rb") as f:
        n = w[assign[0]].shape[0]
        picks = sorted(rng.choice(n, size=min(samples, n), replace=False))
        for ci, name in enumerate(assign):
            d = w[name]
            shape = d.shape[1:]
            for fr in picks:
                off, size, mask = chains[ci]["entries"][str(fr)]
                f.seek(off)
                raw = f.read(size)
                if mask & 1:
                    buf = raw
                else:
                    buf, _ = decompress(raw)
                want = np.frombuffer(buf, d.dtype).reshape(shape)
                got = np.asarray(d[fr])
                if not np.array_equal(want, got):
                    problems.append(f"{name} frame {fr}: recovered file "
                                    f"disagrees with the source bytes")
        print(f"[verify] {len(picks)} frames x {len(assign)} streams = "
              f"{len(picks)*len(assign)} comparisons", flush=True)
    for p in problems[:20]:
        print("  " + p)
    print(f"[verify] {len(problems)} problem(s)")
    return 1 if problems else 0


# ---------------------------------------------------------------- cli

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["index", "write", "verify"])
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--idx", type=Path)
    ap.add_argument("--ref", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--assign", type=Path,
                    help="JSON list mapping chain i -> dataset name")
    ap.add_argument("--samples", type=int, default=200)
    args = ap.parse_args()

    if args.mode == "index":
        idx = build_index(args.src)
        args.out.write_text(json.dumps(idx))
        print(f"[recover] wrote {args.out}")
        return 0

    idx = json.loads(args.idx.read_text())
    idx["chains"] = {int(k): v for k, v in idx["chains"].items()}
    assign = json.loads(args.assign.read_text())

    if args.mode == "write":
        write_recovered(args.src, idx, args.ref, args.out, assign)
        return 0
    return verify(args.src, idx, args.out, assign, args.samples)


if __name__ == "__main__":
    raise SystemExit(main())
