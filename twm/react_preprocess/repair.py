"""Diagnose, and where it is safe to, repair a recording that will not open.

    OSError: Unable to synchronously open file (bad object header version number)

means the recorder died without ``close()``. HDF5 streams raw chunks straight
to disk and keeps object headers, chunk B-trees and the superblock's EOF field
in a metadata cache, so a kill loses the cache and not the recording: every
pixel is on the disk with nothing to reach it by.

pushT/2026-06-18/episode_004.h5 is 79 GB in that state. Recovering it took a
day. This module is that day, made automatic — with one deliberate limit.

WHAT AUTOMATIC REPAIR MUST NOT DO

Recovery gets back what HDF5 had already evicted. For episode_004 that was all
eight image streams (15,447 complete frames) but only 2 of 16 timestamp chunks
and no usable OptiTrack poses. Without timestamps there is no cross-modal
alignment, and React is a tactile-and-pose dataset — a pose-less episode
published as an episode is a lie about the data.

So the repair path deliberately ends in a REFUSAL more often than in a build.
``release_eligibility`` is not a formality: a recovered file must prove it has
complete timestamps and poses before anything downstream may touch it, and the
default answer is no. Silently building whatever survived is the failure mode
this module exists to prevent, not the feature it provides.

Interpolating the missing timestamps is the obvious escape and it is measurably
wrong: fitting a line through the two anchors that survived in episode_004,
checked against episodes whose timestamps are known, misplaces frames by 15.6,
24.5 and 1431. The pipeline's tactile lag is 15 frames.

PREVENTION IS THE REAL FIX

``data_collection.HDF5Writer.FLUSH_INTERVAL_S`` (10 s) means a kill now costs
seconds instead of everything, and ``scripts/test_crash_leaves_readable_h5.py``
proves it by killing a writer with and without the flush. Recovery is for
files recorded before that existed.
"""
from __future__ import annotations

import collections
import json
import os
import shutil
import struct
from dataclasses import dataclass, field
from pathlib import Path

from .h5raw import UNDEF, H5Raw, blosc_decompress

# The recorder lays the same skeleton in every file, so the metadata group's
# object header is at the same address in all of them. Verified identical
# across all five episodes of 2026-06-18.
METADATA_OH_ADDR = 800

SCAN_BLOCK = 64 << 20

# Chain order the sibling-pointer split resolves to, for this recorder's
# eight image datasets. Identity is verified against a reference file at
# recovery time (see verify_stream_identity); this is only the starting guess.
IMAGE_STREAMS = [
    "realsense/cam0/color", "realsense/cam0/depth",
    "realsense/cam1/color", "realsense/cam1/depth",
    "gelsight/left/frames",
    "realsense/cam2/color", "realsense/cam2/depth",
    "gelsight/right/frames",
]

# Datasets an episode needs before the pipeline may build it.
REQUIRED_FOR_RELEASE = ("timestamps",)

METADATA_NEVER_FLUSHED = "metadata-never-flushed"
UNKNOWN_DAMAGE = "unknown-damage"
HEALTHY = "healthy"


@dataclass
class Diagnosis:
    signature: str
    repairable: bool
    detail: str
    superblock: dict = field(default_factory=dict)

    def __str__(self):
        return f"{self.signature}: {self.detail}"


def diagnose(h5_path) -> Diagnosis:
    """Classify why a file will not open, from its superblock.

    The signature this module repairs is specific and checkable, so it is
    checked rather than assumed: the EOF field still holding a value far below
    the file's real size (it is written at close), the consistency flag still
    reading "open for write", and a root object header whose version byte is
    not 1. Anything else is reported and left alone — guessing at unfamiliar
    damage is how a recovery invents data.
    """
    h5_path = Path(h5_path)
    try:
        raw = H5Raw(h5_path)
    except Exception as exc:                                    # noqa: BLE001
        return Diagnosis(UNKNOWN_DAMAGE, False, f"not readable as HDF5: {exc}")
    with raw:
        sb = raw.superblock
        oh = raw.object_header(sb["root_oh_address"])
        eof_short = sb["eof_address"] < sb["file_size"]
        open_flag = bool(sb["consistency_flags"] & 1)
        bad_root = oh.get("version") != 1

        if not bad_root and not eof_short:
            return Diagnosis(HEALTHY, False, "superblock and root header look "
                             "intact; the failure is elsewhere", sb)
        if bad_root and eof_short and open_flag:
            return Diagnosis(
                METADATA_NEVER_FLUSHED, True,
                f"recorder never closed the file — EOF field reads "
                f"{sb['eof_address']:,} against {sb['file_size']:,} bytes on "
                f"disk, consistency flag says open-for-write, root object "
                f"header version is {oh.get('version')}. Raw chunks are "
                f"intact; the metadata cache was lost", sb)
        return Diagnosis(
            UNKNOWN_DAMAGE, False,
            f"does not match the recoverable signature "
            f"(root header version {oh.get('version')}, EOF "
            f"{sb['eof_address']:,} vs {sb['file_size']:,} on disk, "
            f"open-for-write {open_flag}) — do not guess, look at it", sb)


# ─────────────────────────────────────────────────── chunk index rebuild

def _parse_chunk_btree(f, off: int, ndim: int, filesize: int):
    """Parse a chunked-data B-tree node assuming a dataset of rank ``ndim``.

    Returns a score alongside the parse so the caller can choose between
    candidate ranks on evidence rather than assume one.
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
    keys.append((size, mask,
                 struct.unpack(f"<{ndim + 1}Q",
                               body[p + 8:p + 8 + 8 * (ndim + 1)])))

    score = sum(1 for k in keys if 1000 < k[0] < 2_000_000)
    score += sum(1 for k in keys if all(x == 0 for x in k[2][1:]))
    score += sum(1 for c in kids if 0 < c < filesize)
    frames = [k[2][0] for k in keys]
    if all(b > a for a, b in zip(frames, frames[1:])):
        score += 10
    return {"off": off, "level": level, "nused": nused, "left": left,
            "right": right, "keys": keys, "kids": kids, "score": score,
            "ndim": ndim}


def scan_tree_signatures(path: Path, progress=None) -> list[int]:
    out, scanned, prev = [], 0, b""
    with open(path, "rb") as f:
        while True:
            b = f.read(SCAN_BLOCK)
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
            if progress:
                progress(scanned, len(out))
    return out


def build_chunk_index(src: Path, progress=None) -> dict:
    """Rebuild the per-dataset chunk index from orphaned B-tree leaves.

    The split into datasets is STRUCTURAL, not heuristic: leaf nodes carry
    left/right sibling pointers, and chaining them yields one connected
    component per dataset. Each leaf key gives a chunk's byte offset, stored
    size and filter mask — the mask is authoritative and must be carried
    through, because "stored at exactly the raw size means unfiltered" is
    true for 10,635 chunks of episode_004 and false for 37 more.
    """
    src = Path(src)
    filesize = src.stat().st_size
    offsets = scan_tree_signatures(src, progress)

    parsed = {}
    with open(src, "rb") as f:
        for off in offsets:
            best = None
            for ndim in (4, 3):
                r = _parse_chunk_btree(f, off, ndim, filesize)
                if r and (best is None or r["score"] > best["score"]):
                    best = r
            if best and best["score"] >= 2 * best["nused"]:
                parsed[off] = best
    leaves = {o: n for o, n in parsed.items() if n["level"] == 0}

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
    return {"src": str(src), "filesize": filesize,
            "n_chains": len(chains), "chains": index}


# ───────────────────────────────────────────────────────── stream identity

def _normalised_grey(a):
    import numpy as np
    g = a.astype(np.float32).mean(axis=2) if a.ndim == 3 else a.astype(np.float32)
    g = g[::8, ::8]
    return (g - g.mean()) / (g.std() + 1e-6)


def verify_stream_identity(recovered: Path, reference: Path,
                           streams=IMAGE_STREAMS, frame: int = 5000) -> dict:
    """Does each dataset in the recovered file look like its own name?

    Byte-comparing the recovered file against the index it was written from
    proves the copy is faithful and nothing about whether the index put the
    right chunks under the right names — both sides use the same index. This
    is the independent check: read each dataset BY NAME and correlate it with
    the same-named stream of a file that opens.
    """
    import h5py
    import hdf5plugin  # noqa: F401
    import numpy as np

    names = [n for n in streams]
    with h5py.File(recovered, "r") as r, h5py.File(reference, "r") as g:
        usable = [n for n in names
                  if n in r and n in g
                  and frame < r[n].shape[0] and frame < g[n].shape[0]]
        M = np.zeros((len(usable), len(usable)))
        cache = {n: _normalised_grey(np.asarray(g[n][frame])) for n in usable}
        for i, n in enumerate(usable):
            a = _normalised_grey(np.asarray(r[n][frame]))
            for j, m in enumerate(usable):
                M[i, j] = float((a * cache[m]).mean())
    wrong = [usable[i] for i in range(len(usable)) if M[i].argmax() != i]
    return {"streams": usable, "matrix": M.tolist(), "misassigned": wrong,
            "ok": not wrong and len(usable) == len(names)}


# ─────────────────────────────────────────────────────────── writing

# The one place that knows what a recovered file is called. `discover` must
# skip them (a recovered sibling is reached through its source, never as an
# episode of its own) and `open_episode` must not let the suffix leak into the
# published episode name — `episode_004.recovered` would silently become a
# fifth episode in the release.
RECOVERED_SUFFIX = ".recovered"


def recovered_path(src) -> Path:
    return Path(src).with_suffix(RECOVERED_SUFFIX + ".h5")


def is_recovered(path) -> bool:
    return Path(path).name.endswith(RECOVERED_SUFFIX + ".h5")


def source_stem(path) -> str:
    """The episode name a recording publishes under, recovered or not."""
    stem = Path(path).stem
    return stem[:-len(RECOVERED_SUFFIX)] if stem.endswith(RECOVERED_SUFFIX) else stem


def write_recovered(src: Path, index: dict, reference: Path, out: Path,
                    assign=None, limit: int = 0, log=print) -> Path:
    """Copy the indexed chunks into a clean file, uncompressed-unchanged."""
    import h5py
    import hdf5plugin
    import numpy as np                                          # noqa: F401

    assign = list(assign or IMAGE_STREAMS)
    chains = {int(k): v for k, v in index["chains"].items()}
    if len(chains) != len(assign):
        raise ValueError(
            f"{src.name}: rebuilt {len(chains)} dataset chains but the "
            f"recorder writes {len(assign)}. This recording is not the shape "
            f"this recovery was validated on — look at it rather than let it "
            f"guess")
    n_frames = min(v["n"] for v in chains.values())
    if limit:
        n_frames = min(n_frames, limit)
    log(f"[repair] {n_frames} frames present in every one of "
        f"{len(assign)} streams")

    # SCHEMA from the reference; FACTS about the recording from the source.
    with h5py.File(reference, "r") as r:
        spec = {n: {"dtype": r[n].dtype, "chunks": r[n].chunks,
                    "shape": (n_frames,) + r[n].shape[1:]} for n in assign}
        ref_attrs = set(r["metadata"].attrs) if "metadata" in r else set()

    with H5Raw(src) as raw:
        meta_attrs = raw.group_attributes(METADATA_OH_ADDR)
    missing = sorted(ref_attrs - set(meta_attrs))
    if missing:
        log(f"[repair] WARNING: {missing} unreadable from the source and NOT "
            f"copied from the reference — a metadata field carrying another "
            f"episode's value is worse than an absent one")

    with h5py.File(out, "w") as w:
        g = w.create_group("metadata")
        for k, v in meta_attrs.items():
            g.attrs[k] = v
        g.attrs["recovered_from"] = str(src)
        g.attrs["recovery_note"] = (
            "chunk index rebuilt from orphaned B-tree leaves; the recorder "
            "never closed the source. Frame count is the largest index "
            "present in every stream.")

        dsets = {}
        for ci, name in enumerate(assign):
            s = spec[name]
            dsets[ci] = w.create_dataset(
                name, shape=s["shape"], dtype=s["dtype"], chunks=s["chunks"],
                **hdf5plugin.Blosc(cname="lz4", clevel=5,
                                   shuffle=hdf5plugin.Blosc.SHUFFLE))

        # FRAME-major. One frame's chunks sit within a few MB of each other,
        # while one stream's consecutive chunks are ~4.89 MB apart across the
        # whole file — draining a stream at a time reads 79 GB per stream.
        ranks = {ci: len(dsets[ci].shape) for ci in range(len(assign))}
        ents = {ci: chains[ci]["entries"] for ci in range(len(assign))}
        with open(src, "rb") as f:
            for fr in range(n_frames):
                key = str(fr)
                for entry, ci in sorted((ents[ci][key], ci)
                                        for ci in range(len(assign))):
                    off, size, mask = entry
                    f.seek(off)
                    dsets[ci].id.write_direct_chunk(
                        (fr,) + (0,) * (ranks[ci] - 1), f.read(size),
                        filter_mask=mask)
                if fr % 1000 == 0:
                    log(f"[repair]   frame {fr}/{n_frames} "
                        f"({100 * fr / n_frames:.0f}%)")
    return out


def verify_against_source(src: Path, index: dict, out: Path, assign=None,
                          samples: int = 200) -> list[str]:
    """Recovered frames must equal the bytes decoded from the source.

    This proves copy fidelity only. Stream identity is a separate question and
    is answered by verify_stream_identity, because this check and the write it
    checks share an index — if the index were wrong, both would be wrong in
    the same direction and this would still pass.
    """
    import h5py
    import hdf5plugin  # noqa: F401
    import numpy as np

    assign = list(assign or IMAGE_STREAMS)
    chains = {int(k): v for k, v in index["chains"].items()}
    problems = []
    rng = np.random.default_rng(0)
    with h5py.File(out, "r") as w, open(src, "rb") as f:
        n = w[assign[0]].shape[0]
        picks = sorted(rng.choice(n, size=min(samples, n), replace=False))
        for ci, name in enumerate(assign):
            d = w[name]
            for fr in picks:
                off, size, mask = chains[ci]["entries"][str(fr)]
                f.seek(off)
                buf = f.read(size)
                if not mask & 1:
                    buf, _ = blosc_decompress(buf)
                want = np.frombuffer(buf, d.dtype).reshape(d.shape[1:])
                if not np.array_equal(want, np.asarray(d[fr])):
                    problems.append(f"{name} frame {fr}: recovered file "
                                    f"disagrees with the source bytes")
    return problems


# ────────────────────────────────────────────────── release eligibility

def release_eligibility(path) -> tuple[bool, str]:
    """May the pipeline build a release episode from this file?

    Default no. A recovered recording is video until it proves it also carries
    the timestamps every downstream alignment depends on. episode_004 got back
    all eight image streams and 13% of its timestamps; building it would have
    produced an episode whose tactile-to-camera and pose-to-camera alignment
    was invented.
    """
    import h5py
    import hdf5plugin  # noqa: F401

    path = Path(path)
    if not path.exists():
        return False, "file does not exist"
    try:
        with h5py.File(path, "r") as f:
            attrs = dict(f["metadata"].attrs) if "metadata" in f else {}
            recovered = "recovered_from" in attrs
            missing = [n for n in REQUIRED_FOR_RELEASE if n not in f]
            if missing:
                return False, (
                    f"missing {missing}"
                    + (" — recovery cannot rebuild what was still in the "
                       "metadata cache, and interpolating timestamps "
                       "misplaces frames by 15-1431 (measured)"
                       if recovered else ""))
            n_ts = f["timestamps"].shape[0]
            n_img = f[IMAGE_STREAMS[0]].shape[0] if IMAGE_STREAMS[0] in f else 0
            if n_img and n_ts < n_img:
                return False, (f"timestamps cover {n_ts} of {n_img} frames; "
                               f"the rest would have to be invented")
            poses = [k for k in ("optitrack/sensor_left/pose",
                                 "optitrack/sensor_right/pose") if k in f]
            if not poses:
                return False, "no OptiTrack poses"
            if not any(f[k].shape[0] for k in poses):
                return False, "OptiTrack pose datasets are empty"
    except Exception as exc:                                    # noqa: BLE001
        return False, f"unreadable: {exc}"
    return True, "complete"


# ──────────────────────────────────────────────────────── the entry point

def free_bytes(path) -> int:
    return shutil.disk_usage(Path(path).parent).free


def ensure_readable(h5_path, reference=None, auto: bool = True,
                    log=print) -> tuple[Path, str]:
    """Give the pipeline a readable path, or say precisely why it cannot.

    Returns ``(path, note)``. ``path`` is the original when it opens, the
    recovered sibling when repair succeeded, and None when it did not.

    Repair is attempted only for the one diagnosed, validated signature, and
    only when a reference recording from the same rig exists to take the
    schema from. It never overwrites the source.
    """
    import h5py
    import hdf5plugin  # noqa: F401

    h5_path = Path(h5_path)
    try:
        with h5py.File(str(h5_path), "r"):
            return h5_path, "opened"
    except Exception as exc:                                    # noqa: BLE001
        first = str(exc).splitlines()[0]

    out = recovered_path(h5_path)
    if out.exists():
        try:
            with h5py.File(str(out), "r"):
                return out, f"using existing recovery ({out.name})"
        except Exception:                                       # noqa: BLE001
            log(f"[repair] {out.name} exists but does not open; ignoring it")

    d = diagnose(h5_path)
    log(f"[repair] {h5_path.name} will not open: {first}")
    log(f"[repair] diagnosis — {d}")
    if not d.repairable:
        return None, f"unrepairable ({d.signature})"
    if not auto:
        return None, f"repairable ({d.signature}) but --no-repair was given"

    reference = Path(reference) if reference else _pick_reference(h5_path)
    if reference is None:
        return None, ("repairable, but no healthy sibling recording was found "
                      "to take the dataset schema from")

    need = int(h5_path.stat().st_size * 1.05)
    if free_bytes(out) < need:
        return None, (f"repairable, but recovery needs ~{need/1e9:.0f} GB and "
                      f"{free_bytes(out)/1e9:.0f} GB is free")

    log(f"[repair] recovering with schema from {reference.name}")
    index = build_chunk_index(h5_path)
    log(f"[repair] rebuilt {index['n_chains']} dataset chains from "
        f"orphaned B-tree leaves")
    write_recovered(h5_path, index, reference, out, log=log)

    problems = verify_against_source(h5_path, index, out, samples=50)
    if problems:
        out.unlink(missing_ok=True)
        return None, f"recovery failed verification: {problems[0]}"
    ident = verify_stream_identity(out, reference)
    if not ident["ok"]:
        out.unlink(missing_ok=True)
        return None, (f"recovered streams do not match their names: "
                      f"{ident['misassigned']}")
    log(f"[repair] verified: bytes match the source, and all "
        f"{len(ident['streams'])} streams correlate with their own name")
    return out, f"recovered to {out.name}"


def _pick_reference(h5_path: Path) -> Path | None:
    """The largest sibling recording that opens — most likely to be complete."""
    import h5py
    import hdf5plugin  # noqa: F401

    cands = sorted((p for p in h5_path.parent.glob("episode_*.h5")
                    if p != h5_path and not is_recovered(p)),
                   key=lambda p: p.stat().st_size, reverse=True)
    for p in cands:
        try:
            with h5py.File(str(p), "r") as f:
                if all(n in f for n in IMAGE_STREAMS):
                    return p
        except Exception:                                       # noqa: BLE001
            continue
    return None


def save_index(index: dict, path) -> None:
    Path(path).write_text(json.dumps(index))


def load_index(path) -> dict:
    return json.loads(Path(path).read_text())
