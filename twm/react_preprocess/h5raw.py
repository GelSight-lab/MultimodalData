"""Read HDF5 v1 structures off the disk, without the library.

This exists because the library refuses a whole file the moment one structure
is malformed, and a recording whose recorder was killed is exactly that: 79 GB
of intact pixels behind a 24-byte object header that was never written. To
argue that such a file is recoverable you have to be able to read what IS on
the disk.

Nothing here writes. ``repair`` is the module that acts on what this finds.

Layouts implemented: superblock v0, object header v1, local heap, symbol table
node, group B-tree (node type 0), chunked-data B-tree (node type 1), global
heap collection, and the blosc chunk header the recorder's filter writes.
"""
from __future__ import annotations

import ctypes
import struct
from pathlib import Path

SIG = b"\x89HDF\r\n\x1a\n"
UNDEF = 0xFFFFFFFFFFFFFFFF

# Message types we care about, by their HDF5 numbers.
MSG_DATASPACE = 0x01
MSG_LAYOUT = 0x08
MSG_ATTRIBUTE = 0x0C
MSG_CONTINUATION = 0x10
MSG_SYMBOL_TABLE = 0x11


# ─────────────────────────────────────────────────────────────── blosc

_BLOSC_LIB = None


def _blosc():
    """The blosc decompressor hdf5plugin ships, reached directly.

    hdf5plugin registers a filter with the HDF5 library; it exposes no Python
    entry point. Recovery needs to decompress a chunk found at a raw byte
    offset, with no dataset to read it through, so the shared object is loaded
    and called by hand. Verified byte-identical against h5py on a file that
    opens normally.
    """
    global _BLOSC_LIB
    if _BLOSC_LIB is None:
        import hdf5plugin
        so = Path(hdf5plugin.__file__).parent / "plugins" / "libh5blosc.so"
        lib = ctypes.CDLL(str(so))
        lib.blosc_decompress.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                         ctypes.c_size_t]
        lib.blosc_decompress.restype = ctypes.c_int
        try:
            lib.blosc_init()
        except AttributeError:
            pass
        _BLOSC_LIB = lib
    return _BLOSC_LIB


def blosc_header(buf: bytes) -> dict:
    """The 16 bytes that make a blosc chunk self-describing."""
    nbytes, blocksize, cbytes = struct.unpack("<III", buf[4:16])
    return {"version": buf[0], "versionlz": buf[1], "flags": buf[2],
            "typesize": buf[3], "nbytes": nbytes, "blocksize": blocksize,
            "cbytes": cbytes}


def blosc_decompress(buf: bytes) -> tuple[bytes, dict]:
    """Decompress one blosc chunk.

    A chunk whose B-tree filter mask has bit 0 set was stored WITHOUT the
    filter and has no header — do not pass it here. Inferring that from the
    stored size instead of the mask is wrong for 37 chunks of the recording
    this was written for, which compress to exactly the raw length.
    """
    h = blosc_header(buf)
    out = ctypes.create_string_buffer(h["nbytes"])
    n = _blosc().blosc_decompress(bytes(buf), out, h["nbytes"])
    if n <= 0:
        raise ValueError(f"blosc_decompress returned {n}")
    return out.raw[:n], h


# ─────────────────────────────────────────────────────────── raw reader

class H5Raw:
    """A file opened as bytes, with just enough HDF5 to walk its structure."""

    def __init__(self, path):
        self.path = Path(path)
        self.f = open(self.path, "rb")
        self.superblock = self._superblock()

    def close(self):
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def at(self, off: int, n: int) -> bytes:
        self.f.seek(off)
        return self.f.read(n)

    def _superblock(self) -> dict:
        b = self.at(0, 96)
        if b[:8] != SIG:
            raise ValueError(f"{self.path}: not an HDF5 file")
        version = b[8]
        if version != 0:
            raise ValueError(f"{self.path}: superblock version {version} "
                             f"is not implemented here")
        base, freespace, eof, driver = struct.unpack("<QQQQ", b[24:56])
        name_off, root_oh = struct.unpack("<QQ", b[56:72])
        return {"version": version, "offset_size": b[13], "length_size": b[14],
                "consistency_flags": struct.unpack("<I", b[20:24])[0],
                "base_address": base, "eof_address": eof,
                "root_name_offset": name_off, "root_oh_address": root_oh,
                "file_size": self.path.stat().st_size}

    # ── object headers ────────────────────────────────────────────────
    def object_header(self, addr: int) -> dict:
        h = self.at(addr, 16)
        version, nmesg = h[0], struct.unpack("<H", h[2:4])[0]
        refcount, hdrsize = struct.unpack("<II", h[4:12])
        out = {"addr": addr, "version": version, "nmesg": nmesg,
               "refcount": refcount, "hdrsize": hdrsize, "messages": []}
        if version != 1:
            out["error"] = (f"object header version {version} at {addr}; "
                            f"v1 expected")
            return out
        body = self.at(addr + 16, hdrsize)
        p = 0
        for _ in range(nmesg):
            if p + 8 > len(body):
                break
            mtype, msize = struct.unpack("<HH", body[p:p + 4])
            out["messages"].append({"type": mtype, "size": msize,
                                    "flags": body[p + 4],
                                    "data": body[p + 8:p + 8 + msize]})
            p += 8 + msize
        return out

    @staticmethod
    def symbol_table_message(data: bytes) -> tuple[int, int]:
        """(group B-tree address, local heap address)."""
        return struct.unpack("<QQ", data[:16])

    # ── heaps and symbol tables ───────────────────────────────────────
    def local_heap(self, addr: int) -> dict:
        b = self.at(addr, 32)
        if b[:4] != b"HEAP":
            raise ValueError(f"no HEAP signature at {addr}")
        size, free_head, data_addr = struct.unpack("<QQQ", b[8:32])
        return {"addr": addr, "version": b[4], "size": size,
                "free_head": free_head, "data_addr": data_addr}

    def heap_string(self, heap: dict, off: int, limit: int = 512) -> str:
        raw = self.at(heap["data_addr"] + off, limit)
        end = raw.find(b"\0")
        return raw[:end if end >= 0 else limit].decode("utf-8", "replace")

    def global_heap_object(self, addr: int, index: int, length: int = 0):
        """One object out of a GCOL collection — where vlen strings live."""
        blk = self.at(addr, 1 << 16)
        if blk[:4] != b"GCOL":
            return None
        p = 16
        while p + 16 <= len(blk):
            idx, _ref = struct.unpack("<HH", blk[p:p + 4])
            size = struct.unpack("<Q", blk[p + 8:p + 16])[0]
            if idx == index:
                return blk[p + 16:p + 16 + (length or size)]
            if size == 0:
                break
            p += 16 + ((size + 7) // 8 * 8)
        return None

    def snod(self, addr: int) -> list[dict]:
        b = self.at(addr, 8)
        if b[:4] != b"SNOD":
            raise ValueError(f"no SNOD signature at {addr}")
        n = struct.unpack("<H", b[6:8])[0]
        ents = self.at(addr + 8, n * 40)
        out = []
        for i in range(n):
            e = ents[i * 40:(i + 1) * 40]
            name_off, oh_addr = struct.unpack("<QQ", e[:16])
            out.append({"name_off": name_off, "oh_addr": oh_addr,
                        "cache": struct.unpack("<I", e[16:20])[0]})
        return out

    # ── group B-tree (node type 0) ────────────────────────────────────
    def group_btree_leaves(self, addr: int, seen=None) -> list[int]:
        seen = seen if seen is not None else set()
        if addr in seen or addr == UNDEF:
            return []
        seen.add(addr)
        b = self.at(addr, 24)
        if b[:4] != b"TREE":
            raise ValueError(f"no TREE signature at {addr}")
        level = b[5]
        nused = struct.unpack("<H", b[6:8])[0]
        body = self.at(addr + 24, (nused * 2 + 1) * 8)
        children = [struct.unpack("<Q", body[(i * 2 + 1) * 8:(i * 2 + 2) * 8])[0]
                    for i in range(nused)]
        if level == 0:
            return children
        out = []
        for c in children:
            out += self.group_btree_leaves(c, seen)
        return out

    def group_children(self, btree_addr: int, heap_addr: int) -> list[dict]:
        heap = self.local_heap(heap_addr)
        out = []
        for snod_addr in self.group_btree_leaves(btree_addr):
            for e in self.snod(snod_addr):
                out.append({"name": self.heap_string(heap, e["name_off"]),
                            "oh_addr": e["oh_addr"], "cache": e["cache"]})
        return out

    # ── attributes ────────────────────────────────────────────────────
    def group_attributes(self, oh_addr: int) -> dict:
        """A group's attributes, following the continuation chain.

        Recovery takes a broken file's SCHEMA from a healthy sibling, and the
        temptation is to take the sibling's metadata with it. That stamped the
        first recovered episode with the reference's `created_at` — 21 minutes
        wrong, and indistinguishable from right. Four of the five fields are
        per-session and identical, which is what makes the fifth easy to miss.
        A broken file's own attributes usually survive; this reads them.
        """
        out: dict = {}

        def walk(addr: int, length: int, depth: int = 0, seen=None) -> None:
            seen = seen if seen is not None else set()
            if addr in seen or depth > 6:
                return
            seen.add(addr)
            blk = self.at(addr, length)
            p = 0
            while p + 8 <= len(blk):
                mtype, msize = struct.unpack("<HH", blk[p:p + 4])
                data = blk[p + 8:p + 8 + msize]
                if mtype == MSG_ATTRIBUTE and len(data) > 8:
                    nlen, dtlen, dslen = struct.unpack("<HHH", data[2:8])
                    pad = lambda n: (n + 7) // 8 * 8          # noqa: E731
                    name = data[8:8 + nlen].split(b"\0")[0].decode(
                        "utf8", "replace")
                    val = data[8 + pad(nlen) + pad(dtlen) + pad(dslen):]
                    strings, ok = [], True
                    for k in range(0, len(val) - 15, 16):
                        ln = struct.unpack("<I", val[k:k + 4])[0]
                        ga = struct.unpack("<Q", val[k + 4:k + 12])[0]
                        gi = struct.unpack("<I", val[k + 12:k + 16])[0]
                        if not (0 < ln < 4096 and 0 < ga < 10 ** 13):
                            ok = False
                            break
                        s = self.global_heap_object(ga, gi, ln)
                        if s is None:
                            ok = False
                            break
                        strings.append(s.decode("utf8", "replace"))
                    if ok and strings:
                        out[name] = strings if len(strings) > 1 else strings[0]
                elif mtype == MSG_CONTINUATION and msize >= 16:
                    a2, l2 = struct.unpack("<QQ", data[:16])
                    walk(a2, l2, depth + 1, seen)
                if mtype == 0 and msize == 0:
                    break
                p += 8 + msize

        for m in self.object_header(oh_addr)["messages"]:
            if m["type"] == MSG_CONTINUATION:
                a, l = struct.unpack("<QQ", m["data"][:16])
                walk(a, l)
        return out
