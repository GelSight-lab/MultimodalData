"""Read HDF5 v1 group structures straight off the disk, without the library.

Used to recover pushT/2026-06-18/episode_004.h5, whose root group object header
is 24 bytes of zeros at offset 96 — the library refuses the file with "bad
object header version number" before any of the 79 GB behind it can be reached.

Nothing here writes. It exists so that a repair can be justified by what is
actually on the disk rather than by what a healthy file usually looks like.
"""
from __future__ import annotations

import struct
import sys

SIG = b"\x89HDF\r\n\x1a\n"


class H5Raw:
    def __init__(self, path: str):
        self.f = open(path, "rb")
        self.sb = self._superblock()

    def at(self, off: int, n: int) -> bytes:
        self.f.seek(off)
        return self.f.read(n)

    def _superblock(self) -> dict:
        b = self.at(0, 96)
        if b[:8] != SIG:
            raise ValueError("not an HDF5 file")
        ver = b[8]
        if ver != 0:
            raise ValueError(f"superblock version {ver} unsupported here")
        base, freesp, eof, drv = struct.unpack("<QQQQ", b[24:56])
        name_off, root_oh = struct.unpack("<QQ", b[56:72])
        return {"version": ver, "off_size": b[13], "len_size": b[14],
                "flags": struct.unpack("<I", b[20:24])[0],
                "base": base, "eof": eof,
                "root_name_off": name_off, "root_oh_addr": root_oh}

    # -- v1 object header ------------------------------------------------
    def object_header(self, addr: int) -> dict:
        """Parse a v1 object header. Returns its messages, unvalidated."""
        h = self.at(addr, 16)
        version, _, nmesg = h[0], h[1], struct.unpack("<H", h[2:4])[0]
        refcount, hdrsize = struct.unpack("<II", h[4:12])
        out = {"addr": addr, "version": version, "nmesg": nmesg,
               "refcount": refcount, "hdrsize": hdrsize, "messages": []}
        if version != 1:
            out["error"] = f"object header version {version}, expected 1"
            return out
        body = self.at(addr + 16, hdrsize)
        p = 0
        for _ in range(nmesg):
            if p + 8 > len(body):
                break
            mtype, msize = struct.unpack("<HH", body[p:p + 4])
            flags = body[p + 4]
            data = body[p + 8:p + 8 + msize]
            out["messages"].append({"type": mtype, "size": msize,
                                    "flags": flags, "data": data})
            p += 8 + msize
        return out

    @staticmethod
    def symbol_table_message(data: bytes) -> tuple[int, int]:
        """(v1 B-tree address, local heap address) from a type-0x11 message."""
        return struct.unpack("<QQ", data[:16])

    # -- local heap ------------------------------------------------------
    def local_heap(self, addr: int) -> dict:
        b = self.at(addr, 32)
        if b[:4] != b"HEAP":
            raise ValueError(f"no HEAP signature at {addr}")
        dseg_size, free_head, dseg_addr = struct.unpack("<QQQ", b[8:32])
        return {"addr": addr, "version": b[4], "size": dseg_size,
                "free_head": free_head, "data_addr": dseg_addr}

    def heap_string(self, heap: dict, off: int, limit: int = 512) -> str:
        raw = self.at(heap["data_addr"] + off, limit)
        end = raw.find(b"\0")
        return raw[:end if end >= 0 else limit].decode("utf-8", "replace")

    # -- symbol table node -----------------------------------------------
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
            cache = struct.unpack("<I", e[16:20])[0]
            out.append({"name_off": name_off, "oh_addr": oh_addr,
                        "cache": cache, "scratch": e[24:40]})
        return out

    # -- v1 B-tree (group nodes) -----------------------------------------
    def btree_leaves(self, addr: int, seen=None) -> list[int]:
        """Addresses of every SNOD reachable from this group B-tree."""
        seen = seen if seen is not None else set()
        if addr in seen or addr == 0xFFFFFFFFFFFFFFFF:
            return []
        seen.add(addr)
        b = self.at(addr, 24)
        if b[:4] != b"TREE":
            raise ValueError(f"no TREE signature at {addr}")
        node_type, level = b[4], b[5]
        nused = struct.unpack("<H", b[6:8])[0]
        # keys and children alternate: K0 C0 K1 C1 ... Knused
        # group node keys are 8-byte heap offsets
        body = self.at(addr + 24, (nused * 2 + 1) * 8)
        children = []
        for i in range(nused):
            child = struct.unpack("<Q", body[(i * 2 + 1) * 8:(i * 2 + 2) * 8])[0]
            children.append(child)
        if level == 0:
            return children
        out = []
        for c in children:
            out += self.btree_leaves(c, seen)
        return out

    def group_children(self, btree_addr: int, heap_addr: int) -> list[dict]:
        heap = self.local_heap(heap_addr)
        out = []
        for snod_addr in self.btree_leaves(btree_addr):
            for e in self.snod(snod_addr):
                out.append({"name": self.heap_string(heap, e["name_off"]),
                            "oh_addr": e["oh_addr"], "cache": e["cache"],
                            "snod": snod_addr})
        return out


def group_attributes(path: str, oh_addr: int) -> dict:
    """Attributes of one group, read without the library.

    Recovering a file means taking its schema from a healthy sibling, and the
    temptation is to take the sibling's metadata with it. That is how the first
    recovered episode_004 came out stamped `created_at 2026-06-18T17:51:19` —
    episode_003's start time, 21 minutes off. Everything else in that group
    (fps, task, both sets of serials) genuinely is per-session and identical,
    which is exactly what makes the one per-episode field easy to miss.

    So: schema from the reference, facts about the recording from the file
    itself. This reads the latter.
    """
    h = H5Raw(path)
    out: dict = {}

    def gcol(addr: int, index: int, length: int):
        blk = h.at(addr, 1 << 16)
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

    def walk(addr: int, length: int, depth: int = 0, seen=None) -> None:
        seen = seen if seen is not None else set()
        if addr in seen or depth > 6:
            return
        seen.add(addr)
        blk = h.at(addr, length)
        p = 0
        while p + 8 <= len(blk):
            mtype, msize = struct.unpack("<HH", blk[p:p + 4])
            data = blk[p + 8:p + 8 + msize]
            if mtype == 0x0C and len(data) > 8:
                nlen, dtlen, dslen = struct.unpack("<HHH", data[2:8])
                pad = lambda n: (n + 7) // 8 * 8      # noqa: E731
                name = data[8:8 + nlen].split(b"\0")[0].decode("utf8", "replace")
                off = 8 + pad(nlen) + pad(dtlen) + pad(dslen)
                val = data[off:]
                strings, ok = [], True
                for k in range(0, len(val) - 15, 16):
                    ln = struct.unpack("<I", val[k:k + 4])[0]
                    ga = struct.unpack("<Q", val[k + 4:k + 12])[0]
                    gi = struct.unpack("<I", val[k + 12:k + 16])[0]
                    if not (0 < ln < 4096 and 0 < ga < 10 ** 13):
                        ok = False
                        break
                    s = gcol(ga, gi, ln)
                    if s is None:
                        ok = False
                        break
                    strings.append(s.decode("utf8", "replace"))
                if ok and strings:
                    out[name] = strings if len(strings) > 1 else strings[0]
            elif mtype == 0x10 and msize >= 16:
                a2, l2 = struct.unpack("<QQ", data[:16])
                walk(a2, l2, depth + 1, seen)
            if mtype == 0 and msize == 0:
                break
            p += 8 + msize

    oh = h.object_header(oh_addr)
    for m in oh["messages"]:
        if m["type"] == 0x10:
            a, l = struct.unpack("<QQ", m["data"][:16])
            walk(a, l)
    return out


def describe(path: str) -> None:
    h = H5Raw(path)
    sb = h.sb
    print(f"=== {path}")
    print(f"  superblock flags {sb['flags']}  EOF {sb['eof']:,}  "
          f"root OH @ {sb['root_oh_addr']}")
    oh = h.object_header(sb["root_oh_addr"])
    print(f"  root object header: version={oh['version']} nmesg={oh['nmesg']} "
          f"refcount={oh['refcount']} hdrsize={oh['hdrsize']}")
    if oh.get("error"):
        print(f"  !! {oh['error']}")
        return
    for m in oh["messages"]:
        print(f"    message type 0x{m['type']:02x} size {m['size']}")
        if m["type"] == 0x11:
            bt, hp = h.symbol_table_message(m["data"])
            print(f"      symbol table: B-tree @ {bt}, local heap @ {hp}")
            for c in h.group_children(bt, hp):
                print(f"        {c['name']!r:24s} OH @ {c['oh_addr']:,}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        describe(p)
