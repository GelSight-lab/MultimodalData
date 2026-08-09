"""Print what is actually on the disk inside an HDF5 file, structure by structure.

A CLI over ``react_preprocess.h5raw``. Point it at a file that will not open
and a healthy sibling; the difference is usually the whole diagnosis.

    python scripts/h5_forensics.py <broken.h5> [<healthy.h5> ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from twm.react_preprocess import repair                         # noqa: E402
from twm.react_preprocess.h5raw import (H5Raw, MSG_CONTINUATION,  # noqa: E402
                                        MSG_SYMBOL_TABLE)

MSG_NAMES = {0x00: "NIL", 0x01: "Dataspace", 0x03: "Datatype", 0x05: "Fill",
             0x08: "Layout", 0x0B: "Filter", 0x0C: "Attribute",
             0x10: "Continuation", 0x11: "SymbolTable", 0x12: "ModTime"}


def describe(path: str) -> None:
    print(f"=== {path}")
    with H5Raw(path) as h:
        sb = h.superblock
        print(f"  superblock: consistency_flags={sb['consistency_flags']} "
              f"(1 = still open for write)")
        print(f"              EOF field {sb['eof_address']:,}  "
              f"file on disk {sb['file_size']:,}"
              + ("   <-- never updated; the file was not closed"
                 if sb["eof_address"] < sb["file_size"] else ""))
        print(f"              root object header @ {sb['root_oh_address']}")

        oh = h.object_header(sb["root_oh_address"])
        print(f"  root header: version={oh.get('version')} "
              f"nmesg={oh.get('nmesg')} hdrsize={oh.get('hdrsize')}")
        if oh.get("error"):
            print(f"  !! {oh['error']}")
        else:
            for m in oh["messages"]:
                name = MSG_NAMES.get(m["type"], hex(m["type"]))
                print(f"    message {name} ({m['size']} bytes)")
                if m["type"] == MSG_SYMBOL_TABLE:
                    bt, hp = h.symbol_table_message(m["data"])
                    print(f"      symbol table: B-tree @ {bt}, heap @ {hp}")
                    try:
                        kids = h.group_children(bt, hp)
                    except Exception as exc:                    # noqa: BLE001
                        print(f"      unreadable: {exc}")
                        kids = []
                    if not kids:
                        print("      (no children — the root group's "
                              "directory never reached the disk)")
                    for c in kids:
                        print(f"        {c['name']!r:24s} OH @ {c['oh_addr']:,}")

        attrs = h.group_attributes(repair.METADATA_OH_ADDR)
        if attrs:
            print(f"  metadata group attributes (read from THIS file):")
            for k, v in sorted(attrs.items()):
                print(f"    {k:20s} {v}")

    d = repair.diagnose(path)
    print(f"  diagnosis: {d}")
    print(f"  repairable: {d.repairable}")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    for p in sys.argv[1:]:
        describe(p)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
