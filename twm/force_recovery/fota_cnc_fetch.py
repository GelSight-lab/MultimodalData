"""Extract cnc_Mini from FoTa's 392 GB split zip without downloading it.

The FoTa release is a 25-part split zip. Python's ``zipfile`` refuses
multi-disk archives outright, so this module parses the central directory
itself: a seekable HTTP-Range view concatenates the parts, EOCD64 gives the
central directory's (disk, offset), and each member's local header lives at
``part_start[disk] + offset``. Only the tail, the central directory, and the
extracted members are ever transferred.
"""
from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import requests
from huggingface_hub import HfApi, get_token, hf_hub_url

REPO = "alanz-mit/FoundationTactile"
OUT = Path("/media/yxma/Disk1/twm/force_recovery/fota_cnc")


class SplitParts:
    """Ranged reads over the ordered zip parts."""

    def __init__(self, repo: str = REPO):
        api = HfApi()
        info = api.repo_info(repo, repo_type="dataset", files_metadata=True)
        parts = sorted((f.rfilename, f.size) for f in info.siblings
                       if "FoTa_dataset.z" in f.rfilename)
        self.urls = [hf_hub_url(repo, n, repo_type="dataset") for n, _ in parts]
        self.sizes = [s for _, s in parts]
        self.starts = [0]
        for s in self.sizes[:-1]:
            self.starts.append(self.starts[-1] + s)
        self.total = sum(self.sizes)
        self.session = requests.Session()
        tok = get_token()
        if tok:
            self.session.headers["authorization"] = f"Bearer {tok}"
        self.bytes_fetched = 0

    def read_at(self, pos: int, size: int) -> bytes:
        out = bytearray()
        remaining = size
        while remaining > 0 and pos < self.total:
            idx = max(i for i, s in enumerate(self.starts) if s <= pos)
            local = pos - self.starts[idx]
            take = min(remaining, self.sizes[idx] - local)
            r = self.session.get(
                self.urls[idx],
                headers={"Range": f"bytes={local}-{local + take - 1}"},
                timeout=180)
            r.raise_for_status()
            out += r.content
            self.bytes_fetched += len(r.content)
            pos += take
            remaining -= take
        return bytes(out)

    def disk_start(self, disk: int) -> int:
        return self.starts[disk]


@dataclass
class Member:
    name: str
    method: int
    comp_size: int
    file_size: int
    disk: int
    offset: int


def _u16(b, o): return struct.unpack_from("<H", b, o)[0]
def _u32(b, o): return struct.unpack_from("<I", b, o)[0]
def _u64(b, o): return struct.unpack_from("<Q", b, o)[0]


def read_central_directory(parts: SplitParts) -> list[Member]:
    tail = parts.read_at(parts.total - 65536, 65536)
    e = tail.rfind(b"PK\x05\x06")                       # EOCD
    assert e >= 0, "EOCD not found"
    cd_size = _u32(tail, e + 12)
    cd_off = _u32(tail, e + 16)
    cd_disk = _u16(tail, e + 6)
    n_total = _u16(tail, e + 10)
    if 0xFFFFFFFF in (cd_off, cd_size) or n_total == 0xFFFF or cd_disk == 0xFFFF:
        l = tail.rfind(b"PK\x06\x06")                   # EOCD64
        assert l >= 0, "EOCD64 not found"
        cd_disk = _u32(tail, l + 20)
        n_total = _u64(tail, l + 32)
        cd_size = _u64(tail, l + 40)
        cd_off = _u64(tail, l + 48)
    cd = parts.read_at(parts.disk_start(cd_disk) + cd_off, cd_size)

    members, p = [], 0
    while p + 4 <= len(cd) and cd[p:p + 4] == b"PK\x01\x02":
        method = _u16(cd, p + 10)
        comp = _u32(cd, p + 20)
        size = _u32(cd, p + 24)
        nlen, elen, clen = _u16(cd, p + 28), _u16(cd, p + 30), _u16(cd, p + 32)
        disk = _u16(cd, p + 34)
        off = _u32(cd, p + 42)
        name = cd[p + 46:p + 46 + nlen].decode("utf-8", "replace")
        # zip64 extra field overrides any 0xFFFFFFFF placeholders, in order:
        # uncompressed, compressed, offset, disk
        x = p + 46 + nlen
        end_x = x + elen
        while x + 4 <= end_x:
            hid, hsz = _u16(cd, x), _u16(cd, x + 2)
            if hid == 0x0001:
                q = x + 4
                if size == 0xFFFFFFFF:
                    size = _u64(cd, q); q += 8
                if comp == 0xFFFFFFFF:
                    comp = _u64(cd, q); q += 8
                if off == 0xFFFFFFFF:
                    off = _u64(cd, q); q += 8
                if disk == 0xFFFF:
                    disk = _u32(cd, q)
            x += 4 + hsz
        members.append(Member(name, method, comp, size, disk, off))
        p += 46 + nlen + elen + clen
    assert len(members) == n_total, (len(members), n_total)
    return members


def extract_member(parts: SplitParts, m: Member, dest: Path) -> None:
    base = parts.disk_start(m.disk) + m.offset
    head = parts.read_at(base, 30)
    assert head[:4] == b"PK\x03\x04", f"bad local header for {m.name}"
    nlen, elen = _u16(head, 26), _u16(head, 28)
    data_pos = base + 30 + nlen + elen
    dest.parent.mkdir(parents=True, exist_ok=True)
    CHUNK = 8 << 20
    with open(dest, "wb") as f:
        if m.method == 0:                                # stored
            pos, left = data_pos, m.comp_size
            while left:
                take = min(CHUNK, left)
                f.write(parts.read_at(pos, take))
                pos += take
                left -= take
        elif m.method == 8:                              # deflate
            d = zlib.decompressobj(-15)
            pos, left = data_pos, m.comp_size
            while left:
                take = min(CHUNK, left)
                f.write(d.decompress(parts.read_at(pos, take)))
                pos += take
                left -= take
            f.write(d.flush())
        else:
            raise NotImplementedError(f"method {m.method}")


def extract(pattern: str = "cnc_mini", out: Path = OUT) -> list[Path]:
    parts = SplitParts()
    members = read_central_directory(parts)
    print(f"central directory: {len(members)} members "
          f"({parts.bytes_fetched/1e6:.1f} MB fetched)", flush=True)
    picks = [m for m in members
             if pattern in m.name.lower() and not m.name.endswith("/")]
    print(f"{len(picks)} members match {pattern!r}, "
          f"{sum(m.file_size for m in picks)/1e9:.2f} GB", flush=True)
    done = []
    for i, m in enumerate(picks):
        rel = Path(*Path(m.name).parts[1:]) if len(Path(m.name).parts) > 1 \
            else Path(m.name)
        dest = out / rel
        if dest.exists() and dest.stat().st_size == m.file_size:
            done.append(dest)
            continue
        extract_member(parts, m, dest)
        done.append(dest)
        print(f"  [{i+1}/{len(picks)}] {rel} ({m.file_size/1e6:.1f} MB; "
              f"fetched {parts.bytes_fetched/1e9:.2f} GB)", flush=True)
    return done


if __name__ == "__main__":
    import sys

    extract(sys.argv[1] if len(sys.argv) > 1 else "cnc_mini")
