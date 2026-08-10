"""FeelAnyForce: pull the ORIGINAL timestamped PNGs out of the spanned zip.

Why this file exists
--------------------
The FeelAnyForce copy we already held locally ships tactile images with the
force column EMPTY, and the labels live in separate CSVs that reference frames
by *timestamped filename*. Our first attempt re-attached the labels by
inferring a per-capture frame index; that join FAILED its control (pooled
rho 0.455 with the real labels vs 0.442 with labels shuffled *within* each
capture), i.e. all of the apparent signal was between-capture structure and
the frame mapping carried nothing. So the dataset was excluded.

The original archive removes the guesswork: the image filenames ARE the
timestamps the CSVs reference, so the join is exact by construction
(110,109 / 110,109 CSV rows match an archive entry by full path).

The archive is a 4-segment spanned zip totalling ~81 GB
(dataset.z01/.z02/.z03 = disks 0/1/2, dataset.zip = disk 3, the LAST piece
and the one holding the central directory). We never download a whole
segment: every member is fetched with an HTTP Range request against the
segment its local header lives in, then inflated in memory.

Integrity is not assumed. The central directory carries a CRC-32 and an
uncompressed size for every member; both are checked after inflation and a
mismatch raises. A truncated / mis-offset read is exactly the failure mode
that would silently corrupt a frame-label join, which is the thing this whole
exercise is trying to make defensible, so it fails loudly instead.

Layout facts (verified, see task_plan.md ledger):
  Zip64 EOCD in dataset.zip: 305,818 entries, central directory at offset
  17,411,261,400, size 38,401,830. 101,883 entries are
  `dataset/<capture>/tactile/<timestamp>.png` over 42 captures.

Run:
  python -m force_recovery.faf_extract crc      # cache CD CRC-32s (38 MB, once)
  python -m force_recovery.faf_extract select   # stratified subset -> plan json
  python -m force_recovery.faf_extract fetch    # range-fetch + inflate + verify
  python -m force_recovery.faf_extract report   # what is on disk right now
"""
from __future__ import annotations

import concurrent.futures as cf
import csv
import io
import json
import struct
import sys
import threading
import time
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import requests

from .run_episode import OUT_ROOT

BASE = ("https://huggingface.co/datasets/amirsh1376/FeelAnyForce/"
        "resolve/main/")
# disk index -> (filename, size in bytes).  Disk 3 is `dataset.zip`, the last
# segment; the first three are exactly 20 GiB each.
SEGMENTS = [("dataset.z01", 21474836480),
            ("dataset.z02", 21474836480),
            ("dataset.z03", 21474836480),
            ("dataset.zip", 17449663328)]
CD_OFFSET, CD_SIZE, CD_DISK = 17411261400, 38401830, 3

ROOT = OUT_ROOT
CD_JSON = ROOT / "faf_cd.json"                  # cached, do not re-download
CRC_JSON = ROOT / "faf_cd_crc.json"             # name -> [crc32, usize]
LABELS = ROOT / "faf_labels"
IMG_DIR = ROOT / "faf_images"
PLAN_JSON = ROOT / "faf_plan.json"

N_TIER_A = 800             # frames per capture that HAS contact-free frames
# 200 was sized for a first look. Tier A holds 58,630 frames across its 14
# captures and roughly a fifth of any sample survives the fully-imaged
# filter, so 200 each capped the scorable set near 590 — below the 2,000
# the evaluation wants. 800 clears it with room for the filter.
N_TIER_B = 80              # frames per capture that does not (see cmd_select)
N_DECILES = 10             # |Fz| strata within each capture
N_REF = 3                  # lowest-|Fz| frames per session -> median reference
SESSION_GAP = 600.0        # s; a longer hole in the timestamps = new session
ZERO_N = 0.10              # |Fz| below this counts as contact-free
WORKERS = 12
_HDR_SLACK = 1024          # bytes fetched past the 30-byte local header, to
                           # cover name+extra in the same request

_lock = threading.Lock()
_url_cache: dict[int, tuple[str, float]] = {}
_bytes_downloaded = 0


# --------------------------------------------------------------- transport
def _session() -> requests.Session:
    s = getattr(_tls, "sess", None)
    if s is None:
        s = _tls.sess = requests.Session()
    return s


_tls = threading.local()


def _resolved(disk: int) -> str:
    """Final CDN URL for a segment (HF 302s to a signed URL; cache it)."""
    with _lock:
        hit = _url_cache.get(disk)
        if hit and time.time() - hit[1] < 600:
            return hit[0]
    r = requests.head(BASE + SEGMENTS[disk][0], allow_redirects=True,
                      timeout=60)
    r.raise_for_status()
    with _lock:
        _url_cache[disk] = (r.url, time.time())
    return r.url


def read_range(disk: int, off: int, n: int, retries: int = 5) -> bytes:
    """`n` bytes from `off` in segment `disk`, spanning into later segments.

    A spanned zip lets a member's payload run off the end of one segment and
    continue at byte 0 of the next, so this is not optional plumbing.
    """
    global _bytes_downloaded
    out = bytearray()
    while n > 0:
        if disk >= len(SEGMENTS):
            raise EOFError("range ran past the last segment")
        seg_size = SEGMENTS[disk][1]
        take = min(n, seg_size - off)
        if take <= 0:
            disk, off = disk + 1, 0
            continue
        hdrs = {"Range": f"bytes={off}-{off + take - 1}"}
        for attempt in range(retries):
            try:
                url = _resolved(disk)
                r = _session().get(url, headers=hdrs, timeout=120)
                if r.status_code in (403, 401):     # signed URL expired
                    with _lock:
                        _url_cache.pop(disk, None)
                    raise requests.HTTPError(f"{r.status_code} re-resolve")
                if r.status_code != 206:
                    raise requests.HTTPError(f"expected 206, got "
                                             f"{r.status_code}")
                blob = r.content
                if len(blob) != take:
                    raise requests.HTTPError(f"short read {len(blob)}/{take}")
                break
            except Exception:                        # noqa: BLE001
                if attempt == retries - 1:
                    raise
                time.sleep(1.5 * (attempt + 1))
        out += blob
        with _lock:
            _bytes_downloaded += len(blob)
        n -= take
        off += take
    return bytes(out)


# --------------------------------------------------- central directory CRCs
def load_cd() -> list[dict]:
    """Cached parse: {n: name, d: disk, o: local-header offset, c, u, m}."""
    return json.loads(CD_JSON.read_text())


def cmd_crc() -> None:
    """Fetch the 38 MB central directory once and cache CRC-32 + usize.

    The cached `faf_cd.json` records offsets and sizes but NOT the CRC, and
    without an independent checksum "verified extraction" would just mean
    "zlib did not raise". 38 MB against a ~2 GB budget is a cheap way to make
    the integrity claim real, so we pay it once.
    """
    if CRC_JSON.exists():
        print(f"already cached: {CRC_JSON}")
        return
    print(f"fetching central directory: {CD_SIZE / 1e6:.1f} MB", flush=True)
    cd = read_range(CD_DISK, CD_OFFSET, CD_SIZE)
    out, p, n = {}, 0, 0
    while p + 46 <= len(cd):
        if cd[p:p + 4] != b"PK\x01\x02":
            raise ValueError(f"bad central header at {p}")
        (crc, _csz, usz, nlen, elen, clen) = struct.unpack_from(
            "<IIIHHH", cd, p + 16)
        name = cd[p + 46:p + 46 + nlen].decode("utf-8", "replace")
        extra = cd[p + 46 + nlen:p + 46 + nlen + elen]
        if usz == 0xFFFFFFFF:                       # zip64 extra field
            q = 0
            while q + 4 <= len(extra):
                tag, sz = struct.unpack_from("<HH", extra, q)
                if tag == 0x0001:
                    usz = struct.unpack_from("<Q", extra, q + 4)[0]
                    break
                q += 4 + sz
        if name.endswith(".png"):
            out[name] = [crc, usz]
        p += 46 + nlen + elen + clen
        n += 1
    if n != 305818:
        raise ValueError(f"parsed {n} central headers, expected 305818")
    CRC_JSON.write_text(json.dumps(out))
    print(f"-> {CRC_JSON}: {len(out)} png CRCs from {n} entries")


# ------------------------------------------------------------- extraction
def extract_member(rec: dict, crc_expect: int | None,
                   usize_expect: int | None) -> bytes:
    """Range-fetch one member, inflate it, and verify it. Raises on mismatch.

    The local file header's own name/extra lengths are used to find the
    payload -- NOT the central directory's name length. The two are allowed
    to differ (extra fields routinely do), and trusting the wrong one shifts
    the payload start by a few bytes, which yields either a zlib error or,
    worse, a plausible-looking short image.
    """
    want = 30 + _HDR_SLACK + rec["c"]
    blob = read_range(rec["d"], rec["o"], want)
    if blob[:4] != b"PK\x03\x04":
        raise ValueError(f"{rec['n']}: no local header signature at "
                         f"disk {rec['d']}+{rec['o']} (got {blob[:4]!r})")
    flag, method = struct.unpack_from("<HH", blob, 6)
    lcrc, lcsz, lusz = struct.unpack_from("<III", blob, 14)
    nlen, elen = struct.unpack_from("<HH", blob, 26)
    start = 30 + nlen + elen
    lname = blob[30:30 + nlen].decode("utf-8", "replace")
    if lname != rec["n"]:
        raise ValueError(f"local header name {lname!r} != CD name "
                         f"{rec['n']!r}")
    if method != rec["m"]:
        raise ValueError(f"{rec['n']}: method {method} != CD {rec['m']}")
    need = start + rec["c"]
    data = blob[start:need] if need <= len(blob) else (
        blob[start:] + read_range(rec["d"], rec["o"] + len(blob),
                                  need - len(blob)))
    if len(data) != rec["c"]:
        raise ValueError(f"{rec['n']}: got {len(data)} payload bytes, "
                         f"CD says {rec['c']}")
    if method == 8:
        raw = zlib.decompressobj(-15).decompress(data)
    elif method == 0:
        raw = data
    else:
        raise ValueError(f"{rec['n']}: unsupported method {method}")

    # bit 3 => sizes/crc live in a trailing data descriptor, local header
    # zeros are meaningless; otherwise they are a second independent check.
    if not (flag & 0x08):
        if lcsz not in (0xFFFFFFFF, rec["c"]) or lusz not in (0xFFFFFFFF,
                                                              rec["u"]):
            raise ValueError(f"{rec['n']}: local sizes {lcsz}/{lusz} "
                             f"disagree with CD {rec['c']}/{rec['u']}")
        if lcrc and zlib.crc32(raw) & 0xFFFFFFFF != lcrc:
            raise ValueError(f"{rec['n']}: local-header CRC mismatch")
    if usize_expect is not None and len(raw) != usize_expect:
        raise ValueError(f"{rec['n']}: {len(raw)} bytes != CD usize "
                         f"{usize_expect}")
    if len(raw) != rec["u"]:
        raise ValueError(f"{rec['n']}: {len(raw)} bytes != cached usize "
                         f"{rec['u']}")
    if crc_expect is not None:
        got = zlib.crc32(raw) & 0xFFFFFFFF
        if got != crc_expect:
            raise ValueError(f"{rec['n']}: CRC-32 {got:08x} != CD "
                             f"{crc_expect:08x}")
    return raw


def local_path(name: str) -> Path:
    """dataset/<capture>/tactile/<ts>.png -> faf_images/<capture>/<ts>.png"""
    parts = name.split("/")
    return IMG_DIR / parts[1] / parts[-1]


# -------------------------------------------------------------- label join
def load_labels() -> list[dict]:
    """CSV rows -> {key (archive path), capture, ts, fz, split}.

    `FT` is a 6-tuple string; Fz is the THIRD value.
    """
    rows = []
    for split in ("train", "val", "test"):
        with open(LABELS / f"TacForce_{split}_set.csv") as fh:
            for r in csv.DictReader(fh):
                ft = [float(v) for v in r["FT"].split()]
                if len(ft) != 6:
                    raise ValueError(f"FT is not a 6-tuple: {r['FT']!r}")
                tac = r["tactile"].strip()
                rows.append({"key": "dataset/" + tac,
                             "capture": tac.split("/")[0],
                             "ts": Path(tac).stem,
                             "fz": ft[2], "split": split})
    return rows


def joined() -> list[dict]:
    """Label rows that resolve to a real archive entry (must be all of them).

    The three CSVs contain 110,109 rows but only 101,883 distinct frames:
    3,188 frames appear in more than one split (the shipped train/val/test
    split leaks). Every duplicate carries an identical Fz, so de-duplicating
    by path is lossless -- and necessary, or one frame could land in both
    halves of our own fit/eval split.
    """
    cd = {r["n"]: r for r in load_cd()}
    rows = load_labels()
    miss = [r for r in rows if r["key"] not in cd]
    if miss:
        raise ValueError(f"{len(miss)}/{len(rows)} label rows have no archive "
                         f"entry, e.g. {miss[0]['key']}")
    uniq: dict[str, dict] = {}
    for r in rows:
        prev = uniq.get(r["key"])
        if prev is not None and abs(prev["fz"] - r["fz"]) > 1e-9:
            raise ValueError(f"{r['key']}: conflicting Fz "
                             f"{prev['fz']} vs {r['fz']}")
        uniq[r["key"]] = r
    for r in uniq.values():
        r["rec"] = cd[r["key"]]
    return list(uniq.values())


def sessions(rows: list[dict]) -> list[dict]:
    """Tag each row of ONE capture with a session id (gap > SESSION_GAP).

    A capture is not one sitting: `cube28_corner` spans 29 days and 11
    sessions. Illumination and gel state drift between sittings, so a single
    reference frame per capture would be wrong for most of it; references are
    picked per session instead.
    """
    rows = sorted(rows, key=lambda r: int(r["ts"]))
    sid = 0
    for i, r in enumerate(rows):
        if i and (int(r["ts"]) - int(rows[i - 1]["ts"])) / 1000.0 > SESSION_GAP:
            sid += 1
        r["session"] = sid
    return rows


# ------------------------------------------------------------- selection
def _stratified(pool: list[dict], n: int, rng) -> list[dict]:
    """`n` rows spread over |Fz| deciles of `pool`.

    Uniform sampling would be dominated by whatever force level the capture
    dwells at, which lets a model score well by answering "is anything
    touching". Decile stratification inside the capture spends the download
    budget across that capture's own force range instead.
    """
    a = np.array([abs(r["fz"]) for r in pool])
    edges = np.quantile(a, np.linspace(0, 1, N_DECILES + 1))
    bins = np.clip(np.searchsorted(edges, a, side="right") - 1,
                   0, N_DECILES - 1)
    per, picked = n // N_DECILES, []
    for b in range(N_DECILES):
        idx = np.where(bins == b)[0]
        picked += [pool[i] for i in rng.choice(idx, min(per, len(idx)),
                                               replace=False)]
    taken = {id(r) for r in picked}
    left = [i for i in range(len(pool)) if id(pool[i]) not in taken]
    short = min(n - len(picked), len(left))
    if short > 0:
        picked += [pool[i] for i in rng.choice(left, short, replace=False)]
    return picked


def cmd_select() -> None:
    """Stratified subset, split by whether a capture has a CLEAN reference.

    Our pipeline is a difference method: every feature is computed from
    `img - ref`, so a reference frame that already carries load biases every
    downstream number. That is the exact defect that caps FEATS, so it gets
    measured here rather than assumed.

    The labels answer it directly. 14 of the 42 captures contain frames at
    |Fz| < 0.1 N (tier A: a genuine contact-free reference exists, and per
    session, so illumination drift across a multi-day capture is handled).
    The other 28 -- every `<object>_<n>` re-run, all recorded in Jun/Jul 2024
    while the tier-A captures are Apr/May -- have a MINIMUM |Fz| of 4.9-6.0 N:
    there is no unloaded frame anywhere in them, and borrowing one from the
    Apr/May capture of the same object would cross a 2-3 month gap. Tier B is
    therefore run on a median-image reference (the FEATS fallback) and
    reported separately, never mixed into the headline.
    """
    rows = joined()
    by_cap = defaultdict(list)
    for r in rows:
        by_cap[r["capture"]].append(r)
    rng = np.random.default_rng(0)
    plan, ref_plan, report = [], [], []
    for cap in sorted(by_cap):
        rs = sessions(by_cap[cap])
        by_ses = defaultdict(list)
        for r in rs:
            by_ses[r["session"]].append(r)
        zero_ses = {s for s, v in by_ses.items()
                    if min(abs(r["fz"]) for r in v) < ZERO_N}
        tier = "A" if zero_ses else "B"
        if tier == "A":
            refs = []
            for s in sorted(zero_ses):
                refs += sorted(by_ses[s], key=lambda r: abs(r["fz"]))[:N_REF]
            ref_keys = {r["key"] for r in refs}
            pool = [r for r in rs
                    if r["session"] in zero_ses and r["key"] not in ref_keys]
            picked = _stratified(pool, N_TIER_A, rng)
            ref_plan += refs
        else:
            refs = []
            picked = _stratified(rs, N_TIER_B, rng)
        plan += picked
        a_all = np.array([abs(r["fz"]) for r in rs])
        report.append({
            "capture": cap, "tier": tier, "n_rows": len(rs),
            "n_sessions": len(by_ses), "n_zero_sessions": len(zero_ses),
            "n_sampled": len(picked), "n_ref": len(refs),
            "min_absfz": float(a_all.min()), "max_absfz": float(a_all.max()),
            "ref_max_absfz": float(max((abs(r["fz"]) for r in refs),
                                       default=float("nan")))})
    keep = ("key", "capture", "ts", "fz", "split", "session")
    out = {"n_captures": len(by_cap), "n_eval": len(plan),
           "n_ref": len(ref_plan),
           "eval": [{k: r[k] for k in keep} for r in plan],
           "ref": [{k: r[k] for k in keep} for r in ref_plan],
           "per_capture": report}
    PLAN_JSON.write_text(json.dumps(out))
    mb = sum(r["rec"]["c"] for r in plan + ref_plan) / 1e6
    na = sum(r["n_sampled"] for r in report if r["tier"] == "A")
    print(f"{len(by_cap)} captures ("
          f"{sum(r['tier'] == 'A' for r in report)} tier A / "
          f"{sum(r['tier'] == 'B' for r in report)} tier B), "
          f"{len(plan)} eval ({na} tier A) + {len(ref_plan)} ref frames, "
          f"{mb:.0f} MB compressed")
    for tier in ("A", "B"):
        caps = {r["capture"] for r in report if r["tier"] == tier}
        a = np.array([abs(r["fz"]) for r in plan if r["capture"] in caps])
        h, e = np.histogram(a, bins=[0, .5, 1, 2, 4, 8, 16, 32])
        print(f"tier {tier} |Fz| histogram (n={len(a)}): "
              + "  ".join(f"{e[i]:g}-{e[i+1]:g}N:{h[i]}" for i in range(len(h))))
    print(f"-> {PLAN_JSON}")


# ---------------------------------------------------------------- fetching
def cmd_fetch() -> None:
    """Range-fetch every planned member, verify, cache the PNG bytes."""
    plan = json.loads(PLAN_JSON.read_text())
    cd = {r["n"]: r for r in load_cd()}
    crcs = json.loads(CRC_JSON.read_text())
    todo = [e for e in plan["ref"] + plan["eval"]
            if not local_path(e["key"]).exists()]
    print(f"{len(plan['ref']) + len(plan['eval'])} planned, "
          f"{len(todo)} missing -> fetching", flush=True)
    stats = {"ok": 0, "fail": 0, "crc_checked": 0}
    errs = []

    def work(e):
        rec = cd[e["key"]]
        c = crcs.get(e["key"])
        raw = extract_member(rec, c[0] if c else None, c[1] if c else None)
        p = local_path(e["key"])
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(raw)
        return c is not None

    t0 = time.time()
    with cf.ThreadPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(work, e): e for e in todo}
        for i, fut in enumerate(cf.as_completed(futs)):
            try:
                stats["crc_checked"] += int(fut.result())
                stats["ok"] += 1
            except Exception as exc:                      # noqa: BLE001
                stats["fail"] += 1
                errs.append(f"{futs[fut]['key']}: {exc}")
            if (i + 1) % 250 == 0:
                print(f"  {i+1}/{len(todo)}  ok={stats['ok']} "
                      f"fail={stats['fail']}  "
                      f"{_bytes_downloaded/1e6:.0f} MB  "
                      f"{(time.time()-t0)/60:.1f} min", flush=True)
    print(f"fetched ok={stats['ok']} fail={stats['fail']} "
          f"crc-verified={stats['crc_checked']} "
          f"({_bytes_downloaded/1e6:.0f} MB downloaded)")
    for e in errs[:20]:
        print("  FAIL", e)
    if stats["fail"]:
        raise SystemExit(f"{stats['fail']} members failed verification")


def cmd_verify(n: int = 300) -> None:
    """Re-check cached files against the central-directory CRC-32.

    Cheap, offline, and it proves the cache on disk is what the archive says
    it is -- not merely that the download did not raise at the time.
    """
    crcs = json.loads(CRC_JSON.read_text())
    plan = json.loads(PLAN_JSON.read_text())
    entries = plan["ref"] + plan["eval"]
    rng = np.random.default_rng(1)
    sel = [entries[i] for i in rng.permutation(len(entries))[:n]]
    bad = 0
    for e in sel:
        raw = local_path(e["key"]).read_bytes()
        crc, usz = crcs[e["key"]]
        if (zlib.crc32(raw) & 0xFFFFFFFF) != crc or len(raw) != usz:
            bad += 1
            print("  BAD", e["key"])
    print(f"verify: {len(sel) - bad}/{len(sel)} cached files match the "
          f"central-directory CRC-32 and size")
    if bad:
        raise SystemExit("cache corrupt")


def cmd_report() -> None:
    """What is on disk, per capture, plus reference-frame quality."""
    plan = json.loads(PLAN_JSON.read_text())
    have = defaultdict(int)
    for e in plan["ref"] + plan["eval"]:
        if local_path(e["key"]).exists():
            have[e["capture"]] += 1
    print(f"{'capture':22s} {'T':>2s} {'rows':>6s} {'ses':>4s} {'0ses':>5s} "
          f"{'sampled':>8s} {'ref':>4s} {'onDisk':>7s} {'min|Fz|':>8s} "
          f"{'refMax':>7s} {'max|Fz|':>8s}")
    for r in plan["per_capture"]:
        print(f"{r['capture']:22s} {r['tier']:>2s} {r['n_rows']:6d} "
              f"{r['n_sessions']:4d} {r['n_zero_sessions']:5d} "
              f"{r['n_sampled']:8d} {r['n_ref']:4d} {have[r['capture']]:7d} "
              f"{r['min_absfz']:8.3f} {r['ref_max_absfz']:7.3f} "
              f"{r['max_absfz']:8.2f}")
    tot = sum(have.values())
    size = sum(p.stat().st_size for p in IMG_DIR.rglob("*.png"))
    print(f"total {tot} images, {size/1e6:.0f} MB on disk")


CMDS = {"crc": cmd_crc, "select": cmd_select, "fetch": cmd_fetch,
        "verify": cmd_verify, "report": cmd_report}

if __name__ == "__main__":
    CMDS[sys.argv[1] if len(sys.argv) > 1 else "report"]()
