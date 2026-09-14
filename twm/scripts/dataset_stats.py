"""Dataset statistics for the card: collect numbers, then draw two figures.

    python twm/scripts/dataset_stats.py --out-dir assets

Two eras, drawn separately because they are not comparable: the wrist-camera
era (`data/{motherboard,pushT,rope}`) and the earlier Arducam session
(`data/validation`), which carries different wrist cameras and a different
calibration epoch.

WHICH EPISODES COUNT is read from the Hub, never from a table in this file.
The version this replaces held

    NEW = {"motherboard": [...], "pushT": ["2026-09-10", "2026-09-11"], ...}

and 2026-09-12 was simply absent, so pushT's 24 newest segments -- 45 % of
the task -- were missing from the published figure with nothing to indicate
it. A date table is the same defect as `stage_zup.py`'s, which silently
skipped a whole day of Z-up conversion.

Honesty constraints carried over from the first version:
  * the force distribution uses UNSATURATED samples only; the saturated
    fraction is annotated separately rather than mixed into the distribution
  * scale reports segments AND source recordings, or 25 segments reads as 25
    independent recordings
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

REPO = "yxma/React"
CEILING_N = 7.87      # the isotonic stage's top output; see react_calib.F_MAX_N
CONTACT_N = 0.15
_SEG = re.compile(r"_seg\d+$")

# The wrist-camera era is the three task folders; `data/validation` is the
# earlier session (2026-09-09), which carries different wrist cameras and a
# different calibration epoch, so it is drawn on its own axes.
ERAS = {
    "new_era": ["motherboard", "pushT", "rope"],
    "old_era": ["validation"],
}


def published_keys(folder: str, api) -> list[str]:
    """`<date>/<episode>` for every parquet the Hub serves under `data/<folder>`."""
    out = []
    for f in api.list_repo_files(REPO, repo_type="dataset"):
        p = f.split("/")
        if (len(p) == 5 and p[0] == "data" and p[1] == folder
                and p[2] == "meta" and p[4].endswith(".parquet")):
            out.append(f"{p[3]}/{p[4][: -len('.parquet')]}")
    return sorted(out)


def collect(folder: str, keys: list[str], fetch) -> dict:
    """Statistics over the PUBLISHED parquets, fetched from the Hub.

    Not the local tree. The figure describes what a downloader gets, and this
    session found three ways the two diverge -- index files listing episodes
    the release does not publish, `data/validation` existing only on the Hub
    (it was assembled there with CommitOperationCopy, so there is no local
    tree to read), and local parquets rewritten after upload. Reading the
    published copy makes the figure incapable of describing anything else.
    """
    o = dict(segs=0, frames=0, f=[], cL=0, cR=0, both=0, newL=0, srcs=set())
    for key in keys:
        date, ep = key.split("/")
        t = pq.read_table(fetch(f"data/{folder}/meta/{date}/{ep}.parquet"))
        col = lambda c: np.asarray(t.column(c).to_numpy(zero_copy_only=False), float)
        fl, fr = col("force_left_normal_n"), col("force_right_normal_n")
        n = len(fl)
        o["segs"] += 1
        o["frames"] += n
        o["f"].append(np.concatenate([fl, fr]))
        o["cL"] += int((fl > CONTACT_N).sum())
        o["cR"] += int((fr > CONTACT_N).sum())
        o["both"] += int(((fl > CONTACT_N) & (fr > CONTACT_N)).sum())
        o["newL"] += int(np.asarray(
            t.column("tactile_left_is_new").to_numpy(zero_copy_only=False)).sum())
        o["srcs"].add(f"{date}/{_SEG.sub('', ep)}")
    return o


def summarise(o: dict) -> dict:
    f = np.concatenate(o["f"])
    c = f[f > CONTACT_N]
    saturated = c >= CEILING_N - 1e-6
    unsat = c[~saturated]
    return dict(
        segs=o["segs"], sources=len(o["srcs"]),
        minutes=o["frames"] / 1800, frames=o["frames"],
        contact_frames=int(len(c)),
        sat_pct=float(saturated.mean() * 100),
        contact_pct_L=o["cL"] / o["frames"] * 100,
        contact_pct_R=o["cR"] / o["frames"] * 100,
        both_pct=o["both"] / o["frames"] * 100,
        new_pct=o["newL"] / o["frames"] * 100,
        f_q=[float(x) for x in np.percentile(unsat, [5, 25, 50, 75, 95])],
        # ECDF rather than a density: the calibration ends in an isotonic
        # regression, so its output takes a few thousand discrete values and a
        # density draws them as a comb of spikes -- true, but it reads as a
        # plotting error. A step function says the same thing honestly.
        f_ecdf_x=[float(x) for x in np.percentile(unsat, np.linspace(0, 100, 101))],
    )


def gather() -> dict:
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    fetch = lambda f: hf_hub_download(REPO, f, repo_type="dataset")
    out = {"ceiling": CEILING_N, "contact_thresh": CONTACT_N}
    for era, folders in ERAS.items():
        out[era] = {}
        for folder in folders:
            keys = published_keys(folder, api)
            if not keys:
                print(f"  [{era}] {folder}: 未发布，跳过")
                continue
            o = collect(folder, keys, fetch)
            out[era][folder] = summarise(o)
            print(f"  [{era}] {folder}: {o['segs']} 段 / {len(o['srcs'])} 源录制 "
                  f"/ {o['frames'] / 1800:.1f} min")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="assets")
    ap.add_argument("--stats-only", action="store_true")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    d = gather()
    (out / "dataset_stats.json").write_text(json.dumps(d, indent=1) + "\n")
    print(f"  -> {out / 'dataset_stats.json'}")
    if a.stats_only:
        return 0
    import dataset_stats_fig
    dataset_stats_fig.draw(d, out)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
