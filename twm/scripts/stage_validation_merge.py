"""Stage the 2026-09-09 calibration epoch into one `data/validation` folder.

`data/validation` already holds the motherboard episodes recorded at 04:00 that
morning. Two more groups belong with them, for the same reason: they carry the
2026-09-09 extrinsics and must not be inherited by a task tree solved for a
different epoch.

  * `data/pushT_2026-09-09/episode_000` -- one uncut episode, moved whole
  * motherboard 2026-09-09, the 17:20/17:41 session -- 5 cut segments

EPISODE NUMBERS COLLIDE. The recorder reuses them within a date: that day's
`episode_000.h5` was written four times, and the file on disk now (3802 frames,
17:20) is NOT the one published as `validation/episode_000` (6661 frames,
04:01). Same name, different recording. So arrivals are renumbered onto the end
of the folder, and `source_recording` keeps the provenance the number loses.

Renumbering is not a rename: the parquet carries `episode` and `episode_index`
as data, and a folder whose files disagree with their own contents is worse
than one that never had the columns. The cut segments have no index columns at
all -- they are added here.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from twm.react_preprocess.meta import add_index_columns  # noqa: E402

CUT_ROOT = Path("/media/yxma/Disk1/twm/release_cut")
DATE = "2026-09-09"
TASK_INDEX = {"motherboard": 0, "pushT": 1, "rope": 2}
VIDEO_STREAMS = ("view_left", "view_middle", "view_right",
                 "tactile_left", "tactile_right", "wrist_left", "wrist_right")

# Renumbered onto the end of the existing 000/001/002, in the order asked for:
# pushT first, then the motherboard segments.
ARRIVALS = [
    {"task": "pushT", "src": "hf:data/pushT_2026-09-09", "episode": "episode_000"},
    {"task": "motherboard", "src": "cut", "episode": "episode_000_seg00"},
    {"task": "motherboard", "src": "cut", "episode": "episode_001_seg00"},
    {"task": "motherboard", "src": "cut", "episode": "episode_001_seg01"},
    {"task": "motherboard", "src": "cut", "episode": "episode_001_seg02"},
    {"task": "motherboard", "src": "cut", "episode": "episode_001_seg03"},
]


def _reindex(table: pa.Table, task: str, new_ep: str, new_idx: int) -> pa.Table:
    """Set the LeRobot index columns to what this episode is now called.

    Renumbering is not a rename: `episode` and `episode_index` are stored as
    DATA, so a folder whose files disagree with their own contents is worse
    than one that never carried the columns.
    """
    return add_index_columns(table, task, TASK_INDEX[task],
                             f"{DATE}/{new_ep}", new_idx)


def stage(out: Path, start_index: int, download) -> list[dict]:
    (out / "meta" / DATE).mkdir(parents=True, exist_ok=True)
    manifest = []
    for i, a in enumerate(ARRIVALS):
        new_idx = start_index + i
        new_ep = f"episode_{new_idx:03d}"
        task, old_ep = a["task"], a["episode"]

        if a["src"] == "cut":
            src_pq = CUT_ROOT / task / "meta" / DATE / f"{old_ep}.parquet"
        else:
            src_pq = Path(download(f"{a['src'][3:]}/meta/{DATE}/{old_ep}.parquet"))
        t = pq.read_table(str(src_pq))
        pq.write_table(_reindex(t, task, new_ep, new_idx),
                       str(out / "meta" / DATE / f"{new_ep}.parquet"),
                       compression="zstd")

        entry = {"new_episode": new_ep, "episode_index": new_idx, "task": task,
                 "n_frames": t.num_rows,
                 "duration_s": round(t.num_rows / 30.0, 3),   # the convention every published index uses
                 "source_recording": f"{task}/{DATE}/{old_ep}",
                 "src": a["src"]}
        if "source_h5_frame" in t.column_names:
            c = t.column("source_h5_frame")
            entry["source_h5_range"] = [c[0].as_py(), c[-1].as_py()]
        if a["src"] == "cut":
            vsrc = CUT_ROOT / task / "videos" / DATE / old_ep
            vdst = out / "videos" / DATE / new_ep
            vdst.mkdir(parents=True, exist_ok=True)
            for s in VIDEO_STREAMS:
                f = vsrc / f"{s}.mp4"
                if f.is_file():
                    shutil.copy2(f, vdst / f"{s}.mp4")
            entry["videos"] = sorted(p.name for p in vdst.iterdir())
            # The preview was rendered under the segment's own name; it follows
            # the episode to its new number, or the folder would index a
            # preview nobody can match to an episode.
            psrc = CUT_ROOT / task / "previews" / DATE / f"{old_ep}.mp4"
            if psrc.is_file():
                pdst = out / "previews" / DATE / f"{new_ep}.mp4"
                pdst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(psrc, pdst)
                entry["preview"] = pdst.name
        manifest.append(entry)
    (out / "_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/media/yxma/Disk1/twm/publish/validation")
    ap.add_argument("--start-index", type=int, default=3)
    a = ap.parse_args()
    from huggingface_hub import hf_hub_download
    m = stage(Path(a.out), a.start_index,
              lambda p: hf_hub_download("yxma/React", p, repo_type="dataset"))
    for e in m:
        print(f"  {e['new_episode']}  {e['task']:12s} {e['n_frames']:6d} 帧 "
              f"({e['duration_s']:7.1f}s)  <- {e['source_recording']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
