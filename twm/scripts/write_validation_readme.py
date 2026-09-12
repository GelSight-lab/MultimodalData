"""README for the merged `data/validation`, generated from its own indices.

Written from `episodes.jsonl` rather than by hand so the table cannot drift
from the folder: the counts a reader sees are the counts the loader gets.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

TEMPLATE = """# validation — the 2026-09-09 calibration epoch

Everything recorded on 2026-09-09 lives here, kept out of `data/motherboard`
and `data/pushT` because it does not share those trees' calibration epoch.
Same rig, same processing pipeline; a separate folder so nothing silently
inherits the wrong extrinsics. `calibration/` holds the epoch solved that
morning at 07:41 (`calib_points_20260909_074139.json`).

{n} episodes, {minutes:.1f} minutes, two tasks.

| episode | task | frames | duration | segments | bad frames | source recording |
|---|---|---|---|---|---|---|
{table}

## Episode numbers do not identify a recording

The recorder reuses episode numbers within a date. `motherboard/2026-09-09/episode_000.h5`
was written **four times** that day, each write replacing the last.
`episode_000`–`episode_002` here are the 04:00 session; `episode_004`–`episode_008`
are the 17:20/17:41 session — different recordings that happened to be given
the same numbers.

So the arrivals were renumbered onto the end of the folder rather than merged
by name, and every row of `episodes.jsonl` carries `source_recording`, naming
the file it actually came from. `task` is likewise per-episode now:
`segments.json` and `bad_frames.json` used to carry one task at the top, and
this folder holds two.

`episode_003` was previously published as `data/pushT_2026-09-09/episode_000`
and is the same data, renumbered. That path has been removed.

## Two cutting policies

| episodes | floor | how they were published |
|---|---|---|
| 000–003 | 16 frames | whole episodes; `segments.json` indexes the clean spans inside them |
| 004–008 | 900 frames (30 s) | already cut — each file **is** one clean span, so it carries no defects |

For 000–003 you must honour `segments.json`: those videos contain flagged
frames. For 004–008 you need not — the flagged frames are the gaps *between*
episodes, which is why their coverage of the recording is not contiguous.

`data/pushT_2026-09-09/segments.json` shipped `n_segments: 79` over an empty
list while its own `episodes.jsonl` said 62. Recomputing the clean spans from
`bad_frames.json` at the documented 16-frame floor reproduces 62 spans and
1147 bad frames exactly, so the merged file carries the rebuilt list.

## Format

`meta/<date>/<episode>.parquet`, `videos/<date>/<episode>/*.mp4` — three
RealSense views (`view_left`, `view_middle`, `view_right`), two GelSights
(`tactile_left`, `tactile_right`) and the two Arducam wrist cameras
(`wrist_left`, `wrist_right`). Poses are Z-up.

`duration_s` throughout is `n_frames / 30`, the nominal write tick. The
RealSense streams actually run at 29.80 Hz and the GelSights reach about
17.8 Hz against that tick, so roughly 40 % of tactile frames repeat the
previous one — `tactile_*_is_new` marks which are real.

Two gaps worth knowing:

* `depth/` covers `episode_000`–`episode_003` only. The 17:20/17:41 session
  was published without depth.
* `previews/` covers every episode except `episode_003`: the preview panel is
  rendered from the source recording, and that H5 has been deleted.
"""


def render(rows: list[dict]) -> str:
    table = "\n".join(
        f"| {r['episode'].split('/')[-1]} | {r['task']} | {r['n_frames']} | "
        f"{r['duration_s']:.1f} s | {r['n_segments']} | {r['total_bad_frames']} | "
        f"`{r['source_recording']}` |" for r in rows)
    return TEMPLATE.format(n=len(rows), table=table,
                           minutes=sum(r["duration_s"] for r in rows) / 60)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/media/yxma/Disk1/twm/publish/validation")
    a = ap.parse_args()
    out = Path(a.out)
    rows = [json.loads(l) for l in
            (out / "episodes.jsonl").read_text().strip().splitlines()]
    (out / "README.md").write_text(render(rows))
    print(f"  README.md  {len(rows)} 集  "
          f"{sum(r['duration_s'] for r in rows)/60:.1f} 分钟")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
