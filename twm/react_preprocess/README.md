# react_preprocess — building the React release

The producer side of the dataset. `toolbox/` (`react_toolbox`) reads the
published data; this package is what turns raw rig recordings into it, and
ships alongside the data so the release is reproducible.

```bash
python -m react_preprocess build --task pushT [--with-depth]
python -m react_preprocess audit --task pushT
python -m react_preprocess backfill-flags --task pushT
python -m react_preprocess verify-flags --task pushT --against h5
```

## Pipeline

For the v8 normal-force migration, follow the
[force reprocessing runbook](../force_recovery/RUNBOOK.md) before rebuilding
segments. Force estimation covers 0-15 N; force-informed target export has a
separate stiffness-policy check. Do not treat a raw tactile rebuild as a force
recalibration or reuse cached force previews.

```
recording.h5
   │  h5io      read timestamps/poses, resolve tactile↔camera alignment
   │  tactile   pass 1: pick the p01 no-contact reference
   │            pass 2: contact metrics + new-frame flags + encode
   │  encode    H.264 yuv444p CRF18 (RGB), FFV1 gray16le (depth)
   │  meta      per-frame parquet
   ▼
release/<task>/{videos,depth,meta}/<date>/episode_NNN/…      (the master)
   │  curation  find the defective spans, and their clean complement
   │  segment   cut, LAST — after force recovery and the Z-up conversion
   ▼
release_cut/<task>/{videos,meta}/<date>/episode_NNN_segNN/…  (what ships)
```

| Module | Responsibility |
|---|---|
| `config` | paths, camera mapping, encoding and contact constants |
| `h5io` | source reading, pose alignment, **tactile time alignment** |
| `contact` | contact metrics, p01 reference, duplicate-frame detection |
| `encode` | ffmpeg writers |
| `tactile` | two-pass GelSight processing |
| `meta` | parquet assembly and index columns |
| `detect` | bad-interval detectors + clean-span complement |
| `curation` | per-task `bad_frames.json` / `segments.json` / `episodes.jsonl` |
| `segment` | cut episodes down to their clean spans, one span per episode |
| `previews` | preview policy (calibration choice, trim, world offset, layout) |
| `pipeline` | per-episode orchestration |
| `backfill` | recover flags for already-published parquet |
| `publish` | mirror data + code to the Hub |

`previews` holds policy only — the panel renderer needs rig-local calibration
the release does not ship, so it stays in `twm/scripts/build_release_previews.py`
as a thin adapter over `previews.plan()`. Port checks: `detect`/`curation`
reproduced the published `bad_frames.json` and `segments.json` for all 36
episodes with zero differences; the preview adapter re-renders
`pushT/episode_000` bit-identically (first-frame MAD 0.00, same 900 frames).

`tactile_freeze_*` (added 2026-09-11) deliberately breaks that zero-difference
property: it flags a defect the published files do not contain. See below.

## Defects are cut out, not annotated

`curation` describes the defects; it does not remove them. That was the whole
release's model until 2026-09-11, and its weakness is that the description
lives in a sidecar: a reader who loads the parquet and the MP4s and never opens
`bad_frames.json` trains on frozen tactile and teleported poses without ever
being told. Nine of the nineteen 2026-09 episodes carry such a span.

`segment` makes it structural. Every clean span long enough to be a
demonstration becomes its own published episode, so each frame that ships has
passed every detector and there is no annotation left to skip — what was
`bad_frames.json` becomes the gaps between episodes. `release/` stays as the
uncut master; `release_cut/` is what goes to the Hub.

It runs LAST, after force recovery and the Z-up conversion, so every column
those add is carried through by the same row slice and neither has to know
segments exist.

Measured over the 2026-09 sessions: 132.2 min becomes 122.5 min in 38
episodes. 4.7% was defective; 2.6% was clean but under the 30-second floor
(`MIN_PUBLISH_SECONDS`). Span lengths are strongly bimodal — 19 slivers under
1.3 s between nearby defects, then a jump to 2.5 s, 11.6 s and up — so the
floor costs little and removes every fragment too short to be a demonstration.

A row keeps `timestamp` and `source_h5_frame` unrebased, because both stay
true of a slice, and gains `source_episode` / `source_frame_idx`, so a segment
traces back to the recorded frame it came from without an index lookup.

The cut is routed in Python off the decoder rather than by an ffmpeg filter
graph. `select=between(n,a,b),setpts=N/FRAME_RATE/TB` is **not** frame-exact:
a 120-frame probe asked for `[10,29]` returned twenty frames whose contents
were 10, 10, 12, 13, … — right count, wrong pixels, which is exactly the
failure this stage exists to remove and exactly what a count check cannot see.

## A held GelSight frame is not missing data

The recorder repeats a sensor's last frame at every tick until a new one
arrives, so a GelSight that stops delivering produces no gap and no error — it
produces the same frame, and the same metrics, over and over. Nothing in
curation read that until `tactile_freeze_L` / `tactile_freeze_R`, which find
runs of bit-identical tactile intensity the way `ot_loss_*` finds runs of
bit-identical pose.

The threshold is the shared `FREEZE_THRESHOLD_S` (0.25 s, 8 ticks), and the
margin around it is wide because the two populations are far apart. A GelSight
runs at 15-18 Hz against a 30 Hz tick, so short repeat runs are the normal
state of every episode: measured 1-3 ticks, worst case 4. Every genuine
outage on the pushT 2026-09-10 session was 144 ticks or longer.

Unlike the pose version this one is padded by `BUFFER_FRAMES`: the driver's
first frames after a reopen are underexposed — measured mean 55, then 73,
against a settled 75 — and they land just past the end of the freeze.

What it found:

| Data | Episodes touched | Frames added | Where |
|---|---|---|---|
| pushT 2026-09-10 (unpublished) | 3 of 3 | 6,233 (17 %) | mid-episode sensor dropouts, 5 s each, plus two dead tails of 20 s and 121 s |
| motherboard (published) | 32 of 35 | 754 (0.35 %) | the last 0.7-1.1 s of the episode, both sensors |
| pushT (published) | 4 of 5 | 95 (0.16 %) | same, the last 0.7-1.1 s |

The published tails are a shutdown artefact: the GelSight threads stop before
the last ticks are written. Regenerating the two published index files would
trim about a second off each episode's usable span. That has not been done —
it changes files already on the Hub.

## Tactile time alignment

How a GelSight frame is paired with a camera frame depends on the recording:

| | legacy (≤ 2026-06-18) | timestamped (2026-06-27 →) |
|---|---|---|
| Pairing | by tick index | nearest capture timestamp |
| Systematic lag | ~15 frames (0.5 s) | removed at the source |
| Constant shift needed | yes | **no — would double-correct** |

The rig used to decode full 8 MP MJPG frames on the capture thread (~71 ms
each), so tactile really ran at ~8 fps while rows were written at 30 Hz. That
produced both the lag and heavy frame duplication. The rig now decodes at
reduced scale and records `gelsight/<side>/timestamps`.

`h5io.TactileAlignment.needs_legacy_shift` is the guard: it is False for
timestamped recordings, so a latency shift can never be applied twice.

## `tactile_*_is_new`

The GelSight Mini tops out at 18.75 fps while parquet rows are written at
30 Hz, so some rows necessarily repeat the previous tactile frame. Legacy
recordings repeat far more. These boolean columns mark the rows that are
genuinely fresh readings:

```python
df = pq.read_table("episode_000.parquet").to_pandas()
fresh = df[df.tactile_left_is_new]        # train tactile dynamics on these
```

Measured — legacy over the whole published release (480 080 rows, 36 episodes),
fixed rig over `test/2026-06-29/episode_001` (1 294 rows, both sensors):

| | capture rate | duplicate rows | effective rate | longest frozen run |
|---|---|---|---|---|
| legacy | ~8 fps (decode-bound) | 71.8 % | 8.5 fps | 30 frames (1.0 s) |
| fixed rig | 19.3 fps | 39.5–41.5 % | 17.6–18.2 fps | 5 frames (0.17 s) |

The residual ~40 % on the fixed rig is irreducible: a 19 fps sensor sampled onto
a 30 Hz row clock must repeat roughly a third of its rows. Only the legacy
excess above that was a bug.

**How the flags are recovered for already-published data.** A repeated frame
yields a bit-identical contact triple, so a row is fresh exactly when its
triple differs from the previous row's — no video decode required.
`verify-flags --against h5` checks this against source pixels: on all seven
audited episodes (4 pushT + 3 motherboard) it reproduces the source
frame-by-frame with **0 mismatches in 899 frames each**, and independently
recovers the +15 shift baked into the release (every other offset in 0..20
disagrees on >33 % of frames, so the detection is unambiguous).

Checking against the published MP4s can only ever be approximate — H.264 is
lossy, so a duplicated frame does not decode back to identical pixels.
`--against video` therefore compares with a tolerance and reports the observed
separation rather than asserting exactness.

## Conventions

- Frame `i` of every MP4 == parquet row `i` == source frame `trim_offset + i`
- Cameras: `cam0 → view_right`, `cam1 → view_left`, `cam2 → view_middle`
  (verified against calibration serials)
- Tactile frames are RGB in HDF5 and converted to BGR only for ffmpeg
- Depth is uint16 millimetres, `0` = no return
- The 2026-05-19 motherboard world-origin offset is baked into stored poses

Paths come from `REACT_DATA_ROOT` / `REACT_STAGE_ROOT` when set, so the
package runs off the rig.
