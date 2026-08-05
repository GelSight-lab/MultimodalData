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

```
recording.h5
   │  h5io      read timestamps/poses, resolve tactile↔camera alignment
   │  tactile   pass 1: pick the p01 no-contact reference
   │            pass 2: contact metrics + new-frame flags + encode
   │  encode    H.264 yuv444p CRF18 (RGB), FFV1 gray16le (depth)
   │  meta      per-frame parquet
   ▼
release/<task>/{videos,depth,meta}/<date>/episode_NNN/…
```

| Module | Responsibility |
|---|---|
| `config` | paths, camera mapping, encoding and contact constants |
| `h5io` | source reading, pose alignment, **tactile time alignment** |
| `contact` | contact metrics, p01 reference, duplicate-frame detection |
| `encode` | ffmpeg writers |
| `tactile` | two-pass GelSight processing |
| `meta` | parquet assembly and index columns |
| `pipeline` | per-episode orchestration |
| `backfill` | recover flags for already-published parquet |
| `publish` | mirror data + code to the Hub |

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
