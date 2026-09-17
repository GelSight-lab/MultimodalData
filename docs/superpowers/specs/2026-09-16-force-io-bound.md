# The v8 force re-run is I/O bound, and worker count cannot fix it

Measured during the 2026-09-16 full re-run (56 episodes / 112 sensor-sides /
404 min of footage, `FORCE_SINCE=2026-09-10`).

## What was measured

| configuration | aggregate throughput |
|---|---|
| 1 worker (Step 3 pilot) | 53 rows/s |
| 4 disjoint workers | 57 rows/s (**1.08x**) |

Per-worker throughput collapsed from 53 to 14 rows/s. Each worker held only
~49% of a core (12 min CPU in 25 min wall), blocked the rest of the time.

Aggregate disk read, taken from `/proc/<pid>/io` rather than from averaged
`iowait`: **49 MB/s**, against a disk that does ~111 MB/s sequential.

`iowait` was misleading here -- 5-15% averaged over 8 logical CPUs looks
CPU-bound, while four processes each blocked half the time is not.

## Why

The gel frames are a minority stream inside a file dominated by cameras:

    /gelsight/{left,right}/frames   6.79 GB each   chunks (1, 480, 640, 3)
    /realsense/cam{0,1,2}/color     6.79 GB each   chunks (1, 480, 640, 3)
    /realsense/cam{0,1,2}/depth     4.53 GB each   chunks (1, 480, 640)
    -> 28.9 GB on disk, ~47 GB uncompressed, per 7371-frame episode

One frame per chunk is the right layout for random frame access: reading
frame i reads exactly one chunk, with no chunk-granularity waste. The cost is
elsewhere. The recorder writes frame-by-frame across every dataset, so gel
frame i sits on disk between RealSense frame i and depth frame i. Walking the
gel stream in order therefore strides across the entire file, and filesystem
readahead pulls in the RealSense chunks adjacent to each gel chunk.

**CORRECTION (measured later the same evening).** An earlier version of this
note claimed ~1.9x read amplification from readahead. That number was not
measured -- it came from dividing 7.8 GB read per sensor-side by a compressed
gel size ASSUMED from the file's overall compression ratio. Measured directly
with `/proc/self/io` while reading 150 gel frames strided across a file:

    decompressed          138 MB
    actually read off disk  99 MB      -> 0.72x

There is **no read amplification**. The disk delivers almost exactly the
compressed bytes asked for, and `read_ahead_kb` on this volume is 2048 -- so
readahead is not fetching waste, it simply cannot bridge the gap either.

The cost is SEEKS, not wasted bytes. `rotational = 1`: this is a mechanical
disk. Consecutive gel frames sit ~6.4 MB apart (47.5 GB uncompressed over 7371
frames, six other streams between them), so every frame is its own seek and a
2 MB readahead spans none of it.

Adding workers multiplies the seeking rather than the throughput, which is
exactly the 1.08x above. Fewer workers would not help either: one worker
already achieved 53 of the 57 rows/s that four achieve.

## What was NOT done, and why

Not patched mid-run. The runbook archives the code checkout alongside the run
so every NPZ is attributable to known code; swapping the reader halfway would
split the output across two versions and require re-freezing the inventory.
The run was left to finish (~6 h) rather than made faster and less traceable.

## Worth trying next time, in order of expected value

1. **Read both sides in ONE pass — measured 1.34x.** Left and right are
   separate jobs today, so every file is traversed twice. But gel-left frame i
   and gel-right frame i were written at the same moment and sit adjacent on
   disk, so one seek can serve both. Measured on a real recording, 120 frames:

       left only          14.2 s   118.6 ms/frame
       left + right       21.3 s    88.7 ms/frame per side   -> 1.34x

   This is the same argument that made encoding single-pass.
2. Record gel to its own file, or write each dataset contiguously. Removes the
   interleaving at the source, which is what makes the seeks long.
3. NOT `posix_fadvise(POSIX_FADV_RANDOM)`. An earlier version of this note put
   it first, on the strength of an amplification figure that turned out to be
   an assumption rather than a measurement. There is no readahead waste to
   suppress; see the correction above.

## What does NOT follow from this

MP4 is still not the input. The 2026-09-16 A/B measured 132.4 ms/frame from H5
against 134.6 ms from MP4 and read that as "no speed advantage" -- but that
comparison ran on a WARM cache and measured reconstruction, not I/O, so it
does not describe this run, which is I/O bound and where MP4 would in fact be
far cheaper to read. The reason to stay on H5 is accuracy, not speed: the same
A/B found a median difference of 0.0226 N (0.16%) but a maximum of 6.48 N
(86%) on a single frame, and the runbook is explicit that missing raw data is
"a blocker, not permission to estimate from recompressed preview videos".
