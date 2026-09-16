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

Measured: **7.8 GB read per sensor-side** against ~4.2 GB of compressed gel
bytes actually needed -- roughly **1.9x read amplification**, all of it
readahead fetching data no one asked for.

Adding workers multiplies the seeking rather than the throughput, which is
exactly the 1.08x above. Fewer workers would not help either: one worker
already achieved 53 of the 57 rows/s that four achieve.

## What was NOT done, and why

Not patched mid-run. The runbook archives the code checkout alongside the run
so every NPZ is attributable to known code; swapping the reader halfway would
split the output across two versions and require re-freezing the inventory.
The run was left to finish (~6 h) rather than made faster and less traceable.

## Worth trying next time, in order of expected value

1. `posix_fadvise(POSIX_FADV_RANDOM)` on the H5 before walking gel frames.
   It suppresses the readahead that is fetching the RealSense chunks, and on
   these numbers that is the whole 1.9x. Cheapest change by far; verify the
   read volume per side drops toward 4.2 GB before trusting it.
2. Record gel to its own file, or write each dataset contiguously. Removes the
   interleaving at the source rather than working around it.
3. Read both sides in ONE pass. Left and right are currently separate jobs, so
   every file is traversed twice; one traversal serving both halves the
   striding. This is the same change that made encoding single-pass.

## What does NOT follow from this

MP4 is still not the input. The 2026-09-16 A/B measured 132.4 ms/frame from H5
against 134.6 ms from MP4 and read that as "no speed advantage" -- but that
comparison ran on a WARM cache and measured reconstruction, not I/O, so it
does not describe this run, which is I/O bound and where MP4 would in fact be
far cheaper to read. The reason to stay on H5 is accuracy, not speed: the same
A/B found a median difference of 0.0226 N (0.16%) but a maximum of 6.48 N
(86%) on a single frame, and the runbook is explicit that missing raw data is
"a blocker, not permission to estimate from recompressed preview videos".
