"""Does the published VIDEO agree with the published COLUMNS, in time?

`test_force_names_its_frame` already proves force <-> `source_frame` <-> the
raw H5, bit-exactly. It never opens a published mp4, so it cannot see the
defect found on 2026-09-12: the preview renderer displayed
`gelsight/<side>/frames[N]` while the force beside it came from
`source_frame[N]`, about N+2. Force led the tile it was drawn against by two
frames. Everything published was self-consistent; the renderer was not, and no
check compared the two artefacts a USER pairs up.

So this measures the published artefacts against each other and nothing else:
decode `videos/<date>/<ep>/tactile_<side>.mp4`, reduce each frame to how far it
has deformed from a quiet reference, and cross-correlate that against the
parquet's own `tactile_<side>_intensity` / `force_<side>_normal_n`. Aligned
data peaks at lag 0.

TWO THINGS THIS GETS WRONG IF DONE CASUALLY, both learned the hard way:

* **The sign.** Reported lag was labelled backwards on the first pass, which
  turned "force leads" into "force lags" in a bug report. `calibrate()` exists
  so the convention is asserted against a synthetic known delay rather than
  reasoned about.
* **Reading a lag off noise.** A window with no contact has nothing to align;
  a force-lag gate earlier in the same project fired on a segment whose
  correlation was 0.06 (force mean 0.078 N). Below `MIN_CORR` the answer is
  "unmeasurable", never "0".
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

MIN_CORR = 0.25          # below this the window has no contact to align
# Below this margin over the NEIGHBOURING lag, the peak is not resolved and the
# reported lag is noise. Measured on published episodes: a slow press gives a
# genuinely flat curve (rope/episode_000_seg02 peaked at -1 by 0.0033 over
# lag 0, which is nothing), while a real 2-frame offset peaked by 0.011. The
# honest answer on a flat curve is "unmeasurable", not "0" and not "-1".
MIN_SHARPNESS = 0.005
MAX_LAG = 6


def lag_profile(x, y, max_lag: int = MAX_LAG) -> dict[int, float]:
    """Correlation at every lag in [-max_lag, max_lag]."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    x = x - x.mean(); y = y - y.mean()
    out = {}
    for k in range(-max_lag, max_lag + 1):
        xa, ya = (x[:len(x) - k], y[k:]) if k >= 0 else (x[-k:], y[:len(y) + k])
        s = xa.std() * ya.std()
        out[k] = float((xa * ya).mean() / s) if s > 0 else 0.0
    return out


def best_lag(x, y, max_lag: int = MAX_LAG) -> tuple[int, float, float]:
    """Lag of `y` relative to `x`, its correlation, and the peak's margin.

    POSITIVE means y is LATER than x. Verified by `calibrate()`, not by
    reading this sentence.

    The margin is how far the peak stands above its neighbours -- but only
    once `falls_away` has confirmed there is a peak at all. A maximum that
    does not fall away on both sides gets margin 0, which MIN_SHARPNESS reads
    as "unmeasurable". That covers both ways the curve can fail to peak:

    * **A maximum at +/-max_lag.** One neighbour was never observed, so the
      curve may still be rising outside the search. A slow press ramps
      monotonically across the whole window, and pushT/2026-09-12 gave five
      false `-6`s -- each past MIN_SHARPNESS, because a ramp's per-step slope
      (0.010) beats the margin even though nothing peaks.
    * **A ripple on a plateau.** The neighbour margin is a local test, so a
      0.007 bump on a curve that spans 0.025 passes it. See `falls_away`.
    """
    out = lag_profile(x, y, max_lag)
    k = max(out, key=out.get)
    margin = out[k] - max(out[k - 1], out[k + 1]) if falls_away(out, k) else 0.0
    return k, out[k], margin


def falls_away(profile: dict[int, float], k: int, width: int = 2) -> bool:
    """Does the correlation drop for `width` steps on BOTH sides of `k`?

    An alignment peak is a peak: shift either way and the two signals agree
    less. This is what separates one from a ripple on a plateau, and the
    neighbour margin alone cannot -- it is a local test, so any 0.007 bump
    passes it.

    Measured on pushT/2026-09-12/episode_001_seg00, whose left hand never
    presses harder than 1.17 N. Video against `force_left_normal_n` gave

        -8:0.316 -7:0.318 -6:0.324 -5:0.329 -4:0.330 -3:0.329 -2:0.341
        -1:0.334 +0:0.333 +1:0.314 ... +8:0.151

    -- flat from -8 to 0 within 0.025, and the gate called it "lag -2,
    margin 0.0067". It is not a peak: -4 (0.330) stands above -3 (0.329), so
    the curve does not fall away. The same episode's video against
    `tactile_left_intensity` DOES peak, cleanly, at 0.
    """
    for side in (-1, 1):
        prev = profile[k]
        for step in range(1, width + 1):
            j = k + side * step
            if j not in profile or profile[j] >= prev:
                return False
            prev = profile[j]
    return True


def calibrate() -> int:
    """Delay a signal by a known 2 and confirm `best_lag` reports +2."""
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(size=400))
    y = np.roll(x, 2); y[:2] = x[0]
    k, _, _ = best_lag(x, y)
    return k


def video_motion(mp4: Path, lo: int, hi: int, quiet: int = 25) -> np.ndarray:
    """Per-frame deformation of a published video over [lo, hi)."""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(mp4)],
        capture_output=True, text=True)
    w, h = (int(v) for v in probe.stdout.strip().split(",")[:2])
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(mp4),
         "-vf", f"select='between(n\\,{lo}\\,{hi - 1})'", "-vsync", "0",
         "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
        capture_output=True).stdout
    n = len(raw) // (w * h * 3)
    if n < quiet + 20:
        raise RuntimeError(f"{mp4.name}: decoded {n} frames, too few to align")
    v = np.frombuffer(raw[:n * w * h * 3], np.uint8).reshape(n, h, w, 3).astype(np.float32)
    return np.abs(v - v[:quiet].mean(0)).mean(axis=(1, 2, 3))


def episode_lag(root: Path, date: str, episode: str, side: str,
                column: str) -> tuple[int, float, float] | None:
    """Lag between the published tactile video and a published column.

    None when the episode has no window with enough contact to measure.
    """
    import pyarrow.parquet as pq
    t = pq.read_table(str(root / "meta" / date / f"{episode}.parquet"),
                      columns=[column])
    sig = np.asarray(t.column(column).to_numpy(), float)
    if len(sig) < 150:
        return None
    # the most active 200-frame window, with quiet frames ahead of it
    span = 200
    act = np.array([sig[i:i + span].std() for i in range(0, len(sig) - span, 25)])
    if not len(act) or act.max() == 0:
        return None
    lo = max(0, int(np.argmax(act)) * 25 - 40)
    hi = min(len(sig), lo + span)
    if hi - lo < 120:
        return None
    mp4 = root / "videos" / date / episode / f"tactile_{side}.mp4"
    motion = video_motion(mp4, lo, hi)
    m = min(len(motion), hi - lo)
    return best_lag(motion[:m], sig[lo:lo + m])
