"""THE definition of how a camera tick maps onto a GelSight frame.

Why this module exists
----------------------
The GelSight capture lags the camera stream on legacy recordings. That single
fact was implemented independently in four places, with three different values:

    run_episode.LEGACY_SHIFT                       15
    build_latency_correction_clips.SHIFT           15   (a second copy)
    play_react_pt   --tactile_latency  default      3   (contradicts the above)
    latency_align_viewer --latency     default      0   (a fourth value)

and a fifth consumer — the episode preview builder — applied **no** shift at
all, so its tactile tiles ran half a second ahead of the camera views beside
them. That shipped to the dataset repo before anyone noticed.

The failure mode is not "someone typed the wrong number". It is that a fact
about the DATA lived in the CALLERS. Every new consumer had to rediscover it,
and one silently did not. So: one function, imported everywhere.

The rule (from react_preprocess.h5io.TactileAlignment.needs_legacy_shift)
------------------------------------------------------------------------
* Recording WITH per-sensor GelSight timestamps -> already aligned, shift 0.
  Applying a constant on top would DOUBLE-correct it.
* Recording WITHOUT them (legacy) -> constant LEGACY_SHIFT frame lag.

Detection is per file, never assumed, because both kinds exist in this dataset.
"""
from __future__ import annotations

# The dataset documents this itself (yxma/React tasks.json -> tactile_latency):
#   frames_estimate 15 @ 30 fps = 0.5 s
#   applies_to  "all recordings up to and including 2026-06-18"
#   fixed_in_rig 2026-06-27
#   cause  recording-side cv2.VideoCapture V4L2 buffer never flushed
#          (throttled reads + no BUFFERSIZE=1 + default pixfmt)
#   method "tactile streams shifted +15 frames (rebuilt from raw H5) so
#           tactile[i] aligns with view[i]/pose[i]"
#
# IMPORTANT — two different artefacts, two different rules:
#   * The PUBLISHED release (videos + contact scalars) is already corrected;
#     status CORRECTED_IN_DATA, "No loader compensation needed".
#   * The RAW H5 is not. Anything reading gelsight/<side>/frames straight out
#     of the H5 must apply the shift itself, which is what this module is for.
LEGACY_SHIFT = 15
RIG_FIXED_DATE = "2026-06-27"   # recordings after this need no shift
_TS_HINTS = ("time", "ts", "stamp")


def gel_lag_frames(h5file) -> int:
    """Frames the GelSight stream lags the camera stream in THIS recording.

    Returns 0 for timestamp-aligned recordings, LEGACY_SHIFT for legacy ones.
    Takes an open h5py.File so callers cannot forget to check the file they
    are actually reading.
    """
    for side in ("left", "right"):
        node = h5file.get(f"gelsight/{side}")
        if node is None:
            continue
        if any(any(h in k.lower() for h in _TS_HINTS) for k in node.keys()):
            return 0
    return LEGACY_SHIFT


def gel_index(h5file, cam_index: int, side: str = "left") -> int:
    """GelSight frame index to pair with `cam_index`, clamped to the stream."""
    n = len(h5file[f"gelsight/{side}/frames"])
    return min(int(cam_index) + gel_lag_frames(h5file), n - 1)


def describe(h5file) -> str:
    """One line for a status bar / log, so the applied correction is visible."""
    lag = gel_lag_frames(h5file)
    return f"gel+{lag}f (legacy)" if lag else "gel aligned (timestamped)"
