"""A GelSight that drops out is held, not dropped, and nothing caught it.

The recorder repeats a sensor's last frame at every tick until a new one
arrives, so an outage looks like a run of bit-identical tactile readings —
exactly how OptiTrack track loss looks in the pose, which `ot_loss_*` has
always flagged. Until this detector there was no `tactile_freeze_*`, so the
pushT 2026-09-10 session published a left GelSight that was frozen for 5 s
nine times over and then dead for the last 122 s of an episode, and every one
of those frames counted as training data.

Measured on that session (`gelsight/left` hold-run lengths, ticks):
normal holds at the sensor's native 15-18 Hz are 1-3 long, worst case 4;
every outage is >= 144. The 8-frame threshold (0.25 s) sits between the two
populations with a 5x margin on the normal side and an 18x margin on the
outage side.
"""
import numpy as np

from twm.react_preprocess.detect import (
    BUFFER_FRAMES,
    FREEZE_THRESHOLD_S,
    detect_tactile_freezes,
)
from twm.react_preprocess.config import FPS


def trace(pattern):
    """An intensity trace from hold-run lengths: 3 means one new value held
    for 2 more ticks. Neighbouring runs differ so only the repeats are
    bit-identical, and every value stays under TAU_INTENSITY so the spike
    detector has nothing to say about these fixtures."""
    out = []
    for k, run in enumerate(pattern):
        out.extend([10.0 + (k % 5) * 0.25] * run)
    return np.asarray(out, np.float32)


def live_trace(T):
    """A sensor delivering normally for T frames: held every other tick."""
    return trace([2] * (T // 2 + 1))[:T]


def test_a_sensor_running_at_half_the_tick_rate_is_not_flagged():
    """15 Hz sampled at 30 Hz holds every other frame. Flagging that would
    condemn every GelSight episode ever recorded."""
    t = trace([2] * 300)
    assert detect_tactile_freezes(t, len(t)) == []


def test_the_worst_normal_hold_on_the_rig_is_not_flagged():
    """Runs of 4 occur in the real recordings when the sensor is slowest."""
    t = trace([1, 2, 3, 4, 2, 1, 4, 3])
    assert detect_tactile_freezes(t, len(t)) == []


def test_a_five_second_outage_is_flagged():
    t = trace([2] * 50 + [150] + [2] * 50)
    (a, b), = detect_tactile_freezes(t, len(t))
    assert a == 100 - BUFFER_FRAMES              # the last real frame is at 100
    assert b == 100 + 149 + BUFFER_FRAMES


def test_the_dark_frames_a_reconnect_delivers_fall_inside_the_interval():
    """The driver's first frames after a reopen are underexposed — measured
    mean 55 then 73 against a settled 75 — so the interval has to reach past
    the freeze. Three frames of buffer covers the two that were wrong."""
    t = trace([2] * 50 + [150] + [1, 1, 1] + [2] * 20)
    (_, b), = detect_tactile_freezes(t, len(t))
    warmup_first, warmup_last = 250, 252
    assert b >= warmup_last, f"warm-up frames {warmup_first}-{warmup_last} left in"


def test_a_freeze_that_runs_to_the_end_is_clipped_to_the_last_frame():
    t = trace([2] * 50 + [3623])
    (a, b), = detect_tactile_freezes(t, len(t))
    assert a == 100 - BUFFER_FRAMES
    assert b == len(t) - 1


def test_two_outages_close_together_stay_two_intervals():
    """Consecutive restarts leave a short usable span between them; merging
    them would throw that span away."""
    t = trace([2] * 50 + [150] + [2] * 40 + [150] + [2] * 20)
    assert len(detect_tactile_freezes(t, len(t))) == 2


def test_the_threshold_is_a_quarter_second_of_ticks():
    n = int(round(FREEZE_THRESHOLD_S * FPS))
    assert detect_tactile_freezes(trace([n - 1]), n - 1) == []
    assert detect_tactile_freezes(trace([n]), n) != []


def test_an_empty_or_single_frame_trace_is_handled():
    assert detect_tactile_freezes(np.zeros(0, np.float32), 0) == []
    assert detect_tactile_freezes(np.zeros(1, np.float32), 1) == []


# ── curation: the new category has to reach bad_frames.json and the segments ──

def _sidecar(tmp_path, left_intensity, right_intensity, T):
    import torch
    d = tmp_path / "pushT" / "meta" / "2026-09-10"
    d.mkdir(parents=True)
    (d / "episode_001.parquet").write_bytes(b"x")
    torch.save({
        "timestamps": torch.arange(T, dtype=torch.float64) / FPS,
        "sensor_left_pose": torch.zeros(T, 7),
        "sensor_right_pose": torch.zeros(T, 7),
        "tactile_left_intensity": torch.from_numpy(left_intensity),
        "tactile_right_intensity": torch.from_numpy(right_intensity),
        "_contact_meta": {"active_sensors": [], "tactile_timestamped": True},
    }, d / "episode_001._detect.pt")
    return d / "episode_001._detect.pt"


def test_a_frozen_left_sensor_is_reported_for_the_left_side_only(tmp_path):
    from twm.react_preprocess.curation import episode_report
    frozen = trace([2] * 40 + [150] + [2] * 10)
    T = len(frozen)
    report, _ = episode_report(_sidecar(tmp_path, frozen, live_trace(T), T))
    assert len(report["tactile_freeze_L"]) == 1
    assert report["tactile_freeze_R"] == []
    assert report["total_bad_frames"] >= 150


def test_the_frozen_span_is_cut_out_of_the_clean_segments(tmp_path):
    """What "delete those parts" means downstream: the span never appears in
    any segment, and the good data on both sides of it survives as two."""
    from twm.react_preprocess import detect as D
    from twm.react_preprocess.curation import _bad_intervals, episode_report
    frozen = trace([2] * 40 + [150] + [2] * 40)
    T = len(frozen)
    report, _ = episode_report(_sidecar(tmp_path, frozen, live_trace(T), T))
    segments = D.find_clean_segments(T, _bad_intervals(report))
    (a0, b0), (a1, b1) = segments
    assert (a0, b0) == (0, 80 - 1 - BUFFER_FRAMES)
    assert a1 == 80 + 149 + BUFFER_FRAMES + 1 and b1 == T - 1
    covered = np.zeros(T, bool)
    for a, b in segments:
        covered[a:b + 1] = True
    assert not covered[80:80 + 150].any(), "frozen frames still inside a segment"
