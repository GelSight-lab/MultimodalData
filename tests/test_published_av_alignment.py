"""The published video and the published columns must agree in time.

This is the check that would have caught the 2026-09-12 preview defect. The
release itself was self-consistent; the renderer read raw H5 frames while the
force beside it came from `source_frame` (~N+2), and nothing compared the two
artefacts a USER actually pairs up.

`test_force_names_its_frame` proves force <-> source_frame <-> raw H5 bit
exactly, and never opens a published mp4. This is the other half.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "twm" / "scripts"))

from check_av_alignment import (MIN_CORR, MIN_SHARPNESS,  # noqa: E402
                                best_lag, calibrate, episode_lag,
                                video_motion)

CUT = Path("/media/yxma/Disk1/twm/release_cut")


# ── the instrument, before its readings ──────────────────────────────────────

def test_the_sign_convention_is_calibrated_not_assumed():
    """A known +2 delay must report +2.

    This was labelled backwards on the first pass, which turned "force leads"
    into "force lags" in a bug report.
    """
    assert calibrate() == 2


def test_a_lead_reports_negative():
    x = np.cumsum(np.random.default_rng(1).normal(size=300))
    y = np.roll(x, -3); y[-3:] = x[-1]
    k, _, _ = best_lag(x, y)
    assert k == -3


def test_aligned_signals_report_zero():
    x = np.cumsum(np.random.default_rng(2).normal(size=300))
    assert best_lag(x, x.copy())[0] == 0


def test_noise_is_not_read_as_a_lag():
    """A window with no contact has nothing to align; the correlation says so.

    An earlier force-lag gate fired on a segment correlating at 0.06.
    """
    rng = np.random.default_rng(3)
    _, corr, _ = best_lag(rng.normal(size=300), rng.normal(size=300))
    assert abs(corr) < MIN_CORR


# ── the published tree ───────────────────────────────────────────────────────

def _episodes(task: str, limit: int = 3):
    root = CUT / task
    meta = root / "meta"
    if not meta.is_dir():
        return []
    out = []
    for d in sorted(meta.iterdir()):
        for p in sorted(d.glob("episode_*.parquet")):
            if (root / "videos" / d.name / p.stem / "tactile_left.mp4").is_file():
                out.append((root, d.name, p.stem))
    return out[:limit]


@pytest.mark.parametrize("task", ["motherboard", "pushT", "rope"])
def test_published_video_matches_published_columns(task):
    eps = _episodes(task)
    if not eps:
        pytest.skip(f"{task}: no local published tree to check")
    checked = 0
    for root, date, ep in eps:
        for side, column in (("left", "tactile_left_intensity"),
                             ("left", "force_left_normal_n")):
            got = episode_lag(root, date, ep, side, column)
            if got is None:
                continue
            lag, corr, margin = got
            if corr < MIN_CORR or margin < MIN_SHARPNESS:
                # No contact, or a curve too flat to resolve one frame. A slow
                # press genuinely peaks by ~0.003, which is nothing.
                continue
            checked += 1
            assert lag == 0, (
                f"{task}/{date}/{ep} tactile_{side}.mp4 vs {column}: "
                f"lag {lag:+d} (r={corr:.3f}, margin={margin:.4f}) — the "
                f"published video and the published column disagree by "
                f"{abs(lag)} frame(s)")
    if not checked:
        pytest.skip(f"{task}: no window with enough contact to measure")


def _sharp_window():
    """An episode whose correlation peak is actually resolvable, or None.

    Not simply the first episode: a slow press gives a flat curve on which no
    shift of any size is visible, so testing the gate there proves nothing.
    """
    import pyarrow.parquet as pq
    for task in ("motherboard", "rope", "pushT"):
        for root, date, ep in _episodes(task, limit=6):
            t = pq.read_table(str(root / "meta" / date / f"{ep}.parquet"),
                              columns=["tactile_left_intensity"])
            sig = np.asarray(t.column(0).to_numpy(), float)
            span = 200
            if len(sig) < span + 60:
                continue
            act = np.array([sig[i:i + span].std()
                            for i in range(0, len(sig) - span, 25)])
            lo = max(0, int(np.argmax(act)) * 25 - 40)
            try:
                motion = video_motion(
                    root / "videos" / date / ep / "tactile_left.mp4",
                    lo, lo + span)
            except Exception:
                continue
            n = min(len(motion), span)
            k, r, m = best_lag(motion[:n], sig[lo:lo + n])
            if k == 0 and m >= MIN_SHARPNESS and r >= 0.8:
                return motion[:n], sig, lo, n, f"{task}/{date}/{ep}"
    return None


def test_the_gate_catches_a_real_two_frame_shift():
    """It must be able to FAIL, on the size of defect that was reported.

    A gate whose green has never been contrasted with a red is not a gate --
    `pipeline_guard` spent its whole life reporting 14 clean checks, two of
    which could not fail.
    """
    w = _sharp_window()
    if w is None:
        pytest.skip("no published window with a resolvable correlation peak")
    motion, sig, lo, n, name = w
    k2, r2, m2 = best_lag(motion, sig[lo + 2:lo + 2 + n])
    assert k2 == -2, f"{name}: a 2-frame shift read as {k2}"
    assert m2 >= MIN_SHARPNESS, f"{name}: the shift was called unresolvable"
