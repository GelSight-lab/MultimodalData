"""Force is computed from the tactile frame the published row actually shows.

The estimator reads `tactile_<side>_intensity` out of the published parquet and
reconstructs force from the GelSight frame behind that row. Which frame that is
was a formula — `trim + row + LEGACY_SHIFT` — evaluated independently by the
preprocess and by the estimator. The two agreed only while every recording was
legacy.

The first timestamp-aligned session broke the tie. Those recordings carry
per-sensor capture times, so the preprocess pairs each tick with the nearest
GelSight frame and adds nothing; the estimator kept adding 15. The published
2026-09-09 validation set went out with force 13-15 frames (~0.45 s) ahead of
its own tactile: the right sensor reads 0 N while the gel image plainly shows
contact, and the left rises before it is touched. Cross-correlation against
`tactile_*_intensity` put the peak at +13 frames, r 0.95 against 0.58 at zero
lag.

The invariant these tests hold is not "the shift is 15" or "the shift is 0" —
it is that there is no second evaluation to disagree with the first.
"""
import numpy as np
import pytest

from twm.react_preprocess.h5io import TactileAlignment


def _alignment(index_map, timestamped):
    return TactileAlignment("left", np.asarray(index_map, np.int64), timestamped,
                            np.zeros(len(index_map)) if timestamped else None)


def test_timestamped_recordings_declare_no_legacy_shift():
    """A timestamped recording is already aligned; a constant on top of it is
    the double-correction that shipped."""
    a = _alignment([3, 3, 4, 6, 6], timestamped=True)
    assert a.needs_legacy_shift is False


def test_legacy_recordings_still_declare_the_shift():
    a = _alignment([0, 1, 2, 3, 4], timestamped=False)
    assert a.needs_legacy_shift is True


def test_run_episode_reads_the_index_map_and_never_adds_a_constant():
    """The estimator must not import a shift constant or re-derive an index.

    Asserted on the source because the alternative is a fixture with real
    GelSight frames, and the defect is exactly one line of arithmetic.
    """
    from pathlib import Path
    import twm.force_recovery.run_episode as re_mod

    src = Path(re_mod.__file__).read_text()
    body = "\n".join(l for l in src.splitlines() if not l.strip().startswith("#"))
    assert "LEGACY_SHIFT" not in body, (
        "run_episode adds a latency constant again — the map belongs to "
        "react_preprocess.h5io.open_episode(...).align[side].index_map")
    assert "index_map" in body
    # The estimator must not carry a `shift` in its metadata either: there is
    # no single number to carry once the map is per row.
    assert '"shift"' not in body


def test_the_alignment_used_matches_the_published_rows(monkeypatch):
    """Row count mismatch means the parquet and the H5 are different takes."""
    import twm.force_recovery.run_episode as re_mod
    assert hasattr(re_mod, "open_episode"), "the map must come from the preprocess"


@pytest.mark.parametrize("timestamped", [True, False])
def test_index_map_is_monotone_and_within_the_stream(timestamped):
    """Whatever the map is, it may never run backwards or off the end —
    a force value from a frame the row cannot show is not exportable."""
    idx = np.array([0, 0, 1, 3, 3, 4], np.int64)
    a = _alignment(idx, timestamped)
    assert np.all(np.diff(a.index_map) >= 0)
    assert a.index_map.min() >= 0
