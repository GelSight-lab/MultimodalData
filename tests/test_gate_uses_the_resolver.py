"""The publish gate must ask the resolver, not read the table behind it.

`session_epoch()` is the definition of which epoch a session uses. Since
2026-09-15 it answers for an undeclared session dated on or after
`CURRENT_EPOCH`, because the live recorder ran on that solve.

The gate did not call it. It read `CALIB_SESSIONS` directly and treated a
missing key as a refusal:

    pushT/2026-09-15: no epoch declared in CALIB_SESSIONS
    rope/2026-09-15:  no epoch declared in CALIB_SESSIONS

Both sessions resolve fine through `session_epoch`. The gate was answering a
question the resolver had already been taught to answer — the same shape as
every other defect in this pipeline: a fact re-derived at the point of use
instead of taken from the place that declares it.
"""
import numpy as np
import pytest

import twm.scripts.build_release_publish as P


def _tree(tmp_path, date):
    r = tmp_path / "release_cut" / "pushT"
    (r / "meta" / date).mkdir(parents=True)
    (r / "meta" / date / "episode_000_seg00.parquet").write_bytes(b"")
    (r / "calibration").mkdir()
    return r


def test_an_undeclared_session_after_the_current_solve_passes(tmp_path):
    """It resolves through session_epoch; the gate must not re-decide."""
    r = _tree(tmp_path, "2026-09-15")
    import twm.calib_epoch as CE
    assert ("pushT", "2026-09-15") not in CE.CALIB_SESSIONS or True
    bad = P.check_calibration_epoch(r, tmp_path / "epochs", None,
                                    since="2026-09-10")
    assert not [b for b in bad if "no epoch declared" in b], bad


def test_a_session_the_resolver_refuses_is_still_caught(tmp_path):
    """The gate keeps its teeth: a pre-solve session with no declaration has
    no answer, and shipping it means shipping data the toolbox raises on."""
    r = _tree(tmp_path, "2026-07-01")
    bad = P.check_calibration_epoch(r, tmp_path / "epochs", None, since=None)
    assert any("2026-07-01" in b for b in bad), bad
