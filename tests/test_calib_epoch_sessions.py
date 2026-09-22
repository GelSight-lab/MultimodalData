"""A recording session declares which calibration epoch it belongs to.

The rig was recalibrated between sessions, so `task -> epoch` cannot answer
"which extrinsics does this recording need". Every motherboard recording
before 2026-06 uses the May-12 epoch; the 2026-09-09 session was recorded
with the June-26 one, and projecting it through May-12 puts the sensor dot
30-60 px off the sensor.
"""
from pathlib import Path

import pytest

from twm.calib_epoch import CALIB_SESSIONS, calib_dir, check_epoch, epoch_of


def test_no_date_keeps_the_task_default():
    """Every existing caller passes only a task; none of them may change."""
    assert epoch_of("motherboard") == "2026-05-12"
    assert epoch_of("pushT") == "2026-06-26"
    check_epoch("motherboard")
    check_epoch("pushT")


def test_a_declared_session_selects_its_own_epoch():
    # Asserted on the epoch the files carry, not on directory names: the
    # epoch directories get renamed as new ones are measured.
    assert epoch_of("motherboard", date="2026-05-10") == "2026-05-12"
    assert epoch_of("motherboard", date="2026-09-09") == "2026-09-09"
    assert epoch_of("pushT", date="2026-06-18") == "2026-06-26"
    # two sessions on one epoch resolve to the same directory
    assert calib_dir("motherboard", date="2026-05-11") == calib_dir("motherboard", date="2026-05-19")


def test_the_september_session_declares_its_own_epoch_not_the_task_default():
    """Whatever a session declares must beat the task default; the default is
    May-12 and this session is not on it."""
    assert epoch_of("motherboard", date="2026-09-09") == "2026-09-09"
    assert calib_dir("motherboard", date="2026-09-09").name != "result"


def test_epoch_and_check_follow_the_session_too():
    assert epoch_of("motherboard", date="2026-09-09") == "2026-09-09"
    assert epoch_of("motherboard", date="2026-05-19") == "2026-05-12"
    check_epoch("motherboard", date="2026-09-09")     # must not raise


def test_an_undeclared_session_raises_instead_of_falling_back():
    """Falling back to the TASK default is how a session silently ships
    through another session's extrinsics.

    The example is a date BEFORE the current solve. Since 2026-09-15 a session
    dated on or after CURRENT_EPOCH resolves to it without a table entry —
    the recorder ran on that solve, which is a fact about the rig, not a guess
    from the calendar. For anything older the calendar decides nothing, and
    pushT's 2026-06-18 belonging to the June-26 solve, measured eight days
    LATER, is why.
    """
    with pytest.raises(KeyError, match="2026-07-01"):
        calib_dir("motherboard", date="2026-07-01")


def test_every_declared_session_names_a_real_epoch_directory():
    for (task, date) in CALIB_SESSIONS:
        d = calib_dir(task, date=date)
        assert d.is_dir(), f"{task}/{date} -> {d}"
        assert epoch_of(task, date=date) == CALIB_SESSIONS[(task, date)]


def test_every_recorded_session_resolves_to_an_epoch():
    """A session in the release whose epoch cannot be RESOLVED is the gap this
    table closes -- not merely one that lacks a hand-written line.

    This used to demand an explicit `CALIB_SESSIONS` entry per recording day.
    The operator decision of 2026-09-15 retired that: a session dated on or
    after `CURRENT_EPOCH` resolves through the current solve, because the live
    recorder built its own overlay from that solve, so the recording was made
    THROUGH it. That is a fact about how the rig ran, not a guess from the
    calendar, which is why it is allowed where inference is not.

    A session dated BEFORE the current solve still has to be declared, and
    `session_epoch` still raises for it. That refusal is what this test keeps:
    it asserts resolution, so both the declared and the defaulted paths pass
    and an unresolvable session fails. On 2026-09-22 the old form failed on
    pushT/2026-09-17 -- a session that resolves correctly to 2026-09-09, the
    same epoch as every one of its declared neighbours.
    """
    import json
    from pathlib import Path
    from twm.calib_epoch import session_epoch
    root = Path("/media/yxma/Disk1/twm/release")
    if not root.is_dir():
        pytest.skip("release tree not on this machine")
    unresolved = []
    for task in ("motherboard", "pushT"):
        jsonl = root / task / "episodes.jsonl"
        if not jsonl.is_file():
            continue
        dates = {json.loads(line)["date"] for line in jsonl.read_text().splitlines() if line.strip()}
        for d in sorted(dates):
            try:
                session_epoch(task, d)
            except Exception as e:
                unresolved.append(f"{task}/{d}: {e}")
    assert not unresolved, "sessions with no resolvable epoch: " + "; ".join(unresolved)


def test_the_status_line_names_the_session_epoch_not_the_task_default():
    """The preview status bar exists so a viewer can catch a wrong epoch
    without trusting the pipeline. Labelling a June-26 render '2026-05-12'
    defeats exactly that."""
    import sys
    from twm.calib_epoch import describe
    # calib_epoch imports `react_toolbox.frames` (not `twm.react_toolbox`) for
    # the Y-up/Z-up conversion, so it resolves only with twm/ on the path —
    # which is how every render script runs. Pre-existing coupling, not part
    # of what this test is about.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "twm"))
    try:
        assert describe("motherboard", "2026-09-09", "episode_000").startswith("calib 2026-09-09")
        assert describe("motherboard", "2026-05-10", "episode_000").startswith("calib 2026-05-12")
    finally:
        sys.path.pop(0)


def test_the_viewer_picks_the_epoch_from_the_date_in_the_path():
    """`python -m twm.visualize .../motherboard/2026-09-09/episode_000.h5`
    must not silently get the May-12 task default."""
    from twm.calib_epoch import calib_dir_for_path
    sept = calib_dir_for_path("/media/yxma/Disk1/twm/data/motherboard/2026-09-09/episode_000.h5")
    may = calib_dir_for_path("/media/yxma/Disk1/twm/data/motherboard/2026-05-10/episode_000.h5")
    assert sept == calib_dir("motherboard", date="2026-09-09")
    assert may == calib_dir("motherboard", date="2026-05-10")
    assert sept != may


def test_a_path_with_no_date_still_falls_back_to_the_task_default():
    from twm.calib_epoch import calib_dir_for_path
    assert calib_dir_for_path("/some/where/motherboard/episode_000.h5") == calib_dir("motherboard")


def test_a_path_whose_session_is_undeclared_raises():
    from twm.calib_epoch import calib_dir_for_path
    # Before the current solve: after it, the session resolves by default.
    with pytest.raises(KeyError, match="2026-07-04"):
        calib_dir_for_path("/data/motherboard/2026-07-04/episode_000.h5")


def test_cam_calib_accepts_an_epoch_name_as_an_escape_hatch():
    from twm.calib_epoch import resolve_calibration
    from twm.calib_epoch import EPOCH_DIRS
    cams, gl, gr = resolve_calibration(["2026-06-26"], None, None,
                                       "/data/motherboard/2026-09-09/ep.h5")
    # The named epoch wins over the one the path's session declares.
    assert all(c.startswith(str(EPOCH_DIRS["2026-06-26"])) for c in cams)
    assert gl.startswith(str(EPOCH_DIRS["2026-06-26"]))


def test_current_epoch_is_declared_and_present():
    """The LIVE recorder's overlay resolves through this. Renaming an epoch
    directory used to turn the overlay off with only a warning, so the
    declaration and the directory must be checked together."""
    import json

    from twm.calib_epoch import CURRENT_EPOCH, EPOCH_DIRS, current_epoch_dir
    assert CURRENT_EPOCH in EPOCH_DIRS
    d = current_epoch_dir()
    assert d.is_dir()
    created = json.loads((d / "T_mocap_to_cam_middle.json").read_text())["created_at"]
    assert created[:10] == CURRENT_EPOCH


def test_the_world_frame_fingerprint_uses_the_session_epoch():
    """`fingerprint` resolved the task default, so every fingerprint stamped
    into a 2026-09-09 parquet was computed with May-12 extrinsics."""
    import inspect

    from twm import world_frame
    for fn in (world_frame.fingerprint, world_frame.verify_fingerprint,
               world_frame._calib):
        assert "date" in inspect.signature(fn).parameters, fn.__name__
