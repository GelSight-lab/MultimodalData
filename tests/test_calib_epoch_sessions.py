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
    assert epoch_of("motherboard", date="2026-09-09") == "2026-06-26"
    assert epoch_of("pushT", date="2026-06-18") == "2026-06-26"
    assert calib_dir("motherboard", date="2026-09-09") == calib_dir("pushT", date="2026-06-18")


def test_the_september_session_does_not_take_the_epoch_measured_that_same_day():
    """The rig was recalibrated on 2026-09-09 at 07:41, after the recordings.
    Newer is not the same as applicable."""
    assert epoch_of("motherboard", date="2026-09-09") == "2026-06-26"
    assert calib_dir("motherboard", date="2026-09-09").name != "result"


def test_epoch_and_check_follow_the_session_too():
    assert epoch_of("motherboard", date="2026-09-09") == "2026-06-26"
    assert epoch_of("motherboard", date="2026-05-19") == "2026-05-12"
    check_epoch("motherboard", date="2026-09-09")     # must not raise


def test_an_undeclared_session_raises_instead_of_falling_back():
    """Falling back to the task default is how a session silently ships
    through the wrong extrinsics."""
    with pytest.raises(KeyError, match="2027-01-01"):
        calib_dir("motherboard", date="2027-01-01")


def test_every_declared_session_names_a_real_epoch_directory():
    for (task, date) in CALIB_SESSIONS:
        d = calib_dir(task, date=date)
        assert d.is_dir(), f"{task}/{date} -> {d}"
        assert epoch_of(task, date=date) == CALIB_SESSIONS[(task, date)]


def test_the_recorded_sessions_on_disk_are_all_declared():
    """A session that exists in the release but not here is exactly the gap
    this table closes."""
    import json
    from pathlib import Path
    root = Path("/media/yxma/Disk1/twm/release")
    if not root.is_dir():
        pytest.skip("release tree not on this machine")
    for task in ("motherboard", "pushT"):
        jsonl = root / task / "episodes.jsonl"
        if not jsonl.is_file():
            continue
        dates = {json.loads(line)["date"] for line in jsonl.read_text().splitlines() if line.strip()}
        undeclared = sorted(d for d in dates if (task, d) not in CALIB_SESSIONS)
        assert not undeclared, f"{task}: undeclared sessions {undeclared}"


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
        assert describe("motherboard", "2026-09-09", "episode_000").startswith("calib 2026-06-26")
        assert describe("motherboard", "2026-05-10", "episode_000").startswith("calib 2026-05-12")
    finally:
        sys.path.pop(0)
