"""Walk a folder and show every recording in it, so naming can be checked by eye.

Verifying that an episode is filed under the right task means LOOKING at it.
Doing that one `python -m twm.visualize <path>` at a time does not scale to a
date with twelve recordings, and an automated colour check is not evidence --
a throwaway frame grab that flipped BGR turned a blue T orange and reported a
correctly-filed recording as a different setup.

So: point it at a folder, get every recording under it as a labelled
thumbnail, open the ones that look wrong in the full viewer.

Both tree shapes have to work, because the question comes up on both: the
published tree (`<task>/videos/<date>/<episode>/view_middle.mp4`) and the raw
one (`<task>/<date>/episode_NNN.h5`).
"""
import numpy as np
import pytest

from twm.browse import contact_sheet, find_episodes, label_of


def _video(path, value=90, n=4):
    import subprocess
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = np.stack([np.full((48, 64, 3), (value + k) % 256, np.uint8)
                       for k in range(n)])
    p = subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo",
         "-pix_fmt", "bgr24", "-s", "64x48", "-r", "30", "-i", "-",
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
         str(path)], input=frames.tobytes(), capture_output=True)
    assert p.returncode == 0, p.stderr[-300:]


@pytest.fixture
def published(tmp_path):
    for task, date, ep in (("rope", "2026-09-11", "episode_008_seg00"),
                           ("rope", "2026-09-11", "episode_000_seg00"),
                           ("pushT", "2026-09-10", "episode_000_seg00")):
        _video(tmp_path / task / "videos" / date / ep / "view_middle.mp4")
    return tmp_path


def test_every_published_episode_under_the_folder_is_found(published):
    got = {(e.task, e.date, e.episode) for e in find_episodes(published)}
    assert got == {("rope", "2026-09-11", "episode_008_seg00"),
                   ("rope", "2026-09-11", "episode_000_seg00"),
                   ("pushT", "2026-09-10", "episode_000_seg00")}


def test_pointing_at_one_date_lists_only_that_date(published):
    got = {e.episode for e in find_episodes(published / "rope/videos/2026-09-11")}
    assert got == {"episode_008_seg00", "episode_000_seg00"}


def test_raw_recordings_are_found_too(tmp_path):
    """The mislabelled episode is a RECORDING; the published segments are
    downstream of it. Checking the raw tree is how you catch it before the
    eight hours of building."""
    import h5py
    import hdf5plugin  # noqa: F401
    d = tmp_path / "rope" / "2026-09-11"
    d.mkdir(parents=True)
    with h5py.File(d / "episode_008.h5", "w") as f:
        g = f.create_group("realsense/cam1")
        g.create_dataset("color", data=np.zeros((4, 48, 64, 3), np.uint8))
        f.create_dataset("timestamps", data=np.arange(4, dtype=float))
    got = [(e.task, e.date, e.episode, e.kind) for e in find_episodes(tmp_path)]
    assert got == [("rope", "2026-09-11", "episode_008", "h5")]


def test_the_listing_is_ordered_so_a_date_reads_in_order(published):
    eps = find_episodes(published)
    assert [(e.task, e.date, e.episode) for e in eps] == sorted(
        (e.task, e.date, e.episode) for e in eps)


def test_the_label_carries_what_is_needed_to_spot_a_wrong_task(published):
    e = next(x for x in find_episodes(published) if x.episode.startswith("episode_008"))
    lbl = label_of(e)
    assert "rope" in lbl and "008" in lbl


def test_a_sheet_of_n_thumbnails_has_a_cell_for_each(published):
    eps = find_episodes(published)
    sheet, cells = contact_sheet(eps, cols=2, cell=(64, 48))
    assert len(cells) == len(eps)
    assert sheet.shape[0] >= 48 * 2 and sheet.shape[1] >= 64 * 2
    for (x, y, w, h), e in zip(cells, eps):
        assert 0 <= x < sheet.shape[1] and 0 <= y < sheet.shape[0]


def test_a_cell_can_be_found_from_a_click(published):
    eps = find_episodes(published)
    _, cells = contact_sheet(eps, cols=2, cell=(64, 48))
    from twm.browse import hit_test
    x, y, w, h = cells[1]
    assert hit_test(cells, x + 2, y + 2) == 1
    assert hit_test(cells, 10_000, 10_000) is None


def test_an_unreadable_video_still_gets_a_cell(tmp_path):
    """One corrupt file must not take the whole sheet down -- the point is to
    survey a folder, and a file that cannot be read is itself a finding."""
    p = tmp_path / "rope" / "videos" / "2026-09-11" / "episode_000_seg00"
    p.mkdir(parents=True)
    (p / "view_middle.mp4").write_bytes(b"not a video")
    eps = find_episodes(tmp_path)
    sheet, cells = contact_sheet(eps, cols=2, cell=(64, 48))
    assert len(cells) == 1 and sheet is not None
