"""Only the task that tracks an object gets an `object_pose`.

The motherboard has a rigid body on it. The pushT block does not. That is a
fact about the rig, so it is declared once in `config.OBJECT_TRACKED_TASKS`
rather than discovered per recording.

It used to be discovered. `_object_pose` tried the bodies (task, "object",
"motherboard") in order and took the first one present in the file, which is
correct only while Motive never emits a body the task does not use. During the
2026-09-09 pushT session it emitted `motherboard` anyway -- 3 stray samples
over 8.9 minutes, 157 mm apart, a marker cluster mistaken for the board -- and
the nearest-neighbour fill spread those three values across all 13484 rows.
That episode shipped with `object_pose` 100% non-NaN while every other pushT
episode is all NaN, and the column alone cannot tell a reader which is real.
"""
import numpy as np
import pytest

from twm.react_preprocess.config import OBJECT_BODY, OBJECT_TRACKED_TASKS


def test_only_motherboard_tracks_an_object():
    assert OBJECT_TRACKED_TASKS == {"motherboard"}
    assert "pushT" not in OBJECT_TRACKED_TASKS


def test_every_tracked_task_names_its_body():
    missing = [t for t in OBJECT_TRACKED_TASKS if t not in OBJECT_BODY]
    assert not missing, f"{missing} track an object but do not name its rigid body"


def test_the_builder_reads_the_declaration_and_does_not_search():
    """No fallback chain: a body the task does not declare must not be used."""
    from pathlib import Path
    import twm.react_preprocess.pipeline as pl

    src = Path(pl.__file__).read_text()
    body = "\n".join(l for l in src.splitlines() if not l.strip().startswith("#"))
    i = body.index("def _object_pose")
    fn = body[i:body.index("\ndef ", i + 10)]
    # strip the docstring: it describes the removed search, so a literal scan
    # of the whole function would match its own explanation
    fn = fn.split('"""')[0] + "".join(fn.split('"""')[2:])
    assert "OBJECT_TRACKED_TASKS" in fn, "the builder must consult the declaration"
    assert "for body in" not in fn, "the (task, 'object', 'motherboard') search is back"


class _Src:
    def __init__(self, task, T=8):
        self.task, self.T = task, T
        self.trimmed_cam_ts = np.arange(T, dtype=float)
        self.world_offset = (0.0, 0.0, 0.0)


@pytest.mark.parametrize("task", ["pushT", "some_new_task"])
def test_an_untracked_task_gets_nan_even_when_the_body_is_in_the_file(task):
    """The regression itself: a stray `motherboard` body must not be adopted."""
    import twm.react_preprocess.pipeline as pl

    f = {"optitrack/motherboard": object(),
         "optitrack/motherboard/timestamps": np.array([0.0, 3.0, 6.0]),
         "optitrack/motherboard/pose": np.tile([1.0, 2, 3, 0, 0, 0, 1], (3, 1))}
    out = pl._object_pose(f, _Src(task))
    assert out.shape == (8, 7)
    assert np.isnan(out).all(), f"{task} adopted a body it does not declare"


def test_motherboard_still_gets_its_pose():
    import twm.react_preprocess.pipeline as pl

    f = {"optitrack/motherboard": object(),
         "optitrack/motherboard/timestamps": np.arange(8, dtype=float),
         "optitrack/motherboard/pose": np.tile([1.0, 2, 3, 0, 0, 0, 1], (8, 1))}
    out = pl._object_pose(f, _Src("motherboard"))
    assert np.isfinite(out).all()
    assert np.allclose(out[:, :3], [1.0, 2.0, 3.0])
