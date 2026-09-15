"""The certifier has to check the alignment the recording actually declares.

`verify_against_h5` documents its own rule: "parquet row i corresponds to
source frame trim + i + shift". That is the LEGACY mapping, and it was correct
for every recording it was written against. Timestamped recordings
(2026-06-27 onward) pair each camera tick with the NEAREST-IN-TIME gel frame,
which repeats and skips, so an index walk compares row i against a frame the
release never used.

Measured on the published motherboard/2026-09-11/episode_000, left sensor:

    mapping the verifier assumes (index)        49-58 % agreement
    the mapping the release used (timestamp)    75-83 % agreement
    what the release actually stores (pixels)   the remainder

and the disagreement is ONE-DIRECTIONAL — 205 rows where an index walk says
"new frame" and the release says "repeat", against 0 the other way. Every one
of those 205 was checked: the two gel frames are bit-identical. The GelSight
emits duplicate frames of its own at 15-18 Hz against a 30 Hz tick, so novelty
is a property of the PIXELS, not of the frame number.

So the release was right and the certifier was measuring the wrong thing. The
fix must not merely make the gate pass: a genuinely misaligned column still
has to fail, which is the last test here.
"""
import numpy as np
import pytest

from twm.react_preprocess.backfill import verify_against_h5


def _episode(tmp_path, *, timestamped: bool, n=40, gel_hz=15.0, corrupt=False):
    """A recording plus the parquet a correct build would publish for it."""
    import h5py
    import hdf5plugin  # noqa: F401
    import pyarrow as pa
    import pyarrow.parquet as pq

    from twm.react_preprocess.h5io import nearest_index

    rng = np.random.default_rng(0)
    cam_ts = 100.0 + np.arange(n) / 30.0
    n_gel = int(n * gel_hz / 30.0) + 2
    gel_ts = 100.0 + np.arange(n_gel) / gel_hz
    # Distinct frames, except a pair the sensor repeats verbatim — the case
    # that separates "different index" from "different pixels".
    frames = rng.integers(0, 255, (n_gel, 8, 10, 3), dtype=np.uint8)
    frames[3] = frames[2]

    h5 = tmp_path / "episode_000.h5"
    with h5py.File(h5, "w") as f:
        f.create_dataset("timestamps", data=cam_ts)
        for side in ("left", "right"):
            g = f.create_group(f"gelsight/{side}")
            g.create_dataset("frames", data=frames)
            if timestamped:
                g.create_dataset("timestamps", data=gel_ts)

    if timestamped:
        idx = nearest_index(gel_ts, cam_ts)
    else:
        idx = np.clip(np.arange(n), 0, n_gel - 1)
    is_new = np.ones(n, bool)
    for i in range(1, n):
        is_new[i] = not np.array_equal(frames[idx[i]], frames[idx[i - 1]])
    # The verifier recovers novelty from the CONTACT SCALARS, not from the
    # is_new column, so they have to be what a real build would write: a
    # function of the gathered frame, which repeats exactly when it does.
    intensity = np.array([frames[j].mean() for j in idx], np.float32)
    if corrupt:
        is_new = np.roll(is_new, 5)          # a constant misalignment
        intensity = np.roll(intensity, 5)

    pqp = tmp_path / "episode_000.parquet"
    pq.write_table(pa.table({
        "source_h5_frame": np.arange(n, dtype=np.int32),
        "tactile_left_is_new": is_new,
        "tactile_right_is_new": is_new,
        "tactile_left_intensity": intensity,
        "tactile_right_intensity": intensity,
    }), str(pqp))
    return pqp, h5


def test_a_timestamped_recording_certifies_clean(tmp_path):
    """The case that has been failing every new session since 2026-06-27."""
    pqp, h5 = _episode(tmp_path, timestamped=True)
    r = verify_against_h5(pqp, h5, "left", limit=40, shift=0)
    assert r["mismatches"] == 0, (
        f"{r['mismatches']}/{r['compared']} rows called wrong on a correctly "
        f"built timestamped episode")


def test_a_legacy_recording_still_certifies_clean(tmp_path):
    """The index walk stays right where it was right; no gel timestamps in the
    file is exactly how a legacy recording announces itself."""
    pqp, h5 = _episode(tmp_path, timestamped=False)
    r = verify_against_h5(pqp, h5, "left", limit=40, shift=0)
    assert r["mismatches"] == 0


def test_a_repeated_sensor_frame_is_not_counted_as_new(tmp_path):
    """Novelty is a property of the pixels. The GelSight emits duplicates of
    its own, and calling those new is what made an index walk disagree with
    the release 205 times in one episode and 0 times the other way."""
    pqp, h5 = _episode(tmp_path, timestamped=True)
    r = verify_against_h5(pqp, h5, "left", limit=40, shift=0)
    assert r["source_unique"] < r["compared"], \
        "every row counted as a new frame — the duplicate pair was missed"


def test_a_misaligned_column_still_fails(tmp_path):
    """The point of the fix is to measure the right thing, NOT to let the gate
    through. A column shifted by five rows has to be caught."""
    pqp, h5 = _episode(tmp_path, timestamped=True, corrupt=True)
    r = verify_against_h5(pqp, h5, "left", limit=40, shift=0)
    assert r["mismatches"] > 0, "a five-row misalignment was certified clean"
