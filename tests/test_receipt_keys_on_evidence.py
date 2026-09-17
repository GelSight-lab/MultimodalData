"""A certification receipt must key on what its check actually READS.

`_parquet_hash` hashed the whole file, so ANY byte change invalidated every
receipt for that unit. That was fine while the only thing that changed a
parquet was a re-cut. On 2026-09-16 the force-only export adds columns to
every published parquet, which would have re-run the alignment certification
over all 173 segments -- 2000 frames per sensor-side, read out of the source
H5 on a seek-bound mechanical disk, about seven hours.

And it would have proved nothing new. `verify_against_h5` reads exactly
`source_h5_frame` and the tactile contact scalars, compares them against the
source H5 pixels, and never looks at another column. Adding `force_left_
normal_n` cannot change its answer.

So the receipt hashes the EVIDENCE for its check, not the file. The risk being
managed is the opposite one -- a receipt that survives a change it should not
-- so the tests below spend most of their effort on what must still
invalidate.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "twm" / "scripts"))
import certify_release as CR  # noqa: E402


def _write(path, *, n=8, intensity=None, source=None, extra=None):
    cols = {
        "source_h5_frame": np.arange(n, dtype=np.int32) if source is None else source,
        "tactile_left_intensity": (np.linspace(0, 1, n, dtype=np.float32)
                                   if intensity is None else intensity),
        "tactile_left_area": np.linspace(1, 2, n, dtype=np.float32),
        "tactile_left_mixed": np.linspace(2, 3, n, dtype=np.float32),
        "tactile_right_intensity": np.linspace(3, 4, n, dtype=np.float32),
        "tactile_right_area": np.linspace(4, 5, n, dtype=np.float32),
        "tactile_right_mixed": np.linspace(5, 6, n, dtype=np.float32),
        "tactile_left_is_new": np.ones(n, bool),
    }
    cols.update(extra or {})
    pq.write_table(pa.table(cols), str(path))
    return path


def test_adding_a_force_column_does_not_invalidate_alignment(tmp_path):
    """The whole point: the force-only export must not re-certify the world."""
    a = _write(tmp_path / "a.parquet")
    b = _write(tmp_path / "b.parquet",
               extra={"force_left_normal_n": np.zeros(8, np.float32),
                      "force_left_source_frame": np.arange(8, dtype=np.int32)})
    assert CR._evidence_hash(a, "alignment") == CR._evidence_hash(b, "alignment")


def test_changing_a_contact_scalar_does_invalidate_alignment(tmp_path):
    a = _write(tmp_path / "a.parquet")
    b = _write(tmp_path / "b.parquet",
               intensity=np.linspace(0, 9, 8, dtype=np.float32))
    assert CR._evidence_hash(a, "alignment") != CR._evidence_hash(b, "alignment")


def test_changing_the_source_frame_map_does_invalidate_alignment(tmp_path):
    """A re-cut moves the window; that is exactly what must re-certify."""
    a = _write(tmp_path / "a.parquet")
    b = _write(tmp_path / "b.parquet",
               source=np.arange(100, 108, dtype=np.int32))
    assert CR._evidence_hash(a, "alignment") != CR._evidence_hash(b, "alignment")


def test_dropping_a_scalar_column_invalidates_alignment(tmp_path):
    """flags_from_scalars uses whichever scalars are present, so losing one
    changes the proxy it builds."""
    a = _write(tmp_path / "a.parquet")
    t = pq.read_table(str(a)).drop(["tactile_left_area"])
    b = tmp_path / "b.parquet"
    pq.write_table(t, str(b))
    assert CR._evidence_hash(a, "alignment") != CR._evidence_hash(b, "alignment")


def test_a_different_row_count_invalidates_alignment(tmp_path):
    a = _write(tmp_path / "a.parquet", n=8)
    b = _write(tmp_path / "b.parquet", n=9)
    assert CR._evidence_hash(a, "alignment") != CR._evidence_hash(b, "alignment")


def test_previews_still_key_on_the_whole_file(tmp_path):
    """Previews are rendered from far more than the alignment columns, and
    nothing here has established which. Keep the conservative hash."""
    a = _write(tmp_path / "a.parquet")
    b = _write(tmp_path / "b.parquet",
               extra={"force_left_normal_n": np.zeros(8, np.float32)})
    assert CR._evidence_hash(a, "previews") != CR._evidence_hash(b, "previews")
    assert CR._evidence_hash(a, "previews") == CR._parquet_hash(a)


def test_an_unknown_check_falls_back_to_the_whole_file(tmp_path):
    """A check nobody has analysed must not silently get a narrow hash."""
    a = _write(tmp_path / "a.parquet")
    assert CR._evidence_hash(a, "something_new") == CR._parquet_hash(a)
