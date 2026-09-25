"""The receipt has to say WHICH check passed, not just that something did.

Measured on rope, 2026-09-15:

    certify_curation :   2.3 s   (parquet scalars only)
    certify_previews : 238.2 s   (opens the 120 GB source H5 per sampled episode)

So the expensive halves are `alignment` and `previews`, and a receipt that
records only "this unit passed" would let a previews-only rerun skip alignment,
or the reverse. Each check is stamped under its own name, against the parquet
hash it passed at.
"""
import pytest

import twm.scripts.certify_release as C


def _pq(p, payload=b"abc"):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(payload)


@pytest.fixture
def tree(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "RELEASE", tmp_path)
    for n in ("episode_000_seg00", "episode_001_seg00"):
        _pq(tmp_path / "rope/meta/2026-09-11" / f"{n}.parquet")
    return tmp_path


def test_a_check_that_passed_is_skipped_only_for_that_check(tree):
    todo = C.needs_certifying("rope", check="alignment")
    C.write_receipt("rope", todo, check="alignment")
    assert C.needs_certifying("rope", check="alignment") == []
    assert len(C.needs_certifying("rope", check="previews")) == 2


def test_changed_bytes_invalidate_every_check(tree):
    for ck in ("alignment", "previews"):
        C.write_receipt("rope", C.needs_certifying("rope", check=ck), check=ck)
    _pq(tree / "rope/meta/2026-09-11/episode_000_seg00.parquet", b"new")
    for ck in ("alignment", "previews"):
        assert C.needs_certifying("rope", check=ck) == \
            ["2026-09-11/episode_000_seg00"], ck


def test_the_default_check_is_alignment(tree):
    """Keeps the first receipt's meaning: it recorded the source-H5 pass."""
    C.write_receipt("rope", C.needs_certifying("rope"))
    assert C.needs_certifying("rope") == []
    assert C.needs_certifying("rope", check="alignment") == []
