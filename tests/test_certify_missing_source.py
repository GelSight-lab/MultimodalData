"""An episode whose source recording was deleted must not deadlock the gate.

`certify_alignment` compares the published tactile flags against the source H5
pixels and fails when the H5 is absent — "alignment cannot be certified, only
assumed". That is right for data being published now.

But 1.19 TB of raw HDF5 was deleted on 2026-09-09 to make room, and 33 of
motherboard's 47 episodes plus 5 of pushT's 26 are from those sessions. Their
H5 can never come back, so the condition is PERMANENT: the gate could not pass
again for either task, for any change, ever. A gate that cannot pass is a gate
that gets bypassed with --skip-gate, which is the thing it exists to prevent.

The distinction that makes it passable without weakening it: an episode ALREADY
PUBLISHED with an unchanged parquet asserts nothing new, so there is nothing
new to certify — a warning. An episode being published for the first time with
no source to check against is still a failure.
"""
import numpy as np
import pytest


def _release(tmp_path, task="motherboard", date="2026-05-10", ep="episode_000"):
    import pyarrow as pa
    import pyarrow.parquet as pq
    d = tmp_path / task / "meta" / date
    d.mkdir(parents=True)
    n = 10
    pq.write_table(pa.table({
        "source_h5_frame": np.arange(n, dtype=np.int32),
        "tactile_left_intensity": np.linspace(0, 1, n, dtype=np.float32),
        "tactile_right_intensity": np.linspace(1, 0, n, dtype=np.float32),
    }), str(d / f"{ep}.parquet"))
    return d / f"{ep}.parquet"


def test_a_published_episode_with_no_source_is_a_warning_not_a_failure(tmp_path, monkeypatch):
    from twm.scripts import certify_release as C
    monkeypatch.setattr(C, "RELEASE", tmp_path)
    monkeypatch.setattr(C, "H5_ROOT", tmp_path / "nowhere")
    _release(tmp_path)
    monkeypatch.setattr(C, "published_episodes",
                        lambda task: {"2026-05-10/episode_000"}, raising=False)
    errs, warns = C.certify_alignment("motherboard", 100)
    assert errs == [], f"a deleted-source episode still fails the gate: {errs}"
    assert any("episode_000" in w for w in warns), "and it must still be said"


def test_an_unpublished_episode_with_no_source_still_fails(tmp_path, monkeypatch):
    """New data that cannot be checked is exactly what the gate is for."""
    from twm.scripts import certify_release as C
    monkeypatch.setattr(C, "RELEASE", tmp_path)
    monkeypatch.setattr(C, "H5_ROOT", tmp_path / "nowhere")
    _release(tmp_path, date="2026-09-12")
    monkeypatch.setattr(C, "published_episodes", lambda task: set(), raising=False)
    errs, _ = C.certify_alignment("motherboard", 100)
    assert any("episode_000" in e for e in errs), \
        "unverifiable NEW data was let through"
