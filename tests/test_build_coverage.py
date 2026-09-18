"""The build stage has to be measured against the RECORDINGS, not its own output.

`coverage` takes the source set from the release tree for every stage. For
every stage but one that is right -- you cannot cut what was never built. For
`build` it is circular: the denominator IS the output, so `done == total`
always, and a recording that was never built is not missing, it is invisible.

Measured 2026-09-15: rope/2026-09-14 holds six recordings, three had been
built, and coverage reported `build done=11 total=11 missing=[]`. The three
unbuilt ones did not appear anywhere. A scheduler that reports a stage complete
when half its input has not been touched will walk straight past it and cut,
index and publish the half.
"""
import h5py
import pytest

import twm.pipeline_stages as PS


@pytest.fixture
def trees(tmp_path, monkeypatch):
    data, rel = tmp_path / "data", tmp_path / "release"
    for date, eps in (("2026-09-14", ("episode_000", "episode_001", "episode_002")),
                      ("2026-05-11", ("episode_000",))):
        (data / "rope" / date).mkdir(parents=True)
        for e in eps:
            with h5py.File(data / "rope" / date / f"{e}.h5", "w") as f:
                for side in ("left", "right"):
                    f[f"gelsight/{side}/frames"] = [0]
    # only the first was built
    (rel / "rope" / "meta" / "2026-09-14").mkdir(parents=True)
    meta = rel / "rope" / "meta" / "2026-09-14"
    videos = rel / "rope" / "videos" / "2026-09-14" / "episode_000"
    videos.mkdir(parents=True)
    for side in ("left", "right"):
        (videos / f"tactile_{side}.mp4").write_bytes(b"encoded stream")
    (meta / "episode_000._detect.pt").write_bytes(b"sidecar")
    (meta / "episode_000.parquet").write_bytes(b"parquet")
    monkeypatch.setattr(PS, "DATA_ROOT", data)
    monkeypatch.setattr(PS, "RELEASE", rel)
    monkeypatch.setattr(PS, "SCOPE_SINCE", "2026-09-10")
    return data, rel


def test_unbuilt_recordings_are_counted_as_missing(trees):
    c = PS.coverage("build", "rope")
    assert c.total == 3 and c.done == 1
    assert c.missing == ["2026-09-14/episode_001", "2026-09-14/episode_002"]


def test_out_of_scope_recordings_are_not_demanded(trees):
    """The May session is not this window's work; counting it would make the
    stage permanently incomplete."""
    c = PS.coverage("build", "rope")
    assert not any("2026-05" in m for m in c.missing)


def test_a_later_stage_still_measures_against_what_was_built(trees):
    """You cannot cut what was never built, so for every other stage the
    release tree remains the right denominator."""
    c = PS.coverage("zup", "rope")
    assert c.total == 1
