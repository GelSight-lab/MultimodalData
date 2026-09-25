"""main carries one week. Everything older is a different branch's problem.

The cut tree holds every date ever built, because the old data still has to go
somewhere later. What goes to the dataset's main revision is a window:
2026-09-10 onward. 2026-09-09 is excluded too — it was recorded with a
different wrist camera and was published separately as `validation`.

Two halves have to agree, or the upload creates the exact defect the indices
exist to prevent:

  * the FILES uploaded must be the in-scope dates only;
  * the INDEX files uploaded must describe exactly those, and no others.

An episode published but absent from `splits.json` is read as TRAIN by
`ReactVideoDataset._split_filter`. Uploading a full index beside a windowed
file set, or a windowed index beside a full file set, both produce it — from
opposite directions.
"""
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import twm.scripts.build_release_publish as P

SINCE = "2026-09-10"
IN, OUT = "2026-09-11", "2026-05-10"


@pytest.fixture
def tree(tmp_path):
    r = tmp_path / "release_cut" / "pushT"
    keys = []
    for date, name in ((IN, "episode_000_seg00"), (OUT, "episode_000_seg00")):
        (r / "meta" / date).mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({"frame_idx": np.arange(3, dtype=np.int32)}),
                       str(r / "meta" / date / f"{name}.parquet"))
        (r / "videos" / date / name).mkdir(parents=True, exist_ok=True)
        (r / "videos" / date / name / "view_left.mp4").write_bytes(b"x")
        keys.append(f"{date}/{name}")
    (r / "calibration").mkdir()
    (r / "calibration" / "T_mocap_to_cam_left.json").write_text("{}")
    (r / "episodes.jsonl").write_text(
        "".join(json.dumps({"episode": k, "date": k.split("/")[0]}) + "\n"
                for k in keys))
    (r / "bad_frames.json").write_text(json.dumps(
        {"task": "pushT", "episodes": {k: {} for k in keys}}))
    (r / "segments.json").write_text(json.dumps(
        {"task": "pushT", "segments": [{"source_episode": k} for k in keys]}))
    (r / "splits.json").write_text(json.dumps(
        {"seed": 0, "episodes": {k: {"test": []} for k in keys}}))
    return r


def test_an_out_of_scope_date_is_not_uploaded(tree):
    pats = P.scoped_patterns(tree, SINCE)
    assert not any(OUT in p for p in pats), \
        f"May would have gone to main: {[p for p in pats if OUT in p]}"


def test_the_in_scope_dates_are_uploaded(tree):
    pats = P.scoped_patterns(tree, SINCE)
    assert any(p.startswith(f"meta/{IN}") for p in pats)
    assert any(p.startswith(f"videos/{IN}") for p in pats)


def test_the_calibration_always_goes(tree):
    """It describes the frame the poses are in, not a date."""
    pats = P.scoped_patterns(tree, SINCE)
    assert any(p.startswith("calibration/") for p in pats)


def test_the_uploaded_index_describes_only_what_was_uploaded(tree, tmp_path):
    out = tmp_path / "scoped"
    P.scoped_index(tree, SINCE, out)
    eps = [json.loads(l)["episode"]
           for l in (out / "episodes.jsonl").read_text().splitlines() if l.strip()]
    assert eps == [f"{IN}/episode_000_seg00"], eps
    for name, key in (("bad_frames.json", "episodes"),
                      ("splits.json", "episodes")):
        got = list(json.loads((out / name).read_text())[key])
        assert got == [f"{IN}/episode_000_seg00"], (name, got)
    segs = json.loads((out / "segments.json").read_text())["segments"]
    assert [s["source_episode"] for s in segs] == [f"{IN}/episode_000_seg00"]


def test_nothing_uploaded_is_left_out_of_the_splits(tree, tmp_path):
    """The train leak, stated as an equality: the parquet that actually match
    the upload patterns, and the episodes the splits describe, are one set."""
    import fnmatch
    out = tmp_path / "scoped"
    P.scoped_index(tree, SINCE, out)
    pats = P.scoped_patterns(tree, SINCE)
    shipped = set()
    for f in tree.rglob("*.parquet"):
        rel = str(f.relative_to(tree))
        if any(fnmatch.fnmatch(rel, pat) for pat in pats):
            shipped.add(f"{f.parent.name}/{f.stem}")
    described = set(json.loads((out / "splits.json").read_text())["episodes"])
    assert shipped == described, f"only on one side: {shipped ^ described}"


def test_the_index_keeps_its_non_episode_fields(tree, tmp_path):
    """`bad_frames.json` carries the thresholds it was measured at. Filtering
    the episodes must not drop the record of how they were judged."""
    out = tmp_path / "scoped"
    P.scoped_index(tree, SINCE, out)
    assert json.loads((out / "bad_frames.json").read_text())["task"] == "pushT"
    assert json.loads((out / "splits.json").read_text())["seed"] == 0
