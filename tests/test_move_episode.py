"""Refiling a recording under the task it actually belongs to.

Four 2026-09-11 recordings were filed as `rope` and are pushT: the overhead
view shows the blue T, not the rope. Renaming the H5 is the small part. The
recording has products in five trees, its identity is stamped into five parquet
columns, and two tasks' index files describe it -- and a segment cannot be left
behind, because its `source_episode` would then name a recording in another
task.

What must move together, for one recording:

    data/<task>/<date>/<ep>.h5
    release/<task>/{meta/<date>/<ep>.parquet, meta/<date>/<ep>._detect.pt,
                    videos/<date>/<ep>/}
    release_force/<task>/meta/<date>/{<ep>.parquet, <ep>.force.json}
    release_cut/<task>/{meta/<date>/<ep>_segNN.parquet,
                        videos/<date>/<ep>_segNN/, previews/<date>/<ep>_segNN.mp4}
    force_recovery/<task>/<date>/<ep>_{left,right}.npz

and in the cut parquet: task, task_index, episode, episode_index,
source_episode.
"""
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from twm.scripts.move_episode import Move, apply_move, plan_files, rewrite_parquet


def _pq(path, **cols):
    path.parent.mkdir(parents=True, exist_ok=True)
    base = {"frame_idx": np.arange(3, dtype=np.int32)}
    base.update({k: pa.array([v] * 3) for k, v in cols.items()})
    pq.write_table(pa.table(base), str(path))


@pytest.fixture
def trees(tmp_path):
    r = tmp_path
    (r / "data/rope/2026-09-11").mkdir(parents=True)
    (r / "data/rope/2026-09-11/episode_008.h5").write_bytes(b"h5")
    _pq(r / "release/rope/meta/2026-09-11/episode_008.parquet")
    (r / "release/rope/meta/2026-09-11/episode_008._detect.pt").write_bytes(b"pt")
    (r / "release/rope/videos/2026-09-11/episode_008").mkdir(parents=True)
    (r / "release/rope/videos/2026-09-11/episode_008/view_middle.mp4").write_bytes(b"v")
    _pq(r / "release_force/rope/meta/2026-09-11/episode_008.parquet")
    (r / "release_force/rope/meta/2026-09-11/episode_008.force.json").write_text("{}")
    for seg in ("seg00", "seg01"):
        _pq(r / f"release_cut/rope/meta/2026-09-11/episode_008_{seg}.parquet",
            task="rope", task_index=2,
            episode=f"2026-09-11/episode_008_{seg}", episode_index=23,
            source_episode="2026-09-11/episode_008")
        (r / f"release_cut/rope/videos/2026-09-11/episode_008_{seg}").mkdir(parents=True)
        (r / f"release_cut/rope/previews/2026-09-11").mkdir(parents=True, exist_ok=True)
        (r / f"release_cut/rope/previews/2026-09-11/episode_008_{seg}.mp4").write_bytes(b"p")
    (r / "force_recovery/rope/2026-09-11").mkdir(parents=True)
    for side in ("left", "right"):
        (r / f"force_recovery/rope/2026-09-11/episode_008_{side}.npz").write_bytes(b"n")
    # An untouched neighbour in each tree, to prove the move is surgical.
    _pq(r / "release_cut/rope/meta/2026-09-11/episode_000_seg00.parquet",
        task="rope", task_index=2, episode="2026-09-11/episode_000_seg00",
        episode_index=0, source_episode="2026-09-11/episode_000")
    (r / "data/rope/2026-09-11/episode_000.h5").write_bytes(b"other")
    return r


MV = Move("rope", "2026-09-11", "episode_008", "pushT", "2026-09-11", "episode_005")


def test_every_product_of_the_recording_is_in_the_plan(trees):
    got = {str(s.relative_to(trees)) for s, _ in plan_files(MV, trees)}
    assert got == {
        "data/rope/2026-09-11/episode_008.h5",
        "release/rope/meta/2026-09-11/episode_008.parquet",
        "release/rope/meta/2026-09-11/episode_008._detect.pt",
        "release/rope/videos/2026-09-11/episode_008",
        "release_force/rope/meta/2026-09-11/episode_008.parquet",
        "release_force/rope/meta/2026-09-11/episode_008.force.json",
        "release_cut/rope/meta/2026-09-11/episode_008_seg00.parquet",
        "release_cut/rope/meta/2026-09-11/episode_008_seg01.parquet",
        "release_cut/rope/videos/2026-09-11/episode_008_seg00",
        "release_cut/rope/videos/2026-09-11/episode_008_seg01",
        "release_cut/rope/previews/2026-09-11/episode_008_seg00.mp4",
        "release_cut/rope/previews/2026-09-11/episode_008_seg01.mp4",
        "force_recovery/rope/2026-09-11/episode_008_left.npz",
        "force_recovery/rope/2026-09-11/episode_008_right.npz",
    }


def test_a_segment_keeps_its_segment_number(trees):
    dsts = {str(d.relative_to(trees)) for _, d in plan_files(MV, trees)}
    assert "release_cut/pushT/meta/2026-09-11/episode_005_seg01.parquet" in dsts
    assert "release_cut/pushT/previews/2026-09-11/episode_005_seg00.mp4" in dsts


def test_a_neighbour_is_never_swept_in(trees):
    srcs = {str(s.relative_to(trees)) for s, _ in plan_files(MV, trees)}
    assert not any("episode_000" in s for s in srcs)


def test_an_occupied_destination_is_refused(trees):
    (trees / "data/pushT/2026-09-11").mkdir(parents=True)
    (trees / "data/pushT/2026-09-11/episode_005.h5").write_bytes(b"taken")
    with pytest.raises(FileExistsError, match="episode_005"):
        apply_move(MV, trees, episode_index=55)


def test_the_identity_columns_are_rewritten(trees):
    p = trees / "release_cut/rope/meta/2026-09-11/episode_008_seg00.parquet"
    rewrite_parquet(p, MV, episode_index=55)
    t = pq.read_table(p)
    assert t["task"][0].as_py() == "pushT"
    assert t["task_index"][0].as_py() == 1
    assert t["episode"][0].as_py() == "2026-09-11/episode_005_seg00"
    assert t["episode_index"][0].as_py() == 55
    assert t["source_episode"][0].as_py() == "2026-09-11/episode_005"


def test_the_move_leaves_nothing_behind(trees):
    apply_move(MV, trees, episode_index=55)
    assert not (trees / "data/rope/2026-09-11/episode_008.h5").exists()
    assert (trees / "data/pushT/2026-09-11/episode_005.h5").read_bytes() == b"h5"
    assert not list((trees / "release_cut/rope/meta/2026-09-11").glob("episode_008*"))
    assert (trees / "release_cut/pushT/meta/2026-09-11/episode_005_seg01.parquet").is_file()
    # the neighbour is untouched
    assert (trees / "data/rope/2026-09-11/episode_000.h5").read_bytes() == b"other"


def test_the_index_files_follow_the_recording(trees):
    for task, rows in (("rope", [{"episode": "2026-09-11/episode_008_seg00",
                                  "date": "2026-09-11", "task": "rope"},
                                 {"episode": "2026-09-11/episode_000_seg00",
                                  "date": "2026-09-11", "task": "rope"}]),
                       ("pushT", [{"episode": "2026-09-11/episode_000_seg00",
                                   "date": "2026-09-11", "task": "pushT"}])):
        d = trees / "release_cut" / task
        d.mkdir(parents=True, exist_ok=True)
        (d / "episodes.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows))
    apply_move(MV, trees, episode_index=55)
    src = [json.loads(l) for l in
           (trees / "release_cut/rope/episodes.jsonl").read_text().splitlines() if l.strip()]
    dst = [json.loads(l) for l in
           (trees / "release_cut/pushT/episodes.jsonl").read_text().splitlines() if l.strip()]
    assert [r["episode"] for r in src] == ["2026-09-11/episode_000_seg00"]
    assert sorted(r["episode"] for r in dst) == ["2026-09-11/episode_000_seg00",
                                                 "2026-09-11/episode_005_seg00"]
    moved = next(r for r in dst if "005" in r["episode"])
    assert moved["task"] == "pushT"


def test_the_provenance_record_follows_the_recording(trees):
    """`segment_provenance.json` is the record of WHY each segment exists and
    which spans were dropped. Left behind, it has the source task claiming
    segments that are no longer in its tree, and the destination unable to say
    where its new ones came from."""
    for task, segs, drops in (
            ("rope",
             [{"episode": "2026-09-11/episode_008_seg00", "date": "2026-09-11",
               "task": "rope", "source_episode": "2026-09-11/episode_008"},
              {"episode": "2026-09-11/episode_000_seg00", "date": "2026-09-11",
               "task": "rope", "source_episode": "2026-09-11/episode_000"}],
             [{"episode": "2026-09-11/episode_008", "frame_range": [0, 9]},
              {"episode": "2026-09-11/episode_000", "frame_range": [3, 4]}]),
            ("pushT", [], [])):
        d = trees / "release_cut" / task
        d.mkdir(parents=True, exist_ok=True)
        (d / "segment_provenance.json").write_text(json.dumps(
            {"summary": {"task": task}, "thresholds": {"tau": 1},
             "dropped_spans": drops, "segments": segs}))
    apply_move(MV, trees, episode_index=55)
    src = json.loads((trees / "release_cut/rope/segment_provenance.json").read_text())
    dst = json.loads((trees / "release_cut/pushT/segment_provenance.json").read_text())
    assert [s["episode"] for s in src["segments"]] == ["2026-09-11/episode_000_seg00"]
    assert [s["episode"] for s in src["dropped_spans"]] == ["2026-09-11/episode_000"]
    assert [s["episode"] for s in dst["segments"]] == ["2026-09-11/episode_005_seg00"]
    assert dst["segments"][0]["task"] == "pushT"
    assert dst["segments"][0]["source_episode"] == "2026-09-11/episode_005"
    assert [s["episode"] for s in dst["dropped_spans"]] == ["2026-09-11/episode_005"]
    # the thresholds the destination was already judged at are not overwritten
    assert dst["thresholds"] == {"tau": 1}
