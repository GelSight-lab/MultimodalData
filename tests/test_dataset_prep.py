"""Assembling a session folder out of the release trees."""
import json

import numpy as np
import pandas as pd

from twm.dataset_prep import assemble_session

DATE = "2026-09-09"
OTHER = "2026-05-10"


def _release(tmp_path):
    """A release tree holding two sessions, only one of which we want."""
    rel, force = tmp_path / "release" / "motherboard", tmp_path / "release_force" / "motherboard"
    for date, eps in ((DATE, ("episode_000", "episode_001")), (OTHER, ("episode_000",))):
        (force / "meta" / date).mkdir(parents=True)
        for ep in eps:
            pd.DataFrame({"frame_idx": np.arange(5)}).to_parquet(force / "meta" / date / f"{ep}.parquet")
            (force / "meta" / date / f"{ep}.force.json").write_text("{}")
            for kind, name in (("videos", "view_left.mp4"), ("depth", "depth_left.mkv")):
                d = rel / kind / date / ep; d.mkdir(parents=True, exist_ok=True)
                (d / name).write_bytes(b"x")
            pv = rel / "previews" / date; pv.mkdir(parents=True, exist_ok=True)
            (pv / f"{ep}.mp4").write_bytes(b"x")
    (rel / "episodes.jsonl").write_text("\n".join(json.dumps(
        {"episode": f"{d}/{e}", "date": d, "n_frames": 5})
        for d, es in ((DATE, ("episode_000", "episode_001")), (OTHER, ("episode_000",))) for e in es))
    (rel / "bad_frames.json").write_text(json.dumps(
        {"task": "motherboard", "tau_intensity": 1.0,
         "episodes": {f"{DATE}/episode_000": {"x": 1}, f"{OTHER}/episode_000": {"y": 2}}}))
    (rel / "segments.json").write_text(json.dumps(
        {"task": "motherboard", "schema": "v1", "segments": [
            {"source_episode": f"{DATE}/episode_000", "n_frames": 5},
            {"source_episode": f"{OTHER}/episode_000", "n_frames": 5}]}))
    cal = tmp_path / "epoch"; cal.mkdir()
    (cal / "T_mocap_to_cam_middle.json").write_text(json.dumps({"created_at": "2026-09-09T07:41:39"}))
    return rel, force, cal


def test_assemble_takes_only_this_session(tmp_path):
    rel, force, cal = _release(tmp_path)
    out = assemble_session(tmp_path / "stage", "motherboard", DATE,
                           ["episode_000", "episode_001"],
                           release=rel, release_force=force, calib_dir=cal)
    assert sorted(p.name for p in (out / "meta" / DATE).glob("*.parquet")) == \
        ["episode_000.parquet", "episode_001.parquet"]
    assert not (out / "meta" / OTHER).exists()
    assert (out / "videos" / DATE / "episode_000" / "view_left.mp4").is_file()
    assert (out / "previews" / DATE / "episode_001.mp4").is_file()
    assert (out / "calibration" / "T_mocap_to_cam_middle.json").is_file()


def test_the_indices_are_filtered_to_this_session(tmp_path):
    """Shipping the whole task's episodes.jsonl inside one session's folder
    describes episodes that folder does not contain."""
    rel, force, cal = _release(tmp_path)
    out = assemble_session(tmp_path / "stage", "motherboard", DATE,
                           ["episode_000", "episode_001"],
                           release=rel, release_force=force, calib_dir=cal)
    rows = [json.loads(l) for l in (out / "episodes.jsonl").read_text().splitlines() if l.strip()]
    assert [r["episode"] for r in rows] == [f"{DATE}/episode_000", f"{DATE}/episode_001"]
    bad = json.loads((out / "bad_frames.json").read_text())
    assert list(bad["episodes"]) == [f"{DATE}/episode_000"]
    assert bad["tau_intensity"] == 1.0            # the thresholds travel with it
    seg = json.loads((out / "segments.json").read_text())
    assert [s["source_episode"] for s in seg["segments"]] == [f"{DATE}/episode_000"]


def test_a_requested_episode_with_no_parquet_is_refused(tmp_path):
    rel, force, cal = _release(tmp_path)
    try:
        assemble_session(tmp_path / "stage", "motherboard", DATE, ["episode_009"],
                         release=rel, release_force=force, calib_dir=cal)
    except FileNotFoundError as exc:
        assert "episode_009" in str(exc)
    else:
        raise AssertionError("expected a refusal")


def test_the_staged_parquet_carries_the_published_index_columns(tmp_path):
    """Published parquets have task/task_index/episode/episode_index/
    frame_index; freshly built ones do not, because the only function that
    adds them is called from nowhere. A folder that ships without them has a
    different schema from every other folder in the dataset."""
    rel, force, cal = _release(tmp_path)
    out = assemble_session(tmp_path / "stage", "motherboard", DATE,
                           ["episode_000", "episode_001"],
                           release=rel, release_force=force, calib_dir=cal)
    a = pd.read_parquet(out / "meta" / DATE / "episode_000.parquet")
    b = pd.read_parquet(out / "meta" / DATE / "episode_001.parquet")
    for col in ("task", "task_index", "episode", "episode_index", "frame_index"):
        assert col in a.columns, col
    assert a["task"].iloc[0] == "motherboard"
    assert a["episode"].iloc[0] == f"{DATE}/episode_000"
    assert list(a["episode_index"].unique()) == [0]
    assert list(b["episode_index"].unique()) == [1]
    assert list(a["frame_index"]) == list(range(len(a)))
