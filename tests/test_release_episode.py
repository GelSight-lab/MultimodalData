"""Playing back a PUBLISHED episode, not just the raw recording it came from.

The viewer could only open an H5. That is the wrong artefact to review before
publishing: the H5 is 30 GB of raw pixels, while what users actually get is
seven MP4s plus a parquet — re-encoded, tone-curved, trimmed to the release
window. Reviewing the source and shipping the derivative means the thing you
looked at is not the thing you published.

This reads the derivative. Poses come from the parquet (already row-aligned to
the video, no timestamp search needed) and forces from the force npz, so the
viewer can draw both overlays the dataset previews draw.
"""
import numpy as np
import pytest

from twm.release_episode import ReleaseEpisode, resolve


def _write_video(path, n, value):
    import subprocess
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = np.stack([np.full((48, 64, 3), (value + k) % 256, np.uint8)
                       for k in range(n)])
    p = subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo",
         "-pix_fmt", "bgr24", "-s", "64x48", "-r", "30", "-i", "-",
         "-c:v", "libx264", "-preset", "ultrafast", "-crf", "0",
         "-pix_fmt", "yuv444p", str(path)],
        input=frames.tobytes(), capture_output=True)
    assert p.returncode == 0, p.stderr[-400:]


def _episode(root, task="motherboard", date="2026-09-11", ep="episode_000", n=12):
    import pyarrow as pa
    import pyarrow.parquet as pq

    vd = root / task / "videos" / date / ep
    for i, name in enumerate(("view_right", "view_left", "view_middle",
                              "tactile_left", "tactile_right",
                              "wrist_left", "wrist_right")):
        _write_video(vd / f"{name}.mp4", n, i * 20)
    md = root / task / "meta" / date
    md.mkdir(parents=True, exist_ok=True)
    pose = lambda o: [[o + k * 0.001, 0.1, 0.2, 0.0, 0.0, 0.0, 1.0] for k in range(n)]
    pq.write_table(pa.table({
        "frame_idx": np.arange(n, dtype=np.int32),
        "timestamp": 100.0 + np.arange(n) / 30.0,
        "sensor_left_pose": pose(0.3),
        "sensor_right_pose": pose(0.6),
        "object_pose": pose(0.45),
        "tactile_left_intensity": np.linspace(0, 5, n, dtype=np.float32),
        "tactile_right_intensity": np.linspace(5, 0, n, dtype=np.float32),
    }), str(md / f"{ep}.parquet"))
    return vd, md / f"{ep}.parquet"


@pytest.fixture
def release(tmp_path):
    _episode(tmp_path)
    return tmp_path


def test_an_episode_is_found_from_any_of_the_three_names_it_has(release):
    """A user pastes whichever path their shell completed: the parquet, the
    video directory, or the key they read off episodes.jsonl."""
    want = ("motherboard", "2026-09-11", "episode_000")
    for path in (release / "motherboard/meta/2026-09-11/episode_000.parquet",
                 release / "motherboard/videos/2026-09-11/episode_000",
                 "motherboard/2026-09-11/episode_000"):
        e = resolve(path, root=release)
        assert (e.task, e.date, e.episode) == want


def test_a_path_that_is_not_a_published_episode_is_refused_by_name(release):
    with pytest.raises(FileNotFoundError, match="episode_099"):
        resolve("motherboard/2026-09-11/episode_099", root=release)


def test_the_frame_count_comes_from_the_parquet_not_the_video(release):
    """The parquet is the authority on what the episode contains; a video that
    disagrees is a build defect and has to surface as one."""
    e = resolve("motherboard/2026-09-11/episode_000", root=release)
    assert e.n_frames == 12
    assert e.fps == 30.0


def test_every_published_stream_is_read_back_at_the_asked_index(release):
    e = resolve("motherboard/2026-09-11/episode_000", root=release)
    color, gel, wrist = e.frames(5)
    assert len(color) == 3 and len(gel) == 2 and len(wrist) == 2
    for f in (*color, *gel, *wrist):
        assert f.shape == (48, 64, 3) and f.dtype == np.uint8
    # view_right was written with value 0 + k, so frame 5 is 5 everywhere.
    assert abs(int(color[0][0, 0, 0]) - 5) <= 2


def test_seeking_backwards_returns_the_frame_asked_for(release):
    """Sequential decode with a cached position is the fast path; a backward
    jump must not quietly hand back the cached later frame."""
    e = resolve("motherboard/2026-09-11/episode_000", root=release)
    for i in (0, 1, 2, 9, 3, 8, 0):
        color, _, _ = e.frames(i)
        assert abs(int(color[0][0, 0, 0]) - i) <= 2, f"frame {i} came back wrong"


def test_poses_arrive_in_the_shape_the_overlay_already_speaks(release):
    """`draw_projection_overlay` consumes what `optitrack_at` returns:
    {body: (timestamp, 7-vector)} with None for a body that was not tracked.
    Matching it means the release path reuses the overlay unchanged."""
    e = resolve("motherboard/2026-09-11/episode_000", root=release)
    p = e.poses_at(4)
    assert set(p) == {"sensor_left", "sensor_right", "motherboard"}
    ts, vec = p["sensor_left"]
    assert ts == pytest.approx(100.0 + 4 / 30.0)
    assert len(vec) == 7 and vec[0] == pytest.approx(0.304)


def test_an_untracked_object_reads_as_absent_rather_than_as_the_origin(tmp_path):
    """All-NaN is how the build records "this body was never broadcast". A
    (0,0,0) pose would be drawn as a real observation at the world origin."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    _episode(tmp_path)
    p = tmp_path / "motherboard/meta/2026-09-11/episode_000.parquet"
    t = pq.read_table(p)
    n = t.num_rows
    t = t.set_column(t.schema.get_field_index("object_pose"), "object_pose",
                     pa.array([[float("nan")] * 7] * n))
    pq.write_table(t, str(p))
    e = resolve("motherboard/2026-09-11/episode_000", root=tmp_path)
    assert e.poses_at(0)["motherboard"] is None


def test_forces_are_read_when_the_estimator_has_run_and_absent_when_it_has_not(
        release, tmp_path):
    # An empty force root, NOT the default: the real one has npz for this very
    # episode, so a test that left it unset would read live data and pass or
    # fail on whatever the estimator last wrote.
    empty = tmp_path / "no_forces"
    empty.mkdir()
    e = resolve("motherboard/2026-09-11/episode_000", root=release, force_root=empty)
    assert e.forces_at(3) == {}, "no npz on disk, so nothing to draw"

    fr = tmp_path / "forces"
    d = fr / "motherboard" / "2026-09-11"
    d.mkdir(parents=True)
    for side, peak in (("left", 4.0), ("right", 1.0)):
        np.savez(d / f"episode_000_{side}.npz",
                 force_normal_n=np.full(12, peak, dtype=float))
    e2 = resolve("motherboard/2026-09-11/episode_000", root=release, force_root=fr)
    assert e2.forces_at(3) == {"left": 4.0, "right": 1.0}


def test_a_force_array_shorter_than_the_episode_does_not_read_off_its_end(
        release, tmp_path):
    fr = tmp_path / "forces"
    d = fr / "motherboard" / "2026-09-11"
    d.mkdir(parents=True)
    np.savez(d / "episode_000_left.npz", force_normal_n=np.arange(5, dtype=float))
    e = resolve("motherboard/2026-09-11/episode_000", root=release, force_root=fr)
    assert e.forces_at(2) == {"left": 2.0}
    assert e.forces_at(9) == {}, "past the end of the force array"
