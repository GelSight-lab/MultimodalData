import csv
from pathlib import Path

import h5py

from twm.recorder.episode import (EpisodeStore, EpisodeSummary,
                                  next_episode_number)


def summary(**kw):
    base = dict(episode_num=3, path=Path("/x/episode_003.h5"), task="t",
                frame_count=90, fps=30, valid=True, ended_by="operator", reason="",
                max_tick_gap_s=0.04, gap_count=0, queue_peak_fraction=0.1,
                writer_mean_mb_s=200.0, has_optitrack=True)
    base.update(kw)
    return EpisodeSummary(**base)


def test_next_episode_number_scans_only_episode_files(tmp_path):
    assert next_episode_number(tmp_path / "missing") == 0
    (tmp_path / "episode_000.h5").touch()
    (tmp_path / "episode_007.h5").touch()
    (tmp_path / "notes.txt").touch()
    (tmp_path / ".episode_002.h5.tmp").touch()
    assert next_episode_number(tmp_path) == 8


def test_summary_attrs_and_notes():
    ok = summary()
    assert ok.duration_s == 3.0
    assert ok.attrs()["valid"] is True and ok.attrs()["invalid_reason"] == ""
    assert ok.notes() == ""
    bad = summary(valid=False, ended_by="overload", reason="queue full")
    assert bad.attrs()["ended_by"] == "overload"
    assert bad.notes() == "INVALID: overload: queue full"
    assert "INVALID" in bad.describe()
    wd = summary(ended_by="watchdog", reason="sensor_left silent 10.2s")
    assert wd.notes() == "auto-ended: watchdog: sensor_left silent 10.2s"


def test_store_creates_files_in_task_date_dir_and_logs_rows(tmp_path):
    store = EpisodeStore(tmp_path, "pouring", date="2026-09-05")
    assert store.date_dir == tmp_path / "pouring" / "2026-09-05"
    assert store.next_episode_number() == 0
    f, path = store.create(0, ["A"], ["L", "R"], 30)
    f.close()
    assert path == store.date_dir / "episode_000.h5"
    assert store.next_episode_number() == 1
    with h5py.File(path, "r") as g:
        assert g["metadata"].attrs["task"] == "pouring"

    store.log(summary(path=path, task="pouring"))
    store.log(summary(path=path, task="pouring", valid=False, ended_by="disk_low",
                      reason="12 GB free"))
    rows = list(csv.DictReader(open(store.log_path)))
    assert [r["episode"] for r in rows] == ["ep_003", "ep_003"]
    assert rows[0]["notes"] == "" and rows[1]["notes"] == "INVALID: disk_low: 12 GB free"
    assert rows[0]["path"] == "pouring/2026-09-05/episode_000.h5"
    assert rows[0]["optitrack"] == "yes"
