"""The certifier's scope is one date, declared once — it has to reach all of it.

`certify_alignment` skips dates before `--since`. `certify_curation` does not,
so a run scoped to one week still demanded a curation record for every May and
June segment in the tree. Measured 2026-09-14: 55 problems, every one of them
"published but never curated" on 2026-05-10/11/19 — data nobody asked to
reprocess, blocking the publish of data that had passed every check.

A gate that cannot pass on the work it was pointed at is a gate that gets
bypassed, which is how the last one stopped being run at all. The filter has
to be applied wherever the tree is enumerated, not in the half that happened
to get it first.
"""
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import twm.scripts.certify_release as C


def _tree(root, task, entries):
    """entries: [(date, name, curated)] — `curated` puts it in bad_frames."""
    r = root / task
    (r / "meta").mkdir(parents=True, exist_ok=True)
    rows, bad = [], {}
    for date, name, curated in entries:
        d = r / "meta" / date
        d.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({"frame_idx": np.arange(4, dtype=np.int32)}),
                       str(d / f"{name}.parquet"))
        key = f"{date}/{name}"
        if curated:
            bad[key] = {}
            rows.append({"episode": key, "date": date, "n_segments": 0,
                         "active_sensors": []})
    (r / "bad_frames.json").write_text(json.dumps({"episodes": bad}))
    (r / "segments.json").write_text(json.dumps({"segments": []}))
    (r / "episodes.jsonl").write_text(
        "\n".join(json.dumps(x) for x in rows) + "\n")
    return r


@pytest.fixture
def scoped(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "RELEASE", tmp_path)
    monkeypatch.setattr(C, "SINCE", "2026-09-08")
    return tmp_path


def _errs(task="motherboard"):
    out = C.certify_curation(task)
    return out[0] if isinstance(out, tuple) else out


def test_an_out_of_scope_date_is_not_demanded(scoped):
    _tree(scoped, "motherboard",
          [("2026-09-11", "episode_000", True),
           ("2026-05-10", "episode_000_seg00", False)])
    errs = _errs()
    assert not [e for e in errs if "2026-05-10" in e], \
        f"May was still demanded: {errs}"


def test_an_in_scope_publication_with_no_curation_record_still_fails(scoped):
    """The point is the scope, not letting the check through."""
    _tree(scoped, "motherboard",
          [("2026-09-11", "episode_000", True),
           ("2026-09-12", "episode_001_seg00", False)])
    errs = _errs()
    assert any("2026-09-12/episode_001_seg00" in e for e in errs), errs


def test_with_no_scope_every_date_is_checked(scoped, monkeypatch):
    """`--since` absent means 'all of it' — the unscoped run must not quietly
    become a scoped one."""
    monkeypatch.setattr(C, "SINCE", None)
    _tree(scoped, "motherboard",
          [("2026-09-11", "episode_000", True),
           ("2026-05-10", "episode_000_seg00", False)])
    errs = _errs()
    assert any("2026-05-10" in e for e in errs), errs
