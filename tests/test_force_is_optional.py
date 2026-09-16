"""A session may publish without the force channel — declared, never silent.

The operator paused force estimation on 2026-09-16: a new algorithm is coming,
and re-running the old one on data that will be re-estimated is wasted hours.

The chain hard-wired force as required. `curate` declared `needs='force'` and
does not read a single npz; `export` writes the columns; `zup` waits on
`export`; and the publish gate refuses a parquet with no force columns —
correctly, while every published segment had them.

146 published segments carry force. If new ones do not, the dataset holds two
schemas, and a reader joining them gets NaN or a KeyError depending on which
loader they use. That is fine ONLY if the file says so. So:

  * skipping force is expressible, and skips export with it;
  * a task with no force npz is not blocked from curate, which never needed it;
  * `episodes.jsonl` records `force: false` for a session published without it,
    so the absence is a fact in the index rather than a hole in a schema.
"""
import json

import pytest

import twm.pipeline_stages as PS


def test_curate_does_not_depend_on_force():
    """It writes bad_frames / segments / episodes from the release tree and
    reads no npz. Declaring force made a force-free run impossible for a stage
    that never needed it."""
    needs = PS.BY_NAME["curate"].needs
    names = (needs,) if isinstance(needs, str) else tuple(needs or ())
    assert "force" not in names, (
        "curate declares force and does not read it, so pausing force "
        "estimation blocks indexing that has nothing to do with it")


def test_skipping_force_also_skips_export():
    """export exists to write the force columns. Running it without the npz
    is the error the operator is trying to avoid, not a stage to salvage."""
    plan = [s.name for s in PS.plan(skip={"force"})]
    assert "export" not in plan, (
        "export survived a skipped force stage; it would fail on the first "
        "missing npz after everything ahead of it had run")


def test_the_rest_of_the_chain_survives_the_skip():
    plan = [s.name for s in PS.plan(skip={"force"})]
    for name in ("build", "curate", "zup", "segment", "index", "verify",
                 "publish"):
        assert name in plan, f"{name} was dropped along with force"


def test_a_force_free_episode_is_recorded_as_such(tmp_path):
    """The absence has to be IN the index. 146 published segments carry force;
    a reader joining a force-free one against them gets NaN or a KeyError
    depending on the loader, and nothing in the data would say why."""
    from twm.react_preprocess.curation import force_flag
    ep = tmp_path / "meta" / "2026-09-16"
    ep.mkdir(parents=True)
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.table({"frame_idx": np.arange(2, dtype=np.int32)}),
                   str(ep / "episode_000.parquet"))
    assert force_flag(ep / "episode_000.parquet") is False
    pq.write_table(pa.table({"frame_idx": np.arange(2, dtype=np.int32),
                             "force_left_normal_n": np.zeros(2)}),
                   str(ep / "episode_001.parquet"))
    assert force_flag(ep / "episode_001.parquet") is True


def test_the_publish_gate_accepts_a_declared_absence(tmp_path):
    """The gate refuses a parquet with no force columns — right, while every
    published segment had them and an overlay could not reach a cut name.

    A DECLARED absence is a different thing: the operator paused the estimator,
    the index says so, and refusing would block a publish that is correct.
    What must still fail is an UNDECLARED one, which is the residue of a run
    that half-finished.
    """
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import twm.scripts.build_release_publish as P

    stage = tmp_path / "release_cut"
    d = stage / "rope" / "meta" / "2026-09-16"
    d.mkdir(parents=True)
    pq.write_table(pa.table({"frame_idx": np.arange(2, dtype=np.int32)}),
                   str(d / "episode_000_seg00.parquet"))

    # Undeclared: still refused.
    (stage / "rope" / "episodes.jsonl").write_text(json.dumps(
        {"episode": "2026-09-16/episode_000_seg00", "date": "2026-09-16"}) + "\n")
    files, missing = P.force_overlay_plan(stage, tmp_path / "force", ("rope",),
                                          since="2026-09-10")
    assert missing, "an undeclared missing force channel stopped being caught"

    # Declared: accepted.
    (stage / "rope" / "episodes.jsonl").write_text(json.dumps(
        {"episode": "2026-09-16/episode_000_seg00", "date": "2026-09-16",
         "force": False}) + "\n")
    files, missing = P.force_overlay_plan(stage, tmp_path / "force", ("rope",),
                                          since="2026-09-10")
    assert not missing, f"a declared force-free session was refused: {missing}"
