"""A task with no force channel is drawn as such, not as zero contact.

`collect` read `force_left_normal_n` unconditionally and died on `toy`, which
is published without force while a new estimator is written:

    KeyError: Field "force_left_normal_n" does not exist in schema

Filling with zeros would be worse than crashing: panels B (force distribution)
and C (contact occupancy) are derived from force alone, and a zero-filled task
appears on the figure as one that never makes contact. That is a claim about
the data, and it would be false.

So the absence propagates: the task is counted in scale and tactile validity,
which do not need force, and is marked `no_force` so the force-derived panels
can leave it out and say why.
"""
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def _table(tmp_path, with_force):
    n = 4
    cols = {"tactile_left_is_new": np.ones(n, bool)}
    if with_force:
        cols["force_left_normal_n"] = np.linspace(0, 3, n)
        cols["force_right_normal_n"] = np.linspace(3, 0, n)
    p = tmp_path / "e.parquet"
    pq.write_table(pa.table(cols), str(p))
    return p


def test_a_force_free_task_does_not_crash(tmp_path):
    from twm.scripts.dataset_stats import collect
    p = _table(tmp_path, with_force=False)
    o = collect("toy", ["2026-09-16/episode_000"], lambda _: str(p))
    assert o["segs"] == 1 and o["frames"] == 4


def test_it_is_marked_rather_than_zero_filled(tmp_path):
    """Zero force would draw `toy` as a task that never touches anything."""
    from twm.scripts.dataset_stats import collect
    p = _table(tmp_path, with_force=False)
    o = collect("toy", ["2026-09-16/episode_000"], lambda _: str(p))
    assert o.get("no_force") is True
    assert o["cL"] == 0 and not o["f"], (
        "a force-free task contributed contact counts it cannot support")


def test_a_task_with_force_is_unchanged(tmp_path):
    from twm.scripts.dataset_stats import collect
    p = _table(tmp_path, with_force=True)
    o = collect("rope", ["2026-09-16/episode_000"], lambda _: str(p))
    assert o.get("no_force") is False
    assert o["cL"] > 0 and len(o["f"]) == 1


@pytest.mark.parametrize("values", [[], [0., 0.], [8., 8.]])
def test_summary_handles_empty_contact_distributions(tmp_path, values):
    import json
    from twm.scripts.dataset_stats import collect, summarise
    p = tmp_path / "empty.parquet"
    pq.write_table(pa.table({"tactile_left_is_new": [True] * len(values),
                            "force_left_normal_n": values,
                            "force_right_normal_n": values}), p)
    summary = summarise(collect("rope", ["2026-09-16/episode_000"], lambda _: p))
    assert summary["f_q"] is None
    assert summary["f_ecdf_x"] is None
    json.dumps(summary, allow_nan=False)


def test_partial_force_channels_are_not_reported_as_complete(tmp_path):
    from twm.scripts.dataset_stats import collect, summarise
    p = tmp_path / "partial.parquet"
    pq.write_table(pa.table({"tactile_left_is_new": [True],
                            "force_left_normal_n": [1.]}), p)
    summary = summarise(collect("toy", ["2026-09-16/episode_000"], lambda _: p))
    assert summary["no_force"] is True
    assert summary["contact_pct_L"] is None
    assert summary["frames"] == 1


def test_force_free_figure_renders_without_inventing_force(tmp_path):
    from twm.scripts.dataset_stats import collect, summarise, CEILING_N
    from twm.scripts.dataset_stats_fig import panel
    p = _table(tmp_path, with_force=False)
    summary = summarise(collect("toy", ["2026-09-16/episode_000"], lambda _: p))
    output = tmp_path / "stats.png"
    panel({"toy": summary}, CEILING_N, output, "Test")
    assert output.stat().st_size > 0


def test_mixed_force_coverage_preserves_scale_but_withholds_force_summary(tmp_path):
    from twm.scripts.dataset_stats import collect, summarise
    paths = []
    for present in (True, False):
        directory = tmp_path / str(present)
        directory.mkdir()
        paths.append(_table(directory, present))
    files = iter(paths)
    summary = summarise(collect("toy", ["2026-09-16/episode_000",
                                        "2026-09-16/episode_001"], lambda _: next(files)))
    assert summary["frames"] == 8 and summary["sources"] == 2
    assert summary["no_force"] and summary["contact_frames"] is None


def test_empty_collection_summary_is_json_safe():
    import json
    from twm.scripts.dataset_stats import collect, summarise
    summary = summarise(collect("toy", [], lambda _: pytest.fail("unexpected fetch")))
    assert summary["frames"] == 0 and summary["new_pct"] is None
    json.dumps(summary, allow_nan=False)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_nonfinite_force_marks_summary_incomplete(tmp_path, invalid):
    import json
    from twm.scripts.dataset_stats import collect, summarise
    p = tmp_path / "nonfinite.parquet"
    pq.write_table(pa.table({"tactile_left_is_new": [True, True],
                            "force_left_normal_n": [1., invalid],
                            "force_right_normal_n": [1., 2.]}), p)
    summary = summarise(collect("toy", ["2026-09-16/episode_000"], lambda _: p))
    assert summary["frames"] == 2 and summary["no_force"]
    assert summary["contact_pct_L"] is None
    json.dumps(summary, allow_nan=False)


@pytest.mark.parametrize("values", [[], [0., 0.], [8., 8.]])
def test_empty_force_distributions_render(tmp_path, values):
    from twm.scripts.dataset_stats import collect, summarise, CEILING_N
    from twm.scripts.dataset_stats_fig import panel
    p = tmp_path / "distribution.parquet"
    pq.write_table(pa.table({"tactile_left_is_new": [True] * len(values),
                            "force_left_normal_n": values,
                            "force_right_normal_n": values}), p)
    summary = summarise(collect("rope", ["2026-09-16/episode_000"], lambda _: p))
    output = tmp_path / "distribution.png"
    panel({"rope": summary}, CEILING_N, output, "Test")
    assert output.stat().st_size > 0
