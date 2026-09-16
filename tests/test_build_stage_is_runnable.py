"""The build stage has to be invokable by the thing that schedules it.

`_build` took a required `date`, and `run_all` passes none. So the stage sat in
the plan, printed its heading, and died on
`TypeError: _build() missing 1 required positional argument: 'date'` -- every
time. It was only ever run by hand, with a date typed in, which is why nobody
noticed: the manual path worked and the scheduled one had never executed.

Given no date, it should build exactly what `coverage` says is missing, one
command per date. Rebuilding an episode that already has a parquet costs hours
and produces the same bytes.
"""
import pytest

import twm.pipeline_stages as PS


@pytest.fixture
def trees(tmp_path, monkeypatch):
    data, rel = tmp_path / "data", tmp_path / "release"
    for date, eps in (("2026-09-14", ("episode_000", "episode_001")),
                      ("2026-09-15", ("episode_000",))):
        (data / "rope" / date).mkdir(parents=True)
        for e in eps:
            (data / "rope" / date / f"{e}.h5").write_bytes(b"h5")
    (rel / "rope" / "meta" / "2026-09-14").mkdir(parents=True)
    (rel / "rope" / "meta" / "2026-09-14" / "episode_000.parquet").write_bytes(b"")
    monkeypatch.setattr(PS, "DATA_ROOT", data)
    monkeypatch.setattr(PS, "RELEASE", rel)
    monkeypatch.setattr(PS, "SCOPE_SINCE", "2026-09-10")
    return data, rel


def _cmds(task="rope"):
    return [[str(x) for x in c] for c in PS._build(task=task)]


def test_the_runner_can_invoke_it_with_no_date(trees):
    cmds = _cmds()
    assert cmds, "the stage produced no command at all"


def test_it_builds_only_what_is_missing(trees):
    cmds = _cmds()
    flat = " ".join(" ".join(c) for c in cmds)
    assert "2026-09-15" in flat
    assert "episode_001" in flat
    assert "episode_000.h5" not in flat


def test_one_command_per_date(trees):
    dates = {c[c.index("--date") + 1] for c in _cmds() if "--date" in c}
    assert dates == {"2026-09-14", "2026-09-15"}


def test_the_scheduled_build_does_not_ask_for_depth(trees):
    """The operator reversed this on 2026-09-16: "以后都先不要处理depth".

    This test used to assert the OPPOSITE -- that `--with-depth` was on every
    scheduled build -- and by then it contradicted
    `test_encode_and_depth_policy.py::test_the_build_stage_no_longer_asks_for_depth`
    outright. Two tests asserting opposite things about one flag means the
    suite can no longer say what the policy is, so the reversed one is stated
    here rather than left to fail. The flag itself still exists for whoever
    wants it; only the SCHEDULED path stopped asking.
    """
    assert all("--with-depth" not in c for c in _cmds())


def test_nothing_missing_means_nothing_to_run(trees, monkeypatch):
    monkeypatch.setattr(PS, "coverage",
                        lambda s, t: PS.Coverage("build", 3, 3, []))
    assert _cmds() == []


def test_an_explicit_date_still_wins(trees):
    """The manual path -- `--date` typed by an operator -- must keep working."""
    cmds = [[str(x) for x in c] for c in PS._build(task="rope", date="2026-09-14")]
    dates = {c[c.index("--date") + 1] for c in cmds if "--date" in c}
    assert dates == {"2026-09-14"}
