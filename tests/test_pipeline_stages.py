"""The whole chain in one place, in the order it has to run.

`run_stages` covered four of the eight stages a release actually needs. The
other four — curate, the Z-up conversion, the cut, and publish — lived in
`react_preprocess.__main__`, in a standalone script, and in the publisher, with
nothing that knew the order or the dependencies between them.

The cost of that, measured on this release: the build ran without --with-depth
and had to be redone; force estimation was never run and surfaced four stages
later at the publish gate; the Z-up tree was built from 29 of 72 episodes and
the cut silently ran on the stale half. Each was a missing prerequisite that
only appeared when something downstream failed.

A stage list is not a schedule. What makes this one is that every stage
declares what it needs, and `plan()` refuses to start one whose prerequisite
has not produced anything.
"""
import sys

import pytest

from twm import pipeline_stages as PS


def test_the_stages_are_the_whole_chain_in_order():
    assert [s.name for s in PS.STAGES] == [
        "build", "force", "curate", "export", "zup", "segment", "index",
        "verify", "publish"]


def _needs(stage):
    """A tuple when a stage has more than one prerequisite. `export` writes the
    force columns (needs the npz) AND stamps a world-frame declaration read out
    of episodes.jsonl (needs curate); expressing only one of those is how the
    missing force estimation surfaced four stages later, at the publish gate."""
    n = stage.needs
    return () if n is None else ((n,) if isinstance(n, str) else tuple(n))


def test_every_stage_after_the_first_declares_what_it_needs():
    known = {x.name for x in PS.STAGES}
    for s in PS.STAGES[1:]:
        assert _needs(s), f"{s.name} declares no prerequisite"
        for n in _needs(s):
            assert n in known, f"{s.name} needs {n!r}, which is not a stage"


def test_the_order_is_consistent_with_the_dependencies():
    """Each stage's prerequisites must come before it, or the list is a lie."""
    seen = set()
    for s in PS.STAGES:
        for n in _needs(s):
            assert n in seen, f"{s.name} runs before its prerequisite {n}"
        seen.add(s.name)


def test_the_cut_comes_after_the_zup_conversion():
    """The segment module says so in its own docstring: the cut is the LAST
    stage, after force recovery and the Z-up conversion, so every column those
    added is carried through by the same row slice."""
    names = [s.name for s in PS.STAGES]
    assert names.index("zup") < names.index("segment")
    assert names.index("force") < names.index("zup")


def test_publish_is_last_and_needs_the_verification():
    assert PS.STAGES[-1].name == "publish"
    assert PS.STAGES[-1].needs == "verify"


def test_a_plan_skips_what_it_is_told_to_and_keeps_the_rest_in_order():
    plan = PS.plan(skip={"build", "force"})
    # export goes with force: it exists to write the force columns, and
    # without the npz it fails on its first episode after everything ahead of
    # it has run. The operator paused force estimation pending a new
    # algorithm, not to salvage half of it.
    assert [s.name for s in plan] == ["curate", "zup", "segment",
                                      "index", "verify", "publish"]


def test_planning_only_to_a_stage_stops_there():
    plan = PS.plan(until="segment")
    assert [s.name for s in plan][-1] == "segment"
    assert "publish" not in {s.name for s in plan}


def test_an_unknown_stage_name_is_refused_rather_than_ignored(tmp_path):
    """Silently ignoring a typo in --skip would run a stage the operator
    believed was skipped."""
    with pytest.raises(KeyError, match="forse"):
        PS.plan(skip={"forse"})


# ── the three failures this exists to prevent ───────────────────────────────

def test_the_depth_channel_is_off_and_that_is_a_decision(monkeypatch):
    """Ten hours of building were once thrown away because --with-depth was
    left off by ACCIDENT and nothing noticed until the layout check. The
    operator turned it off deliberately on 2026-09-16 — it is the largest
    single cost in this stage — so what must hold now is the opposite of the
    original assertion, and the distinction is the point: the flag still
    exists, it is simply not asked for.
    """
    cmds = PS.BY_NAME["build"].commands(task="pushT", date="2026-09-12")
    assert cmds, "the build stage produced no command at all"
    assert not any("--with-depth" in [str(x) for x in c] for c in cmds)


def test_export_is_blocked_when_its_predecessor_has_produced_nothing(monkeypatch):
    """Force estimation was skipped and surfaced four stages later, at the
    publish gate. The prerequisite has to be checked before the work, not
    after."""
    monkeypatch.setattr(PS, "FORCE_ROOT", PS.Path("/nonexistent"))
    why = PS.blocked(PS.BY_NAME["export"], "pushT")
    assert why and "force" in why


def test_the_cut_is_blocked_when_the_zup_tree_is_missing(monkeypatch):
    """The Z-up tree was rebuilt from 29 of 72 episodes and the cut ran on the
    stale half without complaint."""
    monkeypatch.setattr(PS, "RELEASE_ZUP", PS.Path("/nonexistent"))
    why = PS.blocked(PS.BY_NAME["segment"], "pushT")
    assert why and "zup" in why


def test_a_stage_with_its_prerequisite_satisfied_is_not_blocked(tmp_path, monkeypatch):
    from twm.react_preprocess.complete import STREAMS

    d = tmp_path / "pushT" / "meta" / "2026-09-12"
    d.mkdir(parents=True)
    video_dir = tmp_path / "pushT/videos/2026-09-12/episode_000"
    video_dir.mkdir(parents=True)
    for stream in STREAMS:
        (video_dir / f"{stream}.mp4").write_bytes(b"encoded stream")
    (d / "episode_000._detect.pt").write_bytes(b"sidecar")
    (d / "episode_000.parquet").write_bytes(b"x")
    monkeypatch.setattr(PS, "RELEASE", tmp_path)
    monkeypatch.setattr(PS, "DATA_ROOT", tmp_path / "no-source")
    assert PS.blocked(PS.BY_NAME["force"], "pushT") is None


def test_status_reports_every_stage(monkeypatch):
    monkeypatch.setattr(PS, "RELEASE", PS.Path("/nonexistent"))
    monkeypatch.setattr(PS, "FORCE_ROOT", PS.Path("/nonexistent"))
    monkeypatch.setattr(PS, "RELEASE_ZUP", PS.Path("/nonexistent"))
    monkeypatch.setattr(PS, "RELEASE_CUT", PS.Path("/nonexistent"))
    st = dict(PS.status("pushT"))
    assert set(st) == {s.name for s in PS.STAGES}
    assert st["build"] is False and st["segment"] is False


# ── the tail of the chain: what the CUT tree still needs ────────────────────

def test_the_cut_tree_gets_its_own_indices_before_anything_is_published():
    """Cutting creates new publishing units (`episode_001_seg00`), and they
    need their own bad_frames / segments / splits. An episode present in
    episodes.jsonl but absent from splits.json is read as TRAIN by
    `ReactVideoDataset._split_filter` — a silent training leak, not an error.
    """
    names = [s.name for s in PS.STAGES]
    assert "index" in names
    assert names.index("segment") < names.index("index") < names.index("publish")
    assert PS.BY_NAME["index"].needs == "segment"


def test_the_index_stage_builds_splits():
    cmds = PS.BY_NAME["index"].commands(task="pushT")
    flat = " ".join(" ".join(c) for c in cmds)
    assert "build_splits" in flat, "splits.json is never built"
    assert "curate" in flat, "the cut tree never gets bad_frames/segments"


def test_publish_uploads_the_cut_tree_not_the_uncut_one():
    """The Hub holds segments; uploading the uncut tree beside them would put
    `episode_001` and `episode_001_seg00` in one folder, the same frames
    counted twice."""
    cmds = PS.BY_NAME["publish"].commands(task="pushT")
    flat = " ".join(" ".join(c) for c in cmds)
    assert "release_cut" in flat, "publish still points at the uncut tree"


def check_command(cmd, stage_name: str) -> int:
    """Assert every `--flag` in `cmd` appears in its script's --help.

    Returns how many flags it checked.

    Asking the command itself, rather than reading the `.py`, is what covers
    `-m module subcommand` — the first version only read files and therefore
    skipped every `-m twm.react_preprocess <sub>` call, which is exactly where
    the next bad flag was (`curate --root`, undefined).
    """
    import subprocess

    flags = {a for a in cmd if a.startswith("--")}
    if not flags:
        return 0
    head = list(cmd)
    for i, a in enumerate(head):
        if a.startswith("--"):
            head = head[:i]
            break
    r = subprocess.run([*head, "--help"], capture_output=True, text=True,
                       cwd=str(PS.REPO), timeout=120)
    help_text = r.stdout + r.stderr
    # A script that cannot start has no flags to be missing. Without this the
    # comparison runs against a traceback, every flag is "absent", and the
    # failure names the wrong defect entirely.
    assert r.returncode == 0, (
        f"{stage_name}: `{' '.join(head[-2:])} --help` exited {r.returncode} — "
        f"the script cannot start, so nothing can be said about its flags:\n"
        f"{help_text.strip()[-500:]}")
    for f in sorted(flags):
        assert f in help_text, (
            f"{stage_name} passes {f} to `{' '.join(head[-2:])}`, "
            f"which does not accept it")
    return len(flags)


def test_every_flag_the_stages_pass_actually_exists():
    """A stage that passes an option the script does not define dies on
    argparse the moment it runs. Checking the command STRING for a path is not
    the same as checking the script accepts the flag carrying it — the first
    version of this asserted only the former and shipped `--src` to a
    publisher that had no such option."""
    checked = 0
    for stage in PS.STAGES:
        for cmd in stage.commands(task="pushT", date="2026-09-12"):
            checked += check_command(cmd, stage.name)
    assert checked, "no flags were checked — the test proves nothing"


# ── "produced something" is not "produced everything" ───────────────────────

def test_a_stage_that_covered_only_part_of_its_input_is_reported_incomplete(tmp_path, monkeypatch):
    """`_zupped` returned True on finding ONE parquet, so a Z-up tree built
    from 29 of 72 episodes read as done and `segment` cut the stale half
    without complaint. Coverage has to be counted, not sampled."""
    rel = tmp_path / "release"; zup = tmp_path / "zup"
    for i in range(4):
        d = rel / "pushT" / "meta" / "2026-09-12"; d.mkdir(parents=True, exist_ok=True)
        (d / f"episode_{i:03d}.parquet").write_bytes(b"x")
    d2 = zup / "pushT" / "meta" / "2026-09-12"; d2.mkdir(parents=True)
    (d2 / "episode_000.parquet").write_bytes(b"x")     # only one of four
    monkeypatch.setattr(PS, "RELEASE", rel)
    monkeypatch.setattr(PS, "RELEASE_ZUP", zup)
    cov = PS.coverage("zup", "pushT")
    assert cov.done == 1 and cov.total == 4
    assert not cov.complete
    assert "episode_001" in " ".join(cov.missing)


def test_full_coverage_reads_as_complete(tmp_path, monkeypatch):
    rel = tmp_path / "release"; zup = tmp_path / "zup"
    for root in (rel, zup):
        d = root / "pushT" / "meta" / "2026-09-12"; d.mkdir(parents=True)
        for i in range(3):
            (d / f"episode_{i:03d}.parquet").write_bytes(b"x")
    monkeypatch.setattr(PS, "RELEASE", rel)
    monkeypatch.setattr(PS, "RELEASE_ZUP", zup)
    cov = PS.coverage("zup", "pushT")
    assert cov.complete and cov.done == 3 and cov.missing == []


def test_blocking_uses_coverage_so_a_half_done_prerequisite_stops_the_next_stage(
        tmp_path, monkeypatch):
    rel = tmp_path / "release"; zup = tmp_path / "zup"
    d = rel / "pushT" / "meta" / "2026-09-12"; d.mkdir(parents=True)
    for i in range(3):
        (d / f"episode_{i:03d}.parquet").write_bytes(b"x")
    d2 = zup / "pushT" / "meta" / "2026-09-12"; d2.mkdir(parents=True)
    (d2 / "episode_000.parquet").write_bytes(b"x")
    monkeypatch.setattr(PS, "RELEASE", rel)
    monkeypatch.setattr(PS, "RELEASE_ZUP", zup)
    why = PS.blocked(PS.BY_NAME["segment"], "pushT")
    assert why and "1/3" in why, f"half-done prerequisite did not stop it: {why}"


def test_episodes_of_different_dates_are_counted_separately(tmp_path, monkeypatch):
    """The recorder restarts numbering每天, so `episode_000` exists under many
    dates. Keying coverage on the stem alone collapsed 47 motherboard files
    into 18 names and reported a complete stage as two-thirds done — and would
    just as easily report a two-thirds-done stage as complete."""
    rel = tmp_path / "release"
    # Both in scope: the point here is重名 across dates, not the date floor.
    for date in ("2026-09-11", "2026-09-12"):
        d = rel / "motherboard" / "meta" / date
        d.mkdir(parents=True)
        for i in range(3):
            (d / f"episode_{i:03d}.parquet").write_bytes(b"x")
    monkeypatch.setattr(PS, "RELEASE", rel)
    assert len(PS._episodes_in(rel, "motherboard")) == 6


def test_only_recent_sessions_are_in_scope(tmp_path, monkeypatch):
    """The May/June sessions are published and their source H5 is deleted;
    re-cutting them buys nothing and their missing wrist streams and absent
    sources make every gate argue about them. Scope is a date floor."""
    rel = tmp_path / "release"
    for date in ("2026-05-10", "2026-06-18", "2026-09-11", "2026-09-12"):
        d = rel / "pushT" / "meta" / date
        d.mkdir(parents=True)
        (d / "episode_000.parquet").write_bytes(b"x")
    monkeypatch.setattr(PS, "RELEASE", rel)
    got = PS._episodes_in(rel, "pushT")
    assert got == {"2026-09-11/episode_000", "2026-09-12/episode_000"}, got


def test_the_date_floor_is_stated_not_hidden():
    """main carries one week, and the floor is the single place that says so.

    2026-09-09 sits below it deliberately: that session used a different wrist
    camera and ships separately as `validation`. Asserting the bare constant
    only catches a silent edit; asserting what it EXCLUDES catches a floor
    that was moved without the decision behind it being revisited.
    """
    assert PS.SCOPE_SINCE == "2026-09-10"
    assert "2026-09-09" < PS.SCOPE_SINCE, \
        "the validation epoch would be pulled back into main"
    assert "2026-05-11" < PS.SCOPE_SINCE and "2026-06-18" < PS.SCOPE_SINCE

    # The publish defaults to it rather than restating it.
    import twm.scripts.build_release_publish as P
    assert P.SCOPE_SINCE is PS.SCOPE_SINCE


def test_there_is_a_runner_that_executes_the_plan(monkeypatch):
    """A stage graph nobody runs is a diagram. `run_all` walks the plan,
    refuses a stage whose prerequisite is short, and stops at the first
    failure instead of carrying it downstream."""
    calls = []

    def fake_run(self, **kw):
        calls.append((self.name, kw.get("task")))
        return 0

    monkeypatch.setattr(PS.Stage, "run", fake_run)
    monkeypatch.setattr(PS, "blocked", lambda stage, task: None)
    rc = PS.run_all(tasks=("pushT",), skip={"build", "force", "curate", "export", "zup"})
    assert rc == 0
    assert [c[0] for c in calls] == ["segment", "index", "verify", "publish"]


def test_the_runner_stops_at_the_first_failing_stage(monkeypatch):
    def fake_run(self, **kw):
        return 0 if self.name == "segment" else 3

    monkeypatch.setattr(PS.Stage, "run", fake_run)
    monkeypatch.setattr(PS, "blocked", lambda stage, task: None)
    rc = PS.run_all(tasks=("pushT",), skip={"build", "force", "curate", "export", "zup"})
    assert rc == 3


def test_the_runner_refuses_a_stage_whose_prerequisite_is_short(monkeypatch):
    monkeypatch.setattr(PS.Stage, "run", lambda self, **kw: 0)
    monkeypatch.setattr(PS, "blocked",
                        lambda stage, task: "zup covers only 1/9" if stage.name == "segment" else None)
    rc = PS.run_all(tasks=("pushT",), skip={"build", "force", "curate", "export", "zup"})
    assert rc != 0, "a short prerequisite did not stop the run"


def test_an_episode_the_cut_deliberately_dropped_is_not_counted_as_unprocessed(
        tmp_path, monkeypatch):
    """`2026-09-12/episode_001` is 110 frames — 3.7 s — so the cut correctly
    produced nothing for it. That is a RESULT, not an omission, and treating
    it as one blocked the whole pipeline at `index`. The cut already records
    what it dropped; coverage reads that record rather than guessing."""
    import json
    rel = tmp_path / "release"; cut = tmp_path / "cut"
    d = rel / "motherboard" / "meta" / "2026-09-12"; d.mkdir(parents=True)
    for i in range(2):
        (d / f"episode_{i:03d}.parquet").write_bytes(b"x")
    dc = cut / "motherboard" / "meta" / "2026-09-12"; dc.mkdir(parents=True)
    (dc / "episode_000_seg00.parquet").write_bytes(b"x")
    (cut / "motherboard" / "segment_provenance.json").write_text(json.dumps(
        {"dropped_spans": [{"episode": "2026-09-12/episode_001",
                            "frame_range": [0, 109], "n_frames": 110}]}))
    monkeypatch.setattr(PS, "RELEASE", rel)
    monkeypatch.setattr(PS, "RELEASE_CUT", cut)
    cov = PS.coverage("segment", "motherboard")
    assert cov.complete, f"a deliberately dropped episode blocked the run: {cov.missing}"


def test_an_episode_with_no_output_and_no_record_is_still_missing(tmp_path, monkeypatch):
    """The exemption is only for what the cut SAID it dropped. Silence still
    means unprocessed."""
    import json
    rel = tmp_path / "release"; cut = tmp_path / "cut"
    d = rel / "motherboard" / "meta" / "2026-09-12"; d.mkdir(parents=True)
    for i in range(2):
        (d / f"episode_{i:03d}.parquet").write_bytes(b"x")
    dc = cut / "motherboard" / "meta" / "2026-09-12"; dc.mkdir(parents=True)
    (dc / "episode_000_seg00.parquet").write_bytes(b"x")
    (cut / "motherboard" / "segment_provenance.json").write_text(json.dumps(
        {"dropped_spans": []}))
    monkeypatch.setattr(PS, "RELEASE", rel)
    monkeypatch.setattr(PS, "RELEASE_CUT", cut)
    cov = PS.coverage("segment", "motherboard")
    assert not cov.complete and "2026-09-12/episode_001" in cov.missing


def test_verify_certifies_the_tree_that_will_be_published():
    """The certifier hard-coded the uncut tree while publish sends the cut one,
    so it looked for parquets that exist only in the cut tree and died on
    FileNotFoundError — after passing every check it could still run."""
    cmds = PS.BY_NAME["verify"].commands(task="motherboard")
    flat = " ".join(" ".join(c) for c in cmds)
    assert "release_cut" in flat, "verify certifies a different tree than publish uploads"


def test_the_scope_floor_reaches_the_certifier():
    """`SCOPE_SINCE` excluded the May sessions from the scheduler's accounting,
    but the certifier walked the whole tree and failed on 55 of them — data the
    operator had taken out of scope. A floor declared in one place and unknown
    to the next is not a floor."""
    cmds = PS.BY_NAME["verify"].commands(task="motherboard")
    flat = " ".join(" ".join(c) for c in cmds)
    assert PS.SCOPE_SINCE in flat, "the certifier is not told the scope floor"


def test_a_segment_knows_which_recording_it_came_from(tmp_path):
    """The certifier derived the H5 name from the parquet's stem, so
    `episode_000_seg00` sent it looking for `episode_000_seg00.h5` and it
    reported the source as missing while `episode_000.h5` sat there, 14.9 GB.
    The cut writes `source_episode` into every row for exactly this."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from twm.scripts.certify_release import source_recording

    f = tmp_path / "episode_000_seg00.parquet"
    pq.write_table(pa.table({
        "source_episode": ["2026-09-09/episode_000"] * 3,
        "source_h5_frame": np.arange(3, dtype=np.int32),
    }), str(f))
    assert source_recording(f) == "episode_000"


def test_a_parquet_without_the_column_falls_back_to_its_own_stem(tmp_path):
    """Uncut episodes have no source_episode and ARE their own recording."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from twm.scripts.certify_release import source_recording

    f = tmp_path / "episode_007.parquet"
    pq.write_table(pa.table({"source_h5_frame": np.arange(3, dtype=np.int32)}), str(f))
    assert source_recording(f) == "episode_007"


def test_a_stage_runs_for_every_task_before_the_next_stage_starts(monkeypatch):
    """Stage-major, not task-major. `verify` certifies BOTH tasks in one run,
    so finishing motherboard's whole chain first meant verify ran while
    pushT's index had never happened — it died on a bad_frames.json that no
    stage had been given the chance to write."""
    calls = []
    monkeypatch.setattr(PS.Stage, "run",
                        lambda self, **kw: calls.append((self.name, kw.get("task"))) or 0)
    monkeypatch.setattr(PS, "blocked", lambda stage, task: None)
    PS.run_all(tasks=("motherboard", "pushT"),
               skip={"build", "force", "curate", "export", "zup"})
    order = [c[0] for c in calls]
    assert order.index("index") > order.index("segment")
    # every task's index is done before the first verify
    first_verify = order.index("verify")
    assert order[:first_verify].count("index") == 2, \
        f"verify ran before both tasks were indexed: {order}"


def test_a_command_that_cannot_start_is_reported_as_a_crash(tmp_path):
    """Not as a missing flag.

    `build_release_publish.py` imported `twm.pipeline_stages` before anything
    put the repo root on sys.path, so `--help` died with ModuleNotFoundError.
    The checker compared its flags against that traceback, found none, and
    reported "publish passes --src to build_release_publish.py, which does not
    accept it" -- while `--src` was defined, on line 210. A reader chasing that
    message looks for an argparse option that is already there.
    """
    boom = tmp_path / "boom.py"
    boom.write_text("import a_module_that_is_not_installed\n")
    with pytest.raises(AssertionError) as e:
        check_command([sys.executable, str(boom), "--src", "x"], "publish")
    msg = str(e.value)
    assert "does not accept" not in msg, (
        "a script that cannot start was blamed for the flag it never saw")
    assert "--help" in msg and "ModuleNotFoundError" in msg
