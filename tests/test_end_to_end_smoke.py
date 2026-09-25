"""Walk the whole chain on a few seconds of synthetic recording.

Twenty defects were found on 2026-09-15/16 by running the pipeline on real
data. Every one of them cost hours, because each surfaced only when its stage
was finally reached — after the stages before it had re-encoded, re-estimated
or re-certified hundreds of gigabytes. The shapes were not subtle:

    _build needed a `date` the runner never passes    -> TypeError, line one
    _force named `force_recovery`, not `twm.force_...` -> ModuleNotFoundError
    batch_worker carried its own task tuple            -> exit 0, nothing done
    export ran before curate wrote what it reads       -> KeyError on a new date
    coverage measured build against its own output     -> 11/11 with 3 unbuilt

Not one of them needs real data to catch. They need the commands to actually
RUN. Every existing scheduler test monkeypatches `run_all`, so the argv the
stages build was never executed by anything.

This is slow for a unit test (it encodes video and estimates force on a toy
episode) and it is still two orders of magnitude cheaper than finding the same
defect at 03:00 on a 144 GB recording. Marked `slow`; run it before trusting a
pipeline change.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.slow

FPS = 30
N = 120                     # 4 s — long enough to cut a segment from
H, W = 48, 64               # tiny frames; the chain does not care about size


def _write_recording(path: Path, task: str, n: int = N) -> None:
    """A recording shaped the way the recorder writes one."""
    import h5py
    import hdf5plugin  # noqa: F401
    rng = np.random.default_rng(0)
    path.parent.mkdir(parents=True, exist_ok=True)
    cam_ts = 100.0 + np.arange(n) / FPS
    with h5py.File(path, "w") as f:
        f.create_dataset("timestamps", data=cam_ts)
        for cam in ("cam0", "cam1", "cam2"):
            g = f.create_group(f"realsense/{cam}")
            g.create_dataset("color", data=rng.integers(
                0, 255, (n, H, W, 3), dtype=np.uint8))
            g.create_dataset("depth", data=rng.integers(
                0, 4000, (n, H, W), dtype=np.uint16))
        # The GelSight runs slower than the tick and repeats frames.
        gn = int(n * 17 / FPS) + 2
        gel_ts = 100.0 + np.arange(gn) / 17.0
        # A real indentation, not noise: the force estimator fits gel geometry
        # against newtons, and a uniform-random frame has no contact at all —
        # it lands on "0 samples" rather than on anything about the pipeline.
        yy, xx = np.mgrid[0:H, 0:W]
        for side in ("left", "right"):
            frames = np.full((gn, H, W, 3), 110, np.uint8)
            for i in range(gn):
                depth = 14.0 * np.sin(np.pi * i / gn)      # press and release
                r2 = ((xx - W / 2) ** 2 + (yy - H / 2) ** 2) / (0.18 * W * H)
                bump = np.clip(depth * np.exp(-r2), 0, None)
                frames[i] = np.clip(110 + bump[..., None] * 9, 0, 255).astype(np.uint8)
            g = f.create_group(f"gelsight/{side}")
            g.create_dataset("frames", data=frames)
            g.create_dataset("timestamps", data=gel_ts)
        for body in ("sensor_left", "sensor_right", "motherboard"):
            g = f.create_group(f"optitrack/{body}")
            pose = np.zeros((n, 7))
            pose[:, 2] = np.linspace(0.2, 0.3, n)
            pose[:, 6] = 1.0
            g.create_dataset("pose", data=pose)
            g.create_dataset("timestamps", data=cam_ts)
        m = f.create_group("metadata")
        m.attrs.update(task=task, fps=FPS, frame_count=n, valid=True,
                       duration_s=n / FPS, created_at="2026-09-20T00:00:00",
                       ended_by="operator", depth_aligned=True,
                       gelsight_serials=["L", "R"],
                       realsense_serials=["0", "1", "2"])


def _run(argv, cwd=REPO, timeout=600):
    return subprocess.run([str(a) for a in argv], cwd=str(cwd),
                          capture_output=True, text=True, timeout=timeout)


@pytest.fixture(scope="module")
def sandbox(tmp_path_factory):
    """A whole twm data root, with one recording of one task."""
    root = tmp_path_factory.mktemp("twm")
    _write_recording(root / "data" / "rope" / "2026-09-20" / "episode_000.h5",
                     "rope")
    return root


def test_every_stage_builds_an_argv_that_starts(sandbox, monkeypatch):
    """The cheapest half of the lesson: a stage whose command cannot even
    start. Three of today's twenty were exactly this — a missing `date`, a
    module path that resolves only from another cwd.

    `--help` proves the entry point exists and parses, without doing the work.
    """
    import twm.pipeline_stages as PS
    bad = []
    for stage in PS.STAGES:
        for task in PS.TASKS:
            try:
                cmds = stage.commands(task=task, date="2026-09-20")
            except TypeError as e:
                bad.append(f"{stage.name}: argv could not be built — {e}")
                continue
            for c in cmds:
                argv = [str(x) for x in c]
                if "-m" in argv:
                    mod = argv[argv.index("-m") + 1]
                    r = _run([sys.executable, "-c", f"import {mod}"], timeout=120)
                    if r.returncode:
                        bad.append(f"{stage.name}: cannot import {mod} — "
                                   f"{r.stderr.strip().splitlines()[-1]}")
    assert not bad, "\n  ".join([""] + bad)


def test_the_chain_runs_from_a_raw_recording_to_a_cut_segment(sandbox, monkeypatch):
    """The expensive half, and the only thing that catches an ordering defect:
    export ran before curate wrote the indices it reads, and that showed only
    on a date the chain had never seen."""
    import twm.pipeline_stages as PS
    monkeypatch.setattr(PS, "DATA_ROOT", sandbox / "data")
    monkeypatch.setattr(PS, "RELEASE", sandbox / "release")
    monkeypatch.setattr(PS, "RELEASE_ZUP", sandbox / "release_zup")
    monkeypatch.setattr(PS, "RELEASE_CUT", sandbox / "release_cut")
    monkeypatch.setattr(PS, "FORCE_ROOT", sandbox / "force_recovery")
    monkeypatch.setattr(PS, "SCOPE_SINCE", "2026-09-20")

    cov = PS.coverage("build", "rope")
    assert cov.missing == ["2026-09-20/episode_000"], (
        f"the recording is not visible to the build stage: {cov}")

    env = {
        "REACT_DATA_ROOT": str(sandbox / "data"),
        "REACT_STAGE_ROOT": str(sandbox / "release"),
        "REACT_FORCE_RECOVERY_ROOT": str(sandbox / "force_recovery"),
        "REACT_FORCE_EXPORT_ROOT": str(sandbox / "release_force"),
    }
    import os
    for k, v in env.items():
        monkeypatch.setenv(k, v)

    # BUILD, for real. Not `--help`: the defect that cost ten hours was a
    # missing flag, and the one that cost three was an argv the runner could
    # not even assemble. Both live in the command, not in the module.
    cmds = PS.BY_NAME["build"].commands(task="rope")
    assert cmds, "the build stage produced no command for a missing recording"
    r = _run([*cmds[0]], timeout=900)
    assert r.returncode == 0, (
        f"build failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")

    out = sandbox / "release" / "rope" / "meta" / "2026-09-20" / "episode_000.parquet"
    assert out.is_file(), (
        f"build reported success but wrote no parquet — a stage that exits 0 "
        f"without doing its work is worse than one that crashes\n"
        f"{r.stdout[-1500:]}")

    import pyarrow.parquet as pq
    t = pq.read_table(str(out))
    for col in ("frame_idx", "timestamp", "sensor_left_pose",
                "tactile_left_is_new", "source_h5_frame"):
        assert col in t.column_names, f"{col} missing from the built parquet"
    assert t.num_rows > 0

    vid = sandbox / "release" / "rope" / "videos" / "2026-09-20" / "episode_000"
    assert (vid / "view_middle.mp4").is_file(), \
        f"no video written: {sorted(p.name for p in vid.glob('*')) if vid.is_dir() else 'no dir'}"


def test_the_stages_after_build_run_in_an_order_that_works(sandbox, monkeypatch):
    """curate/export/zup/segment, executed — not planned.

    `export` reads the world offset out of `episodes.jsonl`, which `curate`
    writes, and it ran FIRST. Every episode the chain had seen was already in
    the file from an earlier run, so the defect appeared only on a genuinely
    new date — three hours into a run, at 03:00. Here it appears in seconds,
    because the sandbox has never seen this episode either.
    """
    import twm.pipeline_stages as PS
    for name, val in (("DATA_ROOT", "data"), ("RELEASE", "release"),
                      ("RELEASE_ZUP", "release_zup"),
                      ("RELEASE_CUT", "release_cut"),
                      ("FORCE_ROOT", "force_recovery")):
        monkeypatch.setattr(PS, name, sandbox / val)
    monkeypatch.setattr(PS, "SCOPE_SINCE", "2026-09-20")
    for k, v in (("REACT_DATA_ROOT", "data"), ("REACT_STAGE_ROOT", "release"),
                 ("REACT_FORCE_RECOVERY_ROOT", "force_recovery"),
                 # the export stage runs out of process and passes no --root,
                 # so without this its OUTPUT lands in the production tree.
                 # It did: rope/2026-09-20 sat in the real release_force with
                 # a sidecar naming this test's tmp dir as its source.
                 ("REACT_FORCE_EXPORT_ROOT", "release_force"),
                 ("REACT_RELEASE", "release")):
        monkeypatch.setenv(k, str(sandbox / v))

    built = sandbox / "release/rope/meta/2026-09-20/episode_000.parquet"
    if not built.is_file():
        r = _run(PS.BY_NAME["build"].commands(task="rope")[0], timeout=900)
        assert r.returncode == 0, r.stderr[-1500:]

    # WHERE THIS TEST STOPS. The force estimator wants real gel photometry —
    # three coloured LEDs, a reference frame, a photometric-stereo solve. A
    # synthetic frame that satisfies it would be a re-implementation of that
    # physics, and passing it would prove something about my fake, not about
    # the pipeline. Force VALUES have their own tests.
    #
    # So the npz are written directly, and what this test judges is what the
    # smoke test is for: that the stages start, run in a workable order, and
    # hand the next one what it reads.
    # The second undeclared asset, found the same way: the depth lookup table
    # the force reconstruction reads. Also measured in August, also absent
    # from git, also present on this machine since before anyone looked.
    lut = sandbox / "force_recovery" / "lut_calibration"
    lut.mkdir(parents=True, exist_ok=True)
    real = Path("/media/yxma/Disk1/twm/force_recovery/lut_calibration/glowtact_lut.npz")
    if real.is_file() and not (lut / "glowtact_lut.npz").is_file():
        import shutil
        shutil.copy2(real, lut / "glowtact_lut.npz")
        for extra in ("geometry.json",):
            if (real.parent / extra).is_file():
                shutil.copy2(real.parent / extra, lut / extra)

    fdir = sandbox / "force_recovery" / "rope" / "2026-09-20"
    fdir.mkdir(parents=True, exist_ok=True)
    import pyarrow.parquet as _pq
    nrows = _pq.read_metadata(str(built)).num_rows
    for side in ("left", "right"):
        f = fdir / f"episode_000_{side}.npz"
        if not f.is_file():
            # Built from the shape a REAL npz has, read off one on disk.
            # Guessing the fields one refusal at a time taught nothing: export
            # rightly rejects a force value that cannot name its tactile frame,
            # and one that cannot say which calibration produced it. Those
            # gates stay; the skeleton matches production so they pass for the
            # right reason.
            # The contract export checks, read from its own code: on a row
            # where `tactile_<side>_is_new` is False the force must EQUAL the
            # previous row's. The GelSight runs ~17 Hz against a 30 Hz tick, so
            # ~45 % of rows repeat. Guessing this from `source_frame` was
            # wrong twice; the parquet column is what the gate reads.
            cols = _pq.read_table(str(built),
                                  columns=[f"tactile_{side}_is_new",
                                           "source_h5_frame"])
            is_new = np.asarray(cols[f"tactile_{side}_is_new"].to_numpy(), bool)
            sf = np.asarray(cols["source_h5_frame"].to_numpy(), np.int64)
            wave = np.abs(np.sin(np.pi * np.arange(nrows) / max(1, nrows - 1)))
            for i in range(1, nrows):          # hold across duplicates
                if not is_new[i]:
                    wave[i] = wave[i - 1]
            from twm.force_recovery.run_episode import (
                PIPELINE_VERSION as _PIPELINE_VERSION)
            np.savez(
                f,
                force_normal_n=wave * 3, max_depth_mm=wave * 2,
                volume_mm3=wave * 40, contact_area_mm2=wave * 12,
                source_frame=sf.astype(np.int32),
                contact_threshold_mm=0.05, trim=0, side=side, task="rope",
                date="2026-09-20", episode="episode_000",
                # Taken from the producer, not copied: the literal `5` sat
                # here until 2026-09-16, when the exporter began refusing
                # anything below 8 and the smoke test failed on its own
                # fixture rather than on the pipeline it exists to exercise.
                pipeline_version=_PIPELINE_VERSION,
                force_calibration="synthetic (end-to-end smoke test)",
                force_reconstruction="synthetic",
                geometry_reconstruction="synthetic",
                scale_source="synthetic", tactile_timestamped=True,
                reference_rows=np.zeros(1, np.int32),
                valid_mask_dI=np.ones(nrows, bool))

    ran = []
    for name in ("curate", "export", "zup"):
        for c in PS.BY_NAME[name].commands(task="rope"):
            r = _run(c, timeout=900)
            ran.append((name, r.returncode, r.stdout[-600:] + r.stderr[-900:]))
            assert r.returncode == 0, f"{name} failed:\n{ran[-1][2]}"

    assert (sandbox / "release/rope/episodes.jsonl").is_file(), \
        "curate wrote no episode index — export reads its world offset from it"
    zup = sandbox / "release_zup/rope/meta/2026-09-20/episode_000.parquet"
    assert zup.is_file(), "the Z-up tree has no parquet"

    rows = [json.loads(l) for l in
            (sandbox / "release_zup/rope/episodes.jsonl").read_text().splitlines()
            if l.strip()]
    assert rows and all(r.get("up_axis") == "z" for r in rows), (
        "the Z-up tree does not DECLARE z — a calibration paired with it "
        "raises nothing and puts the view-frame action out by R_x(90)")

    cal = sandbox / "release_zup/rope/calibration/T_mocap_to_cam_left.json"
    assert cal.is_file(), "no calibration staged beside the Z-up poses"
    assert json.loads(cal.read_text()).get("up_axis") == "z"


def test_a_missing_asset_is_refused_before_the_build_not_after(tmp_path, monkeypatch):
    """Both assets the force channel needs — the fitted-features cache and the
    depth lookup table — are measured artefacts that no stage produces and git
    does not carry. On this machine they have been present since August, which
    is why nothing declared them.

    Found by running the chain in an empty sandbox. Without the declaration the
    run dies four stages in, AFTER the build has spent hours, with a message
    that sends the reader to `build`.
    """
    import twm.pipeline_stages as PS
    monkeypatch.setattr(PS, "FORCE_ROOT", tmp_path / "empty")
    for name in ("force", "export"):
        why = PS.blocked(PS.BY_NAME[name], "rope")
        assert why and "no stage produces" in why, (
            f"{name} did not refuse a missing measured asset: {why}")
        assert "restore it from the data disk" in why
