
"""Run the PUBLISHED scripts against ONLY the published data, in a clean room.

Every reproduction claim so far was verified by me, on my disk, with my
defaults. That proves the scripts work for the person who does not need them.

This downloads the scripts and the data pieces they need from the Hub into a
scratch tree, sets REACT_* to point there, and runs them in a subprocess whose
PYTHONPATH does NOT include this repository. If a script silently depended on
something only present here, it fails.

Not everything can be reproduced from the release: the probe context images
need `videos/`, which is large, and `test_frame_consistency` needs the depth
tree, which is not published. Those are declared here rather than quietly
omitted — an omission is how a reproduction section becomes a promise nobody
checked.

    python scripts/test_reproducible.py
"""
from __future__ import annotations


import json
import os
import shutil
import subprocess
import sys


import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from react_toolbox.staging import staging_dir

RESULTS: list[tuple[bool, str, str]] = []
REPO = "yxma/React"


def check(ok: bool, name: str, evidence: str) -> None:
    RESULTS.append((bool(ok), name, evidence))


def stage(root: Path):
    """Fetch scripts + toolbox + the small release metadata into `root`."""
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    files = api.list_repo_files(REPO, repo_type="dataset")
    got = {"scripts": 0, "toolbox": 0, "meta": 0, "calib": 0, "docs": 0}
    for f in files:
        if f.startswith("scripts/") or f.startswith("toolbox/"):
            p = hf_hub_download(REPO, f, repo_type="dataset", local_dir=str(staging_dir()))
            dest = root / f
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(p, dest)
            got["scripts" if f.startswith("scripts/") else "toolbox"] += 1
    # the toolbox must be importable as `react_toolbox` and as `twm.*`
    (root / "react_toolbox").mkdir(exist_ok=True)
    for f in (root / "toolbox").glob("*"):
        shutil.copy(f, root / "react_toolbox" / f.name)
    (root / "twm").mkdir(exist_ok=True)
    (root / "twm" / "__init__.py").write_text("")
    for name in ("splits.py", "calib_epoch.py"):
        if (root / "toolbox" / name).exists():
            shutil.copy(root / "toolbox" / name, root / "twm" / name)
    rel = root / "release" / "motherboard"
    rel.mkdir(parents=True, exist_ok=True)
    for f in files:
        if f.startswith("data/motherboard/") and f.rsplit("/", 1)[-1] in (
                "episodes.jsonl", "bad_frames.json", "segments.json", "splits.json"):
            p = hf_hub_download(REPO, f, repo_type="dataset", local_dir=str(staging_dir()))
            shutil.copy(p, rel / Path(f).name)
            got["meta"] += 1
        # the calibration, from where the dataset publishes it. calib_dir used
        # to resolve a path relative to its own source file, which exists on one
        # machine; it now looks under $REACT_RELEASE/<task>/calibration first.
        if f.startswith("docs/"):
            src = hf_hub_download(REPO, f, repo_type="dataset", local_dir=str(staging_dir()))
            dest = root / f
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dest)
            got["docs"] = got.get("docs", 0) + 1
        if f.startswith("data/motherboard/calibration/"):
            p = hf_hub_download(REPO, f, repo_type="dataset", local_dir=str(staging_dir()))
            dest = rel / "calibration" / Path(f).name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(p, dest)
            got["calib"] = got.get("calib", 0) + 1
    return got, rel


def run(root: Path, rel: Path, script: str, *args, needs_meta=False):
    env = dict(os.environ)
    env["REACT_RELEASE"] = str(root / "release")
    env["REACT_OUT"] = str(root / "out")
    # PYTHONPATH is the SCRATCH tree only. This repo must not be reachable.
    env["PYTHONPATH"] = str(root)
    p = subprocess.run([sys.executable, str(root / "scripts" / script), *args],
                       capture_output=True, text=True, timeout=900,
                       env=env, cwd=str(root))
    return p


def main() -> int:
    root = staging_dir()
    got, rel = stage(root)
    check(got["scripts"] > 10 and got["toolbox"] > 10 and got["meta"] >= 3
          and got["calib"] >= 5,
          "the published scripts, toolbox, metadata and calibration download",
          f"{got['scripts']} scripts, {got['toolbox']} toolbox modules, "
          f"{got['meta']} metadata files, {got['calib']} calibration files")

    # 1 — build the split from the published metadata alone
    p = run(root, rel, "build_splits.py", "--root", str(rel))
    ok = p.returncode == 0 and (rel / "splits.json").is_file()
    stats = {}
    if ok:
        stats = json.loads((rel / "splits.json").read_text())["stats"]
    check(ok, "build_splits runs on published data with this repo unreachable",
          f"test {stats.get('test_fraction', 0)*100:.1f}%, "
          f"{stats.get('n_test_intervals')} intervals"
          if ok else (p.stderr.strip().splitlines() or ["?"])[-1][:130])

    # 2 — and its own test passes there
    p2 = run(root, rel, "test_splits.py")
    check(p2.returncode == 0, "test_splits passes in the clean room",
          (p2.stdout.strip().splitlines() or ["?"])[-1][:110]
          if p2.returncode == 0 else (p2.stderr.strip().splitlines() or ["?"])[-1][:130])

    # 3 — the split it rebuilds is the one that ships. Byte-identical, because
    #     the same seed on the same episodes must give the same answer; anything
    #     else means a hidden dependency on machine state.
    from huggingface_hub import hf_hub_download
    pub = json.loads(Path(hf_hub_download(
        REPO, "data/motherboard/splits.json", repo_type="dataset",
        local_dir=str(staging_dir()))).read_text())
    mine = json.loads((rel / "splits.json").read_text())
    same = pub["episodes"] == mine["episodes"]
    check(same, "the rebuilt split reproduces the published one exactly",
          f"{len(mine['episodes'])} episodes, intervals identical: {same}")

    # 4 — THE PROBE PACKAGE REBUILDS FROM PUBLISHED DATA. The claim that
    #     matters, and the one that used to be false: until the context images
    #     were switched to the release videos, this needed the unpublished raw
    #     tree. Two episodes are staged so the sampler can only choose from
    #     them, which keeps the download small without special-casing anything.
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    files = api.list_repo_files(REPO, repo_type="dataset")
    eps = sorted({f.split("/")[3] + "/" + f.split("/")[4]
                  for f in files if f.startswith("data/motherboard/videos/")})[:2]
    staged, mb = [], 0
    for ek in eps:
        d_, e_ = ek.split("/")
        want = [f"data/motherboard/meta/{d_}/{e_}.parquet"] + [
            f"data/motherboard/videos/{d_}/{e_}/{s_}.mp4"
            for s_ in ("view_left", "view_middle", "view_right",
                       "tactile_left", "tactile_right")]
        okall = True
        for f in want:
            if f not in files:
                okall = False; break
            src = hf_hub_download(REPO, f, repo_type="dataset",
                                  local_dir=str(staging_dir()))
            dest = rel / Path(f).relative_to("data/motherboard")
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dest)
            mb += dest.stat().st_size / 1e6
        if okall:
            staged.append(ek)
    p4 = run(root, rel, "build_probe_testset.py", "--runs", "1",
             "--out", str(root / "rebuilt"))
    n_ctx = len(list((root / "rebuilt").glob("probes/*/context/*.jpg")))
    man = (root / "rebuilt" / "manifest.json")
    streams = json.loads(man.read_text())["context_streams"] if man.is_file() else []
    check(p4.returncode == 0 and n_ctx == 20 and len(streams) == 5,
          "the probe package rebuilds from published data alone",
          f"{len(staged)} episodes staged ({mb:.0f} MB), 1 run built, "
          f"{n_ctx} context images across {len(streams)} streams"
          if p4.returncode == 0 and n_ctx == 20
          else f"rc={p4.returncode} ctx={n_ctx} streams={len(streams)} | "
               + ((p4.stderr.strip() or p4.stdout.strip()).splitlines()
                  or ["no output"])[-1][:120])

    # 5 — and its own test suite passes on what was just rebuilt
    if p4.returncode == 0:
        env_out = dict(os.environ)
        p5 = subprocess.run(
            [sys.executable, str(root / "scripts" / "test_probe_testset.py")],
            capture_output=True, text=True, timeout=900, cwd=str(root),
            env={**os.environ, "PYTHONPATH": str(root),
                 "REACT_RELEASE": str(root / "release"),
                 "REACT_TESTSET": str(root / "rebuilt")})
        tail = (p5.stdout.strip().splitlines() or ["?"])[-1]
        check(p5.returncode == 0, "its test suite passes on the rebuilt package",
              tail[:110] if p5.returncode == 0
              else (p5.stderr.strip().splitlines() or [tail])[-1][:130])

    # 6 — what CANNOT be reproduced from the release is declared, not hidden
    needs_more = {
        "test_frame_consistency.py": "depth/ (NOT published, ~33 GB)",
        "build_probe_clips.py / build_testset_page.py":
            "a built probe package (reproduced above) plus ffmpeg",
    }
    check(True, "steps that need more than the metadata are named",
          "; ".join(f"{k} needs {v}" for k, v in needs_more.items()))

    w = max(len(x) for _, x, _ in RESULTS)
    print()
    for ok_, name, ev in RESULTS:
        print(f"  [{'ok' if ok_ else 'FAIL'}] {name:<{w}}  {ev}")
    n = sum(not x for x, _, _ in RESULTS)
    print(f"\nreproducible: {len(RESULTS)} checks, {n} failing")
    return 1 if n else 0


if __name__ == "__main__":
    raise SystemExit(main())
