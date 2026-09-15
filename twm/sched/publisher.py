"""One publish pass: find what is finished and unpublished, verify it, ship it.

Idempotent and safe to run at any moment, including while the cut is still
writing. What makes that safe is an ordering property of `segment.cut_episode`:
it writes every stream of an episode BEFORE any of its parquets, so a segment
that has a parquet has all seven of its videos. Selecting on the parquet can
therefore never pick up a half-written mp4.

The verification is not skipped for being inconvenient. `check_sync` decodes
each candidate and checks frame counts, force lag and -- the point of the cut
-- that the tactile-freeze detector comes back EMPTY. A segment that fails is
left unpublished and named.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "/home/yxma/MultimodalData")
PUB = Path("/tmp/claude-1004/-home-yxma-MultimodalData/"
           "d734563d-9427-48c6-a0e9-fe7c75ba0ddf/scratchpad/pub")
CUT = Path("/media/yxma/Disk1/twm/release_cut")
PYT = "/home/yxma/miniconda3/envs/twm/bin/python"
TASKS = ("motherboard", "pushT", "rope")


def published() -> set[tuple[str, str, str]]:
    from huggingface_hub import HfApi
    try:
        files = HfApi().list_repo_files("yxma/React", repo_type="dataset")
    except Exception:                                            # noqa: BLE001
        return set()
    out = set()
    for f in files:
        p = f.split("/")
        if p[0] == "data" and len(p) > 5 and p[2] == "videos":
            out.add((p[1], p[3], p[4]))
    return out


def _wrist_map(task: str) -> dict[str, str]:
    """source episode -> which wrist camera recorded it."""
    import json
    idx = Path(f"/media/yxma/Disk1/twm/release_wave/{task}/episodes.jsonl")
    if not idx.is_file():
        return {}
    return {r["episode"]: r.get("wrist_camera")
            for r in (json.loads(l) for l in idx.read_text().splitlines() if l.strip())}


def ready() -> dict[str, list[tuple[str, str]]]:
    """Complete, unpublished segments per task that data/ will actually take.

    The Arducam-era episodes are deliberately NOT published to data/ (they
    belong with data/validation), so counting them as pending would have the
    scheduler start a publisher every couple of minutes forever, each one
    verifying and then declining to upload the same segment.
    """
    import re as _re
    have = published()
    out = {}
    for task in TASKS:
        root = CUT / task
        if not root.is_dir():
            continue
        wrist = _wrist_map(task)
        got = []
        for pq in sorted((root / "meta").rglob("episode_*.parquet")):
            date, name = pq.parent.name, pq.stem
            if len(list((root / "videos" / date / name).glob("*.mp4"))) != 7:
                continue                       # the cut is still writing it
            if (task, date, name) in have:
                continue
            src = f"{date}/{_re.sub(r'_seg[0-9]+$', '', name)}"
            if wrist.get(src, "usb") != "usb":
                continue                       # Arducam era; belongs in validation
            got.append((date, name))
        if got:
            out[task] = got
    return out


def main() -> int:
    pending = ready()
    if not pending:
        print("没有待发布的内容")
        return 0
    for task, segs in pending.items():
        print(f"[{task}] {len(segs)} 段待发布: {', '.join(n for _, n in segs)}")
    # Verify ONLY what is about to be published. Checking the whole tree made
    # the cost of shipping one new segment grow with everything already
    # shipped, and every segment here has already passed this same check.
    only = Path("/tmp/publish_only.txt")
    only.write_text("".join(f"{task}/{d}/{n}\n"
                            for task, segs in pending.items() for d, n in segs))
    rc = subprocess.run([PYT, str(PUB / "check_sync.py"), str(CUT), str(only)],
                        capture_output=True, text=True)
    tail = rc.stdout.strip().splitlines()[-3:]
    print("\n".join("  " + t for t in tail))
    if rc.returncode != 0:
        print("★ 同步检查未通过，不上传")
        return 1
    for task in pending:
        r = subprocess.run([PYT, str(PUB / "upload_new.py"), "--tasks", task,
                            "--execute"], capture_output=True, text=True)
        for line in r.stdout.splitlines():
            if re.search(r"保留不发布|episodes|合计|已提交|没有要上传", line):
                print("  " + line.strip())
        if r.returncode != 0:
            print(f"★ {task} 上传失败\n{r.stdout[-600:]}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
