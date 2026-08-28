
"""Published scripts must not leave their staging directories behind.

Every script here that needs a calibration in `load_calibration`'s layout
does `tempfile.mkdtemp()` and copies into it. Not one removed it. On this
machine that had left 1,296 orphaned directories and, with the clean-room
rebuild staging 273 MB of episodes per run, it filled a 469 GB disk to
100% -- at which point unrelated commands started failing on ENOSPC with
no obvious connection to the cause.

These scripts are published; a user who runs them fills their disk too.

Counts real directories before and after a real subprocess run, because
the failure is a side effect on the filesystem and nothing short of
looking at the filesystem can see it.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

RESULTS: list[tuple[bool, str, str]] = []
TMP = Path("/tmp")
# cheap, no network, and each stages a calibration the way the rest do
SCRIPTS = ("test_frames.py", "test_probe_frame_and_collision.py")


def check(ok: bool, name: str, evidence: str) -> None:
    RESULTS.append((bool(ok), name, evidence))


def _dirs() -> set[str]:
    return {p.name for p in TMP.glob("tmp*") if p.is_dir()}


def main() -> int:
    root = Path(__file__).resolve().parent
    for s in SCRIPTS:
        before = _dirs()
        p = subprocess.run([sys.executable, str(root / s)],
                           capture_output=True, text=True,
                           cwd=str(root.parent))
        leaked = sorted(_dirs() - before)
        check(p.returncode == 0 and not leaked,
              f"{s} leaves no staging directory behind",
              f"exit {p.returncode}; {len(leaked)} leaked"
              + (f": {leaked[:3]}" if leaked else ""))

    w = max(len(x) for _, x, _ in RESULTS)
    print()
    for ok, name, ev in RESULTS:
        print(f"  [{'ok' if ok else 'FAIL'}] {name:<{w}}  {ev}")
    n = sum(not x for x, _, _ in RESULTS)
    print(f"\ntemp leak: {len(RESULTS)} checks, {n} failing")
    return 1 if n else 0


if __name__ == "__main__":
    raise SystemExit(main())
