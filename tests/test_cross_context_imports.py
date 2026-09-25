"""Library modules must import their siblings the same way from anywhere.

`twm/react_toolbox/` is a package inside `twm/`, and also ships as a
top-level `toolbox/` in the dataset. Library code that says
`from react_toolbox.frames import ...` resolves only when `twm/` happens to
be on sys.path — which is true for scripts under twm/scripts/ and false for
`python -m twm.force_recovery.export_force_columns`, where it killed the run
after the whole export had been computed.
"""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def run_from_repo_root(code: str):
    """A subprocess whose sys.path is the repo root only, like `python -m`."""
    return subprocess.run([sys.executable, "-c", code], cwd=str(ROOT),
                          capture_output=True, text=True)


def test_world_offset_converts_up_axis_from_the_repo_root():
    r = run_from_repo_root(
        "from twm.calib_epoch import world_offset_m;"
        "print(world_offset_m('motherboard', '2026-05-19', 'episode_002', up_axis='y'))")
    assert r.returncode == 0, r.stderr
    assert "ModuleNotFoundError" not in r.stderr


def test_world_frame_fingerprint_imports_from_the_repo_root():
    r = run_from_repo_root(
        "from twm.world_frame import build_declaration;"
        "print(type(build_declaration))")
    assert r.returncode == 0, r.stderr


def test_no_library_module_imports_react_toolbox_bare():
    """Scripts may lean on sys.path; library modules may not."""
    offenders = []
    for p in (ROOT / "twm").rglob("*.py"):
        if {"scripts", "react_toolbox", "tests", "examples"} & set(p.parts):
            continue
        for i, line in enumerate(p.read_text().splitlines(), 1):
            s = line.strip()
            if s.startswith(("from react_toolbox", "import react_toolbox")):
                offenders.append(f"{p.relative_to(ROOT)}:{i}: {s}")
    assert not offenders, "\n".join(offenders)
