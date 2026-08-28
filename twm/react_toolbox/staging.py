"""A staging directory that removes itself.

`load_calibration` wants a directory holding `calibration/`, so scripts that
have a bare epoch directory copy it into a temporary one first. Twenty-three
call sites did `tempfile.mkdtemp()`; none removed it. On the author's machine
that left 1,296 orphaned directories, and with the clean-room rebuild staging
273 MB of episodes per run it filled a 469 GB disk -- after which unrelated
commands failed on ENOSPC with nothing pointing at the cause.

These scripts are published. A reader who runs them fills their disk too.

atexit rather than a context manager because most call sites hold the path for
the whole run and wrapping each in a `with` would reindent the body of every
main() -- a large diff for a small fix, and large diffs are where the real
mistakes hide.
"""
from __future__ import annotations

import atexit
import shutil
import tempfile
from pathlib import Path

_MADE: list[Path] = []


def _sweep() -> None:
    for d in _MADE:
        shutil.rmtree(d, ignore_errors=True)


atexit.register(_sweep)


def staging_dir(prefix: str = "react-stage-") -> Path:
    """A fresh temporary directory, deleted when the process exits."""
    d = Path(tempfile.mkdtemp(prefix=prefix))
    _MADE.append(d)
    return d
