"""Where the data is — from the environment, not from someone's disk.

The published READMEs told a reader to run `scripts/build_splits.py`. The
scripts were not published, and when they were checked they carried absolute
paths to one laptop. A reproduction section nobody but the author can run is
worse than none: it reads as a promise.

Three roots, each overridable:

    REACT_RELEASE   the release tree — episodes.jsonl, splits.json, meta/,
                    videos/, segments.json, bad_frames.json.
                    This is what `yxma/React` publishes.
    REACT_FORCE     a release tree whose meta/ carries the force columns.
                    Defaults to REACT_RELEASE, which is correct for the
                    published dataset; it is separate here only because this
                    repo stages force exports before promoting them.
    REACT_RAW       the raw HDF5 capture tree. NOT published — it is ~1 TB.
                    Only steps that need original frames use it, and they say
                    so when it is missing rather than failing on a path.

Defaults point at this repo's working layout so nothing changes for the author;
`REACT_RELEASE=... python scripts/...` is all a user needs.
"""
from __future__ import annotations

import os
from pathlib import Path

_DEF_RELEASE = "/media/yxma/Disk1/twm/release"
_DEF_FORCE = "/media/yxma/Disk1/twm/release_force"
_DEF_RAW = "/media/yxma/Disk1/twm/data"
_DEF_TESTSET = "/media/yxma/Disk1/twm/probe_testset"


def release_root(task: str | None = None) -> Path:
    p = Path(os.environ.get("REACT_RELEASE", _DEF_RELEASE))
    return p / task if task else p


def force_meta(task: str = "motherboard") -> Path:
    """meta/ with the force columns. Falls back to the plain release."""
    p = Path(os.environ.get("REACT_FORCE", _DEF_FORCE)) / task / "meta"
    if p.is_dir():
        return p
    return release_root(task) / "meta"


def raw_root(task: str | None = None) -> Path:
    p = Path(os.environ.get("REACT_RAW", _DEF_RAW))
    return p / task if task else p


def testset_root() -> Path:
    return Path(os.environ.get("REACT_TESTSET", _DEF_TESTSET))


def require(path: Path, what: str, env: str) -> Path:
    """Fail with the variable to set, not with a stranger's directory name."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{what} not found at {path}. Set {env} to your copy, e.g.\n"
            f"    {env}=/path/to/react python {os.path.basename(__file__)}\n"
            f"(REACT_RAW is the original HDF5 capture tree, which the release "
            f"does not publish — steps needing it are marked in the README.)")
    return Path(path)


def out_root(name: str) -> Path:
    """Where a build script writes its artefacts.

    `REACT_OUT` or the working layout. Build scripts wrote to absolute paths
    on one disk, which made them unpublishable: a reader could run them and
    get a permission error rather than a page.
    """
    return Path(os.environ.get("REACT_OUT", "/media/yxma/Disk1/twm")) / name
