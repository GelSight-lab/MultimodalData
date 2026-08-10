"""Publish the force columns into the React dataset itself.

Why this exists
---------------
The force estimation, its validation, and its documentation all lived in this
repo and on the results Space. The dataset a user actually downloads had
neither: `yxma/React`'s README mentioned force zero times in 214 lines, and
not one of its 36 published parquet files carried a force column. Documenting
"how to use the force data" while the force data was unpublished would have
described something that did not exist.

What it uploads
---------------
`release_force/<task>/meta/<date>/<episode>.parquet` REPLACES the published
parquet at the same path. Each is a strict superset: every original column
byte-for-byte, plus six force columns, plus field-level metadata naming the
units, the source npz, and the stiffness behind every target pose.

The superset property is CHECKED, not assumed — `check_superset` compares each
local file against the published one it replaces and refuses the whole upload
if a single original column is missing, renamed, retyped, or if a row count
differs. A partial overwrite of a public dataset is not something to discover
afterwards.

    python -m force_recovery.upload_force_columns --dry-run
    python -m force_recovery.upload_force_columns
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

EXPORT = Path("/media/yxma/Disk1/twm/release_force")
REPO = "yxma/React"
FORCE_PREFIX = "force_"


def _local_files() -> list[Path]:
    return sorted(EXPORT.rglob("meta/*/*.parquet"))


def _repo_path(p: Path) -> str:
    """release_force/motherboard/meta/<date>/<ep>.parquet -> data/motherboard/..."""
    return "data/" + str(p.relative_to(EXPORT))


def check_superset(paths: list[Path]) -> list[str]:
    """Every local file must contain its published counterpart, unchanged.

    Checks names, arrow types and row count. A force export that quietly
    dropped `sensor_left_pose` would still look like a valid parquet and would
    still load — the loss would surface only in someone else's training run.
    """
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    published = {f.rfilename for f in api.repo_info(REPO, repo_type="dataset").siblings}
    bad = []
    for p in paths:
        rel = _repo_path(p)
        if rel not in published:
            bad.append(f"{rel}: no published file at this path (would be new)")
            continue
        old = pq.read_schema(hf_hub_download(REPO, rel, repo_type="dataset"))
        new = pq.read_schema(p)
        newmap = {f.name: f.type for f in new}
        for f in old:
            if f.name not in newmap:
                bad.append(f"{rel}: DROPS published column {f.name!r}")
            elif newmap[f.name] != f.type:
                bad.append(f"{rel}: retypes {f.name!r} "
                           f"{f.type} -> {newmap[f.name]}")
        added = [n for n in newmap if n.startswith(FORCE_PREFIX)]
        if not added:
            bad.append(f"{rel}: adds no force column at all")
        n_old = pq.read_metadata(hf_hub_download(REPO, rel, repo_type="dataset")).num_rows
        n_new = pq.read_metadata(p).num_rows
        if n_old != n_new:
            bad.append(f"{rel}: {n_new} rows vs published {n_old}")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    paths = _local_files()
    if not paths:
        raise SystemExit(f"no exported parquet under {EXPORT}")
    total = sum(p.stat().st_size for p in paths) / 1e6
    print(f"{len(paths)} parquet ({total:.0f} MB) -> {REPO}")

    print("checking each is a superset of the published file ...", flush=True)
    bad = check_superset(paths)
    if bad:
        print(f"REFUSING TO UPLOAD: {len(bad)} problem(s)")
        for b in bad:
            print("   ", b)
        return 1
    print(f"all {len(paths)} files preserve every published column and row count")

    cols = sorted({n for p in paths for n in pq.read_schema(p).names
                   if n.startswith(FORCE_PREFIX)})
    print("adds:", ", ".join(cols))
    if args.dry_run:
        for p in paths[:4]:
            print("   ", _repo_path(p))
        return 0

    from huggingface_hub import HfApi
    api = HfApi()
    for p in paths:
        api.upload_file(path_or_fileobj=str(p), path_in_repo=_repo_path(p),
                        repo_id=REPO, repo_type="dataset")
    # the per-episode sidecars travel with the data, so a reader never has to
    # come here to learn what stiffness produced a target pose
    for p in sorted(EXPORT.rglob("meta/*/*.force.json")):
        api.upload_file(path_or_fileobj=str(p), path_in_repo=_repo_path(p),
                        repo_id=REPO, repo_type="dataset")
    for name in ("force_export_manifest.json", "force_export_verify.json"):
        f = EXPORT / name
        if f.exists():
            api.upload_file(path_or_fileobj=str(f),
                            path_in_repo=f"data/{name}",
                            repo_id=REPO, repo_type="dataset")
    # RECORD WHAT WAS PUSHED. `release_channel.json` surveys the npz on this
    # disk, and the site turned that into "the published dataset carries this
    # channel" — a claim about a Hugging Face repo inferred from local files.
    # If a promote succeeded and this upload failed, the page would have gone
    # on saying published. The sentence now cites this file, which only exists
    # once the upload has actually returned.
    from .react_calib import CALIBRATION_NAME
    from .run_episode import OUT_ROOT
    rec = OUT_ROOT / "feature_cache" / "force_upload.json"
    rec.parent.mkdir(parents=True, exist_ok=True)
    rec.write_text(json.dumps({
        "repo": REPO,
        "n_parquet": len(paths),
        "force_columns": cols,
        "calibration": CALIBRATION_NAME,
        "manifest_sha": hashlib.sha256(
            (EXPORT / "force_export_manifest.json").read_bytes()
        ).hexdigest()[:16],
    }, indent=1))
    print(f"uploaded; recorded -> {rec}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
