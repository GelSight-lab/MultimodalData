"""Freeze the exact V8 code and calibration assets for another PC."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Mapping

from .react_calib import CALIBRATION_NAME
from .run_episode import PIPELINE_VERSION


ASSET_SHA256 = {
    "feature_cache/glowtact_round_mm.json":
        "6ac589b93dd42e77ebf286245f8e354b782f3e748d313dbf1a508b8d62e8c7a3",
    "feature_cache/glowtact_round_8_15_di4.json":
        "9e49f047f5f5b15bc3ca540aa8705e0b660ca17733b00f250ede21da21267e1a",
    "lut_calibration/glowtact_lut.npz":
        "70d145661e8a2aaf26eda91a364b790ba9fd13aad0a469729cbf31856baed925",
}

REQUIREMENTS = """numpy==1.26.4
scipy==1.13.1
scikit-learn==1.1.1
opencv-python==4.5.5.64
h5py==3.14.0
hdf5plugin==6.0.0
pyarrow==21.0.0
Pillow==9.1.1
joblib==1.1.0
huggingface_hub>=0.27,<2
"""

README = r"""# Exact V8 force rebuild for old motherboard data

This directory is a frozen, CPU-only reconstruction package for the 32 old
motherboard episodes from 2026-05-10, 2026-05-11, and 2026-05-19. It requires
the original raw H5 recordings. **Do not use the published tactile MP4 files**:
the measured single-frame difference reached 6.48 N.

## 1. Download

```bash
huggingface-cli download yxma/React \
  --repo-type dataset \
  --include 'old_data/motherboard/v8_rebuild_tools/**' \
            'old_data/motherboard/meta/2026-05-10/*.parquet' \
            'old_data/motherboard/meta/2026-05-11/*.parquet' \
            'old_data/motherboard/meta/2026-05-19/*.parquet' \
  --local-dir react-v8
cd react-v8/old_data/motherboard/v8_rebuild_tools
python3.9 -m venv .venv
source .venv/bin/activate
pip install -r requirements-force-v8.txt
```

## 2. Arrange inputs and set paths

Raw files must be available as:

`$RAW_ROOT/motherboard/<date>/episode_NNN.h5`

Create one run directory and reuse it when resuming:

```bash
export RAW_ROOT=/absolute/path/to/raw
export RUN=/absolute/path/to/old-motherboard-v8-run
export REACT_DATA_ROOT="$RAW_ROOT"
export REACT_STAGE_ROOT="$RUN/input"
export REACT_FORCE_RECOVERY_ROOT="$RUN/force"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
mkdir -p "$REACT_STAGE_ROOT/motherboard/meta" "$REACT_FORCE_RECOVERY_ROOT"
cp -a assets/. "$REACT_FORCE_RECOVERY_ROOT/"
for date in 2026-05-10 2026-05-11 2026-05-19; do
  mkdir -p "$REACT_STAGE_ROOT/motherboard/meta/$date"
  cp ../meta/$date/episode_*.parquet "$REACT_STAGE_ROOT/motherboard/meta/$date/"
done
```

Set all three `REACT_*` variables before starting Python; they are read at
module import time. Confirm there are exactly 32 parquet and 32 raw H5 files.

## 3. Run or resume exact V8

From this directory (the parent of the packaged `twm` module):

```bash
python -m twm.force_recovery.batch_worker 0 1 |& tee -a "$RUN/force.log"
```

Rerun the same command after an interruption. Completed readable V8 outputs
are skipped. One worker is recommended for a mechanical disk; extra workers
increase seeks. The measured runtime on the source PC is about two hours.

## 4. Export force into the action-augmented parquet

Before export, download the latest action-updated files from
`old_data/motherboard/meta/...` into `$REACT_STAGE_ROOT/motherboard/meta/...`,
replacing the input parquet snapshot. Row counts must remain unchanged.

```bash
python -m twm.force_recovery.export_force_columns export \
  --task motherboard --force-only --root "$RUN/release_force"
```

This preserves the action columns and appends the V8 force measurement columns.

## 5. Publish the V8/action superset

Authenticate with `huggingface-cli login`, dry-run, then publish:

```bash
python -m twm.scripts.publish_old_motherboard_update force \
  --root "$RUN/release_force" --dry-run
python -m twm.scripts.publish_old_motherboard_update force \
  --root "$RUN/release_force"
```

The destination is exactly `old_data/motherboard/meta/<date>/...`. The command
requires 32 episodes and refuses any force source not stamped pipeline V8 with
the calibration recorded in `manifest.json`.
"""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_value(source_root: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(source_root), *args],
                            capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _copy_tree(source: Path, destination: Path) -> None:
    for path in sorted(source.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        if path.suffix not in {".py", ".md", ".json", ".toml"}:
            continue
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def build_package(source_root: Path, force_root: Path, output: Path,
                  *, asset_sha256: Mapping[str, str] = ASSET_SHA256) -> dict:
    source_root, force_root, output = map(Path, (source_root, force_root, output))
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.mkdir(parents=True)

    twm_out = output / "twm"
    for rel in ("twm/force_recovery", "twm/react_preprocess"):
        _copy_tree(source_root / rel, output / rel)
    for rel in ("twm/__init__.py", "twm/pipeline_stages.py"):
        source = source_root / rel
        if source.is_file():
            target = output / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    scripts = source_root / "twm/scripts"
    twm_scripts = twm_out / "scripts"
    twm_scripts.mkdir(parents=True, exist_ok=True)
    init = scripts / "__init__.py"
    if init.is_file():
        shutil.copy2(init, twm_scripts / "__init__.py")
    for name in ("publish_old_motherboard_update.py",
                 "export_old_motherboard_actions.py"):
        source = scripts / name
        if source.is_file():
            shutil.copy2(source, twm_scripts / name)

    for rel, expected in sorted(asset_sha256.items()):
        source = force_root / rel
        if not source.is_file():
            raise FileNotFoundError(f"missing V8 asset: {source}")
        actual = _sha256(source)
        if actual != expected:
            raise ValueError(f"asset digest mismatch for {rel}: {actual} != {expected}")
        target = output / "assets" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    (output / "README.md").write_text(README)
    (output / "requirements-force-v8.txt").write_text(REQUIREMENTS)
    diff = _git_value(source_root, "diff", "--binary")
    files = []
    for path in sorted(p for p in output.rglob("*") if p.is_file()
                       and p.name != "manifest.json"):
        files.append({"path": path.relative_to(output).as_posix(),
                      "bytes": path.stat().st_size,
                      "sha256": _sha256(path)})
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": _git_value(source_root, "rev-parse", "HEAD"),
        "source_diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
        "pipeline_version": int(PIPELINE_VERSION),
        "calibration": CALIBRATION_NAME,
        "asset_sha256": dict(sorted(asset_sha256.items())),
        "files": files,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path,
                        default=Path(__file__).resolve().parents[2])
    parser.add_argument("--force-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_package(args.source_root, args.force_root, args.output)
    print(f"files={len(manifest['files'])} pipeline_version={manifest['pipeline_version']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
