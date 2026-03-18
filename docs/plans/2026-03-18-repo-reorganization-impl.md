# Repo Reorganization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Restructure the repo into a multi-project monorepo with shared sensor infrastructure at the top level and project-specific code under `probing_panda/` and `twm/`.

**Architecture:** File moves only — no logic changes. Shared packages (`camera_stream`, `ft_sensor`, `optitrack`, `misc`) stay at root with unchanged imports. `probing_panda/` gains a `scripts/` and `config/` subfolder. `twm/` becomes a new Python package.

**Tech Stack:** Python, hydra (probing_panda config), pytest, git mv

---

## Task 1: Create `twm/` package and move TWM scripts

**Files:**
- Create: `twm/__init__.py`
- Move: `scripts/twm_data_collection.py` → `twm/data_collection.py`
- Move: `scripts/visualize_twm_data.py` → `twm/visualize.py`
- Modify: `tests/test_hdf5_writer.py` (fix import)
- Modify: `pyproject.toml` (add `twm` package)

### Step 1: Create the `twm` package

```bash
mkdir -p /home/yxma/MultimodalData/twm
touch /home/yxma/MultimodalData/twm/__init__.py
```

### Step 2: Move TWM scripts using git mv

```bash
cd /home/yxma/MultimodalData
git mv scripts/twm_data_collection.py twm/data_collection.py
git mv scripts/visualize_twm_data.py twm/visualize.py
```

### Step 3: Fix the HDF5 test import

In `tests/test_hdf5_writer.py`, replace the sys.path hack with a clean package import:

Old (lines 8-10):
```python
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from twm_data_collection import create_episode_file, append_camera_frame, flush_optitrack_to_hdf5
```

New:
```python
from twm.data_collection import create_episode_file, append_camera_frame, flush_optitrack_to_hdf5
```

### Step 4: Add `twm` to `pyproject.toml`

In `pyproject.toml`, update packages list:
```toml
[tool.setuptools]
packages = ["probing_panda", "camera_stream", "optitrack", "ft_sensor", "misc", "twm"]
```

### Step 5: Reinstall package and run tests

```bash
cd /home/yxma/MultimodalData
pip install -e . --no-deps
python -m pytest tests/ -v
```

Expected: all 12 tests pass.

### Step 6: Commit

```bash
git add twm/__init__.py tests/test_hdf5_writer.py pyproject.toml
git commit -m "refactor: create twm package and move TWM scripts"
```

---

## Task 2: Move BC/probing scripts to `probing_panda/scripts/`

**Files:**
- Create: `probing_panda/scripts/` directory
- Move: all non-TWM files from `scripts/` → `probing_panda/scripts/`
- Move: `example_gelsight_stream.py` (root) → `probing_panda/scripts/`
- Move: `find_gelsight_sensors.py` (root) → `probing_panda/scripts/`
- Move: `probingpanda_endeffector-config.json` (root) → `probing_panda/`

### Step 1: Create the scripts directory

```bash
mkdir -p /home/yxma/MultimodalData/probing_panda/scripts
```

### Step 2: Move all remaining scripts with git mv

```bash
cd /home/yxma/MultimodalData
git mv scripts/bc_online_eval.py             probing_panda/scripts/
git mv scripts/bc_traj_collection.py         probing_panda/scripts/
git mv scripts/confirm_camera_order.py       probing_panda/scripts/
git mv scripts/disp_collection.py            probing_panda/scripts/
git mv scripts/download_bc_checkpoints_from_sc.py probing_panda/scripts/
git mv scripts/guide_mode_pose.py            probing_panda/scripts/
git mv scripts/switch_insertion_full.py      probing_panda/scripts/
git mv scripts/sync_bc_checkpooint_to_panda.sh probing_panda/scripts/
git mv scripts/sync_bc_data_from_panda.sh    probing_panda/scripts/
git mv scripts/test_gripper.py               probing_panda/scripts/
git mv scripts/test_raspi_cam.py             probing_panda/scripts/
git mv scripts/train_bc.py                   probing_panda/scripts/
git mv scripts/usb_insertion_full.py         probing_panda/scripts/
git mv scripts/vga_insertion_full.py         probing_panda/scripts/
git mv scripts/visualize_bc_devices.py       probing_panda/scripts/
```

### Step 3: Move loose root files

```bash
cd /home/yxma/MultimodalData
git mv example_gelsight_stream.py            probing_panda/scripts/
git mv find_gelsight_sensors.py              probing_panda/scripts/
git mv probingpanda_endeffector-config.json  probing_panda/
```

### Step 4: Remove now-empty scripts directory

```bash
rmdir /home/yxma/MultimodalData/scripts
```

### Step 5: Verify tests still pass

```bash
python -m pytest tests/ -v
```

Expected: all 12 tests pass (these tests don't import from scripts/).

### Step 6: Commit

```bash
git add -A
git commit -m "refactor: move BC scripts and loose files into probing_panda/"
```

---

## Task 3: Move `config/` to `probing_panda/config/`

**Files:**
- Move: `config/` → `probing_panda/config/`

**Note:** `disp_collection.py` and other probing_panda scripts use `@hydra.main(config_path="../config")`. After moving scripts to `probing_panda/scripts/`, the relative path `../config` resolves to `probing_panda/config/` — exactly where we're moving it. No code changes needed.

### Step 1: Move config directory with git mv

```bash
cd /home/yxma/MultimodalData
git mv config probing_panda/config
```

### Step 2: Verify tests pass

```bash
python -m pytest tests/ -v
```

Expected: all 12 tests pass (tests don't depend on config/).

### Step 3: Commit

```bash
git commit -m "refactor: move config/ into probing_panda/"
```

---

## Task 4: Final cleanup and verification

### Step 1: Verify final directory structure

```bash
find /home/yxma/MultimodalData -not -path "*/.git/*" -not -path "*/__pycache__/*" \
  -not -path "*/data/*" -not -path "*/.pytest_cache/*" -not -path "*.egg-info*" | sort
```

Expected structure:
```
MultimodalData/
├── camera_stream/
├── ft_sensor/
├── misc/
├── optitrack/
├── probing_panda/
│   ├── config/
│   ├── scripts/
│   ├── probingpanda_endeffector-config.json
│   ├── bc_policy.py
│   ├── displacement_data_collection.py
│   ├── dxlgripper_interface.py
│   └── __init__.py
├── twm/
│   ├── __init__.py
│   ├── data_collection.py
│   └── visualize.py
├── tests/
├── docs/
├── pyproject.toml
└── README.md
```

### Step 2: Run TWM data collection to verify it still works

```bash
cd /home/yxma/MultimodalData
python -m twm.data_collection --help 2>&1 || python twm/data_collection.py 2>&1 | head -5
```

Expected: script starts initializing sensors (or prints usage), no ImportError.

### Step 3: Run full test suite one final time

```bash
python -m pytest tests/ -v
```

Expected: 12/12 pass.

### Step 4: Final commit

```bash
git add -A
git commit -m "refactor: complete repo reorganization into multi-project structure"
```
