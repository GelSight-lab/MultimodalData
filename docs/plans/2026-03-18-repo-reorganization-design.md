# Repo Reorganization Design

**Goal:** Restructure the repo from a single Franka BC pipeline into a multi-project monorepo that cleanly separates shared sensor infrastructure from project-specific code.

---

## Motivation

The repo started as a Franka BC / probing data collection pipeline (`probing_panda`). The TWM project (robot-free multimodal collection) has been added, and more projects are expected. Scripts, configs, and loose files are currently intermixed at the root level, making it hard to understand what belongs to what.

---

## Guiding Principles

- **Shared infrastructure stays at the top level** — `camera_stream/`, `ft_sensor/`, `optitrack/`, `misc/` are used across projects and keep their existing import paths (no import churn).
- **Each project is a self-contained subdirectory** — with its own `scripts/`, config, and any project-specific modules.
- **`tests/` stays at the root** — pytest discovers all tests from one place.
- **`docs/` stays at the root** — design docs span projects.

---

## Target Structure

```
MultimodalData/
├── camera_stream/               # shared sensor abstraction (unchanged)
├── ft_sensor/                   # shared sensor abstraction (unchanged)
├── optitrack/                   # shared sensor abstraction (unchanged)
├── misc/                        # shared utilities (unchanged)
│
├── probing_panda/               # Project 1: Franka BC / probing pipeline
│   ├── __init__.py              # (existing)
│   ├── bc_policy.py             # (existing)
│   ├── displacement_data_collection.py  # (existing)
│   ├── dxlgripper_interface.py  # (existing)
│   ├── probingpanda_endeffector-config.json  # moved from root
│   ├── config/                  # moved from root config/
│   │   ├── bc.yaml
│   │   ├── config.yaml
│   │   └── object/
│   └── scripts/                 # moved from root scripts/ (BC/insertion/disp scripts)
│       ├── bc_online_eval.py
│       ├── bc_traj_collection.py
│       ├── confirm_camera_order.py
│       ├── disp_collection.py
│       ├── download_bc_checkpoints_from_sc.py
│       ├── example_gelsight_stream.py   # moved from root
│       ├── find_gelsight_sensors.py     # moved from root
│       ├── guide_mode_pose.py
│       ├── switch_insertion_full.py
│       ├── sync_bc_checkpooint_to_panda.sh
│       ├── sync_bc_data_from_panda.sh
│       ├── test_gripper.py
│       ├── test_raspi_cam.py
│       ├── train_bc.py
│       ├── usb_insertion_full.py
│       ├── vga_insertion_full.py
│       └── visualize_bc_devices.py
│
├── twm/                         # Project 2: TWM multimodal data collection
│   ├── __init__.py
│   ├── data_collection.py       # moved from scripts/twm_data_collection.py
│   └── visualize.py             # moved from scripts/visualize_twm_data.py
│
├── tests/                       # all tests (unchanged)
│   ├── __init__.py
│   ├── test_realsense_stream.py
│   ├── test_optitrack_stream.py
│   └── test_hdf5_writer.py
│
├── docs/                        # design docs (unchanged)
├── pyproject.toml               # updated: add twm package, fix config path
└── README.md                    # updated to reflect new structure
```

---

## Key Changes

| Item | From | To |
|------|------|----|
| TWM data collection script | `scripts/twm_data_collection.py` | `twm/data_collection.py` |
| TWM visualizer | `scripts/visualize_twm_data.py` | `twm/visualize.py` |
| BC/insertion/disp scripts | `scripts/*.py` | `probing_panda/scripts/*.py` |
| Config files | `config/` | `probing_panda/config/` |
| Endeffector config JSON | root | `probing_panda/` |
| Loose root scripts | root | `probing_panda/scripts/` |
| `pyproject.toml` packages | `["probing_panda", "camera_stream", "optitrack", "ft_sensor", "misc"]` | add `"twm"` |

---

## Impact on Imports

- Shared packages (`camera_stream`, `optitrack`, etc.) — **no import changes**.
- `probing_panda` scripts that reference `config/` via hydra — **path in config needs updating** to `probing_panda/config/`.
- `twm/data_collection.py` imports — no changes (imports are from shared packages).
- Tests — no changes (they import from shared packages, not from scripts).

---

## Out of Scope

- Renaming or refactoring existing modules
- Changing the `misc/` package (used by existing code)
- Moving `tests/` into per-project subdirectories (keep flat for simplicity)
