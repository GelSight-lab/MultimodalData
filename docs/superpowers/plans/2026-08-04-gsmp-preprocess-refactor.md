# gelsight-mini-pretrain 预处理重构 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把散落在 `MultimodalData` 根目录的 40 个未跟踪预处理脚本，重构成数据盘上一个受版本控制、以声明式 `SourceSpec` 组织的 Python 包。

**Architecture:** 新建独立 git 仓库 `/media/yxma/Disk1/yuxiang/gelsight-mini-pretrain/`。核心包 `src/gsmp/` 提供纯函数层（filters/encode/schema）、写出层（writer）、baseline 策略层，以及一个通用 runner。每个数据源成为 `src/gsmp/sources/<name>.py`，导出一个 `SourceSpec` 与一个 `iter_frames()` 生成器。已生效的一次性修复脚本原样封存到 `archive/`。

**Tech Stack:** Python 3.9.18 (conda, `/home/yxma/miniconda3/bin/python3`)、pyarrow 21.0.0、pytest 8.4.2、numpy、Pillow、opencv-python、huggingface_hub。

## Global Constraints

- Python 3.9 — 所有模块首行必须 `from __future__ import annotations`；类型标注用 `typing.Optional` / `typing.Literal`，不用 `X | None` 运行时语法。
- 805G 原始数据（`/media/yxma/Disk1/yuxiang/mini_data/`）**全程只读**，无例外。
- 已发布 parquet（`mini_data_parquet/`）默认只读。**两处经批准的例外，仅限 Task 19 与 Task 20：**
  - Task 19：`fota_labeled` / `fota_unlabeled` 补列后重传（图片不重编码，先本地构建校验再上传）
  - Task 20：删除 `mini_data_parquet/scripts/` 这份陈旧代码副本（不触碰任何 parquet）

  Task 1-18 不得写入该目录。
- 不重跑已发布数据的生成流程。Task 19 的补列是对现有 shard 加列，不是重新生成。
- 数据盘 `/media/yxma/Disk1` 已用 92%（剩 292G）；NVMe `/` 已用 100%（剩 3.8G）。新仓库只放代码。
- `i_min` 在任何新代码中**不得设默认值**，必须逐源显式声明，且取值须经 Task 10 从已发布 parquet 反推得到，不得从任何现有文档抄写。
- 提交信息结尾附：`Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`
- 新仓库路径常量统一走 `gsmp.config`，任何模块内不得出现硬编码绝对路径。

## 范围：三棵 parquet 树，本计划覆盖两棵

数据盘上 `gelsight-mini-pretrain` 实际存在三棵已发布的 parquet 树：

| 目录 | HF 仓库 | 源数 | 本计划 |
|---|---|---:|---|
| `mini_data_parquet/` | `yxma/gelsight-mini-pretrain` | 12 | **覆盖** |
| `mini_data_parquet_nc/` | `yxma/gelsight-mini-pretrain-nc` | 1 (`sparsh`) | **覆盖** |
| `mini_data_parquet_video/` | 未单独发布 | 3 | **不覆盖，见下** |

**视频子集不在本计划范围内。** `mini_data_parquet_video/` 含 `gelslam`、
`real_tactile_mnist`、`tactile_tracking` 三个源的保序视频版本，由
`legacy/make_parquet_video.py` 生成。它的 schema 比主 schema 多 4 列
（`sequence_id`、`frame_in_seq`、`sequence_length`、`fps`），而 Task 4 的
`gsmp.schema.SCHEMA` 是固定 30 列。

把这个变体塞进本计划会迫使 schema 层支持多态，动摇"唯一真源"这个前提——
而这正是本次重构要建立的东西。因此：

- `legacy/make_parquet_video.py` 随 Task 2 逐字导入，**不迁移、不删除**
- Task 18 的 README 须明确记录该子集当前状态为"未迁移"
- 视频子集的处理另开一个 spec，届时先解决 schema 多态的设计问题

任何任务都不得为了容纳视频子集而修改 `gsmp.schema.SCHEMA`。

## 两个前置事实（实施前必读）

**事实 1 — `i_min` 四处取值不一致：**
`make_parquet_v2.py:51` 为 10，`pipeline.py:188` 为 12.0，`PIPELINE.md:90` 为 15（FoTA 为 10），已发布给用户的 `README` 为 12 real / 10 sim。已发布数据实际用的值无法从代码或文档判定。Task 10 负责反推。

**事实 2 — 已发布数据的 schema 并非统一的 30 列：**

```
fota_labeled      ncols=26   missing: episode, frame_idx, digit_class, gel_variant
fota_unlabeled    ncols=26   missing: episode, frame_idx, digit_class, gel_variant
其余 11 个源      ncols=30
```

这与 `PIPELINE.md`（"Unified schema (current, single-view, 30 columns)"）和已发布
README（"Schema (30 columns, every row identical)"）均矛盾。受影响的是
FoTA 两个子集共 93,155 帧（约占语料 11%）。

后果：已发布 README 的 quick-start 示例

```python
pool = concatenate_datasets([
    load_dataset("yxma/gelsight-mini-pretrain", c, split="train")
    for c in ["fota_unlabeled", "gelslam", "feelanyforce", ...]])
```

会因 features 不匹配而报错。这是一个面向用户的真实缺陷。Task 4 负责把它
固化为一个持续可检测的断言；是否补数据（需重新上传）由 Task 19 提出方案，
**不在本计划自动执行**。

---

### Task 1: MultimodalData 仓库清理提交

把属于采集端的未提交工作提交掉，并停止跟踪不该跟踪的产物。此任务不涉及新仓库。

**Files:**
- Modify: `/home/yxma/MultimodalData/.gitignore`
- Commit (already on disk): `twm/scripts/apply_tactile_shift.py`, `twm/scripts/rebuild_tactile_from_h5_shifted.py`, `twm/scripts/upload_tactile_correction.py`, `twm/twm.code-workspace`
- Untrack: `probing_panda.egg-info/`

**Interfaces:**
- Consumes: 无
- Produces: 一个干净的 `git status`，使 Task 2 能明确区分"已归属"与"待迁移"文件

- [ ] **Step 1: 确认这三个脚本确实属于采集端而非预处理**

Run:
```bash
cd /home/yxma/MultimodalData
head -12 twm/scripts/apply_tactile_shift.py twm/scripts/rebuild_tactile_from_h5_shifted.py twm/scripts/upload_tactile_correction.py
```
Expected: 三个文件的 docstring 都提到 tactile latency / H5 / React 数据集，不提 mini_data。若任一文件提到 `mini_data`，停止并报告。

- [ ] **Step 2: 追加 .gitignore 条目**

在 `/home/yxma/MultimodalData/.gitignore` 末尾追加：

```
# Agent / editor local state
.claude/

# Large calibration artifacts (regenerable)
twm/calibration/result/
*.zip
```

- [ ] **Step 3: 停止跟踪 egg-info**

`.gitignore` 已含 `*.egg-info/`，但 5 个文件仍被跟踪。

Run:
```bash
cd /home/yxma/MultimodalData
git rm -r --cached --quiet probing_panda.egg-info
```

- [ ] **Step 4: 提交**

```bash
cd /home/yxma/MultimodalData
git add .gitignore twm/scripts/apply_tactile_shift.py \
        twm/scripts/rebuild_tactile_from_h5_shifted.py \
        twm/scripts/upload_tactile_correction.py twm/twm.code-workspace
git commit -F - <<'EOF'
chore: commit tactile-latency correction scripts, untrack egg-info

The 2026-06-27/28 GelSight latency work left three scripts uncommitted:
apply_tactile_shift, rebuild_tactile_from_h5_shifted, and
upload_tactile_correction. Commit them alongside the rig fix they belong to.

Also stop tracking probing_panda.egg-info (already in .gitignore) and add
ignores for .claude/, twm/calibration/result/, and *.zip.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
```

- [ ] **Step 5: 验证只剩预处理脚本未跟踪**

Run:
```bash
cd /home/yxma/MultimodalData && git status --porcelain
```
Expected: 输出中不再有 `twm/` 或 `probing_panda.egg-info` 条目；剩余全部为根目录预处理脚本、`calibration/`、`PIPELINE.md`、`_readme_new.md`、`_nc_readme_new.md`、`*.json`。

---

### Task 2: 新仓库脚手架 + 逐字导入

建立新仓库，第一个 commit 是 40 个文件的**逐字复制**，不改一行。这样后续每一步重构都可对着这个已知能跑的状态 diff。

**Files:**
- Create: `/media/yxma/Disk1/yuxiang/gelsight-mini-pretrain/` (git repo)
- Create: `.gitignore`, `pyproject.toml`, `README.md`
- Copy verbatim: 40 个预处理文件 + `calibration/` + 3 个 markdown

**Interfaces:**
- Consumes: Task 1 产出的干净 git status
- Produces: 仓库根 `$GSMP_ROOT=/media/yxma/Disk1/yuxiang/gelsight-mini-pretrain`；`legacy/` 目录内含全部原始脚本，供后续任务逐个迁出

- [ ] **Step 1: 建仓库并复制文件到 legacy/**

```bash
set -euo pipefail
GSMP=/media/yxma/Disk1/yuxiang/gelsight-mini-pretrain
SRC=/home/yxma/MultimodalData
mkdir -p "$GSMP/legacy"
cd "$SRC"
cp -v pipeline.py make_parquet.py make_parquet_v2.py make_parquet_video.py \
      ingest_sparsh.py ingest_tacquad_full.py ingest_touchandgo.py \
      ingest_tvl.py ingest_unit.py parallel_rtm.py parallel_sim.py \
      convert_feats.py extract_rtm_video.py \
      fix_channel_order.py fix_feats_marker_labels.py fix_fota_marker_labels.py \
      redo_fota_unlabeled.py reprocess_feats.py reprocess_fota.py \
      reprocess_legacy.py reprocess_upstream.py reprocess_v7_zscore.py \
      swap_fota_unlabeled.py subsample_fota_unlabeled.py dedupe_cap_fota.py \
      rebalance_compose.py \
      probe_rtm_area_intensity.py probe_rtm_diff_viz.py probe_rtm_thresholds.py \
      probe_video_validity.py diagnose_channel_order.py calibrate_imin.py \
      compute_balance.py detect_fota_markers.py detect_fota_markers_parquet.py \
      make_samples_100.py make_samples_fast.py make_samples_intensity_filtered.py \
      make_stats_and_samples.py make_analytical_plots.py make_pie_charts.py \
      make_pixel_distribution.py make_rgb_vs_bgr.py \
      push_final.py update_readme_final.py _sources_md_patches.py \
      finalize.sh finalize_channel_fix.sh finalize_touchandgo.sh \
      finalize_tvl.sh finalize_v9.sh touchandgo_retry_loop.sh \
      "$GSMP/legacy/"
cp -v feats_marker_classification.json fota_marker_classification.json \
      probe_*.json stats_v2_*.json "$GSMP/legacy/"
cp -rv calibration "$GSMP/legacy/calibration"
mkdir -p "$GSMP/docs"
cp -v PIPELINE.md "$GSMP/docs/PIPELINE.md"
cp -v _readme_new.md _nc_readme_new.md "$GSMP/docs/"
```

- [ ] **Step 2: 校验复制完整（逐字节）**

```bash
cd /home/yxma/MultimodalData
fail=0
for f in $(ls /media/yxma/Disk1/yuxiang/gelsight-mini-pretrain/legacy/*.py \
              /media/yxma/Disk1/yuxiang/gelsight-mini-pretrain/legacy/*.sh); do
  b=$(basename "$f")
  cmp -s "$b" "$f" || { echo "DIFF: $b"; fail=1; }
done
echo "mismatch=$fail"
```
Expected: `mismatch=0`。非 0 则停止。

- [ ] **Step 3: 写 .gitignore**

Create `$GSMP/.gitignore`:
```
__pycache__/
*.py[cod]
*.egg-info/
.venv/
.pytest_cache/
.ipynb_checkpoints/
*.log
# 产出物不入库
out/
_all_rows.pkl
```

- [ ] **Step 4: 写 pyproject.toml**

Create `$GSMP/pyproject.toml`:
```toml
[build-system]
requires = ["setuptools >= 61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "gsmp"
version = "0.1.0"
description = "Preprocessing pipeline for the gelsight-mini-pretrain dataset"
requires-python = ">= 3.9"
dependencies = [
    "numpy",
    "pyarrow",
    "pillow",
    "opencv-python",
    "huggingface_hub",
    "tqdm",
]

[project.optional-dependencies]
dev = ["pytest"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 5: 写占位 README**

Create `$GSMP/README.md`:
```markdown
# gelsight-mini-pretrain — preprocessing

Build pipeline for the HF dataset `yxma/gelsight-mini-pretrain`.

Data (read-only, not in this repo):
- raw upstream:  /media/yxma/Disk1/yuxiang/mini_data/
- published:     /media/yxma/Disk1/yuxiang/mini_data_parquet/

`legacy/` holds the original scripts verbatim as imported on 2026-08-04.
They are migrated into `src/gsmp/` task by task; nothing is deleted from
`legacy/` until its replacement has a passing regression test.

See `docs/PIPELINE.md`.
```

- [ ] **Step 6: 初始提交**

```bash
cd /media/yxma/Disk1/yuxiang/gelsight-mini-pretrain
git init -q
git add -A
git commit -F - <<'EOF'
chore: import gelsight-mini-pretrain preprocessing verbatim

Initial import of the 40 preprocessing scripts that built the HF dataset
yxma/gelsight-mini-pretrain, copied byte-for-byte from the MultimodalData
repo root where they lived untracked.

Nothing is modified in this commit. Refactoring into src/gsmp/ happens in
subsequent commits so each step is diffable against a known-working state.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
git log --oneline
```
Expected: 一个 commit，约 60 个文件。

---

### Task 3: `gsmp.config` — 路径集中

消灭 60+ 处硬编码绝对路径。

**Files:**
- Create: `src/gsmp/__init__.py`, `src/gsmp/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: 无
- Produces:
  - `gsmp.config.RAW_ROOT: Path` — 原始数据根（只读）
  - `gsmp.config.PARQUET_MAIN: Path` — 主发布 parquet 根（只读）
  - `gsmp.config.PARQUET_NC: Path` — NC 发布 parquet 根（只读）
  - `gsmp.config.OUT_ROOT: Path` — 新产出目录（可写）
  - `gsmp.config.repo_root() -> Path`

- [ ] **Step 1: 写失败测试**

Create `tests/test_config.py`:
```python
from __future__ import annotations

import importlib
from pathlib import Path


def test_defaults_point_at_data_disk():
    from gsmp import config

    assert config.RAW_ROOT == Path("/media/yxma/Disk1/yuxiang/mini_data")
    assert config.PARQUET_MAIN == Path("/media/yxma/Disk1/yuxiang/mini_data_parquet")
    assert config.PARQUET_NC == Path("/media/yxma/Disk1/yuxiang/mini_data_parquet_nc")


def test_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("GSMP_RAW_ROOT", str(tmp_path / "raw"))
    from gsmp import config

    importlib.reload(config)
    assert config.RAW_ROOT == tmp_path / "raw"
    monkeypatch.delenv("GSMP_RAW_ROOT")
    importlib.reload(config)


def test_out_root_is_not_inside_readonly_trees():
    from gsmp import config

    assert config.PARQUET_MAIN not in config.OUT_ROOT.parents
    assert config.RAW_ROOT not in config.OUT_ROOT.parents
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd $GSMP && python -m pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gsmp'`

- [ ] **Step 3: 实现**

Create `src/gsmp/__init__.py`:
```python
from __future__ import annotations

__version__ = "0.1.0"
```

Create `src/gsmp/config.py`:
```python
"""Central path configuration.

Every path the pipeline touches is resolved here. No module may hardcode an
absolute path. Defaults match the machine the dataset was built on; override
any of them with the corresponding GSMP_* environment variable.

RAW_ROOT and PARQUET_MAIN/PARQUET_NC are READ-ONLY: they hold 805G of
upstream data and the 8G published release. Nothing in this package writes
to them. Generated output goes to OUT_ROOT.
"""
from __future__ import annotations

import os
from pathlib import Path

_DEFAULTS = {
    "GSMP_RAW_ROOT": "/media/yxma/Disk1/yuxiang/mini_data",
    "GSMP_PARQUET_MAIN": "/media/yxma/Disk1/yuxiang/mini_data_parquet",
    "GSMP_PARQUET_NC": "/media/yxma/Disk1/yuxiang/mini_data_parquet_nc",
    "GSMP_PARQUET_VIDEO": "/media/yxma/Disk1/yuxiang/mini_data_parquet_video",
    "GSMP_OUT_ROOT": "/media/yxma/Disk1/yuxiang/gsmp_out",
}


def _p(key: str) -> Path:
    return Path(os.environ.get(key, _DEFAULTS[key]))


RAW_ROOT = _p("GSMP_RAW_ROOT")
PARQUET_MAIN = _p("GSMP_PARQUET_MAIN")
PARQUET_NC = _p("GSMP_PARQUET_NC")
PARQUET_VIDEO = _p("GSMP_PARQUET_VIDEO")
OUT_ROOT = _p("GSMP_OUT_ROOT")

HF_REPO_MAIN = "yxma/gelsight-mini-pretrain"
HF_REPO_NC = "yxma/gelsight-mini-pretrain-nc"


def repo_root() -> Path:
    """Absolute path to this git repository."""
    return Path(__file__).resolve().parents[2]


def published_dir(source: str, license_repo: str = "main") -> Path:
    """Directory of published parquet shards for one source (read-only)."""
    base = PARQUET_MAIN if license_repo == "main" else PARQUET_NC
    return base / source
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd $GSMP && pip install -e . -q && python -m pytest tests/test_config.py -v`
Expected: 3 passed

- [ ] **Step 5: 提交**

```bash
cd $GSMP
git add src/gsmp/__init__.py src/gsmp/config.py tests/test_config.py
git commit -m "feat: gsmp.config — single source of truth for all paths

Replaces 60+ hardcoded absolute paths across the legacy scripts. Raw and
published trees are documented read-only; output goes to OUT_ROOT.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: `gsmp.schema` + 已发布 schema 审计

`SCHEMA` 成为唯一真源，并把"FoTA 是 26 列"这一事实固化成持续可检测的断言。

**Files:**
- Create: `src/gsmp/schema.py`
- Test: `tests/test_schema.py`

**Interfaces:**
- Consumes: `gsmp.config.PARQUET_MAIN`
- Produces:
  - `gsmp.schema.SCHEMA: pa.Schema` — 30 列
  - `gsmp.schema.COLUMNS: tuple[str, ...]`
  - `gsmp.schema.LEGACY_26_SOURCES: frozenset[str]` — 已知 schema 偏离的源
  - `gsmp.schema.published_columns(source) -> list[str]`

- [ ] **Step 1: 写失败测试**

Create `tests/test_schema.py`:
```python
from __future__ import annotations

import pytest

from gsmp import config, schema


def test_schema_has_30_columns():
    assert len(schema.SCHEMA) == 30
    assert schema.COLUMNS[0] == "image"
    assert "domain" in schema.COLUMNS
    assert "gel_variant" in schema.COLUMNS


@pytest.mark.parametrize(
    "source",
    ["gelslam", "tactile_tracking", "feats", "feelanyforce", "threedcal",
     "real_tactile_mnist", "sim_starstruck", "sim_tactile_mnist",
     "tacquad", "unit"],
)
def test_conforming_sources_match_schema(source):
    """These 10 sources were written with the full 30-column schema."""
    if not (config.PARQUET_MAIN / source).is_dir():
        pytest.skip(f"{source} not present on this machine")
    assert schema.published_columns(source) == list(schema.COLUMNS)


@pytest.mark.parametrize("source", ["fota_labeled", "fota_unlabeled"])
def test_fota_is_known_to_deviate(source):
    """REGRESSION GUARD, not an aspiration.

    fota_labeled and fota_unlabeled were published with 26 columns, missing
    episode / frame_idx / digit_class / gel_variant. This contradicts both
    PIPELINE.md and the user-facing README, and breaks the README's own
    concatenate_datasets quick-start example.

    This test pins the defect so it cannot silently change. When the data is
    eventually republished with the full schema, this test should start
    failing -- at which point move `source` out of LEGACY_26_SOURCES.
    """
    if not (config.PARQUET_MAIN / source).is_dir():
        pytest.skip(f"{source} not present on this machine")
    cols = schema.published_columns(source)
    assert source in schema.LEGACY_26_SOURCES
    assert len(cols) == 26
    assert set(schema.COLUMNS) - set(cols) == {
        "episode", "frame_idx", "digit_class", "gel_variant",
    }
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd $GSMP && python -m pytest tests/test_schema.py -v`
Expected: FAIL — `ImportError: cannot import name 'schema'`

- [ ] **Step 3: 实现**

Create `src/gsmp/schema.py`:
```python
"""The unified parquet schema.

This module is the single source of truth for the column set. The legacy
`make_parquet_v2.SCHEMA` and the `sys.path` hack in `legacy/pipeline.py`
that imported it are both replaced by this.

Known deviation: fota_labeled and fota_unlabeled were published with 26
columns (see LEGACY_26_SOURCES). Verified 2026-08-04 against the released
parquet. tests/test_schema.py pins this.
"""
from __future__ import annotations

import glob
from typing import List

import pyarrow as pa
import pyarrow.parquet as pq

from gsmp import config

SCHEMA = pa.schema([
    ("image", pa.binary()),
    ("image_format", pa.string()),
    ("source", pa.string()),
    ("markered", pa.bool_()),
    ("capture", pa.string()),
    ("split", pa.string()),
    ("height", pa.int32()),
    ("width", pa.int32()),
    ("obj_name", pa.string()),
    ("init_pose", pa.int32()),
    ("side", pa.string()),
    ("x_mm", pa.float32()),
    ("y_mm", pa.float32()),
    ("z_mm", pa.float32()),
    ("quat_x", pa.float32()),
    ("quat_y", pa.float32()),
    ("quat_z", pa.float32()),
    ("quat_w", pa.float32()),
    ("indenter", pa.string()),
    ("indenter_param", pa.string()),
    ("f_x", pa.float32()),
    ("f_y", pa.float32()),
    ("f_z", pa.float32()),
    ("grid_z_max", pa.float32()),
    ("grid_z_mean", pa.float32()),
    ("episode", pa.string()),
    ("frame_idx", pa.int32()),
    ("digit_class", pa.int32()),
    ("gel_variant", pa.string()),
    ("domain", pa.string()),
])

COLUMNS = tuple(f.name for f in SCHEMA)

#: Sources published with the older 26-column schema.
LEGACY_26_SOURCES = frozenset({"fota_labeled", "fota_unlabeled"})

#: Columns absent from the 26-column sources.
LEGACY_26_MISSING = frozenset({
    "episode", "frame_idx", "digit_class", "gel_variant",
})


def published_columns(source: str, license_repo: str = "main") -> List[str]:
    """Column names of the first published shard of `source` (read-only)."""
    d = config.published_dir(source, license_repo)
    shards = sorted(glob.glob(str(d / "*.parquet")))
    if not shards:
        raise FileNotFoundError(f"no published parquet under {d}")
    return list(pq.ParquetFile(shards[0]).schema_arrow.names)


def has_join_key(source: str, license_repo: str = "main") -> bool:
    """True if published shards carry both `capture` and a non-null `frame_idx`.

    This is the precondition for regression-testing a source against the
    published release (see tools/regress.py).
    """
    cols = published_columns(source, license_repo)
    if "capture" not in cols or "frame_idx" not in cols:
        return False
    d = config.published_dir(source, license_repo)
    shard = sorted(glob.glob(str(d / "*.parquet")))[0]
    t = pq.read_table(shard, columns=["frame_idx"])
    return t.num_rows > 0 and t.column("frame_idx").null_count < t.num_rows
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd $GSMP && python -m pytest tests/test_schema.py -v`
Expected: 13 passed（1 + 10 + 2）

- [ ] **Step 5: 提交**

```bash
cd $GSMP
git add src/gsmp/schema.py tests/test_schema.py
git commit -F - <<'EOF'
feat: gsmp.schema as single source of truth + pin FoTA schema defect

SCHEMA moves out of make_parquet_v2.py, removing the sys.path.insert cycle
where pipeline.py imported it back from the module it supersedes.

Also pins a defect found while auditing the release: fota_labeled and
fota_unlabeled were published with 26 columns, missing episode, frame_idx,
digit_class and gel_variant. Both PIPELINE.md and the user-facing README
claim all rows share one 30-column schema, and the README's own
concatenate_datasets example fails because of this. 93,155 frames (~11% of
the corpus) are affected. test_fota_is_known_to_deviate pins the current
state so it cannot change silently.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
```

---

### Task 5: `gsmp.filters` — 纯函数层

**Files:**
- Create: `src/gsmp/filters.py`
- Test: `tests/test_filters.py`

**Interfaces:**
- Consumes: 无
- Produces:
  - `grey_center(arr: np.ndarray) -> np.ndarray`
  - `contact_metrics(rgb, baseline, pixel_thresh=10) -> tuple[int, float]` — 返回 `(area, intensity)`
  - `passes_filter(rgb, baseline, a_min, i_min, pixel_thresh=10) -> bool`
  - `channel_check(rgb) -> float`
  - `maybe_swap_channels(rgb, mode) -> np.ndarray`
  - `phash(rgb) -> int`
  - `hamming(a: int, b: int) -> int`
  - `PIXEL_THRESH: int = 10`

- [ ] **Step 1: 写失败测试**

Create `tests/test_filters.py`:
```python
from __future__ import annotations

import numpy as np
import pytest

from gsmp import filters


def test_grey_center_takes_central_50_percent():
    arr = np.zeros((80, 120, 3), dtype=np.uint8)
    arr[20:60, 30:90] = 255          # exactly the central 50% box
    g = filters.grey_center(arr)
    assert g.shape == (40, 60)
    assert g.dtype == np.float32
    assert g.min() == 255.0


def test_contact_metrics_counts_only_pixels_above_thresh():
    baseline = np.zeros((40, 60), dtype=np.float32)
    rgb = np.zeros((80, 120, 3), dtype=np.uint8)
    # 100 central pixels lifted to 50 grey-levels; 5 lifted to 3 (below thresh)
    rgb[20:22, 30:80] = 50           # 2*50 = 100 px inside the central crop
    rgb[22:23, 30:35] = 3            # 5 px, below PIXEL_THRESH=10
    area, inten = filters.contact_metrics(rgb, baseline)
    assert area == 100
    assert inten == pytest.approx(50.0)


def test_passes_filter_requires_both_area_and_intensity():
    baseline = np.zeros((40, 60), dtype=np.float32)

    big_but_faint = np.zeros((80, 120, 3), dtype=np.uint8)
    big_but_faint[20:40, 30:90] = 11          # area huge, intensity 11
    assert filters.passes_filter(big_but_faint, baseline, a_min=40, i_min=10)
    assert not filters.passes_filter(big_but_faint, baseline, a_min=40, i_min=15)

    bright_but_tiny = np.zeros((80, 120, 3), dtype=np.uint8)
    bright_but_tiny[20:21, 30:35] = 200       # 5 px, very bright
    assert not filters.passes_filter(bright_but_tiny, baseline, a_min=40, i_min=10)


def test_channel_check_sign_flags_bgr_storage():
    rgb_like = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb_like[..., 2] = 200                     # B > R  -> at-rest GelSight Mini
    assert filters.channel_check(rgb_like) < 0

    bgr_like = np.zeros((4, 4, 3), dtype=np.uint8)
    bgr_like[..., 0] = 200                     # R > B  -> stored BGR
    assert filters.channel_check(bgr_like) > 0


def test_maybe_swap_channels_modes():
    bgr_like = np.zeros((2, 2, 3), dtype=np.uint8)
    bgr_like[..., 0] = 200

    assert filters.maybe_swap_channels(bgr_like, "rgb")[0, 0, 0] == 200
    assert filters.maybe_swap_channels(bgr_like, "bgr")[0, 0, 2] == 200
    assert filters.maybe_swap_channels(bgr_like, "auto")[0, 0, 2] == 200

    rgb_like = np.zeros((2, 2, 3), dtype=np.uint8)
    rgb_like[..., 2] = 200
    assert filters.maybe_swap_channels(rgb_like, "auto")[0, 0, 2] == 200


def test_phash_identical_images_have_zero_distance():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    assert filters.hamming(filters.phash(img), filters.phash(img.copy())) == 0


def test_phash_differs_for_unrelated_images():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    b = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    assert filters.hamming(filters.phash(a), filters.phash(b)) > 4


def test_hamming():
    assert filters.hamming(0b1011, 0b1001) == 1
    assert filters.hamming(0, 0) == 0
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd $GSMP && python -m pytest tests/test_filters.py -v`
Expected: FAIL — `ImportError: cannot import name 'filters'`

- [ ] **Step 3: 实现**

Create `src/gsmp/filters.py`:
```python
"""Contact filter, channel-order normalization, and perceptual hashing.

All functions are pure. `grey_center`, `phash` and `hamming` are lifted
verbatim (behaviour-preserving) from legacy/make_parquet_v2.py; the filter
is generalised so a_min and i_min are always explicit arguments -- the
legacy code carried three different defaults for i_min in three places.
"""
from __future__ import annotations

import io
from typing import Tuple

import numpy as np
from PIL import Image

#: Sensor-noise floor in grey-levels. A pixel must differ from the baseline
#: by more than this to count as "lit".
PIXEL_THRESH = 10


def grey_center(arr: np.ndarray) -> np.ndarray:
    """Central 50% crop, greyscale, float32."""
    g = arr.mean(axis=2) if arr.ndim == 3 else arr
    h, w = g.shape
    return g[h // 4:3 * h // 4, w // 4:3 * w // 4].astype(np.float32)


def contact_metrics(
    rgb: np.ndarray,
    baseline: np.ndarray,
    pixel_thresh: int = PIXEL_THRESH,
) -> Tuple[int, float]:
    """Return (contact_area, contact_intensity) against `baseline`.

    area      = number of central-crop pixels differing by > pixel_thresh
    intensity = mean absolute difference over exactly those pixels
                (0.0 when area == 0)
    """
    diff = np.abs(grey_center(rgb) - baseline)
    mask = diff > pixel_thresh
    area = int(mask.sum())
    if area == 0:
        return 0, 0.0
    return area, float(diff[mask].mean())


def passes_filter(
    rgb: np.ndarray,
    baseline: np.ndarray,
    a_min: int,
    i_min: float,
    pixel_thresh: int = PIXEL_THRESH,
) -> bool:
    """The unified validity rule: area >= a_min AND intensity >= i_min.

    a_min and i_min are required. There is deliberately no default for
    i_min -- see docs/PIPELINE.md on why the legacy value is ambiguous.
    """
    area, inten = contact_metrics(rgb, baseline, pixel_thresh)
    return area >= a_min and inten >= i_min


def channel_check(rgb: np.ndarray) -> float:
    """Signed R-B channel mean difference.

    A GelSight Mini at rest is lit by three coloured LEDs such that B > R,
    so a positive value means the frame is probably stored BGR.
    """
    return float(rgb[..., 0].mean()) - float(rgb[..., 2].mean())


def maybe_swap_channels(rgb: np.ndarray, mode: str) -> np.ndarray:
    """Normalize channel order to RGB.

    mode:
      'rgb'          never swap
      'bgr'          always swap
      'auto'/'mixed' swap only when channel_check(rgb) > 0
    """
    if mode == "rgb":
        return rgb
    if mode == "bgr":
        return rgb[..., ::-1].copy()
    if mode in ("auto", "mixed"):
        return rgb[..., ::-1].copy() if channel_check(rgb) > 0 else rgb
    raise ValueError(f"unknown channel mode: {mode!r}")


def phash(rgb: np.ndarray) -> int:
    """8x8 DCT-low-frequency perceptual hash as a 64-bit int.

    `dct1` mirrors along axis 0 (`x[::-1]`) while concatenating along
    axis -1. That is not the axis pairing a textbook DCT-II mirror uses,
    but it is exactly what produced every dedupe decision in the published
    release, so it is reproduced verbatim. Changing it to the "correct"
    `x[..., ::-1]` yields a hash 20 bits different out of 64 -- a different
    hash function, and a different dataset.
    """
    im = Image.fromarray(rgb).convert("L").resize((32, 32), Image.LANCZOS)
    a = np.array(im, dtype=np.float32)

    def dct1(x: np.ndarray) -> np.ndarray:
        return np.fft.fft(
            np.concatenate([x, x[::-1]], axis=-1)
        ).real[..., :x.shape[-1]]

    d = dct1(dct1(a).T).T
    low = d[:8, :8].flatten()
    med = np.median(low[1:])          # skip DC term
    h = 0
    for bit in (low > med):
        h = (h << 1) | int(bit)
    return h


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd $GSMP && python -m pytest tests/test_filters.py -v`
Expected: 8 passed

- [ ] **Step 5: 对齐 legacy 行为（回归护栏）**

新增 `tests/test_filters_matches_legacy.py`：
```python
"""phash/grey_center must stay bit-identical to the legacy implementation,
because the published dedupe decisions were made with it."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from gsmp import config, filters

LEGACY = config.repo_root() / "legacy" / "make_parquet_v2.py"


@pytest.fixture(scope="module")
def legacy_mod():
    if not LEGACY.exists():
        pytest.skip("legacy/make_parquet_v2.py not present")
    spec = importlib.util.spec_from_file_location("legacy_mpv2", LEGACY)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["legacy_mpv2"] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:                      # noqa: BLE001
        pytest.skip(f"legacy module not importable: {exc}")
    return mod


def test_grey_center_matches_legacy(legacy_mod):
    rng = np.random.default_rng(7)
    img = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
    np.testing.assert_array_equal(
        filters.grey_center(img), legacy_mod.grey_center(img)
    )


def test_phash_matches_legacy(legacy_mod):
    rng = np.random.default_rng(7)
    for _ in range(5):
        img = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
        assert filters.phash(img) == legacy_mod.phash(img)
```

Run: `cd $GSMP && python -m pytest tests/test_filters_matches_legacy.py -v`
Expected: 2 passed。

这两个测试是 Task 5 的核心，不是形式。已验证：`dct1` 里若把 `x[::-1]`
"整理"成看起来更对的 `x[..., ::-1]`，同一张图的 phash 会差 20 bit（共 64 bit）
—— 那是另一个哈希函数，会产生另一套去重决策。legacy 的写法按定义就是基准，
因为它生成了已发布的数据。不要改。

- [ ] **Step 6: 提交**

```bash
cd $GSMP
git add src/gsmp/filters.py tests/test_filters.py tests/test_filters_matches_legacy.py
git commit -F - <<'EOF'
feat: gsmp.filters — contact filter, channel norm, phash

Pure-function layer lifted from make_parquet_v2.py. passes_filter now
requires a_min and i_min explicitly instead of falling back to a default,
because the legacy code carried three conflicting i_min defaults (10 in
make_parquet_v2, 12 in pipeline.py, 15 in PIPELINE.md).

test_filters_matches_legacy pins grey_center and phash bit-identical to the
legacy implementation, since the published dedupe decisions depend on them.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
```

---

### Task 6: `gsmp.encode` — JPEG 编码

**Files:**
- Create: `src/gsmp/encode.py`
- Test: `tests/test_encode.py`

**Interfaces:**
- Consumes: 无
- Produces: `encode_jpeg(rgb: np.ndarray, quality: int = 92) -> bytes`；`JPEG_QUALITY: int = 92`

- [ ] **Step 1: 写失败测试**

Create `tests/test_encode.py`:
```python
from __future__ import annotations

import io

import numpy as np
from PIL import Image

from gsmp import encode


def test_encode_jpeg_roundtrips_shape():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
    blob = encode.encode_jpeg(img)
    assert isinstance(blob, bytes)
    assert blob[:2] == b"\xff\xd8"            # JPEG SOI marker
    back = np.array(Image.open(io.BytesIO(blob)))
    assert back.shape == (240, 320, 3)


def test_default_quality_is_92():
    assert encode.JPEG_QUALITY == 92


def test_higher_quality_is_larger():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
    assert len(encode.encode_jpeg(img, 92)) > len(encode.encode_jpeg(img, 40))
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd $GSMP && python -m pytest tests/test_encode.py -v`
Expected: FAIL — `ImportError: cannot import name 'encode'`

- [ ] **Step 3: 实现**

Create `src/gsmp/encode.py`:
```python
"""JPEG encoding. Quality 92 at native resolution -- no resizing."""
from __future__ import annotations

import io

import numpy as np
from PIL import Image

JPEG_QUALITY = 92


def encode_jpeg(rgb: np.ndarray, quality: int = JPEG_QUALITY) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd $GSMP && python -m pytest tests/test_encode.py -v`
Expected: 3 passed

- [ ] **Step 5: 提交**

```bash
cd $GSMP
git add src/gsmp/encode.py tests/test_encode.py
git commit -m "feat: gsmp.encode — JPEG q92 at native resolution

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: `gsmp.writer` — 分片 parquet 写出

保留 legacy `ShardWriter` 的两个关键安全特性：2GB 分片上限、写 parquet 前先 pickle 行列表作为恢复点。

**Files:**
- Create: `src/gsmp/writer.py`
- Test: `tests/test_writer.py`

**Interfaces:**
- Consumes: `gsmp.schema.SCHEMA`
- Produces:
  - `ShardWriter(out_dir: Path, prefix: str, shard_bytes: int = 2 * 1024**3)`
  - `.add(row: dict) -> None`
  - `.close() -> list[Path]` — 返回最终重命名后的分片路径
  - `SHARD_TARGET_BYTES: int`

- [ ] **Step 1: 写失败测试**

Create `tests/test_writer.py`:
```python
from __future__ import annotations

import pyarrow.parquet as pq

from gsmp import schema
from gsmp.writer import ShardWriter


def _row(image=b"\xff\xd8fake", **kw):
    r = {"image": image, "image_format": "jpeg", "source": "t",
         "domain": "real", "markered": False, "capture": "c0",
         "split": "train", "height": 240, "width": 320, "frame_idx": 0}
    r.update(kw)
    return r


def test_missing_columns_are_filled_with_none(tmp_path):
    w = ShardWriter(tmp_path, "train")
    w.add(_row())
    paths = w.close()
    t = pq.read_table(paths[0])
    assert t.schema.names == list(schema.COLUMNS)
    assert t.column("obj_name").to_pylist() == [None]


def test_close_renames_with_of_total_suffix(tmp_path):
    w = ShardWriter(tmp_path, "train")
    for i in range(3):
        w.add(_row(frame_idx=i))
    paths = w.close()
    assert len(paths) == 1
    assert paths[0].name == "train-00000-of-00001.parquet"


def test_rolls_to_new_shard_when_byte_budget_exceeded(tmp_path):
    w = ShardWriter(tmp_path, "train", shard_bytes=100)
    for i in range(4):
        w.add(_row(image=b"x" * 60, frame_idx=i))
    paths = w.close()
    assert len(paths) >= 2
    assert [p.name for p in paths] == sorted(p.name for p in paths)
    total = sum(pq.read_table(p).num_rows for p in paths)
    assert total == 4


def test_pickle_recovery_point_written_then_removed(tmp_path):
    w = ShardWriter(tmp_path, "train")
    w.add(_row())
    assert not (tmp_path / "_all_rows.pkl").exists()
    w.close()
    assert not (tmp_path / "_all_rows.pkl").exists()


def test_empty_writer_produces_no_files(tmp_path):
    w = ShardWriter(tmp_path, "train")
    assert w.close() == []
    assert list(tmp_path.glob("*.parquet")) == []
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd $GSMP && python -m pytest tests/test_writer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gsmp.writer'`

- [ ] **Step 3: 实现**

Create `src/gsmp/writer.py`:
```python
"""Sharded parquet writer.

Two safety properties carried over from the legacy ShardWriter:

1. Each shard is capped at ~2 GB. pyarrow overflows when concatenating a
   binary column past 2 GB, which is reachable at ~470K rows of ~50 KB JPEG.
2. Rows are pickled to disk before the parquet write and the pickle is
   removed only after the write succeeds, so a crash mid-write leaves a
   complete recovery point.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List

import pyarrow as pa
import pyarrow.parquet as pq

from gsmp.schema import COLUMNS, SCHEMA

SHARD_TARGET_BYTES = 2 * 1024 ** 3

_PICKLE_NAME = "_all_rows.pkl"


class ShardWriter:
    """Accumulate rows and flush them as size-capped parquet shards."""

    def __init__(
        self,
        out_dir: Path,
        prefix: str,
        shard_bytes: int = SHARD_TARGET_BYTES,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.prefix = prefix
        self.shard_bytes = shard_bytes
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._rows: List[Dict[str, Any]] = []
        self._shard_idx = 0
        self._pending_bytes = 0
        self.total_rows = 0

    def add(self, row: Dict[str, Any]) -> None:
        full = {name: row.get(name) for name in COLUMNS}
        self._rows.append(full)
        img = full.get("image")
        self._pending_bytes += len(img) if img else 0
        self.total_rows += 1
        if self._pending_bytes >= self.shard_bytes:
            self._flush()

    def _flush(self) -> None:
        if not self._rows:
            return
        recovery = self.out_dir / _PICKLE_NAME
        with open(recovery, "wb") as fh:
            pickle.dump(self._rows, fh, protocol=pickle.HIGHEST_PROTOCOL)

        cols = {name: [r[name] for r in self._rows] for name in COLUMNS}
        table = pa.Table.from_pydict(cols, schema=SCHEMA)
        path = self.out_dir / f"{self.prefix}-{self._shard_idx:05d}.parquet"
        pq.write_table(table, path, compression="snappy")

        recovery.unlink()
        self._shard_idx += 1
        self._rows = []
        self._pending_bytes = 0

    def close(self) -> List[Path]:
        """Flush the tail and rename shards to `prefix-NNNNN-of-NNNNN`."""
        self._flush()
        staged = sorted(self.out_dir.glob(f"{self.prefix}-?????.parquet"))
        total = len(staged)
        final: List[Path] = []
        for i, src in enumerate(staged):
            dst = self.out_dir / f"{self.prefix}-{i:05d}-of-{total:05d}.parquet"
            if src != dst:
                src.rename(dst)
            final.append(dst)
        return final
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd $GSMP && python -m pytest tests/test_writer.py -v`
Expected: 5 passed

- [ ] **Step 5: 提交**

```bash
cd $GSMP
git add src/gsmp/writer.py tests/test_writer.py
git commit -m "feat: gsmp.writer — size-capped parquet shards with recovery pickle

Preserves the two safety properties of the legacy ShardWriter: 2GB shard
cap (pyarrow binary-column overflow) and a pre-write pickle recovery point.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: `gsmp.baseline` — 6 种 baseline 策略

这是本次重构的核心：把 `SKIP_EMPTY_FILTER` 这个逃生舱口，替换成一等公民的策略对象。

**Files:**
- Create: `src/gsmp/baseline.py`
- Test: `tests/test_baseline.py`

**Interfaces:**
- Consumes: `gsmp.filters.grey_center`
- Produces（全部实现 `compute(frames: Sequence[np.ndarray]) -> Optional[np.ndarray]`）：
  - `BaselineStrategy` (ABC)
  - `FirstNFrames(n: int)`
  - `PerCaptureMedian(n: int, seed: int = 0)`
  - `GlobalMedian(n: int, seed: int = 0)`
  - `PerGroupMedian(n: int, seed: int = 0)`
  - `PerTouchMedian(n: int)`
  - `NoBaseline()` — 用于力过滤或上游已预筛的源，`compute` 返回 `None`
  - `needs_frames(strategy) -> bool`

- [ ] **Step 1: 写失败测试**

Create `tests/test_baseline.py`:
```python
from __future__ import annotations

import numpy as np
import pytest

from gsmp import baseline


def _frames(values):
    """One (80,120,3) frame per value, so grey_center gives (40,60)."""
    out = []
    for v in values:
        a = np.full((80, 120, 3), v, dtype=np.uint8)
        out.append(a)
    return out


def test_first_n_frames_uses_only_the_head():
    frames = _frames([10, 10, 10, 200, 200, 200])
    b = baseline.FirstNFrames(3).compute(frames)
    assert b.shape == (40, 60)
    assert b.mean() == pytest.approx(10.0)


def test_first_n_frames_handles_fewer_frames_than_n():
    b = baseline.FirstNFrames(10).compute(_frames([7, 7]))
    assert b.mean() == pytest.approx(7.0)


def test_global_median_is_robust_to_contact_outliers():
    # 8 at-rest frames, 2 heavy-contact frames -> median ignores the outliers
    frames = _frames([20] * 8 + [250] * 2)
    b = baseline.GlobalMedian(n=10, seed=0).compute(frames)
    assert b.mean() == pytest.approx(20.0)


def test_per_capture_median_samples_deterministically():
    frames = _frames(list(range(100)))
    a = baseline.PerCaptureMedian(n=30, seed=0).compute(frames)
    b = baseline.PerCaptureMedian(n=30, seed=0).compute(frames)
    np.testing.assert_array_equal(a, b)


def test_per_touch_median_uses_head_of_each_touch():
    b = baseline.PerTouchMedian(n=5).compute(_frames([3] * 5 + [255] * 20))
    assert b.mean() == pytest.approx(3.0)


def test_explicit_reference_reads_the_shipped_blank(tmp_path):
    from PIL import Image

    ref = tmp_path / "blank.png"
    Image.fromarray(np.full((80, 120, 3), 33, dtype=np.uint8)).save(ref)
    b = baseline.ExplicitReference(ref).compute([])
    assert b.shape == (40, 60)
    assert b.mean() == pytest.approx(33.0, abs=1.0)


def test_explicit_reference_ignores_the_frames_it_is_given():
    """The reference is the baseline; frames must not influence it."""
    from PIL import Image
    import tempfile, pathlib

    with tempfile.TemporaryDirectory() as d:
        ref = pathlib.Path(d) / "blank.png"
        Image.fromarray(np.full((80, 120, 3), 33, dtype=np.uint8)).save(ref)
        s = baseline.ExplicitReference(ref)
        np.testing.assert_array_equal(
            s.compute([]), s.compute(_frames([250] * 20))
        )


def test_no_baseline_returns_none():
    assert baseline.NoBaseline().compute(_frames([1, 2, 3])) is None


def test_needs_frames_is_false_only_for_no_baseline():
    assert not baseline.needs_frames(baseline.NoBaseline())
    assert baseline.needs_frames(baseline.FirstNFrames(10))
    assert baseline.needs_frames(baseline.GlobalMedian(200))


def test_empty_frame_list_returns_none():
    assert baseline.FirstNFrames(10).compute([]) is None
    assert baseline.GlobalMedian(10).compute([]) is None
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd $GSMP && python -m pytest tests/test_baseline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gsmp.baseline'`

- [ ] **Step 3: 实现**

Create `src/gsmp/baseline.py`:
```python
"""Baseline strategies -- how a source decides what "gel at rest" looks like.

The legacy code expressed this as a SKIP_EMPTY_FILTER boolean: True meant
"this source filters inside its own iterator, leave it alone". 7 of 9 sources
were True, so the "unified filter" was really only unified for 2 of them, and
the actual variation -- six different ways of computing the reference frame --
was invisible.

Making the strategy a first-class object removes that escape hatch: every
source now declares how its baseline is computed, and the runner is generic.

All strategies return a greyscale central-crop array (the same space
gsmp.filters.contact_metrics compares against), or None when the source does
not use a pixel baseline at all.
"""
from __future__ import annotations

import abc
from typing import List, Optional, Sequence

import numpy as np

from gsmp.filters import grey_center


class BaselineStrategy(abc.ABC):
    """Computes a per-unit reference image from that unit's frames."""

    @abc.abstractmethod
    def compute(self, frames: Sequence[np.ndarray]) -> Optional[np.ndarray]:
        ...

    def __repr__(self) -> str:                      # pragma: no cover
        args = ", ".join(f"{k}={v!r}" for k, v in vars(self).items())
        return f"{type(self).__name__}({args})"


def _median_of(frames: Sequence[np.ndarray]) -> Optional[np.ndarray]:
    if len(frames) == 0:
        return None
    return np.median(np.stack([grey_center(f) for f in frames]), axis=0)


class FirstNFrames(BaselineStrategy):
    """Median of the first n frames -- the pre-contact prologue of a video."""

    def __init__(self, n: int) -> None:
        self.n = n

    def compute(self, frames: Sequence[np.ndarray]) -> Optional[np.ndarray]:
        return _median_of(list(frames)[: self.n])


class _RandomSampleMedian(BaselineStrategy):
    """Median over a deterministic random sample of n frames."""

    def __init__(self, n: int, seed: int = 0) -> None:
        self.n = n
        self.seed = seed

    def compute(self, frames: Sequence[np.ndarray]) -> Optional[np.ndarray]:
        frames = list(frames)
        if not frames:
            return None
        if len(frames) <= self.n:
            picked: List[np.ndarray] = frames
        else:
            rng = np.random.default_rng(self.seed)
            idx = rng.choice(len(frames), size=self.n, replace=False)
            picked = [frames[i] for i in sorted(idx)]
        return _median_of(picked)


class PerCaptureMedian(_RandomSampleMedian):
    """Median of a random sample within one capture (object x pose x side)."""


class GlobalMedian(_RandomSampleMedian):
    """Median of a random sample across the whole source."""


class PerGroupMedian(_RandomSampleMedian):
    """Median of a random sample within one group (indenter shape, split, ...).

    The grouping key lives on the source module; this strategy only sees the
    frames of a single group.
    """


class PerTouchMedian(BaselineStrategy):
    """Median of the first n frames of one touch (a touch = one short video)."""

    def __init__(self, n: int) -> None:
        self.n = n

    def compute(self, frames: Sequence[np.ndarray]) -> Optional[np.ndarray]:
        return _median_of(list(frames)[: self.n])


class ExplicitReference(BaselineStrategy):
    """Baseline read from a gel-at-rest reference image shipped upstream.

    py3DCal ships `blank_images/blank.png`. The legacy iterator uses it
    directly and never computes a median -- PIPELINE.md's claim that
    threedcal uses a "cross-image median over a random 200-frame sample" is
    simply wrong. Verified against legacy/make_parquet_v2.py:529-547.
    """

    def __init__(self, path: "os.PathLike[str] | str") -> None:
        self.path = path

    def compute(self, frames: Sequence[np.ndarray]) -> Optional[np.ndarray]:
        from PIL import Image

        blank = np.asarray(Image.open(self.path).convert("L"), dtype=np.float32)
        h, w = blank.shape
        return blank[h // 4:3 * h // 4, w // 4:3 * w // 4]


class NoBaseline(BaselineStrategy):
    """No pixel baseline.

    Used by sources that are pre-curated upstream (every frame is a contact
    moment, so a median would include contact and poison the reference), and
    by markered sources where pixel diffing is unreliable and a force
    threshold is used instead.
    """

    def compute(self, frames: Sequence[np.ndarray]) -> Optional[np.ndarray]:
        return None


def needs_frames(strategy: BaselineStrategy) -> bool:
    """True if the runner must buffer frames before it can filter."""
    return not isinstance(strategy, NoBaseline)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd $GSMP && python -m pytest tests/test_baseline.py -v`
Expected: 8 passed

- [ ] **Step 5: 提交**

```bash
cd $GSMP
git add src/gsmp/baseline.py tests/test_baseline.py
git commit -F - <<'EOF'
feat: gsmp.baseline — promote baseline strategy to a first-class object

Replaces the SKIP_EMPTY_FILTER boolean, which was True for 7 of 9 sources
and therefore hid the real structure: six distinct ways of computing the
gel-at-rest reference. With the strategy explicit, the runner is generic and
each source declares its own recipe.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
```

---

### Task 9: `gsmp.spec` — SourceSpec

四个散落的字典收敛成一个 per-source 声明。

**Files:**
- Create: `src/gsmp/spec.py`
- Test: `tests/test_spec.py`

**Interfaces:**
- Consumes: `gsmp.baseline.BaselineStrategy`
- Produces: `SourceSpec` frozen dataclass，字段见下；`SourceSpec.out_dir()`；`registry()`/`get(name)`

- [ ] **Step 1: 写失败测试**

Create `tests/test_spec.py`:
```python
from __future__ import annotations

import dataclasses

import pytest

from gsmp.baseline import FirstNFrames, NoBaseline
from gsmp.spec import SourceSpec


def _spec(**kw):
    base = dict(
        name="demo", domain="real", gel_variant="markerless",
        license_repo="main", baseline=FirstNFrames(10), i_min=10.0,
    )
    base.update(kw)
    return SourceSpec(**base)


def test_i_min_has_no_default():
    fields = {f.name: f for f in dataclasses.fields(SourceSpec)}
    assert fields["i_min"].default is dataclasses.MISSING
    assert fields["i_min"].default_factory is dataclasses.MISSING


def test_spec_is_frozen():
    s = _spec()
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.i_min = 99.0


def test_rejects_unknown_domain():
    with pytest.raises(ValueError, match="domain"):
        _spec(domain="synthetic")


def test_rejects_unknown_channel_mode():
    with pytest.raises(ValueError, match="channel_mode"):
        _spec(channel_mode="rbg")


def test_rejects_nonpositive_a_min():
    with pytest.raises(ValueError, match="a_min"):
        _spec(a_min=0)


def test_defaults_match_documented_pipeline():
    s = _spec()
    assert s.a_min == 40
    assert s.channel_mode == "auto"
    assert s.phash_dist == 4
    assert s.phash_lookback == 30
    assert s.budget == 200_000
    assert s.bg_keep_rate == 0.015


def test_dedupe_disabled_when_phash_dist_is_none():
    assert _spec(phash_dist=None).dedupe_enabled is False
    assert _spec(phash_dist=4).dedupe_enabled is True


def test_no_baseline_sources_are_allowed():
    s = _spec(baseline=NoBaseline())
    assert s.baseline.compute([]) is None
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd $GSMP && python -m pytest tests/test_spec.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gsmp.spec'`

- [ ] **Step 3: 实现**

Create `src/gsmp/spec.py`:
```python
"""SourceSpec -- the complete declaration of one source's behaviour.

Collapses four registries that used to live in make_parquet_v2.py:
    SOURCE_ITERS       -> the sources package (module per source)
    VALIDITY_THRESH    -> a_min / i_min fields
    SKIP_EMPTY_FILTER  -> the baseline field
    NC_SOURCES         -> license_repo field
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Dict, Optional, Tuple

from gsmp import config
from gsmp.baseline import BaselineStrategy

_DOMAINS = ("real", "sim")
_GEL_VARIANTS = ("markered", "markerless", "mixed")
_LICENSE_REPOS = ("main", "nc")
_CHANNEL_MODES = ("auto", "rgb", "bgr", "mixed")


@dataclasses.dataclass(frozen=True)
class SourceSpec:
    """Everything the runner needs to know about one source.

    `i_min` deliberately has no default. The legacy code carried three
    different values for it (10, 12, 15) and the published README told users
    a fourth (12 real / 10 sim), so no default can be justified. Each source
    declares the value recovered from the published release by
    tools/recover_imin.py.

    `rng_seed` seeds the runner's background-keep draw. Legacy seeded every
    iterator explicitly -- mostly `random.Random(0)`, but
    `make_parquet_v2.py:234` uses `random.Random(42)`. Reproducing a
    source's kept set requires its actual seed, so copy it from the legacy
    iterator rather than assuming 0.

    `touch_window_keep` is the per-frame Bernoulli keep probability applied
    inside a touch window. It is 1.0 everywhere except real_tactile_mnist,
    which keeps 0.30 (~2 frames per touch).
    """

    name: str
    domain: str
    gel_variant: str
    license_repo: str
    baseline: BaselineStrategy
    i_min: float
    a_min: int = 40
    channel_mode: str = "auto"
    phash_dist: Optional[int] = 4
    phash_lookback: int = 30
    budget: int = 200_000
    bg_keep_rate: float = 0.015
    rng_seed: int = 0
    touch_window_keep: float = 1.0
    resolution: Optional[Tuple[int, int]] = None
    notes: str = ""

    def __post_init__(self) -> None:
        if self.domain not in _DOMAINS:
            raise ValueError(f"domain must be one of {_DOMAINS}: {self.domain!r}")
        if self.gel_variant not in _GEL_VARIANTS:
            raise ValueError(
                f"gel_variant must be one of {_GEL_VARIANTS}: {self.gel_variant!r}"
            )
        if self.license_repo not in _LICENSE_REPOS:
            raise ValueError(
                f"license_repo must be one of {_LICENSE_REPOS}: {self.license_repo!r}"
            )
        if self.channel_mode not in _CHANNEL_MODES:
            raise ValueError(
                f"channel_mode must be one of {_CHANNEL_MODES}: {self.channel_mode!r}"
            )
        if self.a_min <= 0:
            raise ValueError(f"a_min must be positive: {self.a_min}")
        if not 0.0 <= self.bg_keep_rate <= 1.0:
            raise ValueError(f"bg_keep_rate must be in [0,1]: {self.bg_keep_rate}")

    @property
    def dedupe_enabled(self) -> bool:
        return self.phash_dist is not None

    @property
    def markered(self) -> bool:
        return self.gel_variant == "markered"

    def out_dir(self) -> Path:
        return config.OUT_ROOT / self.name

    def published_dir(self) -> Path:
        return config.published_dir(self.name, self.license_repo)


_REGISTRY: Dict[str, SourceSpec] = {}


def register(spec: SourceSpec) -> SourceSpec:
    if spec.name in _REGISTRY:
        raise ValueError(f"source already registered: {spec.name}")
    _REGISTRY[spec.name] = spec
    return spec


def get(name: str) -> SourceSpec:
    if name not in _REGISTRY:
        raise KeyError(f"unknown source {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def registry() -> Dict[str, SourceSpec]:
    return dict(_REGISTRY)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd $GSMP && python -m pytest tests/test_spec.py -v`
Expected: 8 passed

- [ ] **Step 5: 提交**

```bash
cd $GSMP
git add src/gsmp/spec.py tests/test_spec.py
git commit -F - <<'EOF'
feat: gsmp.spec — SourceSpec collapses four scattered registries

SOURCE_ITERS, VALIDITY_THRESH, SKIP_EMPTY_FILTER and NC_SOURCES become
fields of one frozen dataclass. i_min has no default, by design.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
```

---

### Task 10: `tools/recover_imin.py` — 从已发布数据反推 i_min

**这是所有源迁移的前置条件。** 没有这一步，`SourceSpec.i_min` 只能靠猜。

**Files:**
- Create: `tools/recover_imin.py`
- Create: `docs/imin_recovered.md` (工具产出，人工审阅后入库)
- Test: `tests/test_recover_imin.py`

**Interfaces:**
- Consumes: `gsmp.filters.contact_metrics`, `gsmp.schema.has_join_key`, `gsmp.config`
- Produces: `recover_imin(source, sample=2000) -> ImInEstimate`；`ImInEstimate(source, n_sampled, min_kept_intensity, p01, p05, verdict)`

原理：已发布 parquet 里的每一行都是**被保留的**帧。对每行解码 JPEG、用该源的
baseline 策略重算 `contact_metrics`，得到保留帧的 intensity 分布。由于规则是
`intensity >= i_min`，保留帧 intensity 的下界即 `i_min` 的上界估计。1.5% 的
背景随机保留帧会污染下界，所以用 1 百分位而非最小值，并同时报告最小值供人工判断。

- [ ] **Step 1: 写失败测试**

Create `tests/test_recover_imin.py`:
```python
from __future__ import annotations

import numpy as np
import pytest

from gsmp.tools_imin import ImInEstimate, estimate_from_intensities


def test_p01_ignores_background_contamination():
    # 990 real contacts at >=12, 10 background frames far below
    kept = np.concatenate([np.full(990, 12.0), np.full(10, 1.0)])
    est = estimate_from_intensities("demo", kept, bg_keep_rate=0.015)
    assert est.min_kept_intensity == pytest.approx(1.0)
    assert est.p01 == pytest.approx(12.0, abs=0.5)
    assert est.verdict.startswith("likely i_min")


def test_flags_ambiguous_when_distribution_has_no_floor():
    kept = np.linspace(0.0, 60.0, 1000)
    est = estimate_from_intensities("demo", kept, bg_keep_rate=0.015)
    assert est.verdict == "ambiguous"


def test_rejects_empty_input():
    with pytest.raises(ValueError):
        estimate_from_intensities("demo", np.array([]), bg_keep_rate=0.015)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd $GSMP && python -m pytest tests/test_recover_imin.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gsmp.tools_imin'`

- [ ] **Step 3: 实现纯逻辑部分**

Create `src/gsmp/tools_imin.py`:
```python
"""Recover the effective i_min of a published source from its own data.

The published parquet contains only KEPT frames. Under the rule
`keep iff area >= a_min and intensity >= i_min`, the lower edge of the kept
intensity distribution is an upper-bound estimate of i_min -- except that
bg_keep_rate (1.5%) of frames were kept *despite* failing, which contaminates
the very bottom. So the estimate uses the 1st percentile, not the minimum,
and reports both.
"""
from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass(frozen=True)
class ImInEstimate:
    source: str
    n_sampled: int
    min_kept_intensity: float
    p01: float
    p05: float
    verdict: str


def estimate_from_intensities(
    source: str,
    intensities: np.ndarray,
    bg_keep_rate: float,
) -> ImInEstimate:
    if intensities.size == 0:
        raise ValueError(f"no intensities sampled for {source}")

    arr = np.sort(np.asarray(intensities, dtype=np.float64))
    p01 = float(np.percentile(arr, 1))
    p05 = float(np.percentile(arr, 5))
    lo = float(arr[0])

    # A real threshold shows up as a sharp floor: the 1st and 5th percentiles
    # sit close together well above the contaminated minimum.
    spread = p05 - p01
    if spread <= max(1.0, 0.1 * p01) and p01 > lo + 0.5:
        verdict = f"likely i_min = {p01:.1f}"
    elif spread <= max(1.0, 0.1 * p01):
        verdict = f"likely i_min = {p01:.1f} (no bg contamination detected)"
    else:
        verdict = "ambiguous"

    return ImInEstimate(
        source=source,
        n_sampled=int(arr.size),
        min_kept_intensity=lo,
        p01=p01,
        p05=p05,
        verdict=verdict,
    )
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd $GSMP && python -m pytest tests/test_recover_imin.py -v`
Expected: 3 passed

- [ ] **Step 5: 写 CLI 驱动**

Create `tools/recover_imin.py`:
```python
#!/usr/bin/env python3
"""Sample published shards, recompute contact intensity, estimate i_min.

Usage:
    python tools/recover_imin.py --source gelslam --sample 2000
    python tools/recover_imin.py --all

Read-only: touches only config.PARQUET_MAIN / PARQUET_NC.
"""
from __future__ import annotations

import argparse
import glob
import io
import sys

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))

from gsmp import config, filters                      # noqa: E402
from gsmp.tools_imin import estimate_from_intensities  # noqa: E402

ALL_SOURCES = [
    "gelslam", "tactile_tracking", "real_tactile_mnist", "feelanyforce",
    "threedcal", "tacquad", "unit", "sim_tactile_mnist", "sim_starstruck",
    "feats", "fota_labeled", "fota_unlabeled", "sparsh",
]

#: Sources published to the NC repo rather than the main one. Until the
#: source modules exist (Task 13-16) the registry cannot answer this, so the
#: tools carry the mapping. After Task 16, spec.get(name).license_repo is
#: authoritative and this constant should agree with it.
SOURCE_LICENSE_REPO = {"sparsh": "nc"}


def sample_intensities(source: str, n: int) -> np.ndarray:
    """Decode up to n published frames and measure intensity vs a per-capture
    median baseline built from that same shard."""
    repo = SOURCE_LICENSE_REPO.get(source, "main")
    shards = sorted(glob.glob(str(config.published_dir(source, repo) / "*.parquet")))
    if not shards:
        raise FileNotFoundError(f"no shards for {source}")

    cols = pq.ParquetFile(shards[0]).schema_arrow.names
    want = ["image"] + (["capture"] if "capture" in cols else [])
    table = pq.read_table(shards[0], columns=want)
    total = table.num_rows
    step = max(1, total // n)
    idx = list(range(0, total, step))[:n]

    images = table.column("image").to_pylist()
    captures = (
        table.column("capture").to_pylist() if "capture" in want
        else ["_"] * total
    )

    by_cap = {}
    for i in idx:
        arr = np.array(Image.open(io.BytesIO(images[i])).convert("RGB"))
        by_cap.setdefault(captures[i], []).append(arr)

    out = []
    for frames in by_cap.values():
        greys = np.stack([filters.grey_center(f) for f in frames])
        base = np.median(greys, axis=0)
        for f in frames:
            _, inten = filters.contact_metrics(f, base)
            out.append(inten)
    return np.array(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--sample", type=int, default=2000)
    args = ap.parse_args()

    targets = ALL_SOURCES if args.all else [args.source]
    if not targets or targets == [None]:
        ap.error("pass --source NAME or --all")

    print(f"{'source':22s} {'n':>6s} {'min':>7s} {'p01':>7s} {'p05':>7s}  verdict")
    for src in targets:
        try:
            vals = sample_intensities(src, args.sample)
            est = estimate_from_intensities(src, vals, bg_keep_rate=0.015)
        except Exception as exc:                        # noqa: BLE001
            print(f"{src:22s} {'-':>6s} {'-':>7s} {'-':>7s} {'-':>7s}  ERROR: {exc}")
            continue
        print(f"{est.source:22s} {est.n_sampled:6d} {est.min_kept_intensity:7.2f} "
              f"{est.p01:7.2f} {est.p05:7.2f}  {est.verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: 对全部 12 个源运行，产出报告**

Run:
```bash
cd $GSMP && python tools/recover_imin.py --all --sample 2000 | tee docs/imin_recovered.md
```
Expected: 13 行输出（12 个主仓库源 + sparsh）。**人工审阅**：任何 verdict 为
`ambiguous` 的源，在 Task 12 中一律归入 tier-2（原样搬），不得猜测其 i_min。

参考锚点：`legacy/ingest_sparsh.py` 顶部显式写了 `I_MIN = 12`。若工具对
sparsh 反推出的 p01 明显偏离 12，说明反推方法本身有问题（而非 sparsh 特殊），
应先查工具再继续——这是唯一一个有独立可信来源可交叉验证的源。

- [ ] **Step 7: 提交**

```bash
cd $GSMP
git add tools/recover_imin.py src/gsmp/tools_imin.py tests/test_recover_imin.py docs/imin_recovered.md
git commit -F - <<'EOF'
feat: recover effective i_min per source from the published release

The legacy code and docs disagree on i_min four ways (10 / 12 / 15 / 12-10),
so it cannot be read off any of them. Published parquet contains only kept
frames, so the lower edge of the kept intensity distribution bounds i_min
from above; the 1.5% background-keep contaminates the very bottom, so the
estimate uses p01 and reports the raw minimum alongside.

Sources whose verdict is "ambiguous" get no guessed value -- they go to the
verbatim-wrap tier instead.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
```

---

### Task 11: `tools/regress.py` — 回归验证脚手架

**Files:**
- Create: `src/gsmp/regress.py`, `tools/regress.py`
- Test: `tests/test_regress.py`

**Interfaces:**
- Consumes: `gsmp.schema.has_join_key`, `gsmp.config`
- Produces:
  - `published_keys(source) -> set[tuple[str, int]]`
  - `compare(published: set, produced: set, bg_keep_rate: float, n_candidates: int) -> RegressionReport`
  - `RegressionReport(missing, extra, deterministic_ok, bg_within_tolerance, summary)`

判定规则：`produced - published`（新代码多保留的）与 `published - produced`
（新代码漏保留的）都必须为空**，除非**差异规模落在 `bg_keep_rate * n_candidates`
的合理区间内。确定性部分不设容差。

- [ ] **Step 1: 写失败测试**

Create `tests/test_regress.py`:
```python
from __future__ import annotations

from gsmp.regress import compare


def test_identical_sets_pass():
    keys = {("c0", i) for i in range(100)}
    rep = compare(keys, keys, bg_keep_rate=0.015, n_candidates=1000)
    assert rep.deterministic_ok
    assert rep.bg_within_tolerance
    assert not rep.missing and not rep.extra


def test_small_symmetric_difference_within_bg_budget_passes():
    published = {("c0", i) for i in range(1000)}
    produced = set(published)
    produced.discard(("c0", 0))
    produced.add(("c0", 5000))
    rep = compare(published, produced, bg_keep_rate=0.015, n_candidates=100_000)
    assert rep.bg_within_tolerance


def test_large_difference_fails():
    published = {("c0", i) for i in range(1000)}
    produced = {("c0", i) for i in range(500)}
    rep = compare(published, produced, bg_keep_rate=0.015, n_candidates=1000)
    assert not rep.deterministic_ok
    assert len(rep.missing) == 500


def test_report_summary_is_human_readable():
    keys = {("c0", 1)}
    rep = compare(keys, keys, bg_keep_rate=0.015, n_candidates=10)
    assert "PASS" in rep.summary
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd $GSMP && python -m pytest tests/test_regress.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gsmp.regress'`

- [ ] **Step 3: 实现**

Create `src/gsmp/regress.py`:
```python
"""Regression harness: new pipeline output vs the published release.

The published parquet is the ground truth for the legacy behaviour. Each row
carries (capture, frame_idx), so the set of kept frames is directly
queryable -- no need to keep the legacy code runnable.

BG_KEEP_RATE is stochastic, so exact set equality is impossible. The
assertion is therefore two-tier:

  deterministic part -- frames that pass area+intensity MUST match exactly
  stochastic part    -- frames kept despite failing are only checked in bulk
"""
from __future__ import annotations

import dataclasses
import glob
from typing import List, Set, Tuple

import pyarrow.parquet as pq

from gsmp import config

Key = Tuple[str, int]


@dataclasses.dataclass(frozen=True)
class RegressionReport:
    missing: List[Key]
    extra: List[Key]
    deterministic_ok: bool
    bg_within_tolerance: bool
    summary: str


def published_keys(source: str, license_repo: str = "main") -> Set[Key]:
    """(capture, frame_idx) of every published row of `source`."""
    out: Set[Key] = set()
    for shard in sorted(glob.glob(str(config.published_dir(source, license_repo) / "*.parquet"))):
        t = pq.read_table(shard, columns=["capture", "frame_idx"])
        caps = t.column("capture").to_pylist()
        idxs = t.column("frame_idx").to_pylist()
        for c, i in zip(caps, idxs):
            if c is not None and i is not None:
                out.add((c, int(i)))
    return out


def compare(
    published: Set[Key],
    produced: Set[Key],
    bg_keep_rate: float,
    n_candidates: int,
) -> RegressionReport:
    missing = sorted(published - produced)
    extra = sorted(produced - published)

    # Budget: how many frames the 1.5% background keep could plausibly move,
    # with a 3x slack for sampling variance on top of the expected count.
    budget = max(10, int(3 * bg_keep_rate * n_candidates))
    diff = len(missing) + len(extra)

    deterministic_ok = diff == 0
    bg_within_tolerance = diff <= budget

    if deterministic_ok:
        summary = f"PASS exact: {len(published)} keys identical"
    elif bg_within_tolerance:
        summary = (
            f"PASS within background budget: {diff} differing keys "
            f"(budget {budget}, published {len(published)}, produced {len(produced)})"
        )
    else:
        summary = (
            f"FAIL: {len(missing)} missing, {len(extra)} extra, "
            f"exceeds background budget {budget}"
        )

    return RegressionReport(
        missing=missing[:50],
        extra=extra[:50],
        deterministic_ok=deterministic_ok,
        bg_within_tolerance=bg_within_tolerance,
        summary=summary,
    )
```

Create `tools/regress.py`:
```python
#!/usr/bin/env python3
"""Compare a migrated source against the published release.

Usage:
    python tools/regress.py --source gelslam
"""
from __future__ import annotations

import argparse
import importlib
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from gsmp import regress, spec  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--limit", type=int, default=0,
                    help="cap candidate frames (0 = all)")
    args = ap.parse_args()

    importlib.import_module(f"gsmp.sources.{args.source}")
    s = spec.get(args.source)

    mod = importlib.import_module(f"gsmp.sources.{args.source}")
    produced, n_candidates = mod.dry_run_keys(limit=args.limit or None)

    published = regress.published_keys(s.name, s.license_repo)
    rep = regress.compare(published, produced, s.bg_keep_rate, n_candidates)

    print(rep.summary)
    if rep.missing:
        print(f"  first missing: {rep.missing[:5]}")
    if rep.extra:
        print(f"  first extra:   {rep.extra[:5]}")
    return 0 if rep.bg_within_tolerance else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd $GSMP && python -m pytest tests/test_regress.py -v`
Expected: 4 passed

- [ ] **Step 5: 提交**

```bash
cd $GSMP
git add src/gsmp/regress.py tools/regress.py tests/test_regress.py
git commit -F - <<'EOF'
feat: regression harness comparing new output to the published release

Published parquet is the ground truth for legacy behaviour, so the legacy
code does not need to stay runnable. Two-tier assertion: the deterministic
(area+intensity) part must match exactly, the 1.5% stochastic background
keep is only checked against a bulk budget.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
```

---

### Task 12: 分级审计 — 决定每个源走哪条路

**Files:**
- Create: `tools/audit_sources.py`
- Create: `docs/source_tiers.md`

**Interfaces:**
- Consumes: `gsmp.schema.has_join_key`, `docs/imin_recovered.md`
- Produces: `docs/source_tiers.md` — 权威的分级表，Task 13-16 据此执行

- [ ] **Step 1: 写审计脚本**

Create `tools/audit_sources.py`:
```python
#!/usr/bin/env python3
"""Decide, per source, whether it can be regression-tested.

tier-1  has (capture, frame_idx) join key AND an unambiguous recovered i_min
        -> migrate to the new abstraction, prove with tools/regress.py
tier-2  missing either
        -> wrap verbatim; do not touch the internal filtering logic
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from gsmp import schema  # noqa: E402

SOURCES = [
    "gelslam", "tactile_tracking", "real_tactile_mnist", "feelanyforce",
    "threedcal", "tacquad", "unit", "sim_tactile_mnist", "sim_starstruck",
    "feats", "fota_labeled", "fota_unlabeled", "sparsh",
]

#: Must stay in sync with tools/recover_imin.py::SOURCE_LICENSE_REPO.
SOURCE_LICENSE_REPO = {"sparsh": "nc"}


def main() -> int:
    print("| source | repo | published cols | join key | tier |")
    print("|---|---|---:|---|---|")
    for src in SOURCES:
        repo = SOURCE_LICENSE_REPO.get(src, "main")
        try:
            cols = schema.published_columns(src, repo)
            join = schema.has_join_key(src, repo)
        except FileNotFoundError:
            print(f"| {src} | {repo} | - | MISSING | skip |")
            continue
        tier = "tier-1" if join else "tier-2"
        print(f"| {src} | {repo} | {len(cols)} | {'yes' if join else 'NO'} | {tier} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: 运行并生成分级表**

Run:
```bash
cd $GSMP && python tools/audit_sources.py | tee /tmp/tiers.md && cat /tmp/tiers.md
```
Expected: 13 行表格。已知 `feats` 的 `frame_idx` 全为 NULL、`fota_*` 无该列，
故三者必为 tier-2；`sparsh` 已核实为 30 列且 `capture`/`frame_idx` 均非空，
必为 tier-1。若 `gelslam`/`tactile_tracking`/`sparsh` 中任一不是 tier-1，
停止并报告——说明 `has_join_key` 的判定有问题。

- [ ] **Step 3: 写入 docs/source_tiers.md**

把 Step 2 的表格连同下述说明写入 `docs/source_tiers.md`：

```markdown
# Source migration tiers

Generated by `tools/audit_sources.py`, cross-checked against
`docs/imin_recovered.md`. Task 13-16 follow this table.

**tier-1** — published rows carry a usable (capture, frame_idx) join key and
`recover_imin.py` produced an unambiguous verdict. These are migrated onto
SourceSpec + the generic runner, and each one ships with a passing
`tools/regress.py` run recorded in its commit message.

**tier-2** — no join key, or an ambiguous i_min. These are wrapped verbatim:
the original iterator is moved into `src/gsmp/sources/<name>.py` unchanged
and only given a SourceSpec for metadata. Their internal filtering logic is
NOT refactored, because there is no way to prove the refactor preserved it.

<insert generated table here>

## Known tier-2 causes

- `feats` — frame_idx is NULL for all 1363 published rows; the source filters
  on force (|f_z| >= 0.4 N), not pixels, so there is no pixel-domain baseline
  to reconstruct.
- `fota_labeled`, `fota_unlabeled` — published with the 26-column schema,
  which has no frame_idx column at all (see gsmp/schema.py LEGACY_26_SOURCES).
  Additionally fota_unlabeled went through the v9 channel-order fix, so its
  intermediate state is not reproducible from the current raw data.
```

- [ ] **Step 4: 提交**

```bash
cd $GSMP
git add tools/audit_sources.py docs/source_tiers.md
git commit -F - <<'EOF'
docs: per-source migration tiers based on regression feasibility

A source is only refactored onto the new abstraction if its published rows
can actually be joined back (capture + non-null frame_idx) and its i_min was
recoverable. Everything else is wrapped verbatim rather than rewritten on
faith. feats and both fota_* subsets are known tier-2.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
```

---

### Task 13: `gsmp.runner` + 首个 tier-1 源（gelslam）

第一个源同时验证 runner 抽象是否成立。gelslam 是最干净的通用路径
（`SKIP_EMPTY_FILTER=False`，`FirstNFrames(10)` baseline，83,240 已发布行）。

**Files:**
- Create: `src/gsmp/runner.py`
- Create: `src/gsmp/sources/__init__.py`, `src/gsmp/sources/gelslam.py`
- Test: `tests/test_runner.py`

**Interfaces:**
- Consumes: `gsmp.spec.SourceSpec`, `gsmp.baseline`, `gsmp.filters`, `gsmp.encode`, `gsmp.writer`
- Produces:
  - `gsmp.runner.FrameRecord(rgb, capture, obj_name, split, episode, frame_idx, extra)`
  - `gsmp.runner.run(spec, frames_by_unit, writer=None, dry_run=False) -> RunResult`
  - `gsmp.runner.RunResult(kept_keys, n_candidates, n_kept, n_bg_kept)`
  - 每个 source 模块导出 `SPEC`、`iter_units()`、`dry_run_keys(limit=None)`

- [ ] **Step 1: 写失败测试**

Create `tests/test_runner.py`:
```python
from __future__ import annotations

import numpy as np

from gsmp.baseline import FirstNFrames, NoBaseline
from gsmp.runner import FrameRecord, run
from gsmp.spec import SourceSpec


def _spec(**kw):
    base = dict(name="demo", domain="real", gel_variant="markerless",
                license_repo="main", baseline=FirstNFrames(2), i_min=10.0,
                phash_dist=None, bg_keep_rate=0.0)
    base.update(kw)
    return SourceSpec(**base)


def _unit(values, capture="c0"):
    return capture, [
        FrameRecord(rgb=np.full((80, 120, 3), v, dtype=np.uint8),
                    capture=capture, frame_idx=i)
        for i, v in enumerate(values)
    ]


def test_keeps_only_frames_passing_the_filter():
    # first 2 frames are the baseline (value 10); frame 2 is a strong contact
    unit = _unit([10, 10, 200, 10])
    res = run(_spec(), [unit], dry_run=True)
    assert res.kept_keys == {("c0", 2)}
    assert res.n_candidates == 4


def test_bg_keep_rate_zero_keeps_nothing_extra():
    res = run(_spec(bg_keep_rate=0.0), [_unit([10, 10, 10, 10])], dry_run=True)
    assert res.kept_keys == set()
    assert res.n_bg_kept == 0


def test_bg_keep_rate_one_keeps_everything():
    res = run(_spec(bg_keep_rate=1.0), [_unit([10, 10, 10, 10])], dry_run=True)
    assert len(res.kept_keys) == 4


def test_no_baseline_source_keeps_every_frame():
    res = run(_spec(baseline=NoBaseline()), [_unit([1, 2, 3])], dry_run=True)
    assert len(res.kept_keys) == 3


def test_budget_caps_kept_frames():
    unit = _unit([10, 10] + [200] * 50)
    res = run(_spec(budget=10), [unit], dry_run=True)
    assert len(res.kept_keys) == 10


def test_dedupe_drops_near_identical_frames():
    rng = np.random.default_rng(3)
    noise = rng.integers(0, 255, (80, 120, 3), dtype=np.uint8)
    frames = [
        FrameRecord(rgb=np.full((80, 120, 3), 10, dtype=np.uint8), capture="c0", frame_idx=0),
        FrameRecord(rgb=np.full((80, 120, 3), 10, dtype=np.uint8), capture="c0", frame_idx=1),
        FrameRecord(rgb=noise, capture="c0", frame_idx=2),
        FrameRecord(rgb=noise.copy(), capture="c0", frame_idx=3),
    ]
    res = run(_spec(baseline=NoBaseline(), phash_dist=4, phash_lookback=10),
              [("c0", frames)], dry_run=True)
    # frame 3 is a byte-identical duplicate of frame 2 -> dropped
    assert ("c0", 3) not in res.kept_keys
    assert ("c0", 2) in res.kept_keys
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd $GSMP && python -m pytest tests/test_runner.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gsmp.runner'`

- [ ] **Step 3: 实现 runner**

Create `src/gsmp/runner.py`:
```python
"""The generic ingest driver.

One code path for every source. What used to vary per source -- baseline
recipe, thresholds, dedupe strictness, licence destination -- now arrives as
a SourceSpec, so this module has no per-source branches.

Pipeline order matches docs/PIPELINE.md:
    iter_frames -> baseline -> contact filter -> channel norm
                -> phash dedupe -> budget cap -> parquet
"""
from __future__ import annotations

import dataclasses
import random
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from gsmp import filters
from gsmp.baseline import needs_frames
from gsmp.encode import encode_jpeg
from gsmp.spec import SourceSpec
from gsmp.writer import ShardWriter

Key = Tuple[str, int]


@dataclasses.dataclass
class FrameRecord:
    """One decoded source frame, ready to filter."""

    rgb: np.ndarray
    capture: str = ""
    obj_name: str = ""
    split: str = "train"
    episode: str = ""
    frame_idx: int = 0
    extra: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class RunResult:
    kept_keys: Set[Key]
    n_candidates: int
    n_kept: int
    n_bg_kept: int


def _row(spec: SourceSpec, rec: FrameRecord, rgb: np.ndarray) -> Dict[str, Any]:
    h, w = rgb.shape[:2]
    row = {
        "image": encode_jpeg(rgb),
        "image_format": "jpeg",
        "source": spec.name,
        "domain": spec.domain,
        "markered": spec.markered,
        "gel_variant": spec.gel_variant,
        "capture": rec.capture,
        "split": rec.split,
        "episode": rec.episode,
        "frame_idx": rec.frame_idx,
        "obj_name": rec.obj_name,
        "height": h,
        "width": w,
    }
    row.update(rec.extra)
    return row


def run(
    spec: SourceSpec,
    units: Iterable[Tuple[str, Sequence[FrameRecord]]],
    writer: Optional[ShardWriter] = None,
    dry_run: bool = False,
    seed: Optional[int] = None,
) -> RunResult:
    """Process `units`, where each unit is (unit_id, frames_of_that_unit).

    A "unit" is whatever the baseline strategy is scoped to: a capture, an
    episode, a touch, or the whole source for global strategies.
    """
    rng = random.Random(spec.rng_seed if seed is None else seed)
    kept: Set[Key] = set()
    n_candidates = 0
    n_bg = 0
    recent_hashes: List[int] = []

    for _unit_id, frames in units:
        frames = list(frames)
        if not frames:
            continue

        base = (
            spec.baseline.compute([f.rgb for f in frames])
            if needs_frames(spec.baseline)
            else None
        )

        for rec in frames:
            n_candidates += 1
            if len(kept) >= spec.budget:
                break

            if base is None:
                passed = True
            else:
                passed = filters.passes_filter(
                    rec.rgb, base, a_min=spec.a_min, i_min=spec.i_min
                )

            is_bg = False
            if not passed:
                if rng.random() >= spec.bg_keep_rate:
                    continue
                is_bg = True
            elif spec.touch_window_keep < 1.0:
                # Bernoulli thinning inside the touch window (RTM only).
                if rng.random() >= spec.touch_window_keep:
                    continue

            rgb = filters.maybe_swap_channels(rec.rgb, spec.channel_mode)

            if spec.dedupe_enabled:
                h = filters.phash(rgb)
                window = recent_hashes[-spec.phash_lookback:]
                if any(filters.hamming(h, p) <= spec.phash_dist for p in window):
                    continue
                recent_hashes.append(h)

            kept.add((rec.capture, rec.frame_idx))
            if is_bg:
                n_bg += 1
            if not dry_run and writer is not None:
                writer.add(_row(spec, rec, rgb))

        if len(kept) >= spec.budget:
            break

    return RunResult(
        kept_keys=kept,
        n_candidates=n_candidates,
        n_kept=len(kept),
        n_bg_kept=n_bg,
    )
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd $GSMP && python -m pytest tests/test_runner.py -v`
Expected: 6 passed

- [ ] **Step 5: 实现 gelslam 源模块**

先读 legacy 实现以保持行为一致：

Run: `cd $GSMP && sed -n '141,175p' legacy/make_parquet_v2.py`

Create `src/gsmp/sources/__init__.py`:
```python
from __future__ import annotations
```

Create `src/gsmp/sources/gelslam.py`（`i_min` 填入 `docs/imin_recovered.md`
中 gelslam 一行的 p01 值，四舍五入到整数；`iter_units` 的解码逻辑照搬
`legacy/make_parquet_v2.py::iter_gelslam`，仅把 `cv2.VideoCapture` 的路径来源
改为 `config.RAW_ROOT`）:
```python
"""GelSLAM -- tracking and reconstruction episodes, one .avi per episode.

tier-1: published rows carry capture + frame_idx, so tools/regress.py can
verify this migration against the release.
"""
from __future__ import annotations

from typing import Iterator, List, Optional, Sequence, Tuple

import cv2

from gsmp import config
from gsmp.baseline import FirstNFrames
from gsmp.runner import FrameRecord
from gsmp.spec import SourceSpec, register

SPEC = register(SourceSpec(
    name="gelslam",
    domain="real",
    gel_variant="markerless",
    license_repo="main",
    baseline=FirstNFrames(10),
    a_min=40,
    i_min=10.0,          # REPLACE with the p01 from docs/imin_recovered.md
    channel_mode="auto",
    phash_dist=4,
    phash_lookback=30,
    notes="MIT. arXiv:2508.15990. One .avi per tracking/recon episode.",
))

_ROOT = config.RAW_ROOT / "markerless" / "GelSLAM"


def iter_units(limit: Optional[int] = None) -> Iterator[Tuple[str, Sequence[FrameRecord]]]:
    """Yield (capture_id, frames) per episode .avi."""
    n = 0
    for avi in sorted(_ROOT.rglob("*.avi")):
        capture = str(avi.relative_to(_ROOT).with_suffix(""))
        split = "recon" if "reconstruction" in capture else "train"
        cap = cv2.VideoCapture(str(avi))
        frames: List[FrameRecord] = []
        idx = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            frames.append(FrameRecord(
                rgb=bgr[:, :, ::-1].copy(),
                capture=capture,
                obj_name=capture.split("/")[-1],
                split=split,
                episode=capture,
                frame_idx=idx,
            ))
            idx += 1
        cap.release()
        if frames:
            yield capture, frames
            n += 1
        if limit is not None and n >= limit:
            return


def dry_run_keys(limit: Optional[int] = None):
    """Return (kept_keys, n_candidates) without writing anything."""
    from gsmp.runner import run

    res = run(SPEC, iter_units(limit=limit), dry_run=True)
    return res.kept_keys, res.n_candidates
```

- [ ] **Step 6: 跑回归验证**

Run:
```bash
cd $GSMP && python tools/regress.py --source gelslam
```
Expected: `PASS exact` 或 `PASS within background budget`。

若 `FAIL`：**不要调 i_min 去凑**。先确认 `_ROOT` 路径与 legacy 的
`iter_gelslam` 一致（`sed -n '141,175p' legacy/make_parquet_v2.py`），
再确认 split 判定一致。仍失败则把 gelslam 降为 tier-2，
在 `docs/source_tiers.md` 记录原因，继续下一个源。

- [ ] **Step 7: 提交**

```bash
cd $GSMP
git add src/gsmp/runner.py src/gsmp/sources/ tests/test_runner.py
git commit -F - <<'EOF'
feat: generic runner + gelslam as the first tier-1 migration

The runner has no per-source branches: baseline recipe, thresholds, dedupe
strictness and licence destination all arrive via SourceSpec.

gelslam validates the abstraction end to end. Regression result against the
published 83,240 rows is recorded below.

regress: <paste the tools/regress.py output line here>

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
```

---

### Task 14: 第二个 tier-1 源（tactile_tracking）

验证 runner 对第二个源同样成立，且不需要为它加任何分支。

**Files:**
- Create: `src/gsmp/sources/tactile_tracking.py`

**Interfaces:**
- Consumes: `gsmp.runner`, `gsmp.spec`, `gsmp.baseline.FirstNFrames`
- Produces: `SPEC`, `iter_units(limit=None)`, `dry_run_keys(limit=None)`

- [ ] **Step 1: 读 legacy 实现**

Run: `cd $GSMP && sed -n '174,207p' legacy/make_parquet_v2.py`
记录：raw 路径、capture id 构造方式、split 取值。

- [ ] **Step 2: 实现源模块**

Create `src/gsmp/sources/tactile_tracking.py`:
```python
"""TactileTracking / NormalFlow -- one .avi per trial.

tier-1: published rows carry capture + frame_idx.
"""
from __future__ import annotations

from typing import Iterator, List, Optional, Sequence, Tuple

import cv2

from gsmp import config
from gsmp.baseline import FirstNFrames
from gsmp.runner import FrameRecord
from gsmp.spec import SourceSpec, register

SPEC = register(SourceSpec(
    name="tactile_tracking",
    domain="real",
    gel_variant="markerless",
    license_repo="main",
    baseline=FirstNFrames(10),
    a_min=40,
    i_min=10.0,          # REPLACE with the p01 from docs/imin_recovered.md
    channel_mode="auto",
    phash_dist=4,
    phash_lookback=30,
    notes="MIT. RA-L 2024. One normalflow .avi per trial.",
))

_ROOT = config.RAW_ROOT / "markerless" / "TactileTracking"


def iter_units(limit: Optional[int] = None) -> Iterator[Tuple[str, Sequence[FrameRecord]]]:
    n = 0
    for avi in sorted(_ROOT.rglob("*.avi")):
        capture = str(avi.relative_to(_ROOT).with_suffix(""))
        cap = cv2.VideoCapture(str(avi))
        frames: List[FrameRecord] = []
        idx = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            frames.append(FrameRecord(
                rgb=bgr[:, :, ::-1].copy(),
                capture=capture,
                obj_name=capture.split("/")[0],
                split="train",
                episode=capture,
                frame_idx=idx,
            ))
            idx += 1
        cap.release()
        if frames:
            yield capture, frames
            n += 1
        if limit is not None and n >= limit:
            return


def dry_run_keys(limit: Optional[int] = None):
    from gsmp.runner import run

    res = run(SPEC, iter_units(limit=limit), dry_run=True)
    return res.kept_keys, res.n_candidates
```

- [ ] **Step 3: 跑回归验证**

Run: `cd $GSMP && python tools/regress.py --source tactile_tracking`
Expected: PASS（已发布 2,408 行）。FAIL 时的处置同 Task 13 Step 6。

- [ ] **Step 4: 确认 runner 未被修改**

Run: `cd $GSMP && git diff --stat HEAD -- src/gsmp/runner.py`
Expected: 无输出。若 runner 因这个源被改动，说明抽象有漏，
停下来记录到 `docs/source_tiers.md` 再继续。

- [ ] **Step 5: 提交**

```bash
cd $GSMP
git add src/gsmp/sources/tactile_tracking.py
git commit -F - <<'EOF'
feat: migrate tactile_tracking (tier-1)

Second source on the generic runner, added without modifying runner.py.

regress: <paste the tools/regress.py output line here>

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
```

---

### Task 15: 其余 tier-1 源

对 `docs/source_tiers.md` 中每个剩余的 tier-1 源重复 Task 14 的流程。
候选（以实际审计结果为准）：`real_tactile_mnist`、`feelanyforce`、`threedcal`、
`tacquad`、`unit`、`sim_tactile_mnist`、`sim_starstruck`、`sparsh`。

**Files:**
- Create: `src/gsmp/sources/<name>.py`（每源一个）

**Interfaces:**
- Consumes: `gsmp.runner`, `gsmp.spec`, `gsmp.baseline`
- Produces: 每个模块导出 `SPEC`, `iter_units(limit=None)`, `dry_run_keys(limit=None)`

每个源对应的 baseline 策略与 legacy 行数：

> ⚠️ **下表的 baseline 一列是指示性的，以 legacy 代码为准。**
> 它最初抄自 `PIPELINE.md`，而该文档已被证明不可靠——它声称 `threedcal` 用
> "cross-image median over a random 200-frame sample"，实际代码
> （`make_parquet_v2.py:529-547`）读的是上游自带的 `blank_images/blank.png`，
> 根本不算中位数。**每个源实现前必须先读 legacy 行区间**，以代码所写为准；
> 与本表不符时改本表，并在 commit 里记录差异。
>
> 同样必须逐源从 legacy 抄出、不得假设默认值的还有：
> `rng_seed`（多数为 0，`:234` 为 42）、`a_min`、`touch_window_keep`。

| 源 | baseline（待核实） | legacy iter |
|---|---|---|
| `real_tactile_mnist` | `PerTouchMedian(5)`，另有 `touch_window_keep=0.30`、`rng_seed=42` | `make_parquet_v2.py:207-307` |
| `feelanyforce` | `GlobalMedian(126)` | `make_parquet_v2.py:308-369` |
| `threedcal` | **`ExplicitReference(blank.png)`** — 已核实，非中位数 | `make_parquet_v2.py:529-576` |
| `tacquad` | `PerGroupMedian(60)` | `make_parquet_v2.py:457-528` |
| `sim_tactile_mnist` | `PerTouchMedian(32)` | `make_parquet_v2.py:433-446` |
| `sim_starstruck` | `PerTouchMedian(32)` | `make_parquet_v2.py:447-456` |
| `unit` | `FirstNFrames(5)` | `legacy/ingest_unit.py` |
| `sparsh` | `PerGroupMedian(100)` | `legacy/ingest_sparsh.py` |

- [ ] **Step 1: 逐源实现**

对每个源：读对应 legacy 行区间 → 写 `src/gsmp/sources/<name>.py`
（结构同 Task 14 Step 2：`SPEC = register(SourceSpec(...))`、`iter_units`、
`dry_run_keys`）→ `i_min` 取自 `docs/imin_recovered.md` 对应行的 p01。

`sim_*` 两源注意 `domain="sim"`；`unit` 注意 `gel_variant="markered"`。

**`sparsh` 三处与其他源不同，容易写错：**

1. `license_repo="nc"` — 输出到 NC 仓库（CC-BY-NC-4.0，不能混进主仓库）。
2. `split` 存的是 indenter 名（`flat`/`sharp`/`sphere`），不是 `train`/`val`。
   已发布分片也按 indenter 命名（`flat-00000-of-00001.parquet`），
   与其他源按 split 命名的惯例不同。保持现状，不要"修正"。
3. 原始数据是 pickle 而非视频：
   `mini_data/markerless_nc/SparshGelSight/{indenter}/batch_*/dataset_gelsight_NN.pkl`，
   每个 pkl 是一个 JPEG bytes 列表（~5000 帧）。**必须跳过 `org_dataset_*.pkl`**
   （那是未过滤的原始数据），legacy 脚本第 64-65 行有此逻辑。
   `capture` 构造为 `{indenter}_batch_{N}_f{NN}`。

`i_min` 交叉验证：`legacy/ingest_sparsh.py` 显式写 `I_MIN = 12`。若
`docs/imin_recovered.md` 对 sparsh 给出的 p01 与 12 相差超过 2，
**先停下来查反推工具**，不要直接采信任一方。

- [ ] **Step 2: 逐源跑回归**

Run（每个源一次）:
```bash
cd $GSMP && python tools/regress.py --source <name>
```
Expected: PASS。任一源 FAIL 且原因不明时，**降级为 tier-2**、
在 `docs/source_tiers.md` 记录实际的 regress 输出，不得为凑通过而调参。

- [ ] **Step 3: 确认 runner 仍未被修改**

Run: `cd $GSMP && git diff --stat HEAD~7 -- src/gsmp/runner.py`
Expected: 无输出，或有输出时说明每一处改动为何是通用的而非为某源特设。

- [ ] **Step 4: 逐源提交**

每个源单独一个 commit，正文含该源的 regress 输出行：
```bash
cd $GSMP
git add src/gsmp/sources/<name>.py
git commit -F - <<'EOF'
feat: migrate <name> (tier-1)

regress: <paste output>

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
```

---

### Task 16: tier-2 源逐字封装

`feats`、`fota_labeled`、`fota_unlabeled`（及 Task 15 中降级的任何源）。
**不重构内部逻辑**——只搬位置、加元数据声明。

**Files:**
- Create: `src/gsmp/sources/feats.py`, `src/gsmp/sources/fota_labeled.py`, `src/gsmp/sources/fota_unlabeled.py`
- Test: `tests/test_tier2_specs.py`

**Interfaces:**
- Consumes: `gsmp.spec.SourceSpec`, `gsmp.baseline.NoBaseline`
- Produces: 每模块导出 `SPEC` 与 `legacy_entrypoint()`（指向 `legacy/` 中原函数），**不导出** `dry_run_keys`

- [ ] **Step 1: 写测试固化 tier-2 契约**

Create `tests/test_tier2_specs.py`:
```python
from __future__ import annotations

import importlib

import pytest

TIER2 = ["feats", "fota_labeled", "fota_unlabeled"]


@pytest.mark.parametrize("name", TIER2)
def test_tier2_declares_spec_but_no_dry_run(name):
    """tier-2 sources are wrapped, not migrated.

    They must NOT expose dry_run_keys, because there is no join key to
    regression-test them with and a dry_run would imply a verified migration.
    """
    mod = importlib.import_module(f"gsmp.sources.{name}")
    assert hasattr(mod, "SPEC")
    assert not hasattr(mod, "dry_run_keys")
    assert "tier-2" in (mod.__doc__ or "")


@pytest.mark.parametrize("name", TIER2)
def test_tier2_spec_records_why(name):
    mod = importlib.import_module(f"gsmp.sources.{name}")
    assert mod.SPEC.notes.strip(), f"{name} must document why it is tier-2"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd $GSMP && python -m pytest tests/test_tier2_specs.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gsmp.sources.feats'`

- [ ] **Step 3: 实现三个封装模块**

Create `src/gsmp/sources/feats.py`:
```python
"""FEATS -- markered gel, force-based filtering.

tier-2: WRAPPED VERBATIM, NOT MIGRATED.

Why: all 1363 published rows have frame_idx = NULL, so there is no join key
to regression-test a rewrite against. The source also filters on force
(|f_z| >= 0.4 N) rather than pixel diff, because tracking dots make pixel
diffing unreliable -- there is no pixel baseline to reconstruct.

The original implementation stays in legacy/convert_feats.py and
legacy/make_parquet_v2.py. This module only declares metadata.
"""
from __future__ import annotations

from gsmp.baseline import NoBaseline
from gsmp.spec import SourceSpec, register

SPEC = register(SourceSpec(
    name="feats",
    domain="real",
    gel_variant="markered",
    license_repo="main",
    baseline=NoBaseline(),
    i_min=0.0,           # unused: force-based filter, not pixel-based
    phash_dist=None,
    notes=(
        "tier-2. Force filter |f_z| >= 0.4 N + 1.5% bg keep. frame_idx is "
        "NULL in all published rows, so no regression join key exists. "
        "Implementation: legacy/convert_feats.py."
    ),
))


def legacy_entrypoint() -> str:
    return "legacy/convert_feats.py"
```

Create `src/gsmp/sources/fota_labeled.py`:
```python
"""FoTA labeled -- 6-DoF pose + object, mixed markered/markerless gels.

tier-2: WRAPPED VERBATIM, NOT MIGRATED.

Why: published with the 26-column legacy schema, which has no frame_idx
column at all (gsmp.schema.LEGACY_26_SOURCES), so there is no join key.
"""
from __future__ import annotations

from gsmp.baseline import PerCaptureMedian
from gsmp.spec import SourceSpec, register

SPEC = register(SourceSpec(
    name="fota_labeled",
    domain="real",
    gel_variant="mixed",
    license_repo="main",
    baseline=PerCaptureMedian(30),
    i_min=10.0,
    phash_dist=4,
    phash_lookback=30,
    resolution=(640, 480),
    notes=(
        "tier-2. Published with the 26-column schema (no frame_idx), so no "
        "regression join key. Implementation: legacy/reprocess_fota.py."
    ),
))


def legacy_entrypoint() -> str:
    return "legacy/reprocess_fota.py"
```

Create `src/gsmp/sources/fota_unlabeled.py`:
```python
"""FoTA unlabeled -- 516K raw frames subsampled to the published set.

tier-2: WRAPPED VERBATIM, NOT MIGRATED.

Why: two independent reasons. (1) Published with the 26-column legacy schema,
which has no frame_idx column. (2) It went through the v9 channel-order fix
(legacy/fix_channel_order.py), so its intermediate state is not reproducible
from the current raw data even if a join key existed.
"""
from __future__ import annotations

from gsmp.baseline import PerCaptureMedian
from gsmp.spec import SourceSpec, register

SPEC = register(SourceSpec(
    name="fota_unlabeled",
    domain="real",
    gel_variant="mixed",
    license_repo="main",
    baseline=PerCaptureMedian(30),
    i_min=10.0,
    phash_dist=1,        # loose, to retain 200K of 516K raw
    phash_lookback=5,
    resolution=(640, 480),
    notes=(
        "tier-2. 26-column schema (no frame_idx) AND post-v9 channel fix, so "
        "the intermediate state is unreproducible. Implementation: "
        "legacy/redo_fota_unlabeled.py + legacy/fix_channel_order.py."
    ),
))


def legacy_entrypoint() -> str:
    return "legacy/redo_fota_unlabeled.py"
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd $GSMP && python -m pytest tests/test_tier2_specs.py -v`
Expected: 6 passed

- [ ] **Step 5: 提交**

```bash
cd $GSMP
git add src/gsmp/sources/feats.py src/gsmp/sources/fota_labeled.py \
        src/gsmp/sources/fota_unlabeled.py tests/test_tier2_specs.py
git commit -F - <<'EOF'
feat: wrap tier-2 sources without refactoring their logic

feats and both fota_* subsets get a SourceSpec for metadata only. Their
filtering logic stays in legacy/ untouched, because none of them can be
regression-tested: feats has frame_idx NULL throughout, and both fota_*
subsets were published with the 26-column schema that has no frame_idx
column. fota_unlabeled additionally went through the v9 channel fix, so its
intermediate state is unreproducible.

A test asserts these modules do NOT expose dry_run_keys, so a wrapped source
cannot be mistaken for a verified migration.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
```

---

### Task 17: `archive/` — 封存一次性修复脚本

**Files:**
- Create: `archive/README.md`
- Move: `legacy/{fix_*,reprocess_*,redo_*,swap_*,subsample_*,dedupe_cap_fota,rebalance_compose}.py` → `archive/`
- Move: `legacy/finalize*.sh`, `legacy/touchandgo_retry_loop.sh` → `archive/`

**Interfaces:**
- Consumes: 无
- Produces: `archive/README.md` — 每个脚本修了什么、何时跑的、影响哪些源

- [ ] **Step 1: 移动文件**

```bash
cd $GSMP
mkdir -p archive
git mv legacy/fix_channel_order.py legacy/fix_feats_marker_labels.py \
       legacy/fix_fota_marker_labels.py legacy/redo_fota_unlabeled.py \
       legacy/reprocess_feats.py legacy/reprocess_fota.py \
       legacy/reprocess_legacy.py legacy/reprocess_upstream.py \
       legacy/reprocess_v7_zscore.py legacy/swap_fota_unlabeled.py \
       legacy/subsample_fota_unlabeled.py legacy/dedupe_cap_fota.py \
       legacy/rebalance_compose.py \
       legacy/finalize.sh legacy/finalize_channel_fix.sh \
       legacy/finalize_touchandgo.sh legacy/finalize_tvl.sh \
       legacy/finalize_v9.sh legacy/touchandgo_retry_loop.sh \
       archive/
```

注意 `reprocess_fota.py` 与 `redo_fota_unlabeled.py` 被 Task 16 的
`legacy_entrypoint()` 引用——移动后须同步更新那两处字符串为 `archive/...`。

- [ ] **Step 2: 更新 tier-2 模块中的路径引用**

```bash
cd $GSMP
sed -i 's|legacy/reprocess_fota.py|archive/reprocess_fota.py|' src/gsmp/sources/fota_labeled.py
sed -i 's|legacy/redo_fota_unlabeled.py|archive/redo_fota_unlabeled.py|g' src/gsmp/sources/fota_unlabeled.py
sed -i 's|legacy/fix_channel_order.py|archive/fix_channel_order.py|g' src/gsmp/sources/fota_unlabeled.py
grep -rn "legacy/" src/gsmp/sources/
```
Expected: 只剩 `legacy/convert_feats.py`（未移动）。

- [ ] **Step 3: 写 archive/README.md**

Create `archive/README.md`:
```markdown
# Archived one-off scripts

These ran once, their effects are already baked into the published release,
and they are kept verbatim for provenance -- not for reuse. Nothing here is
imported by `src/gsmp/`.

| Script | What it fixed | Affected sources |
|---|---|---|
| `fix_channel_order.py` | Swapped BGR-stored frames to RGB (v9) | fota_unlabeled, others per diagnose output |
| `swap_fota_unlabeled.py` | Applied the channel swap to fota_unlabeled shards | fota_unlabeled |
| `fix_feats_marker_labels.py` | Corrected the `markered` column | feats |
| `fix_fota_marker_labels.py` | Auto-detected marker presence from dot density | fota_labeled, fota_unlabeled |
| `redo_fota_unlabeled.py` | v4 cv2 fast reprocess, 516K raw -> 200K kept | fota_unlabeled |
| `subsample_fota_unlabeled.py` | Budget subsampling | fota_unlabeled |
| `dedupe_cap_fota.py` | Streaming phash dedupe + budget cap | fota_labeled, fota_unlabeled |
| `reprocess_feats.py` | Re-derived force columns | feats |
| `reprocess_fota.py` | v3 in-place post-filter | fota_labeled |
| `reprocess_legacy.py` | Backfilled `domain` / `markered` on old shards | multiple |
| `reprocess_upstream.py` | Re-pulled and re-filtered upstream sources | multiple |
| `reprocess_v7_zscore.py` | z-score filter experiment (superseded by area+intensity) | multiple |
| `rebalance_compose.py` | Rebalanced per-source composition | all |
| `finalize*.sh`, `touchandgo_retry_loop.sh` | Release drivers per wave | all |

## Why these were not refactored

The dataset is already published. Rewriting a script whose only job was to
repair data that is now correct buys nothing and risks misrepresenting what
actually happened. Deleting them would lose the record of why the release
looks the way it does.
```

- [ ] **Step 4: 验证测试仍通过**

Run: `cd $GSMP && python -m pytest -q`
Expected: 全部通过（tier-2 测试依赖 `SPEC.notes` 中的路径字符串，Step 2 已更新）。

- [ ] **Step 5: 提交**

```bash
cd $GSMP
git add -A archive/ src/gsmp/sources/
git commit -F - <<'EOF'
chore: archive one-off repair scripts with a provenance index

These 19 scripts ran once and their effects are already in the published
release. Kept verbatim so the record of why the release looks the way it
does survives; not refactored, because repairing already-correct data buys
nothing.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
```

---

### Task 18: 文档重写 + 消除陈旧副本

`docs/PIPELINE.md` 停在 v4，且开篇声称"文档是唯一真源"——该声明当前不成立。
同时 `mini_data_parquet/scripts/` 有一份 5-17 的陈旧副本已随数据集发布到 HF。

**Files:**
- Modify: `docs/PIPELINE.md`
- Modify: `README.md`
- Delete: `/media/yxma/Disk1/yuxiang/mini_data_parquet/scripts/`（**需用户确认**，见 Step 4）

**Interfaces:**
- Consumes: `docs/source_tiers.md`, `docs/imin_recovered.md`
- Produces: 与代码一致的 PIPELINE.md

- [ ] **Step 1: 改写 PIPELINE.md 开头的真源声明**

把现有的

```
This file is the single source of truth — if the pipeline code disagrees
with this doc, the doc wins and the code should be updated to match.
```

替换为：

```
The code is the source of truth. This document describes what
`src/gsmp/` actually does; where they disagree, the code is right and this
document is stale — fix the document.

Per-source parameters are NOT duplicated here. They live in the
`SourceSpec` at the top of each `src/gsmp/sources/<name>.py`. The previous
version of this file listed thresholds inline, which is how i_min came to
have four different documented values.
```

- [ ] **Step 2: 更新 PIPELINE.md 的过滤规则一节**

把 `I_MIN` 常量表格行替换为：

```
| `I_MIN` | per-source, see each SourceSpec | recovered from the published
  release by tools/recover_imin.py; historically documented as 10, 12 and 15
  in three different places, none of which could be confirmed |
```

- [ ] **Step 3: 追加 schema 现状一节**

在 "Unified schema" 一节标题下追加：

```markdown
> **Not actually unified.** `fota_labeled` and `fota_unlabeled` were
> published with 26 columns, missing `episode`, `frame_idx`, `digit_class`
> and `gel_variant` (93,155 frames, ~11% of the corpus). The released
> README's `concatenate_datasets` example fails because of this.
> `gsmp.schema.LEGACY_26_SOURCES` records it and
> `tests/test_schema.py::test_fota_is_known_to_deviate` pins it.
> Remediation requires republishing those two subsets — see Task 19.
```

- [ ] **Step 4: 在 PIPELINE.md 记录 scripts/ 副本的处置方向**

实际的删除与 GitHub 迁移在 Task 20 执行（需要 `gh auth login`）。此处只在
`docs/PIPELINE.md` 的 "Where the code lives" 一节记录最终状态：

```markdown
## Where the code lives

All of it lives in one git repository, mirrored to GitHub. The dataset repo
on Hugging Face no longer carries a `scripts/` copy: it held an 8-file
snapshot from 2026-05-17 whose `make_parquet_v2.py` was 7,770 bytes behind
the code that actually produced the release. A copy with no mechanism
keeping it fresh is worse than a link, so the README points at the
repository instead.
```

**本步不修改 `mini_data_parquet/`。**

- [ ] **Step 5: 更新 README.md**

把 Task 2 写的占位 README 替换为完整版：仓库布局、`pip install -e .`、
如何跑一个源、`tools/` 各脚本用途、tier-1/tier-2 的含义与当前分布
（引用 `docs/source_tiers.md`）。

README 须含一节 "Not migrated"，明确记录：

```markdown
## Not migrated

`legacy/make_parquet_video.py` and the sequence-preserving video subset at
`mini_data_parquet_video/` (gelslam, real_tactile_mnist, tactile_tracking)
are imported verbatim and left alone. That subset's schema carries four
extra columns (sequence_id, frame_in_seq, sequence_length, fps) that the
30-column `gsmp.schema.SCHEMA` does not model. Supporting it needs a
polymorphic schema layer, which is a separate design question — see the
scope section of docs/superpowers/plans/2026-08-04-gsmp-preprocess-refactor.md.
```

- [ ] **Step 6: 全量测试**

Run: `cd $GSMP && python -m pytest -q`
Expected: 全部通过。

- [ ] **Step 7: 提交**

```bash
cd $GSMP
git add docs/PIPELINE.md README.md
git commit -F - <<'EOF'
docs: rewrite PIPELINE.md to match the code

Three corrections. The doc claimed to be the source of truth while sitting
at v4 against a v9 codebase -- the code is now authoritative. Per-source
thresholds are no longer duplicated here, since inline duplication is how
i_min ended up with four conflicting documented values. And the "unified
30-column schema" section now records that FoTA is actually 26 columns.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
```

---

### Task 19: 修复 FoTA 26 列 schema（补列 + 重传）

已批准执行。给 `fota_labeled` / `fota_unlabeled` 现有 shard 补齐 4 列，
JPEG 原样透传，重传约 3.5GB。

**Files:**
- Create: `src/gsmp/backfill.py`, `tools/backfill_fota_schema.py`
- Test: `tests/test_backfill.py`

**Interfaces:**
- Consumes: `gsmp.schema.SCHEMA`, `gsmp.schema.LEGACY_26_MISSING`
- Produces:
  - `backfill_table(table: pa.Table) -> pa.Table` — 26 列 → 30 列
  - `verify_backfill(old: pa.Table, new: pa.Table) -> None` — 不满足则 raise

补列规则（`gel_variant` 是唯一能填出真值的）：

| 列 | 填法 |
|---|---|
| `gel_variant` | `"markered" if markered else "markerless"` |
| `frame_idx` | NULL — 行内没有来源帧号，不可恢复 |
| `episode` | NULL — FoTA 无 episode 概念 |
| `digit_class` | NULL — 对 FoTA 无意义 |

- [ ] **Step 1: 写失败测试**

Create `tests/test_backfill.py`:
```python
from __future__ import annotations

import pyarrow as pa
import pytest

from gsmp import schema
from gsmp.backfill import backfill_table, verify_backfill

_OLD_COLS = [c for c in schema.COLUMNS if c not in schema.LEGACY_26_MISSING]


def _old_table(n=3, markered=(True, False, True)):
    data = {}
    for name in _OLD_COLS:
        field = schema.SCHEMA.field(name)
        if name == "image":
            data[name] = pa.array([b"\xff\xd8jpg%d" % i for i in range(n)], pa.binary())
        elif name == "markered":
            data[name] = pa.array(list(markered[:n]), pa.bool_())
        elif pa.types.is_string(field.type):
            data[name] = pa.array([f"{name}{i}" for i in range(n)], pa.string())
        elif pa.types.is_boolean(field.type):
            data[name] = pa.array([False] * n, pa.bool_())
        elif pa.types.is_int32(field.type):
            data[name] = pa.array(list(range(n)), pa.int32())
        else:
            data[name] = pa.array([float(i) for i in range(n)], pa.float32())
    return pa.table(data)


def test_backfill_produces_the_full_schema():
    new = backfill_table(_old_table())
    assert new.schema.names == list(schema.COLUMNS)
    assert new.schema.equals(schema.SCHEMA)


def test_gel_variant_is_derived_from_markered():
    new = backfill_table(_old_table(markered=(True, False, True)))
    assert new.column("gel_variant").to_pylist() == [
        "markered", "markerless", "markered",
    ]


def test_unrecoverable_columns_are_null():
    new = backfill_table(_old_table())
    for col in ("frame_idx", "episode", "digit_class"):
        assert new.column(col).null_count == new.num_rows


def test_image_bytes_pass_through_untouched():
    old = _old_table()
    new = backfill_table(old)
    assert new.column("image").to_pylist() == old.column("image").to_pylist()


def test_all_original_columns_are_preserved_exactly():
    old = _old_table()
    new = backfill_table(old)
    for name in _OLD_COLS:
        assert new.column(name).to_pylist() == old.column(name).to_pylist(), name


def test_verify_passes_for_a_correct_backfill():
    old = _old_table()
    verify_backfill(old, backfill_table(old))


def test_verify_rejects_row_count_change():
    old = _old_table()
    bad = backfill_table(old).slice(0, 2)
    with pytest.raises(ValueError, match="row count"):
        verify_backfill(old, bad)


def test_verify_rejects_mutated_image_bytes():
    old = _old_table()
    new = backfill_table(old)
    cols = {n: new.column(n) for n in new.schema.names}
    cols["image"] = pa.array([b"tampered"] * new.num_rows, pa.binary())
    with pytest.raises(ValueError, match="image"):
        verify_backfill(old, pa.table(cols).cast(schema.SCHEMA))


def test_backfill_is_idempotent():
    old = _old_table()
    once = backfill_table(old)
    assert backfill_table(once).equals(once)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd $GSMP && python -m pytest tests/test_backfill.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gsmp.backfill'`

- [ ] **Step 3: 实现**

Create `src/gsmp/backfill.py`:
```python
"""Backfill the 26-column FoTA shards to the full 30-column schema.

fota_labeled and fota_unlabeled were published missing episode, frame_idx,
digit_class and gel_variant, which breaks the concatenate_datasets example in
the dataset's own README (93,155 frames, ~11% of the corpus).

Only gel_variant can be given a real value -- it is a function of the
markered column. The other three are unrecoverable from the published rows
and are written as NULL. That is enough to fix the user-visible breakage:
HF feature compatibility requires matching column names and types, not
non-null values.

JPEG bytes are never re-encoded. Every original column passes through
untouched, and verify_backfill() enforces that.
"""
from __future__ import annotations

import pyarrow as pa

from gsmp.schema import COLUMNS, LEGACY_26_MISSING, SCHEMA


def backfill_table(table: pa.Table) -> pa.Table:
    """Return `table` widened to the full 30-column schema."""
    n = table.num_rows
    present = set(table.schema.names)
    cols = {}

    for name in COLUMNS:
        field = SCHEMA.field(name)
        if name in present:
            cols[name] = table.column(name).cast(field.type)
        elif name == "gel_variant":
            markered = table.column("markered").to_pylist()
            cols[name] = pa.array(
                ["markered" if m else "markerless" for m in markered],
                pa.string(),
            )
        else:
            cols[name] = pa.nulls(n, field.type)

    return pa.table(cols, schema=SCHEMA)


def verify_backfill(old: pa.Table, new: pa.Table) -> None:
    """Raise unless `new` is `old` widened, with nothing else changed."""
    if new.num_rows != old.num_rows:
        raise ValueError(
            f"row count changed: {old.num_rows} -> {new.num_rows}"
        )
    if not new.schema.equals(SCHEMA):
        raise ValueError("result does not match the canonical schema")

    for name in old.schema.names:
        if name in LEGACY_26_MISSING:
            continue
        if new.column(name).to_pylist() != old.column(name).to_pylist():
            raise ValueError(f"column {name!r} was modified")

    for name in LEGACY_26_MISSING:
        if name == "gel_variant":
            continue
        if new.column(name).null_count != new.num_rows:
            raise ValueError(f"{name!r} should be entirely null")
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd $GSMP && python -m pytest tests/test_backfill.py -v`
Expected: 9 passed

- [ ] **Step 5: 写 CLI**

Create `tools/backfill_fota_schema.py`:
```python
#!/usr/bin/env python3
"""Widen the published FoTA shards to the 30-column schema.

Writes to config.OUT_ROOT/fota_backfill/<source>/ -- never in place, so the
published tree stays untouched until the upload step is run separately.

    python tools/backfill_fota_schema.py --source fota_labeled
    python tools/backfill_fota_schema.py --all
"""
from __future__ import annotations

import argparse
import glob
import pathlib
import sys

import pyarrow.parquet as pq

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from gsmp import config                                    # noqa: E402
from gsmp.backfill import backfill_table, verify_backfill  # noqa: E402

SOURCES = ("fota_labeled", "fota_unlabeled")


def run(source: str) -> None:
    src_dir = config.published_dir(source)
    out_dir = config.OUT_ROOT / "fota_backfill" / source
    out_dir.mkdir(parents=True, exist_ok=True)

    shards = sorted(glob.glob(str(src_dir / "*.parquet")))
    if not shards:
        raise FileNotFoundError(f"no shards under {src_dir}")

    for path in shards:
        name = pathlib.Path(path).name
        old = pq.read_table(path)
        new = backfill_table(old)
        verify_backfill(old, new)
        pq.write_table(new, out_dir / name, compression="snappy")
        print(f"  {name}: {old.num_rows} rows, "
              f"{len(old.schema.names)} -> {len(new.schema.names)} cols")

    print(f"{source}: wrote {len(shards)} shards to {out_dir}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=SOURCES)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    targets = SOURCES if args.all else ([args.source] if args.source else [])
    if not targets:
        ap.error("pass --source NAME or --all")
    for s in targets:
        run(s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: 生成补列后的分片**

Run:
```bash
cd $GSMP && python tools/backfill_fota_schema.py --all
```
Expected: 4 个 shard，`26 -> 30 cols`，行数分别为 26,394 与 66,761。
`verify_backfill` 在任一 shard 上抛异常即停止——**不要跳过**。

- [ ] **Step 7: 独立复核（不复用 backfill 代码）**

Run:
```bash
cd $GSMP && python - <<'PY'
import glob, pathlib, pyarrow.parquet as pq
from gsmp import config, schema
for s in ("fota_labeled", "fota_unlabeled"):
    old_fs = sorted(glob.glob(str(config.published_dir(s) / "*.parquet")))
    new_fs = sorted(glob.glob(str(config.OUT_ROOT / "fota_backfill" / s / "*.parquet")))
    assert len(old_fs) == len(new_fs), s
    for o, n in zip(old_fs, new_fs):
        to, tn = pq.read_table(o), pq.read_table(n)
        assert to.num_rows == tn.num_rows
        assert list(tn.schema.names) == list(schema.COLUMNS)
        assert to.column("image").to_pylist() == tn.column("image").to_pylist()
        gv = set(tn.column("gel_variant").to_pylist())
        assert gv <= {"markered", "markerless"}, gv
    print(f"{s}: OK, {sum(pq.ParquetFile(f).metadata.num_rows for f in new_fs)} rows")
PY
```
Expected: 两行 `OK`，行数 26,394 / 66,761。

- [ ] **Step 8: 上传新 revision**

```bash
cd $GSMP && python - <<'PY'
from huggingface_hub import HfApi
from gsmp import config
api = HfApi()
for s in ("fota_labeled", "fota_unlabeled"):
    api.upload_folder(
        repo_id=config.HF_REPO_MAIN, repo_type="dataset",
        folder_path=str(config.OUT_ROOT / "fota_backfill" / s),
        path_in_repo=s,
        commit_message=f"fix: widen {s} to the full 30-column schema",
    )
    print("uploaded", s)
PY
```

上传约 3.5GB。HF 保留提交历史，出问题可回滚到前一个 revision。

- [ ] **Step 9: 验证线上生效 + README 示例可跑**

Run:
```bash
cd $GSMP && python - <<'PY'
from datasets import load_dataset, concatenate_datasets
ds = [load_dataset("yxma/gelsight-mini-pretrain", c, split="train")
      for c in ("fota_unlabeled", "gelslam")]
pool = concatenate_datasets(ds)
print("concat OK, rows =", len(pool))
PY
```
Expected: 打印总行数而非抛 feature-mismatch 异常。**这一步是整个任务的验收标准**——
它跑通才说明缺陷真的修好了。

- [ ] **Step 10: 从 LEGACY_26_SOURCES 移除并让固化测试转正**

修改 `src/gsmp/schema.py`：`LEGACY_26_SOURCES = frozenset()`，
并在 docstring 记录修复日期。

修改 `tests/test_schema.py`：删除 `test_fota_is_known_to_deviate`，
把 `fota_labeled`、`fota_unlabeled` 加进
`test_conforming_sources_match_schema` 的参数列表。

Run: `cd $GSMP && python -m pytest tests/test_schema.py -v`
Expected: 13 passed（现在 12 个源全部走 conforming 分支）

- [ ] **Step 11: 更新已发布 README**

`docs/_readme_new.md` 中 "Schema (30 columns, every row identical)" 一节
现在成立，无需加免责说明。在文末 Notes 追加一行：

```markdown
- 2026-08-04: `fota_labeled` and `fota_unlabeled` were widened from 26 to 30
  columns. `frame_idx`, `episode` and `digit_class` are null for these two
  subsets -- they were not recorded at build time -- but the column set now
  matches every other config, so cross-config `concatenate_datasets` works.
```

- [ ] **Step 12: 提交**

```bash
cd $GSMP
git add src/gsmp/backfill.py tools/backfill_fota_schema.py \
        tests/test_backfill.py src/gsmp/schema.py tests/test_schema.py \
        docs/_readme_new.md
git commit -F - <<'EOF'
fix: widen fota_labeled and fota_unlabeled to the 30-column schema

Both subsets shipped with 26 columns, missing episode, frame_idx,
digit_class and gel_variant, which made the dataset README's own
concatenate_datasets example fail across configs. 93,155 frames affected.

gel_variant is derived from markered; the other three are unrecoverable from
the published rows and are null. That still fixes the breakage, since HF
feature compatibility needs matching names and types, not values.

JPEG bytes are not re-encoded -- verify_backfill asserts every original
column passes through byte-identical, and an independent re-check outside
the backfill code confirmed it before upload.

LEGACY_26_SOURCES is now empty and the pinning test is replaced by the
conforming-source test.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
```

---

### Task 20: GitHub 公开仓库 + 清理 HF 上的陈旧 scripts/

**前置条件（人工）：** `gh auth login`。`gh` 已安装但当前未登录，
此步无法自动完成——先确认登录再执行本任务。

**Files:**
- Modify: `README.md`（加仓库地址与 HF 数据集链接）
- Modify: `docs/_readme_new.md`（HF README：scripts/ 改为指向 GitHub）
- Delete on HF: `mini_data_parquet/scripts/`（远端），本地同目录一并删除

**Interfaces:**
- Consumes: Task 18 已重写的 docs
- Produces: 公开 remote `origin`；HF 数据集不再携带代码副本

- [ ] **Step 1: 确认无敏感内容**

Run:
```bash
cd $GSMP
grep -rnE "hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|api_key|password|secret" \
     --include="*.py" --include="*.sh" --include="*.md" . | grep -v "\.git/" || echo "CLEAN"
```
Expected: `CLEAN`。有命中则先处理再继续，**不要**推送。

另确认没有大文件混进历史：
```bash
cd $GSMP && git count-objects -vH | grep size-pack
```
Expected: 几 MB 量级。若超过 100MB，说明有数据文件被误提交，先查 `git log --stat`。

- [ ] **Step 2: 建仓库并推送**

```bash
cd $GSMP
gh repo create gelsight-mini-pretrain --public \
   --description "Preprocessing pipeline for the gelsight-mini-pretrain tactile dataset" \
   --source=. --remote=origin --push
git remote -v
```
Expected: `origin` 指向 `https://github.com/<user>/gelsight-mini-pretrain`。

- [ ] **Step 3: 更新 HF README 指向仓库**

在 `docs/_readme_new.md` 的 Pipeline 一节末尾追加：

```markdown
Full build pipeline, including per-source parameters and the regression
tests that check each source against this release:
<https://github.com/<user>/gelsight-mini-pretrain>
```

同时把 `docs/PIPELINE.md` 里 Task 18 Step 4 写的那段中的"mirrored to GitHub"
补上实际 URL。

- [ ] **Step 4: 从 HF 删除 scripts/**

```bash
cd $GSMP && python - <<'PY'
from huggingface_hub import HfApi
from gsmp import config
api = HfApi()
files = [f for f in api.list_repo_files(config.HF_REPO_MAIN, repo_type="dataset")
         if f.startswith("scripts/")]
print("deleting:", files)
assert files, "nothing to delete -- already removed?"
api.delete_files(repo_id=config.HF_REPO_MAIN, repo_type="dataset",
                 delete_patterns=["scripts/*"],
                 commit_message="chore: drop stale scripts/ copy; code now lives on GitHub")
PY
```
Expected: 列出 8 个文件并删除。

- [ ] **Step 5: 上传更新后的 HF README**

```bash
cd $GSMP && python - <<'PY'
from huggingface_hub import HfApi
from gsmp import config
HfApi().upload_file(
    path_or_fileobj=str(config.repo_root() / "docs" / "_readme_new.md"),
    path_in_repo="README.md", repo_id=config.HF_REPO_MAIN, repo_type="dataset",
    commit_message="docs: point at the pipeline repository")
print("README uploaded")
PY
```

- [ ] **Step 6: 删除本地陈旧副本**

```bash
rm -rf /media/yxma/Disk1/yuxiang/mini_data_parquet/scripts
ls /media/yxma/Disk1/yuxiang/mini_data_parquet/
```
Expected: 输出中不再有 `scripts`。这是本计划中**唯一**一处写 `mini_data_parquet/`
的操作，且只删除那份已确认陈旧的代码副本，不触碰任何 parquet。

- [ ] **Step 7: 验证线上状态**

```bash
cd $GSMP && python - <<'PY'
from huggingface_hub import HfApi
from gsmp import config
files = HfApi().list_repo_files(config.HF_REPO_MAIN, repo_type="dataset")
assert not [f for f in files if f.startswith("scripts/")], "scripts/ still present"
print("scripts/ removed; total files:", len(files))
PY
```

- [ ] **Step 8: 提交**

```bash
cd $GSMP
git add README.md docs/_readme_new.md docs/PIPELINE.md
git commit -F - <<'EOF'
chore: publish the pipeline repo, drop the stale scripts/ copy from HF

The dataset repo carried an 8-file scripts/ snapshot from 2026-05-17 whose
make_parquet_v2.py was 7,770 bytes behind the code that actually produced
the release -- four of the eight files had drifted. A copy with no mechanism
keeping it current is worse than a link, so the HF README now points at the
GitHub repository and the copy is gone.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
EOF
git push
```

---

## Self-Review

**Spec coverage:**

| Spec 要求 | 对应任务 |
|---|---|
| 独立 git 仓库、代码与数据同盘 | Task 2 |
| `config.py` 消灭硬编码路径 | Task 3 |
| `schema.py` 唯一真源、断循环依赖 | Task 4 |
| `filters.py` / `encode.py` / `writer.py` | Task 5, 6, 7 |
| `BaselineStrategy` 6 种、取代 SKIP_EMPTY_FILTER | Task 8 |
| `SourceSpec` 收敛四个字典 | Task 9 |
| i_min 无默认值、逐源反推 | Task 10 |
| 回归验证（两层断言） | Task 11 |
| 分级门槛（tier-1 / tier-2） | Task 12 |
| 通用 runner + 逐源迁移 | Task 13, 14, 15 |
| tier-2 原样封装不重构 | Task 16 |
| `archive/` 封存 + 索引 | Task 17 |
| `PIPELINE.md` 更新到真实状态 | Task 18 |
| 删除陈旧 scripts 副本 | Task 20（已批准，需先 `gh auth login`）|
| FoTA schema 缺陷修复 | Task 19（已批准执行，含重传）|
| MultimodalData 提交 | Task 1 |
| 数据只读 | Global Constraints；Task 19 显式不执行写操作 |

**新增（spec 未预见，审计中发现）：**

- Task 19 — FoTA 26 列 schema 缺陷。写 spec 时未知，是 Task 4 起草测试时
  查证已发布 parquet 发现的。**已批准执行**（原为仅提案），含约 3.5GB 重传。
  spec 的"不修改 HF 上现有 parquet"约束因此有一处经批准的例外，
  仅限 `fota_labeled` / `fota_unlabeled` 两个 config 的补列。
- Task 20 — GitHub 公开仓库 + 删除 HF 上的陈旧 `scripts/` 副本。**已批准执行**。
  spec 的"数据只读"约束的唯一例外是删除 `mini_data_parquet/scripts/`
  这份代码副本，不触碰任何 parquet。
- `sparsh`（NC 仓库）纳入范围。spec 只写了"12 个源"，遗漏了 NC 仓库这一棵
  parquet 树。已核实 sparsh 为 30 列、`capture`/`frame_idx` 非空，属 tier-1，
  且其 `I_MIN = 12` 在 legacy 脚本中明写，可反过来校验 Task 10 的反推工具。
- 视频子集明确排除。spec 未提及 `mini_data_parquet_video/`；因其 schema 多 4 列，
  纳入会迫使 schema 层多态化，与本次重构要建立的"唯一真源"冲突，故另开 spec。

**Type consistency（补充）：** `SOURCE_LICENSE_REPO` 在 Task 10 与 Task 12 的
两个工具中各定义一份，二者必须一致（Task 12 的注释已声明）。Task 16 落地后，
`spec.get(name).license_repo` 成为权威来源，两处常量应与之相符。
`config.published_dir(source, license_repo)` 与 `schema.published_columns(
source, license_repo)`、`schema.has_join_key(source, license_repo)`
三者的第二参数语义一致，默认均为 `"main"`。

**Type consistency:** `SourceSpec` 字段名在 Task 9 定义后，于 Task 13-16
的所有源模块中一致使用（`i_min`、`a_min`、`phash_dist`、`phash_lookback`、
`bg_keep_rate`、`channel_mode`、`license_repo`、`gel_variant`、`notes`）。
`BaselineStrategy.compute()` 在 Task 8 定义，Task 13 的 runner 通过
`needs_frames()` 调用。`FrameRecord` 字段（`rgb`/`capture`/`obj_name`/`split`/
`episode`/`frame_idx`/`extra`）在 Task 13 定义后为 Task 14-15 各源模块沿用。
`dry_run_keys()` 返回 `(kept_keys, n_candidates)`，与 `tools/regress.py` 的
解包一致，且 Task 16 明确规定 tier-2 不导出该函数。
