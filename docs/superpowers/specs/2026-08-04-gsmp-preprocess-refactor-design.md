# gelsight-mini-pretrain 预处理代码重构 — 设计

日期：2026-08-04
状态：已批准，待实施计划

## 背景

`gelsight-mini-pretrain`（HF: `yxma/gelsight-mini-pretrain`，~853K 帧，12 个源）
的全部预处理代码目前以 **40 个未跟踪脚本**的形式散落在
`/home/yxma/MultimodalData/` 仓库根目录，与机器人采集端代码（`twm/`、
`camera_stream/`、`probing_panda/`）混在一起。

### 现状问题

1. **无版本控制。** 85 个未跟踪条目，其中 ~40 个是本项目的全部实现。
2. **代码与数据分离。** 代码在 NVMe（`/` 已 100% 满，仅剩 3.8G），
   数据在 `/media/yxma/Disk1/yuxiang/mini_data*`（805G 原始 + 8G parquet）。
3. **单一真源缺失导致漂移。** `mini_data_parquet/scripts/` 存在一份 5-17 的
   陈旧副本（`make_parquet_v2.py` 32,592 字节 vs 仓库内 40,362 字节），
   且该副本已随数据集发布到 HF。
4. **抽象断裂。** `pipeline.py`（5-20）与 `make_parquet_v2.py`（5-18）是
   同一套抽象的两代，前者 `sys.path.insert` 反向 import 后者的 `SCHEMA`，
   构成循环依赖。
5. **文档与实现脱节。** `PIPELINE.md` 自称 v4，实际已有 `finalize_v9.sh`、
   `fix_channel_order.py` 等后续变更。该文档开篇声明"文档是唯一真源"，
   此声明当前不成立。
6. **关键参数四处取值不一。** `i_min` 在两代代码与两份文档中分别为
   10 / 12 / 15 / 12-10，且发布给用户的 README 值与主实现不符。详见下文。

### 核心发现

`SKIP_EMPTY_FILTER` 中 9 个源有 7 个为 `True`。PIPELINE.md 宣称的
"统一 area+intensity 过滤"实际只对 `gelslam`、`tactile_tracking` 两个源
由通用驱动器执行；其余 7 个各自在 iterator 内部用不同 baseline 策略完成过滤。

这个布尔开关掩盖的真实结构是：**12 个源存在 6 种不同的 baseline 求法**。
`pipeline.py` 假设驱动器负责过滤，该假设对 7/9 的源不成立——这解释了
为何它只被 2 个最新源（touchandgo、tvl）采用：老源套不进去。

## 目标

- 预处理代码进入版本控制，与采集端代码解耦。
- 每个源的完整行为在单个模块内可一眼读完，消灭散落各处的参数字典。
- 新增源 = 新增一个文件。
- 保留历史修复脚本的可追溯性，但不为其投入重写成本。

## 非目标

- 不重跑任何已发布数据的生成流程。
- 不修改 HF 上现有 parquet。
- 不重构 `twm/scripts/`（React 数据集构建流程）——单独一个 spec。
- 805G 原始数据与 8G parquet 全程只读。

## 方案

### 决策记录

| 决策 | 选择 | 理由 |
|---|---|---|
| 仓库边界 | 独立 git 仓库 `~/gelsight-mini-pretrain` | 与 MultimodalData 解耦，只留采集端 |
| 重构深度 | 分层：活代码重构 + 历史归档 | 一次性修复已生效，重写无收益 |
| 范围 | 仅 mini-pretrain | twm/scripts/ 另开 spec |
| 源注册方式 | 声明式 `SourceSpec` | 真正的债是参数散落，非文件散落 |

### 目标结构

```
/home/yxma/gelsight-mini-pretrain/     # 新 git 仓库
├── src/gsmp/
│   ├── config.py          # 所有路径 ← 环境变量 + 默认值
│   ├── schema.py          # 统一 30 列 SCHEMA（唯一真源）
│   ├── filters.py         # grey_center / passes_filter / channel_check / phash
│   ├── encode.py          # JPEG q92
│   ├── writer.py          # ShardWriter：分块 binary、2GB 规避、pickle 恢复点
│   ├── baseline.py        # BaselineStrategy 的 6 种实现
│   ├── spec.py            # SourceSpec dataclass
│   ├── runner.py          # 通用驱动：probe → filter → dedupe → budget → write
│   └── sources/           # 12 个源，每源一个模块（SPEC + iter_frames）
├── tools/                 # 报表 / 样例网格 / 发布
├── probes/                # 诊断脚本
├── archive/               # 一次性修复脚本原样封存 + README 索引
├── docs/PIPELINE.md       # 更新到当前真实状态
└── README.md / SOURCES.md
```

`archive/` 的取舍：`fix_channel_order.py`、`redo_fota_unlabeled.py`、
`reprocess_*.py` 等 11 个脚本的修复已烘进已发布数据。重写无收益，
删除则失去"数据为何长这样"的追溯链。原样封存 + 索引说明是最低成本的正确做法。

### 数据流

```
iter_frames()  →  BaselineStrategy  →  contact filter  →  channel norm
                                                              ↓
                   parquet shards  ←  budget cap  ←  phash dedupe
```

### SourceSpec

```python
@dataclass(frozen=True)
class SourceSpec:
    name: str
    domain: Literal["real", "sim"]
    gel_variant: Literal["markered", "markerless", "mixed"]
    license_repo: Literal["main", "nc"]      # 取代 NC_SOURCES 集合
    baseline: BaselineStrategy               # 取代 SKIP_EMPTY_FILTER
    a_min: int = 40
    i_min: float                             # 无默认值，见下 — 取代 VALIDITY_THRESH
    channel_mode: Literal["auto","rgb","bgr","mixed"] = "auto"
    phash_dist: int | None = 4               # None = 不去重
    phash_lookback: int = 30
    budget: int = 200_000
    resolution: tuple[int,int] | None = None # None = 保持原生
```

四个字典（`SKIP_EMPTY_FILTER`、`VALIDITY_THRESH`、`NC_SOURCES`、
`SOURCE_ITERS`）因此全部消失。

#### i_min 无默认值 — 一个必须先解决的矛盾

同一个参数在四处取值不同：

| 出处 | 值 |
|---|---|
| `make_parquet_v2.py:51` | `I_MIN_DEFAULT = 10` |
| `pipeline.py:188` | `i_min: float = 12.0` |
| `PIPELINE.md:90` | 15（FoTA 为 10） |
| `_readme_new.md:145`（已发布的 HF README） | 12 real / 10 sim |

已发布数据实际使用的值无法从代码或文档判定，且发布给用户的 README
声称的值（12/10）与主实现（10）和主文档（15）均不一致。

因此 `i_min` **不设默认值**，每个源必须显式声明。

#### 修正（2026-08-05，实施 Task 10 时发现）

上面这段的前提**只对全局默认值成立**。实际上每个 legacy iterator
都硬编码了自己的 `I_MIN`：

| 源 | I_MIN | 出处 |
|---|---|---|
| gelslam, tactile_tracking | 10 | `VALIDITY_THRESH` |
| real_tactile_mnist | 15 | `make_parquet_v2.py:231` |
| sim_tactile_mnist, sim_starstruck | 15 | `make_parquet_v2.py:379` |
| feelanyforce, threedcal, tacquad_mini, faf | 10 | `:319 / :538 / :471 / :589` |
| sparsh, unit, tacquad_full | 12 | `ingest_*.py` |

**权威来源是生成数据的 legacy 代码**，不是从已发布 parquet 反推。

反推法本身有一个不可修复的局限：已发布 parquet 只含**被保留的（即接触）**帧，
而 baseline 按定义是**无接触**参考——无接触参考无法从只含接触帧的数据集重建。
`real_tactile_mnist` 与 `feats` 更极端：5127 行对应 5127 个 capture 组，
per-capture 中位数就是该帧自身，差分恒为 0。

`tools/recover_imin.py` 因此降级为**交叉验证工具**，其结论记于
`docs/imin_recovered.md`；权威值记于 `docs/imin_from_code.md`。
在非退化的源上两者互相印证：gelslam 反推 11.05 vs 代码 10，
sparsh p05=12.50 vs 代码 12。

这个错误是被 sparsh 锚点抓到的——`ingest_sparsh.py` 明写 `I_MIN = 12`，
是唯一有独立可信值的源。它的作用不是确认方法可靠，而是证伪。
没有这个锚点，9 个源会拿到"i_min = 0.0"的退化值而看起来一切正常。

### BaselineStrategy

统一接口 `compute(frames) -> np.ndarray | None`，覆盖实际存在的 6 种：

| 策略 | 使用的源 |
|---|---|
| `PerCaptureMedian(n=30)` | fota_labeled, fota_unlabeled |
| `FirstNFrames(n=10)` | gelslam, tactile_tracking, unit(n=5) |
| `GlobalMedian(n)` | threedcal(200), feelanyforce, faf |
| `PerGroupMedian(key)` | feats(按 indenter), tacquad(按 data_* split) |
| `PerTouchMedian(n=5)` | real_tactile_mnist, sim_tactile_mnist, sim_starstruck |
| `ForceThreshold(0.4)` | feats（有 marker，像素 diff 不可靠，改用 \|f_z\|） |

## 回归验证

抽出 7 个源内嵌的过滤逻辑等于改动已发布数据的生成路径。不依赖
"读代码觉得等价"来保证正确性。

已发布的 8G parquet 即旧行为的 ground truth——每行带 `capture` + `frame_idx`：

```
published:  SELECT capture, frame_idx FROM <source>/*.parquet   → 集合 A
new code:   跑新路径，收集保留帧                                  → 集合 B
断言:       A △ B 只能落在 1.5% 背景随机保留的部分内
```

`BG_KEEP_RATE=0.015` 是随机的，精确相等不可能。断言分两层：

- **确定性部分**（通过 area+intensity 的帧）必须完全一致
- **随机部分**（未通过但被保留的）仅校验数量在容差内

### 分级门槛

每个源在重构前先做可行性检查：该源 parquet 是否有可 join 的
`capture` + `frame_idx`。

- **有** → 重构到新抽象，附回归证明
- **无** → 原样搬进 `sources/`，不动内部逻辑，仅包一层 `SourceSpec`

预计 `feats`（力过滤，无像素判据）与 `fota_unlabeled`（经历 v9 通道修复，
中间态无从复现）落入第二档。

交付物是分级的：一部分源真正统一并有回归证明，一部分源仅被规整位置与声明。
不将后者表述为前者。

## 实施顺序

先原样搬，再重构——使每步重构可 diff。

```
1. MultimodalData 提交
   · twm/scripts/{apply_tactile_shift,rebuild_tactile_from_h5_shifted,
                  upload_tactile_correction}.py
   · .gitignore 补: .claude/  *.zip  twm/calibration/result/
   · git rm --cached probing_panda.egg-info/

2. 新仓库 initial commit = 40 个文件逐字复制，不改一行

3. 逐步重构，每步一个 commit：
   config → schema → filters → writer → baseline → spec → runner → 逐源迁移

4. 收尾：删除 mini_data_parquet/scripts/ 陈旧副本，
        改由发布脚本从新仓库同步（消灭漂移源头）
```

## 风险

| 风险 | 缓解 |
|---|---|
| 重构改变已发布数据的生成语义 | 回归验证 + 分级门槛；无基准则不重构 |
| 根分区已满（剩 3.8G） | 仓库仅代码 1.6MB；生成物写 `OUT_ROOT`，仍在数据盘 |
| 抽取内嵌过滤逻辑时引入静默偏差 | 确定性部分要求完全一致，不设容差 |

### 仓库位置变更（2026-08-05）

原计划把仓库放在数据盘 `/media/yxma/Disk1/yuxiang/`。实施中发现
security-guard hook 拒绝对 repo 与 home 之外的路径使用 `Write`/`Edit`，
只剩 Bash heredoc 一条路，且无法增量编辑——对还要新建 ~25 个模块、
且每个任务都有 TDD 修复轮次的计划来说，摩擦过大。

仓库改到 `~/gelsight-mini-pretrain`。原本"代码与数据同盘"的理由是根分区已满，
但仓库只有 1.6MB，该理由不成立。数据与生成物仍全部留在数据盘
（`RAW_ROOT`、`PARQUET_MAIN`、`OUT_ROOT`）。
