# REACT 数据集 v2（含腕部相机）处理与发布计划

**目标**：把 2026-09-09 至 09-11 新录的全部数据处理进 REACT 数据集，作为带腕部相机的新版本发布；旧版本另行保留。

**日期**：2026-09-11
**前置事实**：本次会话已完成 `tactile_freeze_L/R` 检测器、看护线程盲区修复（见下方账本）。

---

## 数据盘点

原始 H5 只剩 2026-09-09 之后的会话（旧会话的 H5 已删，只有派生产物）：

| 会话 | episodes | 体积 | 腕部相机 | 已构建 |
|---|---|---|---|---|
| motherboard/2026-09-09 | 2 (release 里有 3) | 54 G | 有 | ✅ |
| pushT/2026-09-09 | 1 | 49 G | 有 | ✅ |
| **pushT/2026-09-10** | **5** | **270 G** | 有 | ❌ ep_000 有被我中断的半成品 |
| **pushT/2026-09-11** | **1** | 16 G | 有 | ❌ |

v2 最终构成：motherboard 35 条（32 条旧 + 3 条带腕部），pushT 11 条（4 条旧 + 7 条带腕部）。
新增 pushT 有效时长约 43.7 分钟，其中 `2026-09-10/episode_003` 单条 18.2 分钟 / 128 GB。

## 已定的 6 件事（2026-09-11 操作者裁定）

1. **命名**：`yxma/React` 发新版本，旧状态在 Hub 上打 tag **`old`**。
2. **标定**：`pushT/2026-09-10`、`2026-09-11` 声明为 `2026-09-09` 纪元。残余偏移是物理的（刚体质心 ≠ 胶面），不是标定误差，不再作为重解外参的理由。已写进 `calib_epoch.py` 与 memory。
3. **splits**：pushT 按 motherboard 同方法重算，比例先过目再上传。
4. **`reference/`**：不删，重建。方法有文档（per-episode fuzzy mode 为主、session pool 为备，明确不用 p01），但 `fuzzy_mode_background` / `session_reference` 两个函数在仓库和 Hub 上都不存在，需要按文档重新实现；已发布的 136 张 PNG 作为回归靶子。
5. **已发布 36 条**：一并更新 bad_frames（新检测器增加 0.16–0.35% 尾部坏帧）。
6. **新数据只收 2026-09-10 及之后**（pushT 6 条）。0909 那四条用的是 Arducam，不进 v2 新数据。`episodes.jsonl` 写 `wrist_camera: null | "arducam" | "usb"` 而不是布尔值——两种相机的视场、曝光、畸变都不同，布尔值会把它们混成同一模态。

---

## 阶段 0：录制侧修复（不依赖数据，先做）

### 0a. 重连加速 — ✅ 已完成

实测预算（episode_001 九次停摆，4.8–5.2 s/次）：

| 环节 | 现值 | 来源 |
|---|---|---|
| 判定无帧 | `STALL_AFTER_S = 1.0` | `twm/recorder/rig.py:105` |
| 轮询粒度 | `SUPERVISOR_POLL_S = 0.25` | 同上 |
| **固定休眠** | **`time.sleep(3)`** | `camera_stream/base_video_stream.py::restart` |
| 重新打开 | 按序列号扫描 + `VideoCapture` + 设 FOURCC/BUFFERSIZE | `usb_video_stream.py::start` |
| 首帧偏暗 | 2–3 帧（均值 55→73→75） | 实测 |

已做的改动：
- `STALL_AFTER_S` 1.0 → **0.4**。依据：85,031 个正常帧间隔中，左 p99.9=122 ms、最大 203 ms；右 p99.9=116 ms、最大 216 ms。0.4 s 有近 2 倍余量，回测误判 0 次。
- `SUPERVISOR_POLL_S` 0.25 → **0.1**。
- `restart()` 的 `time.sleep(3)` → **`RESTART_SETTLE_S = (0.0, 0.5)` 退避 + 首帧验证**。21 次重开扫描（settle 0→3 s）证明休眠对首帧延迟毫无影响（一律 620–720 ms），但退避保留了扫描无法复现的情形（刚报过 EPROTO 的设备）。最坏 2.9 s，不比原来的固定 3 s 慢。
- `restart()` 现在**失败会抛异常**。旧实现睡完、重开、直接返回，设备死了也当成功——这正是第 9 次重启后 122 s 冻结却无告警的原因。
- `USBVideoStream.start()` 打不开设备时从 `exit()` 改成抛 `RuntimeError`：`exit()` 在非主线程里是 `SystemExit`，会静默杀掉看护线程本身。

实测结果：`restart()` 5 次均值 **809 ms**（最大 846）。端到端一次停摆 0.4 + ≤0.1 + 0.81 ≈ **1.31 s / 39 帧**，旧值 4.95 s / 148 帧（与数据里实测的 144–150 帧吻合）。**损失降到约 1/4。**

**注意**：这只是减小损失。停摆本身来自 USB（EPROTO），真正消除要把两个 GelSight 从 `1-12` hub 上挪到不同的 USB 控制器。

### 0b. 标定声明 — ✅ 已完成

`CALIB_SESSIONS` 加了 `pushT/2026-09-10`、`pushT/2026-09-11` → `2026-09-09`。`tests/test_raw_deleted.py::test_every_live_session_declares_a_calibration_epoch` 此前就在报这个缺口，现已转绿。

### 0c. 重建 reference/ 的两个函数（待做）

按 Hub 上 `reference/README.md` 的算法重新实现 `fuzzy_mode_background`（逐像素时间维 fuzzy mode）与 `session_reference`（整天池化），放进 `twm/react_toolbox/reference.py`，并补 per-episode 驱动脚本（Hub 上只有 session 级的 `build_session_references.py`，且它 import 的函数不存在）。验收：重算 motherboard 三个 session 的输出与 Hub 上已发布的 PNG 比对。

## 阶段 1：构建新 episode（6 条，只收 09-10 及之后）

```bash
rm -rf /media/yxma/Disk1/twm/release/pushT/videos/2026-09-10   # 清掉中断的半成品
python -m twm.react_preprocess build --task pushT --date 2026-09-10
python -m twm.react_preprocess build --task pushT --date 2026-09-11
```

- **不能和录制并行**：两者都打 Disk1。上次并行时 episode_002 的队列峰值从 8–17% 升到 24.4%。
- 读 286 GB，预计 2–4 小时；episode_003 单条就占一半。
- 产物：每条 7 个 mp4（3 视角 + 2 触觉 + 2 腕部）+ parquet + `_detect.pt`。

## 阶段 2：重跑 curation（全部任务）

```bash
python -m twm.react_preprocess curate --task pushT
python -m twm.react_preprocess curate --task motherboard
```

生成 `bad_frames.json` / `segments.json` / `episodes.jsonl`。新的 `tactile_freeze_*` 会在这里生效——pushT 2026-09-10 的四条已投影：切除 36.6%/4.8%/2.8%/… 详见本次会话导出的清单。

## 阶段 3：splits

```bash
python twm/scripts/build_splits.py --root /media/yxma/Disk1/twm/release/pushT
python twm/scripts/build_splits.py --root /media/yxma/Disk1/twm/release/motherboard
```

上传前把 train/guard/test 比例给用户看。任何在 `episodes.jsonl` 里而不在 `splits.json` 里的 episode 会被 `ReactVideoDataset` 静默当成 train——这是训练泄漏。

## 阶段 4：预览

```bash
python twm/scripts/build_episode_previews.py --task pushT --date 2026-09-10
python twm/scripts/build_previews_index.py
```

预览面板含腕部一行（本会话此前已加）。

## 阶段 5：校验（上传前必过）

```bash
python -m twm.dataset_layout check /media/yxma/Disk1/twm/release/pushT
python twm/scripts/certify_release.py
python -m pytest tests/ -q
```

`dataset_layout` 已能处理腕部视频的有无（全无只警告，只有一个则失败）。

## 阶段 6：上传与旧版处置

按第 1 项的决定执行；`dataset_prep.upload` 在 `dataset_layout` 不过时会拒绝上传。

## 阶段 7：文档

数据卡写明：腕部相机自哪一天起存在、两个 GelSight 的 USB 停摆及其在 `bad_frames.json` 中的类别、标定纪元与已知偏差、v1 与 v2 的差异。

---

## 账本

| 发现 | 证据 | 修复 | 验证 |
|---|---|---|---|
| 触觉冻结无人检测 | pushT 09-10 左传感器冻结 9 段、最长 121 s，episode 仍 `valid=True` | 新增 `detect_tactile_freezes` + `tactile_freeze_L/R` 类别 | `tests/test_tactile_freeze.py` 10 项 |
| 重连耗时 5 s，其中 3 s 是未经测量的固定休眠 | 21 次重开扫描 settle 0→3 s，首帧延迟一律 620–720 ms，与 settle 无关 | 退避 + 首帧验证，`STALL_AFTER_S` 1.0→0.4，轮询 0.25→0.1 | 实测 809 ms；`tests/test_stream_restart.py` 8 项 |
| `restart()` 失败也静默返回 | 第 9 次重启后设备已死，看护仍记为成功 | 失败抛异常 | `test_a_dead_camera_raises_instead_of_pretending_to_have_recovered` |
| 打不开设备时 `exit()` 杀掉看护线程 | 非主线程的 `SystemExit` 只结束该线程 | 改抛 `RuntimeError` | `test_an_open_that_raises_is_retried_then_reported` |
| 两个 GelSight 都在报 USB EPROTO | 左 120 次 / 右 45 次；ep_004 与 09-11 ep_000 是右侧重启 | 未修（硬件）：需把两者分到不同 USB 控制器 | — |
| 重开后的流静默退出看护 | 第 9 次重启后不再告警，122 s 冻结 | 流出过时间戳即永久受看护；`None` 按上次见帧算陈旧 | `test_a_stream_that_reopens_but_never_delivers_is_not_mistaken_for_a_dummy` |
| 已发布数据尾部也有冻结 | 32/35 motherboard、4/5 pushT，均在末尾 0.7–1.1 s | 未动（会改 Hub 上已发布文件） | 待第 5 项决定 |
| 预处理与录制争盘 | episode_002 队列峰值 24.4% vs 平时 8–17% | 阶段 1 明确禁止并行 | — |
| 腕部相机画面偏暗 | 同一表面（木桌）上腕部 p75/p90 = 106/120，参考 146/152 | 采集侧 gamma 450→500（+10%，免费）；发布侧按相机代次固定指数 usb 1.5 / arducam 1.2 | `tests/test_wrist_tone.py` 11 项；实测木桌帧 p50 96→132（参考 136） |
| pushT 新数据无物体位姿 | 6 条的 `optitrack/motherboard` 全 0 样本 | 未修（需 Motive 侧广播刚体） | — |
| 09-09 那条的物体位姿是 3 个样本摊出来的 | `motherboard` 仅 3 样本，parquet 却 0% NaN | 未修：`cam_align_poses` 缺陈旧度上限 | — |

### 否决的想法

- **把冻结帧从 mp4 里物理删掉**：会打破 `parquet 第 i 行 ↔ 视频第 i 帧` 的现有约定，且要重编码全部视频。沿用 `segments.json` 索引干净区间，零重编码。
- **靠"找时间戳空洞"检测停摆**：停摆不产生空洞——recorder 把上一帧重复写入，时间戳照常前进。必须查像素/指标的逐位相同。
- **把 settle 直接砍成 0**：扫描是在健康设备上做的，复现不了刚报 EPROTO 的状态。保留 0.5 s 的第二次尝试，代价是最坏情形仍 ≤ 原来的 3 s。
- **`episodes.jsonl` 用 `has_wrist` 布尔值**：会把 Arducam 和 USB 两种腕部相机混成同一模态。改存相机种类。
- **腕部亮度用线性增益补**：腕部的 p99.5 比参考还高（176–213 vs 163–166），增益要么削掉高光尾巴、要么等于没做（0.77–0.95×）。改用幂曲线。
- **按 episode 拟合曝光补偿**：腕部相机装在会动的传感器上，视野在木桌和黑主板之间来回跳，按 episode 拟合会让同一个物体在不同片段里亮度不同。改成每代相机一个常数。
- **把腕部中位数匹配到 RealSense 中位数**：腕部是特写、参考是广角俯视，中位数的差别来自取景而非相机。改为在两者共有的表面（木桌）的高分位上比。
