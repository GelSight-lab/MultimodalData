# Task Plan 2: 力恢复双方法实现 + 评估 + HF 可视化站

## Goal 2(活跃)
选择两个最合适的方法实现,自行 debug 和评估,可视化方法和结果到 HF 网站。
选定方法(基于文献调研):
1. **FEATS 伪力标注**:GelSight Mini 图像 → normal/shear 力分布 → 回填数据集力通道
2. **DexForce 式 action 变换**:F_normal → 沿 sensor normal 的 force-informed
   position target(虚拟穿透),与现有 pose action 融合

## Phases (Goal 2)
- [x] P1: FEATS 跑通 → **负结果**:我们的 gel 无 marker,FEATS(marker gel 训练)
      对最强接触帧输出与无接触帧完全相同(OOD 塌缩)。有证据、已记录。
      方法1 改为 gsrobotics 光度立体深度 + Winkler 弹性地基 F=E*/h·Σδ·dA
- [x] P2: 方法1 实现 + 调试(边界伪影裁剪、中位数零图+MAD 5σ 阈值、
      fresh-only median-3 去尖峰 4-8%→0-0.6%);批量推理进行中(7/16 npz)
- [x] P3: 方法2 实现 + 评估:invariance=0、roundtrip=1e-14N、穿透≤2.6mm;
      关键修正:gel 法向不是 [0,0,1],用标定的 gel_axis_in_rigid(双球标定,1°一致)
- [x] P4: 可视化(timeline、depth panels、DexForce offset 图、overlay mp4)
- [x] P5: 全量 eval + 站点 + 发布 → https://huggingface.co/spaces/yxma/react-force-recovery ✓(回读 4 表)

## Goal 3(批量+外部验证+改进)结果
- **批量**:36 集 × 2 side 全部处理,0 失败(4 worker 双 GPU,~1.3h)
- **外部验证(FEATS 真值)**:val ρ=0.70/r=0.63(≤30N 法向);**未见 indenter ρ=0.85**;
  shear captures ρ=−0.15(记录为盲区);跨 split 绝对刻度漂移 2-4×(集内相对力可靠)
- **验证驱动的改进**(每项都有 before/after):
  1. 1/7 边缘裁剪统一(MLP 训练域)  2. 逐帧稳健背景平面移除(ρ 0.42→0.61,
  React 阈值 50µm→10µm)  3. marker 点 inpaint(替代 SDK mask 路线)
  4. 二次背景面试过并否决(ρ 降到 0.50)  5. FEATS 定标绝对刻度(40× 理论值;
  motherboard 最强按压 0.2N→7N,符合直觉)  6. 下游常数随刻度更新(k=1500N/m)
- **改进后的 React 内部指标**(72 sides):SNR 中位 1209(此前 ~1);
  ρ vs intensity 中位 0.84(此前 0.36-0.60);尖峰滤后 0.00%;力峰 1.5-22.8N
- **事故与修复**:test split 验证覆盖了正式定标文件 → 只允许 val 写 + 
  normalize_scale 从 volume 精确重算(抓到 2 个受污染 npz)

## Debug log (Goal 2)
1. UNet 参数名 output_size→out_sz
2. 点检测阈值 55 检出 0 点(我们的点最低灰度 56)→ 百分位阈值 → 随后发现根本无点
3. FEATS OOD 塌缩 → 换光度立体路线
4. Poisson 边界 1.1mm 假深度 → 裁 12/16px 边缘
5. 参考帧含轻接触 → 阈值爆 30×(0.159 vs 0.006mm)→ median+MAD
6. 单帧尖峰 4-8% → fresh-only median-3(row-wise 会把重复值数 3 次而保留尖峰)
7. gel 法向假设 [0,0,1] 上升沿对齐为负 → 标定轴翻正
8. DexForce 图:offset 0.2mm 在 100mm 轨迹上不可见 → 直接画 offset + 力双轴

## Env facts
- GPU: RTX 2080 Ti (11GB) + GTX 1080;torch 2.1.0+cu121 OK;HF auth = yxma
- FEATS: github.com/feats-ai/feats;权重在 repo `src/feats/models/unet_09042025_124903_80.pt`
  + `normalization_08042025_122519.npy`;输入格式需读代码确认
- 我们的触觉帧:H5 `gelsight/<side>/frames`,640x480 RGB(HDF5 内 RGB)
- 注意:FEATS 训练输入可能是 320x240 或原生 GelSight Mini 分辨率;需查 predict 代码

---

# Task Plan 1(已完成): React 数据集修复 + preprocess 重构

## Goal
1. 自动修复已识别的数据集问题(触觉重复标注、时间戳对齐、双重校正)
2. 把散落在 `MultimodalData/twm/scripts/` 的 41 个 preprocess 脚本重构成一个干净的包
3. 让 preprocess 代码与数据同处(shipped with dataset,像 `toolbox/` 一样),而不是埋在 MultimodalData 里

## Phases
- [ ] Phase 1: 探明现状(41 脚本里哪些是真 pipeline、哪些是一次性;确认"数据folder"位置)
- [ ] Phase 2: 设计新包结构 + 写计划
- [ ] Phase 3: 实现 preprocess 包(含新时间戳对齐 + 防双重校正)
- [ ] Phase 4: 实现数据修复(tactile_is_new 列 + README 数字更正)
- [ ] Phase 5: 验证(跑通、对比新旧输出一致性)
- [ ] Phase 6: 交付(commit,可选上传 HF)

## Key Questions
1. ~~"数据folder" 在哪~~ → **已答**:`twm/react_toolbox/` → HF `toolbox/`。同理 preprocess 应
   为 `twm/react_preprocess/` → HF `preprocess/`,与数据同发布。
   本地暂存区 = `/media/yxma/Disk1/twm/release/<task>/`(镜像 HF `data/<task>/`)
2. ~~41 脚本分类~~ → **已答**(见下)
3. 新旧 H5 如何区分 → 检测 `gelsight/<side>/timestamps` 是否存在

## 脚本分类(41 个 → 真 pipeline 只有 8 个)
**核心 pipeline(要重构进包,~1050L)**
build_video_release(244) detect_bad_intervals(225) build_release_curation(155)
build_release_depth(103) build_release_previews(92) add_object_pose(95)
enrich_parquet_index(58) build_release_publish(84)

**一次性迁移(已执行完,归档)**:apply_world_offset_2026_05_19, apply_tactile_shift,
rebuild_tactile_from_h5_shifted, recut_tactile_latency, upload_tactile_correction,
publish_2026_05_19, publish_wave2, bake_ot_loss, trim_pt, rebuild_post_trim,
build_release_publish_depth
**旧 .pt 格式(已废弃)**:build_episodes_from_h5, build_episodes_multicam,
build_segments, build_mode2_segments, extract_gs_refs
**诊断/分析(一次性)**:freeze_classify, freeze_diagnose, inspect_freezes_v3,
find_real_still, check_recording_fps, test_world_offset
**工具(保留原地)**:play_react_pt, latency_align_viewer, test_latency,
build_latency_clips, build_latency_correction_clips, mp4_5x, regen_figures,
build_previews_index, build_episode_previews, build_lerobot_dataset

## Decisions Made
- **包名/位置**:`twm/react_preprocess/` → HF `preprocess/`(与 `toolbox/` 对称)
- **一次性脚本不删**,移到 `twm/scripts/oneoff/` 归档,保留历史可追溯
- **双重校正防护**:H5 有 per-sensor 时间戳 → 用时间戳对齐 + 禁止 +15 移位

## Errors Encountered
- **验证方法本身错了**:最初用解码后的 MP4 做逐位比对判断重复帧 → 432/600 "失配"。
  原因:H.264 有损,重复的源帧解码后像素并不相同。改用源 H5 作 bit-exact ground truth,
  MP4 比对改为带容差并报告实测分离度。
- **性能缺陷**:shift 搜索对每个候选重读一次 H5(21×900 帧)→ motherboard 超时 >120s。
  改为只读一次源、滑窗比较 → 16s。
- **off-by-one 回归**:滑窗后第 0 帧有了真实前驱,而 proxy 第 0 帧只能按约定取 True,
  产生 1 处假失配。第 0 帧两边都是约定值不含信息 → 比较从第 1 帧开始。回归测试抓到的。

## 关键验证结果(证据)
- 标量代理 vs 源 H5 像素:**pushT 4/4 + motherboard 3/3 全部 0 失配**(899 帧/集)
- 自动检出已发布数据的 `tactile_shift=+15`;其他 offset 失配率 >33%(强区分度)
- 回填幂等:重跑结果逐位一致
- 代码可从数据目录独立运行(模拟用户只下载数据集的情形)✓

## Phase 6 交付记录
- commit `726db47` — react_preprocess 包
- HF `yxma/React` 一次 commit 推送 60 个文件:23 代码 + 36 parquet + README
- 回读验证:Hub 上 preprocess/ 12 文件、toolbox/ 13 文件;
  下载 parquet 确认 19 列、`tactile_*_is_new` 为 bool、fresh 29.2%/29.1% ✓
- 顺带发现并记录:`motherboard/2026-05-19/episode_{003,004}` 只有 4.0s / 7.0s
  (中位数 213s),是中断的录制 → README 标注建议过滤,不删除以保持编号稳定

## Phase 7(追加):补全最后 2 个未迁移脚本
- `detect.py` + `curation.py`:全部 5 类检测区间、36 集,与已发布 JSON **逐区间 0 差异**;
  segments 逐字段 identical(76+17 段)。顺带修复了我重组 scripts 时弄断的
  `build_release_curation.py -> build_segments`(find_clean_segments 移入包)。
- `previews.py`:只迁移策略(标定选择/trim/world offset/输出布局),渲染器留在
  scripts 作 thin adapter(需要 rig 本地标定,不随 release 发布)。
  端到端验证:重渲 pushT/episode_000 与已发布 preview **首帧 MAD 0.00**、同 900 帧。
- 旧脚本移入 `scripts/superseded/` 并附 README(保留 provenance,禁止运行)。
- 8/8 核心 pipeline 脚本全部有包内归属;CLI 增加 `curate` 子命令。

## Status
**全部完成**(Phase 1-7)。commit 726db47 + 本次追加 commit;
HF 已发布 60 文件(代码+parquet+README)并回读验证。
