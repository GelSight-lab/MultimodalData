# Task Plan 6(活跃): 形状保真 + 通用化 + 双数据集逐indenter ρ≥0.95

## Goal 6 验收标准
1. 形状:ball→半球剖线;triangle/quad/star 可辨;A-E 字母可读
2. 方法 general(无逐家族技巧;两个无点数据集通用)
3. Drake 式接触模型;0-20N 各类 indenter ρ≥0.95;误差尽量低
4. 高误差 outlier 逐帧分析原因

## Goal 6 已确认证据
- 形状画廊:round/star/triangle/quad 轮廓已可辨但软;字母 raw 清晰、重建成软团
- 高通测试:LUT 高通里字母 counters 存在但软;**MLP 高通边缘锐利**(互补!)
- nnmini+fast_poisson 直测 GlowTact round:幅值 14×、ρ=0 → 绝对色 MLP 在此传感器出域
  (它只提供可靠高频,不提供幅值)
- **方案:梯度域频带融合** g = lowpass(g_LUT) + s·highpass(g_MLP),
  s 由中频带能量比自校准;单次 fast_poisson
- 渲染注意:字母内浮雕仅占总深~10%,验收渲染用局部对比(高通或分层色标)

## Goal 6 计划与进展
- [x] F0: 证据收集完成,方案定向
- [x] F1a: **半球验收 ✓**——round 剖线形状贴合解析球(幅值 +30% 全局偏置,可重标)
- [x] F1b: **字母在梯度层清晰 ✓**——|g_LUT| 图里 B 的双 counter 锐利;
      模糊源=积分步(光晕色误读 + Poisson 平滑),非色域限制
- [x] **否决**: MLP-LUT 梯度域频带融合——8 种符号组合最高 corr=0.049,
      MLP 在此传感器的"锐边"是 |n|≥1 饱和伪影(rms 220 vs LUT 0.027,已加物理钳制仍不相关)
- [ ] F1c: 积分改进:光晕色处理(halo 色与接触坡色区分:用 dI 幅值分层
      或 valid-mask 内外分治)+ 可选加权 Poisson;字母渲染用梯度-深度双层
- [ ] F2: 全量特征(修 +30% 幅值偏置)+ hydroelastic p(δ) → GlowTact 逐 indenter ρ 表
- [ ] F3: cnc_Mini 通用化:z 监督幅值校正路线(无球家族);参考=最轻帧中位
- [ ] F4: 高误差 outlier 逐帧归因(渲染 top-20 误差帧,分类:贴边/深压饱和/
      gel 损伤区(左上有划痕)/光晕误读)
- [ ] F5: 站点 + commit

## 复跑入口(context 丢失时)
- 融合类: twm/force_recovery/fusion_recon.py (FusionReconstructor;_lut_gradients 可单用)
- LUT: lut_calibration.py, 标定文件 /media/.../force_recovery/lut_calibration/glowtact_lut.npz
- 全量特征缓存: feature_cache/lut_full.json (6264帧, 无 f01, 含 x,y,z,f,vol,vol2,area,maxd,cx,cy)
- 探针小缓存: lut_probe_frames.json (365帧含 peak)
- 最优力模型配置见 Task Plan 5 最终数字节

---

# Task Plan 5(已完成): 光度算法重做 + 接触模型升级 → ρ≥0.9, MAE<1N

## Goal 5
1. 光度重建按 Dong/Yuan 经典管线重做:差分图 dI=(img−ref) → 逐传感器 RGB-LUT → 梯度 → Poisson
2. Winkler → Drake 式非线性弹性地基 p(δ)=k1δ+k2δ² (+Hertz 形状归一)
3. 目标:ρ 0.9–0.95+,MAE < 1N

## 已确认的关键事实
- **诊断**:nnmini 重建峰深 vs 指令压深 ρ=0.39,幅值仅 ~25% 且随深度饱和
  (0.30→0.19)——光度是根本瓶颈,力 ρ=0.65 全靠面积项
- GlowTact `round` 家族 = 球形压头标定源:1152 帧、位置铺满、Hertz r=0.97、
  每家族带 initial.jpg;z 有未知零点 z0 → 用 a²=2R(z−z0) 回归同时解 R 和 z0
- 参考实现:~/gelsight_heightmap_reconstruction (Dong python) +
  ~/gelsight_driver/Bnz (Yuan matlab): dI 分箱 ±90/90bins → LUT(gx,gy) → fast_poisson
- MM_PER_PIXEL(1/7裁剪后 320 宽)= 13.29/320 ≈ 0.0415 mm/px

## Phases
- [x] P1: 精确球面拟合 a²=d(2R−d) → R=3.35mm z0=0.81mm R²=0.948
- [x] P2: LUT 构建 + fast_poisson 重建
- [x] P3-P4: 迭代到 pooled ρ=0.82/MAE 2.46N;round ρ=0.91/1.86N
- [ ] P5: cnc_Mini 迁移(未做——React 传感器需一次球压标定,已明确路径)
- [x] P6: commit(站点更新待做)

## Debug ledger (Goal 5)
| found | evidence | fix |
|---|---|---|
| nnmini 深度幅值坏 | peak vs z ρ=0.39, 幅值25%且饱和 | 换经典 LUT 管线 |
| **gsrobotics Poisson 求解器幅值 bug** | 解析合成场只回 39%(线积分105%证明LUT对) | 换 Dong fast_poisson(100.7%) |
| LUT 幻影尖峰 | 深度ρ塌到0而中心正确 | Yuan validmask(\|dI\|>8 才查表) |
| 陡坡未训练 | inner_frac=0.85 排除外环 | 0.97a + 精确球面梯度 |
| 位置相关增益/基准 | 位置二次解释80%残差 | round 监督 u(x,y) 场 |
| 贴边按压污染 | 全量缓存 ρ 反降到 0.45 | 指令位置内部过滤 |
| quad(平头)拖后腿 | 家族 ρ 0.69 | 形状自条件化 s1=V2A/V², s2=V/(A·D) → 0.81 |
| **否决**: f01 补偿 | vol ρ 无改善且 LUT 稀疏化 | 回退 |
| **否决**: 力监督交替场拟合 | 0.735 < round 监督 0.757 | 回退 |

## 最终数字 (held-out, GlowTact 0-20N)
起点 MLP+Winkler 0.63/4.4N → LUT+求解器修复+场+地基 0.80/2.75N →
+形状自条件 **0.82/2.46N**;家族已知 0.85;**round(球,物理精确处)0.91-0.94/1.5-1.9N**
天花板参考:rho(F,z)=0.975。目标 0.9/1N:球形家族 ρ 达标;pooled 差 0.08;
MAE<1N 超出几何法信息量(FeelAnyForce 20万监督帧也只到 2.1N)

---

# Task Plan 4(已完成): cnc_Mini 方法优化 + FEATS 对比 + 可视化

## Goal 4
1. ~~initial 命名图像~~ → 查证:无(tar+全 zip CD;DoubleTower=随机对)。
   但 z_mm = 指令压深(逐帧真值信号!)
2. 边缘按压过滤 + 回答 max depth vs volume(实证)
3. 优化方法(目标:ρ 达到可标注水平)
4. 我们的方法 vs FEATS 模型,同真值对比(两个数据集:cnc_Mini=双双外域;
   FEATS val=FEATS 本域)
5. 可视化:20 图像样本(raw|深度|3D 面|力)+ 10 视频样本(React clip 加深度列)

## Phases
- [ ] P1: 特征缓存(cnc_Mini train+val ~1300 帧:vol/area/maxd/vol^1.5/soft-vol/
      质心/边距 + FEATS 模型输出 + z_mm)
- [ ] P2: 变体矩阵(fit train / eval val + 留探头):vol | maxd | area |
      power-fit | 小线性组合 | 软阈值 | 边缘过滤 | 照明归一
- [ ] P3: FEATS val 上同样对比(FEATS 本域 head-to-head)
- [ ] P4: 可视化 20 图 + 10 clip
- [ ] P5: 站点更新 + 发布 + commit

---

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

## Goal 6 F2+F4 结果 (2026-08-06, held-out, GlowTact 0-20N)
**Outlier 归因**: top-20 误差帧 19/20 = EDGE+DEEP+HIGH-F(贴边 4-5mm 深压:
印痕出视野 + gel 触底,力涨而可见压入饱和,全部严重低估)→ 物理作用域:
接触完整在视野内 (质心±等效半径在 interior 框内) 且 z≤4.2mm(pad 厚)
**作用域内 7 种子**: pooled ρ 0.904 [0.876,0.919], MAE 0.95N [0.87,1.04]
逐 indenter 中位: quad 0.988✓ triangle 0.971✓ B 0.927 quad_small 0.907
star 0.860 round 0.811(round=受限力程效应:in-view 剔除了其大深压帧)
**途中 bug**: 缓存 area 是像素数,误除 MM_PER_PIXEL 使等效半径×24 →作用域 0/6257
**待续**: round/star 提升(受限力程分析、halo 积分修正)、cnc_Mini z 监督
通用化、站点更新
