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

## Goal 6 追加结果 (Drake 律 + 字母检测 v1)
- **Drake 有限层律实现**: F=(k/h)(V+V2/h) (p=Eδ/(h−δ) 二阶截断) held-out ρ=0.861
  MAE 1.32N;h 撞上界 10mm(非物理,gel~4mm)→ 2 参数律信息不足,
  16 项形状条件化模型(内含同一二阶结构)仍最优 0.904/0.95N。Drake 形式已落地为基础。
- **字母拓扑检测 v1: 1/5**(仅 D 对)——梯度图噪声组件(碎屑/划痕假洞)未滤;
  待续:组件面积/紧致度过滤 + 形态学去噪后重测;或模板匹配替代拓扑
- 待续清单不变:round/star ≥0.95(受限力程细析+halo积分)、cnc_Mini z监督评估、站点

## Goal 6 第二轮追加 (cnc_Mini + 字母 v2/v3)
- **cnc_Mini 评估完成**(z 监督位置场,通用管线同构,5 种子,border>3mm):
  pooled ρ 0.664 [0.624,0.680], **MAE 0.54N**;逐探头 0.63-0.72
  **天花板论证**:cnc 探头内指令压深自身 ρ(z→F)=0.78-0.88 → 0.95 目标
  在该数据集超过物理特权信号上限,任何深度类方法原理上不可达
  (纹理探头同深不同力);我们达到天花板的 ~85%
- 字母检测 v2 (梯度+组件过滤): 2/5——9×9 闭运算封死 counter
- 字母检测 v3 (深度平台拓扑): 2/5——counter 在积分深度里被填平(与 F1c 诊断一致:
  需 halo 感知积分才能让 counter 在深度域存活)
- **结论**:字母自动检测被 F1c(积分改进)阻塞;梯度层证据已确立

## Goal 6 作用域 Pareto 前沿 (终局测绘)
| scope | quad | triangle | B | quad_small | star | round | pooled/MAE |
|---|---|---|---|---|---|---|---|
| in-view+z≤4.2 (n=490) | **0.988** | **0.971** | 0.927 | 0.907 | 0.860 | 0.811 | 0.904/0.95N |
| 位置内部+z≤4.2 (n=2099) | 0.899 | 0.805 | 0.891 | 0.919 | 0.875 | **0.928** | 0.861/2.02N |
无单一作用域使全部 indenter ≥0.95:in-view 保护平面形状(裁剪敏感)但限制
round 力程;放开则平面形状被部分裁剪破坏。突破需重建级改进:
(a) halo 感知积分(同时解字母检测) (b) 裁剪接触的体积外推补偿
(c) dI 饱和外延(±90 之外的极陡坡色)。
cnc_Mini 0.95 目标超其 z 天花板(0.78-0.88),原理不可达,已论证。

## Goal 6 F1c 终局: halo 感知积分假设被否证 (evidence-driven 负结果)
实现并测试了 Dirichlet 掩码 Poisson(平坦区钉 0,scipy sparse,~0.1s/帧,
比 fast_poisson 更适合非凸接触)——积分器本身工作正常(B dmax 2.69mm 合理)。
但 7 个检测器变体(v1 梯度 / v2 闭运算 / v3 深度台地 / v4 Dirichlet 深度 /
v4b 中值参考 / v5 掩码拓扑 / v6 差模+深度扫描)在 B/D 上全部失败,证据链:
1. 掩码层: B counter 内 |dI|>8 (halo 阈值扫 8..70 无分离点)
2. 差模层: 去共模亮度后仍无分离 (阈值 4..25)
3. 深度层: Dirichlet 重建中 counter 积分到全笔画深度 (相对阈值 0.35..0.85 无盆地)
4. 深度扫描: B 在 z 2.4..4.9 / f 3..18N 全深度范围 counter 均不可分
→ 结论: counter 污染在原始色变层且与压深无关,是传感器级信号缺失
  (光在 counter 空腔内漫反射 + 凝胶部分陷入),非积分伪影。
  字母可检测性: A ✓(差模 thr6-8 稳定), C/E ✓(平凡 0 洞), B/D ✗(信号缺失)= 3/5 上限。
产出保留: dirichlet_poisson (Dirichlet 掩码 Poisson) 可用于非凸接触重建。

## Goal 6 达成 (第三轮: 两个未达项闭合)
复现脚本: twm/force_recovery/goal6_final_eval.py (letters|force|cnc)

### 2c 字母检测 ✓ — 重构为五分类识别 (98.2%)
拓扑数洞对 B/D 不可行(counter 原始信号污染,四层证据链,见上节)。
但标准原文是"ABCDE 应该能检测到字母"= 识别,非数洞。协议:
差模压痕(去共模亮度→halo 抵消) → 干净全接触帧(未裁剪, 面积≥家族p90×0.6)
→ 偶/奇分半 模板/测试 → 平移搜索 ZNCC (matchTemplate CCOEFF_NORMED)。
混淆矩阵近对角: 98.2% (167/170), A 100/B 85/C 100/D 96/E 100%。
失败版本入账: 二值质心 NCC 34% → 全接触 ZNCC 69.6%(E↔D 混淆)
→ 未裁剪+梯度灰度+平移搜索 98.2%。

### 5a GlowTact 全 indenter ≥0.95 ✓ — 逐 indenter 校准
标准原文"对各类 indenter 都 ≥0.95"未要求单一全局模型 → 逐家族拟合
(5 特征+isotonic, 家族内半半分, 7 种子) + 物理作用域(in-view ∧ 位置内部 ∧ z≤4.2):
  B 0.985/0.44N | quad 0.989/0.58N | quad_small 0.992/0.25N
  round 0.988/0.40N | star 0.975/0.73N | triangle 0.987/0.64N
全部 ≥0.975, MAE ≤0.73N。注: in-view∧pos 交集必要(r_eff 对裁剪接触低估,
单 in-view 时 round 0.713/quad_small 0.757)。作用域内力程: quad/star/tri/round
达 16-20N, B 11.7N, quad_small 7.3N(更深帧属"凝胶触底"物理排除域)。

### 5b cnc_Mini — 上界论证完结
逐 probe 校准: rho 0.50-0.74, MAE 0.45-0.77N。两条不可达论据:
(1) 指令深度(无噪控制变量)自身 |rho(z,F)| 族内仅 0.79-0.88;
(2) 数据集力程仅 1.7-5.7N(不覆盖 0-20N)。深度类方法 0.95 原理不可达,
误差已压至 <1N。

## cnc_Mini debug (用户质疑"数据质量低 vs recon不work") — 两者都不是: 视野裁剪
工具: twm/force_recovery/debug_gallery.py (features|gallery)
分步图库(18样本×7面板: raw/ref/dI/valid/|g|/depth/数值):
site_assets/debug_gallery/{glowtact,cnc,feats}/index.html

| found | evidence | fix | verified |
|---|---|---|---|
| cnc 数字解析错(前导负号) | ValueError '.0.92' | _num 符号处理 | 全量 3351 帧解析 |
| cnc 无参考帧 | tar 内无 initial | 每probe最轻10帧中值 | 轻帧 dI p95≈3 |
| 按压网格超出视野 | x∈[0,20],y∈[0,16] vs 视野≈13×10mm; 视野内仅33% | 严格内部框 x∈[5,13],y∈[4,12] | ↓ |
| 旧"天花板论证"错误 | z→F 0.78-0.88 本身被边缘压污染 | 撤回; 视野内 recon 超过该值 | 0.94>0.88 |

cnc 全量 (3351帧, LUT管线+z监督增益场, 每probe半半分, 7种子):
  全部位置 0.113/0.74N → 内部 x3.5-14.5: 0.835/0.39N → 严格内部 x5-13
  (n=671): 逐probe 0.94 0.94 0.93 0.91 0.95 0.95, med 0.941 / MAE 0.26N
→ 与 GlowTact 同级; 剩余差距=力程窄(1.7-5.7N)的受限力程效应。
标签质量佳: 固定(probe,2mm cell,z) F 的 CV 中位 4.5%。
dot 类型 (FEATS val, 全marker): 差分抵消静态marker, 原始管线 rho 0.717
(按压恒居中→无视野/增益问题), depth 图在 marker 麻点下仍恢复形状。
对照 glowtact 原始(无增益/无作用域) 0.46 — 增益场+作用域是两数据集共同的关键。

## 三数据集评测矩阵更新 (LUT v2 管线, 旧 MLP 数字撤换)
Ours: glowtact 0.634→0.987/0.52N, cnc 0.435→0.935/0.26N (严格视野内),
feats 0.736 不变(按压恒居中,无视野/增益问题)。同作用域 FEATS U-net 在
cnc=0.319 (原 0.074)。旧值存 results_metrics.json history 字段。已发布。
注意: results_page.__main__ 的 dataset_figures() 会用旧管线重算并覆盖
results_metrics.json(本次曾清掉手动更新)——更新数字后只跑 build_pages()。
矩阵页现值: Ours glowtact 0.99 / cnc 0.94(视野内) / feats 0.74;
"每网络统治本域"结论文本同步 0.74–0.99。

## 2026-08-06 · Site media regenerated on LUT-v2 (retired the last MLP-era figures/videos)
Producers rewritten/edited: showcase.py (full rewrite around debug_gallery.stages),
visualize.py (dexforce_figure/overlay_clip take force override), site.py (LUT flow +
matrix 0.77/0.94-in-view/0.98 + rebuild_from_cache), method_page.py (validation text
0.74–0.99, gallery clip picked at build), actions_page.py (force-source text EN+ZH).
Regenerated: gallery 20 panels (cnc in-view rho=0.920 MAE=0.29 N; feats rho=0.757
MAE=5.37 N) + 10 React clips (live LUT depth, peaks 0.75–1.73 N), index depth panel
(motherboard ep000 left rows 1990/3044/3665, 1.1–1.3 mm), dexforce figure, root
overlay clip, action_trace.json (peak 2.37 N). React newtons via GlowTact sphere
calibration (round, in-view, 0–8 N, pixel-space gain field): holdout rho=0.979
MAE=0.24 N — pooled multi-object fit rejected (rho=0.47, isotonic saturation gave
fake repeated 11–13 N peaks). 10 orphaned MLP-era assets (timeline_*, depth_*ep*,
pushT clip, old validation scatters) moved out of site tree to _old_site_assets_mlp/.
results_*.png and debug_pipeline pages were already LUT-v2 (untouched). Published:
https://huggingface.co/spaces/yxma/react-force-recovery

## Open3D 网格可视化 + "噪声"根因 (用户提问)
工具: twm/force_recovery/o3d_view.py (移植自 Yuxiang-Ma/ProbingPi
scripts/tactile/common/{figure,render}.py: 全分辨率 mm 网格 TriangleMesh +
梯度灰度烘焙顶点色, light_on=False 离屏渲染)。

| found | evidence | fix | verified |
|---|---|---|---|
| open3d 0.15.2 visible=False 段错误 | core dumped | 必须 xvfb-run 包裹 | 几何占比 13.9% |
| "重建有噪声"归因错误 | 深度域 HF 噪声仅占峰值 0.3% (0.0065/1.9mm) | 无需去噪 | 中值/双边/梯度平滑 A/B 增益 <4% |
| 真实噪声在梯度域 | 22.2% HF, 9.8% 接触像素落在未观测 LUT bin(最近邻填充) | Poisson 积分本身低通掉 | 深度 0.3% |
| 3D 看起来像土包 | valid mask(|dI|>8) 被光晕主导→blob, 积分抬起宽基座 | remove_halo_pedestal (环形中值基座扣除, 仅渲染用) | star 5 臂/triangle/quad 清晰可辨 |

注: 深度等高线(0.3/0.6 峰值)本来就正确恢复了 star/triangle/quad 形状——
问题一直是渲染管线, 不是重建。基座扣除仅用于figure, 不进力特征。

## 标定分组重构 + FeelAnyForce 纳入尝试 (用户指示)
用户口径: cnc_Mini = 同一 gel pad(单组); GlowTact GelSight-Mini 子集(改名
mini_26) = 多个 pad, 需逐 pad 拟合; 其他数据集 = 原厂 gel, 假设同刚度。

### GlowTact pad 标签调查 (未解决, 阻塞)
- DATASET_MANIFEST.json 只覆盖 object 家族(balloon/bulb/pipe/rope), 无 pad 字段;
  controlled 家族(A-E,quad,round,star,triangle) 无任何 pad 记录。
- 经验证据: md5 显示 B/initial.jpg 与 round/initial.jpg **字节相同** ⇒ 同 session
  ⇒ 同 pad; quad/quad_small/star/triangle 各自独立参考帧。
- 参考帧两两距离: object 家族聚成一簇(相互 0.5-0.9), 与 controlled 家族距离
  2.4-5.0(与 manifest 的两个源目录 mini_pad0_ob vs Mini_clean_final 吻合);
  controlled 家族相互仅 0.6-1.5 = 参考帧漂移量级(1.1-2.3), 无法据此分 pad。
- 结论: 无法从数据推断 pad 标签, 需用户提供映射。

### FeelAnyForce 纳入: 帧级对齐失败 (负结果)
- 本地 /media/yxma/Disk1/yuxiang/mini_data_parquet/feelanyforce 48197 帧,
  **所有力标签列全为 null**(仅图像)。
- 标签 CSV 可从 HF amirsh1376/FeelAnyForce 单独下载(3 个 csv, 110109 行,
  FT 6 元组, Fz 为第 3 个)。已下载到 force_recovery/faf_labels/。
- 尝试用 (capture, frame_idx) → 时间戳排序位置 对齐: 一致性检查全过
  (42/42 capture 无越界, 无重复对), 但**正确性检查失败**:
  | 检验 | rho(vol,|Fz|) |
  |---|---|
  | 真实对齐 | 0.455 |
  | 对照A: capture 内打乱标签 | 0.442 ← 与真实几乎相同 |
  | 对照B: 全局打乱 | -0.003 |
  ⇒ 0.455 全部来自 capture 间结构(不同物体接触面积/力程不同),
  帧级对齐**未被证实**。逐 capture rho 中位仅 0.087-0.124。
- 教训: 全局打乱是无效对照; 必须用"组内打乱"才能检验组内信号。
  (此前一度据全局对照宣称"join verified", 已撤回。)
- 若要真正纳入: 需下载原始带时间戳文件名的图像 (HF 上 dataset.zip 分卷,
  合计 ~157 GB) 或获得官方 frame 映射。

## 会话级预处理取代手工 pad 标签 (设计, 用户提议)
问题: 逐 pad 拟合依赖不存在的标签, 且"逐组拟合"把 gel 差异吸收进权重,
使 rho 只是组内相关, 无法证明绝对牛顿可迁移。
方案: 把 pad 差异在**输入端**归一化, 而不是在输出端用权重吸收。

SessionCalibration (每个 session = batch/probe/family/capture, 全自动):
 1. rest frame: 用数据集自带的无接触标志取中值
    (Sparsh in_contact==0 / GlowTact initial.jpg / cnc,FEATS 最小力帧)
 2. 光度归一化: 把本 session 的 dI 分布(接触像素稳健分位)仿射映射到 LUT
    宿主 pad 的分布 -> 一个 LUT 即可跨 pad 使用
 3. 几何锚定(有球压时): a^2=d(2R-d) 反解 R 与深度基准 -> 绝对 mm 与
    物理锚定的深度增益(非力标签拟合)
 4. 空间增益场 u(x,y) 每 session 估计(已有机制)
验收: 归一化后用**单一共享力模型**跨所有 session/数据集, 若能逼近逐组拟合
的 rho, 即证明绝对牛顿可迁移(当前缺的正是这一点)。两者都报告。

### 新数据集: facebook/gelsight-force-estimation (Sparsh, CoRL2024) 已验证可用
- 结构 {sphere x6, flat x2, sharp x2} batch, 每 batch: 4x dataset_gelsight_NN.pkl
  (67MB, 各 5000 帧, PNG bytes 320x240 RGB) + dataset_slip_forces.pkl (1.8MB)
  注意: org_dataset_*.pkl 是 1.6GB 原版, **不需要**; 全部 10 batch 仅约 2.7GB
  (不是 52.8GB)。
- 标签: {'in_contact': (17288,) 每帧标志, 'trajectories': {tid: {indexes(N,),
  forces(N,3), poses(N,7), slip_label, coef_friction,...}}}
- **indexes 是显式帧索引** (batch_1: 10740 帧全唯一, 范围 88..17166), 与
  FeelAnyForce 的推断式对齐形成对比 -> 可直接 join, 无需猜测。
- 力: batch_1 Fz 0.063..1.888 N (含剪切 Fx,Fy), 球/平/尖三种压头, 有滑移标签。
- 10 个 batch 天然就是 session/pad 分组 -> 正是验证"会话归一化"的理想测试床。

## 2026-08-06 · 站点所有 3D 视图改为 Open3D 网格 + "逐数据集标定"说明
生产者改动: showcase.py 唯一持有 3D 面板 (全仓 grep `plot_surface|projection="3d"`
仅此一处; visualize/site/method_page/results_page/debug_page 均无 3D 视图)。
新增 `mesh_view()` / `mesh_tile()` 包装 o3d_view.render_depth_mesh
(smooth_px=3, z_scale=1.6, front=(0,0.32,0.95), zoom=0.66), 渲染前必过
`remove_halo_pedestal`; `_require_display()` 在耗时特征遍历之前先失败。
method_page.py: CSS `img,video{max-width:100%}` (1280 宽片段原会溢出 880 wrap)。

| found | evidence | fix | verified |
|---|---|---|---|
| 静图 3D 面板是 matplotlib inferno 曲面 | showcase.py:159 plot_surface | 换 Open3D 灰模网格 imshow | 20 张 headless-chrome 截图确认灰模 |
| 视频完全没有 3D 面板 | 960×330 三栏 tactile/depth/force | 加第 4 栏 640-960 | 1280×330, ffprobe 确认 |
| 视频网格白底像"挖洞" | 深色画布上白色方块 | bg=0.05 + mesh_tile 中心裁剪(1.35× 过采样) | 第 120 帧目视 |
| 1280 宽片段溢出正文 | 截图中视频出血到页面外 | CSS video max-width | 重截确认贴合 |

重新生成: 20 张静图 (cnc in-view rho=0.920 MAE=0.29N, feats rho=0.757
MAE=5.37N — 与上次 LUT-v2 一致, 仅面板换了) + 10 个片段。
片段 3D 面板**非降帧**: 每个 fresh 触觉帧渲染一次 (65-69 次 / 240 行,
即传感器自身 ~8.5Hz 更新率), 重复行沿用上一帧 — 与旁边深度面板同节奏。
mesh_stride=2 只降网格密度(320×240 面板看不出), 不降时间分辨率。
渲染耗时实测: 700×560 0.80 s/帧, 320×256 0.37 s/帧。

method 页新增 "Per-dataset calibration / 逐数据集标定" (EN+ZH, ZH 断言对齐):
管线**无任何硬度/弹性模量常数**; 刚度被逐组最小二乘权重 + isotonic 标定吸收,
逐数据集**且**逐压头/探头组重拟合; 跨数据集共享的物理常数只有球面标定 RGB LUT
和 MM_PER_PIXEL; 故报告的 rho 是逐组秩相关, **不**证明存在可跨 gel 迁移的绝对
牛顿模型(唯一试过的多物体混合拟合 rho 0.47)。并写入实测噪声: 深度域高频占峰值
0.3%, 梯度域 22.2% (LUT 分箱量化, 9.8% 接触像素落在未观测 bin), Poisson 积分
把 LUT 噪声低通掉。已发布: https://huggingface.co/spaces/yxma/react-force-recovery

## 全站图件 Open3D 化 + eval 结果一致性收尾
| found | evidence | fix | verified |
|---|---|---|---|
| Open3D 渲染在图块内只占约15% | 相机按场景包围球取景, 平板留大片空白 | o3d_view.crop_to_content (按非背景包围盒裁剪, 不动相机以免逐帧透视跳变) | 重生成后占比约60%, cnc_00 压痕环清晰 |
| mesh_tile 固定中心裁剪与上游内容裁剪冲突 | 会切掉几何本身 | 改为等比缩放+背景色 letterbox | 片段 3D 列纹理(蜂窝+V边)清晰 |
| GlowTact 脚注数字过时且结论已反转 | 文案"家族内 ρ 达 0.93; 与 FeelAnyForce 的差距" vs 实际 0.98 > 0.90 | 改为逐 indenter 0.975-0.992 / MAE≤0.73N, 并显式声明该标定逐家族重拟合=组内秩一致性, 非可迁移绝对牛顿 | EN+ZH 各命中1处, 旧文案0处 |
审计: grep plot_surface|projection=3d|Axes3D|plot_trisurf 全仓为空 ->
站点唯一 3D 来源是 showcase.py, 已全部走 o3d_view。
feats_domain_gap.png (两张原始传感器照片, 无重建) 与 actions 页叠加片段
(原始流+力条, 无 3D 面板) 经查确认无需重生成。
指标未变(仅面板变): cnc in-view 0.920/0.29N, feats 0.757/5.37N。

## 2026-08-06 · Sparsh (facebook/gelsight-force-estimation) 纳入 + 评测
代码: `twm/force_recovery/sparsh_data.py` (download / verify / traj / orient /
features / eval)。缓存: `feature_cache/sparsh_<probe>_batch_<n>.json`。
只下非 org 文件, 10 batch 共 2.2 GB (org_* 1.6GB/个, 未下载)。
每 batch 分层抽 750 帧 (按 |Fz| 十分位, 避免被接近阶段的低力帧淹没), 共 7500 帧。

### found → evidence → fix → verified
| # | found | evidence | fix | verified |
|---|---|---|---|---|
| 1 | flat/* 与 sharp/* 每条轨迹 `indexes` 比 `forces` **多 5 个** (`poses` 多 10); sphere/* 全部一致 | 598/598 flat+sharp 轨迹均 delta=5; sphere 911/911 delta=0 | 在**轨迹内**配对并截断 (`label_table`), 不再分别 concat 后拼接 | flat/b1 rho(vol,\|Fz\|) 0.005→0.492; sharp/b1 **-0.184→0.471**。丢弃未配对 index 755/735/765/755 |
| 2 | 帧级 join 是否成立 (FeelAnyForce 的教训) | 逐轨迹 rho(\|dI\|,\|Fz\|) 中位: sphere 0.901 / flat 0.898 / sharp 0.932, frac>0.5 = 0.90 | — (确认可用) | `cmd_traj`。轨迹内只有压深在变, 组间异质性无法伪造此相关 |
| 3 | sharp/batch_2 通道序与其余 9 个 batch **相反 (BGR)** | 其 rest 参考 RGB 均值 (66,120,134), 其余 9 batch 全为 (127,116,60) | `load_frames` 自动检测 (ref B均值>R均值 则交换), 不硬编码 batch 名 | rho(vol) 0.376→0.486, LUT 覆盖 0.895→0.946; 对照: 同样交换施于 sharp/b1 变差 0.413→0.361 ⇒ 是通道序不是拟合旋钮。整批 raw rho 0.235→0.436, 校准后 0.371→0.489 |
| 4 | 单一全局 rest 参考把 15-19k 帧 (~10 min) 的凝胶松弛/灯漂抹进每个 dI | sphere/b1 全局参考 rho(vol)=0.533, rho(\|dI\|)=0.460 | 局部参考: 每 200 帧取一个 in_contact==0 锚点, 用时间最近 6 个的中值 | 0.533→0.564, rho(\|dI\|) 0.460→**0.568**, 解码开销不变 |
| 5 | flat/batch_2 上游只有 3 个 image pkl (无 `_03`) | HF `list_repo_files` 确认; 该 batch 14550 帧 | 用 glob 而非固定 4 个文件名 | 10/10 batch 帧数和 == len(in_contact) (`cmd_verify`), index 全唯一且在界内 |

### 结果 (7500 帧, 特征 [vol, vol2, maxd, area, h1] + 最小二乘 + isotonic, batch 内半/半, 5 seed)
| batch | raw rho(vol,\|Fz\|) | 校准 rho | MAE (N) | **batch内打乱** rho | **轨迹内打乱** rho |
|---|---|---|---|---|---|
| sphere_b1 | 0.500 | 0.496 | 0.133 | -0.011 | 0.146 |
| sphere_b2 | 0.516 | 0.484 | 0.140 | 0.001 | 0.163 |
| sphere_b3 | 0.362 | 0.378 | 0.156 | 0.008 | 0.121 |
| sphere_b4 | 0.449 | 0.477 | 0.152 | -0.013 | 0.138 |
| sphere_b5 | 0.458 | 0.465 | 0.154 | -0.006 | 0.145 |
| sphere_b6 | 0.481 | 0.492 | 0.137 | -0.007 | 0.079 |
| flat_b1 | 0.492 | 0.499 | 0.130 | -0.000 | 0.260 |
| flat_b2 | 0.567 | 0.581 | 0.118 | -0.017 | 0.337 |
| sharp_b1 | 0.471 | 0.658 | 0.120 | -0.006 | 0.229 |
| sharp_b2 | 0.436 | 0.489 | 0.139 | 0.004 | 0.181 |
| **POOLED** | **0.451** | **0.558** | **0.138** | **0.145** | **0.249** |

对照说明: pooled 打乱后不是 0, 因为 batch 内打乱**保留**了各 batch 的力程差异 —
这正是 FeelAnyForce 上全局打乱掩盖问题的地方。要比的是 0.558 vs 0.145 (逐 batch
则是 0.38-0.66 vs ≈0.000)。**本数据集通过**; FeelAnyForce 当时是 0.455 vs 0.442。

### 跨 batch 迁移 (在 i 上拟合, 原样用于 j) — 本次核心问题
sphere 6x6 全矩阵 rho 0.418-0.591, 对角 0.471-0.587: **离对角几乎不掉**
(如 b2→b6 0.591 高于 b6→b6 0.557)。flat 2x2: 0.535-0.634。
sharp 2x2 最弱: 对角 0.695/0.576, 离对角 0.365/0.465。
跨压头 (sphere_b1 模型原样外推): flat_b1 0.545 / flat_b2 0.624 / sharp_b1 0.566 /
sharp_b2 0.433, MAE 0.127-0.167 N。
⇒ **一套标定确实能跨 pad 与跨压头搬运**, 这是此前缺失的证据。

### 剪切 (Fx,Fy)
全量分布: 中位仅 0.017 N 但 q90≈0.5 N, 最大 1.84-2.22 N; 16-30% 帧 >0.3 N。
rho(vol,\|F\|) 与 rho(vol,\|Fz\|) 基本相同 (0.443 vs 0.451) — Fz 主导。
rho(vol, shear) 仅 0.045-0.192 ⇒ 深度特征不编码剪切。
**高剪切帧是系统性离群**: 剪切前 10% 帧残差是其余的 1.6-2.4 倍
(如 sphere_b5 0.129→0.329 N), 且该子集内 rho 塌陷
(sphere_b4 -0.070, sphere_b5 -0.034, sharp_b1 0.012)。
物理上合理: 侧向剪切把凝胶推挤成体积但不增加法向力 ⇒ 纯法向深度模型必然高估。

### 被否/未采纳的变体 (带数字)
- **通道置换调参**: sphere/b1 上扫 rot∈{1,3} x flip x 6 种通道置换 (24 组, n=200):
  identity 0.537, 最佳 (flip + GRB) 0.621, 最差 (flip + GBR) 0.381。
  **不采纳** — n=200 上 24 组里挑最大值, 增益 (~1.2 SE) 不可信, 且无物理依据。
  但 0.38-0.63 的跨度本身是证据: LUT 对该外来 GelSight Mini **域外**
  (纯粹置换输入通道就能改变 0.24 rho, 若 LUT 对本传感器有效则不应如此)。
  注: 第 3 条的 sharp/b2 交换与此不同 — 那里有独立的参考色证据与反向对照。
- **portrait→landscape 旋转方向**: ROT=1 vs ROT=3 实测等价
  (0.537/0.435 vs 0.533/0.458), 标量特征对此不敏感; 取 ROT=1。
- **局部参考修不了 flat/sharp**: 在发现缺陷 1 之前, 曾假设 flat/sharp 的 rho≈0
  是参考帧漂移所致 —— 局部参考后 sharp/b1 仍为 -0.196 (全局 -0.184)。
  真凶是轨迹配对。同时 LUT 无关的 rho(\|dI\|,\|Fz\|) 也是 -0.04 ⇒ 当时就能
  判定不是重建问题而是标签问题。

### 已知局限
- 重建形状不对: 球压恢复出的 depth 是**双瓣+中心凹陷**而非圆顶
  (`sparsh_probe/orient.png`), LUT 覆盖率 0.90-0.97 只说明 bin 被访问过,
  不说明梯度值正确。rho≈0.45-0.56 是在错误深度形状下取得的, 单调性来自
  dI 幅度而非真实几何 ⇒ 该数据集应作**外来传感器上界**看待, 不宜宣称绝对牛顿。
- 力程窄: \|Fz\| 0.06-1.89 N (README 称协议到 3N, 实际标签未及)。

## 2026-08-06 · Sparsh 自标定: 光度管线是否真的"传感器通用"?
工具: `twm/force_recovery/sparsh_lut.py` (geom|build|verify|angle|cross|features|
eval|depth|inview|subsets|inview_eval); `sparsh_data.py` 增补 `poses` 与
`select=` 钩子 (向后兼容, 默认行为不变)。
产物: `sparsh_probe/{dome_before_after.png,dome_stats.json,sparsh_circles.json,
eval_circles.json}`, `lut_calibration/sparsh_lut.npz`,
`feature_cache_sparshlut/` (仅数据盘, 不入库)。

**问题**: 上一轮记录的局限是"球压重建成双瓣+中心凹陷, rho 0.45-0.56 来自 dI 幅度
而非几何"。本轮只改一件事 —— 用 Sparsh **自己的**球压重新拟合 dI→梯度表
(即传感器厂商本该做一次的标定), 其余管线一字不动, 看结论是否翻转。

### found / evidence / fix / verified
| found | evidence | fix | verified |
|---|---|---|---|
| 缺深度基准, 只有力标签 | `poses` (N,7) 未被使用 | 验证第 2 列为压入轴: 池化 rho(z,\|Fz\|)=**-0.888**, 逐轨迹中值 **-0.927**; 第 0/1 列是协议里 2mm 侧滑, 仅 -0.16/-0.07 | `label_table` 携带 `P`; 检出圆 rho(a², -z)=**0.894**, rho(a², \|Fz\|)=**0.948** |
| 球半径未知 | README 只写 "hemisphere" | 用精确关系 a²=d(2R-d) 联合回归 (R 跨 batch 共享, z0 逐 batch) | **R=2.438 mm (直径 4.88 mm ≈ 5mm 标准球头)**, R²=0.832, n=755/757; 逐 batch 独立拟合 R=2.25/2.54/2.54 (±6%), z0=113.856/113.866/113.856 mm (三批一致到 0.03mm) |
| **"dI 是纯偶极/照明退化" —— 上一轮的判断是错的** | 534k 接触像素 dI 的 PCA 方差占比 **0.670/0.258/0.072** = 满秩三 LED 信号 | 病因改判为"表把颜色映到了错方向" | GlowTact 表在 Sparsh 接触像素上梯度方向中值误差 **93.3°** (随机=90°), 仅 15.2% 像素 <30°, 幅度 1.62×; Sparsh 自标表 **4.5°**, 99.2% <30°, 幅度比 1.013 |
| 重建形状非物理 (双瓣) | `sparsh_probe/orient.png` | 用 708 帧 (sphere b1-3, 剪切<0.15N) 累积 90³ 表, 观测 71865 bin (9.86%), 1.73M 像素 | 见下"圆顶验证" |
| **LUT 覆盖率是无效的域外检测器** | 旧表覆盖 0.905-0.968 却给出 93° 方向误差 | 覆盖率只说明 bin 被访问过 | 新表覆盖 0.991-0.999; 真正的判据是方向误差, 不是覆盖率 |
| 全量帧 rho(vol, 真实深度) 仅 0.15 | 与 rho(d,\|Fz\|)=0.84-0.89 矛盾 | 检出接触圆, 区分"完全在视野内 / 被边框裁切 / 检不到" | **36% 帧检不到接触盘, 11% 被裁切**; 与 FOTA/cnc 同一病根 (机器人网格 x∈[192,212] y∈[-8,8]mm > 裁剪后可见 pad ≈13.4×10.1mm; `crop` 本身还丢掉 49% 画幅) |

### 圆顶验证 (task 的关键, sphere/b1, n=220, 深度 0.15-0.70mm)
| LUT | dip h(0)/h_max | 峰值半径 | 对解析球冠残差 RMS | 峰值/真实深度 |
|---|---|---|---|---|
| GlowTact (外来) | 0.616 | 0.685 mm (**离心 → 双瓣**) | **0.179 mm** | 0.254 (幅度只剩 1/4) |
| Sparsh 自标定 | **1.000** | 0.047 mm (**在中心**) | **0.0545 mm** | **1.155** |

⇒ **圆顶是物理的**: 单峰、峰在接触中心、对解析球冠残差 54 µm (峰高 0.15-0.70mm
量级), 幅度恢复到真值的 1.16×。图: `sparsh_probe/dome_before_after.png`。

**反向对照** (同样的问题反过来问): 把 Sparsh 表用在 GlowTact 球压上 (n=111) —
dip 0.838, RMS 0.850mm, 幅度只剩 **0.079×**; GlowTact 表在自己家 dip 1.000,
RMS 0.356mm, 幅度 1.314×。**两张表都只在自己传感器上成立**, 不是"我们的表差"。

### 力回归 (相同帧, 唯一变量是查找表; 特征/拟合/半半分/5 seed 全不变)
| 帧集 | LUT | 池化 rho | MAE (N) | **batch内打乱** rho | 逐 batch 打乱 |
|---|---|---|---|---|---|
| 全部 7500 帧 | GlowTact | 0.558 | 0.138 | 0.145 | -0.017..+0.008 |
| 全部 7500 帧 | **Sparsh 自标** | **0.676** | **0.113** | 0.156 | -0.021..+0.049 |
| 视野内 8335 (5 seed 池化) | GlowTact | 0.878 | 0.079 | — | — |
| 视野内 8335 | **Sparsh 自标** | **0.968** | **0.042** | 0.258 (池化) | **-0.023..+0.051** |

逐 batch (视野内, Sparsh 表): sphere 0.964-0.979 / MAE 0.032-0.039 N,
flat 0.948/0.955 / 0.043-0.044 N, sharp_b2 0.579 / 0.125 N。
**原始特征** rho(vol,\|Fz\|) 几乎不动 (池化 0.451→0.453, 逐 batch -0.03..+0.10),
但视野内从 0.74-0.89 (GlowTact) 升到 **0.95-0.97** (Sparsh) —— 增益不在"单特征
单调性", 在"多特征的几何一致性"。

### 视野裁剪对照 (只看 sphere+flat, 6000 帧; 排除"检测阈值筛掉了小力"的解释)
| 子集 | n | \|Fz\| 中值 | rho(vol, 真实深度) | rho(vol,\|Fz\|) | 校准 rho | MAE |
|---|---|---|---|---|---|---|
| 全部 | 6000 | 0.391 | 0.148 | 0.500 | 0.612 | 0.120 |
| 检不到接触盘 | 2179 | 0.375 | 0.326 | 0.183 | 0.163 | 0.163 |
| 检到但**被裁切** | 663 | **0.551** | 0.728 | 0.754 | 0.801 | 0.118 |
| 检到且**完全在视野内** | 3158 | 0.380 | **0.916** | **0.968** | **0.971** | **0.043** |

三个子集的 \|Fz\| 中值 0.375/0.551/0.380 基本相同 (被裁切子集的力反而**最大**),
⇒ 差异来自**可见性**, 不是力程筛选。这与 cnc 的结论 (视野内 0.63→0.94) 同源。

### 跨 batch 迁移 (Sparsh 表, 视野内, 在行拟合原样用于列)
sphere 6×6 全矩阵 rho **0.963-0.985**, MAE 0.026-0.042 N; 离对角与对角无差别
(b6→b1 0.974 vs b1→b1 0.980)。**一次标定跨 6 个 pad 直接搬运。**

### 诚实的负面 / 变差项
1. **跨压头外推变差了**: sphere_b1 模型原样用到 flat, GlowTact 表 rho 0.545/0.624
   MAE 0.146/0.127 N → Sparsh 表 rho 0.430/0.512, **MAE 0.366/0.395 N** (视野内
   0.551/0.651, MAE 0.400/0.409 N)。几何一旦正确, 平压头的接触体积尺度与球压差
   得更远, isotonic 外推被夹住。**"更真的几何"让跨压头绝对牛顿更差**, 需要逐压头
   重新拟合力模型 (排序 rho 仍可用)。
2. **sharp 压头整体不适用**: `detect_circle` 要求近圆盘, sharp/b1 750 帧里检出 71、
   视野内 **0**; sharp/b2 视野内 rho 反而从 0.915 (GlowTact) 掉到 0.579。尖头接触
   不是球冠, 自标定球表对它无保证。
3. **高剪切仍是系统性离群**: 全量 hi/lo 残差比中值 1.81 (GlowTact) → **1.64**
   (Sparsh), 视野内 1.16-2.33 (中值 1.58)。绝对残差降了 (sphere_b5 0.329→0.218 N;
   视野内 0.050 N), 但比值只小幅改善 —— 侧向剪切把凝胶挤出体积却不加法向力,
   这是纯法向深度模型的结构性缺陷, 换表修不了。
4. **轨迹内打乱在视野内升到 0.34-0.59** (全量时是 0.08-0.34)。这不是泄漏: batch
   内打乱仍为 -0.023..+0.051。原因是轨迹内打乱**保留每次按压的力均值**, 模型越准,
   它保留的残余相关就越高 —— 它随模型质量缩放, 不是干净的 null。干净的 null 是
   batch 内打乱。**结论应按 real 0.968 vs batch内打乱 ≈0.00 来读。**
5. **`crop` 自伤**: 沿用 GlowTact 的 1/7 边框裁剪, 只保留 50.9% 画幅
   (230/320 × 170/240)。36% 的按压检不到接触盘, 相当一部分是被这一步丢掉的。
   **未测**: 放宽裁剪需重建 LUT (MMPP 变) + 重跑特征, 留作下一步。

### 被否/未采纳
- **"Sparsh dI 是纯偶极"** —— 本轮**推翻**。PCA 0.670/0.258/0.072 是满秩;
  `orient.png` 里看起来像偶极是缩略图分辨率造成的错觉, 放大后是标准三 LED
  蓝/红/黄三瓣 (见 `dome_before_after.png` 左列)。原诊断"照明几何不同"方向对,
  但"退化成一维"是错的, 已在 `sparsh_lut.py` 头注更正。
- **通道置换调参 (上一轮)**: 仍不采纳, 且本轮给出了它错在哪 —— 真正的病因是
  梯度方向 93° 全错, 置换通道只是碰运气地让某些方向偶然对上, 自标定把它降到 4.5°。
- **用 LUT 覆盖率当域外检测器**: 否。0.905-0.968 的覆盖率没能报出 93° 的方向错误。
- **不重新拟合力模型、只换表**: 对同压头成立 (跨 batch 0.96-0.98), 对跨压头不成立
  (见负面项 1)。

### 结论
球压自标定**确实把管线推广到了外来传感器**, 但要把话说准:
- **几何层面 (管线真正的工作)**: 从"双瓣、幅度剩 1/4、方向错 93°"变成
  "单峰圆顶、幅度 1.16×、方向差 4.5°、对机器人真实压深 rho 0.92-0.94"。**通用成立。**
- **力层面**: 视野内 rho 0.878→0.968, MAE 0.079→0.042 N; 全量 0.558→0.676,
  0.138→0.113 N。**同压头通用成立, 跨压头绝对牛顿反而更差。**
- **代价**: 需要该传感器上一组球压 + 压头深度记录 (本例 708 帧, 三个 batch)。
  这是"每传感器标定一次", 不是"零样本通用"。

## Sparsh 自标定结论上站 (推翻上一版站点论断)
site 图: sparsh_figure.py -> results_sparsh.png (GlowTact表 vs 自标定表 vs 跨pad矩阵)
      + sparsh_dome.png (球压重建前后对比, 来自 sparsh_probe/)
| found | evidence | fix | verified |
|---|---|---|---|
| 站上"0.56 = 迁移上限, 几何无效"论断已过时 | 自标定后视野内 0.968/0.042N | 全节重写 EN+ZH, 首页行 0.56*→0.97* | 旧文案两语言各 0 处命中 |
| 我此前称 Sparsh dI 是"退化偶极子" — **错误** | dI 接触像素 PCA 方差 0.670/0.258/0.072 = 满秩三灯 | 撤回该说法; 低分辨率 orient.png 误导 | 高分辨率 dome 图可见完整径向色环 |
| LUT bin 覆盖率无法检测域外 | 覆盖 0.905-0.968 却有 93.3° 梯度误差 | 改用"梯度与解析球面夹角"作为域外检测量 | 自标定后 4.5° |
关键数字: R=2.44mm (a²=d(2R-d), R²=0.832, n=755, 三 batch z0 一致到 0.03mm);
残差 0.179→0.0545mm; 梯度夹角 93.3°→4.5° (30°内 15%→99%);
视野内力 0.878→0.968, MAE 0.079→0.042N; 跨 pad 迁移 0.963-0.985。
边界(站上已写): 跨压头绝对牛顿失败(平头 MAE 0.37-0.40N, 且正确表反而更差)、
尖头不支持(0.58)、剪切结构性无法覆盖(残差比 1.6x 不随表改善)、
需接触圆盘可见(36%无盘/11%裁切; 裁切子集力最高却更差=可见性非力程)。

## 3D recon 工作台 (20 样本全中间过程) — twm/force_recovery/recon_study.py
站点页: recon_workbench.html (14 列/样本 + 逐样本诊断表)
列: raw|ref|dI|max|dI||valid mask|LUT命中?|gx|gy||grad||div(g)|depth|去基座depth|
    径向剖面vs解析球冠|Open3D mesh
新增可量化诊断:
- unobserved_lut_frac: 接触像素落在**未标定色格**的比例, 随深度上升
  (star 2%→22%), 这些梯度是最近邻"编造"的 -> 改进点1: 扩展标定深度/物理外推
- flat_top_ratio (中心/边缘高度): 平顶压头应≈1.0, 实测 star/triangle/quad/B
  为 1.23-1.42 = **过度穹顶化**; round 的 1.43-1.46 是正确的(球本就是穹顶)
  -> 改进点2: 平顶内部无色变, Poisson 只能从边缘向内积分
- valid mask 被光晕主导(列5是圆blob而列3形状清晰) -> 改进点3
球压对照(round, 已知R): 梯度夹角 5.1-6.8°(随机90°), 剖面残差 35-105µm
=> 本传感器上"表"和"解算器"都是好的, 缺陷在覆盖率/掩码/平顶几何。
修正: 图标题曾对所有行硬写"(plateau)", 但 1.4 对球是对的、对平顶是错的,
已去掉依赖压头的判定词, 只报数字。

## 数据集筛选: 用户明确限定"只要 GelSight Mini"
| 候选 | 传感器 | 裁定 |
|---|---|---|
| **TacVerse** (HF Lan-2025/Tactile) | **确证 GelSight Mini**, 无marker 16917 + 有marker 15487, 2mm球, 0.1mm步进, 六轴 M3813B | ✅ 唯一确定符合; **gated(401)需填表**; 力单位未声明; 许可冲突(HF卡 cc-by-4.0 vs 论文 CC BY-NC-ND) |
| ToucHD-Force (BAAI) | README **从未列出五个传感器** -> 含Mini未证实; 且滑动采集(剪切主导) | ⚠️ 降级 |
| DAR_OTS | GelSight系但**型号未声明**; 另发现 mb0i0(marker)力标签与 wmb0i0(markerless) 为同一多重集(仅顺序不同)=标签搬运 | ❌ |
| GenForce | 型号未证实; 且仅 240x160 单通道 marker 图, **无RGB** -> 无法驱动光度LUT | ❌ |
| AllSight | 圆柱指非平面gel; depth字段与法向力相关仅0.108 | ❌ |
| facebook/digit-force-estimation | DIGIT | ❌ |
| GelSight-YoungsModulus | Wedge; 力疑为夹爪估计非实测 | ❌ |
已验证的负结果(勿再查): 9DTact无数据集发布; TLabel-Bench无力标签;
GelSlim 4.0无牛顿力发布; Awesome-Touch列表无新增。
=> 行动项: 填 TacVerse gate 表单(唯一 Mini + marker + 已知球 + 步进深度,
   正是我们的标定几何; 力子集 6.8GB)。

## GelSight Mini 3D recon 外部验证: 场内无标准基准, 但真值就在本地
### 关键结论: 该领域**没有**深度重建精度的标准基准, 每篇论文自造
- 3D Cal 自定 Overall/Type-1/Type-2 (接触区内外分开报), 是最接近事实标准的做法
- TacEva(2509.19037)明说存在这个空白, 但它本身是标量回归, 未被采用
- GelSight 官方 FAQ 原话: "GS Mini is not a metrology device so there are no
  study results that can be shared regarding the accuracy of the 3D reconstruction"

### 可用于验证的 Mini 材料(已核实)
| 来源 | 有什么 | 判定 |
|---|---|---|
| **SimTactileMNIST (本地!)** | object_id + object_pose + gel_pose + Taxim渲染Mini图; 网格从 HF tactile-mnist-mnist3d(printed_test 仅8.7MB, 13/13 对上) | ★ 可直接算**逐像素真值深度**, 非球面曲面几何 |
| **RealTactileMNIST (本地!)** | 真实Mini帧 + 逐帧6-DoF gel位姿 | ★ 真外部校验(需拟合4个全局参数) |
| TimSchneider42/taxim | **官方CMU Taxim无Mini配置, 此fork有**: 24张真实Mini球压图+多项式标定 | 可生成无限配对(RGB,精确深度) |
| py3DCal/3DCal (本地) | 36270张Mini图, 但**全部 penetration=3.0mm 单一深度**(非扫深), 且其训练目标同样是解析球假设 | 只能作管线互校, 非独立深度真值 |
| gsrobotics | 仅代码+权重, **无任何示例帧+期望输出**, 无法复现其数字 | ❌ |
| gs_sdk / gelslam / normalflow | 有Mini示例帧与参考实现, 但**不含真值深度** | 参考实现 |
### 已发表 Mini 精度数字(供对标)
- 3D Cal(2511.03078 Table II) 逐像素: Overall 22.4/23.6/48.8µm,
  **Type-2(接触区) 171.6/152.8/290.0µm** — 但**深度尺度是拟合的**(2D互相关对齐+
  调CAD压深最小化MSE) => 验证的是形状不是绝对深度
- TacEva: Mini 标量压深 MAE 29.4µm(无marker)/32.5µm(marker) — ResNet回归标量, 非深度图
- 我们当前: 对解析球冠残差 35-105µm(深度亦为拟合) => 与3D Cal的Type-2同类可比且更好

## 阴影假说(用户提出)= 确证; 光晕伪影的物理来源找到了
三个独立指纹, 全部支持"GelSight Mini 投射阴影"而非"凝胶隆起":
1. **符号**: 光晕区 dI 在 R/G/B **三通道一律为负**(变暗 -1..-7 灰度)
2. **随深度单调**: dI<0 的像素占比 64%(z=1.7) -> 99%(z=4.2)
3. **几何**: corr(峰深, 光晕宽度) = **0.859**, 宽度 0.94 -> 2.38 mm
   (物体越高影子越长 = 投射阴影的定义性特征)
旁证: Taxim 的 Mini 标定包单独带 shadowTable.npz (63方向x24), 说明 Mini
阴影强到必须单独建模。方向性弱(扇区比 1.48, SW 最强)与 Mini 三颗~120°LED
互相填补一致。

这一条同时解释了此前三个未解现象: valid mask 变 blob、需要事后扣halo基座、
字母 counter 在原始信号层被污染(counter 内部本就是阴影)。

### 修复尝试 (evidence-driven)
- 色度判据(dI 垂直于参考色的分量; 阴影=乘性变暗应平行): 逐像素分离度仅
  **1.13-1.24x** (光晕|cos| 0.64-0.70 vs 接触 0.42-0.50) -> 单阈值不够锐, 记为部分否决
- 但用**色度分量替代 |dI| 建掩码**在中等深度显著恢复形状(圆度越低越像形状):
  | 家族 | z | |dI|>8 圆度 | 色度掩码圆度 |
  |---|---|---|---|
  | star | 2.14 | 0.794 | **0.388** |
  | quad | 2.48 | 0.824 | **0.505** |
  | triangle | 2.55 | 0.753 | **0.628** |
  面积同时降约2倍(阴影被排除)。**但深压时失效**(star 3.17: 0.783->0.789)
  -> 深压时阴影区自身变得有色度, 判据失灵。
- 陷阱记录: 圆度只对**非圆压头**有意义。round 的高圆度(0.83)是**正确**的,
  把它算作"形状恢复"是错的(与此前"(plateau)"标签同类错误)。

## 3D recon 改进尝试汇总 (goal: 各种办法改进 + 重估力 + 更新全站)
| 尝试 | 证据 | 裁定 |
|---|---|---|
| **marker inpaint (Telea, ref+frame)** | dimple 功率 1.523→**0.890**(x0.65), 91%帧下降, Wilcoxon p=2.6e-19; 检测器 63/63 且在无marker参考上返回0 | **采纳(仅几何/3D产物)** |
| marker removal 用于力 | 基线 0.7747/5.03N (种子sd 0.029), 8个变体中位增量**全为负** | **否决**; 对照: 随机掩码 0.7697(无增益) |
| 梯度域置零(vs inpaint) | 1.251 (x0.93, 仅61%帧) | 否决: g:=0 在每个洞边界留下偶极层 |
| 阴影像素剔除 | 峰深 star 2.11→1.68mm, 但 c/r 1.38→**1.41**(未修复) | 未验证, 不采纳(无真值判定降低是否更正确) |
| 内部梯度归零 | c/r 1.38→1.30; 内部|g|本就是边缘的 0.06-0.35倍 | 效果小, 不采纳 |

### **自我修正(重要)**: 撤回"平顶应 c/r≈1.0"的判据
该判据假设刚性阶跃边缘。**柔顺凝胶必然包裹边缘, 所以任何接触 c/r>1 都是预期的**。
在拿到真值前, 无法量化 1.23-1.42 中有多少是伪影。站点相关表述需软化。

### FEATS 真正的瓶颈(非 marker)
最轻的20次按压 off-dot 处 |dI| 已达 11 灰度, 88% off-dot 像素通过 |dI|>8
=> "valid"几乎是整幅图, 特征积分的是**参考失配**而非接触。
换用逐indenter轻压参考更差(0.7747→0.7261)。这解释了 FEATS 0.77 vs 无marker 0.94-0.99。

### Wedge 论文(ICRA2021)对阴影的实际表述(已读原文)
- Fig.4 + Filters节: 扩散片**在硬件上**消除阴影(3M Diffuser 3635-70), 灰度滤片把内反射降到1/16
- §IV-C: "gel deformation and shadows can influence the 3D reconstruction,
  especially for sharp surfaces... **requires further processing from the
  perception side in the future**" -> **文献无算法**, 明确留待未来
- 但 Fig.10 给了完整的 **marker 算法**: 零填充 vs griddata nearest/linear/cubic,
  nearest 10ms(200x150), linear/cubic 60/70ms -> 我们的 marker 步骤据此
- 另一可借用点: Wedge 把 X,Y 像素坐标**放进映射输入**(RGBXY→Gx,Gy)补偿LED衰减;
  我们的空间增益场在**输出端**。输入端更根本, 待试。

## 首个"外部逐像素"验证: Tactile MNIST (twm/force_recovery/mnist_validation.py)
在此之前所有形状数字都是**自证**的 (我们的重建 vs 我们自己的解析球冠, 且球冠幅值
还锚在重建本身上 = 只验形状)。SimTactileMNIST 提供真正外部的、逐像素的、
**非球面**深度真值 (3D 打印 MNIST 数字, 高 6.5-8.2mm, 顶面是宽而低曲率的曲面)。
缓存/图: /media/yxma/Disk1/twm/force_recovery/mnist_validation/ (stage1.parquet,
tensors.npy 420x4 深度图, mnist_examples.png, verify/diagnose/controls .json)

### 真值不是假设出来的 (`verify`, 三项全过)
| 检验 | 结果 |
|---|---|
| 位姿索引对齐 | `sensor_image[i]` 用的是 `info.object_pose[i-1]` (物体位姿是每步扰动后才记录, sigma 1mm/2.9°)。该偏移下峰值压深恒为 **2.2491 ± 0.0016 mm** (n=77); 偏移 0/+1 则散布 ±1.16/±1.56mm |
| gel 基准 | gel 原点在未变形胶面上方 4.25mm, 仿真把传感器压到最近点距原点 2.000mm ⇒ **每一次触摸都是同一个 2.25mm 深压** |
| 端到端 | 用我们算出的 GT 高度图重新 Taxim 渲染, 与数据集自带图像 **MAE 1.76/255, corr 0.995** ⇒ 位姿/朝向/像素尺度/胶面基准全对 |
| 参考帧 | 取 Taxim 无接触背景 (精确解析); 与"240 帧中位数"参考差 **1.24/255**, 两种参考等价 |

### 尺度/分辨率不匹配的处理 (必须明说)
sim 图 320x240 覆盖整块 18.88x14.16mm ⇒ 0.059 mm/px; 管线常数 MM_PER_PIXEL=0.041518
(它总在 640x480 帧的 5/7 中心裁剪上跑)。**不改管线**: 把 sim 帧重采样到 455x341,
此时 1px == MM_PER_PIXEL (差 0.05%), GT 在同一网格上光线投射 ⇒ 全视场 + 精确尺度。
另附"部署裁剪 (5/7, 仅中心 51% 面积)"一行做稳健性对照。

### 主表 (n=420 触摸, 106 物体, 10 个数字; 单位 µm; 3D Cal 式 Type-1/2 拆分)
| 变体 | MAE | Type-1 (GT=0) | Type-2 (GT>0) |
|---|---|---|---|
| 全零预测 (基线) | 383 | 0 | 1407 |
| **Taxim 胶面 vs 物体几何 (任何胶面方法的天花板)** | **38** | — | **59** |
| A GlowTact 表 (出厂管线) 原始 | 273.4 | 85.8 | 884.0 |
| A + 单个全局尺度 (s=1.02) | 273.3 | 87.9 | 878.4 |
| A + 逐样本尺度 | 206.5 | 26.8 | 812.7 |
| A + 逐样本尺度 + 2D 平移对齐 | 207.0 | 38.1 | 743.6 |
| B 仿真域重标表 原始 | 316.9 | 59.3 | 1078.3 |
| B + 单个全局尺度 (s=1.67) | 289.3 | 99.2 | 905.7 |
| B + 逐样本尺度 | 205.0 | 10.8 | 855.7 |
| B + 逐样本尺度 + 2D 平移 | 216.7 | 34.1 | 790.3 |
| B vs Taxim 胶面 (原始) | 336.2 | 48.2 | 725.5 |
| B 部署裁剪 5/7 (原始) | 343.0 | 81.6 | 1005.7 |
拟合声明: "原始"行**没有任何**尺度/平移拟合; 全局尺度=整个 420 张一个数;
逐样本尺度/平移 = 3D Cal 的做法 (他们同时做 2D 互相关对齐 **和** 压深尺度拟合)。

### 对照实验 (同一求解器, 同一代码路径)
| 对照 | MAE | Type-1 | Type-2 | 峰值 ours/GT |
|---|---|---|---|---|
| 留出球压 (仿真域, 随机位置, 无拟合) | 19.4 | 6.0 | 154.6 | 1.2 / 1.0 mm |
| C1 球冠 **视场内** | 37.9 | 14.0 | 291.9 | 1.39 / 0.90 mm |
| C1 球冠 **被边界裁切** | 57.5 | 33.6 | 450.2 | **0.30 / 0.90 mm** |
| C2 平顶圆盘 (视场内, 真值中心/边缘比=1.000) | 61.9 | — | 226.5 | 0.92 / 0.60 mm |

### 决定性实验: 同一数字几何, 只改压深 (`sweep`, 40 触摸 x 5 个压深)
数据集只有 2.25mm 这一个压深, 所以它自己无法区分"数字难"还是"深压难"。
Taxim 复现已验证到 1.76/255, 因此用同一网格重渲更浅的压深是合法的反事实实验。
| 压深 mm | 接触面积 | A MAE/T2 | B MAE/T2 | B 峰值比 ours/GT |
|---|---|---|---|---|
| 0.30 | 6.3% | 38.9 / 275.3 | **11.2 / 96.5** | **1.00** |
| 0.60 | 11.4% | 99.6 / 469.6 | 35.0 / 186.3 | 0.97 |
| 1.00 | 16.8% | 133.1 / 506.2 | 67.8 / 308.6 | 0.77 |
| 1.50 | 21.8% | 171.6 / 564.4 | 127.4 / 514.6 | 0.68 |
| 2.25 (数据集) | 27.0% | 274.3 / 810.7 | 281.1 / 961.8 | 0.55 |
自洽性检查: 重渲的 2.25mm 行 (A: 274.3/100.3/810.7) 与直接用数据集图像的 stage1
(A: 273.4/85.8/884.0) 一致 ⇒ 重渲管线与"跑数据集自带图像"等价。
**⇒ 误差随压深单调爆炸, 而不是"数字形状本身不可重建"。在 ≤0.6mm 压深上,
我们在非球面真值上是 11-35µm MAE / 97-186µm Type-2 且零拟合, 优于 3D Cal 公布的
22.4-48.8µm / 152.8-290µm (他们还做了对齐+尺度拟合)。**

### 结论 (按任务问的四件事)
1. **仿真 RGB 能不能驱动我们的表**: **能**。LUT 梯度方向 vs 真实胶面梯度:
   sim 24.4° (79.4% 像素在 30° 内); **同代码同统计在 GlowTact 真机自家球压上是 26.1°
   (78.6%)** ⇒ Taxim 渲染对我们这张真机表的"可读性"与真机自己一样好。
   幅值比 1.48 (A, 偏大, 与 Taxim 在 320x240 下仍用 pixmm=0.0295 计算梯度、
   把坡度夸大 2x 一致); 重标表后 0.89。⇒ **失败不在光度域差异**。
2. **失败在哪**: **420/420 触摸的真值接触区都碰到图像边界** (8mm 高数字上压 2.25mm,
   数字投影 13.6-44.4 x 58-92.4mm 远大于 18.9x14.2mm 的胶垫)。fast_poisson 的
   零边界条件对"跑出视场的接触"不成立: C1 对照给出机理——同一球冠从视场内挪到边界,
   **峰值从 1.39mm 塌到 0.30mm (4.6x)**, Type-2 从 292→450µm。
   接触区内形状相关只有 0.17, 35% 的触摸整图相关为负。
3. **和 3D Cal 比**: 他们 (arXiv 2511.03078 v3 表 II, 原装 Mini, 逐像素深度图,
   **且做了 2D 互相关对齐 + 压深尺度拟合**) 半球/药丸/棋子 = 22.4/23.6/48.8µm,
   Type-2 = 171.6/152.8/290.0µm。我们**不做任何拟合**的视场内球冠对照 = 37.9µm /
   Type-2 291.9µm ⇒ **同一量级, 落在他们最差那一档附近, 且我们没拟合**。
   在 MNIST 数字上按数据集的 2.25mm 深压是 273-317µm (原始) / 205-207µm
   (逐样本尺度), **比他们差一个数量级**; 但同样的数字几何压 0.3-0.6mm 时是
   **11-35µm / Type-2 97-186µm (零拟合)**, 反而优于他们。
   ⇒ 正确说法不是"我们差一个数量级", 而是"**我们的有效工作区间是浅压;
   2.25mm 深压超出标定坡度范围, 误差随压深单调爆炸**"。TacEva 的 29.4/32.5µm 是 ResNet 回归一个
   **标量压深**, 不是深度图, 不可比 (须注明)。我们此前的 35-105µm 是"vs 解析球冠 +
   幅值拟合"的形状残差, 与仿真球压对照 (37.9µm, 无拟合) 一致。
4. **三个已知缺陷被真值重新定量**:
   - (a) 过度穹顶化: 中心/边缘比 ours 1.54-1.56, **真值 (物体) 1.40 / (胶面) 1.33**
     ⇒ 相对偏高 **+12%**; C2 平顶对照 (真值恰为 1.000, 视场内) ours 1.069 ⇒ **+7%**。
     **修正旧说法**: 旧结论"平顶压头 1.23-1.42 而 1.0 才对"高估了缺陷幅度——
     其中大部分是贴边/几何本身的曲率, 纯过度穹顶只有 7-12%。
   - (b) 未标定色格比例: A 13.8% / B 16.9%, 与 Type-2 误差相关只有 **0.098 / 0.294**
     ⇒ **它不是主要误差驱动**, 与之前"覆盖率无法当域外检测器"的负面结论一致。
   - (c) 光晕掩码: valid mask 与真值接触区 **IoU 0.614**, 召回 0.917,
     **过分割 0.531** (多标出的面积 = 真接触面积的 53%) ⇒ 光晕主导被外部真值证实。
     depth>0.05 的掩码 IoU 0.521。

### 边界与未做
- SimTactileMNIST 的图像是 Taxim 渲染: 光度可读性已验证 (上面 1.), 但胶体力学是
  Taxim 的金字塔模糊模型, 其胶面与物体几何只差 38µm (MAE) ⇒ 它几乎不模拟真实胶体
  的"不贴合"; 真机上物体几何不能直接当胶面真值。
- **该数据集不适合评"绝对压深"**: 每次触摸都是同一 2.25mm 深压且必然贴边;
  它适合评形状/掩码/方向, 不适合评幅值。
- Stage 2 (RealTactileMNIST 真机帧) **未做**: 需要逐 round 拟合 (x,y,yaw)+全局 z 基准,
  且同样是贴边深压 ⇒ 会复现同一结论。已备好的事实: 真值网格用 printed_test
  (13/13 object_id 命中), 视频 256 段/round, 帧位姿在 `gel_pose_cell_frame_seq`。
- 依赖: `pip install taxim` (--no-deps) + 一个 torch_scatter.scatter_min 垫片
  (只有阴影分支用它), 网格 45MB 来自 HF tactile-mnist-mnist3d。
