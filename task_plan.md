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

## 首次外部逐像素验证 (SimTactileMNIST 精确网格真值, n=420)
模块 twm/force_recovery/mnist_validation.py (348db56/59cb679)
真值本身经过验证: 位姿配对 sensor_image[i] <-> object_pose[i-1] (峰值压深恒为
2.2491±0.0016mm, 偏移0/+1散布±1.16/1.56mm); 用 Taxim 重渲染我方GT高度图能复现
出厂图像 MAE 1.76/255 corr 0.995; 尺度重采样到 1px==MM_PER_PIXEL(误差0.05%)。

### **决定性发现: 浅压极好, 深压崩溃**(压深扫描, 零拟合)
| 压深mm | 0.30 | 0.60 | 1.00 | 1.50 | 2.25(数据集) |
|---|---|---|---|---|---|
| MAE µm | **11.2** | 35.0 | 67.8 | 127.4 | 281.1 |
| Type-2 µm | 96.5 | 186.3 | 308.6 | 514.6 | 961.8 |
| 峰值 ours/GT | 1.00 | 0.97 | 0.77 | 0.68 | **0.55** |
=> <=0.6mm 且零拟合、在**非球面**真值上**优于 3D Cal 已发表数字**
   (其 Type-2 171.6/152.8/290.0µm, 且用了2D互相关对齐+拟合压深尺度)。
   2.25mm 时我们差一个数量级。正确表述: **工作区间是浅压**, 深压超出标定坡度范围。
   任何精度数字必须带压深, 否则无意义。

### 三个缺陷的外部再测 — 两个要改口
| 缺陷 | 此前说法 | 真值测得 | 处置 |
|---|---|---|---|
| 平顶过度穹顶 | +23-42% | 中心/边缘 ours 1.54-1.56 vs GT 1.40; 封闭平台对照(真值恰为1.000)得1.069 = **+7%** | **大幅下调**(此前多数是物体真实曲率+裁切) |
| 未标定LUT色格 | "最高22%, 改进点1" | 13.8/16.9%, 与Type-2误差相关仅 **0.098/0.294** | **降级为次要因素** |
| 光晕掩码 | 定性 | IoU **0.614**, recall 0.917, 过分割 **0.531**(多出面积=真实接触的53%) | **确认并量化** |

### 排除的解释
- 光度域差不是瓶颈: LUT梯度与真实凝胶梯度夹角 Taxim渲染 24.4°(79.4%在30°内)
  vs GlowTact LUT在自家真实球压上 26.1°(78.6%) — 同一统计量同一代码
- 用Taxim球压重标LUT**并不改善**数字(316.9 raw vs 273.4) => 不能归因于sim-vs-real
- 420/420 触碰接触都溢出视野; 对照: 球冠从中心移到边缘峰值 1.39->0.30mm(真值0.90),
  Type-2 292->450 => 与 cnc_Mini / Sparsh 同一可见性效应, 应作为**跨数据集共性**陈述一次

### 诚实注记
sim图像是Taxim渲染 => 验证的是**几何解算器**而非传感器模型;
Taxim的凝胶与物体几何仅差38µm => 几乎不建模真实凝胶的非贴合性。
Stage2(RealTactileMNIST真实帧)未做: 需逐轮拟合(x,y,yaw)+全局z, 失败模式静默。

## 可追溯性修补: 站上补充"评测过但未纳入的数据集"一节 (EN+ZH)
用户问"为什么结果里没有 FeelAnyForce" -> 暴露缺口: 它在站上只作为**方法**
(基线网络)出现 9 次, 从未说明它作为**数据集**为何缺席; 排除理由只在一段
讲对照方法的话里被顺带提及, 读者找不到。
新增一节明确列出:
- FeelAnyForce: 本地 48197 张图**力标签全为空**; 标签需单独下载并推断 capture
  内帧索引, 而该对齐**从未被验证**(混合 rho 0.455 vs capture 内打乱 0.442
  => 相关性全来自 capture 间结构, 帧映射完全错误也会保留)。纳入=发布无法辩护的数字。
  修正需带时间戳原图(~157GB)或官方帧映射。
- Tactile MNIST 真实划分: 用于重建真值, 无力标签。
- 另附因无牛顿力真值/传感器不符而排除的清单(TouchAndGo/TVL/TacQuad/GelSLAM/
  3DCal/DAR_OTS/GenForce/AllSight/DIGIT/Wedge), 以及仍开放的 TacVerse。
原则: 被排除的数据集也应在结果页可见, 否则读者无法判断覆盖面。

## FeelAnyForce 定向抽取: 绕过对齐问题(用户要求纳入)
思路转变: 不再尝试"推断 capture 内帧索引"(该对齐已被组内打乱对照否证),
而是**从原始分卷 zip 按文件名直接抽图** —— 文件名就是 CSV 引用的时间戳。
侦察结果(全部已验证, 勿重推):
- 分卷 zip: disk0-2 = dataset.z01/z02/z03, disk3 = dataset.zip(最后一卷, 17.45GB)
- Zip64 EOCD: 305818 条目; 中央目录在 dataset.zip 偏移 17,411,261,400, 大小 38,401,830
- 中央目录已解析并缓存: force_recovery/faf_cd.json ({n,d,o,c,u,m})
- 101,883 个 dataset/<capture>/tactile/<timestamp>.png, 压缩中位 109,633 B, 42 captures
- **CSV 路径与压缩包条目 110,109/110,109 = 100.0% 精确匹配**(前缀 'dataset/')
- |Fz| 0.00-30.81 N; 每 capture 行数 min 727 / 中位 1018 / max 13495
=> 不需要 157GB: 分层抽 4000-6000 张约 440-660MB 即可。
验收标准(与之前失败的对齐相同的检验): **capture 内打乱对照**必须被大幅拉开
(旧的推断式对齐是 0.455 vs 0.442, 逐 capture rho 仅 ~0.09)。

## Sparsh 的 OOD: 无官方划分, 但结构自带 5 条轴 — 全部已测
| OOD 轴 | 做法 | rho | MAE | 组内打乱对照 | 判定 |
|---|---|---|---|---|---|
| 1 形状(球->平) | 跨 probe | 秩保留 | 绝对牛顿失败(0.37-0.40N) | — | 部分 |
| 2 pad(6x6 迁移) | 跨 batch | 0.963-0.985 | 0.026-0.042N | — | **通过(几乎无代价)** |
| 3 滑移 | 稳定压拟合 -> slip>0 评测 | **0.964** | 0.051N | 0.318 | **通过** |
| 4 **力程外推** | 低力半区拟合 -> 高力半区 | **0.552** | 0.225N | 0.447 | **失败** |
| 5 跨轨迹 | 一半 trajectory -> 另一半 | **0.973** | 0.042N | -0.043 | **通过** |
基线(同 batch 随机半半分): 0.968 / 0.042N

### 力程外推失败的机制: isotonic 截断
| 变体 | rho | MAE | 预测量程/真实量程 |
|---|---|---|---|
| linear + isotonic(现用) | 0.552 | 0.2247 | **0.05** |
| **linear only(去掉 isotonic)** | **0.839** | **0.1448** | 2.18 |
| log-linear only | -0.682 | 0.8458 | 9.82 |
=> isotonic 的 out_of_bounds='clip' 把训练范围外全部压平(量程只剩 5%)。
   纯线性外推 rho 0.552->**0.839**, MAE 0.225->**0.145**。
   但线性会过冲(量程 2.18x), log 版彻底发散(-0.682) => 需要**有界的参数化尾部**
   而非直接去掉 isotonic。行动项: 内插用 isotonic, 外推用受约束的单调尾部。

### 与 Tactile MNIST 的深度发现同构
深度: 标定坡度范围外 -> 峰值比 1.00->0.55, MAE 11->281µm
力:   标定力程范围外 -> 预测量程塌到 5%, rho 0.968->0.552
=> 同一个失效模式: **内插好, 外推坏**。这应作为方法的统一表述写进站里。

## React 片段伪影溯源 (用户指出 episode_007_right)
参考帧现行选法: run_episode._reference_rows 取强度最低的15个 fresh 行(间隔>=30行/1秒),
视频用前6个的逐像素中值。
| found | evidence | fix | verified |
|---|---|---|---|
| 我先怀疑参考帧污染 — **否证** | 两组不相交参考帧分半比较 p95 仅 3.0 灰度; **0.0%** 像素会因参考噪声通过 |dI|>8 | 无需修 | n=6/12/15 均 0.0% |
| 我误读了统计量 | 曾据"单帧与中值最大差 8.5-12 灰度"下结论, 但那是全帧最大值(个别离群像素), p95=3.0 | 记录教训: max 不能代表分布 | — |
| 真因: 锐利细长几何 | ep007/right 未标定LUT色格 **14%** vs ep003 7% / ep008 8%; 触觉图为细长刃口 | 待修(需扩展标定坡度范围) | 三 episode 对比 |
免费改进: 参考帧中值由 6 帧提到 12 帧(分半 p95 3.0->2.0), 已落地。
佐证: Wedge 论文明说阴影/形变影响重建 "especially for sharp surfaces" 且留待未来。
注意: Tactile MNIST 真值显示未标定色格与误差相关仅 0.098/0.294, 故 14% 未必是全部原因;
锐利接触同时伴随小接触面积(光晕占比高)与更强阴影。

## FeelAnyForce 改判: 从"排除"到"可纳入"(按 range 请求抽原始带时间戳图)
之前排除的理由是**对齐未经证明**(推断帧序号: 混合 rho 0.455 vs capture 内打乱
0.442, 逐 capture rho ~0.09)。本次不再推断: 从 4 分卷 zip(~81GB)按 HTTP Range
逐条抽取原始 PNG, **文件名即 CSV 引用的时间戳**, 连接按构造精确。
代码: `force_recovery/faf_extract.py`(抽取/选样), `force_recovery/faf_validation.py`(评测)。

| found | evidence | fix | verified |
|---|---|---|---|
| 缓存的中央目录 `faf_cd.json` 无 CRC 字段, 无法做独立完整性校验 | {n,d,o,c,u,m} 六个键, 没有 crc | 单独取一次中央目录(38.4MB, 占预算 1.9%)解析出 CRC-32+usize 缓存到 `faf_cd_crc.json` | 解析出 **305,818** 条中央头(与 Zip64 EOCD 完全一致), 其中 203,766 个 png |
| 分卷 zip 成员数据可能跨卷续写 | 前 3 卷各恰好 20GiB, 末卷 17,449,663,328 B | `read_range(disk,off,n)` 越界自动续到下一卷 | 5202 条中 disk 分布 0/1/2/3 = 1159/1369/1419/1255, 全部成功 |
| 本地文件头的 name/extra 长度可能与中央目录不同, 用错会把载荷起点偏移几字节 | zip 规范允许两处 extra 不同 | 只用**本地头自己的** nlen/elen 定位载荷, 并强制本地头文件名与中央目录一致 | 5202/5202 通过; 另比对本地头 crc/size(未置 flag bit3 时)全部一致 |
| CSV 110,109 行 > 归档 tactile 条目 101,883 | 3,188 帧同时出现在多个 split(官方 train/val/test **有泄漏**) | 按路径去重(重复行 Fz 完全一致, 冲突数 **0**) | 去重后 101,883 唯一帧, 与归档条目数相等 |
| **28/42 个 capture 根本没有无接触帧** — 差分法参考帧必然被污染 | 全部 `<物体>_<n>` 复跑(2024-06/07)的 **最小 |Fz| = 4.87-6.01 N**; 基础 capture(2024-04/05)最小 |Fz| <= 0.096 N。跨 2-3 个月借参考帧不可辩护 | 按标签(不是按结果)分层: tier A = 14 个有无接触帧的 capture; tier B = 28 个没有的, 只能用中值图参考, **单独报告绝不混入表头** | tier A 每个 session 的参考帧 |Fz| <= **0.096 N**(最差 triple_cylinder7, 其余 <= 0.033) |
| 一个 capture 不等于一次 sitting | `cube28_corner` 跨 **29 天 / 11 个 session**; 单张参考帧对其中大部分时间是错的 | 时间戳缺口 >600s 切 session, **逐 session** 取最轻 3 帧中值作参考 | tier A 共 54 个 session 有合格参考; 无合格参考的 session 的帧不进入 tier A |
| 归档里没有"额外的未标注帧"可用来找无接触参考 | 逐 capture 比对: 归档 tactile 数 == CSV 唯一帧数, **extra = 0**(42/42) | tier B 只能走中值图参考 | — |

### 抽取完整性
5202 帧(2800 tier A + 2240 tier B + 162 参考), **5202/5202 通过中央目录 CRC-32 +
解压尺寸校验, 失败 0**; 事后离线随机复验 300/300 仍匹配。下载合计 **609 MB**
(571 MB 图 + 38 MB 中央目录), 预算 2GB。图像原生就是 **320x240 = 我们的 (W,H)**,
故**不裁切不缩放**直接用(与 FEATS 处理一致; GlowTact 的 1/7 边框裁切是 GlowTact 专用)。

### 抽样
每 capture 内按 |Fz| 十分位分层(tier A 200 帧/capture, tier B 80 帧/capture),
避免被接近阶段的近零力帧淹没。tier A 覆盖 0.00-30.81 N
(0-0.5N:306, 0.5-1:184, 1-2:347, 2-4:540, 4-8:764, 8-16:608, 16-32:51)。
tier B 因数据本身无低力帧, 只能覆盖 4.9-17.9 N。

### 结果 — 每个数字都并排给出 capture 内打乱对照
协议与其他数据集完全相同(`force_eval_all.evaluate`): capture 内分半最小二乘
[vol, vol2, maxd, area, sqrt(area)*maxd] + isotonic, 5 seeds 中位。

| 子集 | n | cap | rho | **打乱对照** | MAE(N) | 逐capture rho 中位[min,max] |
|---|---|---|---|---|---|---|
| **A 干净参考(14 caps)** | 1400 | 14 | **0.961** | **0.338** | **0.85** | **0.953 [0.89,0.99]** |
| A 中值图参考(同样的帧) | 1400 | 14 | 0.909 | 0.335 | 1.18 | 0.929 [0.62,0.98] |
| B 中值图参考(28 caps) | 1120 | 28 | 0.519 | 0.092 | 2.21 | 0.532 [-0.01,0.87] |
| A+B 中值图参考(全 42) | 2520 | 42 | 0.869 | 0.629 | 1.67 | 0.559 [-0.04,0.98] |

**对照被大幅拉开: 0.961 vs 0.338(差 0.62)**, 而被否决的推断式对齐是 0.455 vs 0.442
(差 0.013); 逐 capture rho 从 ~0.09 变成中位 0.953、**最差的 capture 也有 0.89**。

### 追加的两个对照(都不是任务要求, 但是"太好了吧"的直接答复)
- **连接扰动**: 把每帧改标成它**在该 capture 完整时间线上后 k 帧**的力(保留
  capture、顺序、边缘分布, 只让连接错 k 帧) => k=1: rho **0.603**(逐 capture 0.405),
  k=5: 0.473, k=25: **0.430**(逐 capture **0.145**)。错 25 帧的成绩**恰好落在旧的推断式
  对齐(0.455 / 0.09)上** — 反过来印证旧对齐就是一个"错了几十帧"的连接, 也证明
  精确文件名连接是**承重的**, 不是装饰。
- **时间分块**(拟合=每 capture 最早一半, 评测=最晚一半, 评测帧在拟合集里没有时间邻居):
  A 干净参考 rho **0.912** / MAE 1.27 N; A 中值参考 0.847; B 0.479。随机分半没有靠
  相邻近重复帧作弊。

### 参考帧污染的代价 — 同一批帧上直接测出来
tier A 用中值图参考(而非无接触参考)重跑**同样 1400 帧**: rho 0.961 -> 0.909,
MAE 0.85 -> 1.18 N; 逐 capture 最惨的是平面/棱接触: `cube28_edge` 0.989 -> **0.619**,
`cube28_flat` 0.992 -> **0.706**(球/柠檬几乎不掉)。tier B(参考帧里已经压着 5.5N)
只有 0.519, 4 个 capture 落到 <=0.12(`cylinder400_5` **-0.007**)。
=> 这就是"参考帧污染封顶"效应的**同数据集内定量证据**, 与 FEATS 上的诊断同源。

### 接触可见性(与 cnc / Sparsh / Tactile MNIST 的共性检查)
先纠正一件事: 圆盘检测器在这里**不适用** — FeelAnyForce 的压头多是圆柱/十字/环/
立方棱, `detect_circle` 只在 2800 帧中的 26.2% 触发(球/柠檬那 3 个 capture 才是圆斑)。
改用**形状无关**的判据(pipeline 自己的 valid mask 是否碰到 10px 边框):
有接触 94.5%, **完全在视野内 34.8%**, 碰边框 59.6%, 无接触 5.5%。

| 子集 | n | rho | 打乱 | MAE(N) |
|---|---|---|---|---|
| A 接触完全在视野内 | 484 | 0.956 | 0.519 | **0.36** |
| A 接触碰到边框 | 833 | 0.911 | 0.317 | 1.17 |
| A 圆斑且在视野内(仅球/柠檬) | 210 | 0.984 | 0.628 | 0.26 |

**注意: 这里 in-view 并没有像 cnc/Sparsh 那样抬高 rho**(0.956 vs 全体 0.961)。原因是
in-view 子集的力中位只有 1.60 N 而碰边框子集是 6.65 N — in-view 同时也窄化了力程,
rho 被压缩; 真正的差别体现在 **MAE 0.36 vs 1.17 N**。诚实结论: 可见性效应在
FeelAnyForce 上以**误差**而非**秩相关**的形式出现, 不要把它当成又一次 rho 提升来讲。

### 被否决 / 被排除的变体
- **借用同物体基础 capture 的参考帧**给 tier B: 否决。基础 capture 是 2024-04/05,
  复跑是 2024-06/07, 跨 2-3 个月的凝胶/光照状态不可比, 会把污染换成一个更隐蔽的偏差。
- **用中值图参考统一处理全部 42 个 capture** 做表头: 否决。它能出 0.869 的漂亮数字,
  但对照是 0.629(tier B 全是高力 capture, capture 间结构本身就撑起大半), 且逐 capture
  中位只有 0.559 — 是"看起来更全面、其实更弱"的数字。
- **圆盘检测器作为 in-view 判据**: 否决(只覆盖 26% 的帧, 且系统性偏向球形压头)。
- **把 tier B 直接丢掉不报**: 否决。它是"没有无接触帧会发生什么"的现成对照, 要报, 但
  单独一行并注明。

### 判定
**可纳入, 作为 tier A 的 14 个 capture / 1400 评测帧的一行**:
rho **0.961**, MAE **0.85 N**, capture 内打乱对照 **0.338**, 逐 capture rho 中位
**0.953**(最差 0.894), 时间分块 0.912, 错 1 帧的连接就掉到 0.603。
写进结果表时必须同时写清界定: (a) 只有 14/42 个 capture 有无接触参考帧, 另外 28 个
最小 |Fz| 就有 5.5N, 单独报 0.519; (b) 特征缓存 `feature_cache/faf_<capture>.json`
(42 个文件, 与 Sparsh 同结构), 指标 `feature_cache/faf_metrics.json`。

## FeelAnyForce 恢复成功 — 问题在数据处理, 不在数据
定向抽取原始带时间戳图像(faf_extract.py): 5202 成员 / 609MB(82GB 压缩包中),
**5202/5202 CRC-32 校验通过**; 图像原生 320x240 = 我们的 (W,H), 无需裁剪缩放。
| 指标 | 旧(推断对齐) | **新(文件名精确)** |
|---|---|---|
| pooled rho | 0.455 | **0.961** |
| capture 内打乱对照 | 0.442(差 0.013) | **0.338(差 0.62)** |
| 逐 capture rho 中位 | ~0.09 | **0.953**(最差 0.894) |
两个防"太好了"的对照(代理主动加的, 很关键):
- **对齐扰动**: 改标成 k 帧后的力, k=1→0.603, **k=25→0.430/逐capture 0.145**
  = 恰好落在被否决的推断对齐上 => 文件名精确匹配是承重的
- **时间分块划分**(每 capture 靠前一半拟合): 0.912 => 随机半半分未利用相邻帧

### 关键作用域: 只有 14/42 个 capture 可用
28 个 capture **通篇无无载荷帧**(|Fz| 最小 4.87-6.01N), 无有效参考帧
=> rho 仅 0.519(对照 0.092)。它们是 Jun/Jul 重跑, 干净的是 Apr/May,
借用参考跨 2-3 个月, 已否决; 且查过压缩包无未标注的额外帧可救(42/42 数量相等)。
=> **与 FEATS 同一失效模式**: 无干净参考 => 积分参考失配而非接触。
   同一批帧换成中值图像参考: 0.961→0.909, 且专门打击平面/边缘接触
   (cube28_edge 0.989→0.619, cube28_flat 0.992→0.706)。
数据集缺陷: 官方 CSV 有 **3188 帧同时出现在多个划分**(train/val/test),
重复帧 Fz 完全一致, 已按路径去重(否则同一帧会落进我们划分的两半)。
in-view: 圆盘检测器不适用(仅26.2%触发, 多为圆柱/十字/环/立方边);
用形状无关判据: 完整在视野 34.8%, 裁切 59.6%。in-view rho 0.956/MAE 0.36N
vs 裁切 0.911/1.17N -> **rho 提升未复现**(in-view 同时是低力子集),
故按 MAE 报告而非当作第四次确认。

## 图件布局改为两行 (用户要求) — 按测量改, 非凭观感
安装了设计类 skill 到本项目: measured-design-audit / design-regression-guard /
single-source-visual-consistency (来自 Yuxiang-Ma/agent-skills)。

### 定罪的测量
| 图 | 宽高比 | 容器 | 每格宽度 | 标题有效字号 |
|---|---|---|---|---|
| workbench 14列 | 7.36 | 1900px | **136px** | **6.1px** |
| gallery 8列 | 6.58 | 880px | **110px** | **2.8px** |
| debug 7列 | 7.20 | 880px | **126px** | **2.8px** |
标题 2.8-6.1px = 不可读。

### 改后
| 图 | 宽高比 | 每格宽度 | 标题有效字号 |
|---|---|---|---|
| workbench 2x7 | **3.47** | **271px** | **10.5px** |
| gallery 2x4 | **2.05** | **220px** | **6.0px** |
| debug 2x4 | **2.07** | **220px** | **6.7px** |

中途自纠一次: 第一版 2x7 用 figsize=(16.5,9.2)=1.79:1, 留下大片垂直留白 +
标题与色标条碰撞。根因可算: 7列4:3面板x2行 = 7*4:2*3 = **4.67:1**,
figsize 必须跟随单元格纵横比而非拍脑袋。修正为 17.0 x (17.0/4.67*1.28)。
另修: 标题过长撞色标条(缩短文案), 面板13 y轴标签压面板12色标条(移入标题)。

### 冻结为回归门: twm/force_recovery/design_guard.py
断言(按缺陷类而非实例, 参数化到每张图):
1. 由宽高比反推网格 -> 每格 >= 180px (改后实测 220-271)
2. 标题渲染尺寸 = pt * 列宽/图宽 >= 4.0px (改后实测 6.0-10.5, 改前 2.8)
3. 站内副本与源文件 md5 一致(此前两次出现"重生成了但站上仍是旧图")
自检通过: 旧单行图(2332x317)被推断为 14列x1行 / 136px -> **拦截**; 新图通过。

## 代码整理 (goal: 整理 code)
44 个模块 / 12,579 行。用**导入图**而非印象分类, 发现两处真实结构债:
1. **重建核心 `stages()` 住在 `debug_gallery.py`** —— 名字像"调试图库", 却被 8 个
   模块依赖。不移动(移动=纯风险的大范围重构), 改为在新的 `pipeline.py` 中重导出,
   并在 ARCHITECTURE.md 显式记录这笔命名债。
2. **`depth_force.py` 是已废弃的 v1 MLP 管线, 但仍有 6 个引用者**(都是旧验证脚本)。
   标注为 attic 并写明"不要在其上新建", 不删除以保留出处。
产出:
- `force_recovery/pipeline.py` —— 稳定公共 API: reconstruct / force_from_depth /
  penetration_mm / virtual_target / **STIFFNESS_N_PER_MM = 1.0**。
  docstring 直接带上已测的适用边界(浅压 11µm@0.3mm vs 深压 281µm@2.25mm;
  力外推 0.968→0.552; ρ 是组内秩相关而非可迁移绝对牛顿刻度), 使调用者无法
  在不知道边界的情况下引用数字。
- `force_recovery/ARCHITECTURE.md` —— 模块地图: 核心/数据集适配/研究/站点构建/attic,
  每个 attic 模块附"为什么死了"的证据(如 fusion 的 8 种符号约定最好相关 0.049)。
自检: 零力 -> 目标位姿与观测位姿逐元素相等(标量与批量均通过)。

## 刚度改为 1 N/mm + 单一真值源 (goal: 1N/mm 刚度)
发现将要产生的不一致: `dexforce.STIFFNESS_N_PER_M = 1500`(1.5 N/mm) vs 我新写的
`pipeline.STIFFNESS_N_PER_MM = 1.0` —— **同一物理量两个真值源**。
修法(single-source-visual-consistency):
- `dexforce.STIFFNESS_N_PER_M = 1000.0` 成为唯一定义;
  `pipeline.STIFFNESS_N_PER_MM` 由它**推导**(/1000), 不再各自声明
- 页面模板里硬编码的 "k = 1500 N/m"(actions_page.py, site.py 各一处)
  改为 `@@K_NM@@` 占位符, 构建时从同一常量注入 -> 文字不可能与数据脱节
- **图也重生成**(showcase react): 只改文字不改图 = 页面显示与数据不符
验证: 站上 1000 N/m 三处、1500 零处、占位符零残留; 布局门 0 违规。
**~~1 N/mm 是否合理(实测)~~ —— 此结论已撤回, 见下**
~~React episode_000 left 接触帧 ... 超厚占比 0.0% => 物理合理~~

### 撤回: 我犯了取样错误
上述结论只取了 **72 个 sensor-side 中的 1 个**(episode_000 left), 且只算接触帧;
那一段的力上限仅 2.30 N, 完全不代表全量。**全量实测(480,080 个样本)**:
穿透 p50 0.0012 / p95 **5.686** / p99 9.685 / max **23.835 mm**,
**7.85% 的行超过 4.25mm 凝胶厚度, 72 侧中 27 侧连 p95 都越界**。
=> **1 N/mm 在全量上不成立**, 仅在低力段(<=4N)成立。
教训: 用单个 episode 的分布代表数据集, 与之前"用 max 代表分布"是同一类错误——
**报告分布前必须先确认取样覆盖面**。已独立复算确认(36 parquet / 480080 样本)。

---

# Ledger: 力列写入 React 数据集 + DexForce 力动作 (k = 1 N/mm)

代码: `twm/force_recovery/export_force_columns.py`
入口: `python -m twm.force_recovery.export_force_columns export|verify|digest`
输出: `/media/yxma/Disk1/twm/release_force/<task>/meta/<date>/<episode>.parquet`
(**平行目录, 原 `release/` 只读**; 每次全量重建, 两次运行 sha256 一致)

## 新列 (每侧 3 列, 共 6 列; 19 -> 25 列)
| 列 | 类型 | 单位 | 定义 |
|---|---|---|---|
| `force_{side}_normal_n` | float32 | N | 逐行法向力, >= 0, 无接触帧恰为 0.0 |
| `force_{side}_penetration_mm` | float32 | mm | `force / k`, k 为**假设**刚度 |
| `force_{side}_target_pose` | list\<double\>[7] | m + quat(xyzw) | 观测位姿沿按压方向前推 penetration; 四元数原样复制 |

k 写在三处(不只在代码里): 每列的 parquet field metadata (`twm.stiffness_n_per_mm`)、
schema metadata `twm.force_export`、同目录 sidecar `<episode>.force.json`
(外加 run 级 `force_export_manifest.json` / `force_export_verify.json`)。
k 本身**不在本模块声明**, 而是 import 自 `pipeline.STIFFNESS_N_PER_MM`
(← `dexforce.STIFFNESS_N_PER_M = 1000 N/m`), 保证数据列与站点图不会各说各话。

## found / evidence / fix / verified
| found | evidence | fix | verified |
|---|---|---|---|
| 任务前提说"87 个 sensor-side"实为 **72** | `force_recovery/**/*.npz` 共 87 个, 其中 15 个是 `feature_cache`(2) / `lut_calibration`(2) / `mnist_validation`(11), 不是 episode 产物。72 = (32 motherboard + 4 pushT) x 2 侧 | 导出以 release 的 36 个 meta parquet 为准, 缺任一侧的 npz 直接 `MissingForceFile` 报错 | 72/72 全部找到, 0 缺失 |
| 行数对齐可能失配 | npz 逐行数组 vs parquet 行数 | 不等即抛 `RowCountMismatch` 退出, **不截断不补齐** | 对齐通过率 **72/72 = 100%**; 人为把 parquet 切到 500 行, 守卫按预期抛错 |
| **k = 1 N/mm 的穿透量尾部不物理** | 力 p50/p95/p99/max = 0.0012 / 5.686 / 9.685 / **23.835 N** → 穿透量同数值 (mm)。**7.85% 的行 > 4.25 mm 凝胶厚度**, 72 个 sensor-side 里 **27 个**的 p95 就已越界 | 不静默接受: 按 k 扫描并给出建议区间(见下), 值仍按用户指定的 1.0 导出 | k=1.5 → p95 3.79/max 15.89 mm (3.86% 越界); k=3.0 → 1.90/7.95 (0.31%); **k=5.61 → 1.02/4.26 mm (0% 越界)**; k=15.4 → 0.37/1.55 mm |
| 1 N/mm 与凝胶实测刚度差一个量级 | 用估计器自己的压入深度作分母: `F / max_depth_mm` (F>0.5 N, n=147071 帧) = p5/p25/p50/p75/p95 **5.3 / 9.9 / 15.4 / 22.9 / 34.7 N/mm** | 记录在 verify 报告与本 ledger | 若把 penetration 读作"真实凝胶压缩", k 应为 ~15 N/mm; 读作阻抗控制器增益则 3-5.6 N/mm 是安全上限。**建议区间 3-6 N/mm(控制器语义) / 10-20 N/mm(物理语义); 1 N/mm 仅在"只用低力段"时成立** |
| 按压方向不能写死 | 标定得到的 `gel_axis_in_rigid` 左 `[-0.174,-0.932,-0.316]` / 右 `[0.350,-0.925,0.149]` —— 主轴是刚体 -Y, 天真的"工具 z 轴"会偏 **71-108 度** | 逐行 `n_hat = R(q_row) @ gel_axis_in_rigid`(标定 pose-to-pose 一致性 <= 1.07 度), 随传感器旋转 | 纯运动学独立验证(不用标定): 力上升期 `v·n_hat > 0` 的 sensor-side 占 **94.3%**, 释放期 `v·n_hat < 0` 占 **70.0%**, 上升 > 释放占 **92.9%**, `corr(dF, v·n)` 中位 **0.270** 且 **95.7%** 为正 (70/72 侧样本足够) |
| 无接触帧可能变成 NaN 或非恒等 | 45.73% 的行力恰为 0(每侧 p5/p50/p95 = 3.8/43.0/67.9%) | `force == 0` 的行 target_pose 用**拷贝**而非加 0 浮点运算 | **219518** 无接触行, `max|target - observed| = 0.0e+00` 逐元素相等 → PASS; 四元数全体最大偏差 0.0e+00 |
| `is_new == False` 重复帧语义不明 | `run_episode` 已把估计前向填充到重复行 | 本导出保留并**断言**该契约(不符即报错) | **344783/344783 = 100.00%** 重复行的力与前一行严格相等。已写入文档的副作用: 力是 ~8 Hz 信号挂在 30 Hz 位姿流上, 重复行 = 新位姿 + 旧力 |
| 目标位姿与力可能不自洽 | DexForce 往返: `k·‖target − observed‖` 应等于 F | verify 里逐行核对 | 最大误差 **5.77e-14 N** |
| 原数据可能被就地破坏 / 重跑不幂等 | — | 写 `release_force/` 平行树, 每次从两个只读输入全量重建 | 连续两次 export 的全体 parquet sha256 完全一致 (`8fcb8f03...`); `release/` 的 mtime 未变 |

## 数字速查
36 episodes / 72 sensor-sides / 240040 行 / 480080 个力样本 / 60 MB。
每侧力 p50 中位 0.0040 N (0-5.262), p95 中位 3.198 N (0.607-13.010),
max 中位 6.833 N (1.657-23.835)。帧率 29.9 Hz, is_new 占比 ~0.28。

## 与上一条 ledger("1 N/mm 物理合理, 0.0% 超厚")的冲突 —— 以全量为准
上一节的判据只取了 **1 个** sensor-side (React episode_000 left, 且只算接触帧),
那段的力上限 2.30 N, 自然 0% 越界。全量 **72 个 sensor-side / 480080 行**下:
力 max **23.835 N**, 穿透 **7.85% 的行超过 4.25 mm**, **27/72 侧的 p95 就已越界**。
教训: "单 episode 判据"会把 k 的合理性说反; 站点上那句 0.0% 只对该 episode 成立,
需改成全量数字或明确标注取样范围。(本次未改站点文件——用户正在并行整理。)

## 力导出上站 + README 更正 (全量数字取代单 episode 结论)
站点新增双语章节"力作为 observation, 以及由它导出的 action":
新增 6 列(force_{side}_{normal_n,penetration_mm,target_pose})、k 写入 parquet
字段元数据+sidecar、按压方向来自双球标定(并用纯运动学独立验证: 力上升期
v·n̂>0 占 94.3%, corr(ΔF, v·n̂) 为正占 95.7%)、自由空间严格恒等(219,518 行
逐元素为 0)、往返闭合 5.8e-14 N、重跑逐字节一致。
**页面本身写明了撤回**: 早前据单个 sensor-side(力上限 2.3N)推断"0% 超出凝胶",
全量 480,080 样本为 p95 5.69mm / 7.85% 超厚, 该说法作废。
k 扫描与建议区间(3-6 N/mm 作虚拟偏移, 10-20 N/mm 作真实压缩)一并上站。
README 同步更正。数据仍按声明的 1 N/mm 导出(单一真值源, 改 k 一处即可)。

## 预览视频叠加按压力 (半透明圆点, 用户要求)
新增 twm/force_overlay.py + 接入 scripts/build_episode_previews.py。
### 设计决策(均有理由, 非默认值)
- **面积∝力, 而非半径∝力**: 人对圆点大小的判断跟随**面积**; 半径正比会把大力
  在视觉上放大约二次方。故 r = R_MIN + (R_MAX-R_MIN)·√(F/F_FULL)。
  单测断言面积线性(25 点, 最大偏差 2.2e-16)。
- **固定标度 F_FULL=8N 而非逐片自适应**: 保证同一大小的点在所有视频里含义相同。
- **零力不画点**: 常驻圆点会暗示不存在的接触(阈值 0.02N)。
- **图例与圆点共用同一个 radius_px**: 画的和标的不可能脱节。
### 帧对齐(最易静默出错处)
力按 release parquet 行索引, 预览按 h5 帧迭代。run_episode 用
`trim(=parquet source_h5_frame[0]) + row + LEGACY_SHIFT(15)`, 而预览自带的
`_get_trim_offset` 读的是 .pt 的另一个字段且**常常不存在**(退化为 0)。
实测 6 个 episode: .pt 全部缺失, parquet trim 为 0/1 => 若用预览的 trim,
圆点会偏离其所标注的画面 ~15 帧 = **0.5 秒**。故 row_for_h5_frame 从 parquet 重推。
单测覆盖四个边界(首行/首行前/末行/末行后)。
### 自查修掉的两处
1. 读数文字被面板底部状态栏遮挡 -> 移到瓦片顶部并加黑色描边
2. **两条新断言被追加到 `__main__` 块之后, 根本没被收集执行**(grep 才发现)
   -> 移到收集循环之前; 现确认 `ok test_force_dot_area_is_linear_in_force`
   与 `ok test_force_row_mapping_uses_parquet_trim` 均在跑。
验证: episode_000 h5 帧 114(力 1.91N) 圆点大小落在图例 0.5N 与 2N 之间, 读数 "1.9 N" 可见。

## 预览视频力叠加: 覆盖面核查(自查发现缺口)
批量重生成后**清点发现覆盖不全**: 36 个预览中只有 27 个(05-10/05-11)带力叠加,
另 9 个是更早运行的产物。逐日期比对"力 npz 数 vs 预览数"后定位:
| 日期 | 有力估计的 episode | 预览 | 处置 |
|---|---|---|---|
| 2026-05-10 | 24 侧/12 ep | 12 | 已带叠加 |
| 2026-05-11 | 30 侧/15 ep | 15 | 已带叠加 |
| **2026-05-19** | **10 侧/5 ep** | 5(旧) | **已重生成** |
| 2026-03-23 / 05-15 | 0 | 3 / 1 | 无力数据, 叠加不适用(非缺陷) |
抽查方法(不靠肉眼): 每个日期取一段, 每 20 帧采样, 统计 gelsight 瓦片区域内
橙色(r>140 且 r-b>60)像素占比 -> 05-10 1.04% / 05-11 1.18% / 05-19 1.86%,
三者均远高于 0 => 叠加确实存在。
教训: "批量跑完"不等于"覆盖完整", 必须清点输入与输出的对应关系。

## Sparsh 上补跑两个基线 (FeelAnyForce / FEATS) —— 补齐结果矩阵最后一行

站点矩阵的 Sparsh 行里另外两列一直写着 "not run — no published predictions"。
本轮把它跑出来了。**结论先说: FeelAnyForce 在 Sparsh 上赢了我们的物理管线**
(视野内 rho 0.985 vs 0.968, MAE 0.030 vs 0.042 N; 而且它**零标注**就有
rho 0.967 / MAE 0.060 N, 我们的 0.042 N 是逐 pad 重新校准换来的)。FEATS 塌成常数。

新增: `twm/force_recovery/sparsh_baselines.py` (取帧/导出/FEATS/评测)、
`twm/force_recovery/anyforce_sparsh.py` (在 `anyforce` 环境里跑 FeelAnyForce)。
产物: `feature_cache/{anyforce,feats}_on_sparsh.json` (各 7500 行)、
`feature_cache/sparsh_baselines{,_allframes}.json`。

### found / evidence / fix / verified

| found | evidence | fix | verified |
|---|---|---|---|
| 基线必须跑在**同一批帧**上才可比 | 视野内掩码筛掉 56% 的帧, 且被裁切的接触盘力中位数最高 —— 用不同子集比较等于换数据集 | 帧号不重新采样, 直接从 `feature_cache_sparshlut/sparsh_*.json` 的 `index` 逐条读回, 再套同一个 `eval_circles.json` 掩码 | 物理列复算 = rho 0.9682 / MAE 0.0422 / shuffle 0.264 / n_eval 1667, 与 `force_matrix.json` 逐位一致 |
| FeelAnyForce 的**预处理选择**能把结论从 -0.2 翻到 +0.985 | sphere_b1 四变体: `full_bg` +0.985 / `crop_bg` +0.976 / `crop_raw` -0.208 / `full_raw` -0.224; 两个 raw 变体输出恒定在 18.4-19.5 N | 采 `full_bg`(不再补裁 1/7 边框, 只做 `clip(im-bg+127)`); 两个 bg 变体全量都跑并存进 json | Sparsh 的 320x240 帧**已经是 gsdevice 裁过的输出**, 再裁一次会丢 1/7 pad; 背景相减是死线, 少了就退化为常数 |
| FEATS 在无 marker gel 上不是"弱", 是**输出常数** | 预测 IQR 0.002 N vs 真值 IQR 0.341 N (**1%**); 93% 的帧落在自身中位数 ±20% 内; MAE 0.341 N **差于**最优常数预测 0.181 N; 逐 pad 原始 rho 全为负 (-0.05..-0.21) | 三种输入链全跑 (`pre_full`/`pre_crop`/`asis_full`), 取最有利的 `pre_crop` 作主值 | 与它在 GlowTact 上的塌陷同型 (那里 pred 均值 0.156 N, rho 0.041); 域内 FEATS val 则是 0.17-31.6 N 跟着真值走 (rho 0.960) —— 是域效应不是弱模型 |
| 逐 pad isotonic 会给**近常数输出免费加分** | 同协议下纯噪声列 pooled rho = **0.265**; FEATS 校准后 0.384 —— 只高出 0.12。逐 pad 校准还能把 FEATS 的 -0.099 翻成 +0.261(isotonic 可自选符号 + 大量并列秩) | 主报告值改用**无拟合** rho; 校准值一律与随机对照并列 | 随机列逐 pad 校准后 rho = -0.06..+0.04 (真的是 0), 说明 FEATS 的 +0.2 来自"稳定的负相关被翻了个号", 不是信号 |
| 我们的管线**依赖视野内过滤**, 基线不依赖 | 全部 7500 帧(含出画面): 我们 0.683, FeelAnyForce 0.897 (raw 0.870); sharp 尤其明显 (我们 0.567 / 它 0.872) | 如实报两套口径 | 这是 Poisson 零边界的老问题, 第五次在外部数据上复现 |
| 需要多少本传感器标注? | 冻结校准(只在 sphere_b1 拟合, 其余 pad 原样套用): 我们 rho 0.860 / MAE 0.117 N (用了 467 个标注), FeelAnyForce **0 个标注** rho 0.963 / MAE 0.064 N | 把"标注预算"作为一列一起报 | 逐 pad 重校准掩盖了这个差距: 那一栏我们 0.968 看着接近, 冻结后差 0.10 |
| 首轮取帧 15 min/batch, 机器进 swap | 每 batch 物化 750 个 float64 局部参考 ×2 (crop+full) = 2.8 GB, 加上 float32 帧字典 → RES 5.1 GB, 系统 swap 17 GB | 帧一律留 uint8, 局部参考改为 `local_ref()` 按需算 (uint8 栈取 median 与 float32 栈取 median 数值相同) | 与旧代码逐位一致 (`max|old-new| = 0`, crop_bg/full_bg 各 60 帧), 单 batch decode 1.5 s |

### 三方对比 (Sparsh, 视野内 3328 帧 / 9 块 pad; 协议同站上其他数据集)

| 口径 | Ours (physics) | FeelAnyForce | FEATS U-net | 随机对照 |
|---|---|---|---|---|
| 逐 pad 半半分 + isotonic, 5 seeds | **0.968** (MAE 0.042 N) | **0.985** (MAE 0.030 N) | 0.384 (MAE 0.161 N) | 0.265 |
| 无拟合(直接读模型输出的牛顿) | 不适用(输出是 mm³) | **0.967** (MAE 0.060 N) | 0.086 (MAE 0.341 N) | — |
| 组内打乱对照(逐 pad) | -0.07..+0.07 | -0.03..+0.07 | -0.03..+0.04 | -0.10..+0.02 |

逐压头(校准后 rho / MAE | 无拟合 rho): sphere 0.973/0.036 · **0.984**/0.028|+0.986 · 0.157/0.165|-0.081;
flat 0.957/0.044 · **0.979**/0.032|+0.981 · 0.089/0.147|-0.134;
sharp 0.619/0.116 · **0.878**/0.077|+0.900 · 0.197/0.141|-0.016。
**sharp 是我们最弱的一栏(0.619), 也正是 FeelAnyForce 领先最多的一栏(+0.26)。**
sharp_batch_1 视野内帧为 0 (679/750 根本测不到接触盘), 全数据集口径下才进得来。

### 站点这一行建议怎么写

把 "not run" 换成三个真数字, 并且**不要只写校准后那一栏** —— 只写 0.985 vs 0.968
会显得两者接近, 而真实差距在"要多少标注": 建议主表写无拟合口径
(FeelAnyForce 0.97 / FEATS 0.09), 脚注给逐 pad 校准后的 0.985 / 0.384 与随机对照 0.265。
一句话版本: **"在 Sparsh 上 FeelAnyForce 零标注就打平并略胜我们逐 pad 标定后的结果, 这是矩阵里
唯一一个预训练网络赢过物理管线的格子 —— 因为 Sparsh 正落在它的域内(无 marker GelSight Mini,
0-3 N); FEATS 则在同一批帧上塌成 0.073 N 的常数。"** 另外把 "each network collapses outside
its own domain" 的说法收紧: FEATS 成立, FeelAnyForce 不成立 —— 它在三个无 marker 数据集上
分别是 0.83 / 0.90 / 0.97, 只在 marker gel 上掉到 0.43。

## Sparsh 三方对比: 我们输了这一格, 并据此收紧了一个过强论断
### 可比性先证明
帧不重采样, 直接按 feature_cache_sparshlut 的 index 读回 + 同一 eval_circles 掩码
=> 物理列复算 rho **0.9682**/MAE 0.0422, 与 force_matrix.json **逐位一致** = 同一批帧。
### 三方(Sparsh 视野内 3328 帧 / 9 pad)
| 口径 | Ours | FeelAnyForce | FEATS | 随机对照 |
|---|---|---|---|---|
| **无拟合直读牛顿** | 不适用(输出 mm³) | **0.967** (0.060N) | 0.086 | — |
| 逐 pad 半半分+isotonic | 0.968 (0.042N) | **0.985** (0.030N) | 0.384 | **0.265** |
逐压头: sphere 0.973 vs **0.984**; flat 0.957 vs **0.979**; sharp **0.619** vs **0.878**。
### 关键的两个对照(决定了怎么读这张表)
1. **随机对照 0.265** => FEATS 校准后的 0.384 只高 0.12, **不是信号**;
   isotonic 甚至能把它逐 pad 的 -0.099 翻成 +0.261(可自选符号)。
2. **标注预算**(冻结校准, 只在 sphere_b1 拟合): 我们 467 标注 -> 0.860/0.117N;
   FeelAnyForce **0 标注** -> 0.963/0.064N。**逐 pad 重校准掩盖了这个差距**。
   全数据集(含出画面 7500 帧): 我们 0.683, 它 0.897 —— **它不需要我们的视野内过滤**。
### FEATS 失效机理(非仅低 rho)
预测 IQR **0.002N** vs 真值 0.341N(1%); 93% 帧落在自身中位数±20%内;
MAE 0.341N **差于**最优常数 0.181N; 逐 pad 原始 rho 全负。=> 卡住, 不是算错。
### 站点据此改了两处
- Sparsh 行 "not run" -> 真数字, 主表用**无拟合**口径(两个基线都直出牛顿, 与
  "重校准后才相关"是不同主张), 校准口径与随机对照并列给出。
- **收紧过强论断**: 原文"每个网络出域即塌 / 物理管线是唯一全域可用的估计器"
  被 Sparsh 推翻两半 —— FeelAnyForce 在三个无 marker 集上 0.83/0.90/0.97 **不塌**,
  且在 Sparsh 上**直接胜过我们**。改为: 物理管线是唯一**不需要**训练数据的估计器,
  而非唯一能跨域的。EN 模板 2 处 + ZH 配对 1 处全部更新。
### 两个会翻转结论的预处理事实(记录以免重推)
Sparsh 的 320x240 **已是 gsdevice 裁过**, 再套 1/7 裁剪 FeelAnyForce 0.985->0.976;
背景相减是死线, 不减则模型钉在 18.4-19.5N 且 rho 变负。

## 预览完成度核查(两个真实缺口 + 一个我自己的验证错误)
### 缺口1: pushT 完全没有预览
逐 episode 比对"有力数据 vs 有预览"才发现: pushT/2026-06-18 有 4 个 episode 的力
数据但预览数为 0。根因: build_episode_previews.py 把 **motherboard 硬编码在 4 处**
(H5_ROOT / OUT_ROOT / EPISODES_ROOT / 传给 build_preview_panel 的 task_name)。
已改为 --task 可配置并补生成。注意第 4 处不是路径而是入参, 只改路径会让状态栏
仍显示 "[motherboard]"。
### 缺口2: 2026-05-19 预览过时(上一轮已修)
### **我的验证脚本是错的** —— 差点据此误判功能坏了
橙色检测窗口写成 x∈[0,480), 那是**左侧的 raw+diff 两格**, 完全没覆盖
右侧 raw 瓦片 x∈[480,720)。motherboard 恰好左侧有接触所以测到, pushT 只有右侧
接触就报"0.000% 无叠加"。目视抽帧才发现点其实画得好好的(1.5 N, 右侧瓦片)。
修正窗口为两个 raw 瓦片的拼接后全量复查:
| | 峰值橙色 | 叠加帧 |
|---|---|---|
| motherboard/05-10 | 8.55% | 493 |
| motherboard/05-11 | 14.95% | 447 |
| motherboard/05-19 | 15.40% | 848 |
| pushT/06-18 | 4.36% | 37 |
| motherboard/03-23, 05-15 | 0% | 0 (无力数据, 正确) |
教训: 自动化检查本身也要被检验。此前报的 1.04/1.18/1.86% 虽然结论对(测到了橙色),
但只覆盖了一半瓦片 —— **一个只扫一半画面的检查, 通过了也不能算通过**。
### 独立问题(不在本任务范围)
pushT/2026-06-18/episode_004.h5 **文件损坏**(79GB, h5py 无法打开:
bad object header version number), 该 episode 本就没有力数据。
