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
