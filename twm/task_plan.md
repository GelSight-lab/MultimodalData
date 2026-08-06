
## Sparsh 上站 (第四个数据集) — 迁移上限而非重建结果
figure: twm/force_recovery/sparsh_figure.py -> site/assets/results_sparsh.png
(三面板: 逐 pad 标定散点 | pad 内打乱对照 | 跨 pad 迁移矩阵)
数字: rho 0.558 / MAE 0.138N / n=18750, 对照 0.161(有效对照, 保留各 pad 力程);
跨 pad 迁移 非对角 0.418-0.591 vs 对角 0.471-0.587 -> 一套标定可跨 pad。
**站点必须写明的警告**: GlowTact 标定的 LUT 在该传感器上重建无效
(球压→双瓣+中心凹陷, 打光几何不同), 相关性追踪 dI 幅值而非深度。
EN+ZH 各一节, 含三个数据集缺陷说明。已发布。

## 改进研究: 五个候选, 两个存活 (twm/force_recovery/improvement_study.py)
| 候选 | 对照/证据 | 结论 |
|---|---|---|
| 提取剪切特征 | **预言机**(喂真实剪切) 仅 0.9749→0.9757 | 关闭:残差-剪切相关是混杂(都随力大小) |
| 小神经网络 | MLP32x16 0.967 / MLP64x64 0.972 / GB 0.972 < 线性+isotonic 0.974 | 无用:每 pad ~400 帧, 柔性模型过拟合 |
| **对数特征** | 0.9735→0.9822, MAE 0.0361→0.0310 (-14%) | **采纳**(误差乘性,由残差-力相关 0.35-0.48 提示) |
| PINN-lite F/A=g(δ/√A) | 球→平 MAE 0.426→0.150(胜过 MLP 0.297) **但** F=g(体积) 达 0.0736 | **否决**:增益来自单调特征数, 非物理形式; GlowTact 留一上全面更差 |
| **复杂度匹配迁移场景** | 见下 | **采纳** |
结构性发现: 单一体积特征在**未见压头**上泛化更好, 五特征在**已标定形状**上绝对值更好。
GlowTact 留一压头交叉验证(模型从未见过该形状): round 0.958→**0.993**, star 0.977→**0.991**,
quad_small 0.989→0.997, 全部 ≥0.986 —— 正是 Goal 6 遗留的两个短板, 且比结果页
"逐压头标定"(家族内拟合)是更强论断。代价: 形状已知时五特征 MAE 更优
(Sparsh 跨 pad 0.0372 vs 0.0443N)。物理律里的 g 换成 MLP 会崩溃(ρ -0.249):
isotonic 训练范围外单调截断, 网络自由外推。
教训: 一个看似决定性的新方法(PINN-lite)被"更笨的对照"击败——控制实验必须包含
"用更少特征"这一档, 否则会把正则化效应误读成物理发现。

---

# Ledger: marker inpainting shipped for geometry, force re-measured everywhere

## What changed in the pipeline
- **New** `force_recovery/marker_removal.py` — the adopted marker step.
  `marker_mask(ref)` caches a dilated dot mask per reference and returns
  `None` on a markerless gel; `stages_depth(img, ref)` wraps (never edits)
  `debug_gallery.stages()` by inpainting the dots out of BOTH reference and
  frame (cv2 Telea) before differencing. On a markerless gel it is a
  **bit-exact no-op** by construction.
- `marker_study.py` now imports `detect_markers` / `inpaint_img` /
  `dimple_power` from it, so there is one implementation of each.
- Depth/3D consumers switched to `stages_depth`: `showcase._panel` inputs
  (FEATS gallery stills get the marker path, cnc/React are no-ops),
  `showcase._lut_force_rows(keep_depth=True)`, the React depth panel.
  **Force features everywhere still come from the untouched `stages()`.**
- `recon_study.stages_full` gained an opt-in `inpaint_markers` flag, and
  `recon_study page` now GENERATES recon_workbench.html from
  `glowtact_diagnostics.json` instead of it being hand-written.
- **New** `force_recovery/force_eval_all.py` — one protocol, four datasets,
  each with a within-group label-shuffle control, plus `spotcheck` which
  recomputes cached cnc features from raw frames.
- `test_units.py`: `stages_depth` is asserted byte-identical to the studied
  `img_telea` variant on a real FEATS frame, and `marker_mask` is asserted
  `None` on a synthetic markerless gel.

## Refreshed force numbers (per-group half/half + isotonic, 5 seeds, median)
| dataset | n | rho | seeds | MAE [N] | within-group shuffle |
|---|---|---|---|---|---|
| GlowTact (markerless, 0-20 N) | 201 | 0.9864 | 0.981–0.987 | 0.525 | +0.171 |
| FoTa cnc_Mini (markerless, in view) | 337 | 0.9458 | 0.929–0.949 | 0.252 | +0.056 |
| FEATS (marker gel) | 186 | 0.7747 | 0.713–0.787 | 5.025 | -0.003 |
| Sparsh / Meta (markerless, Sparsh LUT, in view) | 1667 | 0.9682 | 0.967–0.971 | 0.042 | +0.264 |
| FEATS (marker gel, dots inpainted — rejected for force) | 186 | 0.7371 | 0.682–0.816 | 4.934 | -0.010 |

Regression check: `spotcheck` recomputed 40 cnc frames from the raw tars,
max |cached − fresh| over the 5 features = **0.000e+00**. The FEATS row
reproduces the marker study's published baseline (0.7747 / 5.025 N) exactly.

Numbers that MOVED, and why: results_page's scatter fit used one RNG
re-seeded identically for every group, a different split from the study
protocol. It now uses the median-rho seed of `force_eval_all`, so the matrix
and the refreshed table are one estimator. Displayed effect: cnc 0.94 → 0.95,
GlowTact 0.98 → 0.99, FEATS unchanged at 0.77. All index/results/method
numbers are now substituted from `force_matrix.json` at build time; the ZH
replacement pass runs BEFORE substitution so a metric can move without
un-translating a paragraph.

## Adopted
- **Marker inpainting, depth/3D only.** Lattice power at the 31.9 px pitch
  1.523 → 0.890 (×0.65), lower on 91% of 120 frames >1 N, Wilcoxon
  p = 2.6e-19. Detector: 63/63 dots, 0 rejects, stable over thresholds 3–16,
  0 blobs on both markerless references.

## Rejected, with their numbers
- Marker removal **for force**: 0.7747 → 0.7371 on identical frames/splits,
  every paired median delta negative. Not adopted.
- `grad_zero` (g := 0 on the holes): dimple 1.251 (×0.93) — a dipole layer on
  every hole boundary. `grad_inpaint` 0.924 is close but loses more rho.
- Random-mask control: 0.7697 (no gain) — the effect is not smoothing.
- Per-indenter light-press reference for FEATS: 0.7747 → 0.7261.
- Shadow-pixel dropping and interior-gradient zeroing: not validated,
  not adopted.

## Corrections propagated to the site (external per-pixel GT, `mnist_validation`)
- **RETRACTED**: "flat-topped indenters should reconstruct at centre/rim ≈1.0".
  Compliant gel wraps a flat edge, so c/r > 1 is expected. GT: true digit
  depth map c/r 1.400 (gel surface 1.334) vs our 1.539–1.562 (+10–12%);
  enclosed plateau control, truth exactly 1.000, we measure 1.069 (+7%).
  The workbench's old "1.23–1.42 is a defect" framing is gone.
- **DEMOTED**: unobserved LUT bins — 13.8%/16.9% of contact pixels,
  correlation with Type-2 error only 0.098/0.294.
- **CONFIRMED**: the |dI|>8 mask is halo-dominated — IoU 0.614, recall 0.917,
  over-segmentation 0.531.
- **RULED OUT**: the photometric table as the weak link — 24.4° off-domain vs
  26.1° on its own real sensor.
- **NEW HEADLINE**: accuracy is a function of press depth (MAE 11.2 / 35.0 /
  67.8 / 127.4 / 281.1 µm at 0.30 / 0.60 / 1.00 / 1.50 / 2.25 mm; peak ratio
  1.00 → 0.55). No accuracy number is quoted on the site without its press
  depth. Caveat stated: Taxim renders validate the geometry solver, not the
  sensor model, and Taxim's gel conforms to the object within ~38 µm.
- **CROSS-DATASET**: contact visibility is the biggest external failure mode
  (cnc field of view, Sparsh clipped discs, 420/420 MNIST touches clipped;
  mid-pad → edge control collapses peak 1.39 → 0.30 mm against 0.90 mm).

## Regenerated
20 gallery stills + 10 clips (`showcase`), debug_gallery samples,
results_page figures + matrix, `sparsh_figure`, recon workbench (20 panels,
generated HTML), index.html, method.html/method_zh.html,
debug_pipeline.html/_zh.html, gallery.html, and the new
`assets/feats_marker_removal.png` (raw | markers | inpainted | depth
before/after | mesh before/after) and `assets/mnist_examples.png`.

## Re-run entry points
```
python -m force_recovery.force_eval_all            # table + shuffle controls
python -m force_recovery.force_eval_all spotcheck  # stages() regression check
xvfb-run -a -s "-screen 0 1400x1000x24" python -m force_recovery.marker_removal figure
xvfb-run -a -s "-screen 0 1400x1000x24" python -m force_recovery.recon_study glowtact
python -m force_recovery.recon_study page
xvfb-run -a -s "-screen 0 1400x1000x24" python -m force_recovery.showcase images react videos
python -m force_recovery.results_page && python -m force_recovery.method_page
python -m force_recovery.debug_page && python -m force_recovery.site
python -m force_recovery.publish_space
```
