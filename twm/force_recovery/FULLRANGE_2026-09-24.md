# 0–15 N 力模型重拟合（2026-09-24）

代码：`fullrange_search.py`（标定集搜索）、`fullrange_react.py`（React 部署抽样）。
产物：`$REACT_FORCE_RECOVERY_ROOT/feature_cache/fullrange_search_2026-09-24/`
（`final_report.json`、`candidate.joblib`、`heldout_model.joblib`、`react_sample.joblib`）。
**生产没有改动**：`react_calib` / `PIPELINE_VERSION = 8` / 已发布的力列都维持原样。

## 结论

| 同一批 283 个 held-out 按压（v8 的位置切分） | v8 | 新模型 |
|---|---|---|
| MAE | 1.975 N | **0.279 N** |
| 95% 位置 bootstrap CI | [1.53, 2.46] | [0.19, 0.38] |
| RMSE | 3.20 N | 0.58 N |
| Spearman ρ | 0.717 | 0.991 |
| v8 够不着的 57 个高载样本 | 5.94 N | 0.73 N |
| 0–1 / 1–4 / 4–8 / 8–12 / 12–15 N | 0.44 / 1.08 / 1.12 / 2.79 / 3.60 | 0.06 / 0.12 / 0.23 / 0.34 / 0.51 |

配对 bootstrap 的改进：1.70 N，95% CI [1.33, 2.10]。

新模型的定义：RBF 核岭回归（α=1e-3，γ=0.3/d），目标是 √F；特征 135 维，
**全部与位置无关**：深度统计（不含 8×6 深度图）+ 图像差分的通道分位数（不含 8×6 图像图）+ cx,cy；
训练数据为 round 的训练位置加上另外 5 种压头（quad、star、triangle、B、quad_small）。

## 选择流程（防止自欺）

- 超参数和特征集**只用 598 个训练行**选，方法是 6 块空间 block CV（KMeans 按 x,y 分块）；
  240 个配置，选中者的 CV MAE 为 0.614 N。283 个测试样本只在最后评估一次。
- 探索阶段我在测试集上看过几个变体（0.49、0.42 N），但最终选择由训练集上的 block CV 决定，
  而且最终结果比那几个都好，不是挑出来的。

## 必须知道的三个限制

1. **v8 的切分偏乐观。** 测试位置离最近的训练位置中位数只有 0.57 mm。把整片区域
   （约 6 mm）拿出去时，新模型在训练集上 block CV 为 0.61 N，而不是 0.28 N。
   同一 sensor 上标定覆盖了整个 pad，因此 0.28 N 对应的是“同一传感器、已覆盖区域”的场景。
2. **换压头形状仍然有误差。** 留一形状（leave-one-shape-out）：quad 1.16（v8 2.47）、
   triangle 1.05（2.24）、B 0.74（1.70）、quad_small 0.80（2.55）；**star 4.00（v8 3.69），
   所有模型在 star 上都失败**，原因未查。
3. **不能部署到 React。** React 的 calibration-free 深度比所有 CNC 形状小 7–10 倍
   （maxd 中位数 1.7 对 11–20；maxd/√area 0.47 对 1.5–3.2），而接触面积相当。这是传感器之间的
   光度尺度差（React gel 色相约 85°，Mini 约 170°），不是物体形状造成的。在 1894 帧 React 抽样上
   （16 个 sensor-side，四个任务，v8 重算与已发布 npz 完全一致）：
   - 100% 的接触帧落在 CNC 训练集 kNN 距离的 p95 之外；
   - 新模型会把 9–18% 的接触帧推到 15 N，中位数是 v8 的 2.5–3 倍。
   v8 在 React 上的绝对力同样**没有验证**（`absolute_force_validated_on_react: False`），
   只是它低维、单调，外推比较温和。要让任何模型在 React 上有可信的牛顿值，需要在 React
   的 gel 上采一批带力标签的按压（CNC 或放一个力传感器），标定集本身无法解决这个问题。

## 试过并放弃的变体（数字都在 `final_report.json` / `cv_search.json`）

训练集 6 块空间 block CV，同一流程、同一分块（除特别注明）：

| 变体 | block CV MAE | 结论 |
|---|---|---|
| **选中：inv135 + 多形状训练** | **0.614 N** | 部署候选（仅限标定传感器） |
| inv133（去掉 cx,cy）+ 多形状 | 0.626 N | 差别在噪声内 |
| inv135，只用 round 训练 | 0.810 N | 多形状训练更好 |
| 只用 25 维深度统计（geo25）+ 多形状 | 1.202 N | 图像分位数特征贡献最大 |
| 759 维全部特征（含 8×6 深度图和图像图）+ 多形状 | 1.760 N | 放弃：学到位置外观；在 v8 切分上却有 0.49 N，被切分掩盖 |
| 759 维，只用 round | 1.946 N | 同上 |
| v8 风格：7 个标量线性打分 + isotonic，全量程重拟合（全 round 的 6 块 CV） | 2.82 N | 放弃：标量特征信息不够 |

√F 目标变换只在 120 对配置中的 70 对更好，属于小优势，不是决定性因素。

## 复现

```bash
OPENBLAS_NUM_THREADS=8 python -c "from twm.force_recovery import fullrange_search as S; S.check_v8_from_basic(); S.final()"
OPENBLAS_NUM_THREADS=4 python -m twm.force_recovery.fullrange_react   # React 抽样，约 15 分钟
```
依赖 `bounded_force_0_15_2026-09-16/*_v1_di4.joblib` 特征缓存（不在 git）。
本机 sklearn 1.1.1 + 新 SciPy 下 `Ridge(solver='cholesky')` 和 `KernelRidge` 都会报
`sym_pos` 错误，所以代码里自带了一个 `KRR`，Ridge 用 `solver="svd"`。
