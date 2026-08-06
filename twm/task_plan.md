
## Sparsh 上站 (第四个数据集) — 迁移上限而非重建结果
figure: twm/force_recovery/sparsh_figure.py -> site/assets/results_sparsh.png
(三面板: 逐 pad 标定散点 | pad 内打乱对照 | 跨 pad 迁移矩阵)
数字: rho 0.558 / MAE 0.138N / n=18750, 对照 0.161(有效对照, 保留各 pad 力程);
跨 pad 迁移 非对角 0.418-0.591 vs 对角 0.471-0.587 -> 一套标定可跨 pad。
**站点必须写明的警告**: GlowTact 标定的 LUT 在该传感器上重建无效
(球压→双瓣+中心凹陷, 打光几何不同), 相关性追踪 dI 幅值而非深度。
EN+ZH 各一节, 含三个数据集缺陷说明。已发布。
