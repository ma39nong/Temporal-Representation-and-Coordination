# P1-0：物理动作空间偏移分解

日期：2026-11-05。仅复用P0-C封印的真实Step4000/6000导出，没有重跑505窗归因或重新抽样。全部窗口mask有效；每窗50步、58个物理关节命令，单位rad。训练来源485窗与候选4来源20窗分别统计；候选来源身份与限制见 `p1_state_relative_hypothesis.md`。

## 定义与可解释范围

每窗n、关节j，`e[t]=A_pred[t]−A_gt[t]`，`b=mean_t(e[t])`，`r[t]=e[t]−b`。由于 `mean_t(r)=0`，有 `mean_t(e²)=b²+mean_t(r²)`。整体偏移MSE=`mean_n,j(b²)`；形状MSE=`mean_n,t,j(r²)`；比例为先池化MSE后的比值，不是逐窗比值的均值。无误差时比例标记为空，而非伪造0或1。

b使用全部未来GT，属于Oracle事后分解，不能用于预测校正。它只剔除了时间常量偏移，并不保证残差具有任务正确的形状、动力学或协调意义。所有关节的b、r所对应的MSE、来源与窗口标识均可追溯；窗口重叠不能当独立试验。

## 总体结果

| Step | 范围 | 总RMSE | 偏移RMSE | 形状RMSE | 偏移MSE占比 |
|---|---|---:|---:|---:|---:|
| 4000 | training_sources | 0.014828 | 0.011372 | 0.009515 | 58.819% |
| 4000 | candidate_unseen_sources | 0.228767 | 0.227351 | 0.025415 | 98.766% |
| 6000 | training_sources | 0.011727 | 0.008505 | 0.008073 | 52.607% |
| 6000 | candidate_unseen_sources | 0.227965 | 0.226590 | 0.024995 | 98.798% |

Step6000候选来源约98.798%的最终MSE是整体偏移，训练来源约52.607%。4000→6000没有消除候选来源这一误差结构。不能把低形状残差解释成原轨迹已经可安全执行。

## Step6000四组结果

| 范围 | 组 | 总RMSE | 偏移RMSE | 形状RMSE | 偏移占比 |
|---|---|---:|---:|---:|---:|
| training_sources | left_arm | 0.018027 | 0.013874 | 0.011510 | 59.23% |
| training_sources | left_hand | 0.011561 | 0.007306 | 0.008960 | 39.93% |
| training_sources | right_arm | 0.013371 | 0.011646 | 0.006569 | 75.86% |
| training_sources | right_hand | 0.008281 | 0.005739 | 0.005970 | 48.02% |
| candidate_unseen_sources | left_arm | 0.355791 | 0.351479 | 0.055228 | 97.59% |
| candidate_unseen_sources | left_hand | 0.250673 | 0.250060 | 0.017526 | 99.51% |
| candidate_unseen_sources | right_arm | 0.219470 | 0.219246 | 0.009924 | 99.80% |
| candidate_unseen_sources | right_hand | 0.136255 | 0.135009 | 0.018385 | 98.18% |

## 四个候选来源逐组结果（Step6000）

下表ID缩写对应假设文档的完整ID。每个来源5窗。Step4000的相同逐来源、逐组结果也已写入CSV/JSON，不用20窗平均替代来源报告。

| Source前8位 | 组 | 总RMSE | 偏移RMSE | 形状RMSE | 偏移占比 |
|---|---|---:|---:|---:|---:|
| 0340d869 | left_arm | 0.397106 | 0.395180 | 0.039058 | 99.03% |
| 0340d869 | left_hand | 0.222184 | 0.221430 | 0.018282 | 99.32% |
| 0340d869 | right_arm | 0.216321 | 0.216088 | 0.010054 | 99.78% |
| 0340d869 | right_hand | 0.128748 | 0.127192 | 0.019961 | 97.60% |
| 0340d869 | full_action | 0.222917 | 0.221851 | 0.021777 | 99.05% |
| 435fc6da | left_arm | 0.406154 | 0.397159 | 0.085004 | 95.62% |
| 435fc6da | left_hand | 0.288726 | 0.288487 | 0.011725 | 99.84% |
| 435fc6da | right_arm | 0.215321 | 0.215150 | 0.008579 | 99.84% |
| 435fc6da | right_hand | 0.162066 | 0.161055 | 0.018072 | 98.76% |
| 435fc6da | full_action | 0.259013 | 0.256964 | 0.032511 | 98.42% |
| b710689f | left_arm | 0.170698 | 0.166670 | 0.036863 | 95.34% |
| b710689f | left_hand | 0.196186 | 0.194591 | 0.024967 | 98.38% |
| b710689f | right_arm | 0.182744 | 0.182499 | 0.009459 | 99.73% |
| b710689f | right_hand | 0.102822 | 0.099853 | 0.024529 | 94.31% |
| b710689f | full_action | 0.161730 | 0.159741 | 0.025288 | 97.56% |
| e4bdeafd | left_arm | 0.393139 | 0.390471 | 0.045722 | 98.65% |
| e4bdeafd | left_hand | 0.283076 | 0.282840 | 0.011558 | 99.83% |
| e4bdeafd | right_arm | 0.257129 | 0.256877 | 0.011391 | 99.80% |
| e4bdeafd | right_hand | 0.144388 | 0.144300 | 0.005033 | 99.88% |
| e4bdeafd | full_action | 0.254825 | 0.254180 | 0.018117 | 99.49% |

## 系数0阶与物理偏移是否对应？

P0-C的98.75%是 `mean(e0 * sum_k ek)` 相对于Prediction−Oracle MSE的signed allocation，保留跨阶干涉。这里98.798%是最终Prediction−GT误差在时间常量子空间的正交投影占比。两者的误差对象、投影和分母不同，不能画等号。

在物理坐标计算0阶分量 `d0 = mean(B[:,0])*s0*(C_pred[0]−C_gt[0])*(q99−q01+eps)/2`，再与本轮的b逐窗逐关节比较：

| Step6000范围 | b的RMSE | d0的RMSE | b−d0的RMSE | 池化相关系数 |
|---|---:|---:|---:|---:|
| training_sources | 0.008505 | 0.008469 | 0.000619 | 0.99731612 |
| candidate_unseen_sources | 0.226590 | 0.226444 | 0.001011 | 0.99999026 |

这构成了零阶偏移与物理整体偏移高度对应的直接证据，但不是恒等关系：高阶基在50步截取网格上的均值不必为0，且b包含Oracle−GT。不能用这份高相关性证明某种网络改造有效。

## q0、GT[0]和Prediction[0]

q0严格取8帧measured history的最后一项，等于当前实测state；不从未来command构造参考。下表为Step6000 Full Action RMSE（rad）；各组和各source的MAE/P95/P99/Max同样输出。

| 范围 | GT0−q0 | Prediction0−q0 | Prediction0−GT0 |
|---|---:|---:|---:|
| training_sources | 0.033270 | 0.036016 | 0.012012 |
| candidate_unseen_sources | 0.037831 | 0.220468 | 0.220625 |

q0是同名位置关节的合法坐标原点，不是命令第一步的真值。参考可逆并不要求q0=A_gt[0]。

## B1：不用未来标签的简单校正

定义 `A_B1[t]=A_pred[t]+q0−A_pred[0]`。固定平移使用合法q0与模型自己的第0步输出，不用GT；它强制第0步等于q0、保持预测的时间差分。仅作离线诊断，未核验控制延迟或实机适用性。

| Step6000范围 | B0原预测RMSE | B1校正RMSE | 重复q0基线RMSE |
|---|---:|---:|---:|
| training_sources | 0.011727 | 0.036567 | 0.055988 |
| candidate_unseen_sources | 0.227965 | 0.065744 | 0.067696 |

B1在候选来源改善，但在训练来源显著退化；候选来源的成绩也仅略优于重复当前状态。这反驳“只要强制第一步对齐当前状态就普遍更好”，尚未检验需要训练的B2相对坐标。

## 输出与复现

`outputs/p1_state_relative/offset_final/` 保存offset_groups、offset_windows、offset_per_joint_window、coefficient0_vs_physical_bias、q0_gt0_pred0、B0_B1_controls、episode_initial_distribution和window_reference_distribution的CSV及封印输入/代码hash。Figure1为分组物理MSE堆叠，Figure2为B0/B1/重复q0对照，均明确标REAL。`offset_components_step4000.npz`、`offset_components_step6000.npz`保存每窗每关节的b、完整r及三种MSE，窗口顺序严格对应封印导出；报告记录这些数组文件hash。

```bash
.venv/bin/python scripts/diagnose_p1_offsets.py \
  --exports outputs/p0c_fullft/export_step4000 outputs/p0c_fullft/export \
  --audit outputs/p1_state_relative/source_audit.json \
  --stats /path/to/matching/norm_stats.json \
  --output outputs/p1_state_relative/new_offset_analysis
```

结论：候选来源确实以整体位置偏移为主，值得验证可逆状态相对坐标；既有B1反例要求保留真实command0残差，不能把q0硬设为策略第一步。
