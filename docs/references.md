# 相关工作（2026-11-03核验）

以下仅依据原论文摘要/官方项目描述记录研究关系，不引用未经复现的实验数值。具体实现与公平比较需在P0-B阅读全文；这些文献不证明本项目假设。

| 工作 | 原始链接 | 与研究的关系 |
|---|---|---|
| FLASH: Efficient Visuomotor Policy via Sparse Sampling | [arXiv:2605.15492](https://arxiv.org/abs/2605.15492)、[官方代码](https://github.com/NTUMARS/FLASH-Policy) | Legendre系数表示、稀疏时间采样与历史系数初始化；本项目仅计划检验同预算表示容量，不复现其完整策略 |
| Coordi-π: Coordination-Aware Dual-Hand Action Representation for Bimanual Dexterous Manipulation | [arXiv:2610.19753](https://arxiv.org/abs/2610.19753) | 左右侧独立action token及交错序列、身份与时间编码；提供协调结构参照，不是异构多项式阶数的证据 |
| RTC: Real-Time Execution of Action Chunking Flow Policies | [论文](https://arxiv.org/abs/2506.07339)、[PI官方介绍](https://www.pi.website/research/real_time_chunking) | 异步生成/执行与跨chunk连续性；本阶段只检查数据边界，不运行RTC |
| MambaAct: Efficient Mamba Action Experts for Flow-Matching Vision-Language-Action Models | [arXiv:2610.22951](https://arxiv.org/abs/2610.22951) | 以Mamba动作专家和Context Compressor改进计算效率；与表示预算问题相关，但本阶段不替换专家 |
| T-Rex: Tactile-Reactive Dexterous Manipulation | [arXiv:2606.17055](https://arxiv.org/abs/2606.17055)、[项目](https://tactile-rex.github.io/) | 多速率MoT和时间触觉编码，启发非均匀时间需求；当前平台输入为RGB，不能假定具备该文触觉条件 |

T-Rex有同名论文：2506.19498为Task-Adaptive Spatial Representation Extraction，1904.06387为reward extrapolation，均非本表所指。此处暂按双臂灵巧/多速率语境选择2606.17055；如研究团队另有所指需更新引用。

P0-C补充（2026-11-05）：[ProAct / Beyond Reconstruction: What Matters in Action Tokenization for Robot Policies?](https://arxiv.org/abs/2610.09170) 将可预测性、鲁棒性作为action tokenization的研究问题。它是本项目“低重建误差不等于低策略误差”的相关工作，不能用其结论代替本平台实证。另有同名[ProAct / World-Calibrated Proposal-to-Action Flow](https://arxiv.org/abs/2610.02323)，不能混为同一论文。最小P1方案与这些工作的区别及待验证边界见 `p1_expert_design_options.md`。
