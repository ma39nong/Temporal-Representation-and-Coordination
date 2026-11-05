# P1：由真实诊断支持的最小算法实验

状态：设计与接口调查，未实现新算法、未修改原训练仓库、未启动训练。当前只保留 **一个** 值得先做的候选方案。Step8000未落盘，Step6000不是收敛证明；方案依据及范围见 `p0c_fullft_diagnosis.md`。

## 为什么只保留一个方案

训练来源 Step4000→6000 的最终 RMSE 从0.014828降至0.011727 rad，候选独立4来源则从0.228767变为0.227965。后者0阶物理误差的 signed allocation 约占预测MSE的98.75%，第0步已有0.220625 rad误差；固定K6的Oracle仍只有0.002383 rad。它们还表现出更高的训练分位范围外输入比例。

因此，优先测试“降低绝对姿态对系数条件学习的干扰”这一具体假设。没有证据要求增加多项式阶数、分割阶段、增加Mamba/双Expert或专门Coordination模块。下面方案可能无效，也尚不足以构成RA-L创新；需先确认Step8000并通过公平消融，不能把坐标改写本身包装为已验证方法。

## 唯一候选：在合法当前 measured state 周围平移系数输入坐标

### 对应的假设

原FLASH **已经** 用历史拟合初始化系数并预测残差速度，不能声称首次提出 History Anchoring。现有原模型仍将含绝对姿态的七个系数token送进Expert。候选假设是：保持原HAF的source/velocity监督与表示容量，只给Expert提供相对当前姿态的系数坐标，可以减少候选来源上的0阶偏移。

这不保证有效：绝对姿态对视觉/动作映射可能有用，移除它可能损害任务。state仍按原流程进入VLM，但离散state token不保证足够表达精确姿态。该方案不修复未知归一化因果问题，不承诺物理协调改善。

### 数学机制与边界

设原action归一化为 `F(A)=2(A−q01)/(q99−q01+eps)−1`，原解码为 `A=F⁻¹(B diag(s) C)`。`C∈R^(7×58)` 是现有训练scaled系数，K6每关节7系数，总预算406。合法当前 measured state 为 `q_t=history[-1]`，定义理想常量系数

`C_a[k,j] = 1[k=0] · F(q_t)[j] / s[0]`。

注意这里使用 **action** 的固定Quantile参数作用于measured state；observation自身原state归一化与tokenization保持不变。不从候选评价来源重估统计，不用A[0]代替q_t。

先严格使用原实现得到 `C_h=FitHistory(F(history))`、`C_gt=FitTarget(F(target_ext))`，再做

`C_h' = C_h − C_a`，`C_gt' = C_gt − C_a`。

原历史拟合有ridge项，不严格保持常量。**不能**未经验证地用 `FitHistory(F(history)−F(q_t))` 替换 `C_h−C_a`；否则连原HAF source也改变，无法将实验归因于输入坐标。

训练时以相同噪声 `η`（若原配置启用）和时间 `τ` 构造

`z'_τ = (1−τ)(C_h+η−C_a) + τ(C_gt−C_a)`，

`v_target = C_gt − (C_h+η)`。

原速度目标不因平移改变。原flow matching MSE和one-step consistency MSE形式、系数缩放、权重均保持。改变的是Expert接收到的系数坐标，VLM条件与time embedding不变。推理从 `C_h'` 沿原Euler/NFE=1得到 `C_pred'`，加回同一个合法当前anchor再用原解码：

`A_pred = F⁻¹(B diag(s) (C_pred' + C_a))`。

转换可逆，Oracle物理轨迹与406系数预算不变，没有额外模型参数，没有新的Flow source分布或未来条件。不能通过新坐标复用旧Checkpoint直接宣称新算法性能：旧网络没有在此坐标下训练，必须进行受控重训消融。

不约束预测第0步等于 q_t。真实command0与measured state0的差异已有0.033/0.038 rad；强制相等会混淆两种信号。C_a来自measured feedback，不是Oracle；P0-C中用C_gt替换0阶的成绩也不作为此方案预期成绩。

### 原代码最小修改点清单（后续实施）

以只读核验的 `paper-flash-test@d208d414efbb78483f44bfef4c2a9314115dd632` 为定位基准。本表不授权或执行对原仓库的修改；P1应在用户指定的独立实验代码中实施，复用原实现并记录差异。

| 位置 | 当前真实行为 | 候选最小变化 |
|---|---|---|
| `src/openpi/models/pi0_flash.py::coefficient_velocity` | 7×58 coefficient tokens 经58→1024投影；时间条件经MLP和AdaRMS；共享Expert解码回7×58速度 | 函数结构/参数不变，由调用处传平移后的token |
| 同文件 `loss_from_cache` | history拟合source、54帧拟合target、source noise、插值FM和单步consistency | 在原fit结果和相同噪声基础上减同一C_a；保持目标速度、loss权重和随机数 |
| 同文件 `sample_from_cache` | history source + 原系数速度Euler积分，原normalize/reconstruct/denormalize | source减C_a，积分后加回C_a；保持NFE和解码 |
| 同文件 `encode_condition`；`pi0.py::embed_prefix` | π₀.₅图像/语言/state prefix及原逐层KV条件 | 不变 |
| `src/openpi/shared/flash_polynomial.py` | K6基、scale、ridge历史拟合、训练target边界投影和50步重建 | 不变；写数值不变性检查，不重写拟合 |
| `src/openpi/training/flash_native.py` | 原数据索引、三相机预处理、合法history/target和固定归一化身份 | 不变；anchor只从已提供history取 |
| 模型配置与checkpoint契约 | 明确K6/Absolute/FullFT/源数据和统计 | 若P1获准，新增默认关闭的坐标模式字段并保存其身份；本轮不加开关代码 |
| 研究评估导出 | 保存原C_pred及物理动作 | 同时记录内部C_pred'和加回anchor的C_pred；跨方法指标统一在原物理动作/原系数坐标评价 |

当前58D通过同一线性层进入每个系数token，原代码没有显式四组Expert。四组布局为Left Arm0:7、Left Hand7:29、Right Arm29:36、Right Hand36:58；继续用于评价，不因为分组误差不同就新增四路网络。视觉条件、系数生成、FM loss、动作解码及分组位置均已定位；没有需要在P0-C补建的模型框架。

### 与已有方法的关系

- [FLASH原论文](https://arxiv.org/abs/2605.15492)与[官方实现](https://github.com/NTUMARS/FLASH-Policy)：保留其Legendre表示与History-Anchored Flow。候选只改变条件网络输入的坐标原点，不是新的多项式表示或新的历史初始化方法。是否已有等价做法需P1前阅读全文及更完整检索，当前不作新颖性保证。
- [Coordi-π](https://arxiv.org/abs/2610.19753)：关注左右token组织、手身份与时间编码；本候选不改变左右token序列、注意力或身份编码，不能称为Coordi-π式协调机制。
- 本文所指 [ProAct / Beyond Reconstruction: What Matters in Action Tokenization for Robot Policies?](https://arxiv.org/abs/2610.09170)：讨论tokenization的可预测性及鲁棒性。候选保留固定连续K6并做可逆坐标平移，不学习离散tokenizer，也不搬用该论文实验结论证明本平台收益。

另有同名 [ProAct / World-Calibrated Proposal-to-Action Flow](https://arxiv.org/abs/2610.02323)，涉及proposal/world校准，不能混用标题或贡献。本候选没有proposal expert/world expert。以上只说明研究关系，不是完整相关工作或已确定的创新区别。

### 公平消融与可证伪判据

后续训练需另行授权。本轮仅定义协议：

1. 原FLASH与坐标平移版本使用相同π₀.₅ backbone、初始权重、数据来源、合法窗口、任务文本、图像预处理和固定统计；相同全量可训练参数、K6/406系数、7token、历史/监督/评价索引、source噪声、NFE=1、batch/优化器/学习率和更新预算。记录参数量与实际计算/显存，不把不同FullFT/冻结或不同数据结果当算法消融。
2. 先做训练前不变性验收：给任意同一C，减/加C_a后原物理解码一致；target-source和一致性残差代数一致；同一source noise下不改变目标；第0步不强制复制measured state；修改未来标签不影响合法输入。不能用这些代数测试代替模型收益实验。
3. 训练来源固定窗口做诊断，现有4来源20窗只做发现性探针；增加按recording/会话隔离且未用于选型的独立来源后再做确认性评价。重叠窗口不独立，置信区间按source/会话cluster重采样；4来源不足以稳定支撑统计结论。建议匹配3个seed，预算不足时如实注明探索性单seed。
4. 主指标为独立来源Full Action物理RMSE，伴随四组RMSE/P99、时刻0/远期曲线、每阶物理误差分量、source级误差和constant measured-state基线。观察是否只是保守重复当前状态导致轨迹幅度损失；同时评价相邻动作增量误差。没有可靠FK/接触事件仍不使用物理协调成功率。
5. 在看新确认集结果前固定有效改善阈值（例如至少5%的相对RMSE，作为待团队预注册的实用阈值，不是目前已承诺的统计阈值）与不可接受的组别P99退化。若新来源/多seed无稳定改善、只改训练集、损害动作变化/臂手尾部、或收益可完全由不公平数据/预算解释，则否定这一方案。若Step8000或进一步输入核查已消除偏移，则降低或取消P1改造优先级。

此候选的价值是以几乎不增加混杂因素的实验检验条件坐标问题；不会因为用了“相对”就自动解决双臂协调。若失败，先返回观察证据定位条件分布、优化或数据问题，不继续拼接新架构。
