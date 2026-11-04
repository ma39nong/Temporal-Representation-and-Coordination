# FLASH Baseline Error Attribution — 首组真实结果

日期：2026-11-04。研究主线为Predictable and Coordination-Aware Action Generation for Bimanual Dexterous Manipulation。先定位表示误差、预测差异及时间结构，再决定是否改造Action Expert，不预设新方法有效。

## 本轮结论

**P0-B最低数值条件已满足，且已完成真实历史FLASH预测的误差归因。当前FullFT权重尚未落盘，未重新加载Checkpoint或运行当前FullFT推理。**

主结果采用历史冻结VLM Step250的101个独立验证来源、505个固定均匀窗口。相对它自己的历史395训练来源未见；相对当前all480训练，其中97个来源已进入训练。不能把它称作当前FullFT泛化结果。另在相同505窗口评价历史Step100，并保留813窗Step100补充结果，不混用两种采样均值。

这些都是原验证入口保存的真实模型输出，不是合成或插值生成的Prediction。重新核验prediction hash、plan/dataset/split/norm身份，并用原load_episode/make_batch逐窗读取Parquet，GT和8帧实测历史完全一致。原推理predict(mask,cache,history)没有未来标签输入；原source数学函数重新计算Oracle，最大偏差7.153e−7 rad。当前无可用历史参数目录，因此本次没有重新执行历史Checkpoint推理，保留这个限制。

## A/B/C：同一505窗口的物理动作误差

均在canonical58、rad、同一50步和有效掩码上池化。RMSE不能直接相加。

| 关节组 | E_repr RMSE | E_pred RMSE | E_total RMSE |
|---|---:|---:|---:|
| Left Arm 7D | 0.002952 | 0.141658 | 0.141682 |
| Left Hand 22D | 0.003724 | 0.068910 | 0.069009 |
| Right Arm 7D | 0.001370 | 0.098481 | 0.098483 |
| Right Hand 22D | 0.001783 | 0.057745 | 0.057753 |
| All Arms 14D | 0.002301 | 0.121995 | 0.122010 |
| All Hands 44D | 0.002919 | 0.063573 | 0.063631 |
| Full Action 58D | **0.002783** | **0.081599** | **0.081638** |

| Full Action | MAE | RMSE | P95绝对误差 | P99绝对误差 | Max |
|---|---:|---:|---:|---:|---:|
| E_repr | 0.000742 | 0.002783 | 0.003486 | 0.010505 | 0.239366 |
| E_pred | 0.049802 | 0.081599 | 0.171857 | 0.308370 | 1.412003 |
| E_total | 0.049817 | 0.081638 | 0.171848 | 0.308724 | 1.410364 |

预测差异RMSE约为表示RMSE的29.3倍，这支持**在该历史模型和该采样范围内优先调查预测过程**。不表示K6在所有接触瞬间都足够；Oracle仍有0.239rad的局部极值，不能用低均值掩盖尾部。

同一uniform505计划下，历史Step100的Full Action三项RMSE依次为0.002783、0.094893、0.094932 rad。两者Oracle完全相同，指标计算方式和GT一致；本轮不将其与当前FullFT训练步数作因果比较。813窗补充Step100的三项为0.006226、0.094261、0.094429 rad，采样覆盖不同，应单独解读。

### 非加性检查

r_repr=Oracle−GT，r_pred=Prediction−Oracle。Step250：

- MSE_repr = 0.0000077437122
- MSE_pred = 0.0066583770879
- 2 mean(r_repr × r_pred) = −0.0000013277122
- MSE_total = 0.0066647930878

按恒等式相加成立；不使用E_total=E_repr+E_pred，不把两个RMSE当作可直接相加的组成比例。

### 系数误差

C_gt由原fit_target生成，包含原训练监督的边界投影。旧预测档案没有C_pred张量，因此本轮对原54帧解码矩阵作线性逆，恢复scaled C_pred，并再用原reconstruct解码核对：Step100最大误差7.153e−7 rad，Step250为4.768e−7 rad。报告明确标为recovered coefficient，不能冒充直接导出的原模型内部张量。新native入口可直接保存C_pred，但尚待正式Checkpoint验证。

coefficient_by_degree.csv分别报告degree0…6，每组的scaled training、raw normalized和raw physical rad系数误差；不混合不同degree尺度作单一可解释指标。未用未来动作重新约束C_pred或预测轨迹。

## 图表与完整产物

主结果：outputs/flash_attribution/results_step250_505/；对照：results_step100_505/。每个目录包含：

1. Figure1 Representation vs Prediction Error：三项Full Action RMSE。
2. Figure2 Arm-Hand Group Error：四组逐窗口总MAE分布。
3. Figure3 Temporal Error Profile：双臂、双手、58D的50步逐时RMSE。
4. Figure4 High-Error Trajectory Cases：分别选择左右臂、左右手各自最大误差案例，显示GT、Oracle和Prediction。

此外有group_metrics.csv、temporal_metrics.csv、per_joint.csv、per_window.csv、worst_group_cases.csv、coefficient_by_degree.csv、joint_increment_profile.csv及完整report.json。图表标明REAL、历史Checkpoint与评价范围；基础测试的合成数组没有用于科研图表。输出默认不进Git。

export目录含原配置、数据/模型/统计/源提交身份、桥接脚本hash和结果seal。report保存本仓库commit和analysis文件hash。全部窗口可追溯到原sample、Episode、当前frame、source ID及最坏joint/tick。

## 初步时间与协调诊断

Step250总RMSE在tick0与tick49：Full Action为0.049142→0.117566 rad；双臂为0.070816→0.180715；双手为0.039845→0.088478。表明该采样下后段预测误差较大，但不是某一操作阶段失败的证据。

目标系数的边界投影只构造训练监督；原推理没有真实未来端点，不保证采样输出精确满足Oracle边界。这个源码事实与轨迹误差一起值得后续调查，不足以直接判定优化瓶颈。

各组最差总误差（Step250）：

| 分组 | 关节 | Episode/current frame | 预测tick | 最大误差rad |
|---|---|---|---:|---:|
| Left Arm | left_fr3_joint7 | 1 / 4579 | 45 | 1.410364 |
| Left Hand | left_ring_MCP_FE | 152 / 1011 | 49 | 0.957445 |
| Right Arm | right_fr3_joint7 | 159 / 626 | 49 | 0.649379 |
| Right Hand | right_index_MCP_FE | 4 / 33 | 43 | 0.763402 |

Episode编号仅在各自dataset root内有意义，完整source/sample标识在CSV，不能仅凭Episode号关联其他数据源。关节增量CSV输出GT/Prediction各组rad/s时间曲线和误差，供检查两侧及臂手时间关系；不据此构造未经核验的物理协调指标。

没有可靠阶段标注或本轮核验的URDF/FK，因此不输出稳定抓持阶段、抓握同步成功率、双末端相对位姿或任务失败率。**预测关节的时间偏差不等于真实机器人已协作失败。**

## 当前阻塞与下一步

1. 正式FullFT：19:23快照Step454、batch256、无params；当前计划8000不是已完成8000。等待保存实际Checkpoint后，用native配置接口加载匹配契约/统计，先跑同一训练内固定64窗口诊断；不得当泛化。
2. 当前独立来源：历史101中有4个source/20窗口不在all480 manifest。候选清单已保存outputs，下一步接当前统计和真实RGB/当前Checkpoint做独立推理，不重复划分训练源。
3. 历史结果只能支持预测差异远大于表示误差的局部判断，不能直接推出网络结构、K6阶数、归一化、优化器或数据噪声中哪个是根因。
4. 当前优先目标是复查高误差窗口的模型条件、历史和预测系数时间行为；尚无证据要求新增Mamba/Dual-Expert或把RTC当创新。

## 验证记录

原42项测试保持通过，本轮增加8项合成指标测试，共50项通过。新测试覆盖误差恒等式/抵消、真实分组索引、统一mask、50/54时间形状拒绝、分degree尺度、关节时间增量及拒绝合成科研报告/损坏seal。

真实一致性：formal480逐集数字验收；505窗Step100和505窗Step250的raw GT/history逐值相等；原Oracle和恢复C_pred的解码容差通过；原make_batch未来标签/未来state扰动后历史与当前state不变，targets改变。原40项跨分支对照是P0-A历史记录，本轮源分支已演进，不将其谎称为对当前e64c9a4重新完成的40项测试。

没有读取成功的正式FLASH参数Checkpoint，没有新推理或新训练，没有修改原工程或数据；已完成的是可复查的真实保存输出误差归因，以及等待正式权重的最小推理接入口。
