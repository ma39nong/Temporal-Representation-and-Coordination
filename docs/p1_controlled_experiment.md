# P1-0：B0/B1/B2受控实验与启动条件

状态：实现数学适配、单元测试和真实Batch前向；没有启动B2训练。本报告不把未训练B2的前向输出/loss作为性能结果。

## 三个对照的精确定义

| 对照 | 训练 | 推理坐标/输出 | 参考来源 | 本轮状态 |
|---|---|---|---|---|
| B0 | 当前原标准FLASH | 原Absolute系数、原物理绝对动作 | 原历史HAF | 已有4000/6000封印预测；同图新Batch与原API一致 |
| B1 | 不重新训练 | `P[t]+q0−P[0]`，固定平移 | 合法当前measured history[-1]与模型自身P[0] | 已对同一505窗做离线对照，无未来标签 |
| B2 | 需要匹配的重新训练 | 原拟合系数减C_anchor，Expert预测后加回再原解码 | 与B1相同的合法q0；不强制预测0步等于q0 | 适配层与原模型真实前向通过；未训练，未报告策略成绩 |

B1强制第0步对齐当前反馈，保留预测时间差分，不是B2。q0不是当前命令真值，未确定原始传感器采样/发布、图像、动作下发和控制周期的端到端延迟，因此本轮不能将B1用于机器人控制。若后续要实机测试，需独立核验q0采样时间、预测返回时间、Chunk首个实际执行时刻和旧命令处理；本轮不实现这些控制逻辑。

Step6000的B0/B1/重复q0 Full Action RMSE：训练来源0.011727/0.036567/0.055988，候选来源0.227965/0.065744/0.067696 rad。B1训练来源退化，候选成绩接近保守重复q0基线，不能把它当成已解决动作预测或协作问题。Paper T3混合Delta历史结果不进入这三者的因果对照。

## B2接口和训练数学

`FlashCoordinates`是固定的可逆数据坐标层，不复制Expert，不增加参数。配置中的 `mode=absolute` 返回原系数/动作，不改变原Absolute实现；`mode=state_relative` 只作用于显式兼容组。示例默认Absolute，不能由数据维数自动开启。

```python
from trc.models.flash_coordinates import FlashCoordinates
adapter = FlashCoordinates(
    config['coordinates']['mode'], action_q01, action_q99, original_polynomial_scale,
    tuple(config['coordinates']['compatible_groups']),
)
# 两个fit仍必须来自原FLASH实现
source_relative = adapter.coefficients(source_original, measured_history, xp=array_backend)
target_relative = adapter.coefficients(target_original, measured_history, xp=array_backend)
# 原FM/consistency使用同一source noise、tau与原VLM条件
# 原Expert推理只接收relative source/history/observation，没有target
absolute_predicted_coeff = adapter.coefficients(
    predicted_relative_coeff, measured_history, inverse=True, xp=array_backend,
)
# 继续原polynomial.reconstruct + 原denormalize
```

array_backend默认NumPy，真实前向显式传原环境的jax.numpy。基础包不依赖JAX。history固定8帧，系数固定7×58；真实模型输出50×58。相对物理动作采用 `(A−q0)`，归一化只用原训练action Quantile的斜率；具体公式与原ridge历史保留原因见假设文档，禁止直接用F(A−q0)或重新拟合相对history替代此协议。

设原source为S、target为T、噪声η、理想常量系数为Q：`S'=S−Q`、`T'=T−Q`。同一噪声和时间下 `z'_tau=z_tau−Q`，速度目标 `T'−(S'+η)=T−(S+η)`；原FM和consistency的目标尺度/权重不变，网络输入坐标改变。训练插值含监督target是原Flow Matching训练定义；推理函数不接收target或未来端点。未改变target的54帧边界投影、原history重参数化网格、50步评价范围或K6容量。

## 公平比较必须固定的内容

1. B0/B2使用同一π₀.₅ VLM、同一原FLASH初始化流程、同一随机seed和初始化后逐叶相同的参数。正式对照都从相同pi05_base和新优化器开始；不能一个用Step6000、另一个从base开始。Step6000仅用于本轮功能检查。
2. 相同all480数据版本、原recording身份、epoch_shuffle、合法窗口、prompt、三路相机预处理及action/state统计hash。原action保持Absolute58D。B2没有从候选来源拟合新的mean/std/Quantile；额外诊断用的分布统计也不进入训练。
3. 相同K6/每关节7系数/406总系数、7个58D token、参数量、Flow/consistency/source noise、NFE=1、优化器、学习率、batch、训练步数及参数更新范围。保持实际batch配置，不能由名称含b512就断言实跑512；当前FullFT实际global batch256。
4. 相同推理运行时、精度、执行图结构、单窗/Batch协议、缓存和解码逻辑；记录源Commit、配置、权重内容hash、统计、prepared observation身份。B0和B2的报告都在同一原物理动作坐标和统一valid mask下计算。
5. 不通过给B2更高阶、多token、更多独立训练数据、不同冻结策略或不同训练预算制造优势。若以后做arms-only等分组消融，需要事先登记，与全58D坐标方案区分。

## 数值复现发现与正式训练前的门槛

同一次前向中，Absolute坐标路径与原sample_from_cache的最大差为0.0 rad；坐标往返的50步解码差约2.38e−7 rad。它们是适配正确性的证据。

**但重新加载同一权重的新执行图/进程没有逐位复现历史P0-C预测。** 两个真实窗（0与173）在本轮原单窗图重放相对历史封印输出的最大差分别约0.00753/0.00918 rad，RMSE约0.00101/0.00243 rad；同进程单窗图与两分支Batch图的最大差约0.00896 rad。原始RGB哈希、GT/history、源Commit、统计/配置契约、参数内容hash均已核对相同；同图原API仍完全一致。原模型使用低精度计算，但本轮尚未隔离运行时、编译图与跨进程数值差异各自的贡献，**不能直接宣布差异由Batch大小或BF16导致**。额外的原P0评估脚本原样单窗重放结果另保留，不能用这些小样本估计完整噪声分布。

这不改变对已封印4000/6000数组的物理偏移分解，也远小于候选来源约0.228 rad的整体问题；但它与训练来源约0.012 rad的误差相比不能忽略，可能干扰小幅算法收益的解释。

进一步原样运行P0-C评估脚本、使用原启动包装器和同一个window0，得到相对旧预测最大差0.004555 rad、RMSE 0.000906 rad，参数内容身份一致。因此差异并非只有新坐标适配器才出现，仍未定位根因。该单窗回放仅取预测数组做数值核验：旧脚本的固定混合505范围标签和单窗“排除编译后耗时”不适用于此回放，不引用它们作为新科研统计。

在正式P1前，应固定prepared observation、原运行时启动环境和同一推理图/Batch协议，重复B0并量化source级误差的数值噪声，记录编译/精度设置。若仍无法得到可接受的复现范围，应先解决基线而不是跑大型B2。不得把简单放宽allclose阈值写成“已解决复现”。目前保留这一门槛，未宣称正式P1评估已完全就绪。

## 评价协议与停止规则

训练来源485窗用于监测，4候选来源20窗是发现性探针；它们已经用于选择本方案，不能再包装成完全未见的确认集。补充按原始recording/采集会话隔离、内容身份可追溯的新来源，保留不参与设计的新确认集。当前候选缺少原始bag内容链，补证据时不修改数据本身。

至少报告四组/Full Action的MAE、RMSE、P95/P99、Max，50步误差、每source/窗口误差，物理偏移/形状分解及系数到动作的贡献。并列B1、重复q0，检查动态轨迹是否退化成保守静止。相邻时间差分误差可用于运动形状诊断；未核验FK、接触事件和实机结果时不报告物理协调成功或失败。

建议在数值门槛闭合后先做匹配的单seed、短预算探索试验，具体步数和batch在运行前固定并写入配置；同一预算跑B0/B2，不把短试验称正式RA-L验证。得到稳定信号后再开展多seed和独立会话实验。主指标/实用改善阈值应在看新确认集前预注册，并显著大于B0的数值重放差异；先前提出的5%仅是候选实用阈值，当前不能忽略复现噪声就直接采用。

证伪情形：训练后新来源上没有稳定改善；只改善已用于选型的4来源；收益来自复制q0而形状/远期/某组P99明显退化；公平预算下不优于B0；或经评价修正/完整Step8000验证后原偏移已不再存在。出现这些情况应停止扩展相对Expert，返回数据条件/优化或视觉状态表示的诊断。

**当前建议：不直接启动正式大规模P1 FullFT。** 物理证据与数学/真实Batch验收已支持最小适配器，但训练收益未知、确认集不足，并发现需先收敛的基线数值复现问题。下一步优先闭合复现门槛，再决定小预算B0/B2受控训练。本轮没有自动启动任何训练。

## 可运行入口与产物

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/diagnose_p1_offsets.py \
  --exports outputs/p0c_fullft/export_step4000 outputs/p0c_fullft/export \
  --audit outputs/p1_state_relative/source_audit.json \
  --stats /path/to/matching/norm_stats.json \
  --output outputs/p1_state_relative/new_offset_analysis
```

`audit_p1_sources.py`与`check_state_relative_source.py`是显式原环境工具，参数来自对应 `*.example.json` 模板，不能在纯NumPy环境宣称已运行模型。研究包wheel仅约40KB，解包到原工程之外的研究runtime目录；使用原环境解释器与显式配置的源目录，保留原启动环境的库设置。没有向原共享环境安装包、复制整个OpenPI或把源目录写入普通研究包。

实际原环境执行配置和日志、Step6000真实前向报告、RGB身份、原样单窗重放以及数值差异保留在 `outputs/p1_state_relative/`；结果默认不提交Git。核心适配器、脚本和测试正常保存在本研究仓库。新报告的代码/输入/统计hash与最终Commit进入delivery manifest，原模型参数身份为 `89da5fe135d6ae92c74f4f59634ff0c35bbf28301b85ffe5ce91bc3776931134`。
