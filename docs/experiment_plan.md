# P0-B / P0-C 可执行实验计划（仅规划）

## 前置门槛（P0-B）

1. 固定主训练仓库2ab68c8…及LeRobot版本/锁文件；记录数据manifest、各目录指纹、episode到原始recording/片段映射和统计SHA。提供311/480及连续多颗数据的只读挂载配置，不能用本地早期样本充当正式训练集。
2. 接入主仓库load_bundle与norm_provenance验证适配器，核验实际T3的50×58统计、50×14 arm统计和floor。固定同一输入、同一anchor、同一窗口逐数值比较原始action/训练target/解码物理命令。无统计时可以研究明确已核验的原始物理命令，但不能称训练目标逆变换验证完成。
3. 在固定LeRobot环境验证第一/中间/末尾采样点、短episode、跨episode、不同action_hz的最近邻/padding/掩码行为，比较Batch和offset。现有轻量数值对照不代替此步骤。
4. 核验有效帧定义、源时间映射、摄像同步与清洗边界；无法恢复来源的片段单独标记/排除主结论。定义stage：approach/contact/grasp/detach/transfer等由标注规则决定，允许缺失，不预设所有任务都有所有阶段。
5. 以原始recording/采集session分train/validation/test，冻结随机种子和文件清单；全部重叠窗口及其裁剪来源只能落入同一split。连续多颗数据另设分布外测试集。

## E0–E4（P0-C）

共同输入接口：action_schema核验后的[time,58]物理命令、anchor实测state、timestamp、episode/source/segment/stage ID、valid mask以及统计身份。使用连续完整窗口，时间映射到[-1,1]；优先NumPy Legendre least-squares，不训练VLA。有效帧策略未知时阻塞正式实验。

| 实验 | 表示与选择方法 | 系数预算 |
|---|---|---|
| E0 | 统一六阶Legendre、一个连续窗口 | 58×(6+1)=406 |
| E1 | 统一高阶，建议degree=8,10,12；验证集决定候选范围 | 各为522,638,754；作为容量曲线，不能直接与406预算E0比优劣 |
| E2 | 四组不同degree，各组内统一；用train/validation误差分配预算 | B=7(dLA+1)+22(dLH+1)+7(dRA+1)+22(dRH+1)，主比较B=406及E1对应预算 |
| E3 | 按操作阶段分段，阶段内统一阶数 | B=Σs58(ds+1)，约束边界/连续性项单独计；只报告实际可行预算 |
| E4 | 阶段×关节组联合分配degree | B=Σs,g Dg(dsg+1)，含所有分段、边界和自适应选择开销 |

预算不可行时：禁止先给每段六阶再与单段E0声称同容量。按精确整数预算搜索；若不能精确相等，比较相同预算上限且报告实际使用量，并给最近均匀预算双侧基线/误差-预算Pareto曲线。约束消除的自由度同时报告“存储系数数”和“独立自由度”，结点坐标、阶段数、边界位置和共享参数都列出；离散边界的选择开销不能藏在实数系数预算中。E3 oracle人工阶段与可由历史观测估计的阶段分开；只用测试轨迹未来信息切割的结果必须标为oracle表示上界。不给E2/E4测试窗口逐例自由搜索后再称可部署。

## 指标与分析

主指标：以rad计算每关节RMSE，再按四组宏平均，防止44D手部掩盖双臂；同时报告全维RMSE、各组P95/max绝对误差、每window及每episode分布。另报同一训练集尺度归一化后的误差，不能只报归一化空间。速度/加速度误差使用真实dt，断点处不差分；报告段边界跳变和左右臂/手相对轨迹误差作为协调代理，明确不是物理接触成功率。

固定窗口长度50、采样率/时间跨度、样本数、solver、精度、权重和split；除预算变量外使用相同数据。分阶段误差、单颗/连续多颗、平静/快速动作分层，但不根据测试结果事后改主指标。用episode或原始recording为bootstrap单位，而非大量重叠窗口；记录配对95%置信区间、运行时间和内存。固定可接受差异阈值后才看test。

## 预期交付接口（尚未实现算法）

P0-B：verified_dataset_manifest.json、normalization_identity.json、与原loader的真实Batch对照报告、来源分组split清单。P0-C：fit配置（basis/degree_by_group/stage_policy/budget）、逐窗口重建指标Parquet/JSON、budget accounting与配对bootstrap报告。先实现E0/E1及预算计算器，再实现E2，最后E3/E4；若门槛失败就记录失败原因。P0结束前不改Action Expert，不训练、不开展机器人控制。
