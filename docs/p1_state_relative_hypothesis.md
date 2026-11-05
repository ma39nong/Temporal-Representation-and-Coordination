# P1-0：状态相对轨迹表示的假设核验

日期：2026-11-05。以研究仓库 `50bc25d` 为起点，只读引用 `tomato-openpi/paper-flash-test@d208d414efbb78483f44bfef4c2a9314115dd632`。本轮新增无参数坐标适配器与真实Batch前向检查，没有修改原训练工程、数据、Checkpoint或控制程序，没有启动正式训练。

## 当前判断

**证据支持继续验证状态相对坐标，但尚未证明它能提高训练后策略性能。** 候选20窗的物理误差确实约98.8%为时间常量偏移；这不是直接照抄P0-C的系数98.75%。同时，简单把预测第一步强制移到q0会损害训练来源性能，说明“可逆坐标参考”和“正确动作目标”必须区分。详见 `p1_offset_decomposition.md`。

未发现动作顺序、状态语义或匹配统计方面的阻塞性错误，因此完成了最小适配器。源身份、相机校准和推理数值复现仍有明确限制；当前不建议直接投入正式大规模FullFT，先做预注册的B0/B2小预算受控实验，协议见 `p1_controlled_experiment.md`。

## 四个候选来源的身份审计

本轮用原 `flash_native.read_numeric` 对480个训练Episode及4个候选Episode执行针对性检查，不重新建设数据系统。训练来源97个/485窗，候选4个/20窗的固定plan不变。

| 简称 | 完整 source_recording_id | Episode | 全Episode帧数 | 源开始时间（北京时间） |
|---|---|---:|---:|---|
| 0340d869 | `0340d869ea074774b09441f2339cec30` | 2 | 5474 | 2026-11-02 14:45:09 |
| 435fc6da | `435fc6da99284d7ba15cf09193352c06` | 2 | 4542 | 2026-11-02 15:08:20 |
| b710689f | `b710689fe5084126b0ab3dedde821f29` | 3 | 2789 | 2026-11-02 15:52:25 |
| e4bdeafd | `e4bdeafd7d834062bfba2ad01049a5b8` | 1 | 7654 | 2026-11-02 13:24:50 |

四个source ID互不相同，均不在训练480来源中；候选源时间范围与480训练源时间范围无重叠。候选4 Episode共20459帧。将每帧按canonical顺序的58D action与58D measured position拼接为float32指纹，与全部451270训练帧比较：**每个候选来源均0个完全相同帧，也没有按同一训练record顺序出现的复制片段**。这比仅检查UUID更强，能排除保留数值的直接转换/裁剪副本；不能排除有损、重采样、数值改写或改时间戳后的副本。

服务器提供的这些数据副本没有 `conversion/`、`ingest/` 原始bag审计目录。仅有源ID、segment和源起止时间，缺少内容校验链。参考本地转换器 `src/convest/ingest/publish.py`、`ingest/inputs.py` 可知完整发布流程原本可保存原始输入hash证明，但不能假设当前服务器副本保留它。**结论仍为候选独立来源，未升级为已完整证明的独立泛化测试集**。未对原始bag做大规模扫描、复制或修改。

## 动作、状态、相机和配置一致性

484个Episode的action/state完整feature、三路camera元信息、fps及英文task组合只有一种schema身份（SHA256 `f4dec0e183f2481ef13fcefff15a38d1fe2034e16caa0fe2d77114325079dc20`）。全部使用原命名映射：

| 组 | 模型索引 | action | 当前参考state | 离线平移可用性 |
|---|---|---|---|---|
| Left Arm | 0:7 | 同名7关节绝对位置命令 | `.position`实测位置 | 可用 |
| Left Hand | 7:29 | 同名22关节`.command`位置命令 | `.position`实测位置 | 可用 |
| Right Arm | 29:36 | 同名7关节绝对位置命令 | `.position`实测位置 | 可用 |
| Right Hand | 36:58 | 同名22关节`.command`位置命令 | `.position`实测位置 | 可用 |

原始存储action为LA/RA/LH/RH，state为116D位置/速度；原读取器按名称选取58个位置并重排为LA/LH/RA/RH。没有将state速度当位置，也没有套用54D或T3。单位依据已核验的 `fr3_sharpa_protocol.py` 的rad契约及本地转换器 `schemas/sharpa_contract.yaml` 的radian、同名JointState.position提取。手的command与measured position是不同信号，但同一关节位置坐标；不能认为两者逐值相等。没有发现需排除的某个58D关节组。

以上“可用”仅指可逆离线坐标变换；不包含机器人驱动标定、手内部耦合执行或相机外参的实机安全认证。若未来数据action为速度、力、协同量或不同手关节单位，必须明确拒绝，不能按58D自动适配。

相机字段/编码/分辨率一致：cam0为400×640×3，cam1/2为480×640×3；h264/yuv420p、30Hz。原配置映射 `base=cam0,left_wrist=cam2,right_wrist=cam1`；原 `NativeDataset.resize_image` 与tokenization保持不变，英文任务完全一致。P0-C的1515张当前帧原始解码像素身份已核验；本轮真实前向也检查所用原始RGB哈希。不宣称物理相机序列号、曝光/内外参、摆位或视觉场景相同，这些未包含在现有info中。

FullFT4000/6000导出均使用匹配模型配置与当前all480统计，不用历史Step250统计。统计SHA256为 `a5d7bd2dd422148736c69f1bf52a721d5770ba810f3bace9a2dd413395544d52`，训练契约为 `50e26190703b466987ee5a8db9879fb04b43a5e221d3c07df5bcf8121c80e1e8`。离线新入口再次检查封印与统计hash；真实前向调用原preflight并核对完整契约、原参数内容身份。

## 状态分布：Episode开始与窗口当前状态要分开

Episode首个保留帧不一定是原始录制起点。下表把候选Episode首帧实测位置与480训练Episode首帧比较，按训练首帧各关节std标准化后，报告到最近训练首帧的组内RMS距离；仅为描述性距离，不是概率或显著性。

| Source | Left Arm | Left Hand | Right Arm | Right Hand |
|---|---:|---:|---:|---:|
| 0340d869 | .0011 | .3551 | .0675 | .4897 |
| 435fc6da | .0049 | .4546 | .0901 | .3697 |
| b710689f | .0066 | .4165 | .0821 | .5065 |
| e4bdeafd | .0030 | .6405 | .0952 | .5140 |

候选机械臂Episode起始姿态与已有训练首帧很接近，不支持“初始机械臂明显不同导致偏移”的直接解释。手初始有一定差异，但多数关节仍在480训练首帧的min/max内；只有b710689f右手1/22关节超出该范围。4个初始状态不足以声称总体分布显著不同。

固定评价窗口取自后续不同时间位置，情况不同：候选窗口q0相对训练485窗q0的平均绝对z值，Left Arm约1.02–1.41、Left Hand约1.04–1.58、Right Arm约0.44–0.61、Right Hand约0.74–1.08。候选Left Arm超出训练state q01/q99的比例约2.86%–17.14%，Left Hand约6.36%–19.09%，Right Arm均为0，Right Hand约3.64%–10.91%。这些指标和最近窗口距离按四来源分别输出；只能作为过程中的状态变化线索，不能证明它是模型误差的因果来源。所有参考统计只来自训练数据；没有给B2重估候选统计。

## 数据质量检查的范围

原读取器确认全部记录有限、frame_index逐帧连续、30Hz时间间隔；没有NaN/Inf或跨Episode采样。候选全Episode的action/state最大相邻跳变如下，单位rad（不是速度）：

| Source | Action最大跳变 | Action P99跳变 | State最大跳变 | command−state RMSE |
|---|---:|---:|---:|---:|
| 0340d869 | .631222 | .011209 | .243160 | .035969 |
| 435fc6da | .614662 | .012272 | .147655 | .040181 |
| b710689f | .178425 | .016874 | .083724 | .037506 |
| e4bdeafd | .383372 | .012931 | .159874 | .039001 |

训练记录最大action/state跳变约.915125/.912962。候选存在稀疏大跳变，值得定位人工复核，但不是仅候选具有的现象；不能以“比训练最大值小”宣称动作安全或数据完全正常。尚无核验的关节限位、接触事件和原始控制延迟，因此没有构造无依据的异常阈值、清洗或删除窗口。时间网格一致不证明原始传感器端到端延迟为0。

## 状态相对表示的精确定义

物理坐标：对核验兼容的关节组，`R=A_abs−q_ref`，`A_abs=R+q_ref`。q_ref只取原合法history[-1]的实测位置；history[-7,0]与target[-2,51]共用这一参考。不用GT[0]，不强制R[0]=0，不把Paper T3混合Delta当成这个全关节相对表示。

保留原训练Quantile的跨度 `d=q99−q01+eps`，定义原 `F(A)=2(A−q01)/d−1`。相对坐标归一化为 `G(R)=2R/d`，因此 `G(A−q_ref)=F(A)−F(q_ref)`。不能直接套用 `F(A−q_ref)`，两者相差常量F(0)；也不能在已归一化动作上直接减物理rad的q_ref。没有新增或从验证来源计算任何训练统计。未选中的关节继续保持原Absolute/F变换。

scaled系数的原解码是 `U=B diag(s) C`。Legendre0阶为常量1，令 `C_anchor[0]=F(q_ref)/s[0]`，其余阶为0，则 `C_rel=C_abs−C_anchor`；1–6阶保持不变。加回anchor后原50步物理解码一致，时间导数不变。训练target原边界投影与物理动作约束不变，推理不引入未来端点。

**必须先用原算法拟合，再平移系数。** 原history source使用ridge以及原FLASH的历史重参数化网格。直接重新拟合归一化后的相对history一般不等于原source减理想常量系数；本轮真实Batch两者的scaled系数最大差达到8.718206，不能把它当浮点噪声。适配器选择原HAF多项式的可逆坐标平移，保留原source定义，不偷偷改成新的历史拟合器。target具有常量平移等变性，原边界投影后的相对重拟合解码也已数值核验。

## 最小适配器及真实前向验收

`src/trc/models/flash_coordinates.py::FlashCoordinates` 只含固定数组和数学运算，无训练参数、VLM或模型副本。配置显式选 `absolute` / `state_relative` 和兼容组；默认示例为Absolute。两种模式使用同一原Expert的7×58 token与50×58输出。所有基础测试只依赖NumPy。

原环境测试以真实Step6000权重、window0（训练来源）与window173（候选435fc6da）组成Batch=2，原VLM/Expert严格加载；原模型参数和Checkpoint均未更新。**B2使用Absolute模型现有权重只是未训练的前向功能测试，不是B2预测性能实验。**

| 检查 | Batch=2真实结果 |
|---|---:|
| 物理A−q再加q最大误差 | 0 rad |
| 高阶系数变化 | 0 |
| 系数往返最大差 | 4.77e−7（scaled坐标） |
| 50步物理解码最大差 | 2.38e−7 rad |
| 原边界target重拟合解码最大差 | 5.36e−7（归一化坐标） |
| target−source速度目标变化 | 9.54e−7（scaled坐标） |
| 物理时间差分变化 | 0 rad/s |
| B0适配路径与同图原sample API | 0 rad |

原FM与one-step consistency的前向计算都有限；同一时间与source噪声用于两种坐标。其未训练loss数值不当作算法收益证据。真实模型推理函数无target参数，target只在前向预测之后用于拟合/数学检查与loss前向；没有optimizer或梯度更新。

数值复现还需固定原运行时、JIT图及推理Batch：同图B0与原API一致并不保证不同图/不同Batch逐位相同。本轮对该问题进行额外单窗原图重放，具体结果写在受控实验文档和独立数值报告中；不把不同执行图的差异解释为B2收益。

## 产物与限制

`outputs/p1_state_relative/source_audit.json`：484记录的schema、numeric/Parquet hash、来源/时间、精确复制检查及质量信息；`offset_final/`：两Checkpoint全部分解与分布指标；`real_batch_verified/`：真实前向、原图重放及数学检查；所有真实数据产物默认不提交Git。模板在 `configs/p1_coordinates.example.json`、`p1_forward.example.json`、`p1_source_audit.example.json`。

结论范围：物理偏移与0阶差异高度对应，状态参考在四个关节组上具有兼容的位置语义，坐标变换可逆且可保持原FLASH监督/HAF定义。仍未知：B2训练后的收益、独立会话泛化、图像/状态分布各自的因果作用，以及实机协调结果。这些未知不能由数学往返测试填补。
