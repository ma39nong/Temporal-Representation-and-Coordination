# 数据契约（P0-A）

本契约经本地源代码和本地早期Sharpa元信息核验，不能推广至所有数据集。用户确认主训练工程为 `tomato-openpi`，分支 `delta-paper-training-test`，Commit `2ab68c8b8090284fca6d258f86bf5098d2f1302b`。原始字段、训练目标、部署物理动作是三个不同层次。

## 两种明确布局（索引从0开始，闭区间）

| 分组 | canonical58 原始/模型顺序 | converter58 原始action | converter116 原始state位置 |
|---|---|---|---|
| Right Arm 7D | 29–35 | 7–13 | 14–20 |
| Right Hand 22D | 36–57 | 36–57 | 72–93 |
| Left Arm 7D | 0–6 | 0–6 | 0–6 |
| Left Hand 22D | 7–28 | 14–35 | 28–49 |

canonical58为LA,LH,RA,RH；converter58存储为LA,RA,LH,RH。converter116为LA位置0–6/速度7–13、RA位置14–20/速度21–27、LH位置28–49/速度50–71、RH位置72–93/速度94–115。训练只提取位置并重排。action_schema.py 中 ORDER 和 expected_names 保存每个关节的完整名称；必须同时匹配feature shape、float32与names才能审计，不仅检查58D。手部顺序：thumb_CMC_FE, thumb_CMC_AA, thumb_MCP_FE, thumb_MCP_AA, thumb_IP, index_MCP_FE, index_MCP_AA, index_PIP, index_DIP, middle_MCP_FE, middle_MCP_AA, middle_PIP, middle_DIP, ring_MCP_FE, ring_MCP_AA, ring_PIP, ring_DIP, pinky_CMC, pinky_MCP_FE, pinky_MCP_AA, pinky_PIP, pinky_DIP。每臂joint1–7。canonical名称带left_/right_前缀；converter action另带.position或.command，state带.position/.velocity。

早期Wuji为54D、每手20D：明确不支持，拒绝当作Sharpa解释。未知命名或新布局必须先新增可追溯契约，不能自动猜测。

## 字段与语义

| 字段 | 约定/证据 | 状态 |
|---|---|---|
| action | float32[58]，存储层为关节命令位置，单位rad；不是实测state | 本地Sharpa协议/转换器已核验；每个新数据源仍需确认 |
| observation.state | canonical为实测位置58D；converter为位置/速度116D | 两种适配源码已核验 |
| timestamp | 秒，episode局部网格，通常float32，帧0为0，fps来自info | 本地已读数据30Hz |
| episode_index/frame_index | 整数；每个文件一个episode；frame_index应0..N-1 | 审计逐文件检查 |
| source_timestamp_ns | 原始绝对时间映射，可选，原转换器会写 | 本次4个真实数据目录均未在Parquet找到 |
| observation.images.cam0/1/2 | 三路RGB，元信息引用视频；cam0全局、cam2左腕、cam1右腕为已读训练配置定义 | 未解码视频/未核验同步 |
| engagement/image_mask | 是否能作为有效示范/监督掩码需按数据版本确认 | 不擅自把engaged=false全部删除；报告计数 |
| episodes.jsonl | episode_index,length，转换器可附source_recording_id、segment_id、source起止等 | 本次读取的早期数据未含所需原始映射证据 |

LeRobot v2.1读取meta/info.json、meta/episodes.jsonl，按data_path/chunks_size解析每episode Parquet。仅读numeric列，不触及视频数据或缓存。根目录逃逸模板拒绝。路径只从配置读取，相对路径基于配置文件所在目录。

## 原始动作 → 训练目标 → 物理动作

对明确采用Delta arms的配置，窗口以采样点t的实测state为锚：未来每个k的双臂目标为 commanded_arm(t+k) − measured_arm(t)。双手仍为 commanded_hand(t+k)。不是 action(t+k)−action(t+k−1)，也不是由模型目标累积积分。

训练顺序：原始布局核验 → 重排/提取state → Delta arms → Normalize → T3 ArmNormalization bridge（若配置启用）。输出逆序：T3逆bridge → Unnormalize → Absolute arms（加回原观测锚点state）→ 58D物理位置命令。实际部署程序的发送顺序/插值/限幅本阶段未运行，不宣称已经验证。

## 归一化

`norm_stats.json` 根为 `norm_stats`，下有state/actions，每项有mean,std,q01,q99。通用源实现：zscore=(x−mean)/(std+1e−6)，quantile=2(x−q01)/(q99−q01+1e−6)−1，不clip；对应逆变换使用相同epsilon。

T3源实现从quantile坐标恢复双臂Delta，再以 `(delta−arm_mean)/max(arm_std,std_floor)` 替换双臂，双手保持quantile。这里没有在std上加1e−6；不能与通用zscore混同。默认floor在主仓库PaperSharpaData中为1e−4，但实际实验必须读配置/manifest，不在新接口中暗设此值。

T3需要匹配的norm_stats.json、norm_metadata.json、delta_paper.json、delta_paper.npz及SHA-256。动作stats为[50,58]，arm_mean/arm_std为[50,14]，manifest验证mode、horizon、floor、beta、time-major、complete_only和数据contract。已有早期资产仅发现[58]统计，不能替代T3统计。load_stats解析并校验数值JSON；assets.load_assets现已实现bundle内部完整性、实际数据内容绑定和独立预期契约比较，pipeline.ActionCodec串联完整编码解码。默认codec只接受verified资产。本机尚无匹配真实T3 bundle，因此**实现完成不等于真实T3物理恢复已完成**。不估计、不填充虚构统计。没有锚点state时也无法恢复双臂绝对命令。

## 时间段与窗口

P0接口仅提取同episode连续段内完整50帧窗口，不padding、不插值、不重采样。30Hz下首末样本跨度49/30秒；不能把50帧误称50秒或无条件1秒。严格检查dt>0、dt与1/fps误差≤配置tolerance_s（默认1e−4秒）、frame_index连续，若有source_timestamp_ns也检查原始网格。连续段区间采用[start,end)，窗口数量max(0,N−50+1)。任何数值/维度错误的episode不提供可用窗口计数。

主训练仓库CompleteWindows仅据episode/frame_index筛完整窗口；本接口额外检查时间/原始时间断点，因此坏数据时有意更严格。通用LeRobot路径按round(k*fps/action_hz)/fps申请动作时间；其实际padding、最近邻容差和版本尚未端到端对照，不能声称全Batch等价。

早期export_lerobot.py将裁剪段分别保存episode，外部导出报告可记录source_start/end_index和crop_interval；转换器v2.1另有source_timestamp_ns/segment metadata。元信息有能力保存不等于每份数据实际保留。时间戳局部连续不能证明裁剪前物理连续。缺失映射的数据只能用于标记为来源连续性未知的分析，不能跨episode拼接；分割训练/验证必须避免来自同一原始recording的裁剪片段泄漏。

## 当前工程接口补充

多源manifest对元信息与Parquet计算内容SHA，来源映射可通过独立source_mapping_path提供但不可与原元信息冲突。split按source_recording_id分组，不以Episode替代原始recording；原始映射缺失时阻塞正式split。iter_windows显式返回同帧锚点/时间与raw/canonical动作；validity_column仅接受真实一维boolean字段，不推断engagement语义。T3绑定还检查action_sequence_keys和action_hz；当前不重采样，要求action_hz等于实际FPS。用法与状态层级见usage.md。
