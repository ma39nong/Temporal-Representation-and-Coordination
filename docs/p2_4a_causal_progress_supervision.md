# P2-4A：真实目标确认、因果事件审核与紧凑条件验收

日期：2026-11-12；起点`feat/order-specific-velocity-readout@8b97b72`。
判定：**NEED HUMAN ANNOTATION**。已确认两条的真实目标并解决本轮限定域的条件长度问题；没有把候选事件或最终成功转成计数监督。本轮没有参数更新、GPU训练、机器人操作或远端推送。

## 1. 真实目标已经确认，不再索要一个不存在的指定数量

用户对这两条的原始目标直接确认：

> 没有指定，我们就是摘完一串番茄，先摘左边的，发现右边摘不了，就选择放回去，右手调整一下，转一下方向，然后重新抓取，最后放回空的杆子

据此固定`task_goal=harvest_all`、`target_count=null`。这是已知的定性任务目标，不是目标未知，更不是“最终摘到N颗，所以目标=N”。目标审核证据保存在`configs/p2_4a_task_protocol.json`，审核日期不是动作事件时刻。整体任务成功沿用此前用户审核；采集软件为什么在某个时刻停止录制仍无单独证据，不能把任务意图当成停止日志。

两条分别为：

|录制ID|原bag|导出Episode|帧数|目标/整任务审核|逐颗因果事件|
|---|---|---|---:|---|---|
|`b6cda8f38ada48f293f1f43b7fc8373c`|A/episode803|A/1109_783_818_normal Episode2|3297|摘完果串；整任务人工成功|待人工确认|
|`ab41cacd1b154778849b002cfde263dc`|A/episode804|同目录Episode3|3539|同上|待人工确认|

原导出的`meta/tasks.jsonl`仍是历史“摘一颗”文本，与本次确认的完整多轮任务不一致。没有修改原数据；新研究接口使用独立审核的目标元数据，不能把旧task_index文本直接当多颗真实目标。原480条单颗训练任务继续独立保留，未混入本轮连续数据。

## 2. 实际检查范围与事件证据

本轮实际完成：

- 顺序解码全部6路导出视频，未报码流错误；每秒取源帧0/30/60…生成时序导航，并生成14张三视图阶段接触表。重点查看两轮harvest、右手调整及末段return的图像。**全量解码不是全程人工逐帧事件审核**。
- 读取两条原始bag的collection_state/metadata、转换summary和trim_report，复用已有`lineage`核验，两条均verified；没有重哈希26GB原bag或重新转换。
- 对10个删除区间以SQLite只读、时间索引查询各相机附近1帧，实际读取30个图像消息、26,114,160字节。28个header时刻位于删除区间内；两个极短切口的cam0样本落在区间外，明确标记，不冒充缺口内证据。原消息只读，缩略图写入独立输出。它只是缺口抽样，**没有完成所有被删除时间的事件审核**。
- 两条各7个既有reviewed阶段，记录为14个离线phase标注；它们没有在线可确认时刻，不进入因果输入。
- 建立14个真实视频导航候选区间（每来源7个），含可能摘取/放置、重定向和末段放杆。全部`pending_human`，不是14次真实摘取，也不是完整事件清单。没有为不确定物体编造fruit ID或精确detach时刻。

主要候选导航：

|来源|摘取/放置候选（导出秒）|调整|末段放回|
|---|---|---|---|
|Episode2|10–15、15–20、20–24、24–29、90–98|48–66|101–109|
|Episode3|6–12、12–18、18–24、24–30、96–104|52–75|112–117|

这些是宽导航区间，不是已验证事件发生边界；其中跨切口的区间显式列出`crosses_cut_rows`，必须回看原时钟，不能在导出时钟上跨缝插值。物体身份、成功脱离、释放和放置地点、是否重复同一果实均待确认。没有从harvest次数或机械臂往返次数计数。

末段右盘中仍可见红色果实：需要核实是已脱落果实还是仍连接在枝上，以及是否应放入左篮。这个观察没有被自动判成整任务失败，也没有被整任务success标签覆盖成“全部已正确放置”。摘取成功与左篮放置成功必须分开标注。

## 3. 时间映射及连续段：已落实，不沿用表面连续假设

从两个实际trim_report的`compact_shared_time_mapping`恢复原始时钟，沿用既有`inverse_compact_time`。事件原始纳秒只接受整数/字符串，拒绝浮点纳秒损失；`export_seconds`按帧/30解释。原时钟事件落在删除区间、或时间区间跨缝时拒绝，不替其生成精确导出时刻。相机物理曝光延迟没有在本轮重新标定。

|来源|切口首行（右侧）|删除时长|忽略切口窗口数|合法[-7,+51]连续窗口|拒绝窗口|
|---|---|---:|---:|---:|---:|
|Episode2|311,1328,2284,2285,2843,3057|16.602744098s|3239|2948|291|
|Episode3|207,1445,2504,3385|17.218939803s|3481|3249|232|
|合计|10个切口|33.821683901s|6720|6197|523|

实际枚举了6197个合法窗口，逐一通过`reviewed_progress`，全部保持目标all、计数/剩余数unknown、`progress_valid=False`，没有把待审核标签送入输入。另对这两条既有P1-7的10个固定开发窗口核查，**0/10跨这些切口**；这个结论仅适用于这两条，未扩展到其余来源或重跑方法选型。

## 4. 因果监督接口

`src/trc/data/causal_supervision.py`复用既有切口检查，增加审核表到任务条件的薄适配：

- 物理事件至少具有recording_id、event_type、event_timestamp（允许区间）、timestamp_domain、evidence、confidence、review_status、verified_by。
- 还必须有独立的`confirmation_available_at`及其时钟、`causal_availability_verified`、唯一object_id。发生区间用上界，确认区间也用上界；到确认上界才更新，绝不取区间中点伪装精确时刻。
- 已审核完整reset/事件覆盖是有效计数的前提；尚未解决的过去摘取候选不能靠“列表里没事件”产生一个可靠0。跨过删除区间后，缺口事件未审核则全局计数未知。
- 已知未来确认只在到达可用时刻后生效；未来事件不改变当前计数。离线phase没有可用时刻，保持未知。
- 只有唯一果实的成功摘取递增，placement和retry不递增；重复同一物体不能重复计数。果串放回/转向/重抓不是pick_success。

新增`remaining_count`派生属性：只在数值目标已知且计数可靠时为`max(target-completed,0)`；all任务即使将来有可靠completed_count，数字remaining仍未知，不能用最终总数回填。all任务也不会触发数值到数停止；空杆确认、最后放置和结束请求需要自己的因果证据及安全控制器。

## 5. 紧凑模板：保留完整类型信息，不声称模型已经理解简写

新增固定版本`truss-progress-v1`，三阶段均应调用**同一个**`condition_observation(mode='compact_v1')`。不改变原Tokenizer、状态离散化或256上限。

```text
Right truss left pick basket 3/0 h0
Right truss left pick basket */? ?0
```

它是**任务宏+结构化字段**，不是任意自然语言省略：宏字典完整定义为“右手抓起/操持果串，左手采摘至左侧篮；all任务必要时放回、重定向、重抓，最终放回空杆”。元数据依次为`target/completed phase+terminate`：`*`=all、`?`=unknown，p/h/r/a为pickup/harvest/return/adjust，0/1为结束请求。`progress_valid`由completed是否为unknown无损恢复；不重复打印remaining。反解析器逐字段往返校验。

本轮限定目标1–9或unknown/all，completed为0–9或unknown；大于9显式报错，绝不截位。它仍需未来数据证明语言通道能学到含义，不能因短语中出现数字就宣称数量理解。原长语言模式保留作历史对照，但不允许把all/unknown目标错误渲染为字面“None”，这些新目标必须显式选compact_v1。

实际原SentencePiece检查：

|检查|结果|
|---|---:|
|合法元数据组合及反解析|1122组，全字段相同|
|离散状态bin|257个值，58个位置|
|混合状态分词可加性抽查|固定seed42，2000组通过|
|紧凑条件最坏长度|252–254 Token|
|本轮真实Batch紧凑输入|已知数量200；all+unknown进度198 Token|
|原训练Prompt真实Batch / 最坏组合|215 / 269 Token|
|P2-4旧长条件最坏组合|283 Token|

最坏值依赖已核验Tokenizer的数值分词可加性；没有宣称穷举257^58个状态。逐输入长度拒绝始终开启，未知Tokenizer不能直接套用此预算。原任务文本在这个真实状态下与原Tokenizer逐元素相同；原长Prompt最坏仍可能超长，原native的截断行为没有被悄悄改动。disabled严格走原Observation，保持原Step8000回归；今后启用条件的训练/验证/推理必须全部选择已锁定模板与哈希，不能训练长文本、部署短宏，不能直接调用未适配NativeDataset就声称已使用进度。

## 6. 真实Batch及回归

本轮实际使用**上述Episode2帧100**，完整支持[93,151]，位于首个切口311之前。真实三路RGB/动作/状态由固定原FLASH NativeDataset读取；动作与历史与本仓库LeRobotAdapter独立读出的canonical数组逐元素相同。不是用合成动作替代真实Batch。

CPU、FP32、Batch1、固定key42；Step8000全参数SHA仍为`40b2c2c4dcdd6fb88dd847d5762d2071fba5028af0942c6dfe457409fe5f5919`，统计SHA仍为`a5d7bd2dd422148736c69f1bf52a721d5770ba810f3bace9a2dd413395544d52`。

- 关闭条件的velocity、物理动作最大差均0；shape保持(1,50,58)。
- compact数值目标1/2/3仅是**合成功能条件**；实际all目标来自本轮用户确认，completed未知，未伪造真实数量监督。
- 四种条件前向和Loss均有限；all+unknown条件Loss约1.0057466，完整Expert梯度19叶/430,151,738元素有限，梯度范数约12.8532，VLM梯度0叶。该Loss不是新任务学习有效性的证据。
- 原Head+条件JSON保存恢复，物理输出差0；主体全部参数前后SHA一致。不是新训练的模型或Adam状态。
- 新增可训练参数0、优化器更新0，原Absolute/K6/7 Token/50×58及归一化不变。未启动GPU训练。

基础测试151项通过，包括unknown/all目标、数值单颗默认、多颗计数变化、迟到确认、未来事件不泄漏、raw/export时钟、跨删除区间拒绝、episode支持边界及条件协议往返。Token边界另外由原Tokenizer真实检查，不用假Tokenizer代替。

## 7. 科研判断

1. **两条能否形成可靠进度监督？** 目标已确认；逐颗成功/确认/放置及缺口覆盖未完成，当前不能。
2. **接口信息是否够用？** 字段和时间门控已具备；真实审核值尚未齐全。
3. **是否有不同指定数量监督？** 没有；这两条都是all。原480条只支撑单颗基本技能，不能回填2/3目标。
4. **能否开始基础对照？** 仅功能与数值对照可以；真实目标+进度训练仍不就绪。
5. **能否学习数量控制？** 当前数据不足。all与数值目标必须分开，不借最终结果制造指令。
6. **简单计数器/状态机能做什么？** 在可靠外部事件及安全结束规则已经存在的前提下，可累计确认成功、比较明确数值目标、选择继续/停止请求及既知阶段Prompt；它本身不能解决遮挡下成功识别、空杆判定、可达性、重新抓取姿态或动作生成可靠性。
7. **什么才需要Expert学习？** 相同合法观测下，目标/进度变化是否要求不同且可靠的下一段动作；重定向与重抓后的阶段衔接、保持果串及局部动作精度。只有共享FLASH在公平A–D对照中超越“外部事件+重复调用”，才支持内部条件利用的研究价值；本轮不新增结构。

## 8. 人工最小操作、下一轮数据条件与产物

可直接打开`outputs/p2_4a/review/review.html`并查看三路完整导出视频，对照`event_review.csv`：

1. 核对14个候选导航，并补齐漏掉的真实果实事件；为同一果实建立贯穿摘取/重试/放置的object_id。候选行数不是果实数。
2. 每个事件填写可观察区间、确认可用区间、证据/审核人；确认可用性时只能看截止该时刻的观测前缀。无法确认则保留pending/unknown。
3. 对10个删除区间，结合原bag逐段确认是否藏有成功/失败/放置事件，不能凭30张缺口抽样图宣称完整审核。确认完整起始状态/reset；末段分清空杆、右盘可见果实、最后放置和真正录制停止原因。
4. 所有审核通过后，再生成6197候选连续窗口的因果快照；当前没有任何窗口被伪装成进度训练样本。

最小下一轮数据条件：先让两条all录制具有完整因果事件覆盖；若要研究**指定数量控制**，另外需要采集时已经给定1/2/3等不同目标的独立成功录制（协议探索可先每类至少2条，不代表统计充分），保留同样的逐颗/放置/重试/结束审核，并按recording另留未参与设计的新来源。旧480可以保留基础技能，但不能靠重新标目标凑多目标样本。两条开发录制不能成为任务泛化证据。

产物均在`outputs/p2_4a/`：

- `raw_provenance.json`、`lineage.json`、`clocks.json`、`prior_window_cut_check.json`：原始谱系、映射、切口及旧窗核查。
- `video_review/`、`raw_gaps/`：完整解码记录、阶段导航、有限原bag缺口抽样与实际header时间。
- `review/event_reviews.json`、`event_review.csv`、`review.html`、`alignment.json`：真实目标、候选证据、人工审核入口与实际窗口检查。
- `token_budget.json`、`acceptance.json`、`tests.log`、`delivery_manifest.json`：Tokenizer和最终代码数值验收身份。

运行入口：

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/prepare_causal_event_review.py PRIVATE_REVIEW_CONFIG.json NEW_OUTPUT_DIR
# 显式固定版本OpenPI环境，所有资源来自私人配置；CPU，无参数更新
python scripts/check_compact_task_budget.py PRIVATE_RUNTIME.json TOKEN_RESULT.json
python scripts/check_task_condition_source.py PRIVATE_RUNTIME.json
```

私人路径、视频、原始/衍生数据、模型与输出不入Git；协议及实现本地提交，不推送。最后状态为 **NEED HUMAN ANNOTATION**，不是Token长度或新网络阻塞。

人工补齐后，直接编辑独立的`event_reviews.json`（不是重跑候选生成器覆盖审核），用以下入口检查并导出仅有确认进度的快照：

```bash
.venv/bin/python scripts/validate_causal_event_reviews.py \
  REVIEWED_EVENTS.json outputs/p2_4a/clocks.json NEW_VALIDATION.json
```

该入口本轮已在真实6197连续窗口上执行，确认快照输出0条、unknown6197；523个跨界窗口被拒绝。完整性标志需要审核者证据，不能为让脚本通过自行设为verified。人工可先填写CSV作记录，最终将经确认字段同步到事件JSON；程序没有将CSV候选自动升级为verified。
