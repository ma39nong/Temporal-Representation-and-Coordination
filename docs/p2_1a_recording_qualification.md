# P2-1A：90个候选录制实际资格复核

日期：2026-11-10；起点`main@17f544a`。最终判定：**BLOCKED BY REVIEW/DATA**。

本轮已实际逐一检查90/90个来源，完成原始bag元信息/转换谱系核对、全部数字动作与状态读取、270路视频完整压缩包时间轴检查及首/中/尾抽样解码，没有运行D0/D3推理或训练。程序可确认的资格已落实；任务成功、实际单颗/多颗类型和场景相对训练条件仍需要人工证据，未擅自判成功。

## 1. 交付数字

| 项目 | 实际结果 |
|---|---:|
| 候选逐一检查 | 90/90 |
| 有可信完整任务成功审核 | 0 |
| 有可信完整任务失败审核 | 0 |
| success_status=uncertain / 待人工审核 | 90 |
| 原始rosbag元信息与卷文件可访问 | 90 |
| 来源谱系核验通过（下述证据范围） | 90 |
| 与480训练及已用研究来源隔离核验通过 | 90 |
| 最低离线数值/时间轴数据合同通过 | 90 |
| 初始状态距离完成 | 90 |
| 明确near / far | 2 / 3 |
| ambiguous，不能强制归格 | 85 |
| 场景near/different已审核 | 0 |
| 实际single_pick / continuous_multi已人工确认 | 0 / 0 |
| 处理数字帧 / 视频 | 387,996 / 270 |
| SHA匹配的本地完整视频播放入口 | 270/270 |

`verified`只用于有证据的对应技术字段；`failed`用于确认不满足的技术条件；`unknown`用于缺证。任务结果单独使用success/failure/uncertain。技术通过90不等于任务成功90，也不等于90个来源已经可以进入评价。

## 2. 可立即使用的完整审核材料

- `configs/p2_1a_recording_review.json`：**90条真实来源记录**，已填写技术状态、冻结状态距离/类别、谱系证据SHA、有效片段及排除原因；全部成功状态仍为uncertain，reviewer未冒充人工。
- `outputs/p2_1a/recording_qualification.csv`：90条总表，列出原始ID、bag/collection_state/metadata路径、Parquet路径、三路本地与远端视频、五组距离/类别、批次、连续段与合同状态。
- `outputs/p2_1a/human_review.html`：90条可展开审核卡片，含首中尾图、**完整导出三路视频播放器**、证据链接、任务成功/失败、单颗/多颗和场景审核表单。可填写实际审核人并导出reviewed JSON；未完整核对的来源保持uncertain，未附训练参考与理由的场景保持unknown。导出不会修改原始数据，也不会自动把来源入组。
- `outputs/p2_1a/video_review_index.csv`：270路视频逐条地址、SHA及其对应来源。
- `outputs/p2_1a/priority5_review.csv`：先审核明确near/far的5条；完整并排视频播放时长合计约15.08分钟。
- `outputs/p2_1a/missing_review_items/`：每个来源一份缺失项清单，共90份。
- `outputs/p2_1a/evidence/`：每个来源一份真实审核证据，共90份，含数值审计、原始bag身份、转换验证报告、删除映射、视频PTS与SHA等。

在本机仓库根目录运行`xdg-open outputs/p2_1a/human_review.html`，用本地浏览器打开；若应用只显示HTML源码，请改用桌面浏览器打开该文件。全部270个播放器使用SHA核对后的本地文件，不依赖SFTP网页播放。

89个来源复用了已有本地视频，没有重复复制。唯一缺本地视频的来源`2099dc5c4d9443338d0d7c3b90f1df0c`（B/1107_696_696 Episode0）已将3个文件、共49,403,091字节约47.11MiB缓存到**仓库外**`/home/user/datasets/trc_p2_1a_review/`，并核对SHA。原始数据未变，视频不进入Git。

## 3. 原始录制、转换与隔离证据

沿用P2-1冻结的90唯一ID，不新增或删选候选。两个1109 Episode0别名仍各只计一次，选择其1107实际来源作为主路径，全部别名保存在证据中。

实际在数据服务202上找到90个对应原bag目录：`.../rosbag/sharpa/no-block-more/{A,B}/episodeN`。逐一检查collection_state的source_recording_id、metadata.yaml、列出的.db3卷文件存在与大小。旧采集机source_bag路径只作历史映射，当前确认可访问的是202上的身份吻合副本，不声称旧采集机文件仍存在；**没有读取大型消息数据库正文，也没有重新计算数TB bag的全量SHA**。

逐来源连接同服务发布的LeRobot原导出：原ID、帧数及source_start_ns/source_end_ns必须与当前normal Episode完全一致，再将ingest/sources/episodeN的summary、trim_report连接到同一路由/原bag。重用原转换记录中的original_sha256_verified、passed独立对齐验证和dataset_verification_passed，并核对当前bag卷大小与验证时readiness记录相同。全部90条通过。证据文件位置、SHA、转换验证时间及原验证结果随来源封存。

这里的verified是**受控注册表及既有可信转换证据范围内的来源链确认**，不是本轮独立重放全部原消息或新做全量bag内容认证。相同ID/文件存在本身没有被当作完整谱系证据。

当前90份Parquet全部读取，计算canonical动作+位置反馈内容SHA及Parquet SHA，与封存480训练身份比较；ID、完整数字内容和完整文件均不重叠，候选之间也没有完整数字副本。所有ID排除既往4候选、P1-7全部10来源。没有宣称做了任意部分轨迹相似性检测；原bag身份与转换链提供了额外原录制隔离证据。

缺失的原bag/视频路径：**本轮已找到的对应90个来源中为0**。仍缺的是90条完整任务人工结果、实际任务类型及场景相对训练条件证据，逐条见missing_review_items；不是继续索要一个笼统的新路径。

## 4. 数据合同与真实时间段

复用既有LeRobotAdapter、verify_features、canonicalize、audit_episode。全部90源确认LeRobot v2.1、normal30Hz、float32动作58D、状态116D、同名58位置反馈恢复、NaN/Inf/长度/帧号/episode/时间戳检查。存储顺序与canonical顺序按已验证字段转换，不按维度猜测，不使用T3 Delta统计。

270个视频总量约2.963GiB；单工顺序读取压缩包PTS及文件SHA，全部视频包数与对应帧数匹配，排序后的呈现时间逐帧等于frame/30，最大偏差0s；每视频首/中/尾共810帧抽样解码通过。没有全量转码，也没有声称所有帧码流都已完整解码；完整视觉任务审核仍由人完成。

相机cam0/base、cam1/right_wrist、cam2/left_wrist的设备身份元信息在90源中相同，三路视频与动作共享导出帧网格。原转换独立传感器对齐验证被溯源复用；这不代表物理传感器没有延迟、完全同时曝光或本轮恢复了每个传感器逐包时钟真值。

**发现并落实了原始清洗拼接边界：90/90来源均有时间压缩删除映射，当前Episode内共有633个边界。** 从原trim_report的compact_shared_time_mapping反向恢复导出帧网格在原bag时钟中的位置，边界取删除区间右侧；恢复端点均在bag元信息区间±0.2s核查范围内。这个范围只是本轮端点一致性核查容差，不是动作/图像同步误差指标，也未用来改变模型规则。

**完整导出视频不是未经剪辑的原始任务录像。** 人工结果审核必须说明删除区间是否影响完整任务判定，必要时结合原录制关键区间或采集者明确证据；若不能确认，仍保留uncertain。无需对所有bag做无差别全量解码。

不能因为normal导出时间连续就把拼接两侧当作原连续动作。每来源已经输出valid_segments和cut_rows；history[-7,0]与监督[-2,51]必须落在同一有效段。90源合法完整支持窗口共351,695；若忽略原始删除边界会得到382,776，**多算31,081个**。

这只是数据资格/片段统计，没有提取这些窗口做模型评价。现有后续评价入口必须消费本轮段边界sidecar后才能使用；不能沿用只看导出帧连续的窗口索引。合同通过指保存边界并限制在有效段内可用，不指整条拼接后的Episode天然连续。

## 5. 冻结状态分层：真实结果

直接加载P2-1的480训练初始状态参考、均值/尺度和q95/q99，没有重新拟合或改阈值。q_init仍为首个导出有效帧的canonical58位置反馈，不改为某个更容易归格的中间帧，也不声称它就是裁剪前复位姿态。原P2-0四折规则未改动，本轮未调用其分支预测。

| 分组 | 边际near | 缓冲区 | far |
|---|---:|---:|---:|
| left_arm | 90 | 0 | 0 |
| left_hand | 47 | 42 | 1 |
| right_arm | 88 | 2 | 0 |
| right_hand | 76 | 12 | 2 |
| full_action | 88 | 2 | 0 |

边际near只表示该组最近邻≤既定q95，不等于full58共同参考条件满足。最终**2 near、3 far、85 ambiguous**；85中37个各组边际都near，但找不到同一个训练参考同时满足所有组/全维q95；其余48个至少一个边际处于缓冲区。不能将这些来源改阈值或用不同参考拼接成near。

3个far全部由手部组触发；本批没有明确机械臂far来源，不能据此声称已经覆盖臂初始分布偏移。

### 优先人工审核的5条

| recording ID | 状态 | 来源 / Episode | 偏离组 | 完整时长 |
|---|---|---|---|---|
| `b0e4b49bfa1641beb783ec4a2e4e66ee` | far | A/1103_620_642_normal_lerobot / 4 | right_hand | 175.5s |
| `b693693c0fd34115989b89177cbe78c2` | far | A/1106_702_739_normal_lerobot / 16 | right_hand | 123.4s |
| `cc129554ec8340dab7716bdd1a772a5a` | far | B/1103_604_632_normal_lerobot / 0 | left_hand | 205.9s |
| `031906d13dc3439a82f379aee2ed5273` | near | B/1103_604_632_normal_lerobot / 1 | — | 247.6s |
| `bb378a3fc949476d98a5a21b68a157c5` | near | A/1106_702_739_normal_lerobot / 0 | — | 152.5s |

这些只是在状态条件上明确，不是已成功或已确认单颗。每条直接审核入口和原视频路径见priority5_review.csv及HTML对应ID卡片。优先这5条即可判断目前最可能入格的少数来源，不要求先花时间看完85条ambiguous才补录缺口。

## 6. 场景与任务类型：已知和未知分开

有89条来源带人工reviewed阶段标注及任务文本“Harvest all tomatoes into the left basket and return stem to right plate”；其中88条有2个harvest阶段、1条有1个。另1条无对应阶段标注。这个信息提示需要重点区分多轮采摘、单颗任务和失败重试；**阶段数量及任务意图并不等于实际成功摘取数量**。因此没有把89条直接标成continuous_multi，也没有把只含一个harvest阶段的录制标成single_pick。

任务成功/失败和实际类型90条均未找到适用的完整人工审核结果；此前用户对P1-7十条的确认不适用于本90条。数据录制quality/finalized/转换passed也不充当物理任务结果。

已记录批次、设备序列号与三相机视角语义。果实实际数量/位置、果串构型、遮挡、机器人—目标初始关系和场景接近性没有足够可信的统一标签，全部保留unknown。没有使用D0/D3误差或目录A/B猜场景。

为使人工对照可执行，另从7个不同训练根目录各选最早Episode（预定元信息规则，无性能筛选），提取三相机首/中/尾参考图，共7组；已嵌入审核页并封存reference_id和源视频路径/采样时间。它们是**待人工确认代表性的训练场景参考候选**，不是自动建立的场景真值标签；审核人必须给具体参考ID和理由才填写near/different。

## 7. A–D交叉覆盖与具体缺口

只允许完整人工成功+最低合同+谱系+训练/研究隔离+明确状态+明确场景+**实际single_pick**的独立recording进入主评价。连续多颗另列，不能剪出多个片段冒充多个原来源。

| 格子 | 状态—场景 | 当前合格独立来源 | 至少4条的缺口 |
|---|---|---:|---:|
| A | near—near | 0 | 4 |
| B | near—different | 0 | 4 |
| C | far—near | 0 | 4 |
| D | far—different | 0 | 4 |

即使最乐观地把现有2 near、3 far都审核为单颗成功且场景可合理分配，A/B合计仍需≥8条near，至少缺**6条near**；C/D合计需≥8条far，至少缺**5条far**。因此本90条在冻结规则下无法单独形成四格各4条；至少还需11条明确状态的独立成功录制，这是**条件性乐观下界**，并非保证补11条就够。若现有5条实际为多颗、失败或场景集中，缺口更大。

应补的组合具体是：

1. 训练相近状态+训练相近场景，以及训练相近状态+新场景，合计至少补6条；由现有2条的人工场景分类决定A/B分配。
2. 偏离状态+训练相近场景，以及偏离状态+新场景，合计至少补5条；由现有3条的实际资格/场景决定C/D分配。
3. far采集应包含臂姿态偏离且手姿尽量匹配的条件，同时保留手部偏离标记；不能全部只用手姿far来代表整个58D状态偏移。全部沿用冻结距离标准，不给机器人下发姿态指令。

本90来源作为探索资格库登记，不把人审后入组的来源包装成方法选型后的全新最终独立测试；最终锁定评价仍须另留未参与设计的新录制。

明确状态的5条集中在3个目录批次（A1103、B1103、A1106），其中2 near来自2个目录，3 far来自3个目录；A/B路线或同一天并不保证独立场景条件。当前无入组来源，因此无法证明任何格子具有跨批次稳定性。补录仍按匹配区组交错覆盖状态与场景，记录操作者、果串与实际物理批次，不以“4个ID”替代批次/场景覆盖。

## 8. 人工最小操作与继续方式

1. 打开human_review.html，先看5条明确状态来源的**完整**三路视频（可并排播放），确认实际摘取数量、是否完整成功，区分重试和多颗任务。90个来源均有完整播放与缺项入口，不必自行找文件。
2. 对照7组训练参考候选；确认代表性，记录果串构型/遮挡/初始相对位置以及具体参考ID。证据不足就保留unknown，不由界面自动推定。
3. 填审核人、结果理由并勾选已完整核对，导出reviewed JSON。未查看来源保持uncertain，场景无参考/理由保持unknown。原数据和初始技术配置不会被网页修改。
4. 交回JSON后核对source_id、证据SHA、审核时间、single/multi及场景证据，再用既有资格门重新生成覆盖统计；该步骤尚未发生，当前cell全部为null。人工结果不能覆盖冻结状态分类或重新定阈值。
5. 将连续多颗独立统计，按实际A–D缺口补录单颗成功来源。未建立交叉覆盖前，不运行方法比较或训练。

## 9. 工程、验证与提交范围

新增的必要程序已实际运行：collect_recording_provenance只读bag/conversion小文件；audit_recording_qualification复用已有reader/schema/audit，串行扫描约2.963GiB压缩视频而非大型bag正文；report_recording_qualification把真实结果物化成90条JSON、CSV及人工审核页。基础数学与资格门在trc.data.qualification，无模型依赖。

普通单元测试仍只需NumPy；Parquet测试按既有可选依赖运行。真实审核可选依赖为`.[review]`（PyArrow/PyAV/Pillow/PyYAML），本次复用已安装环境，无CUDA、OpenPI或ROS驱动导入，也未启动训练。113项测试通过，新增覆盖共同参考约束、冻结近/远/缓冲区、时间压缩反向映射、uncertain不能入格、不同裁剪/缺映射/卷大小不一致不能被错误认证。审核页还做了90卡/270本地播放器检查和JavaScript导出保护的**合成DOM测试**；合成测试没有写入任何真实成功审核结果。

可复现入口（路径在本地配置中，审计输出重跑须指定新目录）：

```bash
.venv/bin/python -m unittest discover -s tests -q
# 在可访问原bag与导出证明的数据服务环境，使用配置执行：
python scripts/collect_recording_provenance.py --config provenance_config.json
# 在可访问normal数据的只读审核环境，使用配置执行：
python scripts/audit_recording_qualification.py --config audit.local.json
# 在研究仓库本机，复用真实结果生成审核材料：
.venv/bin/python scripts/report_recording_qualification.py --config outputs/p2_1a/report.local.json
xdg-open outputs/p2_1a/human_review.html
```

源路径、原bag证明、运行配置和图像均在Git忽略的outputs下；实际90条审核状态配置按用户要求保存为configs/p2_1a_recording_review.json，不包含私人绝对路径、原始动作数组或视频。外部原数据、训练代码、权重及部署程序只读；本轮没有GPU训练、D0/D3新比较或机器人控制。只本地提交，不推送。

**结论：BLOCKED BY REVIEW/DATA。技术复核已完成90条，剩余阻塞明确为90条人工任务/场景审核，以及冻结分类下实际的near/far交叉覆盖缺口。**
