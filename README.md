# Temporal Representation and Coordination

**P2-3已完成：三组各250步，12个Checkpoint真实配对评价完成。** 原480条单颗录制、冻结VLM和同一采样/优化器协议；131项测试及8卡BF16保存恢复通过。423连续窗RMSE：B1=0.008368、B2=0.008366、B3=0.008363 rad。B3仅有极小平均优势，左臂末9步与Max未超过B1，暂不扩大分阶模块；无独立泛化结论。见[完整结果、来源与逐阶统计](docs/p2_3_velocity_adapter_experiment.md)。

**最新进展（2026-11-11）：12项高误差条目已逐项核对来源、动作对齐、清洗映射及视频抽帧。** 其中6项（3个独立窗口）跨删除边界；3个有效左臂来源仍有明显远期预测偏离。选定第一个最小网络实验为按Legendre阶次的低秩速度读出，保留Absolute/K6/7 Token/Step8000，尚未实现或训练新Expert。见[证据、正常对照与具体修改方案](docs/p2_action_expert_mechanism_selection.md)。判定`READY FOR MINIMAL EXPERT IMPLEMENTATION`，不代表方法已有效。

**当前执行重点（2026-11-11）：回到all480单颗基本技能，保留连续采摘作为技能复用与数量/结束研究。** 已复用Step8000的97个训练来源、485个真实窗口生成12项高误差审核案例；全动作RMSE复现为0.009867 rad，偏移/形状MSE约52%/48%。本轮无新训练或模型前向。任务身份、已有结论的任务变化限制、实际产物与后续通过条件见[当前研究执行记录](docs/research_execution_focus.md)。下列P2-1A等为历史阶段记录，候选审核不再是唯一研究推进路径。

**历史阶段（2026-11-10）：P2-1A完成90个候选的实际技术资格复核，`BLOCKED BY REVIEW/DATA`。** 90条谱系/训练隔离/最低数值与时间轴合同通过；冻结状态2 near、3 far、85 ambiguous，完整任务及场景仍待人工审核，A–D合格均0。已提供90条审核卡与270路本地视频、7组训练场景参考；记录633个清洗拼接边界，后续窗口须限制在有效片段内。见[实际资格报告与优先5条审核](docs/p2_1a_recording_qualification.md)及[90条审核状态](configs/p2_1a_recording_review.json)。未训练或运行D0/D3新对比。

历史P2-1：**2026-11-10：P2-1状态—场景交叉覆盖检查完成，`DATA INSUFFICIENT`。** 589个唯一共享来源中，90个非rejected候选尚缺完整成功、谱系和场景核验，四格交叉覆盖仍未建立。已固定训练初始状态分层与原P2-0四折规则，没有新模型评价、训练或调参。下一步只复核候选并补齐A–D各至少4个探索成功录制。见[盘点证据、四格采集规格与停止条件](docs/p2_1_cross_coverage.md)。

历史P2-0：**2026-11-10：P2-0条件参考选择可行性分析完成，`INSUFFICIENT EVIDENCE`。** 97训练来源/4开发来源的录制隔离验证中，最近邻状态规则将开发RMSE从0.196709降至0.154930 rad，但训练RMSE上升10.86%；端点优势几乎与数据集合身份重合，未训练门控、未修改Expert。此前6个P1-7测试来源未参与本轮数值分析，后续最终评价须另锁新录制。见[完整P2-0证据与计算成本](docs/p2_0_conditional_reference_feasibility.md)。

历史P1-5：**2026-11-07：P1-5同Step8000初始化的B0/B2各250步配对微调与8×505窗评价已完成，`MIXED RESULTS`。** B0/B2终点RMSE：训练来源0.007730/0.015527rad，未拒绝候选0.242875/0.062107rad，失败示范0.215549/0.068484rad。B2候选主收益在零更新切换时已出现；训练97来源均差于原模型与B0，左臂末9步和尾部仍有代价，不能宣称独立成功泛化或实机改善。见[实际迁移结果、Step0效应与三层对照](docs/p1_5_step8000_transfer_results.md)。[P1-4权重核验](docs/p1_4_fullft_reuse.md)、[P1-3机制分析](docs/p1_3_mechanism_analysis.md)和[P1-2基础权重配对训练](docs/p1_2_pilot_training_results.md)保留各自比较边界；不自动延长训练或开发新Expert。P1-1的[复现证据](docs/p1_replay_parity.md)、[q0分析](docs/p1_reference_state_analysis.md)和[受控协议](docs/p1_small_budget_training_plan.md)保留追溯。

面向双FR3＋双Sharpa（58D）、三路RGB和π₀.₅系统的独立研究工程。基础包提供NumPy离线数据/误差分析；可选原环境适配层保留原FLASH架构，只研究动作坐标。代码保存在本仓库，默认main开发；原训练工程、数据和已有权重只读，仅在用户明确授权范围内执行训练，不进行机器人控制。

普通运行与基础测试不依赖旧仓库。只有用户显式选择的FLASH推理/训练准备入口使用固定版本OpenPI运行环境；不复制整个训练工程，不使用隐式源路径。历史P0-A/B/C与P1-0结果按各报告日期和数值协议解读，不将其BF16旧图指标混作新FP32受控评价。

## 安装与快速运行

Python≥3.10，基础环境只需NumPy；Parquet额外需要PyArrow。基础测试无需OpenPI、Torch、JAX、CUDA、ROS或机器人驱动；可选FLASH原环境入口另需已核验的训练依赖。

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[parquet]'
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m trc audit --config configs/p0_data_audit.yaml --output outputs/audit.json
```

默认配置未挂载数据，产生`blocked`报告并退出2，这是明确的未配置状态。实际使用时复制配置为`configs/my_dataset.local.json`，填写真实dataset路径和经核验的layout；相对路径基于配置文件所在目录。配置采用JSON语法（兼容YAML 1.2）。不要把示例中的统计参数视为已核验的正式实验配置。

## 统一命令

安装后可以使用`trc`或`python -m trc`：

| 命令 | 功能 |
|---|---|
| audit | 单/多数据源逐Episode审计、连续时间段、完整有效窗口数、来源和统计状态 |
| manifest | 元信息/Parquet内容指纹、数据源身份和全局Episode映射 |
| split | 按原始recording确定性分组，拒绝来源缺失或同源裁剪泄漏 |
| assets-check | T3 bundle、stats、SHA、schema、形状、实验参数与数据/变换契约校验 |
| roundtrip | 原始存储动作→训练目标→物理关节命令的离线往返核验 |

```bash
.venv/bin/python -m trc audit --config configs/my_dataset.local.json --output outputs/audit.json
.venv/bin/python -m trc manifest --config configs/my_dataset.local.json --output outputs/manifest.json
.venv/bin/python -m trc split --config configs/my_dataset.local.json --output outputs/split.json --seed 42
.venv/bin/python -m trc assets-check --config configs/my_t3.local.json --output outputs/assets.json
.venv/bin/python -m trc roundtrip --config configs/my_t3.local.json --output outputs/roundtrip.json --max-windows 3
```

`configs/p0_collection.example.json`演示多数据源；`configs/p0_t3.example.json`演示T3配置。用法、来源映射格式和状态解释见[使用指南](docs/usage.md)。原有`scripts/audit_dataset.py`保留为统一命令兼容入口。

## 已提供的能力

- 两种显式布局：canonical58为左臂、左手、右臂、右手；converter58存储为左臂、右臂、左手、右手，state116提取位置后重排。逐名称校验，不按维度猜测，拒绝54D Wuji混用。
- 双臂Delta以当前窗口起点实测state为锚，双手Absolute；提供全局Quantile、通用Z-score及T3双向bridge。解码返回canonical物理命令，不执行部署发送。
- 单Episode数字列读取，检查维度、float32、NaN/Inf、帧号、Episode ID、时间网格与原始时间映射。窗口不padding、不跨边界；显式boolean有效帧列可通过配置使用。
- 内容指纹与有序多源manifest；拒绝重复root/完全重复内容和不一致的相机/FPS。source_recording_id缺失时允许审计，但阻止正式split。
- 统计校验区分integrity_only、dataset_bound_transform_unpinned和verified。默认codec只接受与实际数据绑定且固定预期变换契约的资产；不生成虚构统计。
- 输出路径校验，拒绝覆盖配置、源映射、数据集及统计资产。数据、权重、私人配置与outputs产物默认不提交。

## 验证与当前限制

本机Python3.12验证：完整环境42项测试全部通过；仅NumPy环境29项通过、13项Parquet测试明确跳过。多分支数值对照共40项通过；T3资产加载与原实现的6项合成对照通过；真实tier_a首个Episode的首/中/尾三个窗口编码与物理往返通过。对照统计均明确标为合成，不替代真实统计。

新版多源审计读取本地tier_a、tier_b1、tier_c共143个Episode、129206帧，122199个完整50帧窗口。各目录可能存在同源或内容部分重叠，来源未核验前不视为独立示范数。另在P0-A已读episode284：1个Episode、1547帧、1498窗口。结果均为基础审计通过但有来源/统计警告。

以下为P0-A当时的T3状态；本轮FLASH Absolute验收与归因见顶部新报告。T3匹配bundle仍未在已检查路径找到；本地早期数据缺少所需原始recording映射，因此真实split按预期blocked。视频未解码、固定LeRobot完整Batch/最近邻/padding及部署行为未验证。`vector_contract_ready`仅表示本工具矢量契约条件满足；`formal_data_ready`目前始终false，不能把离线模块测试通过称作正式实验或实机验证完成。

详见[验证结果](docs/validation_results.md)、[数据契约](docs/data_contract.md)和[复用清单](docs/code_reuse_audit.md)。可选的源码一致性检查读取用户显式指定的旧仓库Git对象，不修改或切换其分支：

```bash
.venv/bin/python scripts/compare_upstream.py --source-root /path/to/tomato-openpi --check-assets --output outputs/upstream.json
```

## 目录

```text
├── README.md / AGENTS.md / pyproject.toml / MANIFEST.in / .gitignore
├── .github/workflows/tests.yml
├── configs/
│   ├── p0_data_audit.yaml
│   ├── p0_collection.example.json
│   └── p0_t3.example.json
├── src/trc/
│   ├── __init__.py / __main__.py / cli.py / config.py / io.py / project.py
│   ├── analysis/__init__.py
│   └── data/
│       ├── __init__.py / action_schema.py / lerobot_adapter.py
│       ├── validation.py / chunks.py / windows.py / manifest.py
│       └── semantics.py / normalization.py / assets.py / pipeline.py
├── scripts/audit_dataset.py / compare_upstream.py
├── tests/helpers.py / test_data.py / test_assets.py / test_project.py
├── third_party/openpi/LICENSE / NOTICE.md
├── docs/
│   ├── architecture.md / usage.md / validation_results.md
│   ├── code_reuse_audit.md / data_contract.md
│   └── research_hypothesis.md / experiment_plan.md / references.md
└── outputs/.gitkeep
```

研究与公平系数预算见[假设](docs/research_hypothesis.md)和[实验计划](docs/experiment_plan.md)。六阶Legendre每关节7系数，58D单段预算406。先完成真实数据/统计/Batch核验，再复用FLASH轻量工具开展P0-C，不提前更改模型架构。

P0-C（2026-11-05）：真实FullFT Step4000/6000已分别在固定505窗上完成配对诊断。训练来源与4个候选独立来源必须分开解读；尚无Step8000或物理协调失败证明。见[真实诊断报告](docs/p0c_fullft_diagnosis.md)和[唯一保留的P1最小方案](docs/p1_expert_design_options.md)。本轮未修改Expert、启动训练或控制机器人。

P1-0：已完成[物理偏移分解](docs/p1_offset_decomposition.md)、[状态相对坐标假设与真实Batch检查](docs/p1_state_relative_hypothesis.md)和[B0/B1/B2受控协议](docs/p1_controlled_experiment.md)。候选来源的整体偏移假设得到描述性支持；无参数坐标适配器已实现，B2尚未训练。基线跨执行图/进程的数值复现问题仍需闭合，当前不建议直接启动正式FullFT。

P1-1（2026-11-06）：[数值复现已闭合](docs/p1_replay_parity.md)，后续B0/B2共享FP32/highest评价；[q0首步残差与时移诊断](docs/p1_reference_state_analysis.md)保留真实命令—反馈差异；[配对500步训练准备](docs/p1_small_budget_training_plan.md)通过CPU gate，默认入口只做preflight。本轮未训练。状态：`P1 SMALL-BUDGET TRAINING READY`，不代表B2有效或最终论文验证完成。
