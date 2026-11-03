# Temporal Representation and Coordination

面向RA-L双臂灵巧操作研究的P0-A离线工程。平台：双FR3 + 双Sharpa，共58D，三路RGB，当前动作窗口50×58。当前不修改π₀.₅/Action Expert、不训练、不控制硬件；研究假设尚未验证。

用户确认主训练仓库为 `tomato-openpi`：`delta-paper-training-test` / `2ab68c8b8090284fca6d258f86bf5098d2f1302b`。先读[代码复用清单](docs/code_reuse_audit.md)和[数据契约](docs/data_contract.md)。本仓库独立保存研究所需代码，运行和基础测试不依赖旧训练仓库。旧仓库名称和Commit仅用于记录实现来源；完整OpenPI不是当前依赖。已补查本地main、缓存远端main、delta-paper、RTC+T3与paper-flash分支，分支关系及现成多项式工具见复用清单。

## 安装与命令

Python≥3.10，基础测试仅需NumPy与标准库unittest。Parquet读取额外需要PyArrow；无需CUDA、Torch、OpenPI或视频解码器。

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[parquet]'
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/audit_dataset.py --config configs/p0_data_audit.yaml --output outputs/audit.json
```

默认配置未指定真实数据，返回JSON `status=blocked`，退出码2；这是尚未配置的明确提示。审计有错误也返回2；基础检查通过且有研究阻塞提示时为 `passed_with_warnings`，退出码0。退出码0不代表T3物理恢复/视频/正式数据集已经验证。

配置是JSON语法（兼容YAML 1.2），不支持任意YAML语法，以减少依赖。复制成被Git忽略的 `configs/my_dataset.local.json`，按实际情况设置dataset_root、layout（canonical58或converter58）和可选norm_stats_path。相对数据/统计路径相对配置文件目录解析。不得按维度猜测layout；特征名称必须与契约完全匹配。所有数据路径来自配置，代码没有专有绝对数据路径。

```bash
.venv/bin/python scripts/audit_dataset.py --config configs/my_dataset.local.json --output outputs/my_dataset.json
# 可选：在有只读源代码的机器上核验已审阅的固定版本数值方法
.venv/bin/python scripts/compare_upstream.py --source-root /path/to/tomato-openpi --output outputs/upstream_comparison.json
# 可选增加真实episode首/中/尾完整窗口的编码对照
.venv/bin/python scripts/compare_upstream.py --source-root /path/to/tomato-openpi --dataset-config configs/my_dataset.local.json
```

输出含原始元信息、逐episode错误/警告、float32/58D/NaN/Inf、帧数、时间间隔与observed_fps、连续区间[start,end)、完整窗口数、四组维度、原始时间映射可用性和归一化状态。仅单episode数字列驻留内存，视频不加载。原始数据始终只读。

## 目录与文件清单

除原有README更新外，下列为本次新增文件；`*.local.json`、`.venv`与outputs中的JSON为本机产物、不提交。

```text
.
├── README.md
├── AGENTS.md
├── pyproject.toml
├── .gitignore
├── configs/p0_data_audit.yaml
├── src/trc/
│   ├── __init__.py
│   ├── analysis/__init__.py
│   └── data/
│       ├── __init__.py
│       ├── lerobot_adapter.py
│       ├── action_schema.py
│       ├── semantics.py
│       ├── normalization.py
│       ├── chunks.py
│       └── validation.py
├── scripts/
│   ├── audit_dataset.py
│   └── compare_upstream.py
├── tests/test_data.py
├── docs/
│   ├── code_reuse_audit.md
│   ├── data_contract.md
│   ├── research_hypothesis.md
│   ├── experiment_plan.md
│   └── references.md
└── outputs/.gitkeep
```

## 本次验证结果（2026-11-03）

环境Python3.12、NumPy2.5.4、PyArrow26.0.0。17项unittest通过，含临时合成Parquet集成测试；无硬件指令。固定主训练Commit的10项原始数值方法对照通过，浮点rtol=atol=1e−6，关节顺序/窗口索引为精确相等。对照通过不表示完整训练Batch或部署程序已验证；原始类只读按AST选取，不导入模型/驱动。另以tier_a首个真实episode（859帧）的起点0、405、809验证Delta/Absolute及物理命令往返，均通过；该验证未使用不匹配统计。

| 已读取的本地真实数据目录（简称） | Episodes | 帧数 | 完整50帧窗口 | 基础审计 |
|---|---:|---:|---:|---|
| episode284 | 1 | 1547 | 1498 | passed_with_warnings |
| tier_a | 7 | 5962 | 5619 | passed_with_warnings |
| tier_b1 | 62 | 53785 | 50747 | passed_with_warnings |
| tier_c | 74 | 69459 | 65833 | passed_with_warnings |

上述目录可能相互重叠，不把144个episode读取记录称作独立示范数。各目录action/state元信息均为canonical58/float32，fps=30；所有episode均未发现维度错误、NaN/Inf、时间断点或不足50帧。Parquet中均缺少source_timestamp_ns，所读episode元信息不含所需来源映射；局部连续不证明裁剪前物理连续。私有配置与完整JSON报告已保留本机configs/outputs，均被Git忽略；默认无数据配置的blocked报告也已生成。

已确认canonical顺序为左臂0:7、左手7:29、右臂29:36、右手36:58；converter存储另一布局，经主训练适配源码核验。Delta以窗口起点实测位置为锚，双手Absolute。归一化通用公式和T3 bridge已与源方法对照；本次对照统计明确为合成测试数据，未用于真实数据解码。

## 尚未完成与下一步

- 服务器311/480条正式训练集、连续多颗数据及匹配T3 bundle/norm_metadata/SHA尚未挂载确认；本次真实审计是本地早期数据，不替代正式数据审计。
- 原始recording/裁剪映射、有效监督掩码、相机同步，以及固定LeRobot版本下的全Batch/nearest timestamp/padding一致性仍待验证。
- 后续需要的项目代码正常整理到本仓库，保留来源与第三方许可证；不通过机器上的旧仓库路径拼接运行环境。
- P0-B先补齐正式数据和统计身份，核验真实编码/解码、来源映射与Batch；P0-C再按[计划](docs/experiment_plan.md)实现E0/E1和预算核算，之后做E2–E4。六阶Legendre每关节7系数，58D单段预算406；所有异构/分段方法必须控制总预算。

研究假设、可证伪条件见[研究说明](docs/research_hypothesis.md)；FLASH、Coordi-π、RTC、MambaAct、T-Rex的原始来源和研究关系见[参考文献](docs/references.md)。本地提交不推送远端。
