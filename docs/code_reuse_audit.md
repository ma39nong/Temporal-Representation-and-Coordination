# P0-A 代码复用调查（2026-11-03）

先调查后实现。指定研究路径原先不存在，已从用户给定 GitHub 地址克隆；原始 HEAD 为 504fdb4b8d7ebac5aa34cbbbe713d578cb2feb49，仅 README。当前聊天目录是独立的未提交资料仓库，未改动、未复制配置导出。

| 原始仓库/分支 | Git Commit | 原始文件路径 | 实际功能 | 58D | 重依赖 | 许可证/公开权限 | 复用方式 | 研究模块 |
|---|---|---|---|---|---|---|---|---|
| pi05/main | 09bab9d792aeed178644575efda40753cf974e51 | examples/fr3_wuji/convert_dataset.py; src/openpi/policies/fr3_wuji_policy.py | Wuji 54D、20D 手部重排；非 Sharpa | 否 | LeRobot/OpenPI | 根 Apache-2.0；项目代码按用户要求正常复用 | Skip | 无 |
| sharpa-pi05/train-rtc | a3997fbd72a6e758707f61c316d994962090f87e | src/openpi/policies/fr3_sharpa_protocol.py | 明确左臂7、左手22、右臂7、右手22，rad | 是 | NumPy | 根 Apache-2.0；项目代码按用户要求正常复用 | Adapter（独立契约，未复制源码） | action_schema.py |
| sharpa-pi05/train-rtc | 同上 | src/openpi/training/config.py; data_loader.py | pi05_fr3_sharpa 等配置为50×58，delta mask 7,-22,7,-22；LeRobot按 round(t*fps/action_hz)/fps 取未来时间 | 是 | Torch/JAX/LeRobot | 同上 | Adapter（直接读标准元信息/Parquet，不模拟训练采样器） | lerobot_adapter.py; chunks.py |
| sharpa-pi05/train-rtc | 同上 | src/openpi/transforms.py; shared/normalize.py | 起点state减/加；quantile epsilon=1e-6，无clip；norm_stats JSON根为norm_stats | 是，但通用维度 | OpenPI导入链重 | Apache-2.0；不复制 | Adapter（独立数值接口，与原方法对比） | semantics.py; normalization.py |
| sharpa-pi05/train-rtc | 同上 | examples/fr3_sharpa/export_lerobot.py | 实测state/命令action；裁剪各段save_episode；导出报告记录source_start/end_index及crop_interval | 是 | LeRobot/图像 | 项目代码按用户要求正常复用 | Skip（仅读调查） | data_contract.md |
| tomato-openpi/delta-paper-training-test | 2ab68c8b8090284fca6d258f86bf5098d2f1302b | src/openpi/policies/fr3_sharpa_multimodal_policy.py | 转换器116D state提取位置，58D action由LA,RA,LH,RH重排为LA,LH,RA,RH | 是 | OpenPI/图像 | 根 Apache-2.0；项目代码按用户要求正常复用 | Adapter（未复制源码，数值接口对照） | action_schema.py |
| tomato-openpi/delta-paper-training-test | 同上 | src/openpi/shared/delta_paper.py | ArmNormalization在全局quantile与双臂zscore间桥接；ARMS=0:7,29:36；std=max(std,std_floor)；bundle SHA校验 | 是 | 数值类仅NumPy；完整pipeline重 | 同上 | Adapter（独立数值实现；原类只读动态对照） | normalization.py; compare_upstream.py |
| tomato-openpi/delta-paper-training-test | 同上 | src/openpi/training/delta_paper.py | CompleteWindows筛选每episode完整窗口N-H+1；frame_index必须0..N-1 | 是 | Torch/JAX/OpenPI | 同上 | Adapter（离线窗口额外拒绝时间断点；不宣称训练loader完全等价） | chunks.py |
| convert_data_job-worker/auto-convert-transfer-ui | 8fea5fea8d6f1a2ee03e2f9d9e80c134cadf5bdb | schemas/sharpa_contract.yaml; src/convest/formats/lerobot_v21.py; training/normal.py | 原始action=LA,RA,LH,RH；state116D含速度；source_timestamp_ns；segment及source边界可写入episodes；normal投影保留原矢量 | 是 | PyArrow/视频/转换链 | 项目代码按用户要求正常复用；保留第三方声明 | Skip（只读记录协议，不复制） | data_contract.md |

## 证据范围与工程独立性

三个指定参考仓库及额外匹配服务器提交的 tomato-openpi 初始均无未提交修改。分支/Commit来自本地Git，不等同于确认服务器当前进程。聊天已有2026-11-02导出记载服务器分支 delta-paper-training-test、共享Commit 2ab68c8…，本地 tomato-openpi 恰好匹配。未连接服务器；311/480数据集与T3统计不能宣称已审计。

现有轻量数学与映射可通过接口复用；完整LeRobot训练Batch、图像、增强、sampler及部署都未导入。本次以本仓库内的轻量实现完成接口，未引入训练模型或驱动。用户已明确这是自有工程整理与独立测试，后续所需项目代码正常复用到本仓库，保留来源与第三方许可证，不增加常规复用审批。本仓库对照脚本仅在用户本地显式指定源目录时抽取已审阅数值方法执行，禁写bytecode，不导入整个训练工程，不写源目录。

## 一致性验证边界

compare_upstream.py 对同一带唯一值的合成输入验证真实源方法：协议ORDER、116→58位置提取、58D重排、Delta/Absolute、Quantile/Z-score及T3 bridge、CompleteWindows索引。数学对照用明确标为synthetic的统计，不代替真实统计。实际训练LeRobot解码、nearest timestamp选择、padding及Batch逐样本一致性仍须在固定LeRobot环境中做P0-B；本阶段审计使用完整窗口且拒绝不连续，不冒充所有训练loader的边界策略。

## 主训练仓库确认与Batch细节

用户进一步明确 `tomato-openpi` 才是实际训练仓库，后续一切训练语义以它为主，另外两份仅历史对照。主仓库 `src/openpi/training/data_loader.py` 实际导入LeRobot/Torch/JAX：TransformedDataset应用repack→data transforms→Normalize→model transforms，TorchDataLoader将窗口交给torch.utils.data.DataLoader；epoch_shuffle用sampling.epoch_shuffle_indices构造固定预算计划及PlanSampler，支持按已完成steps seek。普通/加权采样路径仍独立，不能把epoch_shuffle行为外推到所有配置。P0不复制Batch sampler，只读取episode并提供完整窗口索引。

本次真实数据读取来自早期canonical58布局，与主训练converter116路径分开记录。归一化资产只读查看到早期actions统计为[58]；未把它冒充T3的[50,58]统计。对照结果及数据集覆盖数量在README，完整产物在Git忽略的outputs。

工程约定：来源仓库路径不属于运行依赖。src/trc内使用包内导入；只有可选的compare_upstream.py在显式指定--source-root时读取上游源码，日常审计和基础测试不需要旧训练仓库。数据及统计仍通过配置指定，不能混同为跨仓库代码依赖。
