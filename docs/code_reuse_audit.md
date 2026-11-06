# 代码复用与来源清单

用户确认主训练工程为tomato-openpi。本仓库整理其适用的P0模块，代码保存在仓库内；无跨仓库运行依赖。源训练仓库/数据/权重只读，未fetch或切换源分支。以下远端引用指本地缓存，不断言远端最新状态。

## 主训练分支

| 引用 | 已核验Commit | 已检查内容 |
|---|---|---|
| 本地main | 4d0cb07e02181ed3bcdd876cad69446b22cfb075 | 58D协议、converter116位置适配、Delta/Absolute、Normalize/Unnormalize；基础契约和Delta分支相同 |
| origin/main、github/main缓存 | 897cb917097c1d56a3fa7a517bf561b9e878036e | 比本地main多RTC模型/config/UI相关提交；基础数值方法另作对照 |
| delta-paper-training-test | 2ab68c8b8090284fca6d258f86bf5098d2f1302b | T3桥接、bundle/normalization provenance、完整窗口、集合与采样配置 |
| github/feat/trainrtc-correlated-t3缓存 | bd2ee4077b236e9f64dbf135625269de8bf576e1 | RTC+T3集成；核心shared/delta_paper、training/delta_paper和multimodal适配与Delta分支无差异 |
| paper-flash-test | 534ed023542a20dad55e1827b0a5adad9b760d4d | NumPy多项式、JAX FLASH表示和source级数据分割；作为P0-C复用候选 |

本地没有名为delta-paper的精确分支；对应现有引用为delta-paper-training-test及其后续RTC+T3集成分支。

## 当前模块的复用方式

| 原始路径（tomato-openpi） | 功能/58D | 依赖 | 方式与本仓库对应模块 |
|---|---|---|---|
| policies/fr3_sharpa_protocol.py | LA7,LH22,RA7,RH22逐关节名称与rad | NumPy | 本仓库action_schema保存独立显式契约，逐名称与原ORDER对照 |
| policies/fr3_sharpa_multimodal_policy.py | converter116位置与converter58动作重排 | OpenPI/图像 | 提取必要的数值映射到action_schema，无模型/图像依赖；与main及Delta源码逐值比较 |
| transforms.py | 锚点Delta/Absolute、quantile/zscore | 完整导入链重 | semantics和normalization本地数值实现，保留epsilon与无clip行为；四分支正逆变换对照 |
| shared/delta_paper.py | T3 bridge、bundle/形状/哈希校验 | 数值部分NumPy | 适配为normalization和assets；保留源算法/许可证与Commit，增强数值、schema与身份检查；未引入噪声训练算法 |
| training/norm_provenance.py | 数据文件内容与变换contract绑定 | OpenPI config/inspect | assets与manifest用显式契约替代训练config导入；比较实际repo/数量/FPS/files及独立expected contract |
| training/dataset_collection.py | 多源Episode映射和feature一致性 | 标准库 | manifest本地实现，进一步检测重复内容/来源映射；不重写原ID |
| training/delta_paper.py | CompleteWindows与PaperSharpaData | Torch/JAX/OpenPI | chunks/windows提供离线接口，额外校验时间/validity；原窗口索引对照通过；模型/训练适配类不引入 |
| training/data_loader.py、sampling.py | LeRobot采样、Torch Batch与epoch_shuffle计划 | Torch/JAX/LeRobot | 仅读调查；未声称完整Batch等价，P0不复制训练采样器 |
| scripts/flash/polynomial.py | K为degree、K+1系数；普通0..49与扩展-2..51拟合 | 仅NumPy | P0-C优先复用候选，本次未提前实现E0–E4 |
| shared/flash_polynomial.py、training/flash_dataset.py | 历史重参数化/缩放/边界目标、source split与RGB | JAX/图像/远程读取 | 只读参考。54为扩展时间采样点数，不是54D动作；上下文窗口不能等同N−50+1 |

原始路径相对于src/openpi，表中scripts路径除外。Apache-2.0文本及适配来源见third_party/openpi/LICENSE和NOTICE.md；用户自有工程代码按要求正常整理复用，不增加常规审批。代码路径不指向机器上的旧训练仓库。

## 历史对照来源

| 仓库/分支 | Commit | 读取与判断 |
|---|---|---|
| pi05/main | 09bab9d792aeed178644575efda40753cf974e51 | examples/fr3_wuji及policy是54D/20D手，不能套用58D Sharpa |
| sharpa-pi05/train-rtc | a3997fbd72a6e758707f61c316d994962090f87e | 早期canonical58协议、30Hz导出、裁剪save_episode与source_start/end报告 |
| convert_data_job-worker/auto-convert-transfer-ui | 8fea5fea8d6f1a2ee03e2f9d9e80c134cadf5bdb | schemas/sharpa_contract.yaml、formats/lerobot_v21.py、training/normal.py确认converter58/116、原始时间与段边界可保存；只读，没有导入转换或控制链 |

## 一致性覆盖与边界

compare_upstream.py从显式指定源仓库的固定Git对象读取已审阅定义，不切换分支、不写源目录，不属于普通运行依赖。main与缓存远端main各9项、Delta与RTC+T3各11项，共40项数值/映射/窗口检查通过。--check-assets另用临时合成bundle比较原load_bundle与本地assets，正常元信息/数组精确一致，四种配置不匹配和损坏哈希均拒绝（6项）。真实早期tier_a的首个Episode首/中/尾锚点编码与往返亦通过。

合成统计只用于数学与失败路径验证，没有用于真实数据恢复。实际训练使用的T3 bundle仍缺失；视频同步、完整LeRobot解码/nearest timestamp/padding/Batch及部署发送未验证。FLASH分支已有结果仅确认存在，未作为本项目已复现实验。当前code完整性不等于正式实验完成。

## P0-C FullFT 实测补充（2026-11-05）

`paper-flash-test@d208d414efbb78483f44bfef4c2a9314115dd632`：原服务器与本地核验提交一致。下列均为58D适配后的真实源码核验，不根据文件名推断；无原训练文件修改。

| 原仓库/Commit | 原文件 | 功能及58D | 重依赖 | 许可证/方式 | 对应研究模块 |
|---|---|---|---|---|---|
| tomato-openpi / d208d414 | `src/openpi/training/flash_native.py` | 原生window/observation、三RGB resize、state/action统计契约，Absolute58D | Torch/JAX/视频环境 | 原仓库既有许可及定制来源见NOTICE；Adapter，未复制实现 | `scripts/evaluate_fullft_paired_source.py` |
| 同上 | `src/openpi/models/pi0_flash.py` | 原HAF、7×58 velocity、VLM条件、单步推理、物理解码 | Flax/JAX/OpenPI | Adapter；未复制或修改Expert | 同上 |
| 同上 | `src/openpi/shared/flash_polynomial.py` | K6 history/54帧target拟合与50帧解码 | JAX | Adapter；不重写拟合 | 同上 |
| 同上 | `src/openpi/models/model.py`、`training/flash_checkpoint.py` | 严格完整模型加载、参数身份 | Orbax/Flax | Adapter；不加载驱动/训练循环 | 同上 |
| 研究仓库原创 | `src/trc/analysis/flash/diagnosis.py` | 导出后逐阶物理误差归因、时间/合法当前state诊断 | NumPy | 本仓库实现；不是FLASH模型 | `scripts/diagnose_fullft.py` |

实际Step4000/6000各505窗完成原API数值一致性核验，C_pred为直接张量，未使用历史线性恢复代替FullFT系数。代码、契约、权重hash及结果见 `p0c_fullft_diagnosis.md`。原代码运行依赖只由用户显式指定的原环境提供；普通研究包和基础测试无跨仓库导入、专有路径或CUDA依赖。

## P1-0 状态相对坐标（2026-11-05）

仍以 `paper-flash-test@d208d414` 为源。`scripts/audit_p1_sources.py` 只读复用原read_numeric与camera映射；`scripts/check_state_relative_source.py` 显式原环境加载原权重、原NativeDataset、原VLM条件、原Polynomial及coefficient_velocity/sample API。没有复制或修改Expert。新增 `trc.models.flash_coordinates` 是本仓库原创的无参数可逆坐标层，NumPy基础测试可独立运行；源环境仅在用户指定的一致性检查中调用。原ridge历史先拟合再平移，不重写历史锚定。复制数值、来源/相机/统计身份、真实Batch结果与数值复现限制见三份P1文档，B2未训练不作为性能结果。

## P1-1：固定原始架构的实验入口（2026-11-06）

只读依据仍为 `paper-flash-test@d208d414`，不跟随本机当前其他工作分支。

| 原始模块 | 复用方式 | 研究对应变化 |
|---|---|---|
| `Pi0Flash` / `Pi0FlashConfig` | 继承、Adapter；原Absolute直接super | `trc.models.flash_openpi`只增加静态coordinate_mode及必要的系数坐标变换，无新参数 |
| `FlashPolynomial` | 原fit/target边界/reconstruct | 无更改；保留ridge历史先fit后平移 |
| `scripts.train_flash.main/step` | 显式原runtime Dependency | `prepare_p1_training_source.py`仅替换模型Config和新输出/共同预算；优化器/FSDP/噪声/采样/Checkpoint不复制 |
| `NativeDataset.index/observation` | 原接口 | 两配置前128000窗口逐项hash核对；预处理与时间/动作映射不重写 |
| 原 `sample_actions` | 原API数值对照 | 分层trace定位低精度图敏感性；共同FP32三入口严格gate |

小段原系数loss/积分逻辑在派生类中适配，来源及Apache-2.0许可见third_party/openpi/NOTICE.md。普通NumPy包不隐式导入OpenPI；这些模块仅由用户显式指定的原运行环境使用。训练启动开关本轮未调用。
