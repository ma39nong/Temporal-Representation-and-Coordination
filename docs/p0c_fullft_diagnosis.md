# P0-C：正式 FullFT 的配对误差诊断

核验日期：2026-11-05。所有下列实验数值来自真实数据和真实 Checkpoint；合成数据只用于单元测试。研究仓库从 `b89170b` 继续开发，未重建 P0-A/B 数据框架。最终分析结果内记录实际分析代码 Commit、文件哈希及输入封印。

## 结论与当前状态

正式 FullFT **Step4000 与 Step6000 已完整加载，并分别完成同一组 505 个窗口的新推理**。训练尚未结束；本次最后一次只读快照为 2026-11-05 16:30:48 北京时间，日志到 Step6400，目录存在完整 2000/4000/6000，没有 Step8000。因此不能回答“最终充分训练模型已收敛”，也不能报告不存在的 Step8000 结果。

训练来源窗口明显改善，但候选独立来源的误差未随 4000→6000 明显改善。最显著异常是候选来源的零阶系数偏移，已从时刻0出现。现有证据支持先验证一种保持 K6 容量的条件坐标改造；不支持宣称 K6 对灵巧手表示不足，也没有物理协调失败证据。P1 最小方案见 `p1_expert_design_options.md`，本轮没有实现或训练新 Expert。

## 输入验收与版本身份

P0-B 最低数值条件继续满足：480 Episode、451270 帧、423430 合法窗口；Absolute 58D，全局 action Quantile，不使用 T3 Delta/Per-Timestamp Z-score。模型物理输出是绝对关节命令，单位 rad；历史是实际关节位置，不是未来 command。

| 项目 | 核验身份 |
|---|---|
| 原训练代码 | `paper-flash-test`，`d208d414efbb78483f44bfef4c2a9314115dd632` |
| 实际配置快照 | `training_configs.experiments.tomato_sharpa.flash.snapshots.flash_all480_b512_step8000_20261104`；实际运行 batch=256，不能由模块名称推定512 |
| 运行标识 | `mengxiang_sharpa_flash_absolute_k6_s1_all480_fullft_s42_1104_162556` |
| 训练契约 SHA256 | `50e26190703b466987ee5a8db9879fb04b43a5e221d3c07df5bcf8121c80e1e8` |
| 数据 manifest SHA256 | `d27dd1d976f8766d096658f97c415a98e42576b4604e93c4845bd4e5b42368c2` |
| 归一化 SHA256 | `a5d7bd2dd422148736c69f1bf52a721d5770ba810f3bace9a2dd413395544d52` |
| 固定历史窗口 plan SHA256 | `d666f5a62e92e4ddc8df4a00c52df9eea984d33530ddd3cbb9b4c37fa4545fa4` |
| Step4000 参数内容 SHA256 | `5dd7bd15629940419e5567c778f265548e21a73d49845242a04f08d461836eda` |
| Step6000 参数内容 SHA256 | `89da5fe135d6ae92c74f4f59634ff0c35bbf28301b85ffe5ce91bc3776931134` |

两份权重都通过 Orbax 完成标记和参数元信息检查，以原 `model.restore_params` 和严格 `model.load(remove_extra_params=False)` 加载，没有随机补齐或部分加载。51个参数叶：视觉23叶/414803696元素，语言9叶/2508531712元素，Expert及投影19叶/430151738元素，共3353487146元素。加载全部推理参数；没有恢复优化器或验证其继续训练状态。配置与数据 preflight 必须等于训练契约，`freeze_vlm=False`。

实际推理在原服务器原 Python 环境中执行，脚本与产物放在独立研究输出目录；仅使用一张 GPU 的受限内存，不启动优化器。原仓库跟踪文件保持干净，Checkpoint 前后文件大小与mtime清单一致。没有修改原始数据、Checkpoint、训练程序或控制训练进程。参数未完整下载到本机，本机缓存只有统计和元信息，不伪称存在本地完整模型。

## 505 个窗口的身份与时间对齐

| 范围 | Source 数 | 窗口数 | 解释 |
|---|---:|---:|---|
| 当前 FullFT 训练来源 | 97 | 485 | 来源在 all480 训练清单中；是训练来源诊断，不能称泛化测试 |
| 候选独立来源 | 4 | 20 | 4个连续多颗番茄来源不在当前训练来源清单中；仅小样本分布外探针 |
| 合计 | 101 | 505 | 混合数值只供追溯，不作为总体泛化性能 |

4个候选来源为 `0340d869ea074774b09441f2339cec30`、`435fc6da99284d7ba15cf09193352c06`、`b710689fe5084126b0ab3dedde821f29`、`e4bdeafd7d834062bfba2ad01049a5b8`。按原始 recording 身份排除当前480来源重叠；不能据此证明所有机器人预训练数据也独立。它们的连续任务、时长与训练分布可能不同；20窗口不等于20个独立试验。这些探针已用于选择下一方案，后续论文确认性实验必须另留未用于选型的新来源/采集会话。

每窗 `(50,58)`，30Hz：历史 ticks `[-7,0]` 的8帧；原目标 `[-2,51]` 的54帧，Oracle 使用全54帧拟合，只有 `[2:52]` 即 ticks `[0,49]` 用于误差评价。不能将54帧称为另一个预测 Chunk。固定合法窗口遵守 Episode/连续片段边界，无 padding、无跨边界采样，评价 mask 全有效；保留已核验 P0-B 源时间映射的适用范围，不补造缺失的事件标签。

58D 模型顺序：Left Arm `[0:7]`，Left Hand `[7:29]`，Right Arm `[29:36]`，Right Hand `[36:58]`。从原数据协议和现有 schema 验证，不根据7+22猜测。

另一个必须保留的原FLASH细节：history的实际观测时刻是[-7,0]，但 `fit_history` 将这8帧重参数化到 middle-grid 前8行（对应target ticks[-1,6]），并非在物理历史时刻上做常规外推。本轮原样复用，未将此实现习惯静默改为另一个拟合算法。其输入仍全为合法历史。target边界约束使用真实扩展监督，只用于Oracle/训练target，推理不接收未来端点，也不额外做未来端点投影。

相对于历史505导出，两次 FullFT 的 GT 和8帧 history 逐值完全一致；1515张原始解码 RGB 的像素哈希一致。FullFT 使用自己训练时的原生 `NativeDataset` 图像 resize、state归一化与 tokenizer，不能为了强行相同而套用历史预处理。4000和6000之间配置、统计、GT、history、图像身份、Oracle及原始系数均完全一致，适合配对诊断。

时刻0的历史最后一项严格等于当前 measured state，而评价 A[0] 是同一时刻的 command。两种信号不相同：GT command0 与 measured state0 的 RMSE 分别为训练来源0.033270、候选来源0.037831 rad。这是合法当前反馈与目标命令的区别，不构成将未来标签当输入；也不能把预测第0步强行设成 measured state。

推理函数只接收 observation/history，不接收 target。实际原模型中替换第一窗未来 action/state，验证 observation/history 及 Prediction 不变、监督目标改变；该实测扰动覆盖一个窗口，其余所有窗口沿用同一无 target 的推理入口。原生 `sample_from_cache(num_steps=1)` 与导出的系数解码在全部505窗口的最大差异均为0.0。C_pred 来自模型直接张量；历史 Step250 的系数仍为已验证线性恢复，不能将其称原生输出。FullFT与历史 Oracle 的最大物理差为 `9.5367431640625e-7 rad`。

## 配对科研结果

定义 A=GT，O=Oracle，P=Prediction；表示误差 O−A，预测差异 P−O，最终误差 P−A。所有指标先在同一有效物理关节值上计算，RMSE/MAE不可相加；报告保留MSE交叉项。下表是 pooled RMSE，单位 rad。

| 范围 | Oracle | 历史冻结 Step250 Final | FullFT4000 Final | FullFT6000 Prediction–Oracle | FullFT6000 Final |
|---|---:|---:|---:|---:|---:|
| 训练来源485窗 | 0.002798 | 0.080250 | 0.014828 | 0.011405 | 0.011727 |
| 候选独立来源20窗 | 0.002383 | 0.110065 | 0.228767 | 0.227932 | 0.227965 |
| 混合505窗 | 0.002783 | 0.081638 | 0.047789 | 0.046717 | 0.046800 |

4000→6000 的训练来源 Final RMSE 下降约20.9%，候选来源下降约0.35%。它们来自同一 FullFT 运行、同一窗口与归一化，支持“后续优化继续改善训练来源，但尚未解决候选来源异常”的描述；不证明最终 Step8000 必然如此。

历史 Step250 与 FullFT 的冻结策略、训练来源、统计、条件预处理和更新预算不同，表中差异不是只改变步数的受控实验。混合505平均还掩盖了训练来源与候选来源的巨大差别。

### Step6000：哪些组最难预测？

以下是最终动作误差的绝对值分位数，单位 rad。完整表示/预测/最终三类、七组及每时间步指标在 JSON/CSV。

| 训练来源 | MAE | RMSE | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|
| Left Arm | .012140 | .018027 | .036221 | .063251 | .236872 |
| Right Arm | .010168 | .013371 | .026710 | .039234 | .081225 |
| Left Hand | .006370 | .011561 | .020723 | .042558 | .428990 |
| Right Hand | .004568 | .008281 | .014872 | .031146 | .339560 |
| All Arms | .011154 | .015871 | .031412 | .052413 | .236872 |
| All Hands | .005469 | .010056 | .018038 | .037806 | .428990 |
| Full Action | .006841 | .011727 | .022887 | .043116 | .428990 |

| 候选独立来源 | MAE | RMSE | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|
| Left Arm | .245528 | .355791 | .753251 | 1.253641 | 1.474739 |
| Right Arm | .174530 | .219470 | .407763 | .579371 | .617433 |
| Left Hand | .172846 | .250673 | .568312 | .805879 | .914854 |
| Right Hand | .085845 | .136255 | .325946 | .506784 | .860985 |
| All Arms | .210029 | .295597 | .581877 | 1.082509 | 1.474739 |
| All Hands | .129345 | .201746 | .465731 | .733547 | .914854 |
| Full Action | .148821 | .227965 | .505580 | .772713 | 1.474739 |

平均上 Left Arm 最大，训练来源最大单值则出现在 Left Hand。不能从手部最大值反推出表示阶数不足或特定操作阶段。

### 什么时候最困难？

训练来源第0步/49步 RMSE 为0.012012/0.015679，49次相邻转换中30次增大，最差为49步；不能说处处单调。候选来源为0.220625/0.235795，46次转换增大、峰值46步：大误差在第0步就存在，远期增长不是唯一原因。训练来源误差最高10%窗口承担约32.2%的MSE；候选来源约17.0%，并非仅一个异常窗口解释全部问题。

四个候选 source 的 Full Action RMSE 分别约0.222917、0.259013、0.161730、0.254825，均较大。最大值案例为 window173，source `435fc6da99284d7ba15cf09193352c06`，Episode2/frame3864，预测tick49，left_fr3_joint7，1.474739 rad。其余组的最差窗口及50步 GT/Oracle/Prediction 曲线保留在 worst_group_cases.csv / Figure4，不凭帧号命名抓取、摘取或释放阶段。

### 哪个系数阶次影响最大？

不能直接把训练坐标的系数 MAE 混合排名。令原基矩阵为 B、原缩放为 s、Quantile 跨度为 d，则每阶物理误差分量为 `e_k = B[:,k] * s[k] * (C_pred[k]−C_gt[k]) * d/2`（包含原实现epsilon）。逐窗验证 `sum_k e_k = P−O`，报告最大数值残差。为保留跨阶干涉，使用 signed allocation `mean(e_k * sum_l e_l)`；其和等于预测MSE，可为负，不能当作独立正的因果重要性。

训练来源0阶分量 RMSE 为0.008469，signed allocation占预测MSE约55.3%；候选来源0阶分量为0.226444，约98.75%。用真实未来 C_gt 仅替换0阶后，最终 RMSE 分别变成0.008097和0.025015。这是使用未来标签的 Oracle 敏感性分析，**不能用于实际预测，也不能当作新方法成绩**。结果表明候选来源主要为持续偏移，尚无证据指向需要提高 K6 阶数。

### 合法的额外 sanity checks

只重复当前 measured state 50次的无学习基线，在训练来源/候选来源的 RMSE 为0.055988/0.067696。FullFT训练来源优于它，但候选来源反而更差。该基线只用于离线诊断，不能据此生成机器人控制策略；它同样不是 command0 真值。

候选来源当前state超出训练 q01/q99 的元素比例约8.79%，history约8.62%，future GT约10.11%；训练来源约1.93%/1.71%/1.71%。这些是输入分布变化线索，不能证明归一化就是因果原因，不能将超分位数数据当成无效帧删除，也不能利用候选来源重新拟合统计后宣称公平泛化。

## 协调问题的证据边界

已有逐关节误差、相邻时间差分误差、时间曲线和异常窗口可以核验。当前没有经过核验的整机 URDF/关节方向/基座外参/FK，也没有可靠的抓持接触和操作事件标注；Episode元信息含source时间/segment，但不含这些事件。因此不报告双末端相对位姿、抓持稳定性分数或任意拼接的“协调分数”。左右及臂手误差不均匀是动作层观察，不能等同机器人协作失败。需要后续带同步执行日志、任务结果和可靠运动学/事件的验证。

## 复用与新增边界

直接复用原实现：`flash_native.NativeDataset.observation/window/image`、`bind_stats/preflight`、`model.restore_params`、`Pi0Flash.encode_condition/coefficient_velocity/sample_from_cache`、原 `FlashPolynomial.fit_history/fit_target/reconstruct`、原归一化与状态 tokenizer。模型、Flow及History-Anchored Flow没有重写。研究仓库新增的只是显式原环境评估脚本和 NumPy 事后归因；普通测试不依赖 OpenPI/JAX/CUDA 或原仓库路径。

研究代码新增：`scripts/evaluate_fullft_paired_source.py`、`scripts/diagnose_fullft.py`、`scripts/plot_fullft_diagnosis.py`、`scripts/compare_fullft_checkpoints.py`、`src/trc/analysis/flash/diagnosis.py`、对应测试和配置模板。现有 report 只修正 fresh/cached 的限制文字。无模型结构修改。

54项基础测试通过，涵盖线性分量/相互抵消/Oracle修复、解码不一致拒绝、时间曲线与第0步一致性；原有50项继续通过。既有40项跨分支数值对照属于P0-A/B记录，本轮没有伪称重新运行这些对照。新增真实数值一致性：两Checkpoint各505窗与原采样API一致（最大差0.0）、同GT/history/RGB、跨Checkpoint完全相同Oracle、第一窗未来扰动无泄漏以及每阶物理解码闭合。

## 产物与复现入口

所有真实数组、私人路径及图表保留在被Git忽略的 `outputs/p0c_fullft/`：

- `export/`（Step6000）、`export_step4000/`：GT、Oracle、Prediction、原生C_gt/C_pred、history/current_state、mask、原basis/scale/stats、逐窗身份、配置、参数hash与seal。
- `step6000_final/`、`step4000_final/`：JSON、七组CSV、50步误差、分阶物理贡献/交互、source分组、最大误差窗口；Figure1–4原有图及Figure5来源分层/Figure6分阶贡献/Figure7来源时间曲线。
- `paired_fullft_final/`：4000/6000严格配对CSV/JSON、Figure8训练进展。
- `paired_inputs.json`、`reference_arrays.npz`：固定505输入身份、RGB哈希和参考真值；`final_training_snapshot.txt`：只读训练状态证据。

实际路径不写死在研究模块中。先在原训练环境的核验提交上，使用显式本机配置运行 source 脚本，输出必须位于原工程之外；该脚本需要原训练依赖，但基础测试和离线归因不需要它们。模板 `configs/p0c_fullft_paired.example.json` 的训练/数据hash必须替换为真实身份，不填虚构值。

```bash
# 研究环境：核心只需NumPy，画图需analysis额外依赖
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/diagnose_fullft.py \
  --export outputs/p0c_fullft/export \
  --historical-export outputs/flash_attribution/export_step250_b89170b \
  --output outputs/p0c_fullft/new_step6000_analysis \
  --current-stats /path/to/matching/norm_stats.json
.venv/bin/python scripts/plot_fullft_diagnosis.py \
  --diagnosis outputs/p0c_fullft/new_step6000_analysis --export outputs/p0c_fullft/export
.venv/bin/python scripts/compare_fullft_checkpoints.py \
  --exports outputs/p0c_fullft/export_step4000 outputs/p0c_fullft/export \
  --output outputs/p0c_fullft/new_paired_comparison
# 原运行环境：从显式指定的只读source工作目录运行，脚本/配置/输出均位于研究目录
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda \
 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.37 \
 bash scripts/python.sh /path/to/research/scripts/evaluate_fullft_paired_source.py \
 /path/to/research/configs/p0c_fullft.local.json
```

这些评估入口拒绝覆盖已有输出；切换权重须更改配置 checkpoint、expected_step及输出目录，保持同一paired inputs/reference身份。显存比例是本次服务器的运行参数，其他设备须按实际可用资源调整，不属于算法设置。未来完整8000落盘后沿用相同入口；不等待它而持续扩建数据设施。

## 仍需什么

1. 完整Step8000及其匹配契约，才能回答当前预算终点表现；当前没有最终收敛证明。
2. 更多按原始recording/采集会话隔离、未参与选型的独立连续任务数据，才能检验真实泛化与候选坐标方案。
3. 核验预训练重叠、FK/双基座外参及同步事件/执行结果，才能讨论物理协调和实机失败。

当前可以支持：候选来源存在远大于K6重建误差的预测偏移，且0阶占主导；训练来源预测误差随优化下降。当前不能支持：新Expert有效、K6不足、模型已充分收敛、所有独立数据泛化差、或机器人已发生协调失败。
