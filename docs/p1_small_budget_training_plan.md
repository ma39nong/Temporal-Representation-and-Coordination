# P1-1：B0/B2受控小预算训练准备

日期：2026-11-06。

**P1 SMALL-BUDGET TRAINING READY**，限定为本轮核验的源代码、资产、共享评价策略与配置。尚未启动B0/B2训练，未证明B2有效，未验证500步的梯度/显存稳定性或收敛；这些正是下一阶段小预算试验要回答的问题。

## 通过的启动前条件

- 基线三入口在共享FP32/highest策略下最大差4.77e−7 rad；两次独立进程相同。具体机制、严格容差与硬件/shape范围见 `p1_replay_parity.md`。
- 原Step6000真实Batch=2集成检查：B0预测和带source noise的Loss与原实现最大差均0；B0/B2直接系数解码与原生推理最大差均0；batch2对两个单窗差5.96e−7 rad。
- B0/B2参数布局与全部FullFT可训练参数相同，3353487146个参数。真实检查加载同一已有参数数组对象，没有新Parameter；这只是功能检查权重，不作为B2训练起点或成绩。
- CPU预检查对两个配置分别遍历前128000个采样位置，序列SHA256均为 `356ccdde7cc1ac7956110106cc56dfc0db2dd19e486e8c451362fb0bfe229ee1`。
- 新训练统一从原pi05_base及相同原初始化流程开始。基础参数目录全部文件做内容SHA，两个配置绑定同一文件清单身份；不把已有6000模型与新模型不公平比较。新Expert随机参数使用相同model_seed。未执行完整GPU参数/优化器初始化，真实初始化继续由原 `init_train_state` 完成。
- Checkpoint/统计/数据/源目录保持只读；新Checkpoint、W&B offline和临时文件在独立研究输出目录。两种模式不允许覆盖、自动resume或每组另加超参数。

实际CPU preflight报告 `outputs/p1_1_replay/p1_pair_preflight_ready.json` 保存基础权重文件内容身份、共同采样hash、两份解析配置、参数布局、gate/code hash与新输出目录。报告 `training_started=false`。

## 实验定义与公平性

| 项目 | B0 | B2 |
|---|---|---|
| 模型 | 原π₀.₅ VLM + 同一Transformer FLASH Expert | 相同 |
| 参数/容量 | 3353487146参数，K6，7×58系数Token/406系数 | 相同，无新增参数 |
| 输出 | Absolute物理命令50×58 | 加回合法参考后仍是相同Absolute命令 |
| 模型系数坐标 | 原Absolute | 同一原fit结果减理想常量系数 |
| history / target / eval | [-7,0] / [-2,51] / [0,49] | 相同 |
| 物理数据 | all480，451270帧，423430合法窗口 | 相同 |
| 统计 | 原训练全局action/state Quantile | 相同，未从候选来源计算新统计 |
| 训练dtype | bfloat16 | 相同 |
| 更新范围 | FullFT，VLM与Expert全部参数 | 相同 |
| 优化/随机/采样/预算 | 以下共同配置 | 相同 |
| 评价 | 以下统一FP32协议 | 相同 |

共同配置：500 steps、global batch256、fsdp_devices8；model_seed42、flow_seed42、data_seed42001、augmentation_seed42002；原epoch_shuffle、无分组加权。500步共128000个Chunk采样位置，约0.30个合法窗口池遍历；这不是充分训练预算。

AdamW：b1=.9、b2=.95、eps=1e−8、weight_decay=1e−10、clip_gradient_norm=1。原warmup1000、peak_lr=2.5e−5、decay_steps30000、decay_lr=2.5e−6保留。500步仍在warmup中；不为B2单独缩短warmup或调大学习率。若决定1000步，必须两组同时改预算、生成新协议/输出目录并重做preflight；仍不是最终方法优劣结论。

## B2究竟改变什么

`trc.models.flash_openpi.CoordinateFlashConfig`继承原Config，`CoordinateFlash`继承原Pi0Flash。B0的loss/sample直接调用super；B2仅在原history/target fit与Euler前后加入既有 `FlashCoordinates`。原Polynomial、VLM条件、Transformer、参数初始化、优化器、FSDP、采样和Checkpoint管理均复用，不复制整个OpenPI。

原source S与target T共同减合法q0对应的理想常量系数Q；所有样本各用自己的history[-1]。相对物理动作满足 `R=A−q0`，相对归一化用原action跨度的斜率，不直接套用F(R)。原ridge历史先fit再平移，不改history拟合器。推理只用观测/历史，不用GT首步或未来反馈；不会强制R[0]=0。

**这个最小方案的FM速度监督仍是原T−S（含同一source noise），并没有因坐标平移降低其幅度或增加信息量。** 改变的是端点系数/输入坐标及必要的参考加回路径，检验条件学习的归纳偏置。q0本来就在合法输入中；不增加传感器、token或模型容量。π₀.₅的state token化与原history压缩也意味着精确参考加回是计算路径的变化，不能把未来收益唯一解释成某种目标统计量下降。

本轮不把State-Relative/Delta本身描述成学术创新，不添加Mamba/MoE/双Expert/RTC、分组网络或新Loss。B1强制P[0]=q0仍只作离线诊断，不进入模型训练/控制。

## 共同评价协议

1. 新B0/B2都使用FP32、高精度matmul、固定XLA flags、batch1、不padding、NFE1；原训练器内部BF16的64个训练来源monitor可以保留为训练日志，但**不能替代本研究的规范配对评价**。
2. 同一505窗口plan、相同GT/history/RGB身份。训练来源485窗与4候选来源20窗分开，逐source保存，不称候选结果为充分独立泛化。
3. 原 `evaluate_fullft_paired_source.py` 已支持显式coordinate_mode和训练adapter SHA，匹配Checkpoint训练契约后用统一FP32推理，保存原Absolute坐标下GT/Oracle/Prediction/C_gt/C_pred，继续核对原sample API。不是把旧6000 BF16结果直接当新B0受控对照。
4. 原评价模块输出MAE/RMSE、Oracle误差、50步曲线、四组及聚合组误差。`analyze_reference_state.py`复用现有指标/偏移分解，补充训练定义的初始状态/q0分层、source统计；CSV明确记录coordinate_mode，避免两组均为Step500时身份混淆。
5. 指标必须包含整体偏移MSE与形状MSE；并列重复q0与B1，防止“改进”只是把轨迹变成保守静止。四候选来源已用于设计，确认性论文结果需另留未参与选型的新recording/会话；当前不报告显著性或物理协调成功率。

保留：Checkpoint内容身份、source/研究Commit、训练/评价配置、基础权重内容身份、adapter/wheel哈希、统计、采样序列与固定评价plan。新环境/设备变更须重跑parity gate。

## 入口与启动约束

基础测试无OpenPI/CUDA依赖：

```bash
.venv/bin/python -m unittest discover -s tests -v
```

原环境配置模板：`configs/p1_small_budget.example.json`。本轮实际路径仅在被忽略的 `configs/p1_small_budget.local.json`；研究wheel解包在原工程外的显式research_runtime，不改原共享环境。以下命令从核验的原训练source工作目录调用，脚本和配置路径指向研究副本。

```bash
# 默认仅CPU预检查：本轮已经执行，不创建优化器或进行更新
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu \
 bash scripts/python.sh /path/to/research/scripts/prepare_p1_training_source.py \
 --config /path/to/research/configs/p1_small_budget.local.json
```

**本轮未执行**的下一阶段启动形式如下，必须由用户显式选择B0或B2；两者使用同一配置、分别新建输出。脚本会重新核验gate、adapter、基础权重和采样身份。存在Checkpoint目录时拒绝自动续训/覆盖；无GPU或设备数不匹配也拒绝启动。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 JAX_PLATFORMS=cuda \
 bash scripts/python.sh /path/to/research/scripts/prepare_p1_training_source.py \
 --config /path/to/research/configs/p1_small_budget.local.json --start-training B0
# B0完成并保留结果后，同一协议显式选择 --start-training B2
```

原 `scripts.train_flash.main` 执行训练。新路径只换模型Config与输出/预算，不改原训练文件；日志默认W&B offline，目录在新输出内。检查报告可重用但不覆盖：关键身份不一致时拒绝启动。不是“预检查成功后自动训练”。

新Checkpoint评价模板为 `configs/p1_evaluation.example.json`，两组分别填写匹配的训练契约、权重、coordinate_mode与adapter SHA：

```bash
XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true" \
 JAX_DEFAULT_MATMUL_PRECISION=highest CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda \
 XLA_PYTHON_CLIENT_PREALLOCATE=false \
 bash scripts/python.sh /path/to/research/scripts/evaluate_fullft_paired_source.py \
 /path/to/research/configs/p1_B0_eval.local.json
# 使用同一评价协议导出B2，再在研究环境运行现有报告/分层入口
.venv/bin/python scripts/analyze_reference_state.py \
 --exports /path/to/B0_export /path/to/B2_export \
 --audit outputs/p1_state_relative/source_audit.json --output outputs/new_B0_B2_strata
```

在新确认集结果之前预注册主指标/可接受退化：source级Full Action物理RMSE为主，四组P99、偏移/形状及时间曲线共同约束。若只改善4个发现性来源、损害训练来源/某组尾部或不能优于保守基线，不扩展训练预算。不能用500步的一次seed宣布算法胜出；后续匹配预算和多seed需要单独决定。

## 仍然未知与不构成当前阻塞的事项

- 没有B2训练成绩，未验证梯度与显存的连续训练稳定性；小预算试验正用于检验学习有效性。
- 没有充分独立确认集、完整raw bag内容证明或可靠FK/事件标签；不能声称普遍泛化或协调失败。
- q0与命令的系统性差异需作为真实残差学习；未做硬件延迟/零位校准，不执行机器人控制。
- 历史BF16可执行kernel未保留，但后续共享FP32严格复现门槛已经通过；不能混用两个数值协议的旧新指标。

当前这些限制不阻塞**离线、发现性、配对500步试验的启动准备**，但阻止将其包装成最终RA-L验证。最终状态：**P1 SMALL-BUDGET TRAINING READY**。本轮训练更新数为0。
