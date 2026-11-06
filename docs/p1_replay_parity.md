# P1-1：FLASH基线数值复现闭合

日期：2026-11-06。基于研究提交 `5248353`；原模型/数据/权重均只读。最终分析Commit与产物hash记录在本轮delivery manifest。

## 结论和适用范围

**PARITY_PASSED：在统一FP32/highest评价协议下，原训练模型API、P0系数导出图和研究Absolute适配入口达到严格数值一致。** 两个真实窗口的最大入口差 `4.76837158203125e-7 rad`，两次独立进程输出最大差0；严格门槛为1e-5 rad，没有放宽门槛。窗口0来自训练来源，173来自候选来源435fc6da，使用同一Step6000权重。

原0.00091 rad差异已定位到**低精度VLM计算对XLA执行图/编译策略的敏感性**这一机制：改变图的可观测输出或编译策略，会改变视觉前缀的数值，随后传播到系数/动作。它不是随机Flow采样，也未发现输入、关节顺序、统计或时间索引错误。历史那一次的已编译可执行文件和具体CUDA kernel未保存，因此不能追溯唯一kernel或逐位重造当次0.00091。这里通过同权重/同输入的可重复机制对照及精度干预闭合后续评价；不宣称历史BF16图也已经变成FP32容差。

FP32是**两组共同的评价设置**，不是B2算法组件；训练仍统一使用原BF16。旧P0-C/P1-0数字保留为历史BF16图结果，不将它们混作新FP32协议下的受控训练成绩。

## 固定身份与输入

- 原参考代码：`paper-flash-test@d208d414efbb78483f44bfef4c2a9314115dd632`，28服务器仍为此提交且跟踪文件干净。本机训练仓库当前另检出 `tomato-coarse-subtasks-agent@b37a06b`，未切换或修改它；源码调查通过Git对象读取d208d414。
- Checkpoint：正式FullFT6000，参数内容SHA256 `89da5fe135d6ae92c74f4f59634ff0c35bbf28301b85ffe5ce91bc3776931134`；原restore/load严格完整加载。
- 统计SHA256 `a5d7bd2dd422148736c69f1bf52a721d5770ba810f3bace9a2dd413395544d52`，训练契约SHA256 `50e26190703b466987ee5a8db9879fb04b43a5e221d3c07df5bcf8121c80e1e8`。
- 原始RGB逐字节hash验证；resize后的float32图像、state、prompt token/mask、8帧history、54帧target分别存入prepared快照。跨所有策略/进程逐数组核对完全相同。
- canonical LA/LH/RA/RH，单位rad，Absolute物理命令；history[-7,0]，target[-2,51]，评价[0,49]。没有重采样、padding或未来state作为推理输入。

## 三个入口与随机性

1. 原训练路径的 `Pi0Flash.sample_actions(key, observation, history, num_steps=1)`。
2. P0诊断图：原encode_condition、fit_history、coefficient_velocity，额外返回系数及直接sample输出。
3. 研究入口：Absolute坐标恒等适配，原Expert/解码，返回物理预测。

每个入口、每个真实窗口分别用固定key 0、1、42重复。最大差均0。源码 `sample_actions` 忽略key，推理source来自实测history，train=False、无高斯source噪声，单步Euler/NFE=1。训练中的history noise不属于推理随机性，因此不能用“随机推理正常波动”解释重放差异。

## 逐层证据

六个独立进程：默认BF16两次、禁用autotune/确定性算子BF16两次、FP32/highest两次。同一策略同一图的跨进程结果均一致；不同BF16图之间仍有偏差，说明“加deterministic flag”本身不能解决不同算术图的精度差异。

以下是**相同分层trace图、相同BF16模型、仅改变编译策略**的最大逐元素差；这是最早可观测差异定位，不声称能无扰动观察未保存的历史图内部。

| 层 | window0 | window173 |
|---|---:|---:|
| 原始/预处理图像 | 0 | 0 |
| state/prompt/mask/history输入 | 0 | 0 |
| 归一化history / 历史系数 | 0 | 0 |
| 视觉prefix tokens | 2.0 | 2.0 |
| 语言token区段 | 0 | 0 |
| 条件KV（第一个cache张量） | 1.1875 | .75 |
| Expert velocity | .029468 | .049379 |
| 物理动作（rad） | .006365 | .008741 |

prefix/KV/velocity行是各自内部坐标，不是rad；不能跨层数值直接比较量纲。最早不同的模块是SigLIP视觉前缀分支，归一化历史和源多项式相同；不是归一化反变换凭空制造rad误差。仅按相同C解码的basis/scale/q01/q99身份一致。

默认BF16下，原API与研究入口最大差0，但P0多输出图相对原API在window173可达.011954 rad；固定XLA flags的BF16图仍可达.007514 rad。额外返回中间量会改变编译图，故同图内“Prediction与Direct一致”不能证明跨图历史结果逐位一致。没有将这种差异当成模型学习结果。

联合使用显式FP32、高精度matmul和固定XLA策略后：

| window | 研究入口↔原API Max | P0入口↔原API Max | FP32跨进程Max |
|---|---:|---:|---:|
| 0 | 0 | 4.77e-7 rad | 0 |
| 173 | 0 | 4.77e-7 rad | 0 |

B0/B2集成前向还验证了batch2对两个单窗的原API最大差 `5.960464477539062e-7 rad`。后续正式配对评价固定batch1、无padding，而不是任意切换batch后假设位级一致。

## 后续共同评价协议

```text
model inference dtype = float32
JAX_DEFAULT_MATMUL_PRECISION=highest
XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true"
eval batch=1, no padding, NFE=1, complete valid (50,58) windows
```

使用原服务器固定运行环境（JAX0.5.3、原共享启动包装器及其库设置）和已核验GPU。换硬件/软件/精度/shape后重跑gate，不把本轮两窗通过外推到任意平台。完整CPU VLM推理未做；CPU图像预处理/host基矩阵与GPU各阶段已单独核对，没有把CPU/GPU交叉设备逐位相同当作要求。

`evaluate_fullft_paired_source.py`已最小扩展：可选CoordinateFlashConfig、显式FP32评价与adapter hash校验，仍复用原数据/模型/Polynomial。B0/B2模式拒绝缺失共享精度或XLA策略，系数解码与原sample路径继续核对。旧不带P1字段的Absolute导出入口保留，历史输出不覆盖。

## 产物与复现

`outputs/p1_1_replay/{default_a,default_b,controlled_a,controlled_b,fp32_a,fp32_b}/` 含prepared快照、各层张量、三入口输出、配置/运行时/输入hash。`parity_analysis/`含layer_differences、entry_parity、cross_process_parity CSV和 `parity_gate.json`。这些是真实诊断产物，默认不提交Git。

```bash
# 原环境中显式指定本研究脚本与配置；模板为configs/p1_replay.example.json
XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true" \
 JAX_DEFAULT_MATMUL_PRECISION=highest JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0 \
 XLA_PYTHON_CLIENT_PREALLOCATE=false \
 bash scripts/python.sh /path/to/research/scripts/trace_flash_replay_source.py \
 /path/to/research/replay.local.json
# 本研究环境中比较已保存的六次真实运行
.venv/bin/python scripts/compare_flash_replays.py \
 --root outputs/p1_1_replay --output outputs/p1_1_replay/new_parity_report
```

每次原环境输出指定新目录；编译flags在JAX初始化前设置。报告不记录未知的“历史唯一kernel根因”，而是保留观测、干预、限制和可复用的严格通过标准。
