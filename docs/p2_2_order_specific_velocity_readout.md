# P2-2：Order-Specific Low-Rank Velocity Readout

2026-11-11；独立分支`feat/order-specific-velocity-readout`，起点`3bc2272`。

**总体交付状态：BLOCKED_WITH_REASON。** 模块已实现，真实Batch FP32前向/反向验收通过，B1/B2/B3各完成5步CPU完整Expert功能Smoke及状态恢复；沿用上轮方案的8卡GlobalBatch256 BF16预检和配对250步尚未执行。26/28两台8卡机器均由其他训练占用，没有挤占、停止或恢复它们。本报告没有250步结果，也不作B3优于B2的判断。

## 1. 真实结构与保留的FLASH语义

依据只读固定依赖`paper-flash-test@d208d414efbb78483f44bfef4c2a9314115dd632`：

- `FlashPolynomial`使用`eye(7)`逐列生成Legendre基，Token索引0…6分别是0…6阶。阶次不是频率分箱。
- target扩展范围−2…51共54帧；history为−7…0的8步实测位置；输出及评价0…49共50帧。保留原历史拟合的重参数化、ridge=.1及目标边界投影，不重写拟合。
- `coefficient_velocity`输入`(B,7,58)`，`action_in_proj`得到`(B,7,1024)`；原Transformer返回同样的hidden形状，原速度头为共享`nnx.Linear(1024,58)`，输出`(B,7,58)`、FP32。
- 这是**系数空间沿flow时间τ的速度**，不是关节物理角速度。FM目标`C_target−C_source`，source按原规则加入训练噪声；consistency目标及权重保持不变。
- 归一化仍是原Absolute动作Quantile仿射，`F(A)=2(A−q01)/(q99−q01+1e−6)−1`；拟合系数除原7项scale。解码为`F⁻¹(Basis·diag(scale)·C)`。无新统计、裁剪或q0平移。
- NFE=1时`C_pred=C_source+v(C_source,τ=0)`，所以`C_pred−C_gt`等于该推理点的velocity残差；**不能**称它为所有随机训练τ的FM误差。真实Batch另外计算了原生FM/consistency loss及反向。

## 2. 逐阶证据：可试验，尚未证明必要

实际读取三个连续左臂窗口（原export行62、2、464）及9个连续、同源低误差对照，共12个不同窗口。没有使用旧候选/连续任务来源选择新超参数。336行四组×七阶统计见`outputs/p2_2/order_residuals.csv`，同时保存target scaled coefficient的均值/标准差、scaled残差MSE、原scale/实际basis/Quantile跨度映射后的物理分量、阶间signed allocation和Oracle指标。

|组别|三个focus窗口最大物理self-energy的阶次分布|九个对照的分布|
|---|---|---|
|左臂|0阶2个，2阶1个|0阶8个，3阶1个|
|左手|0阶3个|0阶8个，1阶1个|
|右臂|0阶3个|0阶8个，1阶1个|
|右手|0阶3个|0阶7个，1阶1个，2阶1个|

三个左臂focus的mean standalone component MSE，按0…6阶为`[8.325e−4,4.056e−4,2.898e−4,4.627e−5,6.328e−5,9.673e−6,1.296e−5] rad²`；对照为`[1.375e−4,2.595e−5,1.631e−5,1.738e−5,4.560e−6,4.314e−6,2.908e−6] rad²`。self-energy不是可直接相加的总误差。考虑阶间交互的signed allocation中，focus左臂0/1/2阶合计89.14%，对应上一轮已核验的结果。

结论是存在低阶主导、但随窗口和关节组变化的误差结构；**没有发现统一的高阶失败模式，也没有证明共享读出是原因**。基函数及原缩放本身也影响物理分量，不能从这些数值直接推出专门化必然更好。原Transformer已有位置条件，不能声称原模型没有阶次信息。保留统一适配及普通微调，正是为了检验是否需要阶次专门化；不增加K或Token，不改变所有阶的loss权重。

Oracle及有效性：三个左臂focus全程Oracle RMSE分别0.008110、0.008629、0.005943 rad，预测↔Oracle为0.042527、0.046247、0.035408 rad；优先问题仍是预测偏离。此前12条case中的6条跨清洗切口，未纳入本轮focus/对照分析。精确接触事件仍未人工确认，不新增协调失败结论。

## 3. B1/B2/B3实际实现及参数量

设原hidden为`h[b,k,:]`、原共享速度读出为`hW+b`。

- **B1：none。** 完整保留原读出，不增加参数；后续仅更新原Expert，VLM冻结。
- **B2：unified。** `v=hW+b+(hV)U`，V为1024×11，U为11×58，所有阶共享。
- **B3：order_specific。** `v[b,k]=h[b,k]W+b+(h[b,k]V)U[k]`，V为1024×8，U为7×8×58。

|版本|新增参数计算|新增参数|原Expert＋Adapter可训练参数|
|---|---|---:|---:|
|B1|0|0|430151738|
|B2|1024×11＋11×58|11902|430163640|
|B3|1024×8＋7×8×58|11440|430163178|

普通无bias统一低秩层参数量必为`r×(1024+58)=1082r`，11440不可整除1082。rank10为10820，差−620；rank11为11902，差＋462，故选择更接近的rank11，新增参数多4.04%。没有用无效参数、特殊截断层凑数。两种结构的函数类不同，参数接近不等于严格容量相同，也不能把B2称为B3的容量超集。

B2/B3都令U=0，V以独立固定seed42003、fan-in标准差`1/sqrt(1024)`非零初始化；不会全部矩阵同时为0。原Step8000参数完全相同；新增矩阵因形状不同而并非逐元素相同，但使用相同初始化分布与固定seed，不消费原模型/采样/flow随机流。

实现路径：

- `src/trc/models/flash_order_readout.py`：`ReadoutFlashConfig/ReadoutFlash/ResidualVelocityHead`。继承原`coefficient_velocity`全部过程，只替换最后Linear；原kernel/bias保留原参数路径。none模式连头包装都不加入。
- `src/trc/models/velocity_readout.py`：NumPy可测的形状、残差、参数计算和严格参数扩展；不依赖JAX/OpenPI。
- `ReadoutFlashConfig.warm_start`：校验完整旧参数SHA、集合、shape及dtype，只有`action_out_proj/adapter_down`与`adapter_up`允许新初始化。增强checkpoint恢复使用相同mode的严格加载；不能将warm-start当作恢复函数。
- `src/trc/data/flash_sampling.py`与可选`models/flash_order_data.py`：复用原数据窗口/图像/观测实现，只在已证明的连续区间上取样，并把切口sidecar加入新产物。

理论新增MAC/样本/一次velocity forward：B2约83314，B3约60592；不是实际延迟测量。没有第二个完整Expert、视觉编码器、坐标选择器或新token。

## 4. 当前可读的Step8000权重身份

原训练仓库的`checkpoints/`目录在本轮检查时已空，原Step8000路径不可直接读取。本轮没有修改或清空它。

实际使用先前保留的P1-5 B0 Step0快照作为恢复源。它的**完整参数SHA**仍为：

`40b2c2c4dcdd6fb88dd847d5762d2071fba5028af0942c6dfe457409fe5f5919`

与已核验Step8000完全相同。没有用B0-FT250、B2-FT或近似权重代替。匹配统计SHA为：

`a5d7bd2dd422148736c69f1bf52a721d5770ba810f3bace9a2dd413395544d52`

各版本完整加载原VLM＋Expert，并重新计算旧参数子树SHA；新增参数单独计数。新优化器全部重置，未读取旧Step8000 Adam作为某一组的初始化。原保存目录、数据及部署代码只读。

## 5. FP32真实Batch数值验收

实际使用B370/source `be79493d531944aba5bc9e85eeaa7af2`，当前帧848，Batch1，三路真实RGB、合法state/history，与冻结GT/history逐元素一致；支持区间不跨切口。调用固定原源码的读取、tokenization、CPU图像resize和FLASH方法。

在隔离CPU运行环境、最高matmul精度下：

|检查|B1|B2|B3|
|---|---:|---:|---:|
|原生velocity最大差|0|0|0|
|物理动作最大差 rad|0|0|0|
|原生loss最大差|0|0|0|
|禁用Adapter的velocity差|原路径|0|0|
|完整Expert梯度有限|通过|通过|通过|
|进入反向的VLM叶子|0|0|0|

真实hidden为`(1,7,1024)`、velocity为`(1,7,58)`、动作`(1,50,58)`。三个初始真实Batch FM＋consistency loss均为0.0396182872。

完整Expert梯度叶子数B1/B2/B3为19/21/21，初始范数分别1.927594、1.930600、1.928301。初始U梯度非零，V梯度为0；读出单独更新后V梯度变非零，符合零残差初始化预期。B3对只涉及阶3的读出loss，其他U切片梯度最大值为0。

另有明确标为synthetic的小型JAX测试：非零U时关闭Adapter仍逐元素等于原Linear；B3仅改变指定阶输出/梯度。它不替代上述真实Batch验收。

跨设备重放：本轮CPU原生FP32预测对既有GPU FP32相同窗口，最大差`1.192093e−6 rad`，RMSE差`1.779038e−7 rad`，通过既定1e−5 rad阈值。没有放宽容差。

证据：`outputs/p2_2/acceptance.json`、`full_smoke_acceptance.json`、`cross_device_replay.json`、`routing_checks.json`及`runtime_versions.json`。这不是8卡BF16验收；后者尚未执行。

## 6. 已执行Smoke Test的准确范围

为避免争抢GPU，限制为服务器CPU的8个核、`CUDA_VISIBLE_DEVICES=''`、`JAX_PLATFORMS=cpu`：

1. 在真实Expert hidden上对各读出运行10步Adam功能检查，检查新增梯度及小型读出状态保存恢复。
2. **随后各自重新从同一原参数初始化**，B1/B2/B3分别执行5步完整Expert更新；不是从上面的10步结果继续。
3. VLM参数逐字节不变。各组Batch1、相同真实连续窗口重复使用、同一固定视觉前缀，FP32，flow/noise key按同一步号变化，无图像增强；AdamW(.9,.95,1e−8)、wd1e−10、clip1、固定LR2.5e−6。
4. 所有loss/梯度有限，无OOM或NaN。每组保存完整可训练Expert＋Adapter＋Adam的薄checkpoint，VLM由原参数SHA引用；每份约5.16GB，保存在独立研究输出目录。恢复全部参数/优化器叶子逐元素一致，恢复前后物理预测最大差0。

|功能更新步|B1 loss|B2 loss|B3 loss|
|---:|---:|---:|---:|
|1|0.01975834|0.01975834|0.01975834|
|2|0.00963152|0.00963084|0.00963064|
|3|0.00837790|0.00837500|0.00837600|
|4|0.03310394|0.03310264|0.03310086|
|5|0.11448265|0.11448313|0.11447931|

不同步的随机time/noise不同，不能用这条loss曲线判断收敛；逐步配对设置相同。没有把CPU Batch1/固定前缀改名为要求的8卡GlobalBatch256 BF16训练。

### 单窗口功能预测指标（不是研究终点评价）

|模型|Full RMSE rad|MAE rad|左臂末9步RMSE rad|P99 rad|>0.5rad元素|
|---|---:|---:|---:|---:|---:|
|B0原冻结参数|0.01613407|0.00737016|0.07526833|0.06659215|0|
|B1 CPU5步|0.01105936|0.00665844|0.02101873|0.04954077|0|
|B2 CPU5步|0.01106095|0.00666137|0.02101757|0.04954171|0|
|B3 CPU5步|0.01105905|0.00665957|0.02100588|0.04953837|0|

它们在同一个已训练过的窗口继续更新再评价，三组几乎一致。**不能据此宣布B3优越、B1充分、泛化改善或250步收益。** 它只是说明模型/梯度/解码和恢复可工作。完整四组、各阶、Oracle/Prediction/Total、偏移/形状MSE、50步误差、相邻变化、P95/P99/Max保存在`cpu_smoke_metrics.json`及`cpu_smoke_groups.csv`，scope明确为single-window functional smoke。

## 7. 250步统一协议及实际资源阻塞

沿用上一轮拟定的GlobalBatch256/8卡预算，`configs/p2_2_velocity_readout.json`固定B0不更新，B1/B2/B3各250步；VLM全部冻结、保持原FP32权重字节，计算BF16，GlobalBatch256、8卡。相同旧参数、全部原Expert更新mask、seed/model42/data42001/augmentation42002/flow42，Adapter独立seed42003；Adam统一重置；25步warmup至2.5e−6，其后固定；保存0/5/50/100/250。模型仍为Absolute/K6/7 Token/50×58，不沿用P1-5 FullFT的可训练VLM设置。

**初始化注意：** 原`train.py::init_train_state`会把冻结参数转为BF16。直接照搬会改变Step8000 VLM权重的字节和FP32重放。必须用本轮严格warm_start保留旧参数dtype，再对训练mask构造新Adam及FSDP状态；不能关闭VLM冻结来绕过，也不能把dtype变化吞进容差。本轮CPU功能入口就是不做该隐式转换的路径。8卡初始化、BF16前向、分片状态和完整训练入口仍需在空闲资源上验收，不能宣称已通过。

生产训练每个样本必须使用自身合法当前观测计算前缀；不得将本轮单窗口Smoke的缓存用于其他样本。保持原augmentation、FM source noise及consistency路径，不能只为Adapter改loss。

资源检查时，26的8张RTX PRO5000各占约53.4GB；28各约66.2GB，均由已有训练占用。某一时刻利用率暂降不代表该卡空闲。未建立可用8卡独占资源，未尝试分配GPU或启动250步，没有自行改变正式Batch/卡数。不会后台等待后自动开训。

### 共同训练窗口正确性修复（已实际完成）

只读取7组已有conversion JSON，逐一绑定冻结480来源的ID、长度、源起止时间；不重读bag或全量Parquet/video。得到1546个切口。按所有history/target支持区间−7…51限制：

- 原压缩时钟窗口423430。
- 原始连续支持窗口**362918**，排除60512。
- 480来源、451270帧、动作及统计未改；三组共用相同过滤和permutation种子。
- 250×256=64000样本的共同采样序列已在本地及实际NumPy1.26.4训练运行环境分别生成，三组hash完全相同，见`sampler.json`和`runtime_sampler.json`。不是已经执行了64000个训练样本。

原485个训练诊断窗中62个跨界，统一可用423窗，仍覆盖97个来源。其冻结B0 RMSE=0.00950533 rad；这是过滤后子集，不是Step8000变好了。新的评价清单固定到`evaluation_windows.json`，三组后续都用这423窗；不能给新方法单独删除困难窗口或与旧485总体直接做因果比较。连续任务及反复使用的旧候选不混入本轮单颗评价。

此处只为训练正确性补了必要的索引与proof，不新增数据管理框架。配置校验拒绝逐组超参数覆盖、未冻结VLM、未固定连续性sidecar或错误参数预算。

## 8. 可运行入口与产物

基础环境：

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/analyze_order_residuals.py \
  --export outputs/p1_4_fullft/export_step8000 \
  --cases outputs/action_expert_cases/case_attribution.json --output outputs/p2_2
.venv/bin/python scripts/report_readout_checks.py \
  --output outputs/p2_2 --basis-export outputs/p1_4_fullft/export_step8000
```

真实模型检查使用显式固定OpenPI/JAX环境及配置：`check_order_readout_source.py <private_config.json>`；它强制CPU且拒绝覆盖输出。需要配置source root/manifest、保留的参数checkpoint、统计、录制、参考数组和新输出目录，没有隐式旧仓库导入。`full_expert_smoke_steps=5`是本轮功能检查，不是250步训练开关。后台CPU任务均已退出。

源码实际复用原系数速度函数、视觉条件、polynomial、数据读取与图像预处理，不复制或修改原训练仓库；仅可选固定版本runtime依赖。原共享训练/部署仓库、原始数据及已有权重未写入。

主要产物在忽略的`outputs/p2_2/`：

- 逐阶CSV/JSON、`order_residual_structure.png`。
- 真Batch前向/梯度验收、跨设备重放及环境版本。
- 完整Expert CPU5步记录、薄checkpoint身份、`cpu_smoke_loss.png`、全部物理指标。
- 480来源切口proof、共同采样hash、423窗固定清单。
- 配置、脚本、参数、归一化和产物SHA清单。

约15.5GB的新Expert＋Adam状态保留在配置指定的独立服务器研究输出；只回传小型身份文件及预测，不把大型权重复制进Git。每个trial的step5身份记录说明需加载同一原VLM及配置才能恢复，不将薄checkpoint伪称自包含完整VLA。

## 9. 下一轮唯一执行顺序与研究判断

基础单元测试124项通过；另外实际完成固定runtime的真实模型/梯度/状态恢复与小型synthetic非零Adapter关闭/阶次路由检查。它们说明实现可工作，不证明算法有效。

下一步等待**明确空闲**8卡后，先验收保留FP32 frozen参数的FSDP初始化、Step0 FP32一致性及BF16 GlobalBatch256的5–10步Smoke/完整训练checkpoint恢复，再顺序执行B1/B2/B3匹配250步。当前没有一条已通过验收的8卡启动命令，不把准备配置说成训练已运行。

判断保持用户标准：B3稳定优于B2才支持阶次专门化；B2相当则选统一适配；B1相当则不需要新结构；都不改善左臂远期/形状/尾部则重新审查机制，不扩大模型。现有CPU单窗结果不能应用这些算法去留标准。

**BLOCKED_WITH_REASON：最小实现及CPU真实Smoke已完成，正式8卡BF16 Smoke与三组250步因资源占用未执行，算法效果尚无结论。**
