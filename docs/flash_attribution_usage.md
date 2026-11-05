# FLASH 归因入口

## 复用边界

研究包的metrics/report只依赖NumPy和可选Matplotlib；它不实现FLASH模型。显式source adapter在指定的原OpenPI环境子进程中复用原代码。source/runtime/data/checkpoint路径全部来自本机配置，不作为本仓库常规运行依赖。原工程不修改，第三方来源见third_party/openpi/NOTICE.md。

`flash_source_bridge.py`的cached模式读取原flash_data.load_episode/make_batch与shared.flash_polynomial.FlashPolynomial，重新构造C_gt并核对原Oracle。历史预测文件没有保存C_pred，因此通过原54帧解码矩阵的线性逆恢复其scaled系数，并验证再解码误差；明确标为recovered，不宣称是直接导出的原模型张量。没有重新执行一个替代Expert。

native模式复用原load_experiment/bind_stats/preflight、NativeDataset.observation、model.restore_params及cfg.model.load，随后原encode_condition/history fit/coefficient_velocity/reconstruct；与sample_from_cache直接输出比较。target_ext在模型条件处理后仅传给Oracle。Checkpoint须已完整落盘，必须固定实际训练契约hash，并与实际模型配置/统计/数据preflight匹配；不允许随机初始化补齐参数。该模式代码已提供，因缺实际权重尚未完成真实推理验收。

## 命令

```bash
.venv/bin/pip install -e '.[analysis]'
# 将示例复制为*.local.json，填写实际路径与核验提交
.venv/bin/python scripts/export_flash_attribution.py --config configs/flash_step250.local.json --output outputs/new_export
.venv/bin/python scripts/evaluate_flash_attribution.py --export outputs/new_export --output outputs/new_results
```

输出目录必须是新目录且位于源工程、Checkpoint和输入数据之外；脚本拒绝覆盖已有评价。cached与native配置模板在configs/flash_attribution_*.example.json，模板未包含真实路径/已核验checkpoint。native的expected_step可在Checkpoint落盘后设为实际2000/4000/8000，不能因为训练预算写8000就指定未保存的8000。

可在有原环境和真实数据的服务器上直接通过原解释器运行flash_source_bridge.py，但需确保工作目录/包环境指向核验source；在本仓库运行export包装器会自动配置显式子进程环境。只读数据验收脚本audit_flash_native_source.py可用stdin送到原服务器解释器运行，不需把脚本安装或写入训练仓库。

## 输出与指标

export目录含experiment_config.json、metadata.json、aligned.npz、seal.json。记录数据/统计/固定plan/Checkpoint参数/源代码身份；历史mode记录是否重新加载权重、C_pred恢复方式和逐窗来源。NPZ为真实已审计输出，不提交Git。

results目录含report.json、group_metrics.csv、temporal_metrics.csv、per_joint.csv、per_window.csv、coefficient_by_degree.csv、joint_increment_profile.csv、worst_group_cases.csv以及Figure1–4。每个组报告MAE/RMSE/绝对误差P95/P99/Max；时间指标基于相同50步/mask。重叠窗口池化统计不是独立试验，不对其声称统计显著性。

定义：r_repr=Oracle−GT，r_pred=Prediction−Oracle，r_total=Prediction−GT；三者均为同一物理关节坐标/rad。报告MSE交叉项，不能相加RMSE/MAE。系数错误分别按degree和scaled-training/raw-normalized/raw-physical三坐标报告，不混成单个可解释的系数精度。

关节增量为相邻预测时间点的30×差分（rad/s），与GT比较。不是末端相对位姿或物理协调分数。高误差案例追溯sample/Episode/current frame/joint/tick，无可靠阶段标签时不加阶段解释。

科学图表入口拒绝synthetic导出；单元测试只用明确标记的合成数据测试计算，未据其生成论文图。当前图表渲染只接受全有效窗口；指标函数支持逐元素显式boolean mask。若遇非全有效，先用--no-figures核验指标，不把无效值画入图中。

## 同一窗口重复评估

保持plan、数据/统计和导出顺序一致，分别导出不同Checkpoint。必须验证各窗口GT、Oracle、历史、来源一致后才比较，不以两份不同抽样的总均值推断训练效果。本轮Step100/Step250主结果共用uniform505 plan；813窗另列为Step100补充结果，不与505窗直接作训练步数对照。历史冻结VLM与当前FullFT即使步数相同，也不是受控消融。

## P0-C 正式 FullFT 配对评估

2026-11-05 已在原环境严格加载真实FullFT4000/6000，各完成固定505窗，先前“缺权重尚未真实推理”的状态仅对应P0-B旧记录。使用当前 `scripts/evaluate_fullft_paired_source.py` 与 `configs/p0c_fullft_paired.example.json`；复用原 `NativeDataset` 预处理和原采样API。离线入口 `scripts/diagnose_fullft.py`、`scripts/plot_fullft_diagnosis.py`、`scripts/compare_fullft_checkpoints.py`。完整命令、结果、source范围和限制见 `p0c_fullft_diagnosis.md`。历史cached模式及其系数恢复标记保持不变。
