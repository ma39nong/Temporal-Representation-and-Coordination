# 当前验证结果（2026-11-03）

## 环境与测试

本地Python3.12、NumPy2.5.4、PyArrow26.0.0，trc-research0.2.0。

| 验证 | 结果 | 范围 |
|---|---|---|
| 完整环境unittest | 42项通过 | schema/NaN/Inf/时间/边界/缺失参数；bundle/hash/schema/数据/变换身份；两布局批量codec往返；manifest/split/有效帧/CLI只读保护 |
| 仅NumPy新环境 | 29项通过、13项明确跳过 | 没有安装PyArrow，Parquet集成测试跳过；不是将跳过计为通过 |
| 源码四分支数值对照 | 40项通过 | main9、origin/main缓存9、Delta11、RTC+T3缓存11，浮点rtol=atol=1e-6；顺序/索引精确比较 |
| 原load_bundle对照 | 6项通过 | 合成bundle元信息/数组完全一致，mode/beta/horizon/floor不匹配及损坏哈希均拒绝 |
| 真实数据编码对照 | 3个窗口通过 | tier_a首个859帧Episode的start0、405、809，Delta/Absolute与canonical物理往返；没有使用不匹配统计 |
| Wheel独立安装 | 通过 | 已构建0.2.0，打包包含第三方许可证；在仅NumPy环境安装wheel后29项通过、13项跳过，导入位置为site-packages，无旧仓库依赖 |
| CI | 已提供，未在远端运行 | Python3.10/3.12 × 有/无PyArrow矩阵；未推送仓库 |

测试使用临时目录生成明确标记的合成数据/统计，不提交这些资产，也不把它们当作真实数据验证结果。没有硬件指令、训练、模型加载或GPU测试。

## 本地真实数据覆盖

| 目录简称 | Episodes | 帧数 | 完整50帧窗口 |
|---|---:|---:|---:|
| tier_a | 7 | 5962 | 5619 |
| tier_b1 | 62 | 53785 | 50747 |
| tier_c | 74 | 69459 | 65833 |
| 多源合计读取记录 | 143 | 129206 | 122199 |

新版audit/manifest已对上述三源执行。早期P0-A另读取episode284（1/1547/1498）。这些目录可能部分重叠，不能将合计当作独立示范数。所读数据为canonical58/state58、float32、30Hz，数值及局部时间审计通过；本次没有把它们冒充正式311/480数据。

审计报告为passed_with_warnings，vector_contract_ready=false、formal_data_ready=false。原始recording和原始时间映射不完整，split真实运行返回blocked，符合预期。匹配的正式T3 bundle未找到，因此未运行真实T3 assets-check/roundtrip成功案例，也未填充虚构统计。

## 可复查命令

```bash
.venv/bin/python -m unittest discover -s tests -v
# 仅NumPy验证环境（本机outputs下，Git忽略）
outputs/basic_env/bin/python -m unittest discover -s tests -v
# 本机私人数据配置，不提交
.venv/bin/python -m trc audit --config configs/early_collection.local.json --output outputs/early_collection_audit.json
.venv/bin/python -m trc manifest --config configs/early_collection.local.json --output outputs/early_collection_manifest.json
.venv/bin/python -m trc split --config configs/tier_a.local.json --output outputs/tier_a_split.json
# 可选外部源码对照，路径由调用者指定
.venv/bin/python scripts/compare_upstream.py --source-root /path/to/tomato-openpi --check-assets --output outputs/upstream_complete_comparison.json
```

outputs中的完整报告和*.local.json仅保留本机。全部基础测试不依赖旧训练目录或真实数据挂载。

## 尚未验证

正式311/480数据及连续多颗数据；真实T3统计/配置完整身份；清洗来源边界；有效监督掩码的具体业务含义；RGB视频帧同步；固定LeRobot版本的完整Batch/最近邻/padding一致性；部署发送/插值/限幅；E0–E4实验与研究假设。当前工程能力完成不代表上述事项已经完成。
