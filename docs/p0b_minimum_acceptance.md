# P0-B 最小验收（2026-11-04）

## 结论与适用范围

已满足开展**Absolute FLASH 离线误差归因**的最小输入条件。现有数据读取、schema、归一化和窗口框架继续使用，没有重构。新增代码仅为原FLASH接口适配、指标和报告。正式FullFT预测仍受Checkpoint未保存阻塞；这不是数据框架未完成。

原工程当前在tomato-openpi的paper-flash-test，Commit e64c9a41ecb24494ea6c81e2df3dd4bbbfdcc45c，工作区干净；旧的独立FLASH worktree已不存在。原仓库、服务器源代码、数据、统计、权重均未修改，没有新训练。

## 核验协议

| 项目 | 结果 |
|---|---|
| 动作 | Absolute commanded joint position，rad；输出(50,58)，双臂14D＋双手44D |
| canonical索引 | LA0:7、LH7:29、RA29:36、RH36:58；逐名称核对原ORDER |
| 转换器存储 | action顺序LA,RA,LH,RH；state116含位置速度；原FLASH按名称提取位置并重排 |
| 条件历史 | 实测位置ticks−7…0，共8帧；只用≤当前帧的state |
| 扩展监督 | 命令动作ticks−2…51，共54帧；输出仅切片2:52，即ticks0…49；额外四帧是前后各两帧，不是另一个Chunk |
| 边界 | 需要start≥7且start≤N−52，计数N−58；不padding、不跨Episode |
| 归一化 | action/history均用该FLASH实验的global q01/q99与epsilon1e−6，无clip；不是T3 Delta＋逐时间Z-score |
| Oracle | 原FlashPolynomial.fit_target，包含原训练目标边界投影；未来标签只供Oracle |
| 推理 | 原sample逻辑：history系数＋单步coefficient_velocity，解码；不施加未来端点约束、裁剪或额外平滑 |

## 数据分别登记，不能相加成独立示范数

| 数据 | 本次覆盖 | 关系与限制 |
|---|---|---|
| 早期单颗数据 | 既有P0-A审计记录：tier_a/b1/c合计143 Episode/129206帧，另episode284 | canonical58/state58；与新normal版本来源关系未完全确认，不能当作新增独立训练/测试源 |
| 当前正式normal480 | 28服务器逐集只读读取480 Episode、451270帧；423430合法因果窗口；480个source ID | 全部来源参与当前FullFT；没有独立验证split |
| 其中新采摘311子集 | 正式480的六个A/B根，311 Episode、294206帧 | 明确为480的子集；余下历史169集157064帧，不能把311和480当独立来源 |
| 连续多颗no-block-more | 旧FLASH数值缓存3个根：3＋7＋6＝16 Episode | 与单颗normal合并缓存共有496记录；真实窗口原数据可读，不能假定每个episode具备阶段标签 |
| 历史FLASH验证 | 101个原始source，固定505均匀窗口；另有813个按既有规则选出的窗口 | 相对历史395训练source独立，非相对当前FullFT独立。101中97进入当前480训练，仅4未进入 |

已保存当前FullFT清单之外4个候选source的20个固定窗口到本机outputs，供后续独立来源评估准备。它们只被证实不在当前480 manifest中，不保证未在通用预训练或其他历史实验出现；还需使用当前Checkpoint/统计/真实当前RGB读取进行新推理，不能把历史预测冒充当前模型结果。

## 正式480最小验收证据

通过原flash_native.read_numeric逐集读取，并额外检查Episode ID。读取帧数、逐关节重排、有限性、原始frame_index、30Hz时间连续性、每集首/中/尾真实54/8帧边界、资产manifest/seal/norm/计划指纹和实际训练契约均通过。480集均有source_timestamp_ns，原始时间映射单调且间隔符合30Hz；但这不构成采集语义或阶段标注的核验。

使用实际统计转float32，与模型配置的q01/q99精度一致；每集三个合法窗口正逆变换最大误差3.427267074584961e−7 rad。未替换、重估或生成正式统计。

资产指纹：

- manifest：d27dd1d976f8766d096658f97c415a98e42576b4604e93c4845bd4e5b42368c2
- norm_stats：a5d7bd2dd422148736c69f1bf52a721d5770ba810f3bace9a2dd413395544d52
- evaluation plan：c763a1d9ca9989c5f35b1e92b37477d0ee82dbf43b7d5a960e421b536cec95ce
- actual training contract：50e26190703b466987ee5a8db9879fb04b43a5e221d3c07df5bcf8121c80e1e8

完整记录在本机outputs/flash_attribution/remote_p0b_float32.json。无RGB同步全量检查，不声称完成整个训练Batch或部署验收。当前目标为数字轨迹归因，这些未完成项不阻塞对已审计真实推理输出的重分析。

## 当前训练与Checkpoint

2026-11-04 19:23（上海）只读快照：实际FullFT run后缀162556，batch256，最新记录Step454；8卡已有训练进程占用显存，没有启动额外模型进程争抢资源。配置导出的batch512/8000是计划快照，不能作为当前实际完成状态。当前run目录下未发现params目录，未读取或重新运行当前Checkpoint。

历史Step100/Step250参数目录在本地也未发现可用副本，但保留了原验证入口产出的真实prediction/GT/Oracle/history以及Checkpoint参数hash、零更新/权重不变审计。已对这些文件hash、固定plan、train-only统计身份、source split和**每一评价窗口**的原始Parquet标签/实测历史重新核对。故可以做可信的历史误差归因，但不能声称本次重新恢复了权重。

## 不阻塞当前归因但限制解释的事项

没有经过本轮核验的URDF/FK或可靠抓取/摘取/释放阶段标签，不输出末端相对位姿、稳定抓持或协作成功率。关节级时间增量仅为可验证诊断。没有开展新的模型训练，也不根据帧号推测阶段。当前FullFT新推理等待实际落盘Checkpoint；原生推理配置接口已留出，尚未用实际FullFT权重测试。
