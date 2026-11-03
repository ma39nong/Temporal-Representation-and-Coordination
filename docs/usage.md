# P0 使用指南

## 配置与运行

基础安装`pip install -e .`；实际读取数据使用`pip install -e '.[parquet]'`。统一入口为`python -m trc`，各命令要求`--config`和`--output`。JSON配置兼容YAML 1.2；不支持任意YAML语法。未知配置键、非法horizon、非有限时间容差会被拒绝，避免拼错后默默采用默认值。

单数据源沿用dataset_root/layout/horizon/tolerance_s/norm_stats_path，可另填dataset_id和真实repo_id以绑定统计；未指定dataset_id时为dataset，缺repo_id时不能声称与真实统计的repo身份已匹配。多数据源使用datasets列表，每项必须明确id、repo_id、root、layout；不可与dataset_root同时配置。所有路径相对配置文件解析。layout只能为canonical58或converter58，必须通过元信息中的逐关节名称核对。

配置示例是结构模板，不包含真实数据或已核验的正式实验统计。创建自己的`*.local.json`后填写真实路径。不要把示例beta/floor直接当作所有实验通用配置。

## 审计与manifest

`audit`逐Episode检查数字列，读取元信息中的相机定义但不解码视频。`manifest`对info.json、episodes.jsonl、tasks.jsonl和data下全部Parquet计算SHA-256，与主训练仓库norm_provenance的矢量文件选择一致；不hash大型视频。manifest中保留配置路径供本机复查，但data_sha256基于内容/来源/窗口配置，不依赖绝对挂载路径。

不同数据源全局Episode编号仅存在内存/报告中，原始文件和local ID从不改写。完全相同的数据目录内容会被拒绝；部分重叠/同源内容仍须原始recording映射识别。manifest命令成功表示指纹构建成功，不代表所有数值帧、统计或视频验证成功，应结合audit报告使用。

## 原始recording映射与split

优先读取episodes.jsonl的source_recording_id。缺失时可通过source_mapping_path指定独立JSON，不改原数据：

```json
{
  "source_a": {
    "0": {"source_recording_id": "actual-recording-id", "segment_id": "actual-segment-id"}
  }
}
```

外层键必须是配置datasets中的id（单数据源兼容模式为dataset）；内层是实际episode_index字符串。标识必须来自真实采集/裁剪记录，不允许编造。若与原始元信息冲突或包含不存在的ID会拒绝。不同目录中的同一原始recording必须使用相同ID，所有裁剪片段才能共同分到同一集合。

`split`默认train=.8、validation=.2、test=0，允许通过三个fraction参数修改；按recording而非帧数分配，保证请求的非空集合至少一个recording，少量recording时实际比例会偏离目标。输出同时记录目标比例、实际recording数、seed、dataset指纹与所有episode分配。缺失来源时直接blocked，不降级按Episode随机分割。

## 有效窗口

Python接口`trc.data.windows.iter_windows(config, manifest=...)`逐次返回ActionWindow，包含storage_actions、storage_anchor、canonical物理动作、canonical锚点、dataset/episode/frame ID、时间戳及可用的来源映射。完整窗口限定连续段内，窗口首帧与anchor/时间戳同帧，不padding、不重采样。

validity_column可指定一个实际Parquet字段；只接受一维boolean有效帧列。任一帧false的窗口会排除，字段缺失则报错。不自动把engaged=false或图像mask解释为无效示范。未配置时仅执行已定义的数值和时间有效性检查。

## T3资产与往返核验

asset_dir必须含delta_paper.json、delta_paper.npz、norm_stats.json、norm_metadata.json。配置必须明确mode、beta和std_floor，horizon来自同一配置。可额外固定expected_bundle_sha256、expected_norm_sha256。

expected_contract_path指向**经过核对并独立固定**的预期source_contract JSON：通常从所用实验资产提取，再与真实配置和源代码核对后固定。它必须含真实的变换指纹和数据源信息，不能为了通过检查随意重写或把未知契约称作已核验。

状态含义：

| 状态 | 已完成 |
|---|---|
| integrity_only | 文件哈希、实验参数、数组形状、schema及资产内部契约相符，尚未绑定当前数据 |
| dataset_bound_transform_unpinned | 另核对实际数据的repo ID、数量、FPS和内容hash，但没有独立固定预期变换契约 |
| verified | 实际数据绑定和预期完整契约比较均通过 |

assets-check在完整verified时退出0；部分状态仍输出结果但退出2。roundtrip默认只接受verified资产。当前接口支持已核验T3 Delta arms/Absolute hands协议，按源实现执行quantile→arm bridge；inverse按相反顺序再加回原实测锚点。输出为canonical物理关节命令，不触发部署。source action_hz必须等于数据FPS，本工具不擅自模仿训练loader的重采样。

## 退出码与保护

审计passed_with_warnings退出0；failed/blocked、来源缺失split、未完全verified资产及参数错误退出2。审计退出0不意味着研究已经可开展：同时查看来源、统计状态和not_verified。formal_data_ready当前固定false，因为视频/完整训练Batch/部署核验尚未接入。

报告不能写入数据集/资产目录，也不能覆盖配置或来源映射；通过符号链接指向这些输入同样拒绝。原始数据/权重不修改。config、manifest与报告包含本机数据路径，默认保留在被Git忽略的local配置/outputs中。

## 可选源码对照

compare_upstream.py只读用户给定源仓库的Git对象，固定main、origin/main缓存、Delta及RTC+T3四个提交；分支引用改变时拒绝继续，需先重新核验。--check-assets在临时目录生成明确的合成bundle，与原load_bundle比较。原源代码使用hashlib.file_digest，该附加对照要求Python≥3.11；本仓库运行/基础测试仍支持Python≥3.10。

--dataset-config可另检查首个真实Episode的首/中/尾窗口Delta/Absolute，不代表完整LeRobot Batch或真实T3统计已经核验。该对照脚本不属于普通安装、审计或测试依赖。
