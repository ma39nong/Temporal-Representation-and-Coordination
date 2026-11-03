# 工程结构与接口

本仓库整理P0需要的代码，不依赖源训练工程的安装目录。旧分支用于追溯协议与数值对照，源仓库不作为sys.path、不建跨仓库软链接、不导入训练模型。

| 模块 | 职责 |
|---|---|
| config.py / io.py | 严格配置、相对路径解析、指纹与只读输入输出保护 |
| data/action_schema.py | 逐关节名称、两种布局、四组索引、116D位置提取 |
| data/lerobot_adapter.py | v2.1元信息、原始Episode数字列读取，PyArrow按需导入 |
| data/validation.py / chunks.py | 字段、dtype、数值、时间、帧号和连续区间检查 |
| data/windows.py | 带锚点、时间戳、来源和显式有效掩码的完整窗口迭代 |
| data/manifest.py | 多源内容身份、内存global Episode ID和原recording分割 |
| data/normalization.py / assets.py | 数值归一化及T3文件/数据/预期变换契约校验 |
| data/semantics.py / pipeline.py | 存储动作重排、Delta编码、训练目标、canonical物理解码 |
| project.py / cli.py | 组合审计与对外命令；JSON状态及运行版本 |

数据流：配置 → 元信息与schema → 源数据审计/manifest → 连续有效窗口 → 已绑定统计资产 → ActionCodec。编码保存并使用同帧锚点；输出恢复为canonical58物理位置。manifest与统计身份分开，避免将文件完整误判成实验配置匹配。

数据与统计均为外部只读资源；模块只依赖NumPy/标准库，读取Parquet才加载PyArrow。批处理支持以[B,H,58]编码/解码；窗口迭代按单Episode读取，未持有全部数据。审计会读取全部数字数据与矢量文件hash，适合离线检查，不能称为实时机器人流程。

没有模型层、训练循环、ROS控制层、GPU代码或完整E0–E4实现。后续P0-C多项式工具应从已核验FLASH轻量实现整理进analysis，保持数据坐标系/时间网格/系数预算显式；完整训练和机器人控制是后续独立阶段。
