# 离线几何验证

天玑安装项目环境并取得官方 DROID 样本后，可在仓库根目录执行 `.venv/bin/python -m tianji_adapter.verify_droid`。它使用公开 parquet 的原始欧拉角分别核对状态和动作的旋转约定。

**DROID 来源。** 固定版本的 NVIDIA 样本转换器从 `observation.state.cartesian_position` 和 `action.cartesian_position` 生成对应的 `eef_9d`。验证使用这两个原始列，根据 DROID 实际 FK 的欧拉约定分别重建物理旋转，再与策略列的解码结果比较。三集 844 行中，状态与动作的最大姿态误差均小于 7.1e-8 rad。仅对策略矩阵右乘 `C.T` 时，两者误差的中位数分别达到 1.227 和 1.217 rad。多轴真值测试还覆盖了名义工具偏心转动与整块共用单个观测锚点。[DROID 实际 FK](https://github.com/droid-dataset/droid/blob/33ae6a67274f36d2e29525b86f23a56616ef43a7/droid/franka/robot.py#L158-L167) · [NVIDIA 样本转换](https://github.com/NVIDIA/Isaac-GR00T/blob/376ba890cff8c9de64d71d982772a9c36185fdd7/scripts/download_droid_sample.py#L185-L202)

**天玑模型。** `nominal_world_right_tool(joint_names, joint_positions, urdf_path)` 用实测关节做 FK，按 `Rz(+π/2)` 把共享 URDF 世界坐标转到 IK 的 APEX world。共享资源版本为 `47c51f0`，模型位置可通过 `TIANJI_EEF_CONTROL` 配置。名义 `right_tool` 距 `TCP_Link_R` 145 mm。

**录像证据。** 34,599 条 EEF 消息分别配对 bag 时刻之前最近的关节消息，时间差中位 4.917 ms、最大 18.663 ms。使用 APEX world 和 `right_tool` 时，位置差中位 0.296 mm、最大 1.862 mm，姿态差中位 0.795 mrad、最大 3.132 mrad。换用 URDF 原世界系后，中位位置差为 0.891 m。只换成 `TCP_Link_R` 时为 145 mm。由此可识别录像反馈的数值约定，实体 TCP 精度、历史发布节点使用的具体模型版本仍未测定。[ROS 发布者及时间来源](observation_sources.zh-CN.md)
