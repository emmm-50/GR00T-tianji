# 原生观测的来源与时钟

`from tianji_adapter.observations import load_snapshot` 后，用 `arrays, meta = load_snapshot(path)` 读取同名 NPZ 和 JSON。两路图像的顺序都是前一秒、当前帧。

| 原生字段 | 来源和定义 |
|---|---|
| `right_eye`、`right_wrist` | 同一张四宫格的右上、右下，RGB uint8。录像每路 960×540，现场 JPEG 每路 320×180。 |
| `joint_names`、`joint_positions` | `/tj/joint_states` 的双臂命名关节，单位 rad。现场发布者 `marvin_robot_node` 也发布 `/tj/info/joint_feedback`，两路 14 轴读数相近。参考机器人节点源码的 `ReadRobotState()` 从 SDK `ReadState()` 取值。安装版与参考源码的逐行一致性尚未确认。 |
| `eef_xyz`、`eef_xyzw` | `/tj/info/eef_right` 的 PoseStamped，单位 m，父帧记录在 `eef_frame_id`。本次现场发布者为 `motion_node`。历史包的 child frame 和运行模型版本缺少发布端记录。有关数值对照见[几何证据](geometry_sources.zh-CN.md)。 |
| `gripper_motor` | `/tj/info/gripper_feedback_R` 的完整 Float32MultiArray。安装驱动的五项依次为电机位置、速度、力矩、MOS 温度、电机温度。已有协议审计按 rad 解码位置，速度、力矩及温度单位本轮未独立核实。驱动以 200 Hz 重发最近解码值，消息无 header。要换算手指开口仍需物理标定。 |

## 采集与回放

在天玑运行，先配置本机 ROS 环境。实时入口订阅现有三路反馈和四宫格 JPEG。

```bash
cd /path/to/gr00t-tianji
export TIANJI_BAG=/path/to/recording
source /etc/apex/apex_ros_env.sh
/usr/bin/python3 -m tianji_adapter.capture_ros --source recording --frame 330 --output /tmp/adapter_observation_recording.npz
/usr/bin/python3 -m tianji_adapter.capture_ros --source ros --timeout 15 --output /tmp/adapter_observation_live.npz
```

现场未启动相机时，在另一终端运行以下相机预览。采样完成后在该终端按 `Ctrl+C` 结束预览。

```bash
source /etc/apex/apex_ros_env.sh
ros2 launch gmsl_quadcam quad_csi_quickview.launch.py
```

录像入口通过 `--bag` 或 `TIANJI_BAG` 指定同类包，存在多个视频子包时用 `--video` 指定目录名。程序解码 H.264 原始四宫格，用视频时间戳选距当前帧之前 1 秒最近的帧，开头允许约一个采样周期的误差。每路状态选视频参考时刻之前最后一条反馈。帧 300/330 与既有 v3 转码的平均像素绝对差为 2.36–2.52（0–255）。

## 如何解释年龄

`observation_timestamp_ns` 与 `observation_clock` 一起标识快照时刻。`ages_ms` 按路给出差值，`image_history_seconds` 给出实际帧间隔。录像使用当前视频的 bag 时间，现场使用本机单调时钟的采样时刻与各路回调接收时刻。相机、关节和 EEF 的 header 时间另存。相机曝光、夹爪电机采样及跨设备物理同步仍缺少共同时间依据。

橙子包 1,071 个视频消息的 header 全等于各自 bag 时间，所有相邻时间差均为 33,333,333 ns。这与规则的 30 Hz 时间轴一致，时间戳生成机制待核实。现场话题为 `/quad_tile/jpeg/compressed`，历史话题为 `/recorder/quad_tile/compressed_undistorted`。当前 SH5 开启校正且四路均配置翻转。历史配置及两条通路的畸变校正、翻转、裁切是否完全相同仍待确认。
