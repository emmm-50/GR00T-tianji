# 天玑右臂观测与 GR00T 末端动作

本模块读取相机与反馈，生成模型输入，解析已解码的策略动作。天玑当前名义 `world/right_tool` 由实测关节和共享 URDF 可得。DROID 关节状态、两种机器人的工具对应及夹爪开度仍缺少跨具身依据，影子入口将策略预测的末端目标锚定到同次天玑 FK，供外部 IK 使用。

## 复跑

在项目根目录执行。以下命令用系统 Python 采集双相机图像和反馈，并把快照保存在临时目录。

```bash
cd /path/to/gr00t-tianji
export TIANJI_BAG=/path/to/recording
out=$(mktemp -d)
source /etc/apex/apex_ros_env.sh
/usr/bin/python3 -m tianji_adapter.capture_ros --source recording --frame 330 --output "$out/bag.npz"
# 现场已有相机和反馈时，可采集另一份
/usr/bin/python3 -m tianji_adapter.capture_ros --source ros --timeout 15 --output "$out/now.npz"

```

`load_snapshot(path)` 返回原生数组和同名 JSON 元数据。两路 RGB 均按历史帧、当前帧排序。右眼取代 DROID 外部相机，右腕取代 DROID 腕相机，两套视角和光学参数仍有差异。夹爪保存五个原始反馈字段。可用 `--bag` 和 `--video` 读取同布局的其他录像。[采集接口、话题、时间域和相机启动条件](observation_sources.zh-CN.md)

## 模型状态和动作时间

当前策略使用同一快照的三个状态向量。`eef_9d[9]` 按 DROID 约定编码名义 FK，`joint_position[7]` 按索引填入天玑右轴反馈（rad），`gripper_position[1]` 取首个电机反馈除以 1.6 后截断。后两项是 Franka 输入的近似。DROID 的夹爪反馈为 `1 - width/max_width`。[DROID 实际机器人状态定义](https://github.com/droid-dataset/droid/blob/33ae6a67274f36d2e29525b86f23a56616ef43a7/droid/franka/robot.py#L150-L172)

本 checkpoint 的视频索引为 `[-15, 0]`，状态索引为 `[0]`，动作索引为 `[0..39]`。样本 15 Hz，名义相邻动作间隔 66.67 ms，首行至末行相隔 2.60 s。DROID 采集循环先读状态和图像、计算动作、等待控制周期，再下发或记录跳过动作，随后将观测和该步动作写成同一条记录。第 0 行是同索引的目标，曝光至下发或到位的实际延迟仍需独立测量。天玑单调时钟与录包时间也需分别保留。[模型固定版本](../README.md) · [采集顺序](https://github.com/droid-dataset/droid/blob/33ae6a67274f36d2e29525b86f23a56616ef43a7/droid/trajectory_utils/misc.py#L69-L115) · [控制周期](https://github.com/droid-dataset/droid/blob/33ae6a67274f36d2e29525b86f23a56616ef43a7/droid/robot_env.py#L31)

## 末端到 IK 的交接

`get_action()` 已用同一个当前状态把整块恢复成绝对目标。前三维为米，后六维是旋转矩阵的前两行。DROID 物理四元数先按 SciPy `xyz` 提取三个欧拉角，NVIDIA 用相同数值按 `XYZ` 重建矩阵，再右乘 C。解码按逆顺序恢复物理朝向。接近欧拉奇异点时映射病态，SciPy 报告万向节锁时转换报错。工具偏置随后按完整刚体变换处理，转动工具也会改变有偏置的工具点位置。[DROID 角定义](https://github.com/droid-dataset/droid/blob/33ae6a67274f36d2e29525b86f23a56616ef43a7/droid/misc/transformations.py#L6-L13) · [NVIDIA 编码器](https://github.com/NVIDIA/Isaac-GR00T/blob/376ba890cff8c9de64d71d982772a9c36185fdd7/gr00t/data/state_action/droid_frame.py#L46-L65)

`T_A_B` 表示 B 在 A 系的位姿。绝对交接 `geometry.ik_targets` 接收显式任务坐标约定 `T_D_W` 和工具约定 `T_R_E`，其中 W 为天玑 IK world，R 为 right_tool，D、E 为 DROID 基座和物理末端。解码每步按 `T_W_R[k] = inv(T_D_W) @ T_D_E[k] @ inv(T_R_E)`。在仅定义相对动作迁移的任务中，`anchored_ik_targets` 用同次推理的观测和天玑 FK 配对消去基座对应，仍须提供有依据的工具点及轴向。绝对交接所需的两项对应关系均由调用方显式给入。

```python
from tianji_adapter.geometry import anchored_ik_targets

# 已取得同一观测的名义 FK、明确工具对应，且模型输入具有 DROID 语义时
steps = anchored_ik_targets(
    observed_eef_9d, world_right_tool_position_m, world_right_tool_quaternion_xyzw,
    actions["eef_9d"][0], actions["gripper_position"][0],
    observation_timestamp_ns=meta["observation_timestamp_ns"],
    observation_clock=meta["observation_clock"],
    right_tool_from_model_tool=T_R_E,
)
```

每步含 `frame_id="world"`、`tool_frame_id="right_tool"`、`position_m`、`quaternion_xyzw`、观测时间戳与时钟、索引和 `nominal_offset_s`。独立字段 `gripper_position_droid` 保留策略的闭合比例。下游 IK 可取位姿。关节初值另取同次实测双臂关节。模型的七关节动作和夹爪电机目标均未接入控制。

## 天玑验证结果（2026-11-20）

| 验证 | 结果 |
|---|---|
| 原包与现场 ROS | 录像图像间隔 1.000 s，现场 1.013 s。现场采样时相机、EEF、关节、夹爪的本机接收年龄分别为 8.23、0.066、0.881、0.677 ms。夹爪节点可能重发缓存值。 |
| 天玑几何 | 橙子包 34,599 条 EEF 与之前最近的关节反馈比较。`base_link` 消息数值相对 APEX world 的名义 `right_tool`，位置差中位 0.296 mm、最大 1.862 mm，方向差中位 0.795 mrad。若换作 `TCP_Link_R`，位置差中位约 145 mm。实测关节可提供当前名义工具点。实体抓取 TCP 仍待测量。[几何约定与历史对照](geometry_sources.zh-CN.md) |
| DROID 编解码 | 三集共 844 个状态和 844 个动作，对照原始欧拉列重建的物理朝向，最大误差分别为 7.09e-8 和 6.23e-8 rad。 |
| 策略格式 | 原包与现场只读快照都在 GPU 返回 40 步。模型坐标中，录像首步和末步距观测位置 13.2、108.9 mm，现场为 9.0、125.9 mm。两份快照的关节 1/2/3/6 及 rot6d 第 4 分量超出 DROID 状态 q01–q99，录像 z 也超出。模型接口可跑，策略迁移效果未测。 |

上表列出当次实验结果。用新快照运行完整策略和模拟 Host 的步骤见[影子运行说明](../shadow_run/README.zh-CN.md)。要把策略输出交给真实 IK，仍需定义可检验的跨机器人任务与工具对应，并解决 Franka 所需七轴状态和真实夹爪闭合比例。改变训练分位数或设定任意基座平移无法补齐这些信息。
