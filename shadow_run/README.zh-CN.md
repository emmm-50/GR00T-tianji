# GR00T 天玑右臂影子运行

在天玑用已有模型从右眼、右腕和机械臂反馈预测末端目标。系统 Python 取同一快照的双臂关节计算名义 FK，项目 venv 的常驻 GPU 策略产生一块 40 个目标，外置 IK 生成右 J1…J7 的绝对关节角（rad）。安装版 VLAHOST 的 WebSocket 路由在进程内处理动作，唯一接收者是假的右臂发布器。记录中可以逐条核对它收到的七个数、回调时间和 ACK。[最新长时管线复测](PIPELINE.zh-CN.md) · [模型与 IK 性能复测](PERFORMANCE.zh-CN.md) · [首次录像和现场实测](RESULTS.zh-CN.md)

## 在天玑复跑

本版对应 `tianji-openpi` 的 `8c32a25`。在新工作区，从 GR00T 仓库根目录取相邻的依赖。天玑已有匹配版本时，直接使用现有目录。[固定版本源码](https://github.com/emmm-50/tianji-openpi/tree/8c32a25ae45fe2ae45dd0bf6ae0f3927c03f702b/eef-control)

```bash
git clone --single-branch --branch codex/gr00t-shadow-ik https://github.com/emmm-50/tianji-openpi.git ../pi05
git -C ../pi05 switch --detach 8c32a25ae45fe2ae45dd0bf6ae0f3927c03f702b
```

`native_ik/robot_model` 通过相对链接读取与 `pi05` 同级的 `tianji-robot-resources/resources/marvin/models/Pro-m6-696`，验证使用资源提交 `47c51f06afcf`。在有权使用这些 URDF、网格和本机 APEX 安装的环境构建。默认从相邻的 `pi05/eef-control` 加载；其他布局用 `TIANJI_EEF_CONTROL` 指定。接收器复用已加载的 Host 和事件循环，并用 `on_right_publish` 记录实际回调。[原生 IK 构建依赖](https://github.com/emmm-50/tianji-openpi/tree/8c32a25ae45fe2ae45dd0bf6ae0f3927c03f702b/eef-control/native_ik) · [Host 模拟接口](https://github.com/emmm-50/tianji-openpi/tree/8c32a25ae45fe2ae45dd0bf6ae0f3927c03f702b/eef-control/host_mock)

```bash
source /etc/apex/apex_ros_env.sh
cmake -S "${TIANJI_EEF_CONTROL:-../pi05/eef-control}/native_ik" -B /tmp/gr00t-eef-module-build -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/gr00t-eef-module-build -j2
```

录像 FK 测试使用调用者提供的原始录像和匹配的 `TIANJI_FIXTURE_PROVENANCE` JSON，包含 `source_video_frames`、`bag_timestamps_ns`。可用 `TIANJI_BAG` 指定含 `data/` 的录像根目录，省略时读取 sidecar 的 `bag` 字段。天玑默认用相邻 `gr00t-tianji/recording/provenance.json`。可用 `GR00T_IK_BINARY` 或运行参数 `--binary` 指定编译产物，`--config` 指定任务配置的绝对路径。

从天玑项目根目录运行。输出目录由程序新建，每块保存 NPZ、原始时间元数据、模型输入与绝对 EEF/夹爪输出、候选位姿、逐槽 IK 残差和 Host 实收消息。模型子进程自动使用原项目的离线环境，整个有限任务完成即退出。

```bash
cd /path/to/gr00t-tianji
source /etc/apex/apex_ros_env.sh
export LD_LIBRARY_PATH="$(/usr/bin/python3 -m site --user-site):${LD_LIBRARY_PATH:-}"
/usr/bin/python3 -m unittest shadow_run.test_receiver

# 对本地保存的影子结果复查名义 IK 保持残差，只做离线计算。
/usr/bin/python3 -m shadow_run.hold_probe /path/to/private-run.json.gz --output /tmp/gr00t-shadow-hold.json

# 指定已有录像（也可设置 TIANJI_BAG；--video 可选）。
export TIANJI_BAG=/path/to/recording
/usr/bin/python3 -m shadow_run.run --frames 300 330 360 --out logs/shadow-bag-new

# 同次加载依次测录像和三次新现场观测，结束后可读 logs/shadow-mixed-new/run.json。
/usr/bin/python3 -m shadow_run.run --frames 300 330 360 --live 3 --out logs/shadow-mixed-new
```

现场需要已有反馈和 `/quad_tile/jpeg/compressed`。只有相机缺席时，在另一个天玑终端运行下列预览，测完用 Ctrl+C 关闭本次预览。

```bash
source /etc/apex/apex_ros_env.sh
(cd /tmp && ros2 launch gmsl_quadcam quad_csi_quickview.launch.py signal_host:=127.0.0.1)
```

## 坐标、状态和时间

- 本次把天玑名义 world 和 right_tool 数值作为虚拟 DROID 基座与工具坐标。用同次 14 轴反馈做共享 URDF FK，按 DROID 的物理姿态约定编码为输入。模型返回绝对目标后，对输入与输出都走已有的完整姿态解码，再把相对运动锚定到这次 FK。记录同一输出在工具沿 z 偏移 50 mm、工具轴绕 z 转 90° 时的变化。[几何约定](../tianji_adapter/README.zh-CN.md)
- 右七轴按名称放入 Franka 状态槽；电机位置除以 1.6 后截断作闭合比例；右眼和右腕替代 DROID 两路相机。这三项都是本次实验的输入假设。下游只消费策略的 EEF。夹爪预测单独记录，Host 接收的七轴全部来自天玑 IK。
- 样本 0–39 的偏移为 0–2.60 s。第 i 个目标保持到观测时刻加 (i+1)/15 s，最后一个区间终点是 2.667 s。规划结果记最后完成的 2 ms 积分时刻。录像按 bag 虚拟时间发布，到下一次观测时截去旧块的尾部。逐行另存真实进程内发布回调的单调时间，可据此核对程序耗时。
- 现场在快照组装时取单调时间，分别记录相机与状态回调、消息 header 和采集开始时的 UTC。有些反馈可能晚于图像回调。新块从新观测的 14 轴初始化。先剔除到期项，按 `--ik-lead-ms` 为剩余样本预留求解时间。优化后的外置 IK 默认预留 35 ms；旧二进制可显式指定此前使用的 1000 ms。求解后仍核对原截止时间，偶发慢区间可能过期。跳过的区间保持上个种子，只积分入选目标自己的区间，算完等其原终点。计算过期或错过整拍则丢弃，跨块保留 60 ms 的模拟发布最小间隔。可用 `--ik-lead-ms 0` 复查无提前量时能发出多少条。
- 多块现场运行持续订阅 ROS。模型只处理一份观测，采样端只保留最新完整 JPEG 与反馈。下一块就绪时，接收线程先完成已经领取的 IK/Host 槽，再按新观测的真实时刻接手可用后缀。每块用同次真实 14 轴反馈初始化，旧块未领取的尾部记入 superseded_tail。末块继续到 40 个名义时间槽结束。每块快照当场保存，整次 run.json 在有限运行退出时写出。[提交点与长时结果](PIPELINE.zh-CN.md)
- 录像真值用 300/302 帧共享 URDF FK，叠加已知工具旋转与偏置，再经相同重定向、IK 和 Host。测试检查 14 轴种子顺序、实际右七轴消息、位姿残差与时序。GPU 策略影子实验使用同次观测及其历史图像，结果另列。[实测记录](RESULTS.zh-CN.md)

要验证实物跟踪，仍需明确抓取点与工具轴、测量夹爪实际开度，再经用户另行授权对少量已知小位姿测量目标与实到偏差。单任务后训练可先用天玑自己的关节、FK 和夹爪命令准备状态与动作标签，保留原始标签来源和按整集留出的评估集。
