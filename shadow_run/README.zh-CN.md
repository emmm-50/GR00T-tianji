# GR00T 天玑右臂影子运行

在天玑用已有模型从右眼、右腕和机械臂反馈预测末端目标。系统 Python 取同一快照的双臂关节计算名义 FK，项目 venv 的常驻 GPU 策略产生一块 40 个目标，外置 IK 生成右 J1…J7 的绝对关节角（rad）。安装版 VLAHOST 的 WebSocket 路由在进程内处理动作，唯一接收者是假的右臂发布器。记录中可以逐条核对它收到的七个数、回调时间和 ACK。[录像和现场实测](RESULTS.zh-CN.md)

## 在天玑复跑

外部 IK 和共享 URDF 默认位于相邻的 `pi05/eef-control`，可用 `TIANJI_EEF_CONTROL` 指向该目录。接收测试还需要外部项目的录包 FK fixture、已安装 VLAHOST、系统 Python 依赖及已构建的 native IK；fixture 不随本库发布；该测试默认使用原站点的历史档案，可用 `TIANJI_BAG` 指定录像根目录，并用 `TIANJI_FIXTURE_PROVENANCE` 指定匹配的时间戳 sidecar。其余四项模拟时钟测试不读取录包。默认二进制为 `/tmp/gr00t-eef-module-build/native_ik`，可设置 `GR00T_IK_BINARY`，运行脚本也可用 `--binary` 和 `--config` 覆盖。

```bash
cd /path/to/gr00t-tianji
cmake -S "${TIANJI_EEF_CONTROL:-../pi05/eef-control}/native_ik" -B /tmp/gr00t-eef-module-build -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/gr00t-eef-module-build -j2
```

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
ros2 launch gmsl_quadcam quad_csi_quickview.launch.py signal_host:=127.0.0.1
```

## 坐标、状态和时间

- 本次把天玑名义 world 和 right_tool 数值作为虚拟 DROID 基座与工具坐标。用同次 14 轴反馈做共享 URDF FK，按 DROID 的物理姿态约定编码为输入。模型返回绝对目标后，对输入与输出都走已有的完整姿态解码，再把相对运动锚定到这次 FK。记录同一输出在工具沿 z 偏移 50 mm、工具轴绕 z 转 90° 时的变化。[几何约定](../tianji_adapter/README.zh-CN.md)
- 右七轴按名称放入 Franka 状态槽；电机位置除以 1.6 后截断作闭合比例；右眼和右腕替代 DROID 两路相机。这三项都是本次实验的输入假设。下游只消费策略的 EEF。夹爪预测单独记录，Host 接收的七轴全部来自天玑 IK。
- 样本 0–39 的偏移为 0–2.60 s。第 i 个目标保持到观测时刻加 (i+1)/15 s，最后一个区间终点是 2.667 s。规划结果记最后完成的 2 ms 积分时刻。录像按 bag 虚拟时间发布，到下一次观测时截去旧块的尾部。逐行另存真实进程内发布回调的单调时间，可据此核对程序耗时。
- 现场在快照组装时取单调时间，分别记录相机与状态回调、消息 header 和采集开始时的 UTC。有些反馈可能晚于图像回调。新块从新观测的 14 轴初始化。先剔除到期项，按 `--ik-lead-ms` 为剩余样本预留求解时间。默认 1000 ms 来自本机完整区间的实测耗时。跳过的区间保持上个种子，只积分入选目标自己的区间，算完等其原终点。计算过期或错过整拍则丢弃，跨块保留 60 ms 的模拟发布最小间隔。可用 `--ik-lead-ms 0` 复查无提前量时能发出多少条。
- 录像真值用 300/302 帧共享 URDF FK，叠加已知工具旋转与偏置，再经相同重定向、IK 和 Host。测试检查 14 轴种子顺序、实际右七轴消息、位姿残差与时序。GPU 策略影子实验使用同次观测及其历史图像，结果另列。[实测记录](RESULTS.zh-CN.md)

要验证实物跟踪，仍需明确抓取点与工具轴、测量夹爪实际开度，再经用户另行授权对少量已知小位姿测量目标与实到偏差。单任务后训练可先用天玑自己的关节、FK 和夹爪命令准备状态与动作标签，保留原始标签来源和按整集留出的评估集。
