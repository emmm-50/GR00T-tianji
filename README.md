# GR00T-tianji

在天玑 Jetson AGX Orin 上运行 NVIDIA GR00T N1.7 3B，用右眼、右腕图像及机器人反馈预测右臂末端目标。现场入口只订阅观测。关节候选由外部天玑 IK 生成，送入安装版 VLAHOST 的进程内模拟发布器。**目前验证到软件接收。实物跟踪、夹爪开度和零样本抓取均未验收。**

## 已验证的结果

| 环节 | 天玑上的结果 |
|---|---|
| 官方 DROID 样本 | 固定版本的策略与动作处理器完成 BF16 GPU 推理。单独测量的 5 次热机推理中位为 518 ms。三集共 844 行的状态和动作分别对照原始欧拉角，物理朝向最大误差均低于 7.1 × 10⁻⁸ rad。 |
| 录像回放 | 三个观测块共 68 条候选经进程内 Host 接收。录像按虚拟时间编排，实际回调间隔中位约 837 ms。 |
| 现场只读快照 | 三个块共 120 个候选，3 条经模拟 Host 接收。GPU 推理 0.777–0.860 s，入选目标的 IK 计算 0.730–0.740 s，快照至实际回调 1.935–2.068 s。当前无法持续以 15 Hz 接收。 |
| 名义 IK | 共享 95 mm URDF 对应的 IK 可由原始双臂 14 轴反馈初始化。入选目标在单个区间结束时仍有 9.09–13.47 mm、9.00–15.38° 的名义残差；离线保持同一目标可进一步收敛。物理 TCP 精度尚待外测。 |

上述延迟来自不同输入和计时区间。GPU 计时含 CUDA 同步，端到端计时还含快照处理、IK、等待和模拟接收。[影子实验汇总](shadow_run/RESULTS.zh-CN.md)

## 从官方样本开始

验证平台为 Jetson AGX Orin 64 GB、L4T R36.4.4、CUDA 12.6、Python 3.10.12。先按环境说明取得固定版本的上游源码、Orin 依赖、模型和 DROID 样本。以下命令均在仓库根目录执行。官方样本路径可独立复跑，无需天玑 ROS 或 IK。[环境与下载步骤](ENVIRONMENT.md)

```bash
source activate.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1 OMP_NUM_THREADS=2

# 处理器往返，读取模型配置与统计量，使用 CPU
python infer_offline.py --dataset data/droid_sample --episode 1 --start 16 \
  --check-only --output results/droid_roundtrip.json

# GPU 推理，输出动作块、形状和延迟
python infer_offline.py --dataset data/droid_sample --episode 1 --start 16 \
  --count 5 --stride 8 --warmup 2 --output results/droid_gpu.json
```

对天玑录像或现场快照，先按观测适配说明采样。完整影子运行还需要外部 `eef-control` 中的 `native_ik`、`host_mock`、共享 URDF 和 APEX 安装版 VLAHOST。可用 `TIANJI_EEF_CONTROL` 指定外部工作区，用 `--bag` 或 `TIANJI_BAG` 指定自己的录像。外部 IK、厂商 SDK、URDF、模型权重和现场采样需分别准备。实测使用了外部工作区中的配套改动，仅克隆本仓库不足以复现 Host 全链路。[观测采集与格式](tianji_adapter/README.zh-CN.md) · [IK 构建与影子复跑](shadow_run/README.zh-CN.md)

## 动作和时间约定

输入使用当前和前一秒的两路图像，以及当前状态。策略接口已完成反归一化，并以同一观测为锚点恢复全部 40 个绝对目标。`eef_9d` 的前三项是米，后六项是旋转矩阵的前两行。DROID 先按 SciPy `xyz` 提取欧拉角，编码器把同一组三个数按 `XYZ` 重建并右乘固定矩阵 C。解码需逆转这两步，再按任务坐标与工具偏置重定向。额外累加会重复施加相对位移。[编解码与 FK 对照](tianji_adapter/geometry_sources.zh-CN.md)

天玑右七轴和夹爪电机反馈仅用作 Franka 状态、闭合比例的输入代理，两路相机视角及工具对应也有差异。下游消费策略末端目标，送给模拟 Host 的七轴来自天玑 IK。DROID 自带的七关节动作不用于天玑执行，夹爪预测单独记录。

40 个样本在 15 Hz 下首末相隔 2.60 s，最后一个保持区间在 2.667 s 结束。IK 的 2 ms 是内部积分步长。VLAHOST 收到候选后立即执行进程内模拟发布，录像的虚拟时间、实际计算耗时和实际回调时间分别记录。

## 源码与依赖来源

自研部分包括根目录的离线部署入口、`tianji_adapter/` 的观测和几何适配、`shadow_run/` 的影子运行程序。上游 GR00T 代码按其 Apache-2.0 许可提供，GR00T 与 Cosmos 权重的使用条款以各模型卡为准。本仓库未为自研代码指定许可证。[上游代码许可](https://github.com/NVIDIA/Isaac-GR00T/blob/376ba890cff8c9de64d71d982772a9c36185fdd7/LICENSE) · [GR00T 模型卡](https://huggingface.co/nvidia/GR00T-N1.7-3B) · [Cosmos 模型卡](https://huggingface.co/nvidia/Cosmos-Reason2-2B)
