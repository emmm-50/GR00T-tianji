# 在 Jetson AGX Orin 上准备推理环境

实测平台使用 JetPack 6 系列（L4T R36.4.4）、CUDA 12.6、系统 Python 3.10.12 和 FFmpeg。以下步骤在刚克隆的仓库根目录执行，使用 PATH 中已有的 Git LFS 和 uv。官方 DROID 样本可独立复跑。天玑录像和现场采集使用系统 ROS 环境，影子接收另需 IK 及 VLAHOST。[影子运行依赖](shadow_run/README.zh-CN.md)

| 依赖 | 实测版本 |
|---|---|
| Isaac-GR00T | `376ba890cff8c9de64d71d982772a9c36185fdd7`，使用该提交的 Orin 锁文件与 DROID 样本 |
| GR00T-N1.7-3B | `373d34e2723d928870157d07dc216a19a624b5bb` |
| Cosmos-Reason2-2B | `9ce19a195e423419c349abfc86fd07178b230561` |
| 推理库 | PyTorch 2.10.0，torchvision 0.25.0，Transformers 4.57.6，FlashAttention 2.8.3，cuDSS 0.8.0.10 |

## 取得上游与 Orin 依赖

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --no-checkout https://github.com/NVIDIA/Isaac-GR00T.git upstream
GIT_LFS_SKIP_SMUDGE=1 git -C upstream checkout --detach 376ba890cff8c9de64d71d982772a9c36185fdd7
git -C upstream lfs pull --include='scripts/deployment/orin/wheels/*,demo_data/droid_sample/**'
mkdir -p data
ln -s ../upstream/demo_data/droid_sample data/droid_sample

python3.10 -m venv .venv
export UV_PROJECT_ENVIRONMENT="$PWD/.venv" UV_CACHE_DIR="$PWD/.cache/uv" UV_LINK_MODE=copy
uv sync --project upstream/scripts/deployment/orin --locked --no-install-project --no-dev
uv pip install --python .venv/bin/python --no-deps nvidia-cudss-cu12==0.8.0.10 \
  upstream/scripts/deployment/orin/wheels/torchcodec-0.10.0a0-cp310-cp310-linux_aarch64.whl
printf '%s\n' /usr/lib/python3.10/dist-packages > .venv/lib/python3.10/site-packages/jetpack-system-packages.pth
source activate.sh
```

`activate.sh` 把固定的上游源码加入 Python 路径，并应用 Orin 的 CUDA 库设置。该 PyTorch wheel 需要 `libcudss.so.0`。若重新执行 `uv sync`，需再安装单独补充的 cuDSS 和 torchcodec。[上游 Orin 构建材料](https://github.com/NVIDIA/Isaac-GR00T/tree/376ba890cff8c9de64d71d982772a9c36185fdd7/scripts/deployment/orin)

## 取得模型并切换为离线推理

使用具有 Cosmos 模型访问权限的 Hugging Face 账号，通过官方 CLI 登录。以下命令分别下载策略、处理器与骨干。骨干的固定快照保存在项目内缓存，上游按模型名解析的 `main` 引用也指向该快照。[GR00T 权重与条款](https://huggingface.co/nvidia/GR00T-N1.7-3B) · [Cosmos 权重与条款](https://huggingface.co/nvidia/Cosmos-Reason2-2B)

```bash
hf auth login
hf download nvidia/GR00T-N1.7-3B --revision 373d34e2723d928870157d07dc216a19a624b5bb \
  --include '*.safetensors' '*.json' --local-dir models/GR00T-N1.7-3B
python fetch_backbone.py
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1
```

冒烟验证可直接运行 README 中的 CPU 往返与 GPU 推理。此前的纯离线实验还在独立网络命名空间 (`unshare -Urn`) 中运行过。[复跑命令](README.md)
