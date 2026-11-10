# 安装说明

本说明针对 Thor 的宿主机环境。所有模型计算在 Thor 上执行，目录固定为 `/home/thor/rsi`，shell 使用 bash。安装前阅读 [AGENTS.md](../AGENTS.md)；脚本不会替你配置操作系统、JetPack 或基础 CUDA 栈。

## 安装前检查

需要已验证的 `venv-cosmos3`、CUDA 13.0、Git、Git LFS、ffmpeg 和 tegrastats。策略环境复制自基础环境；不要在原环境执行 pip 安装，也不要直接用普通 x86 CUDA wheel 替换 Thor 的 ARM 构建。

```bash
uname -m                         # 预期 aarch64
hostname                         # 部分入口要求主机名 thor
command -v git ffmpeg tegrastats
git lfs version
/usr/local/cuda-13.0/bin/nvcc --version
nvidia-smi
df -h /home/thor
```

这些命令只用于检查。环境缺失时先完成宿主机准备；本仓库没有覆盖所有 JetPack 版本的全新系统安装流程。模型和数据均落在工作目录，空间需求随下载组件及示范规模变化。下载器会检查文件总量并预留空间，训练缓存还需单独核算。

## 复制策略环境并核对上游

若工作目录尚不存在：

```bash
git clone https://github.com/lovelyyoshino/Cosmos-RSI.git /home/thor/rsi
cd /home/thor/rsi
source scripts/thor_env.sh
export RSI_BASE_ENV=/home/thor/cosmos_validation/venv-cosmos3
test -x "$RSI_BASE_ENV/bin/python"

# bootstrap 会重写 PINS.md，先保存本仓库给出的参考版本
mkdir -p runs/setup/logs
cp third_party/PINS.md runs/setup/PINS.reference.md
bash scripts/p0_bootstrap.sh
```

`RSI_BASE_ENV` 可以指向 Thor 上其他已验证的同名基础环境。脚本只允许 aarch64 的 `/home/thor/rsi`，采用复制而非硬链接创建 `envs/policy`，并克隆 starVLA、Transfer2.5 和 Isaac-GR00T。**bootstrap 按分支克隆当前版本，不会自动检出参考 SHA。**

在新克隆、尚未应用补丁的目录中，按保存的版本清单核对：

```bash
git -C third_party/starVLA fetch --depth 1 origin 3422b9f2387b6f682cf02802904a77b23ab13afd
git -C third_party/starVLA switch --detach 3422b9f2387b6f682cf02802904a77b23ab13afd
git -C third_party/cosmos-transfer2.5 fetch --depth 1 origin 2ff49d0717af02057ae79bc75c00fbff9da1b4e7
git -C third_party/cosmos-transfer2.5 switch --detach 2ff49d0717af02057ae79bc75c00fbff9da1b4e7
git -C third_party/Isaac-GR00T fetch --depth 1 origin 7b304f95b95790490b79bd263ba57015c3c12dfd
git -C third_party/Isaac-GR00T switch --detach 7b304f95b95790490b79bd263ba57015c3c12dfd
```

starVLA 必须来自稳定分支 `starVLA`；此处 detached HEAD 用于固定该稳定分支的已记录提交。已有带补丁的工作副本先检查 `git status --short`，不要覆盖本地修改或直接切到其他版本。

## 策略依赖与补丁

基础环境保留已验证的 torch、torchvision 和 transformers；辅助依赖按约束安装：

```bash
envs/policy/bin/python -m uv pip install \
  --python envs/policy/bin/python \
  -c configs/p0_policy_constraints.txt -r configs/p0_policy_requirements.txt
envs/policy/bin/python -m pip check

git -C third_party/starVLA apply --check "$PWD/patches/starVLA/0001-lazy-pytorch3d-rotation.patch"
git -C third_party/starVLA apply "$PWD/patches/starVLA/0001-lazy-pytorch3d-rotation.patch"
```

要求基础环境已提供 `uv`；缺失时只在复制出的环境内补装。约束清单保护已验证 CUDA 栈，发生冲突应核对基础环境，不要去掉约束强装。上游的 DeepSpeed 和 FlashAttention 不属于这条单卡 sdpa 路径的必需项。

重复安装时，可用 `git apply --reverse --check <补丁绝对路径>` 判断补丁是否已经应用；已应用就跳过，不要重复打补丁。

## 仿真环境与资产

```bash
bash scripts/p0_setup_sim.sh
git -C third_party/LIBERO fetch --depth 1 origin 8f1084e3132a39270c3a13ebe37270a43ece2a01
git -C third_party/LIBERO switch --detach 8f1084e3132a39270c3a13ebe37270a43ece2a01
git -C third_party/LIBERO-plus fetch --depth 1 origin 4976dc30028e805ff8094b55501d532c48fec182
git -C third_party/LIBERO-plus switch --detach 4976dc30028e805ff8094b55501d532c48fec182
envs/sim/bin/python -m pip check
```

`envs/sim` 从基础环境独立复制，然后安装 `configs/p0_sim_requirements.txt`。它使用的 numpy 与策略环境不同；不要将两套依赖合并。LIBERO 和 LIBERO-plus 包名相同，脚本通过明确的导入路径选择仓库。

LIBERO 本体资产按固定版本上游说明准备到项目目录。LIBERO-plus 的额外资产可以用项目下载器获取：

```bash
envs/policy/bin/python -u scripts/p0_fetch_dataset.py plus-assets
```

从该次打印的记录目录读取 `dataset_manifest.json` 中的 `assets_root`，之后传给仿真命令的 `--assets-dir`。这个入口使用社区镜像，会记录来源、版本或内容快照和逐文件 SHA256。

## Transfer 生成环境

先阅读固定版本上游 `third_party/cosmos-transfer2.5/docs/setup.md` 和 `docs/inference.md`，以及本仓库的 [补丁说明](../patches/cosmos-transfer2.5/README.md)。Jetson AGX 使用 **CUDA 13.0 / `--extra cu130`**。Transfer 的扩展与策略环境不同，安装到独立的 `envs/transfer`。

对上述固定 Transfer SHA，依次检查并应用运行适配补丁：

```bash
for patch in \
  0001-local-resource-paths.patch \
  0002-guardrail-fail-closed.patch \
  0003-experiment-overrides.patch \
  0006-transfer-only-registry.patch \
  0007-siglip2-local-path.patch \
  0008-i2v-attention-backend.patch \
  0009-local-multicontrol-checkpoints.patch; do
  RSI_PATCH_PATH="$PWD/patches/cosmos-transfer2.5/$patch"
  if git -C third_party/cosmos-transfer2.5 apply --reverse --check "$RSI_PATCH_PATH"; then
    continue
  fi
  git -C third_party/cosmos-transfer2.5 apply --check "$RSI_PATCH_PATH" || break
  git -C third_party/cosmos-transfer2.5 apply "$RSI_PATCH_PATH" || break
done
```

如检查失败，停止并核对版本与修改，不继续安装。传输补丁 `0004`、`0005` 单独处理：

```bash
# setup 脚本在检测到 0004 补丁文件时会要求锁文件已完成镜像转换
envs/policy/bin/python -u scripts/p0_transfer_mirror.py
bash scripts/p0_setup_transfer.sh
envs/transfer/bin/python -m pip check
```

镜像转换只替换 PyPI 传输地址，校验依赖版本及哈希保持一致。官方 ARM 扩展仍从官方来源下载。不要直接套用其他机器的 `0005` 本地 wheel 地址；网络受限时，先运行 `p0_prefetch_cuda_wheels.py`，再运行 `p0_setup_transfer_local.py`，由脚本核验文件并在 loopback 临时提供安装资源。安装日志保留在相应记录目录，不关闭 TLS 或安全组件来绕过失败。

`p0_bootstrap.sh` 会在后续环境安装中再次更新 PINS；全部完成后再运行一次 bootstrap，将实际 SHA 和已应用补丁写入 `third_party/PINS.md`，并与 `runs/setup/PINS.reference.md` 对照。

## 模型下载

先在模型平台确认账号许可。需要门控访问的模型通过进程环境读取 `HF_TOKEN`；用静默输入避免令牌写进命令历史：

```bash
read -rsp 'HF token: ' HF_TOKEN
printf '\n'
export HF_TOKEN
envs/policy/bin/python -u scripts/p0_download_model.py --repo nvidia/Cosmos-Reason2-2B
envs/policy/bin/python -u scripts/p0_download_model.py --repo nvidia/Cosmos-Transfer2.5-2B
unset HF_TOKEN
```

下载位置为 `checkpoints/base/<模型名>/`。每次下载将解析到的 revision、文件摘要和后端写进 `runs/p0/model_download_*/model_manifest.json`。脚本不提供任意 `--revision` 参数；复用旧实验时应恢复其对应快照并核对清单。

Transfer 还需要 VAE、文本编码器和完整安全组件。`p0_download_resources.sh` 批量下载这些组件，**后端固定为 ModelScope**；如果使用 Hugging Face，逐项调用下载器并确认许可：

```bash
for RSI_MODEL_REPO in \
  nvidia/Cosmos-Predict2.5-2B nvidia/Cosmos-Guardrail1 \
  Qwen/Qwen3Guard-Gen-0.6B google/siglip-so400m-patch14-384 \
  nvidia/Cosmos-Reason1-7B; do
  envs/policy/bin/python -u scripts/p0_download_model.py --repo "$RSI_MODEL_REPO" || break
done
```

运行该段前按需重新设置 `HF_TOKEN`。默认 Predict2.5-2B 下载筛选只取 VAE 等辅助文件，不能据此认为 Predict 动作条件分支权重已安装。base Transfer 的 SigLIP2、几何检查的 SAM2，以及 Predict 专用权重按实际任务另行下载。

## 最小验证

每项成功后再做下一项，不在备份机运行这些 GPU 命令：

```bash
envs/policy/bin/python -u scripts/p0_probe.py
envs/sim/bin/python -u scripts/p0_sim_smoke.py --repo LIBERO --task 0 --steps 32 --seed 42
envs/transfer/bin/python -u scripts/p0_transfer_probe.py
```

验证样本与策略命令见[使用说明](runbook.md)。`p0_policy_smoke.py` 必须传 `--fixture <样本JSON>`；不能直接把仿真日志目录或任意图片当作 fixture。判官命令应传 `--episode` 指向真实仿真记录目录。

每项会打印记录目录，并产生 `probe.json` 与 `logs/tegrastats.log`。验收应同时检查退出码、`status: passed`、产物摘要和内存采样。这里的步数和种子是验证配置，不是本次整理仓库时新跑的性能结果。

## Predict 可选环境

Predict 不属于默认整圈的前置环境。需要离线动作条件诊断时，按 PINS 中的 Predict SHA 单独克隆上游、阅读其安装文档并建立 `envs/predict`；应用 `patches/cosmos-predict2.5/` 的两个补丁。该环境目前没有一键安装脚本，也不应与 Transfer 环境混装。

## CPU 测试环境

CPU 检查可以在本地独立环境完成，无需下载模型或数据集：

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install pytest numpy PyYAML Pillow scipy opencv-python pyarrow
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest
```

完整测试使用 Python ≥ 3.11；`pyarrow` 用于数据索引测试。不要在 Thor 上照抄 CPU torch 安装命令覆盖 CUDA 栈。只检查发布同步时可直接使用系统 Python 的标准库命令，无需这套依赖。

## 常见问题

| 现象 | 核查与处理 |
| --- | --- |
| bootstrap 提示路径或架构不符 | 在 Thor 的 `/home/thor/rsi` 执行；本地只编辑与测试 |
| source 后路径异常 | 用 bash；`thor_env.sh` 使用 `BASH_SOURCE`，不支持直接在 zsh 中 source |
| 依赖冲突、torch 被替换 | 核对基础环境和策略 constraints；仿真与 Transfer 依赖分别装在其副本中 |
| Transfer 锁文件语义检查失败 | 核对固定 SHA、`0004` 转换及当前 uv.lock；本地 wheel 安装只用同机校验过的清单 |
| HF 401/403 或下载不到模型 | 核对账号许可、令牌作用域和网络；镜像需显式选择并记录，不能代替原站许可 |
| EGL / 场景资产缺失 | 确认已 source 环境、资产路径及 `--assets-dir`；先跑单任务仿真 |
| Qwen3-VL patch embedding 的 conv3d 失败 | 使用已验证宿主机 venv；已有 Thor 验证发现特定容器栈不兼容，不能把容器可启动当成算子通过 |
| 判官很慢 | 检查 CUDA Graph；避免将 eager 模式作为默认运行配置 |
| GPU busy 或内存不够 | 确认上一模型阶段已退出；检查进程和 tegrastats，不停止其他项目服务 |
