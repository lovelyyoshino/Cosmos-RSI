# AGENTS.md：给 Codex 的执行规则

目标、阶段和验收标准见 [PLAN.md](PLAN.md)；文件格式以 [docs/contracts.md](docs/contracts.md) 为准；判断是否“进化”以 [docs/eval_protocol.md](docs/eval_protocol.md) 为准。本文件只写**怎么干活**。

## 工作方式

- 按 PLAN.md §3 的里程碑顺序推进。当前阶段没有通过验收，就不开始下一阶段。
- 汇报和文档都用中文。每个数字都要写明测量条件：机器、功耗模式、精度、batch、步数、种子、样本量。失败和跳过的步骤照实写。
- 每个阶段先写一个最小可运行的脚本，跑通一个样本后再扩大规模。
- 训练、生成、批量评测这类长任务，用 `nohup` 或 tmux 放到后台跑，日志写到 `runs/.../logs/`。汇报时附上进度和日志路径。
- 上游行为和文档不一致时，记到 `docs/notes/` 里，写清现象、复现步骤和上游 SHA。

## 上游代码

- 全部克隆到 Thor 的 `~/rsi/third_party/` 下，SHA 写进 `third_party/PINS.md`：
  - starVLA：用稳定分支 `starVLA`，不要用 `starVLA_dev`。
  - nvidia-cosmos/cosmos-transfer2.5：生成用。
  - NVIDIA/Isaac-GR00T：用作延迟参照和备胎。
- 不要直接修改上游源码。必须改的话，把 patch 放进 `patches/<repo>/`，并在 PINS.md 里注明。
- 先读上游已有的代理 skill 和文档：
  - starVLA 接入数据集：`docs/agent_skills/integrate-starvla-dataset/`。
  - Transfer2.5 推理：`docs/inference.md`（distilled edge 用 `--model=edge/distilled`，`num_steps=4`）。setup 中写明 Jetson AGX 必须用 CUDA 13.0（`--extra cu130`）。
- 门控模型（`nvidia/Cosmos-Reason2-2B`、Cosmos-Guardrail1 许可）需要 HF token。向用户索取，不要写进仓库。

## 机器规则（必须遵守）

### Thor：主机器，全部阶段都在这里跑
- 地址 `thor@<thor-host>`。ssh 密码向用户索取，sudo 免密。
- 本项目的文件全部放在 `~/rsi/` 下。不要动 `~/originman-cosmos-validation` 的数据和服务。
- **阶段串行**：切换阶段之前，先确认上一阶段的服务（policy server、判官 vLLM、生成、训练）已经退出，再启动下一阶段。运行时用 `tegrastats` 记录内存峰值，写进对应清单的 `peak_mem_gb` 字段。
- Python 环境：
  - 优先用宿主机上已验证的 `venv-cosmos3`。已知 NGC vLLM 26.01 容器在 Thor 上跑 conv3d 会失败，而 Qwen3-VL 的 patch embedding 用的就是 conv3d。
  - 需要新环境时，复制 venv-cosmos3 再改，不要破坏原环境。
  - 用容器时，先把镜像名和 run 命令给用户看。
- **不要擅自修改 `jetson_clocks`、nvpmodel 和系统服务。** 训练和生成如果需要 MAXN 模式，先说明理由，等用户批准。
- 跑 Reason2 时开 CUDA Graph，eager 模式非常慢。starVLA 的注意力用 sdpa，单卡训练不用 DeepSpeed。

### lenovo：只做备份
- 地址 `<user>@<host>`，目录 `<backup-root>/`，ssh 密码向用户索取。
- 用 `scripts/backup_to_lenovo.sh` 同步。密码通过环境变量 `SSHPASS` 传入，不要写进任何文件。
- **不要占用它的 GPU**，上面经常有别人的训练在跑。除非用户明确要求把某个阶段临时挪过去。
- 备份内容：代码、文档、配置、每圈的元数据、已晋升的权重。原始视频和 LeRobot 数据默认不备份，量太大。

### 本机笔记本
- 只用来编辑和查看，不下载大模型，也不存数据集。
- **禁止 `docker run --privileged`，禁止在容器里跑 systemd 或 `/sbin/init`**。这台机器曾因此两次黑屏崩溃。

### 真机（P2 起）
- **没有用户在场并明确同意，不发送任何运动指令。** 顺序是：先离线回放，再空跑，最后上真机。
- 急停方式和复位协议写进 `docs/real_robot.md` 之后才能开始。

## Git

- 本目录是 git 仓库。每个里程碑开一个分支，例如 `p0-thor-bringup`。**没有用户同意，不提交到 main，也不 push。**
- `data/`、`runs/`、`checkpoints/`、`third_party/` 不入库（见 `.gitignore`）。只提交代码、配置、文档和 PINS.md。
