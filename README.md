# Cosmos-RSI：在 Thor 上运行具身策略改进闭环

在 NVIDIA Jetson AGX Thor 上串行执行策略采集、困难提取、Cosmos 外观增强、策略微调和冻结评测。各阶段通过文件与 SHA256 交接，可恢复中断的执行，并保留人工晋升审批。

```text
策略运行 → 环境真值与困难提取 → 成功示范外观增强 → 候选/对照训练 → 评测与部署门禁
    ↑                                                           │
    └────────────────── 门禁通过并获人工审批后 ──────────────────┘
```

- **策略**：starVLA CosmosGR00T，以 Cosmos-Reason2 为视觉语言骨干。
- **仿真**：LIBERO / MuJoCo；任务成败以环境真值为准，Reason2 判官提供诊断证据。
- **增强**：Transfer2.5 生成桌面纹理，经 UV 图集、仿真重渲染和质量门后交接给训练。
- **对照**：候选、D0 复制对照与解析外观消融从同一父策略出发，区分数据变化和额外训练的影响。
- **离线分支**：Predict2.5 接收候选动作，预测后续画面；当前用于诊断。

当前提供可执行框架；已有实验的统计门禁结论为拒绝，尚无生产晋升或连续多圈改善的证明。

## 安装

模型运行限定在 Thor 的 `/home/thor/rsi`。本地电脑可编辑文档、整理发布内容和执行 CPU 检查。项目依赖 Thor 上已验证的 `venv-cosmos3`，暂不提供从空白操作系统开始的一键安装。

```bash
# 在 Thor 上准备工作副本；目标目录应尚不存在
git clone https://github.com/lovelyyoshino/Cosmos-RSI.git /home/thor/rsi
cd /home/thor/rsi
source scripts/thor_env.sh
export RSI_BASE_ENV=/home/thor/cosmos_validation/venv-cosmos3
mkdir -p runs/setup/logs
cp third_party/PINS.md runs/setup/PINS.reference.md
bash scripts/p0_bootstrap.sh
```

接着完成上游版本校验、策略依赖、仿真环境、Transfer CUDA 13.0 环境及模型下载。具体顺序与检查命令见 **[安装说明](docs/installation.md)**。`p0_bootstrap.sh` 只复制基础环境并克隆仓库，不会自动装齐依赖、恢复数据或下载模型。

## 使用

完整前置条件是：可重载的父检查点、通过准入的生成数据交接文件，以及同一冻结集上的父策略完整评测。

```bash
cd /home/thor/rsi
source scripts/thor_env.sh

# 替换为本机实际产物，文件必须存在且摘要校验通过
export PARENT=/home/thor/rsi/runs/p0/policy_train_实际目录/checkpoints/step_005000
export HANDOFF=/home/thor/rsi/runs/p0/lerobot_handoff_实际目录/handoff.json
export EVALUATION=/home/thor/rsi/runs/p0/policy_evaluation_实际目录/evaluation.json
export RSI_LAUNCH_DIR="runs/launch/$(date +%Y%m%dT%H%M%S)"
mkdir -p "$RSI_LAUNCH_DIR/logs"

nohup envs/policy/bin/python -u rsi/loop.py \
  --parent "$PARENT" --handoff "$HANDOFF" \
  --incumbent-evaluation "$EVALUATION" \
  --budget configs/iteration_budget.yaml --iteration 1 --seed 42 \
  > "$RSI_LAUNCH_DIR/logs/iteration.log" 2>&1 &
echo "$!" > "$RSI_LAUNCH_DIR/iteration.pid"
```

这是预算配置中的执行示例，不代表性能测量。整圈使用既有合格纹理池，不会自动完成新的 Transfer 生成或晋升。首次运行、数据准备、日志查看、恢复、晋升与备份见 **[使用说明](docs/runbook.md)**。

## 文档与目录

| 入口 | 内容 |
| --- | --- |
| [安装说明](docs/installation.md) | 基础环境、上游 SHA、补丁、依赖、模型与安装排错 |
| [使用说明](docs/runbook.md) | 数据准备、初训、整圈执行、恢复、门禁、备份与发布同步 |
| [工作流](docs/workflow.md) | 五段接口、模型调用和产物交接 |
| [数据契约](docs/contracts.md) / [评测协议](docs/eval_protocol.md) | 文件字段与验收规则 |
| [阶段目标](PLAN.md) / [执行规则](AGENTS.md) | 阶段顺序和机器约束 |
| [上游资料](docs/references.md) / [版本清单](third_party/PINS.md) | 依赖来源、固定版本与许可边界 |

| 目录 | 内容 |
| --- | --- |
| `rsi/`、`scripts/` | 整圈编排、阶段入口与运维工具 |
| `configs/`、`eval/` | 预算、质量门、依赖约束和冻结评测定义 |
| `patches/` | 与固定上游版本配套的适配补丁 |
| `tests/` | CPU 单元与契约测试 |
| `release/` | 按依赖挑选文本源文件的同步工具 |

上传范围为代码、配置、必要文档、测试、补丁及版本清单。图片、PPT、实验过程记录、模型、数据集、环境和运行目录不随仓库分发。完整实验工作区与备份单独保留。

## 检查与许可

同步工具的检查只需 Python 标准库：

```bash
python3 -m unittest discover -s tests -p test_release_sync.py -v
```

项目完整测试使用 Python ≥ 3.11，依赖及命令见[安装说明](docs/installation.md#cpu-测试环境)。本项目代码采用 [MIT 许可](LICENSE)；补丁、模型和数据集同时受各自上游许可约束。
