# 使用说明

先完成[安装说明](installation.md)，再按本文运行。命令在 Thor 的 `/home/thor/rsi` 中用 bash 执行，开始前 `source scripts/thor_env.sh`。每次只启动一个模型阶段，不改变既有功耗模式，不停止其他项目服务。

文中的中文占位目录必须替换为本机真实路径。运行结果以实际记录为准；命令中的 batch、步数和 seed 是输入配置，不是已测性能。

## 1. 第一次运行前准备什么

| 输入 | 内容与检查方式 |
| --- | --- |
| 基础模型 | `checkpoints/base/` 中的真实权重，下载清单可核对 revision 与文件摘要 |
| 仿真资产 | 固定版本 LIBERO / LIBERO-plus 所需场景与材质，单任务先跑通 |
| D0 缓存 | 项目 `data/` 下真实成功示范、初态、动作、归一化信息和 train/validation 划分 |
| 冻结评测快照 | 冻结定义引用的实际初态文件及其摘要；JSON 定义不能替代快照 |
| 父检查点 | `checkpoint.json` 与 `state.pt`，可保存重载，保留父训练谱系 |
| 完整父评测 | 同一冻结集和环境条件下的 `evaluation.json` |
| 合格生成窗口 | 经过独立质量门、原生读取核验的 `handoff.json` 与实际数据文件 |

仓库不附带这些权重、数据和运行产物。复现实验需要从完整 Thor 工作区或备份恢复对应文件及目录关系；只克隆仓库无法直接启动整圈。冻结 JSON 中包含原始快照路径，恢复时逐项核对，不能仅创建空目录绕过检查。

## 2. 数据与初始策略

### 先验证一个任务

```bash
# 默认只下载一个任务的官方原始示范，实际来源与 revision 记入清单
envs/policy/bin/python -u scripts/p0_fetch_raw_libero.py
```

读取该次打印的记录目录及其中 `raw_dataset_manifest.json`，然后进行小规模再生：

```bash
export RAW_MANIFEST=/home/thor/rsi/runs/p0/raw_libero_download_实际目录/raw_dataset_manifest.json
envs/sim/bin/python -u scripts/p0_regenerate_d0.py \
  --raw-manifest "$RAW_MANIFEST" --per-task 2 --workers 1
```

再生使用 MuJoCo 回放真实动作并检查环境成功，结果在 `data/` 中生成缓存，阶段报告在 `runs/p0/`。先看 `probe.json` 的状态和 `regeneration.json` 指向的缓存，确认成功真值、图像/状态/动作对齐、初态和划分。未通过的回合不能作为成功监督。

正式全量入口是 `p0_fetch_raw_libero.py --task all`，再用该次清单运行 `p0_regenerate_d0.py --raw-manifest …`。全量再生必须后台执行并保存日志。`--reuse-run` 只用于明确指定已校验的再生执行目录，不能随意复用其他数据快照。

旧 LeRobot 镜像的排查入口 `p0_map_raw_libero.py`、`p0_raw_alignment.py` 不属于每次再生的必经步骤。使用镜像缓存时先完成分片、帧数和控制语义检查。

### 冻结与基线

`eval/frozen_eval_set.json` 是已经冻结的定义，摘要见 `eval/frozen_eval_set.sha256`。已有实验应恢复对应初态快照，不重新生成不同初态替代它。

```bash
envs/policy/bin/python - <<'PY'
import hashlib
from pathlib import Path
path = Path('eval/frozen_eval_set.json')
expected = Path('eval/frozen_eval_set.sha256').read_text().strip()
assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
print('冻结定义摘要通过；仍须验证实际初态文件')
PY
```

新建独立实验时，`p0_eval_catalog.py` 生成 `eval_draft.json`，`p0_freeze_eval.py --cache … --draft … --probe` 用于初态复现探针。正式冻结会写入规范评测文件，应在独立版本中完成，更新相关预算摘要并重新建立完整基线；不能为了让当前候选通过而改已有冻结集。

缓存与冻结快照都就绪后，启动基线流水线：

```bash
export CACHE=/home/thor/rsi/data/实际缓存目录
export RSI_BASELINE_DIR="runs/launch/baseline_$(date +%Y%m%dT%H%M%S)"
mkdir -p "$RSI_BASELINE_DIR/logs"
nohup envs/policy/bin/python -u scripts/p0_run_baseline.py \
  --cache "$CACHE" --steps 5000 --first-steps 500 --batch 8 --seed 42 \
  --evaluation-workers 1 \
  > "$RSI_BASELINE_DIR/logs/baseline.log" 2>&1 &
echo "$!" > "$RSI_BASELINE_DIR/baseline.pid"
```

流水线先短训和子集预检，再续训到预先固定的最终步数，最后完整评测。多 worker 评测还要求对应最终权重的串并行等价性验证。输出 `baseline_pipeline.json`，其中给出检查点和评测来源。

策略单样本验证使用 `p0_policy_smoke.py --fixture <样本JSON>`。fixture 需要 `sample_npz`、双视角 `images` 和 `instruction`；NPZ 里的 `normalized_action` 形状为 `[8, 7]`，`normalized_state` 为 `[1, 8]`。使用经数据对齐验证的 fixture，避免从任意观测手工拼出错误监督。

## 3. 准备增强数据交接

整圈要求已有合格的 `handoff.json`，不会自动把任何新生成视频当作训练数据。生成与准入的职责见 [workflow §3.3](workflow.md#33-泛化生成)。

| 步骤 | 入口 | 必须核对 |
| --- | --- | --- |
| 选择成功源 | `p0_prepare_verified_transfer.py --cache … --episode-id …` | 来自 `demo_train`，成功、帧范围和真实标签一致 |
| 原生生成 | `p0_transfer_smoke.py --spec … --model edge/distilled --num-steps 4` | 原生 spec、权重和安全组件完整；原生视频不会直接取得训练资格 |
| 固定纹理与 UV | `p0_table_texture_design.py` → `p0_table_uv_atlas_design.py` | 生成来源、纹理摘要、设计方法和控制组绑定 |
| 仿真重渲染 | `p0_table_uv_validation.py render/controls` | 设计源先通过完整控制，再读取独立源 |
| 窗口准入 | `p0_uv_window_gate.py` | render、design、heldout 报告与逐视频复核齐全，`eligible_for_training: true` |
| 原生格式交接 | `p0_lerobot_handoff.py` | LeRobot 落盘、动作块与 starVLA 原生回读一致 |

这些设计入口部分绑定特定示范、纹理和历史报告 SHA，配置是可追溯的复现实验配方。**换新数据时必须重新建立设计及独立控制证据，不能仅替换文件名或去掉摘要检查。** `p0_run_lerobot_handoff.sh` 等历史包装脚本也可能绑定原始目录；新运行优先使用显式参数的组件入口。

准入报告齐全后：

```bash
envs/policy/bin/python -u scripts/p0_lerobot_handoff.py \
  --parent "$PARENT" --gate "$DESIGN_GATE" --gate "$HELDOUT_GATE"
```

`DESIGN_GATE`、`HELDOUT_GATE` 指向对应角色的真实 `quality_gate.json`。交接输出 `handoff.json`，不能只拷贝该 JSON 而漏掉其中引用的数据文件。

## 4. 运行一圈

```bash
export PARENT=/home/thor/rsi/runs/p0/policy_train_实际目录/checkpoints/step_005000
export HANDOFF=/home/thor/rsi/runs/p0/lerobot_handoff_实际目录/handoff.json
export EVALUATION=/home/thor/rsi/runs/p0/policy_evaluation_实际目录/evaluation.json
test -f "$PARENT/checkpoint.json" && test -f "$PARENT/state.pt"
test -f "$HANDOFF" && test -f "$EVALUATION"

export RSI_LAUNCH_DIR="runs/launch/$(date +%Y%m%dT%H%M%S)"
mkdir -p "$RSI_LAUNCH_DIR/logs"
nohup envs/policy/bin/python -u rsi/loop.py \
  --parent "$PARENT" --handoff "$HANDOFF" \
  --incumbent-evaluation "$EVALUATION" \
  --budget configs/iteration_budget.yaml --iteration 1 --seed 42 \
  > "$RSI_LAUNCH_DIR/logs/iteration.log" 2>&1 &
echo "$!" > "$RSI_LAUNCH_DIR/iteration.pid"
```

预算由 `configs/iteration_budget.yaml` 固定，候选、复制对照和解析消融按配置分别训练。合格来源不足会停止并报告，不会自动填入失败轨迹或降低门槛。保守正式配方另见 `configs/iteration_budget_conservative_formal.yaml`；配方变更应预先声明并使用对应来源。

若复用某次反馈，追加 `--feedback-chain <确切的 feedback_chain.json>`。第 2 圈起追加 `--parent-promotion <上一圈 promotion.json>`，父策略须来自真实晋升，不能把被拒候选当作下一圈生产父策略。

当前编排从合格纹理库按反馈选样，不在圈内执行新 Transfer 生成，Predict 也不进入规划。完整阶段顺序见 [workflow](workflow.md)。

## 5. 看进度与结果

```bash
tail -f "$RSI_LAUNCH_DIR/logs/iteration.log"
```

日志中的 `记录目录：` 会出现多次：整圈和各子阶段都有独立目录。认准整圈的 `rsi_iteration_…` 目录，并设置：

```bash
export ITERATION_RUN=/home/thor/rsi/runs/p0/rsi_iteration_实际目录
cat "$ITERATION_RUN/iteration_state.json"
cat "$ITERATION_RUN/progress.json"
```

| 文件 | 用途 |
| --- | --- |
| `iteration_state.json` | 当前阶段、输入身份和已完成阶段摘要，恢复时使用 |
| `logs/*.log` | 采集、训练、评测等实际子阶段输出 |
| `probe.json` | 阶段退出状态、耗时、`peak_mem_gb` 和产物摘要 |
| `iteration.json` | 整圈结束结果与门禁报告路径 |
| 门禁目录的 `iter.json` | `decision`、拒绝原因和晋升资格 |

如需核验队列，`p1_workflow_status.py --config <执行清单>` 只读取 PID、输入摘要和封存产物。仓库中的 `configs/p1_execution.yaml` 是原实验的确切队列，换运行必须准备自己的清单，不能把历史 PID 或路径当作当前状态。

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  envs/policy/bin/python -u scripts/p1_workflow_status.py --config "$EXECUTION_CONFIG"
```

`EXECUTION_CONFIG` 是已按本次执行填写的清单。检查过程不启动或停止模型；`complete_rsi` 不能靠补齐配置文件获得。

## 6. 中断恢复

确认上一次的本项目子进程已退出后，保持相同父策略、handoff、预算、评测、圈号和 seed，再运行原整圈命令并追加：

```text
--resume-from /home/thor/rsi/runs/p0/rsi_iteration_中断目录
```

恢复会建立新的记录目录，复用摘要和阶段校验通过的产物；不要手工把失败状态改成成功。数据、脚本或输入摘要发生变化时，先检查失效原因，再决定是否新开实验。质量门或统计门禁的拒绝是实验结论，不能通过重启包装为通过。

## 7. 门禁与人工晋升

候选必须同时通过对照增益、相对父策略提升、标准/非目标回退及延迟检查，阈值以 [eval_protocol.md](eval_protocol.md) 为准。`reject` 不晋升；只有 `accept_pending_approval` 才进入人工审核。

由人审核真实 `iter.json` 和完整候选结果，取得可核验的审批记录后执行：

```bash
envs/policy/bin/python -u scripts/p1_promote.py \
  --gate "$ITER" --approve-gate-sha256 "$SHA" --approval-reference "$APPROVAL"
```

`ITER` 是门禁文件，`SHA` 是审核的该文件摘要，`APPROVAL` 是明确审批记录。成功后保存晋升权重、原子更新 `checkpoints/current` 与 `previous`，并执行生产冒烟；失败自动恢复原链接并写 `rollback.json`。下一圈须绑定此次 `promotion.json`。

## 8. 可选诊断

Reason 配置比较、Predict 离线验证和反馈后的新 Transfer 生成在整圈释放模型后独立串行执行。入口分别是 `p1_reason_followup.py`、`p1_predict_validation.py`、`p1_feedback_generation.py`。等待关系要传入确切上游目录；新生成仍需独立准入，不回填到已经结束的训练中。

CPU 素材导出脚本保留为代码工具，其生成的图片与视频不进入发布仓库。

## 9. 备份与发布同步

两条路径用途不同：`backup_to_lenovo.sh` 保存实验工作区的代码、元数据、精选媒体及可选晋升权重；`release/sync_from_workspace.py` 整理公开代码仓库，只带运行所需文本源文件。

备份示例：

```bash
export RSI_BACKUP_DEST='备份用户名@备份主机:/实际备份目录'
read -rsp 'SSH password: ' SSHPASS
printf '\n'
export SSHPASS
bash scripts/backup_to_lenovo.sh --dry-run
# 检查目标和预览后执行实际备份；--with-ckpt 只增加已晋升权重
bash scripts/backup_to_lenovo.sh --with-ckpt
unset SSHPASS
```

密码只存于进程环境；已配置 SSH key 时无需 `SSHPASS`。推送备份带 `--delete`，务必核对目标目录与预览。拉取时同时设置 `RSI_THOR_SOURCE` 与本地 `RSI_BACKUP_DEST`，再加 `--pull-from-thor`；拉取不删除目标文件。

在独立发布仓库根目录整理上传内容：

```bash
python3 release/sync_from_workspace.py --workspace /实际完整工作区 --dry-run
python3 release/sync_from_workspace.py --workspace /实际完整工作区
git status --short
```

工具按入口和 import 依赖选择脚本、测试与配置，README、安装与使用说明、数据契约及评测协议由本仓库独立维护，避免被工作区旧文档覆盖。旧过程文档和素材会从发布目录清理；不会删除源工作区、运行数据或第三方克隆。源工作区与发布仓库必须是两个不同目录。

同步清单写入 `release/sync_manifest.json`。`.gitignore` 进一步排除图片、PPT、过程笔记、模型、运行产物与缓存。同步和本地提交均不会自动推送远端。
