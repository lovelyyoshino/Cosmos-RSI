# 阶段间数据约定

> 各阶段只通过这里定义的文件交接。字段可以增加，但不能改名，也不能删除。如果必须改，先改本文件，再改代码。所有 jsonl 都是一行一条记录，UTF-8 编码；时间用 ISO 8601 格式并带时区。

下文的 JSON 数字为格式示例，不是实验实测。§1 的 `iter_k` 为逻辑目录，当前实现按阶段写入 `runs/p0/<阶段>_<时间>/`，由编排记录关联。实际读取前必须检查报告的 `schema`、源码定义和 SHA256，不能照抄示例造出缺失字段。

当前实现的交接文件如下，具体调用关系见 [workflow](workflow.md)：

| 阶段 | 实际产物 | 主要校验 |
| --- | --- | --- |
| 反馈采集与挖掘 | `feedback_chain.json`、`episodes.jsonl`、`failure_records.jsonl` | 父权重身份、独立初态和策略种子、环境真值、上游产物摘要 |
| 生成准入与交接 | `quality_gate.json`、`handoff.json` | 实际视频摘要、完整方法控制与复核、逐帧标签、原生读取一致性 |
| 数据准入与训练 | `prepared_data.json`、`train_run.json`、`checkpoint.json`、`state.pt` | 冻结集隔离、配额、父优化器及权重、输出摘要 |
| 部署与评测 | `deployment_smoke.json`、`evaluation.json`、`analysis.json` | 切换和回滚、完整冻结定义及运行条件、逐回合真值 |
| 门禁与晋升 | `iter.json`、`promotion.json`、失败时 `rollback.json` | 统计门槛、人工审批摘要、权重和链接状态 |
| 编排状态 | `iteration_state.json`、`progress.json`、`iteration.json` | 确切输入绑定、完成阶段及其 SHA256、恢复条件 |

门禁实际决策为 `reject` 或 `accept_pending_approval`；后者仍需人工审批后运行晋升入口。示例中的逻辑门禁与部署清单不代替实际文件。

## 1. 目录布局（每圈一个目录，写完即只读）

```text
runs/
  iter_000/                      # 基线 π_0 的运行
    iter.json                    # 本圈总览：父 ckpt、各阶段起止时间、结论
    rollouts/                    # §2
    failures/                    # §3
    gen/                         # §4
    train/candidate/  train/control/   # §5
    gate/gate_report.json        # §6
    deploy/deploy_manifest.json  # §7
  iter_001/ ...
eval/
  frozen_eval_set.json           # 冻结评测集（eval_protocol §1），带 sha256
checkpoints/
  current -> ...                 # Thor 上同名软链接，指向当前在线的 ckpt
  previous -> ...
```

规则：
- 主副本在 Thor 的 `~/rsi/`（本仓库加 `runs/`、`checkpoints/` 等）。每圈结束后，把本圈元数据（`runs/iter_k/` 中除视频和 LeRobot 数据之外的部分）和已晋升的权重 rsync 到 lenovo 的 `<backup-root>/`。
- 每圈的产物写进 `runs/iter_k/` 之后不再修改。需要重跑时，开新目录 `iter_k_rerun1/`。
- 所有记录都要带上 `iter`、`policy_ckpt_sha256`，以及 `git_sha`（本仓库加各 third_party 的 SHA）。

## 2. 回合（rollouts）

数据本体用 **LeRobot v3** 格式，starVLA 和 GR00T 都能直接读取。另外附一份元数据文件 `rollouts/episodes.jsonl`：

```json
{
  "episode_id": "iter001-libplus-t03-light2-s0042",
  "iter": 1,
  "policy_ckpt_sha256": "…",
  "env": "libero_plus | real_<robot>",
  "task_id": "libero_10/3",
  "instruction": "put both the alphabet soup and the tomato sauce in the basket",
  "conditions": {"light": "warm_dim_2", "background": "default", "layout": "default", "camera": "default", "init": "default"},
  "init_seed": 42,
  "outcome": {"success": false, "source": "env | judge | human", "judge_score": 0.12, "judge_text": "…"},
  "metrics": {"steps": 520, "duration_s": 26.0, "regrasp_count": 2, "action_dispersion_p90": 0.031},
  "interventions": 0,
  "lerobot_index": {"dataset": "rollouts/lerobot", "episode_index": 17},
  "is_eval": false
}
```

- `is_eval` 为 true 的回合只能用于评测，不能出现在任何训练集中（§5 会校验）。
- `action_dispersion_p90`：同一观测下用 K 个噪声种子采样动作，计算离散度，再取整个回合的 p90。K 的取值和算法写在 `configs/rollout.yaml` 中。

## 3. 失败记录（failures/failure_records.jsonl）

```json
{
  "episode_id": "…",
  "kind": "failure | hard_success",
  "category": "visual | behavioral | unknown",
  "sub_type": "lighting | texture | background | sensor_noise | grasp_miss | collision | timeout | layout | init_pose | other",
  "evidence": {"key_frames": [120, 340], "judge_text": "…", "rule_hits": ["regrasp>=2"]},
  "route": "cosmos_restyle | self_distill | need_human_demo | ignore",
  "target_condition": {"light": "warm_dim_2"},
  "weight": 1.0
}
```

另外输出一份 `failures/failure_matrix.csv`：行为 task_id，列为各条件，值为失败率和样本数。生成预算按这个矩阵分配。

## 4. 生成清单（gen/gen_manifest.jsonl）

```json
{
  "gen_id": "iter001-g000123",
  "source_episode_id": "demo-libero10-t03-0007",
  "source_kind": "demo | rollout_success",
  "window": {"start_frame": 120, "end_frame": 213, "phase": "pre_grasp | grasp | transport | place"},
  "target_condition": {"light": "warm_dim_2"},
  "view": "agentview | wrist",
  "control": {"types": ["edge"], "source": "rgb_canny | sim_gt | rgbd", "mask": "robot+target_objects", "mask_source": "sim_seg | sam2"},
  "prompt": "…",
  "seed": 7,
  "model": {"name": "nvidia/Cosmos-Transfer2.5-2B", "variant": "distilled/edge | base", "num_steps": 4, "resolution": "480p | 720p", "revision": "…", "runtime": "cosmos-transfer2.5@<sha>"},
  "output": "gen/videos/iter001-g000123_agentview.mp4",
  "gate": {"edge_f1": 0.71, "robot_mask_iou": 0.88, "object_mask_iou": 0.83, "flicker_ok": true, "passed": true, "reject_reason": null},
  "lerobot_index": {"dataset": "gen/lerobot", "episode_index": 55},
  "wall_s": 41.2,
  "peak_mem_gb": 0
}
```

- 每个生成窗口作为一段**部分 episode** 入库，只覆盖 `[start_frame, end_frame)`。其中的动作、状态和语言字段必须与源 episode 同一区间逐帧一致，只替换图像，入库时要校验。训练采样时，动作块不能跨出窗口边界。
- `gate` 中各阈值写在 `configs/gen_gate.yaml`，被拒样本同样要记录，便于统计拒绝率。
- 入训资格必须同时满足几何、全帧身份/任务语义和**目标外观条件**验收。生成前的成功源证据须显式写入 `target_condition`；禁止生成后修改源证据来补目标条件。历史源未声明时保持拒绝，不能靠事后更名获得资格。
- P0的 `quality_gate.json` 另外保存 `output_video_sha256`、`target_condition`、`target_condition_passed`、`target_condition_review`。检查先核对源证据与生成probe绑定的摘要，以及实际视频与probe产物清单的摘要，再加载评价模型。语义与目标条件复核均须绑定同一生成probe和实际视频摘要，`reviewed_frames` 必须为有序完整的0至92。
- `--condition-review` 接收独立复核JSON：字段为 `source_probe_sha256`（此处指生成probe摘要）、`output_video_sha256`、`target_condition`、`reviewed_frames`、`reviewer`、`passed`、`region_checks`。每个区域包含 `criterion`（验收依据）、`observations`（实际观测）和 `passed`。光照/传感器噪声要求机器人、任务目标物、背景分别验收；背景/纹理要求背景目标达成及非目标前景保持；组合条件取所需区域的并集。只改变背景不能算全场光照达成，几何通过也不能替代目标扰动验收。
- 原始候选布尔mask及其相对源ROI的误差分解仅用于评价器诊断，不是生成侧几何真值。不得清理碎片、按生成图真值换候选或删除拒绝帧后放行。
- 背景派生数据必须与原生Cosmos输出分别登记。新源在首次生成前声明 `target_condition` 与 `composite_recipe`；历史无目标预览始终不可入训。`background_composite` 保存原生probe/视频摘要、源标签摘要和无损RGB输出。旧 `thor-roi-sam2-v1` 及历史拒绝保持。
- 背景派生路线各有独立的方法版本、适用区域和控制报告，不能继承另一方法的准入结论，也不能覆盖已有拒绝。无损前景或桌体像素证明仍须同时通过语义、目标、时序和边界复核。
- 当前 UV 重渲染路线使用 `derived_kind=cosmos_mirrored_uv_atlas_sim_rerender` 和 `geometry_evidence_type=same_sensor_table_uv_render_pixels_v1`。它只改变同帧可见桌体纹理；原生来源、安全检查、纹理与配置摘要、设计及独立源的完整正负控制、物理状态、标签和双视角必须绑定。
- 每次重渲染前用 `declaration.json` 固定 `target_condition` 和配方。语义、背景、时序及边界复核绑定实际视频摘要并覆盖完整窗口。训练时重新调用 `checked_gate` 验证来源、方法报告与复核，不只读取资格布尔值。该数据登记为 `background_composite`，不得冒充原生整段生成或全场光照增强。

## 5. 训练记录（train/{candidate,control}/train_run.json）

```json
{
  "role": "candidate | control",
  "parent_ckpt_sha256": "…",
  "framework": "starVLA CosmosGR00T",
  "data_mix": {
    "demo": {"count": 379, "ids_file": "train/demo_ids.txt"},
    "self_distill": {"count": 120, "ids_file": "train/sd_ids.txt"},
    "generated": {"count": 568, "ids_file": "train/gen_ids.txt"},
    "duplicate_fill": {"count": 0}
  },
  "eval_overlap_check": {"passed": true, "eval_set_sha256": "…"},
  "recipe": {"steps": 1000, "lr": 5e-5, "freeze": ["vision"], "lora": {"r": 16}, "batch": 32, "seed": 42},
  "hardware": "Jetson AGX Thor (power mode: …)",
  "peak_mem_gb": 0,
  "wall_s": 5400,
  "output_ckpt": "…",
  "output_ckpt_sha256": "…"
}
```

- control 的 `data_mix` 中，generated 和 self_distill 两项为 0，由 `duplicate_fill` 用 D_0 的复制补齐到与 candidate 相同的样本量；其余配置完全相同。
- `eval_overlap_check` 不通过时，训练脚本必须直接退出。

## 6. 门禁报告（gate/gate_report.json）

```json
{
  "iter": 1,
  "incumbent": "…sha", "candidate": "…sha", "control": "…sha",
  "splits": {
    "target_perturbed":   {"incumbent": {"p": 0.640, "n": 800, "reused_from": "iter000"}, "candidate": {"p": 0.710, "n": 800}, "control": {"p": 0.650, "n": 800}, "se_diff_vs_control": 0.023},
    "standard":           {"incumbent": {"p": 0.912, "n": 200, "reused_from": "iter000"}, "candidate": {"p": 0.915, "n": 200}},
    "nontarget_perturbed":{"incumbent": {"p": 0.700, "n": 200, "reused_from": "iter000"}, "candidate": {"p": 0.705, "n": 200}}
  },
  "rules": {"min_gain_vs_control": "max(0.025, 2*se_diff)", "max_drop": "max(0.03, 2*se_diff)", "latency_p95_max_ratio": 1.2},
  "decision": "promote | reject",
  "reason": "…",
  "approved_by": "human | auto"
}
```

- `p` 是成功率，`n` 是回合数。`reused_from` 表示该结果复用了哪一圈的评测（eval_protocol §2）。多个评测种子的结果合并后计算 p；如有需要，另加 `per_seed` 字段，列出各个种子的成功率。真机阶段每项加 `wilson95: [lo, hi]`。

## 7. 部署清单（deploy/deploy_manifest.json）

```json
{
  "ckpt_sha256": "…",
  "thor_path": "/home/thor/rsi/checkpoints/iter001_candidate",
  "server_cmd": "…",
  "env": "venv-cosmos3 | <其他>",
  "latency_ms": {"p50": 0, "p95": 0, "measured_over": 200, "includes": "obs encode + VLM + action head（同机，无网络）"},
  "smoke_test": {"episodes": 5, "success": 5},
  "deployed_at": "…",
  "rollback_to": "…sha"
}
```
