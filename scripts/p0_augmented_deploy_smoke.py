#!/usr/bin/env python3
"""在独立槽位加载已完成接入冒烟的新权重并回滚，不改生产链接或发送环境动作。

@author pony
@date 2026-11-08
@version v1.3.0
@last_modified 2026-11-09
@changelog
    - v1.3.0 (2026-11-09): 后续圈允许真实已审批父权重位于promoted，仍核验生产current及晋升绑定。
    - v1.2.0 (2026-11-09): 接受经正式训练来源核验的P1候选，仍只演练且不晋升。
    - v1.1.0 (2026-11-08): 支持原生交接预算候选，核对真实父权重并记录实际训练条件。
    - v1.0.0 (2026-11-08): 使用真实新权重验证服务身份、同观测延迟及原权重逐值回滚。
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import pathlib

from p0_background_preview_materials import checked_artifact
from p0_common import ROOT, ensure_gpu_idle, run_stage
from p0_deploy_smoke import atomic_slot, server_phase
from p0_policy_runtime import DemoCache, sha256


def checked_parent_origin(parent, training, stage):
    if stage != "p1_policy_train":
        if not parent.is_relative_to(ROOT / "runs/p0"):
            raise ValueError("P0服务演练父权重不属于本项目运行记录")
        return
    iteration = training["iteration"]
    if iteration == 1:
        if not parent.is_relative_to(ROOT / "runs/p0"):
            raise ValueError("首圈父权重必须来自正式基线运行")
        return
    from p1_eval_gate import checked_current_promotion

    binding = training["data_qualification"].get("parent_promotion")
    if not isinstance(binding, dict):
        raise ValueError("后续圈服务演练缺少真实父策略晋升绑定")
    _, actual = checked_current_promotion(parent, training["parent_ckpt_sha256"], binding["path"],
                                         preceding_iteration=iteration - 1)
    if actual != binding:
        raise ValueError("正式训练绑定的父晋升摘要已变化")


def checked_training(path):
    """P0计算验证与P1正式训练分别验真；后续圈必须承接真实晋升。"""
    stages = {"training_smoke.json": "augmented_training_smoke", "train_run.json": "augmented_budget"}
    stage = stages.get(path.name)
    if stage is None:
        raise ValueError("不支持的P0训练记录")
    actual_stage = json.loads((path.parent / "probe.json").read_text()).get("stage")
    if path.name == "train_run.json" and actual_stage == "p1_policy_train":
        from p1_eval_gate import checked_update

        training, _ = checked_update(path, "candidate")
        stage = "p1_policy_train"
    probe = checked_artifact(path, stage)
    training = json.loads(path.read_text())
    if training["role"] != "candidate" or training["eligible_for_promotion"] is not False \
            or training.get("policy_gain_evaluated", False) is not False:
        raise ValueError("当前只验证未晋升的P0候选")
    candidate = pathlib.Path(training["output_ckpt"]).resolve(strict=True)
    if not candidate.is_relative_to(path.parent):
        raise ValueError("候选检查点不在实际训练目录内")
    metadata_path = candidate / "checkpoint.json"
    relative = str(metadata_path.relative_to(path.parent))
    if probe["artifacts_sha256"].get(relative) != sha256(metadata_path):
        raise ValueError("候选元数据未绑定实际训练阶段")
    metadata = json.loads(metadata_path.read_text())
    if metadata["state_sha256"] != training["output_ckpt_sha256"] \
            or metadata["parent_ckpt_sha256"] != training["parent_ckpt_sha256"] \
            or metadata["role"] != "candidate" or metadata["eligible_for_promotion"] is not False \
            or metadata["paired_plan_sha256"] != training["paired_plan_sha256"]:
        raise ValueError("候选检查点与训练来源不一致")
    parent = pathlib.Path(metadata["parent_checkpoint"]).resolve(strict=True)
    if training.get("parent_checkpoint", str(parent)) != str(parent):
        raise ValueError("训练报告与候选声明的父权重不同")
    checked_parent_origin(parent, training, stage)
    if not candidate.is_relative_to(ROOT / "runs/p0"):
        raise ValueError("候选检查点不属于本项目实际训练记录")
    for item, expected in [(parent, training["parent_ckpt_sha256"]), (candidate, training["output_ckpt_sha256"])]:
        if sha256(item / "state.pt") != expected:
            raise ValueError("部署检查点越界或与实际训练摘要不同")
    parent_meta = json.loads((parent / "checkpoint.json").read_text())
    conditions = {"steps": training["training_steps"] if stage == "augmented_training_smoke" else training["steps"],
                  "batch": training["batch"], "seed": training["seed"]}
    if any(type(conditions[key]) is not int for key in conditions) or conditions["steps"] <= 0 \
            or conditions["batch"] <= 0 or not 0 <= conditions["seed"] < 2 ** 32 \
            or metadata["step"] - parent_meta["step"] != conditions["steps"] \
            or any(metadata["recipe"][key] != value for key, value in conditions.items()):
        raise ValueError("训练步数、batch或种子与权重记录不同")
    return training, parent, candidate, conditions


def verify(directory, args):
    ensure_gpu_idle()
    import numpy as np
    from PIL import Image

    training, parent, candidate, training_conditions = checked_training(args.training_report)
    meta = json.loads((parent / "checkpoint.json").read_text())
    cache = DemoCache(pathlib.Path(meta["cache_manifest"]).parent)
    if sha256(cache.manifest_path) != meta["cache_manifest_sha256"]:
        raise ValueError("服务探针的示范数据版本已改变")
    location = meta["prediction_probe"]["locations"][0]
    episode, = [r for r in cache.manifest["episodes"] if r["episode_id"] == location["episode_id"]]
    if episode["split"] != "demo_validation" or not 0 <= location["frame"] < episode["length"]:
        raise ValueError("只允许原有固定离线验证观测")
    offset = episode["offset"] + location["frame"]
    images = []
    for view, name in enumerate(["agentview", "wrist"]):
        image = Image.fromarray(np.array(cache.images[offset, view]))
        image.save(directory / (name + ".png"))
        stream = io.BytesIO(); image.save(stream, format="PNG")
        images.append(base64.b64encode(stream.getvalue()).decode())
    payload = {"instruction": episode["instruction"], "state": cache.states[offset].tolist(),
               "images_png_base64": images, "seed": 44}
    frozen = sha256(ROOT / "eval/frozen_eval_set.json")
    production_paths = [ROOT / "checkpoints/current", ROOT / "checkpoints/previous"]

    def production_state():
        return {str(p): {"exists": p.exists(), "link": os.readlink(p) if p.is_symlink() else None} for p in production_paths}

    production = production_state()
    slots = directory / "deployment_slots"; slots.mkdir()
    atomic_slot(slots, "current", parent, ROOT)
    initial = server_phase(directory, "initial", slots / "current", payload, 18793, 5, 50)
    atomic_slot(slots, "previous", parent, ROOT)
    atomic_slot(slots, "current", candidate, ROOT)
    updated = server_phase(directory, "updated", slots / "current", payload, 18793, 5, 50)
    atomic_slot(slots, "current", parent, ROOT)
    rollback = server_phase(directory, "rollback", slots / "current", payload, 18793, 0, 1)
    original_actions = np.asarray(initial["response"]["normalized_actions"])
    rollback_actions = np.asarray(rollback["response"]["normalized_actions"])
    exact = bool(np.array_equal(original_actions, rollback_actions))
    ratio = updated["latency_ms"]["server_ms"]["p95"] / initial["latency_ms"]["server_ms"]["p95"]
    unchanged = production == production_state() and frozen == sha256(ROOT / "eval/frozen_eval_set.json")
    result = {"passed": exact and unchanged, "training_report": str(args.training_report),
              "training_report_sha256": sha256(args.training_report), "policy_ckpt_sha256": training["output_ckpt_sha256"],
              "parent_ckpt_sha256": training["parent_ckpt_sha256"], "rollback_action_exact": exact,
              "production_slots_and_frozen_eval_unchanged": unchanged, "latency_p95_ratio": ratio,
              "phases": [{k: v for k, v in phase.items() if k not in {"rows", "response"}}
                         for phase in [initial, updated, rollback]],
              "precision": meta["precision"], "batch": 1, "action_horizon": 8, "noise_seed": 44,
              "warmup_per_model": 5, "latency_requests_per_model": 50, "distinct_observations": 1,
              "training_steps_added": training_conditions["steps"], "training_batch": training_conditions["batch"],
              "training_seed": training_conditions["seed"],
              "environment_actions_sent": False, "production_deployed": False,
              "eligible_for_promotion": False, "policy_gain_evaluated": False,
        "scope": "独立服务槽位的切换与回滚；完整任务评测与正式晋升由另外的门禁负责"}
    (directory / "deployment_smoke.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    if not result["passed"]:
        raise ValueError("新权重服务或逐值回滚未通过")
    return {key: result[key] for key in ["passed", "policy_ckpt_sha256", "rollback_action_exact", "latency_p95_ratio",
                                       "production_deployed", "eligible_for_promotion"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--training-report", type=pathlib.Path, required=True)
    arguments = parser.parse_args()
    arguments.training_report = arguments.training_report.resolve(strict=True)
    if not arguments.training_report.is_relative_to(ROOT / "runs/p0"):
        parser.error("训练报告必须位于本项目P0目录")
    raise SystemExit(run_stage("augmented_deploy_smoke", lambda directory: verify(directory, arguments),
        {"compute": "Thor策略服务，无环境动作", "precision": "VLM bf16/LoRA与动作头fp32", "batch": 1,
         "action_horizon": 8, "seed": 44, "warmup": 5, "repeats": 50, "distinct_observations": 1,
         "training_report": str(arguments.training_report)}))
