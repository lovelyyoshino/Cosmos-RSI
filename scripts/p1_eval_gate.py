#!/usr/bin/env python3
"""用真实完整冻结评测及配对训练记录计算晋升门禁，不修改生产权重。

@author pony
@date 2026-11-09
@version v1.1.0
@last_modified 2026-11-09
@changelog
    - v1.1.0 (2026-11-09): 为后续圈采集及部署统一核验真实晋升记录、当前链接与父权重。
    - v1.0.0 (2026-11-09): 绑定训练、逐回合真值、完整冻结集和服务演练，保留拒绝结论。
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import pathlib

from p0_background_preview_materials import checked_artifact
from p0_closeout import validate_budget
from p0_common import ROOT, run_stage
from p0_lerobot_handoff import dump
from p0_paired_sampling import paired_plan_sha256


SPLITS = {"standard": 200, "target_perturbed": 800, "nontarget_perturbed": 200}


def sha256(path):
    """P1自有的流式摘要兼容备份机Python3.10，不改变已准入P0方法的依赖源码。"""
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value, *, minimum=0.):
    if type(value) not in {int, float} or not math.isfinite(value) or value < minimum:
        raise ValueError("统计量必须为规定范围内的有限数")
    return float(value)


def rates(counts):
    result = {}
    if set(counts) != set(SPLITS):
        raise ValueError("冻结子集不完整")
    for split, expected in SPLITS.items():
        row = counts[split]
        if type(row["n"]) is not int or row["n"] != expected \
                or type(row["success"]) is not int or not 0 <= row["success"] <= expected:
            raise ValueError("冻结回合数或成功计数不合法")
        p = row["success"] / expected
        if finite(row["p"]) != p:
            raise ValueError("成功率与实际计数不一致")
        result[split] = p
    return result


def decision(candidate, control, incumbent, latency_ratio):
    """严格保留既有协议阈值；流程运行成功与科学假设通过分别记录。"""
    c, k, old = map(rates, [candidate, control, incumbent])
    latency_ratio = finite(latency_ratio, minimum=1e-12)
    target = "target_perturbed"

    def se(a, b, split):
        n = SPLITS[split]
        return math.sqrt(a[split] * (1 - a[split]) / n + b[split] * (1 - b[split]) / n)

    target_se = se(c, k, target)
    threshold = max(.025, 2 * target_se)
    # 同一冻结规模下先减整数计数，避免恰好2.5点被浮点相减误判为低于阈值。
    gain = (candidate[target]["success"] - control[target]["success"]) / SPLITS[target]
    reasons = []
    if gain < threshold:
        reasons.append("目标扰动相对匹配对照的增益未达到max(2.5点,2SE)")
    if c[target] <= old[target]:
        reasons.append("目标扰动未严格超过当前策略")
    regression = {}
    for split in ["standard", "nontarget_perturbed"]:
        limit = max(.03, 2 * se(c, old, split))
        drop = (incumbent[split]["success"] - candidate[split]["success"]) / SPLITS[split]
        regression[split] = {"drop": drop, "maximum_drop": limit, "passed": drop <= limit}
        if drop > limit:
            reasons.append(f"{split}回退超过max(3点,2SE)")
    if latency_ratio > 1.2:
        reasons.append("同观测服务演练的p95延迟增加超过20%")
    return {"decision": "accept_pending_approval" if not reasons else "reject",
            "statistical_gate_passed": not reasons, "eligible_for_promotion": not reasons,
            "target_gain_vs_control": gain, "target_se_diff": target_se,
            "target_minimum_gain": threshold,
            "target_gain_vs_incumbent": (candidate[target]["success"] - incumbent[target]["success"]) / SPLITS[target],
            "regression": regression, "latency_p95_ratio": latency_ratio, "reject_reasons": reasons,
            "human_approval_required": True, "production_deployed": False, "complete_rsi": False,
            "uncertainty_boundary": "协议SE将回合作为计数单位；同初态策略种子相关，另报聚类区间及训练种子复验"}


def sealed(path, stage):
    path = path.resolve(strict=True)
    if not path.is_relative_to(ROOT / "runs/p0"):
        raise ValueError("阶段产物必须位于本项目的实际运行目录")
    probe = checked_artifact(path, stage)
    if probe.get("hardware", {}).get("host") != "thor" \
            or probe.get("hardware", {}).get("arch") != "aarch64":
        raise ValueError("阶段没有Thor运行证据")
    return json.loads(path.read_text()), probe


def bound_file(run, probe, relative):
    path = (run / relative).resolve(strict=True)
    if not path.is_relative_to(run) or probe["artifacts_sha256"].get(relative) != sha256(path):
        raise ValueError("阶段附属文件越界或摘要不符")
    return path


def checked_update(path, role):
    report, probe = sealed(path, "p1_policy_train")
    if report.get("schema") != "thor-feedback-policy-update-v1" or report["role"] != role \
            or report.get("training_performed") is not True or report.get("eligible_for_evaluation") is not True \
            or report.get("eligible_for_promotion") is not False \
            or finite(report["reload_max_abs_diff"]) > 1e-5:
        raise ValueError("正式配对训练记录不合法，预算冒烟不可替代")
    checkpoint = pathlib.Path(report["output_ckpt"]).resolve(strict=True)
    if not checkpoint.is_relative_to(path.parent):
        raise ValueError("检查点不属于实际训练阶段")
    meta_path = bound_file(path.parent, probe, str((checkpoint / "checkpoint.json").relative_to(path.parent)))
    weights = bound_file(path.parent, probe, str((checkpoint / "state.pt").relative_to(path.parent)))
    metadata = json.loads(meta_path.read_text())
    if metadata["state_sha256"] != report["output_ckpt_sha256"] or sha256(weights) != report["output_ckpt_sha256"] \
            or metadata.get("formal_policy_update") is not True or metadata.get("budget_only") is not False \
            or metadata["role"] != role or metadata["paired_plan_sha256"] != report["paired_plan_sha256"] \
            or metadata["parent_ckpt_sha256"] != report["parent_ckpt_sha256"] \
            or metadata["recipe"] != report["recipe"] or metadata["iteration"] != report["iteration"]:
        raise ValueError("正式训练与真实检查点错绑")
    plan_path = bound_file(path.parent, probe, "paired_plan.json")
    plan = json.loads(plan_path.read_text())
    if paired_plan_sha256(plan) != report["paired_plan_sha256"] or plan["sha256"] != report["paired_plan_sha256"]:
        raise ValueError("训练的实际抽样计划不符")
    return report, metadata


def checked_promotion(path):
    """后续圈仅承接真实审批、生产链接冒烟和保存权重一致的晋升记录。"""
    report, _ = sealed(path, "p1_promotion")
    gate_path = pathlib.Path(report["gate"]).resolve(strict=True)
    gate, _ = sealed(gate_path, "p1_evaluation_gate")
    checkpoint = pathlib.Path(report["promoted_checkpoint"]).resolve(strict=True)
    if report.get("schema") != "thor-rsi-promotion-v1" or report.get("production_deployed") is not True \
            or report.get("explicit_user_approval") is not True or not report.get("approval_reference", "").strip() \
            or report.get("production_link_smoke_passed") is not True \
            or report["gate_sha256"] != sha256(gate_path) or gate["eligible_for_promotion"] is not True \
            or gate["decision"] != "accept_pending_approval" or gate["iteration"] != report["iteration"] \
            or gate["candidate_ckpt_sha256"] != report["policy_ckpt_sha256"] \
            or not checkpoint.is_relative_to(ROOT / "checkpoints/promoted") \
            or sha256(checkpoint / "state.pt") != report["policy_ckpt_sha256"] \
            or sha256(checkpoint / "checkpoint.json") != report["promoted_metadata_sha256"]:
        raise ValueError("晋升记录、具体审批或实际保存权重不符")
    return report


def checked_current_promotion(checkpoint, policy_sha, promotion_path, *, preceding_iteration):
    """后续圈使用明确的已审批晋升；不能仅凭文件位于promoted目录获得资格。"""
    if promotion_path is None:
        raise ValueError("已晋升父权重缺少明确晋升记录")
    checkpoint = pathlib.Path(checkpoint).resolve(strict=True)
    promotion_path = pathlib.Path(promotion_path).resolve(strict=True)
    promotion = checked_promotion(promotion_path)
    current = ROOT / "checkpoints/current"
    if not checkpoint.is_relative_to(ROOT / "checkpoints/promoted") \
            or not current.is_symlink() or current.resolve(strict=True) != checkpoint \
            or pathlib.Path(promotion["promoted_checkpoint"]).resolve(strict=True) != checkpoint \
            or promotion["policy_ckpt_sha256"] != policy_sha \
            or type(preceding_iteration) is not int or preceding_iteration < 1 \
            or promotion["iteration"] != preceding_iteration:
        raise ValueError("父权重、上一圈晋升与生产当前链接不一致")
    return promotion, {"path": str(promotion_path), "sha256": sha256(promotion_path)}


def checked_evaluation(path, expected_sha, frozen):
    report, probe = sealed(path, "policy_evaluation")
    if report["policy_ckpt_sha256"] != expected_sha or report["eval_set_sha256"] != frozen["sha256"] \
            or report["full_eval_complete"] is not True or report["completed_episodes"] != sum(SPLITS.values()) \
            or report["server_stopped"] is not True or report["server_forced_kill"] is not False \
            or report["server_returncode"] != 0 or report["real_robot_commands_sent"] is not False:
        raise ValueError("评测版本、完整性或服务退出证据不符")
    episodes_path = bound_file(path.parent, probe, "episodes.jsonl")
    rows = [json.loads(line) for line in episodes_path.read_text().splitlines()]
    expected = {row["eval_id"]: row for row in frozen["episodes"]}
    actual = {row["eval_id"]: row for row in rows}
    if len(actual) != len(rows) or set(actual) != set(expected):
        raise ValueError("实际逐回合记录不等于完整冻结计划")
    counts = {key: {"n": 0, "success": 0} for key in SPLITS}
    runtime = {}
    for identity, row in actual.items():
        fixed = expected[identity]
        if any(row[key] != fixed[key] for key in ["condition_id", "split", "init_seed", "policy_seed", "initial_state_sha256"]) \
                or row["is_eval"] is not True or row["eligible_for_training"] is not False \
                or row["outcome"].get("source") != "env" or type(row["outcome"].get("success")) is not bool \
                or row["policy_ckpt_sha256"] != expected_sha or row["eval_set_sha256"] != frozen["sha256"] \
                or row["batch"] != 1 or row["control_hz"] != 20 or row["action_horizon"] != 8:
            raise ValueError("实际评测的冻结身份、环境真值或执行方式不符")
        count = counts[row["split"]]
        count["n"] += 1
        count["success"] += int(row["outcome"]["success"])
        runtime[identity] = {key: row[key] for key in ["env", "precision", "policy_noise_schedule", "control_hz", "action_horizon"]}
    for count in counts.values():
        count["p"] = count["success"] / count["n"] if count["n"] else None
    if counts != report["splits"]:
        raise ValueError("报告汇总与逐回合环境成败不一致")
    rates(counts)
    runtime["hardware"] = probe["hardware"]
    runtime["upstream"] = {key: probe["upstream_sha"][key] for key in ["starVLA", "LIBERO", "LIBERO-plus"]}
    return report, runtime


def evaluate_gate(directory, args):
    import yaml

    budget = yaml.safe_load(args.budget.read_text())
    frozen_path = ROOT / budget["evaluation"]["definition"]
    validate_budget(budget, sha256(frozen_path))
    frozen = {**json.loads(frozen_path.read_text()), "sha256": sha256(frozen_path)}
    candidate, cm = checked_update(args.candidate_training, "candidate")
    control, km = checked_update(args.control_training, "control")
    fields = ["iteration", "parent_ckpt_sha256", "paired_plan_sha256", "steps", "batch", "seed", "recipe", "data_qualification"]
    if any(candidate[key] != control[key] for key in fields) \
            or candidate["steps"] != budget["training"]["steps_per_role"] \
            or candidate["batch"] != budget["training"]["batch"] or candidate["seed"] not in budget["training"]["seeds"] \
            or candidate["data_qualification"]["budget_sha256"] != sha256(args.budget):
        raise ValueError("候选和复制对照未使用同一真实父权重、数据计划及预算")
    for role, report in [("candidate", candidate), ("control", control)]:
        if {key: row["count"] for key, row in report["data_mix"].items() if row["count"]} != \
                {key: value for key, value in budget["training"][role + "_quotas"].items() if value}:
            raise ValueError("实际训练配额与预声明不符")
    if cm["base_manifest_sha256"] != km["base_manifest_sha256"] or cm["cache_manifest_sha256"] != km["cache_manifest_sha256"]:
        raise ValueError("候选和对照基座或原始示范版本不同")
    qualification = candidate["data_qualification"]
    for name, expected_stage in [("collection", "feedback_rollout"), ("mining", "feedback_mine"), ("handoff", "lerobot_handoff")]:
        path = pathlib.Path(qualification[name]).resolve(strict=True)
        if sha256(path) != qualification[name + "_sha256"]:
            raise ValueError("正式训练的原始反馈或准入交接已经改变")
        _, source_probe = sealed(path, expected_stage)
        if name == "collection":
            rollout_started_at = source_probe["started_at"]
    reports, runtimes = {}, {}
    for role, expected_sha in [("candidate", candidate["output_ckpt_sha256"]),
                               ("control", control["output_ckpt_sha256"]), ("incumbent", candidate["parent_ckpt_sha256"])]:
        reports[role], runtimes[role] = checked_evaluation(getattr(args, role + "_evaluation"), expected_sha, frozen)
    if any(runtimes[role] != runtimes["incumbent"] for role in ["candidate", "control"]):
        raise ValueError("评测环境、上游版本、功耗或执行约定变化，不能复用当前策略基线")
    smoke, _ = sealed(args.deployment_smoke, "augmented_deploy_smoke")
    if smoke["passed"] is not True or smoke["policy_ckpt_sha256"] != candidate["output_ckpt_sha256"] \
            or smoke["parent_ckpt_sha256"] != candidate["parent_ckpt_sha256"] \
            or smoke["training_report_sha256"] != sha256(args.candidate_training) \
            or smoke["production_deployed"] is not False or smoke["rollback_action_exact"] is not True \
            or smoke["production_slots_and_frozen_eval_unchanged"] is not True:
        raise ValueError("服务演练未通过或不属于待评候选")
    result = decision(*(reports[role]["splits"] for role in ["candidate", "control", "incumbent"]), smoke["latency_p95_ratio"])
    bindings = {name: {"path": str(value), "sha256": sha256(value)} for name, value in vars(args).items()}
    result.update(schema="thor-rsi-evaluation-gate-v1", iteration=candidate["iteration"],
                  candidate_ckpt_sha256=candidate["output_ckpt_sha256"], parent_ckpt_sha256=candidate["parent_ckpt_sha256"],
                  candidate_checkpoint=candidate["output_ckpt"], parent_checkpoint=candidate["parent_checkpoint"],
                  frozen_eval_sha256=frozen["sha256"], inputs=bindings,
                  rollout_started_at=rollout_started_at,
                  rollout_to_gate_wall_s=(dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(rollout_started_at)).total_seconds(),
                  manual_interventions=None, manual_interventions_status="尚未完整登记，不能据此宣称逐圈下降",
                  measurements={role: {key: report[key] for key in ["splits", "precision", "batch", "policy_noise_schedule", "latency_ms", "sample_count"]}
                                for role, report in reports.items()},
                  new_cosmos_generation_performed=candidate["new_cosmos_generation_performed"],
                  lighting_generation_qualified=candidate["data_qualification"]["lighting_generation_qualified"])
    dump(directory / "iter.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ["candidate-training", "control-training", "candidate-evaluation", "control-evaluation", "incumbent-evaluation", "deployment-smoke"]:
        parser.add_argument("--" + name, type=pathlib.Path, required=True)
    parser.add_argument("--budget", type=pathlib.Path, default=ROOT / "configs/iteration_budget.yaml")
    args = parser.parse_args()
    for name, value in vars(args).items():
        setattr(args, name, value.resolve(strict=True))
    raise SystemExit(run_stage("p1_evaluation_gate", lambda directory: evaluate_gate(directory, args),
                               {"compute": "Thor CPU来源和统计核验，无训练、策略推理或真机运动"}))
