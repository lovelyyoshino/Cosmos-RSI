#!/usr/bin/env python3
"""只读核验真实组件交接并固定首圈预算，区分P0完成、P1校准与RSI收益。

@author pony
@date 2026-11-09
@version v1.1.0
@last_modified 2026-11-09
@changelog
    - v1.1.0 (2026-11-09): 正式执行前拒绝非整数训练预算及改变既有统计、回退、延迟门槛。
    - v1.0.0 (2026-11-09): 复用有SHA的实际产物、核对父权重与数据交接，不重跑组件冒烟。
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib

from p0_common import ROOT, run_stage
from p0_lerobot_handoff import dump
from p0_pipeline_status import evidence_for
from p0_policy_runtime import sha256


REQUIRED = ["baseline", "feedback", "reason", "mining", "generated_data", "candidate_budget",
            "control_budget", "analytic_budget", "deployment_smoke", "predict_transition", "predict_embeddings", "predict"]


def validate_budget(budget, frozen_sha256):
    if budget["schema"] != "thor-rsi-iteration-budget-v1" or budget["phase"] != "P1" \
            or budget["evaluation"]["sha256"] != frozen_sha256:
        raise ValueError("首圈预算版本、阶段或冻结定义不符")
    training = budget["training"]
    if training["roles"] != ["candidate", "control", "analytic_ablation"] \
            or type(training["steps_per_role"]) is not int or type(training["batch"]) is not int \
            or type(training["sample_slots_per_role"]) is not int \
            or training["steps_per_role"] <= 0 or training["batch"] <= 0 \
            or training["sample_slots_per_role"] != training["steps_per_role"] * training["batch"] \
            or sum(training["candidate_quotas"].values()) != training["sample_slots_per_role"] \
            or sum(training["control_quotas"].values()) != training["sample_slots_per_role"] \
            or any(type(value) is not int or value < 0 for quotas in [training["candidate_quotas"], training["control_quotas"]]
                   for value in quotas.values()) \
            or training["control_quotas"]["generated"] or training["control_quotas"]["self_distill"] \
            or training["candidate_quotas"]["demo"] != training["control_quotas"]["demo"] \
            or training["absent_qualified_pool"] != "stop_and_report":
        raise ValueError("候选与对照预算不匹配或缺失合格来源时未停止")
    expected = {"standard": 200, "target_perturbed": 800, "nontarget_perturbed": 200}
    if any(budget["evaluation"][role] != expected for role in ["candidate", "control"]) \
            or budget["promotion"]["human_approval_required"] is not True \
            or budget["promotion"]["real_robot_enabled"] is not False \
            or budget["promotion"]["min_gain_vs_control"] != "max_0.025_or_2se" \
            or budget["promotion"]["require_gain_vs_incumbent"] is not True \
            or budget["promotion"]["max_standard_nontarget_drop"] != "max_0.03_or_2se" \
            or budget["promotion"]["max_latency_p95_ratio"] != 1.2 \
            or budget["promotion"]["consecutive_accepted_iterations_required"] != 2:
        raise ValueError("冻结门禁规模或晋升审批边界不符")


def validate_links(evidence):
    if any(not evidence.get(name, {}).get("artifact_verified") for name in REQUIRED):
        raise ValueError("P0尚有缺失或未封存的组件产物")
    def data(name):
        return evidence[name]["latest_valid"]["data"]
    def digest(name):
        return evidence[name]["latest_valid"]["sha256"]
    baseline, feedback, reason, mining = [data(name) for name in ["baseline", "feedback", "reason", "mining"]]
    parent = baseline["policy_ckpt_sha256"]
    if any(item["policy_ckpt_sha256"] != parent for item in [feedback, reason, mining]) \
            or reason["collection_sha256"] != digest("feedback") \
            or mining["collection_sha256"] != digest("feedback") or mining["judge_sha256"] != digest("reason"):
        raise ValueError("正式基线与独立反馈/判官/挖掘的父权重或来源错绑")
    candidate, control, analytic = [data(name) for name in ["candidate_budget", "control_budget", "analytic_budget"]]
    for role in [candidate, control, analytic]:
        if role["parent_ckpt_sha256"] != parent or role["handoff_sha256"] != digest("generated_data") \
                or role["paired_plan_sha256"] != candidate["paired_plan_sha256"] \
                or role["reload_max_abs_diff"] != 0 or role["eligible_for_promotion"] is not False \
                or any(role[key] != candidate[key] for key in ["steps", "batch", "seed"]):
            raise ValueError("P0训练预算的父权重、真实交接、配对或重载不符")
    deployed = data("deployment_smoke")
    if deployed["training_report_sha256"] != digest("candidate_budget") \
            or deployed["policy_ckpt_sha256"] != candidate["output_ckpt_sha256"] \
            or deployed["passed"] is not True or deployed["production_deployed"] is not False:
        raise ValueError("部署演练没有使用当前P0预算候选或错误宣称生产晋升")
    prediction = data("predict")
    if prediction["transition_sha256"] != digest("predict_transition") \
            or prediction["embeddings_sha256"] != digest("predict_embeddings") \
            or prediction["same_current_state"] is not True or prediction["same_sampling_seed"] is not True \
            or prediction["eligible_for_planning"] is not False:
        raise ValueError("Predict动作证据错绑或超出接口验证资格")
    return {"parent_ckpt_sha256": parent, "feedback_chain_linked": True,
            "generated_data_training_deployment_linked": True, "predict_chain_linked": True,
            "feedback_directed_generation_demonstrated": False,
            "boundary": "组件交接已核对；现有纹理窗口早于本批反馈，不冒充反馈定向生成"}


def close(directory, args):
    import yaml
    from p0_augmented_budget import checked_handoff
    from p0_augmented_deploy_smoke import checked_training

    pipeline, budget = [yaml.safe_load(path.read_text()) for path in [args.config, args.budget]]
    frozen = ROOT / budget["evaluation"]["definition"]
    validate_budget(budget, sha256(frozen))
    evidence = {name: evidence_for(ROOT, pipeline["evidence"][name]) for name in REQUIRED}
    linked = validate_links(evidence)
    candidate_path = pathlib.Path(evidence["candidate_budget"]["latest_valid"]["artifact"])
    training, parent, _, _ = checked_training(candidate_path)
    checked_handoff(pathlib.Path(evidence["generated_data"]["latest_valid"]["artifact"]),
                    json.loads((parent / "checkpoint.json").read_text()))
    stages = {name: {key: evidence[name]["latest_valid"][key]
                     for key in ["artifact", "sha256", "probe", "probe_sha256", "wall_s", "peak_mem_gb", "hardware", "conditions"]}
              for name in REQUIRED}
    feedback = evidence["feedback"]["latest_valid"]
    baseline = evidence["baseline"]["latest_valid"]["data"]
    collection_episodes = len(budget["collection"]["conditions"]) * budget["collection"]["per_condition"]
    collection_estimate = collection_episodes * feedback["wall_s"] / feedback["data"]["sample_count"]
    training_estimates = {}
    for name in ["candidate_budget", "control_budget", "analytic_budget"]:
        row = evidence[name]["latest_valid"]; value = row["data"]
        samples = value["measured_samples_per_s"]
        if not math.isfinite(samples) or samples <= 0:
            raise ValueError("实际吞吐缺失，不能估算首圈训练预算")
        training_estimates[name] = max(0., row["wall_s"] - value["training_wall_s"]) \
            + budget["training"]["sample_slots_per_role"] / samples
    eval_episodes = sum(sum(budget["evaluation"][role].values()) for role in ["candidate", "control"])
    estimates = {"collection_episodes": collection_episodes, "collection_serial_linear_estimate_s": collection_estimate,
        "training_per_role_linear_estimate_s": training_estimates, "evaluation_episodes": eval_episodes,
        "evaluation_linear_estimate_s": eval_episodes / baseline["rollout_episodes_per_s"],
        "source_measurements": stages,
        "limitation": "均为按P0实测线性排期；未包含新窗口生成/质量拒绝、额外校准、审批和重试，不是整圈实测"}
    result = {"schema": "thor-p0-closeout-v1", "p0_component_acceptance_passed": True,
        "config": str(args.config), "config_sha256": sha256(args.config),
        "budget": str(args.budget), "budget_sha256": sha256(args.budget), "locked_budget": budget,
        "component_links": linked, "artifacts": stages, "budget_estimates": estimates,
        "reason_calibration_passed": evidence["reason"]["latest_valid"]["data"]["calibration"]["calibration_passed"],
        "p1_final_acceptance_passed": False, "complete_rsi": False, "promoted_iterations": 0,
        "production_deployed": False, "real_robot_commands_sent": False,
        "claim": "P0各组件至少单样本运行、真实交接与首圈预算验收；不等于P1收益、自动编排或连续RSI完成"}
    dump(directory / "p0_acceptance.json", result)
    print(json.dumps({key: result[key] for key in ["p0_component_acceptance_passed", "reason_calibration_passed",
        "p1_final_acceptance_passed", "complete_rsi"]}, ensure_ascii=False), flush=True)
    return {key: result[key] for key in ["p0_component_acceptance_passed", "complete_rsi"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=pathlib.Path, default=ROOT / "configs/rsi_pipeline.yaml")
    parser.add_argument("--budget", type=pathlib.Path, default=ROOT / "configs/iteration_budget.yaml")
    args = parser.parse_args()
    args.config, args.budget = args.config.resolve(strict=True), args.budget.resolve(strict=True)
    raise SystemExit(run_stage("p0_closeout", lambda directory: close(directory, args),
        {"compute": "Thor CPU只读验收，无模型重跑", "sample_count": None}))
