#!/usr/bin/env python3
"""从实际probe与产物摘要导出统一配置状态，不凭文件数量或文档勾选宣称RSI完成。

@author pony
@date 2026-11-09
@version v1.4.0
@last_modified 2026-11-09
@changelog
    - v1.4.0 (2026-11-09): 从当前生产权重反查具体审批和连续父子晋升，不再固定显示零圈。
    - v1.3.0 (2026-11-09): 单列Predict多状态诊断；像素重建不得取得规划或任务收益资格。
    - v1.2.0 (2026-11-09): 核验三角色正式权重及同预算绑定，完整门禁与通过结论分开显示。
    - v1.1.0 (2026-11-09): 单列P0组件收口与Reason比较，校准仍为P1最终验收条件。
    - v1.0.0 (2026-11-09): 配置引用、实际证据、校准门槛、正式收益及递归晋升分别核验。
"""
from __future__ import annotations

import argparse
import json
import pathlib

from p0_common import ROOT, run_stage
from p0_lerobot_handoff import dump
from p0_policy_runtime import sha256


def evidence_for(root, specification):
    accepted, rejected = [], []
    for path in sorted((root / "runs/p0").glob(specification["glob"])):
        try:
            data = json.loads(path.read_text())
            if any(type(data.get(key)) is not type(value) or data.get(key) != value
                   for key, value in specification.get("require", {}).items()):
                continue
            probe = json.loads((path.parent / "probe.json").read_text())
            if probe.get("stage") != specification["stage"] or probe.get("status") != "passed" \
                    or probe.get("artifacts_sha256", {}).get(path.name) != sha256(path):
                raise ValueError("阶段或实际产物摘要未通过")
            if specification["stage"] == "p1_policy_train":
                from p1_eval_gate import checked_update

                # 正式训练须核对真实保存权重和配对计划，不能只信训练报告存在。
                checked_update(path, data["role"])
            accepted.append({"artifact": str(path), "sha256": sha256(path), "probe": str(path.parent / "probe.json"),
                "probe_sha256": sha256(path.parent / "probe.json"), "wall_s": probe.get("wall_s"),
                "peak_mem_gb": probe.get("peak_mem_gb"), "hardware": probe.get("hardware"),
                "conditions": probe.get("conditions"), "data": data})
        except (ValueError, KeyError, OSError) as exc:
            rejected.append({"artifact": str(path), "error": str(exc)})
    selected = accepted[-1] if accepted else None
    return {"artifact_verified": selected is not None, "latest_valid": selected, "invalid_reports": rejected}


def anchor_feedback(root, evidence, specifications):
    """主链按封存交接绑定，不让后续独立判官研究覆盖已完成的主圈采集。"""
    chain = evidence.get("feedback_chain", {}).get("latest_valid")
    if chain is None:
        return
    for name, key in [("feedback", "collection"), ("reason", "judge"), ("mining", "mining")]:
        path = pathlib.Path(chain["data"][key]).resolve(strict=True)
        if not path.is_relative_to(root / "runs/p0") \
                or chain["data"]["artifacts_sha256"].get(str(path)) != sha256(path):
            raise ValueError("主反馈交接的采集、判官或挖掘来源失效")
        specification = {**specifications[name], "glob": str(path.relative_to(root / "runs/p0"))}
        selected = evidence_for(root, specification)
        if not selected["artifact_verified"]:
            raise ValueError("主反馈交接的实际阶段证据未通过")
        evidence[name] = selected


def promotion_history(root):
    """只计当前生产分支上的完整晋升链；历史记录数和连续递归圈数分别报告。"""
    from p1_eval_gate import checked_promotion, checked_update, sealed

    records, invalid = {}, []
    for path in sorted((root / "runs/p0").glob("p1_promotion_20*/promotion.json")):
        try:
            report = checked_promotion(path)
            gate, _ = sealed(pathlib.Path(report["gate"]), "p1_evaluation_gate")
            if type(report["iteration"]) is not int or report["iteration"] < 1 \
                    or gate.get("statistical_gate_passed") is not True \
                    or gate["parent_ckpt_sha256"] != report["parent_ckpt_sha256"] \
                    or report["policy_ckpt_sha256"] == report["parent_ckpt_sha256"]:
                raise ValueError("晋升圈数、统计结论或父子权重不符")
            for binding in gate["inputs"].values():
                source = pathlib.Path(binding["path"]).resolve(strict=True)
                if not source.is_relative_to(root) or sha256(source) != binding["sha256"]:
                    raise ValueError("晋升门禁的实际输入已改变")
            training, _ = checked_update(pathlib.Path(gate["inputs"]["candidate_training"]["path"]), "candidate")
            if training["iteration"] != report["iteration"] \
                    or training["output_ckpt_sha256"] != report["policy_ckpt_sha256"] \
                    or training["parent_ckpt_sha256"] != report["parent_ckpt_sha256"]:
                raise ValueError("晋升记录没有绑定同一正式候选")
            parent = training["data_qualification"]["parent_promotion"]
            if report["iteration"] == 1:
                if parent is not None:
                    raise ValueError("首圈不能继承另一晋升记录")
            elif not isinstance(parent, dict) or set(parent) != {"path", "sha256"}:
                raise ValueError("后续圈缺少具体父策略晋升证明")
            records[str(path.resolve())] = {
                "path": str(path.resolve()), "sha256": sha256(path), "iteration": report["iteration"],
                "policy_ckpt_sha256": report["policy_ckpt_sha256"],
                "parent_ckpt_sha256": report["parent_ckpt_sha256"],
                "checkpoint": str(pathlib.Path(report["promoted_checkpoint"]).resolve(strict=True)),
                "frozen_eval_sha256": gate["frozen_eval_sha256"], "parent_promotion": parent,
            }
        except (ValueError, KeyError, OSError, TypeError) as error:
            invalid.append({"path": str(path), "error": str(error)})

    result = {"verified_promotion_records": len(records), "promoted_iterations": 0,
              "current_production_verified": False, "current_checkpoint": None,
              "current_chain": [], "invalid_records": invalid, "current_chain_error": None}
    slot = root / "checkpoints/current"
    if not slot.is_symlink() and not slot.exists():
        return result
    try:
        if not slot.is_symlink():
            raise ValueError("当前生产槽位不是软链接")
        target = slot.resolve(strict=True)
        result["current_checkpoint"] = str(target)
        choices = [row for row in records.values() if row["checkpoint"] == str(target)]
        if len(choices) != 1:
            raise ValueError("当前生产权重没有唯一有效晋升记录")
        chain, row = [], choices[0]
        while True:
            if row["path"] in {item["path"] for item in chain}:
                raise ValueError("父策略晋升证明存在循环")
            chain.append(row)
            if row["iteration"] == 1:
                break
            binding = row["parent_promotion"]
            previous = records.get(str(pathlib.Path(binding["path"]).resolve(strict=True)))
            if previous is None or previous["sha256"] != binding["sha256"] \
                    or previous["iteration"] != row["iteration"] - 1 \
                    or previous["policy_ckpt_sha256"] != row["parent_ckpt_sha256"] \
                    or previous["frozen_eval_sha256"] != row["frozen_eval_sha256"]:
                raise ValueError("连续晋升的圈数、父权重、冻结版本或具体证明断链")
            row = previous
        result.update(current_production_verified=True, promoted_iterations=len(chain),
                      current_chain=list(reversed(chain)))
    except (ValueError, KeyError, OSError, TypeError) as error:
        result["current_chain_error"] = str(error)
    return result


def summarize_readiness(evidence, recursive_orchestrator_status, *, promotions=None):
    def data(name):
        row = evidence.get(name, {}).get("latest_valid")
        return row["data"] if row else {}

    judge, mining, predictor = data("reason"), data("mining"), data("predict")
    multistate = data("predict_multistate")
    multistate_complete = bool(multistate.get("schema") == "thor-predict-validation-v1"
        and multistate.get("diagnostic_complete") is True and multistate.get("summary", {}).get("states", 0) > 1
        and multistate.get("summary", {}).get("eligible_for_planning") is False
        and multistate.get("summary", {}).get("eligible_for_training") is False)
    feedback = data("feedback")
    feedback_linked = bool(judge and mining and feedback
        and judge.get("collection_sha256") == evidence["feedback"]["latest_valid"]["sha256"]
        and mining.get("collection_sha256") == judge.get("collection_sha256")
        and mining.get("judge_sha256") == evidence["reason"]["latest_valid"]["sha256"])
    budget_matched = bool(data("candidate_budget") and data("control_budget")
        and data("candidate_budget").get("paired_plan_sha256") == data("control_budget").get("paired_plan_sha256")
        and data("candidate_budget").get("parent_ckpt_sha256") == data("control_budget").get("parent_ckpt_sha256"))
    formal = [data("formal_" + role) for role in ["candidate", "control", "analytic_ablation"]]
    formal_matched = all(row.get("schema") == "thor-feedback-policy-update-v1"
        and row.get("training_performed") is True and row.get("eligible_for_evaluation") is True for row in formal)
    formal_keys = ["parent_ckpt_sha256", "paired_plan_sha256", "iteration", "steps", "batch", "seed"]
    formal_matched = bool(formal_matched and all(formal[0].get(key) is not None
        and all(row.get(key) == formal[0][key] for row in formal) for key in formal_keys))
    gate = data("formal_gate")
    gate_linked = bool(formal_matched and gate.get("schema") == "thor-rsi-evaluation-gate-v1"
        and gate.get("iteration") == formal[0]["iteration"]
        and gate.get("candidate_ckpt_sha256") == formal[0].get("output_ckpt_sha256")
        and gate.get("parent_ckpt_sha256") == formal[0]["parent_ckpt_sha256"]
        and all(gate.get("inputs", {}).get(role + "_training", {}).get("sha256")
                == evidence["formal_" + role]["latest_valid"]["sha256"] for role in ["candidate", "control"]))
    missing = []
    if not feedback_linked: missing.append("独立运行、判官与挖掘的同源交接")
    if judge.get("calibration", {}).get("calibration_passed") is not True: missing.append("独立判官校准与成败召回率达标")
    if mining.get("generation_routes", 0) == 0: missing.append("足量独立反馈支持的视觉弱条件路由")
    if not predictor: missing.append("Predict外部动作条件在Thor的实际冒烟")
    if not multistate_complete: missing.append("Predict多状态真实未来与复制基准诊断")
    missing.append("Predict任务关键误差、动作排序与真实规划收益验证")
    missing.append("新增弱条件窗口扩量与光照资格")
    if not formal_matched: missing.append("三角色同起点、同计划、同预算正式训练")
    if not gate_linked: missing.append("正式候选/匹配对照完整冻结门禁")
    elif gate.get("eligible_for_promotion") is not True: missing.append("正式候选通过既定统计门禁")
    promoted = promotions["promoted_iterations"] if promotions and promotions["current_production_verified"] else 0
    if promoted < 2: missing.append("审批晋升、新策略再次采集及连续两圈对照增益")
    missing.append("最后一圈追加训练种子复验与人工介入完整登记")
    return {"p0_component_acceptance_passed": data("p0_acceptance").get("p0_component_acceptance_passed", False),
        "feedback_reports_linked": feedback_linked, "matched_budget_artifacts": budget_matched,
        "reason_calibration_passed": judge.get("calibration", {}).get("calibration_passed", False),
        "reason_calibration": judge.get("calibration"), "qualified_generation_routes": mining.get("generation_routes", 0),
        "reason_comparison_calibration": data("reason_comparison").get("calibration"),
        "qualified_successes": mining.get("qualified_successes", 0),
        "predict_interface_verified": bool(predictor), "predict_validated_for_planning": False,
        "predict_multistate_diagnostic_completed": multistate_complete,
        "formal_three_role_training_completed": formal_matched,
        "formal_gate_completed": gate_linked,
        "formal_gate_passed": bool(gate_linked and gate.get("statistical_gate_passed") is True
                                   and gate.get("eligible_for_promotion") is True
                                   and gate.get("decision") == "accept_pending_approval"),
        "formal_gate_decision": gate.get("decision") if gate_linked else None,
        "promoted_iterations": promoted,
        "current_production_verified": bool(promotions and promotions["current_production_verified"]),
        # 连续晋升只是项目验收的一部分；判官、生成、追加种子与人工记录仍须独立验收。
        "complete_rsi": False,
        "recursive_orchestrator_status": recursive_orchestrator_status, "remaining": missing,
        "boundary": "P0真实组件证据与预算汇总；不将短训、预测图、配置文件或服务响应计为RSI收益"}


def audit(directory, args):
    import yaml
    config = yaml.safe_load(args.config.read_text())
    if config.get("schema") != "thor-rsi-pipeline-v1" or config["hardware"]["root"] != str(ROOT) \
            or config["hardware"]["host"] != "thor" or config["promotion"]["real_robot_enabled"] is not False:
        raise ValueError("统一配置的版本、主机或仿真边界不符")
    references = {}
    for key, value in config["workflow"].items():
        if key.endswith(("entrypoint", "config", "admission", "handoff", "eval", "protocol", "contract")):
            path = (ROOT / value).resolve(strict=True)
            if not path.is_relative_to(ROOT) or not path.is_file():
                raise ValueError("统一配置引用越界或缺失")
            references[key] = {"path": str(path), "sha256": sha256(path)}
    evidence = {name: evidence_for(ROOT, specification) for name, specification in config["evidence"].items()}
    anchor_feedback(ROOT, evidence, config["evidence"])
    promotions = promotion_history(ROOT)
    result = {"schema": config["schema"], "config": str(args.config), "config_sha256": sha256(args.config),
        "references": references, "evidence": evidence,
        "promotion_history": promotions,
        "readiness": summarize_readiness(evidence, config["workflow"]["recursive_orchestrator_status"],
                                         promotions=promotions),
        "real_robot_commands_sent": False}
    dump(directory / "pipeline_status.json", result)
    print(json.dumps(result["readiness"], ensure_ascii=False), flush=True)
    return result["readiness"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=pathlib.Path, default=ROOT / "configs/rsi_pipeline.yaml")
    args = parser.parse_args()
    args.config = args.config.resolve(strict=True)
    raise SystemExit(run_stage("pipeline_status", lambda directory: audit(directory, args),
                     {"compute": "Thor CPU证据汇总，无模型运行", "sample_count": None}))
