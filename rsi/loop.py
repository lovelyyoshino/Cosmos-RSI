#!/usr/bin/env python3
"""串行执行反馈、正式配对训练、冻结评测和门禁，保留可核验的阶段恢复点。

@author pony
@date 2026-11-09
@version v1.1.0
@last_modified 2026-11-09
@changelog
    - v1.1.0 (2026-11-09): 后续圈向实际反馈链传递具体父策略晋升记录。
    - v1.0.0 (2026-11-09): 复用已运行反馈，缺合格池即停止；到具体晋升审批前完成全部实验。

生成阶段当前选择已经独立准入的Cosmos纹理池，不冒称本圈重新调用了生成模型。
Predict保持离线研究分支；未校准的Reason不决定仿真成功源资格。
"""
from __future__ import annotations

import argparse
import fcntl
import json
import pathlib
import signal
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from p0_background_preview_materials import checked_artifact
from p0_closeout import validate_budget
from p0_common import ROOT, ensure_gpu_idle, run_stage
from p0_feedback_chain import execute_stage
from p0_lerobot_handoff import dump
from p1_eval_gate import sha256


def input_bindings(args):
    return {name: {"path": str(value), "sha256": sha256(value / "checkpoint.json" if value.is_dir() else value)}
            for name, value in vars(args).items() if isinstance(value, pathlib.Path)
            and name not in {"feedback_chain", "resume_from"}}


def resume_records(path, bindings):
    if path is None:
        return {}
    state = json.loads((path / "iteration_state.json").read_text())
    if state["inputs"] != bindings:
        raise ValueError("恢复的父权重、预算或输入发生变化，须创建独立实验")
    return dict(state["completed_stages"])


def wait_feedback(path, timeout):
    """等待明确指定的正在运行任务；不扫描最新文件，避免接错其他实验。"""
    started = time.monotonic()
    while not (path.parent / "probe.json").exists():
        if time.monotonic() - started > timeout:
            raise TimeoutError("等待现有反馈超时；保留原任务，不重复启动采集")
        time.sleep(5)
    checked_artifact(path, "feedback_chain")
    report = json.loads(path.read_text())
    if report.get("completed_feedback_chain") is not True:
        raise ValueError("现有反馈没有完整完成")
    return report


def iteration(directory, args):
    import yaml

    # 一台Thor仅允许一个本项目完整编排器；锁不改变其他项目的进程和服务。
    with (ROOT / "runs/.rsi_iteration.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("另一个RSI整圈编排器仍在运行") from error
        budget = yaml.safe_load(args.budget.read_text())
        validate_budget(budget, sha256(ROOT / budget["evaluation"]["definition"]))
        bindings = input_bindings(args)
        # 反馈可能尚未落盘；绑定已存在的父目录，而不是提前哈希一个不存在的文件。
        if args.feedback_chain:
            bindings["feedback_chain"] = {"path": str(args.feedback_chain)}
        bindings["iteration"] = args.iteration
        completed = resume_records(args.resume_from, bindings)

        def progress(phase):
            state = {"phase": phase, "iteration": args.iteration, "inputs": bindings,
                     "completed_stages": completed, "human_approval_required": True, "production_deployed": False}
            # 子阶段执行器会写progress.json；恢复证据单独保存，避免运行中断丢失已完成记录。
            dump(directory / "iteration_state.json", state)
            dump(directory / "progress.json", state)

        def stage(name, script, expected, artifact, arguments):
            progress(name)
            if name in completed:
                previous = completed[name]
                path = pathlib.Path(previous["path"]).resolve(strict=True)
                if sha256(path) != previous["sha256"]:
                    raise ValueError("恢复阶段的真实产物已经改变")
                checked_artifact(path, expected)
            else:
                path = execute_stage(directory, name, script, expected, artifact, arguments)
                completed[name] = {"path": str(path), "sha256": sha256(path), "stage": expected}
            progress(name + "_completed")
            return path

        if args.feedback_chain:
            progress("waiting_for_existing_feedback")
            feedback_path = args.feedback_chain
            feedback = wait_feedback(feedback_path, args.wait_timeout)
            if "feedback" in completed and completed["feedback"]["sha256"] != sha256(feedback_path):
                raise ValueError("恢复时反馈产物已改变")
            completed["feedback"] = {"path": str(feedback_path), "sha256": sha256(feedback_path), "stage": "feedback_chain"}
        else:
            initial_seed = budget["collection"]["initial_seed"] + (args.iteration - 1) * 1000
            policy_seed = budget["collection"]["policy_seed"] + (args.iteration - 1) * 1000
            parent_args = ["--parent-promotion", args.parent_promotion] if args.parent_promotion else []
            feedback_path = stage("feedback", "p0_feedback_chain.py", "feedback_chain", "feedback_chain.json",
                ["--checkpoint", args.parent, "--rollout-config", ROOT / "configs/rollout_budget.yaml",
                 "--judge-config", ROOT / "configs/judge_calibration.yaml", "--mining-config", ROOT / "configs/mining.yaml",
                 "--workers", 1, "--conditions", *budget["collection"]["conditions"],
                 "--init-seed", initial_seed, "--policy-seed", policy_seed,
                 "--per-condition", budget["collection"]["per_condition"], *parent_args])
            feedback = json.loads(feedback_path.read_text())
        ensure_gpu_idle()
        stage("feedback_materials", "p0_feedback_materials.py", "feedback_materials", "materials_manifest.json",
              ["--collection", feedback["collection"], "--judge", feedback["judge"]])
        common = ["--parent", args.parent, "--collection", feedback["collection"], "--mining", feedback["mining"],
                  "--handoff", args.handoff, "--budget", args.budget, "--iteration", args.iteration, "--seed", args.seed]
        if args.parent_promotion:
            common += ["--parent-promotion", args.parent_promotion]
        stage("data_qualification", "p1_policy_update.py", "p1_policy_data", "prepared_data.json",
              [*common, "--role", "candidate", "--prepare-only"])
        reports = {}
        for role in budget["training"]["roles"]:
            reports[role] = stage("train_" + role, "p1_policy_update.py", "p1_policy_train", "train_run.json",
                                  [*common, "--role", role])
        smoke = stage("deployment_smoke", "p0_augmented_deploy_smoke.py", "augmented_deploy_smoke", "deployment_smoke.json",
                      ["--training-report", reports["candidate"]])
        evaluations = {}
        for role in ["candidate", "control"]:
            checkpoint = json.loads(reports[role].read_text())["output_ckpt"]
            evaluations[role] = stage("evaluate_" + role, "p0_evaluate_policy.py", "policy_evaluation", "evaluation.json",
                ["--checkpoint", checkpoint, "--frozen-eval", ROOT / budget["evaluation"]["definition"],
                 "--workers", budget["evaluation"]["workers"]])
            stage("analysis_" + role, "p0_analyze_evaluation.py", "evaluation_materials", "analysis.json",
                  ["--evaluation", evaluations[role].parent])
        gate = stage("evaluation_gate", "p1_eval_gate.py", "p1_evaluation_gate", "iter.json",
            ["--candidate-training", reports["candidate"], "--control-training", reports["control"],
             "--candidate-evaluation", evaluations["candidate"], "--control-evaluation", evaluations["control"],
             "--incumbent-evaluation", args.incumbent_evaluation, "--deployment-smoke", smoke, "--budget", args.budget])
        gate_result = json.loads(gate.read_text())
        result = {"schema": "thor-rsi-iteration-v1", "iteration": args.iteration,
                  "decision": gate_result["decision"], "gate": str(gate), "gate_sha256": sha256(gate),
                  "inputs": bindings, "stages": completed, "production_deployed": False,
                  "complete_rsi": False, "human_approval_required": True,
                  "new_cosmos_generation_performed": False, "lighting_generation_qualified": False,
                  "predict_used_for_planning": False, "reason_owns_simulation_truth": False}
        dump(directory / "iteration.json", result)
        progress("awaiting_promotion_approval" if gate_result["eligible_for_promotion"] else "scientific_gate_rejected")
        return result


if __name__ == "__main__":
    def interrupted(number, _frame):
        raise RuntimeError(f"整圈编排收到信号{number}，清理当前子任务")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ["parent", "handoff", "incumbent-evaluation"]:
        parser.add_argument("--" + name, type=pathlib.Path, required=True)
    parser.add_argument("--budget", type=pathlib.Path, default=ROOT / "configs/iteration_budget.yaml")
    parser.add_argument("--feedback-chain", type=pathlib.Path)
    parser.add_argument("--parent-promotion", type=pathlib.Path)
    parser.add_argument("--resume-from", type=pathlib.Path)
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wait-timeout", type=int, default=10800)
    args = parser.parse_args()
    for name, value in vars(args).items():
        if isinstance(value, pathlib.Path):
            value = value.resolve(strict=name != "feedback_chain")
            if not value.is_relative_to(ROOT):
                parser.error("全部输入必须位于Thor本项目")
            setattr(args, name, value)
    if args.iteration < 1 or args.wait_timeout < 1 or not 0 <= args.seed < 2 ** 32 \
            or (args.iteration > 1 and not args.parent_promotion):
        parser.error("后续圈必须绑定真实晋升记录；圈数、种子和等待时长须合法")
    raise SystemExit(run_stage("rsi_iteration", lambda directory: iteration(directory, args),
        {"iteration": args.iteration, "seed": args.seed, "execution": "Thor模型阶段串行，完整冻结门禁后等待具体晋升审批"}))
