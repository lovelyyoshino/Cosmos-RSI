#!/usr/bin/env python3
"""在确切零更新诊断通过后做保守配对短训及独立开发回合，不触碰冻结门禁或生产槽。

@author pony
@date 2026-11-09
@version v1.1.0
@last_modified 2026-11-09
@changelog
    - v1.1.0 (2026-11-09): 训练前核验真实种子隔离；仅从种子冲突失败中复用已封存训练，不重训或覆盖旧结果。
    - v1.0.1 (2026-11-09): 动作产物封存后另等确切前序退出并释放CUDA，不把文件锁释放当成GPU空闲。
    - v1.0.0 (2026-11-09): 复用正式成功源资格；固定低学习率及开发种子，避免退化时直接扩大评测。
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import json
import os
import pathlib
import signal
import time
from types import SimpleNamespace

from p0_common import ROOT, ensure_gpu_idle, run_stage
from p0_feedback_chain import execute_stage
from p0_feedback_judge import checked_collection
from p0_feedback_rollout import build_cases
from p0_lerobot_handoff import dump
from p1_eval_gate import bound_file, sealed, sha256
from p1_policy_regression_probe import project
from p1_policy_update import configure_optimizer, optimizer_recipe, prepare
from p1_stage_transition import wait_stage_teardown
from p1_workflow_status import logged_run


DEPENDENCIES = ["scripts/p1_conservative_pilot.py", "scripts/p1_policy_update.py",
    "scripts/p0_paired_sampling.py", "scripts/p0_policy_runtime.py", "scripts/p0_feedback_rollout.py",
    "scripts/p0_feedback_judge.py", "scripts/p0_lerobot_handoff.py", "scripts/p0_common.py",
    "scripts/p1_eval_gate.py", "scripts/p1_policy_regression_probe.py", "scripts/p1_stage_transition.py",
    "scripts/p0_feedback_chain.py", "scripts/p1_workflow_status.py"]
CONDITIONS = ["t00-standard", "t00-lighting", "t00-background"]
INITIAL_SEED, POLICY_SEED, PER_CONDITION = 153000, 163000, 3


def development_plan(frozen, cache, max_steps):
    """先用实际冻结注册表查冲突；物理初态隔离仍在真实采集时另验。"""
    conditions = [row for row in frozen["conditions"] if row["condition_id"] in CONDITIONS]
    if {row["condition_id"] for row in conditions} != set(CONDITIONS):
        raise ValueError("开发条件不在原冻结注册表中")
    cases = build_cases(conditions, frozen, cache, init_seed=INITIAL_SEED, policy_seed=POLICY_SEED,
                        count=PER_CONDITION, max_steps=max_steps)
    return {"conditions": CONDITIONS, "initial_seed": INITIAL_SEED, "policy_seed": POLICY_SEED,
            "per_condition": PER_CONDITION, "workers": 1, "cases": cases,
            "seed_isolation_passed": True, "actual_scene_isolation_pending": True}


def training_arguments(state, budget):
    feedback_path = project(state["completed_stages"]["feedback"]["path"])
    feedback, _ = sealed(feedback_path, "feedback_chain")
    if sha256(feedback_path) != state["completed_stages"]["feedback"]["sha256"]:
        raise ValueError("开发短训反馈交接已经改变")
    return SimpleNamespace(parent=project(state["inputs"]["parent"]["path"]),
        handoff=project(state["inputs"]["handoff"]["path"]), collection=project(feedback["collection"]),
        mining=project(feedback["mining"]), budget=project(budget), iteration=1,
        parent_promotion=None, seed=42, role="candidate", prepare_only=True)


def bind_inputs(iteration, budget):
    import yaml

    state = json.loads(project(iteration / "iteration_state.json").read_text())
    if state["iteration"] != 1:
        raise ValueError("本开发试验仅诊断当前首圈退化，不冒充后续递归圈")
    args = training_arguments(state, budget)
    if sha256(args.parent / "checkpoint.json") != state["inputs"]["parent"]["sha256"] \
            or sha256(args.handoff) != state["inputs"]["handoff"]["sha256"]:
        raise ValueError("开发试验的父权重元数据或派生交接偏离原整圈")
    parent, config, cache, plan, proof, _, _, _, episodes = prepare(args)
    expected = {"lr_vlm": 2e-6, "lr_head": 1e-5, "weight_decay": .01, "warmup_steps": 20,
        "optimizer_state_restored": True, "scheduler": "linear_warmup_cosine_min0.1"}
    if optimizer_recipe(config) != expected or config["training"]["steps_per_role"] != 250 \
            or config["training"].get("development_only") is not True \
            or config["training"]["candidate_quotas"] != {"demo": 1600, "self_distill": 200, "generated": 200}:
        raise ValueError("保守试验的预声明优化器、步数或配额发生变化")
    manifest = cache.manifest
    development = development_plan(json.loads((ROOT / "eval/frozen_eval_set.json").read_text()), manifest,
                                   yaml.safe_load((ROOT / "configs/rollout_budget.yaml").read_text())["max_steps"])
    instructions = {row["episode_id"]: row["instruction"] for row in manifest["episodes"]}
    instructions.update({r["episode_id"]: r["instruction"] for r in episodes})
    files = {"budget": args.budget, "parent_metadata": args.parent / "checkpoint.json",
        "feedback": project(state["completed_stages"]["feedback"]["path"]), "collection": args.collection,
        "handoff": args.handoff, "mining": args.mining, "rollout_config": ROOT / "configs/rollout_budget.yaml",
        "frozen_eval": ROOT / "eval/frozen_eval_set.json"}
    return {"schema": "thor-conservative-pilot-v1", "iteration_run": str(iteration), "iteration_inputs": state["inputs"],
        "files": {key: {"path": str(path), "sha256": sha256(path)} for key, path in files.items()},
        "dependencies_sha256": {name: sha256(ROOT / name) for name in DEPENDENCIES},
        "parent_ckpt_sha256": parent["state_sha256"], "paired_plan_sha256": plan["sha256"],
        "source_qualification": proof, "optimizer": expected,
        "sampling_by_task": {role: dict(Counter(instructions[r["source_episode_id"]] for r in plan[role]))
                             for role in ["candidate", "control"]},
        "development": development,
        "authority": {"development_only": True, "formal_gate_executed": False,
                      "production_promotion": False, "real_robot_enabled": False}}


def verify_inputs(bindings):
    if any(sha256(project(row["path"])) != row["sha256"] for row in bindings["files"].values()) \
            or bindings["dependencies_sha256"] != {name: sha256(ROOT / name) for name in DEPENDENCIES}:
        raise ValueError("排队期间保守试验的输入或实现发生变化")


def parent_optimizer_audit(directory, iteration, budget):
    """用实际父权重检查优化器矩，不在备份机加载模型或占用GPU。"""
    import torch
    import yaml

    torch.set_num_threads(1)
    state = json.loads(project(iteration / "iteration_state.json").read_text())
    checkpoint = project(state["inputs"]["parent"]["path"])
    metadata = json.loads((checkpoint / "checkpoint.json").read_text())
    if sha256(checkpoint / "checkpoint.json") != state["inputs"]["parent"]["sha256"] \
            or sha256(checkpoint / "state.pt") != metadata["state_sha256"]:
        raise ValueError("实际父权重或元数据已改变")
    saved = torch.load(checkpoint / "state.pt", map_location="cpu", weights_only=True, mmap=True)
    tensors = {name: torch.nn.Parameter(value) for name, value in saved["trainable_state"].items()}
    config = yaml.safe_load(budget.read_text())
    recipe = optimizer_recipe(config)
    optimizer, scheduler, names = configure_optimizer(tensors, saved, recipe, config["training"]["steps_per_role"])
    steps = [float(value["step"]) for value in optimizer.state.values()]
    if not steps or min(steps) != metadata["step"] or max(steps) != metadata["step"]:
        raise ValueError("父优化器实际步数与父权重不同")
    result = {"schema": "thor-parent-optimizer-audit-v1", "parent_checkpoint": str(checkpoint),
        "parent_ckpt_sha256": metadata["state_sha256"], "budget_sha256": sha256(budget),
        "restored_optimizer_steps_min": min(steps), "restored_optimizer_steps_max": max(steps),
        "parameter_tensors": len(tensors), "optimizer_group_parameter_tensors": list(map(len, names)),
        "initial_effective_learning_rates": [group["lr"] for group in optimizer.param_groups],
        "scheduler_peak_lrs": scheduler.base_lrs, "recipe": recipe,
        "optimization_steps_executed": 0, "gpu_used": False}
    dump(directory / "optimizer_preflight.json", result)
    return result


def bind_probe(execution, bindings):
    path = project(execution / "launch.json")
    launch = json.loads(path.read_text())
    command = launch["command"]
    if "actions" not in command or bindings["iteration_run"] not in command \
            or not any(value.endswith("scripts/p1_policy_regression_probe.py") for value in command):
        raise ValueError("保守试验只能等待同一整圈的确切动作复现任务")
    return {"launch": str(path), "launch_sha256": sha256(path), "pid": launch["pid"],
            "command": command, "log": str(project(launch["log"]))}


def wait_probe(binding, timeout):
    deadline = time.monotonic() + timeout
    while True:
        if sha256(project(binding["launch"])) != binding["launch_sha256"]:
            raise ValueError("等待期间动作诊断启动记录发生变化")
        paths = [project(line.split("：", 1)[1]) for line in pathlib.Path(binding["log"]).read_text().splitlines()
                 if line.startswith("记录目录：")]
        if len(paths) > 1:
            raise ValueError("确切动作诊断包含多个阶段，不按最新目录猜测")
        if paths and (paths[0] / "probe.json").exists():
            path = paths[0] / "action_probe.json"
            report, _ = sealed(path, "p1_policy_action_probe")
            if set(report["zero_update_reproduction"]) != {"incumbent", "candidate", "control", "analytic_ablation"} \
                    or any(row["passed"] is not True for row in report["zero_update_reproduction"].values()):
                raise ValueError("原权重动作没有复现，禁止继续用新优化器训练掩盖接口问题")
            return path, report
        try:
            actual = pathlib.Path(f"/proc/{binding['pid']}/cmdline").read_bytes().rstrip(b"\x00").split(b"\x00")
        except FileNotFoundError as error:
            raise RuntimeError("确切动作诊断已退出且没有封存产物，停止保守试验") from error
        if actual != [value.encode() for value in binding["command"]]:
            raise RuntimeError("等待的动作诊断进程身份改变")
        if time.monotonic() >= deadline:
            raise TimeoutError("等待动作诊断超时，保留既有队列")
        time.sleep(min(5., max(.01, deadline - time.monotonic())))


def validate_development(rows, policy_sha):
    expected = {(condition, INITIAL_SEED + i, POLICY_SEED + i)
                for condition in CONDITIONS for i in range(PER_CONDITION)}
    actual = {(row["condition_id"], row["init_seed"], row["policy_seed"]) for row in rows}
    if len(rows) != len(expected) or actual != expected \
            or any(row["is_eval"] is not False or row["eligible_for_training"] is not False
                   or row["policy_ckpt_sha256"] != policy_sha for row in rows):
        raise ValueError("实际开发回合的数量、固定种子、策略或隔离边界不符")
    return {condition: {"success": sum(row["outcome"]["success"] for row in rows if row["condition_id"] == condition),
                        "n": PER_CONDITION} for condition in CONDITIONS}


def validate_reused_training(report, bindings, role):
    """开发种子可以修正；已有训练的监督、预算、优化器和父权重必须完全相同。"""
    expected = {"role": role, "steps": 250, "batch": 8, "seed": 42,
        "parent_ckpt_sha256": bindings["parent_ckpt_sha256"], "paired_plan_sha256": bindings["paired_plan_sha256"],
        "data_qualification": bindings["source_qualification"], "development_only": True,
        "training_performed": True, "eligible_for_evaluation": False, "eligible_for_promotion": False}
    recipe = {"steps": 250, "batch": 8, "seed": 42, **bindings["optimizer"]}
    if any(report.get(key) != value for key, value in expected.items()) \
            or any(report.get("recipe", {}).get(key) != value for key, value in recipe.items()):
        raise ValueError("续用训练与当前父权重、监督、预算、配方或开发权限不一致")


def reusable_training(run, bindings):
    run = project(run)
    if not run.is_relative_to(ROOT / "runs/p0"):
        raise ValueError("续用训练只接受本项目确切失败目录")
    probe_path = run / "probe.json"
    probe = json.loads(probe_path.read_text())
    failure = run / "logs/development_incumbent.log"
    if probe.get("stage") != "p1_conservative_pilot" or probe.get("status") != "failed" \
            or "development_incumbent" not in probe.get("error", "") \
            or "运行采集与D0/冻结评测的种子相交" not in failure.read_text():
        raise ValueError("只接续开发种子冲突的已结束试验，不复用其他失败或未完成任务")
    # 不放宽原隔离检查；只允许本编排脚本变化，其余训练与采集实现按旧快照核对。
    for name in DEPENDENCIES + ["configs/iteration_budget_conservative_pilot.yaml", "configs/rollout_budget.yaml",
                                "eval/frozen_eval_set.json"]:
        if name == "scripts/p1_conservative_pilot.py":
            continue
        expected = probe["source_files_sha256"].get(name)
        if expected != sha256(ROOT / name) or expected != sha256(run / "source_snapshot" / name):
            raise ValueError(f"续用训练的来源实现或配置已变化：{name}")
    training = {}
    for role in ["candidate", "control", "analytic_ablation"]:
        child = logged_run(ROOT, run / f"logs/train_{role}.log")
        if child is None:
            raise ValueError("失败目录没有唯一封存训练来源")
        path = child / "train_run.json"
        report, sealed_probe = sealed(path, "p1_policy_train")
        validate_reused_training(report, bindings, role)
        checkpoint = project(report["output_ckpt"])
        if checkpoint != child / "checkpoint" or not 0 <= report["reload_max_abs_diff"] <= 1e-5:
            raise ValueError("续用训练的实际检查点越界或没有通过重载")
        metadata = json.loads(bound_file(child, sealed_probe, "checkpoint/checkpoint.json").read_text())
        weights = bound_file(child, sealed_probe, "checkpoint/state.pt")
        if sha256(weights) != report["output_ckpt_sha256"] or metadata["state_sha256"] != report["output_ckpt_sha256"] \
                or metadata.get("formal_policy_update") is not False \
                or any(metadata.get(key) != report.get(key) for key in ["role", "recipe", "development_only",
                    "eligible_for_promotion", "parent_ckpt_sha256", "paired_plan_sha256", "data_qualification"]):
            raise ValueError("续用训练的实际权重、元数据或监督摘要不一致")
        training[role] = {"path": str(path), "sha256": sha256(path)}
    return {"failed_run": str(run), "failed_probe_sha256": sha256(probe_path), "training": training,
            "training_steps_repeated": 0, "reason": "修正与冻结注册表冲突的开发种子，保留原始失败和三份训练"}


def pilot(directory, bindings, previous, reuse=None):
    verify_inputs(bindings)
    state = {"inputs": bindings["iteration_inputs"], "completed_stages": {"feedback": bindings["files"]["feedback"]}}
    args = training_arguments(state, bindings["files"]["budget"]["path"])
    common = ["--parent", args.parent, "--collection", args.collection, "--mining", args.mining,
              "--handoff", args.handoff, "--budget", args.budget, "--iteration", 1, "--seed", 42]
    training, checkpoints = {}, {"incumbent": args.parent}
    if reuse is not None:
        if reusable_training(project(reuse["failed_run"]), bindings) != reuse:
            raise ValueError("接续预检后旧训练证据发生变化")
        dump(directory / "reused_training.json", reuse)
    for role in ["candidate", "control", "analytic_ablation"]:
        if reuse is None:
            path = execute_stage(directory, "train_" + role, "p1_policy_update.py", "p1_policy_train", "train_run.json",
                                 [*common, "--role", role])
        else:
            path = project(reuse["training"][role]["path"])
        report, _ = sealed(path, "p1_policy_train")
        if report["paired_plan_sha256"] != bindings["paired_plan_sha256"] \
                or report.get("development_only") is not True or report["eligible_for_evaluation"] is not False:
            raise ValueError("实际短训采样计划与等待前预声明不符")
        training[role] = {"path": str(path), "sha256": sha256(path)}
        checkpoints[role] = project(report["output_ckpt"])
    collections, initial_scenes = {}, None
    for role in ["incumbent", "candidate", "control"]:
        path = execute_stage(directory, "development_" + role, "p0_feedback_rollout.py", "feedback_rollout", "feedback_rollout.json",
            ["--checkpoint", checkpoints[role], "--config", bindings["files"]["rollout_config"]["path"],
             "--conditions", *CONDITIONS, "--init-seed", INITIAL_SEED, "--policy-seed", POLICY_SEED,
             "--per-condition", PER_CONDITION, "--workers", 1])
        report, rows, _ = checked_collection(path)
        if report["eval_overlap_check"]["eval_set_sha256"] != bindings["files"]["frozen_eval"]["sha256"]:
            raise ValueError("开发采集偏离既有冻结隔离版本")
        counts = validate_development(rows, json.loads((checkpoints[role] / "checkpoint.json").read_text())["state_sha256"])
        scenes = {(r["condition_id"], r["init_seed"], r["policy_seed"]): r["initial_scene_sha256"] for r in rows}
        if initial_scenes is not None and scenes != initial_scenes:
            raise ValueError("开发配对策略没有使用同一初始物理场景")
        initial_scenes = scenes
        collections[role] = {"path": str(path), "sha256": sha256(path), "counts": counts}
    result = {"schema": bindings["schema"], "inputs": bindings, "after_action_probe": previous,
        "training": training, "development": collections, "authority": bindings["authority"], "training_reuse": reuse,
        "complete_rsi": False, "causal_attribution_established": False,
        "boundary": "低学习率、优化器状态及重放配额同时变化的开发试验；小样本不证明增益，不自动扩大或晋升"}
    dump(directory / "pilot.json", result)
    return {"development_complete": True, "production_deployed": False, "complete_rsi": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--iteration-run", type=pathlib.Path, required=True)
    parser.add_argument("--budget", type=pathlib.Path, default=ROOT / "configs/iteration_budget_conservative_pilot.yaml")
    parser.add_argument("--after-action-probe-exec", type=pathlib.Path)
    parser.add_argument("--resume-training-from", type=pathlib.Path,
                        help="只接续确切种子冲突失败目录的已封存三份训练；仍做完整开发运行")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--audit-parent-optimizer", action="store_true")
    parser.add_argument("--wait-timeout", type=int, default=86400)
    args = parser.parse_args()
    if args.prepare_only or args.audit_parent_optimizer:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            parser.error("CPU预检必须显式禁用CUDA")
        if args.prepare_only and args.audit_parent_optimizer:
            parser.error("CPU数据预检与实际优化器审计各自记录，不合并两个阶段")
        if args.audit_parent_optimizer:
            if args.resume_training_from:
                parser.error("优化器审计不接续开发任务")
            return run_stage("p1_parent_optimizer_audit", lambda directory:
                parent_optimizer_audit(directory, project(args.iteration_run), project(args.budget)),
                {"compute": "Thor CPU，CUDA禁用、单线程；实际父AdamW状态，零优化步"})
        return run_stage("p1_conservative_pilot_input", lambda directory:
                         prepare_input(directory, project(args.iteration_run), project(args.budget), args.resume_training_from),
                         {"compute": "Thor CPU，CUDA禁用；仅资格和固定开发配方预检，无训练推理"})
    bindings = bind_inputs(project(args.iteration_run), project(args.budget))
    reuse = reusable_training(project(args.resume_training_from), bindings) if args.resume_training_from else None
    if args.after_action_probe_exec is None or args.wait_timeout < 1:
        parser.error("必须绑定确切动作诊断及正等待时长")
    with (ROOT / "runs/.conservative_pilot.lock").open("a") as queue:
        fcntl.flock(queue, fcntl.LOCK_EX | fcntl.LOCK_NB)
        probe = bind_probe(project(args.after_action_probe_exec), bindings)
        print(json.dumps({"phase": "waiting_for_exact_action_probe", "inputs": bindings, "probe": probe}, ensure_ascii=False), flush=True)
        path, report = wait_probe(probe, args.wait_timeout)
        if report["inputs"]["iteration_run"] != bindings["iteration_run"] \
                or report["inputs"]["files"]["checkpoint_incumbent"] != bindings["files"]["parent_metadata"]:
            raise ValueError("零更新诊断来源与保守试验不一致")
        teardown = wait_stage_teardown(probe)
        with (ROOT / "runs/.rsi_iteration.lock").open("a") as lock:
            deadline = time.monotonic() + 30
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise RuntimeError("动作诊断封存后共享锁仍未释放，不抢占其他任务")
                    time.sleep(1)
            ensure_gpu_idle()
            return run_stage("p1_conservative_pilot", lambda directory: pilot(directory, bindings,
                {"path": str(path), "sha256": sha256(path)}, reuse), {"compute": "Thor单卡串行保守开发试验",
                "predecessor_teardown": teardown,
                "precision": "VLM bf16、LoRA和动作头fp32", "batch": 8, "steps_per_role": 250,
                "training_seed": 42, "development_episodes_per_policy": 9})


def prepare_input(directory, iteration, budget, resume):
    bindings = bind_inputs(iteration, budget)
    if resume:
        dump(directory / "reused_training_preflight.json", reusable_training(project(resume), bindings))
    return dump_input(directory, bindings)


def dump_input(directory, bindings):
    dump(directory / "pilot_input.json", bindings)
    return {"prepared": True, "training_performed": False, "production_deployed": False}


if __name__ == "__main__":
    def interrupted(number, _frame):
        raise RuntimeError(f"保守试验收到信号{number}；清理本次子任务，不修改既有结果")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    raise SystemExit(main())
