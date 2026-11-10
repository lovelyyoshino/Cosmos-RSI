#!/usr/bin/env python3
"""等待确切整圈释放模型后比较Reason推理配置，达标才运行独立初态验证。

@author pony
@date 2026-11-09
@version v1.0.0
@last_modified 2026-11-09
@changelog
    - v1.0.0 (2026-11-09): 同源开发比较、条件独立验证、共享阶段锁及逐阶段真实证据。

此入口不修改当前整圈、训练数据、冻结评测或判官权限。开发集上的改善不算独立校准。
"""
from __future__ import annotations

import argparse
import fcntl
import json
import pathlib
import signal
import time

from p0_common import ROOT, ensure_gpu_idle, run_stage
from p0_feedback_chain import execute_stage
from p0_feedback_judge import checked_collection
from p0_lerobot_handoff import dump
from p1_eval_gate import sealed, sha256


def project_file(value):
    path = pathlib.Path(value).resolve(strict=True)
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError("判官接续输入必须是本项目已有文件")
    return path


def bind_inputs(iteration_run, config_path):
    import yaml

    if not iteration_run.is_relative_to(ROOT / "runs/p0"):
        raise ValueError("仅等待本项目明确指定的整圈，不扫描最新任务")
    config = yaml.safe_load(config_path.read_text())
    validation, authority = config["validation"], config["authority"]
    if config.get("schema") != "thor-reason-followup-v1" \
            or config.get("development") != "exact_iteration_feedback_only" \
            or validation.get("run_only_if_development_passes") is not True \
            or validation.get("use_same_parent_as_development") is not True \
            or validation.get("tune_after_validation") is not False \
            or validation.get("workers") != 1 \
            or set(authority) != {"automatic_judge_permission_change", "modify_training_pool", "production_promotion", "real_robot_enabled"} \
            or any(value is not False for value in authority.values()):
        raise ValueError("判官比较不能改变既定验证、串行、权限或真机边界")
    count = validation["per_condition"]
    if type(count) is not int or count < 1 or not validation["conditions"] \
            or len(set(validation["conditions"])) != len(validation["conditions"]):
        raise ValueError("独立验证数量或条件非法")
    for key in ["initial_seed", "policy_seed"]:
        if type(validation[key]) is not int or not 0 <= validation[key] < 2 ** 32 - count:
            raise ValueError("独立验证种子范围非法")

    state = json.loads((iteration_run / "iteration_state.json").read_text())
    feedback_binding = state["completed_stages"]["feedback"]
    feedback_path = project_file(feedback_binding["path"])
    feedback, _ = sealed(feedback_path, "feedback_chain")
    if feedback_binding["stage"] != "feedback_chain" or feedback_binding["sha256"] != sha256(feedback_path) \
            or feedback.get("completed_feedback_chain") is not True:
        raise ValueError("开发反馈没有绑定该整圈的完整封存交接")
    collection = project_file(feedback["collection"])
    if feedback["artifacts_sha256"].get(str(collection)) != sha256(collection):
        raise ValueError("开发采集与反馈链摘要不符")
    source, episodes, _ = checked_collection(collection)
    if not set(validation["conditions"]).issubset({row["condition_id"] for row in episodes}) \
            or set(range(validation["initial_seed"], validation["initial_seed"] + count)) \
                & {row["init_seed"] for row in episodes} \
            or set(range(validation["policy_seed"], validation["policy_seed"] + count)) \
                & {row["policy_seed"] for row in episodes}:
        raise ValueError("验证条件不属于开发范围，或验证种子与开发反馈重叠")
    parent = pathlib.Path(state["inputs"]["parent"]["path"]).resolve(strict=True)
    if not parent.is_relative_to(ROOT) or sha256(parent / "checkpoint.json") != state["inputs"]["parent"]["sha256"] \
            or json.loads((parent / "checkpoint.json").read_text())["state_sha256"] != source["policy_ckpt_sha256"]:
        raise ValueError("独立验证的父策略与开发反馈错绑")
    files = {"config": config_path, "feedback": feedback_path, "collection": collection,
             "parent_metadata": parent / "checkpoint.json",
             "judge_config": project_file(ROOT / config["judge_config"]),
             "rollout_config": project_file(ROOT / config["rollout_config"])}
    if "parent_promotion" in state["inputs"]:
        promotion = state["inputs"]["parent_promotion"]
        files["parent_promotion"] = project_file(promotion["path"])
        if sha256(files["parent_promotion"]) != promotion["sha256"]:
            raise ValueError("父策略晋升证明已改变")
    return {"iteration_run": str(iteration_run), "iteration": state["iteration"], "iteration_inputs": state["inputs"],
            "parent": str(parent),
            "files": {name: {"path": str(path), "sha256": sha256(path)} for name, path in files.items()}}


def wait_for_completion(iteration_run, timeout, expected_inputs):
    deadline = time.monotonic() + timeout
    while not (iteration_run / "probe.json").exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("等待指定整圈超时；没有停止或重复启动原实验")
        time.sleep(min(5, max(.01, deadline - time.monotonic())))
    # 统计拒绝也可进入独立判官研究；工程失败则停下，不能伪造已完成的评测。
    report, _ = sealed(iteration_run / "iteration.json", "rsi_iteration")
    if report.get("production_deployed") is not False or report.get("complete_rsi") is not False \
            or report.get("inputs") != expected_inputs:
        raise ValueError("指定整圈状态不符合本次接续边界")
    return {"path": str(iteration_run / "iteration.json"), "sha256": sha256(iteration_run / "iteration.json")}


def validate_holdout(original, heldout, validation, expected_policy_sha):
    """验证实际条件与种子组合，而非仅相信采集命令参数。"""
    expected = {(condition, validation["initial_seed"] + index, validation["policy_seed"] + index)
                for condition in validation["conditions"] for index in range(validation["per_condition"])}
    actual = {(row["condition_id"], row["init_seed"], row["policy_seed"]) for row in heldout}
    if len(heldout) != len(expected) or actual != expected \
            or len({row["episode_id"] for row in heldout}) != len(heldout) \
            or any(row["policy_ckpt_sha256"] != expected_policy_sha or row["is_eval"] is not False for row in heldout) \
            or {row["init_seed"] for row in original} & {row["init_seed"] for row in heldout} \
            or {row["episode_id"] for row in original} & {row["episode_id"] for row in heldout}:
        raise ValueError("实际独立验证的数量、条件、父策略或种子隔离不符")


def compare(directory, bindings):
    import yaml

    for item in bindings["files"].values():
        if sha256(pathlib.Path(item["path"])) != item["sha256"]:
            raise ValueError("排队期间输入发生变化，须另建明确实验")
    files = {name: pathlib.Path(item["path"]) for name, item in bindings["files"].items()}
    config = yaml.safe_load(files["config"].read_text())
    development = execute_stage(directory, "development_judge", "p0_feedback_judge.py", "feedback_judge", "judge_report.json",
        ["--collection", files["collection"], "--config", files["judge_config"]])
    dev_report, _ = sealed(development, "feedback_judge")
    if dev_report["collection_sha256"] != bindings["files"]["collection"]["sha256"]:
        raise ValueError("Reason开发比较读错采集")
    execute_stage(directory, "development_materials", "p1_reason_diagnostic_materials.py", "reason_diagnostic_materials",
                  "materials_manifest.json", ["--judge", development])
    result = {"schema": "thor-reason-followup-v1", "inputs": bindings, "development_judge": str(development),
              "development_calibration": dev_report["calibration"], "heldout_validation_performed": False,
              "independent_validation_passed": False, "judge_authority_changed": False,
              "training_pool_changed": False, "production_promotion_performed": False, "complete_rsi": False}
    if dev_report["calibration"]["calibration_passed"] is not True:
        result["decision"] = "development_rejected_no_independent_calibration_claim"
        dump(directory / "reason_followup.json", result)
        return result

    validation = config["validation"]
    parent_args = ["--parent-promotion", files["parent_promotion"]] if "parent_promotion" in files else []
    collection = execute_stage(directory, "heldout_collection", "p0_feedback_rollout.py", "feedback_rollout", "feedback_rollout.json",
        ["--checkpoint", bindings["parent"], "--conditions", *validation["conditions"],
         "--init-seed", validation["initial_seed"], "--policy-seed", validation["policy_seed"],
         "--per-condition", validation["per_condition"], "--workers", 1, "--config", files["rollout_config"], *parent_args])
    original_source, original, _ = checked_collection(files["collection"])
    _, heldout, _ = checked_collection(collection)
    validate_holdout(original, heldout, validation, original_source["policy_ckpt_sha256"])
    if sha256(files["judge_config"]) != bindings["files"]["judge_config"]["sha256"]:
        raise ValueError("开发比较后不能更换验证提示")
    judge = execute_stage(directory, "heldout_judge", "p0_feedback_judge.py", "feedback_judge", "judge_report.json",
        ["--collection", collection, "--config", files["judge_config"]])
    report, _ = sealed(judge, "feedback_judge")
    if report["collection_sha256"] != sha256(collection) or report["config"] != dev_report["config"]:
        raise ValueError("验证判官与新采集或固定模型配置错绑")
    execute_stage(directory, "heldout_materials", "p1_reason_diagnostic_materials.py", "reason_diagnostic_materials",
                  "materials_manifest.json", ["--judge", judge])
    passed = report["calibration"]["calibration_passed"] is True
    result.update(heldout_validation_performed=True, heldout_collection=str(collection), heldout_judge=str(judge),
                  heldout_calibration=report["calibration"], independent_validation_passed=passed,
                  decision="independent_validation_passed_pending_integration" if passed else "independent_validation_rejected")
    dump(directory / "reason_followup.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--iteration-run", type=pathlib.Path, required=True)
    parser.add_argument("--config", type=pathlib.Path, default=ROOT / "configs/reason_followup.yaml")
    parser.add_argument("--wait-timeout", type=int, default=86400)
    args = parser.parse_args()
    args.iteration_run = args.iteration_run.resolve(strict=True)
    args.config = project_file(args.config)
    if args.wait_timeout < 1:
        parser.error("等待时长必须为正")
    if ROOT != pathlib.Path("/home/thor/rsi"):
        parser.error("判官模型只能在Thor本项目运行")
    bindings = bind_inputs(args.iteration_run, args.config)
    # 排队时不启动模型或内存监控；独立锁避免重复排队，同一阶段锁避免抢占下一圈。
    with (ROOT / "runs/.reason_followup.lock").open("a") as queue_lock:
        fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(json.dumps({"phase": "waiting_for_exact_iteration", "inputs": bindings}, ensure_ascii=False), flush=True)
        bindings["completed_iteration"] = wait_for_completion(args.iteration_run, args.wait_timeout, bindings["iteration_inputs"])
        with (ROOT / "runs/.rsi_iteration.lock").open("a") as model_lock:
            fcntl.flock(model_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            ensure_gpu_idle()
            return run_stage("p1_reason_followup", lambda directory: compare(directory, bindings),
                             {"compute": "Thor模型阶段串行；等待不计入实验阶段墙钟", "inputs": bindings})


if __name__ == "__main__":
    def interrupted(number, _frame):
        raise RuntimeError(f"Reason接续收到信号{number}，清理本次子阶段")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    raise SystemExit(main())
