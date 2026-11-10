#!/usr/bin/env python3
"""核验正式更新的真实参数、原生数据对齐与零更新动作复现，不据离线误差宣称策略收益。

@author pony
@date 2026-11-09
@version v1.1.0
@last_modified 2026-11-09
@changelog
    - v1.1.0 (2026-11-09): 四份权重改用独立子进程，逐份核验退出及GPU空闲后汇总相同诊断。
    - v1.0.2 (2026-11-09): 封存后等待确切生成进程与CUDA上下文退出，避免锁释放早于解释器退出。
    - v1.0.1 (2026-11-09): 等待前绑定冻结定义及判官读取依赖，拒绝漂移的隔离版本。
    - v1.0.0 (2026-11-09): CPU参数/读取审计；确切生成结束后串行运行固定验证动作比较。
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gc
import json
import math
import pathlib
import signal
import subprocess
import sys
import time

from p0_common import ROOT, ensure_gpu_idle, run_stage, stop_process_group
from p0_lerobot_handoff import dump
from p0_policy_runtime import DemoCache, normalize
from p1_eval_gate import checked_update, sealed, sha256
from p1_stage_transition import wait_stage_teardown


ROLES = ["incumbent", "candidate", "control", "analytic_ablation"]
DEPENDENCIES = ["scripts/p1_policy_regression_probe.py", "scripts/p0_policy_runtime.py",
                "scripts/p0_lerobot_handoff.py", "scripts/p0_augmented_budget.py",
                "scripts/p0_augmented_smoke.py", "scripts/p0_feedback_judge.py",
                "scripts/p1_eval_gate.py", "scripts/p0_common.py", "scripts/p1_stage_transition.py"]
CONFIG = {"schema": "thor-policy-regression-probe-v1",
    "selection": {"split": "demo_validation", "expected_tasks": 10, "episodes_per_task": 2,
                  "positions": ["initial", "middle", "last_valid"]},
    "inference": {"batch": 1, "seeds": [44, 45], "action_horizon": 8, "saved_probe_max_abs_error": 1e-5},
    "authority": {"optimization_steps": 0, "validation_used_for_training": False,
                  "formal_gate_executed": False, "production_promotion": False, "real_robot_enabled": False}}


def project(path):
    path = pathlib.Path(path).resolve(strict=True)
    if not path.is_relative_to(ROOT):
        raise ValueError("诊断输入必须位于Thor本项目")
    return path


def validation_locations(manifest, selection):
    """按任务及ID预先选样；验证数据不进入任何优化或成功源池。"""
    if selection != CONFIG["selection"] or manifest["horizon"] != 8:
        raise ValueError("验证样本配方或动作块版本不符")
    episodes = manifest["episodes"]
    ids = [row["episode_id"] for row in episodes]
    if len(ids) != len(set(ids)):
        raise ValueError("示范身份重复，无法隔离训练与验证")
    groups = defaultdict(list)
    for row in episodes:
        if row["split"] == "demo_validation":
            if type(row["length"]) is not int or row["length"] < 10:
                raise ValueError("验证回合过短，不能覆盖三个独立动作起点")
            groups[row["instruction"]].append(row)
    if len(groups) != selection["expected_tasks"]:
        raise ValueError("独立验证任务覆盖不完整")
    result = []
    for instruction, rows in sorted(groups.items()):
        count = selection["episodes_per_task"]
        if len(rows) < count:
            raise ValueError("某任务缺少预声明数量的独立验证示范")
        for episode in sorted(rows, key=lambda row: row["episode_id"])[:count]:
            last = episode["length"] - 8
            for position, first in [("initial", 0), ("middle", last // 2), ("last_valid", last)]:
                result.append({"episode_id": episode["episode_id"], "instruction": instruction,
                               "frame": first, "horizon": 8, "split": "demo_validation", "position": position})
    return result


def cached_example(cache, location, split):
    import numpy as np
    from PIL import Image

    matches = [r for r in cache.manifest["episodes"] if r["episode_id"] == location["episode_id"]]
    if len(matches) != 1 or matches[0]["split"] != split or split not in {"demo_train", "demo_validation"}:
        raise ValueError("动作检查的源回合或分区错绑")
    row = matches[0]
    first, horizon = location["frame"], location["horizon"]
    if type(first) is not int or horizon != 8 or not 0 <= first <= row["length"] - horizon:
        raise ValueError("动作检查越过源回合")
    start = row["offset"] + first
    return {"image": [Image.fromarray(np.array(cache.images[start, view])) for view in range(2)],
            "lang": row["instruction"],
            "state": normalize(cache.states[start:start + 1], cache.manifest["normalizer"]["state"]),
            "action": normalize(cache.actions[start:start + horizon], cache.manifest["normalizer"]["action"])}


def assert_example_equal(actual, expected, generated=False):
    import numpy as np

    if actual["lang"] != expected["lang"] or any(not np.array_equal(actual[k], expected[k]) for k in ["state", "action"]) \
            or len(actual["image"]) != 2 or len(expected["image"]) != 2 \
            or any(not np.array_equal(np.asarray(actual["image"][view]), np.asarray(expected["image"][view]))
                   for view in ([1] if generated else [0, 1])):
        raise ValueError("实际原生训练样本的语言、视角、归一化状态或动作存在漂移")


def bind_inputs(iteration, config_path):
    import yaml

    config_path = project(config_path)
    if yaml.safe_load(config_path.read_text()) != CONFIG:
        raise ValueError("零更新诊断配方发生变化，应另建实验版本")
    state = json.loads(project(iteration / "iteration_state.json").read_text())
    files = {"config": config_path}
    checkpoints = {"incumbent": project(state["inputs"]["parent"]["path"])}
    reports = {}
    for role in ROLES[1:]:
        binding = state["completed_stages"]["train_" + role]
        path = project(binding["path"])
        if sha256(path) != binding["sha256"]:
            raise ValueError("整圈正式训练交接发生变化")
        report, _ = checked_update(path, role)
        files["training_" + role] = path
        checkpoints[role] = project(report["output_ckpt"])
        reports[role] = report
    parent = json.loads((checkpoints["incumbent"] / "checkpoint.json").read_text())
    for role, checkpoint in checkpoints.items():
        metadata = checkpoint / "checkpoint.json"
        record = json.loads(metadata.read_text())
        if sha256(checkpoint / "state.pt") != record["state_sha256"] \
                or any(record[key] != parent[key] for key in ["cache_manifest_sha256", "base_manifest_sha256",
                                                            "normalizer", "model_config", "precision"]):
            raise ValueError("待比较策略的权重、D0、基座或数值条件不一致")
        if role != "incumbent" and record["parent_ckpt_sha256"] != parent["state_sha256"]:
            raise ValueError("正式三角色不属于同一父权重")
        files["checkpoint_" + role] = metadata
    cache_manifest = project(parent["cache_manifest"])
    if sha256(cache_manifest) != parent["cache_manifest_sha256"]:
        raise ValueError("独立示范验证清单已改变")
    files["cache_manifest"] = cache_manifest
    files["frozen_eval"] = project(ROOT / "eval/frozen_eval_set.json")
    if sha256(files["frozen_eval"]) != parent["eval_overlap_check"]["eval_set_sha256"]:
        raise ValueError("诊断冻结定义与父权重隔离证明不一致")
    candidate = reports["candidate"]
    files["collection"] = project(candidate["data_qualification"]["collection"])
    files["handoff"] = project(candidate["data_qualification"]["handoff"])
    for key in ["collection", "handoff"]:
        if sha256(files[key]) != candidate["data_qualification"][key + "_sha256"]:
            raise ValueError("训练数据来源与正式更新交接不一致")
    return {"schema": CONFIG["schema"], "iteration_run": str(iteration), "iteration": state["iteration"],
            "checkpoints": {role: str(path) for role, path in checkpoints.items()},
            "files": {key: {"path": str(path), "sha256": sha256(path)} for key, path in files.items()},
            "dependencies_sha256": {name: sha256(ROOT / name) for name in DEPENDENCIES},
            "locations": validation_locations(json.loads(cache_manifest.read_text()), CONFIG["selection"]),
            "authority": CONFIG["authority"]}


def verify_bindings(bindings):
    for binding in bindings["files"].values():
        if sha256(project(binding["path"])) != binding["sha256"]:
            raise ValueError("排队期间诊断输入发生变化")
    if bindings["dependencies_sha256"] != {name: sha256(ROOT / name) for name in DEPENDENCIES}:
        raise ValueError("排队期间诊断实现发生变化")


def tensor_deltas(parent, candidate):
    """按真实参数分组汇总更新量；参数距离不能解释或证明闭环收益。"""
    import torch

    if not parent or set(parent) != set(candidate):
        raise ValueError("父子权重参数集合不一致")
    totals = defaultdict(lambda: {"parameters": 0, "elements": 0, "parent_sq": 0., "delta_sq": 0., "max_abs_delta": 0.})
    for name, before in parent.items():
        after = candidate[name]
        if before.shape != after.shape or before.dtype != after.dtype or not before.is_floating_point() \
                or not torch.isfinite(before).all() or not torch.isfinite(after).all():
            raise ValueError("父子权重形状、精度或有限性不一致")
        group = "vlm_lora" if name.startswith("qwen_vl_interface.") else "action_head" if name.startswith("action_model.") else None
        if group is None:
            raise ValueError("权重中出现未声明的可训练模块")
        row = totals[group]
        row["parameters"] += 1
        # 分块以免对大权重构造整份fp64副本，CPU检查不占用GPU。
        for left, right in zip(before.reshape(-1).split(262144), after.reshape(-1).split(262144)):
            x, y = left.double(), right.double()
            delta = y - x
            row["elements"] += x.numel()
            row["parent_sq"] += float(torch.sum(x * x))
            row["delta_sq"] += float(torch.sum(delta * delta))
            row["max_abs_delta"] = max(row["max_abs_delta"], float(delta.abs().max()) if delta.numel() else 0.)
    return {name: {"parameter_tensors": row["parameters"], "elements": row["elements"],
                   "parent_l2": math.sqrt(row["parent_sq"]), "delta_l2": math.sqrt(row["delta_sq"]),
                   "relative_delta_l2": math.sqrt(row["delta_sq"] / row["parent_sq"]) if row["parent_sq"] else None,
                   "max_abs_delta": row["max_abs_delta"]} for name, row in totals.items()}


def reader_audit(bindings, cache):
    import numpy as np
    from PIL import Image
    from p0_augmented_budget import checked_handoff
    from p0_feedback_judge import checked_collection
    from p0_lerobot_handoff import StrictNativeWindows

    parent = json.loads(pathlib.Path(bindings["files"]["checkpoint_incumbent"]["path"]).read_text())
    handoff = checked_handoff(pathlib.Path(bindings["files"]["handoff"]["path"]), parent)
    collection, episodes, probe = checked_collection(pathlib.Path(bindings["files"]["collection"]["path"]))
    runtime = StrictNativeWindows(pathlib.Path(collection["dataset"]), cache.manifest["normalizer"])
    lookup = {r["source_episode_id"]: i for i, r in enumerate(runtime.provenance["episodes"])}
    training = json.loads(pathlib.Path(bindings["files"]["training_candidate"]["path"]).read_text())
    plan_path = pathlib.Path(bindings["files"]["training_candidate"]["path"]).parent / "paired_plan.json"
    descriptions = json.loads(plan_path.read_text())["descriptions"]
    selected_ids = {r["source_episode_id"] for r in descriptions if r["kind"] == "self_distill"}
    selected = [row for row in episodes if row["episode_id"] in selected_ids]
    if len(selected) != len(selected_ids) or len(selected) != training["data_qualification"]["self_success_episodes"]:
        raise ValueError("自身成功池与真实回合不完整对应")
    checked = []
    for row in selected:
        trajectory = project(row["trajectory"])
        relative = str(trajectory.relative_to(pathlib.Path(bindings["files"]["collection"]["path"]).parent))
        if row["outcome"] != {"success": True, "source": "env"} or sha256(trajectory) != row["trajectory_sha256"] \
                or probe["artifacts_sha256"].get(relative) != row["trajectory_sha256"]:
            raise ValueError("自身源的真实动作或环境成功证明失效")
        with np.load(trajectory, allow_pickle=False) as source:
            last = len(source["action"]) - 8
            for frame in sorted({0, last // 2, last}):
                expected = {"image": [Image.fromarray(source[k][frame]) for k in ["agentview", "wrist"]],
                    "lang": row["instruction"], "state": normalize(source["state"][frame:frame + 1], cache.manifest["normalizer"]["state"]),
                    "action": normalize(source["action"][frame:frame + 8], cache.manifest["normalizer"]["action"])}
                assert_example_equal(runtime.example_at(lookup[row["episode_id"]], frame), expected)
                checked.append({"kind": "self_distill", "episode_id": row["episode_id"], "frame": frame,
                                "instruction": row["instruction"], "views_checked": [0, 1]})
    for variant in ["cosmos", "analytic"]:
        native = StrictNativeWindows(pathlib.Path(handoff["datasets"][variant]), cache.manifest["normalizer"])
        for index, row in enumerate(native.provenance["episodes"]):
            first, stop = [row["source_window"][key] for key in ["start_frame", "end_frame"]]
            for frame in sorted({0, (stop - first - 8) // 2, stop - first - 8}):
                expected = cached_example(cache, {"episode_id": row["source_episode_id"], "frame": first + frame, "horizon": 8}, "demo_train")
                assert_example_equal(native.example_at(index, frame), expected, generated=True)
                checked.append({"kind": variant, "episode_id": row["source_episode_id"], "frame": first + frame,
                                "instruction": expected["lang"], "views_checked": [1]})
    return {"checked_blocks": checked, "self_success_instructions": dict(Counter(r["instruction"] for r in selected)),
            "source_reading_passed": True, "scope": "每个真实自身成功及两种派生源的首中末合法动作块；非全部动作块"}


def cpu_audit(directory, bindings):
    import torch

    verify_bindings(bindings)
    torch.set_num_threads(1)
    weights = lambda role: torch.load(pathlib.Path(bindings["checkpoints"][role]) / "state.pt", map_location="cpu", weights_only=True, mmap=True)["trainable_state"]
    parent = weights("incumbent")
    deltas = {}
    for role in ROLES[1:]:
        current = weights(role)
        deltas[role] = tensor_deltas(parent, current)
        del current
    del parent
    gc.collect()
    cache = DemoCache(pathlib.Path(bindings["files"]["cache_manifest"]["path"]).parent)
    result = {"schema": CONFIG["schema"], "inputs": bindings, "weight_deltas": deltas,
              "reader_audit": reader_audit(bindings, cache), "authority": CONFIG["authority"],
              "causal_attribution_established": False, "compute": "Thor CPU，CUDA禁用、单线程；无模型推理和训练"}
    dump(directory / "weight_reader_audit.json", result)
    return {"reader_audit_passed": True, "checked_roles": ROLES[1:], "optimization_steps": 0}


def action_metrics(prediction, target, incumbent):
    import numpy as np

    arrays = [np.asarray(value, dtype=np.float64) for value in [prediction, target, incumbent]]
    if any(value.shape != (8, 7) or not np.isfinite(value).all() for value in arrays):
        raise ValueError("动作比较只接受有限的8×7归一化动作块")
    pred, truth, old = arrays
    return {"action_mse": float(np.mean((pred - truth) ** 2)),
            "arm_mse": float(np.mean((pred[:, :6] - truth[:, :6]) ** 2)),
            "gripper_mse": float(np.mean((pred[:, 6] - truth[:, 6]) ** 2)),
            "drift_vs_incumbent_rms": float(np.sqrt(np.mean((pred - old) ** 2)))}


def ensure_no_other_gpu_processes():
    """同一进程加载下一权重时允许自身CUDA上下文，但不允许其他模型进程并行。"""
    import os
    from p0_common import command

    result = command(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"])
    if result.get("returncode") != 0:
        raise RuntimeError("模型间切换无法核验GPU进程")
    pids = [int(line.strip()) for line in result["stdout"].splitlines() if line.strip()]
    if any(pid != os.getpid() for pid in pids):
        raise RuntimeError("零更新诊断发现其他GPU进程，停止加载下一权重")


def role_action_probe(bindings, role, bindings_path, output_path):
    """模型只驻留于本角色子进程；退出前的张量统计不能替代父进程的退出检查。"""
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from p0_policy_runtime import build_model, restore_checkpoint

    ensure_gpu_idle()
    verify_bindings(bindings)
    if role not in ROLES:
        raise ValueError("独立诊断角色未声明")
    cache = DemoCache(pathlib.Path(bindings["files"]["cache_manifest"]["path"]).parent)
    locations, seeds = bindings["locations"], CONFIG["inference"]["seeds"]
    checkpoint = pathlib.Path(bindings["checkpoints"][role])
    metadata = json.loads((checkpoint / "checkpoint.json").read_text())
    manifest = project(metadata["base_manifest"])
    if sha256(manifest) != metadata["base_manifest_sha256"] \
            or any(sha256(project(pathlib.Path(metadata["base_model"]) / row["file"])) != row["sha256"]
                   for row in json.loads(manifest.read_text())["files"]):
        raise ValueError("复现动作时的基座或处理器发生改变")
    model = build_model(pathlib.Path(metadata["base_model"]), "lora", metadata["recipe"]["seed"], OmegaConf.create(metadata["model_config"]))
    _, stored = restore_checkpoint(checkpoint, model)
    del stored
    model.eval()

    def predict(examples, seed):
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]), torch.no_grad():
            torch.manual_seed(seed)
            return np.asarray(model.predict_action(examples)["normalized_actions"]).copy()

    saved_probe = metadata["prediction_probe"]
    observed = predict([cached_example(cache, row, "demo_validation") for row in saved_probe["locations"]], saved_probe["seed"])
    expected = np.asarray(saved_probe["normalized_actions"], dtype=np.float32)
    if observed.shape != expected.shape or not np.isfinite(observed).all():
        raise ValueError("原保存动作复现格式非法")
    error = float(np.max(np.abs(observed - expected)))
    reproduction = {"max_abs_diff": error, "passed": error <= CONFIG["inference"]["saved_probe_max_abs_error"],
                    "locations": saved_probe["locations"], "seed": saved_probe["seed"], "optimization_steps": 0}
    dump(output_path.with_suffix(".reproduction.json"), reproduction)
    if not reproduction["passed"]:
        raise ValueError(f"{role}零更新未复现原动作；停止后续比较，先定位加载路径")
    current = []
    for location in locations:
        example = cached_example(cache, location, "demo_validation")
        current.append([predict([example], seed)[0].tolist() for seed in seeds])
    del predict, model
    gc.collect()
    torch.cuda.empty_cache()
    result = {"schema": "thor-isolated-role-action-probe-v1", "role": role,
              "bindings_sha256": sha256(bindings_path), "checkpoint": str(checkpoint),
              "state_sha256": metadata["state_sha256"], "reproduction": reproduction,
              "observed_saved_probe": observed.tolist(), "predictions": current,
              "precision": metadata["precision"], "inference_seeds": seeds,
              "allocated_bytes_before_process_exit": torch.cuda.memory_allocated(),
              "optimization_steps": 0}
    dump(output_path, result)


def checked_role_output(path, bindings, bindings_path, role):
    import numpy as np

    report = json.loads(path.read_text())
    metadata = json.loads(pathlib.Path(bindings["files"]["checkpoint_" + role]["path"]).read_text())
    predictions = np.asarray(report["predictions"], dtype=np.float32)
    expected = np.asarray(metadata["prediction_probe"]["normalized_actions"], dtype=np.float32)
    observed = np.asarray(report["observed_saved_probe"], dtype=np.float32)
    if report.get("schema") != "thor-isolated-role-action-probe-v1" or report["role"] != role \
            or report["bindings_sha256"] != sha256(bindings_path) \
            or report["checkpoint"] != bindings["checkpoints"][role] \
            or report["state_sha256"] != metadata["state_sha256"] \
            or report["precision"] != metadata["precision"] \
            or report["inference_seeds"] != CONFIG["inference"]["seeds"] or report["optimization_steps"] != 0 \
            or predictions.shape != (len(bindings["locations"]), len(CONFIG["inference"]["seeds"]), 8, 7) \
            or not np.isfinite(predictions).all() or observed.shape != expected.shape \
            or not np.isfinite(observed).all():
        raise ValueError("独立角色动作产物的身份、配方、形状或有限性不符")
    error = float(np.max(np.abs(observed - expected)))
    reproduction = {"max_abs_diff": error, "passed": error <= CONFIG["inference"]["saved_probe_max_abs_error"],
                    "locations": metadata["prediction_probe"]["locations"],
                    "seed": metadata["prediction_probe"]["seed"], "optimization_steps": 0}
    if report["reproduction"] != reproduction or not reproduction["passed"]:
        raise ValueError("独立角色实际保存动作未复现，禁止汇总或继续训练")
    return report, predictions


def execute_role(directory, bindings, bindings_path, role):
    """协调进程不创建CUDA上下文；只有子进程退出且设备空闲才读取其结果。"""
    ensure_gpu_idle()
    output = directory / "roles" / (role + ".json")
    output.parent.mkdir(exist_ok=True)
    command = [sys.executable, "-u", str(ROOT / "scripts/p1_policy_regression_probe.py"), "role-actions",
               "--iteration-run", bindings["iteration_run"], "--role", role,
               "--bindings-file", str(bindings_path), "--role-output", str(output)]
    log_path = directory / "logs" / (role + ".log")
    with log_path.open("x") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        dump(directory / "progress.json", {"phase": "role_action_probe", "role": role,
                                           "pid": process.pid, "command": command, "log": str(log_path)})
        try:
            process.wait(timeout=1800)
        finally:
            cleanup = stop_process_group(process)
    if process.returncode or cleanup["forced_kill"] or cleanup["remaining_live_pids"]:
        raise RuntimeError(f"独立角色{role}诊断失败；停止下一权重，保留日志{log_path}")
    teardown = wait_stage_teardown({"pid": process.pid, "command": command})
    report, predictions = checked_role_output(output, bindings, bindings_path, role)
    return report, predictions, {"pid": process.pid, "command": command, "teardown": teardown,
                                "output": str(output), "output_sha256": sha256(output)}


def action_probe(directory, bindings):
    import numpy as np
    from p0_policy_runtime import verify_eval_isolation

    ensure_gpu_idle()
    verify_bindings(bindings)
    bindings_path = directory / "bindings.json"
    dump(bindings_path, bindings)
    cache = DemoCache(pathlib.Path(bindings["files"]["cache_manifest"]["path"]).parent)
    isolation = verify_eval_isolation(cache.path, pathlib.Path(bindings["files"]["frozen_eval"]["path"]))
    if isolation["eval_set_sha256"] != bindings["files"]["frozen_eval"]["sha256"]:
        raise ValueError("动作诊断的冻结隔离版本偏离排队输入")
    locations, seeds = bindings["locations"], CONFIG["inference"]["seeds"]
    targets = [cached_example(cache, location, "demo_validation")["action"] for location in locations]
    predictions, reproduction, workers = {}, {}, {}
    for role in ROLES:
        report, predictions[role], workers[role] = execute_role(directory, bindings, bindings_path, role)
        reproduction[role] = report["reproduction"]
        dump(directory / "saved_probe_reproduction.json", reproduction)
        print(json.dumps({"role": role, "completed_blocks": len(locations), "seeds": seeds,
                          "reproduction": reproduction[role]["max_abs_diff"], "process_exited": True}), flush=True)
    rows = []
    with (directory / "validation_predictions.jsonl").open("x") as stream:
        for role in ROLES:
            for index, location in enumerate(locations):
                for offset, seed in enumerate(seeds):
                    actual, old = predictions[role][index][offset], predictions["incumbent"][index][offset]
                    record = {"role": role, **location, "policy_seed": seed,
                              **action_metrics(actual, targets[index], old)}
                    rows.append(record)
                    stream.write(json.dumps({**record, "normalized_actions": actual.tolist(),
                                             "target_actions": targets[index].tolist()}) + "\n")
    with (directory / "validation_metrics.csv").open("x") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = {}
    for role in ROLES:
        summary[role] = {instruction: {key: float(np.mean([r[key] for r in rows if r["role"] == role and r["instruction"] == instruction]))
                                      for key in ["action_mse", "arm_mse", "gripper_mse", "drift_vs_incumbent_rms"]}
                         for instruction in sorted({r["instruction"] for r in rows})}
    result = {"schema": CONFIG["schema"], "inputs": bindings, "zero_update_reproduction": reproduction,
              "summary_by_task": summary, "validation_blocks": len(locations), "inference_seeds": seeds,
              "precision": report["precision"], "attention": "sdpa", "batch": 1, "eval_overlap_check": isolation,
              "isolated_role_processes": workers,
              "authority": CONFIG["authority"], "causal_attribution_established": False,
              "scope": "独立示范验证上的离线动作误差与漂移；不代表仿真任务成功或Cosmos独立贡献"}
    dump(directory / "action_probe.json", result)
    return {"zero_update_reproduction_passed": True, "validation_blocks": len(locations),
            "roles": ROLES, "inference_seeds": seeds, "optimization_steps": 0}


def bind_generation(execution, bindings):
    launch_path = project(execution / "launch.json")
    launch = json.loads(launch_path.read_text())
    source = project(launch["input"])
    prepared, _ = sealed(source, "p1_feedback_generation_input")
    if sha256(source) != launch["input_sha256"] or prepared["files"]["collection"] != bindings["files"]["collection"]:
        raise ValueError("等待的新生成不属于相同的独立反馈来源")
    return {"launch": str(launch_path), "launch_sha256": sha256(launch_path), "pid": launch["pid"],
            "input": str(source), "input_sha256": sha256(source), "log": str(project(launch["log"])),
            "after_predict_exec": launch["after_predict_exec"], "command": launch["command"]}


def wait_generation(binding, timeout):
    deadline = time.monotonic() + timeout
    while True:
        if sha256(project(binding["launch"])) != binding["launch_sha256"]:
            raise ValueError("确切新生成启动记录在等待期间改变")
        paths = [project(line.split("：", 1)[1]) for line in pathlib.Path(binding["log"]).read_text().splitlines()
                 if line.startswith("记录目录：")]
        if len(paths) > 1:
            raise ValueError("待接续的新生成阶段目录不唯一")
        if paths and (paths[0] / "probe.json").exists():
            path = paths[0] / "generation_report.json"
            report, _ = sealed(path, "p1_feedback_generation")
            if report.get("input_sha256") != binding["input_sha256"] or report.get("generation_performed") is not True:
                raise ValueError("新生成结果没有绑定确切排队输入")
            return {"path": str(path), "sha256": sha256(path)}
        try:
            arguments = pathlib.Path(f"/proc/{binding['pid']}/cmdline").read_bytes().split(b"\x00")
        except FileNotFoundError as error:
            raise RuntimeError("确切新生成任务已退出但没有封存产物；禁止自行抢占或重启") from error
        if not any(value.endswith(b"scripts/p1_feedback_generation.py") for value in arguments) \
                or binding["input"].encode() not in arguments or binding["after_predict_exec"].encode() not in arguments:
            raise RuntimeError("待接续的新生成进程身份失效")
        if time.monotonic() >= deadline:
            raise TimeoutError("等待确切新生成超时，保留既有任务")
        time.sleep(min(5, max(.01, deadline - time.monotonic())))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=["cpu", "actions", "role-actions"])
    parser.add_argument("--iteration-run", type=pathlib.Path, required=True)
    parser.add_argument("--config", type=pathlib.Path, default=ROOT / "configs/policy_regression_probe.yaml")
    parser.add_argument("--after-generation-exec", type=pathlib.Path)
    parser.add_argument("--wait-timeout", type=int, default=86400)
    parser.add_argument("--role", choices=ROLES)
    parser.add_argument("--bindings-file", type=pathlib.Path)
    parser.add_argument("--role-output", type=pathlib.Path)
    args = parser.parse_args()
    args.iteration_run = project(args.iteration_run)
    if args.action == "role-actions":
        if not args.role or args.bindings_file is None or args.role_output is None:
            parser.error("独立角色入口必须绑定角色、输入与输出")
        args.bindings_file = project(args.bindings_file)
        bindings = json.loads(args.bindings_file.read_text())
        output = args.role_output.resolve()
        if bindings["iteration_run"] != str(args.iteration_run) \
                or not args.bindings_file.is_relative_to(ROOT / "runs/p0") \
                or output != args.bindings_file.parent / "roles" / (args.role + ".json") or output.exists():
            parser.error("独立角色输入越界、来源错绑或输出已存在")
        role_action_probe(bindings, args.role, args.bindings_file, output)
        return 0
    bindings = bind_inputs(args.iteration_run, args.config)
    if args.action == "cpu":
        import os

        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            parser.error("CPU审计必须显式设置CUDA_VISIBLE_DEVICES为空")
        return run_stage("p1_policy_weight_reader_audit", lambda directory: cpu_audit(directory, bindings),
                         {"compute": "Thor CPU，无GPU模型加载；允许与既有评测并行", "optimization_steps": 0})
    if args.after_generation_exec is None or args.wait_timeout < 1:
        parser.error("动作检查须绑定确切生成任务和正等待时间")
    import fcntl

    with (ROOT / "runs/.policy_regression_probe.lock").open("a") as queue:
        fcntl.flock(queue, fcntl.LOCK_EX | fcntl.LOCK_NB)
        generation = bind_generation(project(args.after_generation_exec), bindings)
        print(json.dumps({"phase": "waiting_for_exact_generation", "inputs": bindings, "generation": generation}), flush=True)
        previous = wait_generation(generation, args.wait_timeout)
        teardown = wait_stage_teardown(generation)
        with (ROOT / "runs/.rsi_iteration.lock").open("a") as lock:
            # 上游先封存probe再退出；只等待其释放锁，不抢占其他阶段。
            deadline = time.monotonic() + 30
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise RuntimeError("生成封存后仍有阶段持锁，禁止抢占")
                    time.sleep(1)
            ensure_gpu_idle()
            return run_stage("p1_policy_action_probe", lambda directory: action_probe(directory, bindings),
                             {"after_generation": previous, "predecessor_teardown": teardown,
                              "compute": "Thor单卡串行，每次仅加载一份策略", "optimization_steps": 0})


if __name__ == "__main__":
    def interrupted(number, _frame):
        raise RuntimeError(f"零更新诊断收到信号{number}；不启动其他模型")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    raise SystemExit(main())
