#!/usr/bin/env python3
"""绑定独立失败反馈与成功示范，准备并串行执行新Cosmos窗口，质量资格另验。

@author pony
@date 2026-11-09
@version v1.0.0
@last_modified 2026-11-09
@changelog
    - v1.0.0 (2026-11-09): 新生成绑定实际统计路由，精确等待Predict并复用已验证生成器。

本入口不改变已运行整圈；新原生视频不能沿用旧纹理、UV方法或旧窗口的入训资格。
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import fcntl
import json
import pathlib
import shutil
import signal
import time
from types import SimpleNamespace

from p0_common import ROOT, ensure_gpu_idle, run_stage
from p0_feedback_judge import checked_collection
from p0_feedback_mine import wilson
from p0_lerobot_handoff import dump
from p1_eval_gate import bound_file, sealed, sha256


CONFIG = {
    "schema": "thor-feedback-generation-v1",
    "selection": {"route": "cosmos_restyle", "category": "visual", "subtype": "background",
                  "source": "verified_same_task_successful_demo", "maximum_windows": 1},
    "target_condition": {"background": "matte_dark_wood", "scope": "visible_table_texture_only"},
    "sampling": {"model": "edge/distilled", "precision": "bf16", "attention": "torch_sdpa", "batch": 1,
                 "frames": 93, "fps": 20, "diffusion_steps": 4, "resolution": "480", "seed": 42},
    "authority": {"eligible_for_training": False, "alter_current_iteration": False,
                  "production_promotion": False, "real_robot_enabled": False},
}
DEPENDENCIES = [
    "scripts/p1_feedback_generation.py", "scripts/p0_transfer_smoke.py", "scripts/p0_table_background.py",
    "scripts/p0_lossless_background.py", "scripts/p0_feedback_judge.py", "scripts/p0_feedback_mine.py",
    "scripts/p0_common.py", "scripts/p1_eval_gate.py", "configs/table_background_gate.yaml",
]
COPIED = ["source.mp4", "edge.mp4", "scene_mask.mp4", "roi_masks.npy", "source_reset_snapshot.npz",
          "table_elements.npy", "editable_table.npy"]


def project_file(path):
    path = pathlib.Path(path).resolve(strict=True)
    if not path.is_file() or not path.is_relative_to(ROOT):
        raise ValueError("新生成输入必须是本项目的真实文件")
    return path


def checked_config(path):
    import yaml

    if yaml.safe_load(path.read_text()) != CONFIG:
        raise ValueError("本单窗口新生成须保持预声明配方；扩大规模须另建版本")
    return CONFIG


def feedback_bindings(path):
    chain, _ = sealed(path, "feedback_chain")
    if chain.get("completed_feedback_chain") is not True:
        raise ValueError("新生成只接受完整独立反馈链")
    files = {"feedback": path}
    for key in ["collection", "judge", "mining"]:
        actual = project_file(chain[key])
        if chain["artifacts_sha256"].get(str(actual)) != sha256(actual):
            raise ValueError("新生成绑定的反馈阶段摘要改变")
        files[key] = actual
    collection, episodes, _ = checked_collection(files["collection"])
    mining, probe = sealed(files["mining"], "feedback_mine")
    if mining.get("collection_sha256") != sha256(files["collection"]) \
            or mining.get("judge_sha256") != sha256(files["judge"]) \
            or mining.get("policy_ckpt_sha256") != collection["policy_ckpt_sha256"] \
            or mining.get("decision_authority") != "environment_truth" \
            or mining.get("environment_truth_owns_success_pool") is not True or mining.get("is_eval") is not False:
        raise ValueError("新生成缺少同一策略的环境真值反馈")
    files["failure_records"] = bound_file(files["mining"].parent, probe, "failures/failure_records.jsonl")
    failures = [json.loads(line) for line in files["failure_records"].read_text().splitlines()]
    if len(failures) != mining["failure_records"]:
        raise ValueError("实际失败记录不完整")
    return files, collection, episodes, mining, failures


def select_route(episodes, records, instruction, minimum):
    """重新计算实际成败计数；不能只相信route布尔值或未校准判官的解释。"""
    identities = {row["episode_id"]: row for row in episodes}
    if len(identities) != len(episodes) or len({r["episode_id"] for r in records}) != len(records) \
            or type(minimum) is not int or minimum < 1:
        raise ValueError("反馈身份重复或统计样本要求非法")
    selected = []
    for record in records:
        if record.get("route") != "cosmos_restyle" or record.get("category") != "visual" \
                or record.get("sub_type") != "background":
            continue
        row = identities.get(record["episode_id"])
        if row is None or record.get("is_eval") is not False or record.get("eligible_for_generation") is not True \
                or record.get("routing_authority") != "environment_truth_condition_comparison" \
                or row.get("is_eval") is not False or row.get("isolation", {}).get("passed") is not True \
                or row.get("outcome") != {"success": False, "source": "env"} \
                or row.get("condition") != "background" \
                or record.get("policy_ckpt_sha256") != row.get("policy_ckpt_sha256") \
                or record.get("target_condition") != row.get("conditions"):
            raise ValueError("声称可生成的路由与真实失败、条件或父策略不一致")
        if row["instruction"] != instruction:
            continue
        stats = {}
        for condition in ["standard", "background"]:
            members = [r for r in episodes if r["task_id"] == row["task_id"] and r["condition"] == condition]
            if len(members) < minimum or any(r["instruction"] != instruction or r["is_eval"] is not False
                    or r["outcome"]["source"] != "env" or type(r["outcome"]["success"]) is not bool
                    or r["policy_ckpt_sha256"] != row["policy_ckpt_sha256"] for r in members):
                raise ValueError("视觉路由缺少足量的同任务独立环境反馈")
            failures = sum(not r["outcome"]["success"] for r in members)
            lower, upper = wilson(failures, len(members))
            stats[condition] = {"task_id": row["task_id"], "condition": condition, "failures": failures,
                "episodes": len(members), "failure_rate": failures / len(members),
                "wilson95_lower": lower, "wilson95_upper": upper}
        if stats["background"]["wilson95_lower"] <= stats["standard"]["wilson95_upper"] \
                or record["evidence"]["condition_statistics"] != stats["background"] \
                or record["evidence"]["standard_statistics"] != stats["standard"]:
            raise ValueError("背景薄弱条件缺少预声明统计支持，或失败矩阵与实际回合不符")
        selected.append({"episode_id": row["episode_id"], "task_id": row["task_id"],
            "condition_id": row["condition_id"], "policy_ckpt_sha256": row["policy_ckpt_sha256"],
            "target_condition": row["conditions"], "statistics": stats})
    if not selected:
        raise ValueError("没有与成功源同任务且得到统计支持的背景弱条件")
    return sorted(selected, key=lambda r: r["episode_id"])


def generation_spec(directory, instruction):
    # 只把有统计支持的弱条件映射到既定桌体目标；不根据冻结评测或判官自由文本改提示。
    return {"name": "p1_feedback_matte_wood_single_window", "video_path": str(directory / "source.mp4"),
        "prompt": "A fixed camera records the same Panda robot arm performing the task: " + instruction + ". "
            "Change only the visible tabletop surface to fine, continuous matte dark brown wood grain. "
            "Preserve the robot, gripper and every task and non-task object, their identities, colors, "
            "positions, geometry, visibility and motion. Preserve illumination and camera. "
            "Do not add, remove or replace objects, obstacles, text, patches, or overlays.",
        "max_frames": 93, "num_video_frames_per_chunk": 93, "num_steps": 4, "seed": 42,
        "resolution": "480", "keep_input_resolution": False,
        "edge": {"control_path": str(directory / "edge.mp4"), "control_weight": 1.0,
                 "mask_path": str(directory / "scene_mask.mp4")}}


def prepare(directory, args):
    checked_config(args.config)
    files, collection, episodes, mining, failures = feedback_bindings(args.feedback_chain)
    from p0_table_background import checked_table_source

    evidence, _, _, _, _, proof = checked_table_source(args.source_input)
    if proof["eval_overlap_check"]["frozen_eval_set_sha256"] != collection["eval_overlap_check"]["eval_set_sha256"]:
        raise ValueError("反馈与成功生成源没有绑定相同冻结隔离版本")
    routes = select_route(episodes, failures, evidence["instruction"], mining["config"]["min_condition_episodes"])
    for name in COPIED + ["source_labels.npz"]:
        checksum = evidence["labels_sha256"] if name == "source_labels.npz" else evidence["files_sha256"][name]
        original = project_file(args.source_input.parent / name)
        if sha256(original) != checksum:
            raise ValueError("成功源文件在准备时改变")
        shutil.copyfile(original, directory / name)
        if sha256(directory / name) != checksum:
            raise ValueError("新生成输入没有逐字节保留源标签、RGB或掩码")
    spec = generation_spec(directory, evidence["instruction"])
    from p0_transfer_smoke import validate_generation_spec

    validate_generation_spec(spec, "edge/distilled", 4)
    dump(directory / "spec.json", spec)
    derived = copy.deepcopy(evidence)
    derived.update(spec=str(directory / "spec.json"), control_recipe="p1-feedback-table-edge-distilled-v1",
        target_condition={"background": "matte_dark_wood"}, eligible_for_training=False,
        reason="独立反馈后的新生成输入；原生输出、纹理提取及派生数据均须另验资格",
        parent_verified_input=str(args.source_input), parent_verified_input_sha256=sha256(args.source_input),
        files_sha256={name: sha256(directory / name) for name in COPIED})
    # 新输入只复用原源的重放事实，不声称由旧准备器重新运行，也不携带旧候选资格。
    derived.pop("predeclared_experiment", None)
    derived.pop("composite_recipe", None)
    dump(directory / "verified_input.json", derived)
    files.update(source_input=args.source_input, config=args.config)
    result = {"schema": CONFIG["schema"], "files": {name: {"path": str(path), "sha256": sha256(path)}
                for name, path in files.items()}, "source_episode_id": evidence["source_episode_id"],
        "source_proof": proof, "source_instruction": evidence["instruction"], "routes": routes,
        "sampling": CONFIG["sampling"], "target_condition": CONFIG["target_condition"],
        "spec_sha256": sha256(directory / "spec.json"), "verified_input_sha256": sha256(directory / "verified_input.json"),
        "input_files_sha256": {name: sha256(directory / name) for name in COPIED + ["source_labels.npz"]},
        "dependencies_sha256": {name: sha256(ROOT / name) for name in DEPENDENCIES},
        "prepared_at": dt.datetime.now(dt.timezone.utc).isoformat(), "generation_performed": False,
        "eligible_for_training": False, "alter_current_iteration": False, "complete_rsi": False,
        "boundary": "一个成功D0窗口；失败只提供条件反馈；新生成及其后续派生必须独立验收"}
    dump(directory / "generation_input.json", result)
    return {"source_episode_id": result["source_episode_id"], "feedback_routes": len(routes),
            "generation_performed": False, "eligible_for_training": False}


def checked_input(path):
    report, probe = sealed(path, "p1_feedback_generation_input")
    if report.get("schema") != CONFIG["schema"] or report.get("generation_performed") is not False \
            or report.get("eligible_for_training") is not False or report.get("alter_current_iteration") is not False \
            or report.get("sampling") != CONFIG["sampling"] or report.get("target_condition") != CONFIG["target_condition"]:
        raise ValueError("新生成输入没有保持单窗口版本及待验权限")
    for name, expected in report["dependencies_sha256"].items():
        if name not in DEPENDENCIES or sha256(ROOT / name) != expected:
            raise ValueError("排队期间生成实现已改变")
    if set(report["dependencies_sha256"]) != set(DEPENDENCIES):
        raise ValueError("新生成依赖清单不完整")
    for binding in report["files"].values():
        if sha256(project_file(binding["path"])) != binding["sha256"]:
            raise ValueError("排队期间反馈、配置或成功源改变")
    config_path = pathlib.Path(report["files"]["config"]["path"])
    checked_config(config_path)
    files, collection, episodes, mining, failures = feedback_bindings(pathlib.Path(report["files"]["feedback"]["path"]))
    if any(report["files"].get(name) != {"path": str(actual), "sha256": sha256(actual)} for name, actual in files.items()):
        raise ValueError("新生成来源没有绑定同一条实际反馈链")
    from p0_table_background import checked_table_source

    original = pathlib.Path(report["files"]["source_input"]["path"])
    evidence, _, _, _, _, proof = checked_table_source(original)
    if proof != report["source_proof"] or evidence["source_episode_id"] != report["source_episode_id"] \
            or evidence["instruction"] != report["source_instruction"] \
            or report["routes"] != select_route(episodes, failures, evidence["instruction"], mining["config"]["min_condition_episodes"]) \
            or proof["eval_overlap_check"]["frozen_eval_set_sha256"] != collection["eval_overlap_check"]["eval_set_sha256"]:
        raise ValueError("新生成成功源、统计路由或隔离版本错绑")
    expected_files = {name: evidence["files_sha256"][name] for name in COPIED}
    expected_files["source_labels.npz"] = evidence["labels_sha256"]
    if report["input_files_sha256"] != expected_files:
        raise ValueError("新生成文件清单与原成功源不一致")
    for name, expected in expected_files.items():
        if sha256(bound_file(path.parent, probe, name)) != expected:
            raise ValueError("实际新生成输入文件已改变")
    spec_path = bound_file(path.parent, probe, "spec.json")
    evidence_path = bound_file(path.parent, probe, "verified_input.json")
    if sha256(spec_path) != report["spec_sha256"] or json.loads(spec_path.read_text()) != generation_spec(path.parent, evidence["instruction"]) \
            or sha256(evidence_path) != report["verified_input_sha256"]:
        raise ValueError("新生成配方或逐帧源证据已改变")
    actual = json.loads(evidence_path.read_text())
    if actual["files_sha256"] != {name: expected_files[name] for name in COPIED} \
            or actual["labels_sha256"] != expected_files["source_labels.npz"] \
            or actual["parent_verified_input"] != str(original) \
            or actual["parent_verified_input_sha256"] != sha256(original):
        raise ValueError("原生生成器读取的源与已核验输入错绑")
    return report


def generate(directory, input_path):
    ensure_gpu_idle()
    report = checked_input(input_path)
    from p0_transfer_smoke import generate as transfer_generate

    args = SimpleNamespace(spec=input_path.parent / "spec.json", num_steps=4, model="edge/distilled",
                           resolution="480", seed=42)
    native = transfer_generate(directory, args)
    outputs = native.get("outputs", [])
    if native.get("guardrails_enabled") is not True or native.get("recipe") != "edge/distilled" \
            or any(native.get(key) != value for key, value in {"num_steps": 4, "seed": 42, "batch": 1}.items()) \
            or len(outputs) != 1:
        raise ValueError("实际新生成没有保持四步单窗口配方或安全检查")
    output = pathlib.Path(outputs[0]).resolve(strict=True)
    if not output.is_file() or not output.is_relative_to(directory):
        raise ValueError("实际原生视频不属于本次新生成阶段")
    result = {"schema": CONFIG["schema"], "input": str(input_path), "input_sha256": sha256(input_path),
        "feedback": report["files"]["feedback"], "source_episode_id": report["source_episode_id"],
        "sampling": report["sampling"], "target_condition": report["target_condition"],
        "control_scope": "同帧非桌体及八像素桌体边界的edge控制；原生输出仍须另验",
        "native_generation": native, "generation_performed": True, "feedback_bound_before_generation": True,
        "output_video_sha256": sha256(output),
        "eligible_for_training": False, "production_promotion_performed": False, "complete_rsi": False,
        "real_robot_commands_sent": False,
        "boundary": "新原生窗口未取得质量资格；不得称合格UV派生、光照增强、策略收益或完整RSI"}
    dump(directory / "generation_report.json", result)
    return result


def wait_predict(exec_path, input_report, timeout):
    launch_path = project_file(exec_path / "launch.json")
    launch_sha = sha256(launch_path)
    launch = json.loads(launch_path.read_text())
    pid = launch.get("pid")
    if type(pid) is not int or pid < 1:
        raise ValueError("Predict等待任务缺少具体进程")
    log = exec_path / "logs/predict_validation.log"
    deadline = time.monotonic() + timeout
    while True:
        if sha256(launch_path) != launch_sha:
            raise ValueError("等待期间Predict启动记录改变")
        paths = [pathlib.Path(line.split("：", 1)[1]).resolve() for line in log.read_text().splitlines()
                 if line.startswith("记录目录：")]
        if len(paths) > 1 or any(not path.is_relative_to(ROOT / "runs/p0") for path in paths):
            raise ValueError("确切Predict任务的实际阶段目录不唯一或越界")
        if paths and (paths[0] / "probe.json").exists():
            report_path = paths[0] / "predict_validation.json"
            report, _ = sealed(report_path, "p1_predict_validation")
            if report.get("diagnostic_complete") is not True \
                    or report["inputs"]["iteration_run"] != launch["iteration_run"] \
                    or report["inputs"]["files"]["feedback"] != input_report["files"]["feedback"]:
                raise ValueError("新生成没有接续同一反馈来源的确切Predict任务")
            return {"path": str(report_path), "sha256": sha256(report_path), "launch_sha256": launch_sha}
        try:
            command = pathlib.Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\x00")
        except FileNotFoundError as error:
            raise RuntimeError("确切Predict任务已退出且没有封存结果，不自行重启") from error
        if not any(part.endswith(b"scripts/p1_predict_validation.py") for part in command) \
                or launch["iteration_run"].encode() not in command:
            raise RuntimeError("确切Predict进程身份已经失效")
        if time.monotonic() >= deadline:
            raise TimeoutError("等待确切Predict超时；保留原任务")
        time.sleep(min(5, max(.01, deadline - time.monotonic())))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="action", required=True)
    preparation = commands.add_parser("prepare")
    preparation.add_argument("--feedback-chain", type=pathlib.Path, required=True)
    preparation.add_argument("--source-input", type=pathlib.Path, required=True)
    preparation.add_argument("--config", type=pathlib.Path, default=ROOT / "configs/feedback_generation.yaml")
    generation = commands.add_parser("generate")
    generation.add_argument("--input", type=pathlib.Path, required=True)
    generation.add_argument("--after-predict-exec", type=pathlib.Path, required=True)
    generation.add_argument("--wait-timeout", type=int, default=86400)
    args = parser.parse_args()
    if ROOT != pathlib.Path("/home/thor/rsi"):
        parser.error("新生成只能在Thor本项目执行")
    for name in ["feedback_chain", "source_input", "config", "input"]:
        if hasattr(args, name):
            setattr(args, name, project_file(getattr(args, name)))
    if args.action == "prepare":
        return run_stage("p1_feedback_generation_input", lambda directory: prepare(directory, args),
            {"compute": "仅CPU校验与文件复制，不加载模型", "source_windows": 1, "frames": 93,
             "precision": "源RGB uint8、动作/状态fp32逐字节保留", "model_sampling": CONFIG["sampling"]})
    args.after_predict_exec = args.after_predict_exec.resolve(strict=True)
    if not args.after_predict_exec.is_relative_to(ROOT / "runs/p0") or args.wait_timeout < 1:
        parser.error("只允许等待本项目确切任务及正等待时长")
    report = checked_input(args.input)
    with (ROOT / "runs/.feedback_generation.lock").open("a") as queue_lock:
        fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(json.dumps({"phase": "waiting_for_exact_predict_validation", "input": str(args.input),
                          "input_sha256": sha256(args.input)}, ensure_ascii=False), flush=True)
        previous = wait_predict(args.after_predict_exec, report, args.wait_timeout)
        with (ROOT / "runs/.rsi_iteration.lock").open("a") as model_lock:
            deadline = time.monotonic() + 30
            while True:
                try:
                    fcntl.flock(model_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise RuntimeError("其他阶段仍在运行，禁止抢占")
                    time.sleep(1)
            ensure_gpu_idle()
            return run_stage("p1_feedback_generation", lambda directory: generate(directory, args.input),
                {"sampling": CONFIG["sampling"], "source_windows": 1, "after_predict": previous,
                 "compute": "Thor模型阶段串行；等待不计入生成墙钟"})


if __name__ == "__main__":
    def interrupted(number, _frame):
        raise RuntimeError(f"反馈新生成收到信号{number}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    raise SystemExit(main())
