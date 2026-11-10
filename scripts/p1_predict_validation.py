#!/usr/bin/env python3
"""在确切Reason接续结束后串行验证Predict多状态，保存全部端点及复制当前图基准。

@author pony
@date 2026-11-09
@version v1.0.0
@last_modified 2026-11-09
@changelog
    - v1.0.0 (2026-11-09): 预绑定分层样本与时刻、真实终态、环境隔离运行及逐像素指标复核。

本入口是既有反馈上的离线开发诊断，不代表独立任务收益、有效规划或想象数据训练。
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import pathlib
import signal
import subprocess
import time

from p0_common import ROOT, ensure_gpu_idle, run_stage, stop_process_group
from p0_feedback_judge import checked_collection
from p0_lerobot_handoff import dump
from p1_eval_gate import bound_file, sealed, sha256


def checked_config(path):
    import yaml

    config = yaml.safe_load(path.read_text())
    selection, sampling = config["selection"], config["sampling"]
    if config.get("schema") != "thor-predict-validation-v1" \
            or selection.get("outcomes") != [False, True] \
            or selection.get("episode_order") != "lexicographic_id" \
            or selection.get("positions") != ["initial", "frame80_or_last_valid", "terminal_16"] \
            or sampling != {"precision": "bf16", "attention": "torch_sdpa", "batch": 1,
                            "diffusion_steps": 4, "seed": 42, "action_horizon": 16} \
            or config.get("authority") != {"eligible_for_training": False, "eligible_for_planning": False,
                                         "production_promotion": False, "real_robot_enabled": False} \
            or config["metrics"].get("pixel_error_is_task_success") is not False:
        raise ValueError("Predict诊断版本、样本约定、采样或权限边界不符")
    low, high = [selection[key] for key in ["minimum_episodes_per_stratum", "maximum_episodes_per_stratum"]]
    if type(low) is not int or type(high) is not int or not 1 <= low <= high <= 4 \
            or not selection["conditions"] or len(set(selection["conditions"])) != len(selection["conditions"]):
        raise ValueError("Predict预声明分层数量或条件非法")
    return config


def select_states(episodes, selection):
    """只按预声明条件、环境成败和ID选择；不足照实报告，不观察预测后挑样本。"""
    identities = [row["episode_id"] for row in episodes]
    if len(set(identities)) != len(identities) or any(row["is_eval"] is not False
            or row["outcome"]["source"] != "env" or type(row["outcome"]["success"]) is not bool for row in episodes):
        raise ValueError("Predict来源必须是独立且无重复的环境真值反馈")
    cases, strata = [], []
    for condition in selection["conditions"]:
        for outcome in selection["outcomes"]:
            matching = sorted((row for row in episodes if row["condition_id"] == condition
                and row["outcome"]["success"] is outcome and row["metrics"]["steps"] >= 16),
                key=lambda row: row["episode_id"])
            selected = matching[:selection["maximum_episodes_per_stratum"]]
            if len(selected) < selection["minimum_episodes_per_stratum"]:
                raise ValueError(f"Predict条件{condition}的环境成败层{outcome}缺少足量完整动作源")
            strata.append({"condition_id": condition, "environment_success": outcome,
                           "available_episodes": len(matching), "selected_episodes": len(selected),
                           "maximum_requested": selection["maximum_episodes_per_stratum"]})
            for episode in selected:
                count = episode["metrics"]["steps"]
                if type(count) is not int:
                    raise ValueError("实际动作数量须为整数")
                positions = {"initial": 0, "frame80_or_last_valid": min(80, count - 16), "terminal_16": count - 16}
                for frame in sorted(set(positions.values())):
                    cases.append({"case_id": f"state_{len(cases):03d}", "source_episode_id": episode["episode_id"],
                        "condition_id": condition, "source_environment_success": outcome,
                        "init_seed": episode["init_seed"], "policy_seed": episode["policy_seed"],
                        "policy_ckpt_sha256": episode["policy_ckpt_sha256"], "first_frame": frame,
                        "future_frame": frame + 16, "positions": [key for key, value in positions.items() if value == frame],
                        "action_count": count, "trajectory": episode["trajectory"],
                        "trajectory_sha256": episode["trajectory_sha256"]})
    return {"cases": cases, "strata": strata, "source_episodes": len(set(row["source_episode_id"] for row in cases)),
            "source_initial_seed_count": len(set(row["init_seed"] for row in cases)), "states": len(cases),
            "boundary": "多时刻、动作分支和条件存在相关性；不把端点图或视角数计为独立回合"}


def bind_inputs(args):
    config = checked_config(args.config)
    for path in [args.iteration_run, args.after_reason_exec]:
        if not path.is_relative_to(ROOT / "runs/p0"):
            raise ValueError("Predict只等待本项目明确指定的真实任务")
    state = json.loads((args.iteration_run / "iteration_state.json").read_text())
    binding = state["completed_stages"]["feedback"]
    feedback_path = pathlib.Path(binding["path"]).resolve(strict=True)
    feedback, _ = sealed(feedback_path, "feedback_chain")
    if sha256(feedback_path) != binding["sha256"] or feedback.get("completed_feedback_chain") is not True:
        raise ValueError("Predict来源没有绑定确切整圈反馈")
    collection = pathlib.Path(feedback["collection"]).resolve(strict=True)
    if feedback["artifacts_sha256"].get(str(collection)) != sha256(collection):
        raise ValueError("Predict源采集摘要与主反馈交接不同")
    source, episodes, probe = checked_collection(collection)
    plan = select_states(episodes, config["selection"])
    import numpy as np

    for row in {item["source_episode_id"]: item for item in plan["cases"]}.values():
        path = pathlib.Path(row["trajectory"]).resolve(strict=True)
        if not path.is_relative_to(collection.parent) or sha256(path) != row["trajectory_sha256"] \
                or probe["artifacts_sha256"].get(str(path.relative_to(collection.parent))) != row["trajectory_sha256"]:
            raise ValueError("Predict分层样本的实际轨迹已改变")
        with np.load(path, allow_pickle=False) as stored:
            if len(stored["action"]) != row["action_count"]:
                raise ValueError("Predict动作长度与源回合不符")
    launch_path = args.after_reason_exec / "launch.json"
    launch = json.loads(launch_path.read_text())
    if launch["iteration_run"] != str(args.iteration_run) or type(launch["pid"]) is not int:
        raise ValueError("待接续Reason任务不属于确切首圈")
    files = {"config": args.config, "collection": collection, "feedback": feedback_path,
             "reason_launch": launch_path}
    return {"iteration_run": str(args.iteration_run), "iteration_inputs": state["inputs"],
            "reason_exec": str(args.after_reason_exec), "reason_pid": launch["pid"], "plan": plan,
            "policy_ckpt_sha256": source["policy_ckpt_sha256"],
            "files": {key: {"path": str(path), "sha256": sha256(path)} for key, path in files.items()}}


def wait_reason(bindings, timeout):
    log = pathlib.Path(bindings["reason_exec"]) / "logs/reason_followup.log"
    deadline = time.monotonic() + timeout
    while True:
        paths = [pathlib.Path(line.split("：", 1)[1]).resolve() for line in log.read_text().splitlines()
                 if line.startswith("记录目录：")]
        if len(paths) > 1 or any(not path.is_relative_to(ROOT / "runs/p0") for path in paths):
            raise ValueError("Reason后续实际阶段目录不唯一或越界")
        if paths and (paths[0] / "probe.json").exists():
            result_path = paths[0] / "reason_followup.json"
            report, _ = sealed(result_path, "p1_reason_followup")
            if report["inputs"]["iteration_run"] != bindings["iteration_run"] \
                    or report["inputs"]["files"]["collection"] != bindings["files"]["collection"]:
                raise ValueError("Reason接续结果与Predict来源不同")
            return {"path": str(result_path), "sha256": sha256(result_path), "decision": report["decision"]}
        try:
            command = pathlib.Path(f"/proc/{bindings['reason_pid']}/cmdline").read_bytes().split(b"\x00")
        except FileNotFoundError as error:
            raise RuntimeError("Reason确切等待任务已退出且没有完整封存结果，禁止自行重启") from error
        if not any(value.endswith(b"scripts/p1_reason_followup.py") for value in command) \
                or str(bindings["iteration_run"]).encode() not in command:
            raise RuntimeError("Reason确切进程身份缺失，不能依据旧PID启动模型")
        if time.monotonic() >= deadline:
            raise TimeoutError("等待Reason接续超时；保留原任务，不重复启动")
        time.sleep(min(5, max(.01, deadline - time.monotonic())))


def execute_probe(directory, phase, environment, script, expected, artifact, arguments):
    if environment not in {"policy", "sim", "predict"}:
        raise ValueError("未知Predict阶段环境")
    ensure_gpu_idle()
    log_path = directory / "logs" / (phase + ".log")
    command = [str(ROOT / "envs" / environment / "bin/python"), "-u", str(ROOT / "scripts" / script), *map(str, arguments)]
    with log_path.open("x") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        dump(directory / "progress.json", {"phase": phase, "pid": process.pid, "command": command, "log": str(log_path)})
        try:
            process.wait()
        finally:
            cleanup = stop_process_group(process)
    if process.returncode or cleanup["forced_kill"]:
        raise RuntimeError(f"Predict阶段{phase}失败，保留日志{log_path}")
    paths = [pathlib.Path(line.split("：", 1)[1]).resolve() for line in log_path.read_text().splitlines()
             if line.startswith("记录目录：")]
    if len(paths) != 1 or not paths[0].is_relative_to(ROOT / "runs/p0"):
        raise ValueError("Predict子阶段记录目录不唯一或越界")
    result = paths[0] / artifact
    sealed(result, expected)
    ensure_gpu_idle()
    return result


def checked_metrics(case, transition_path, prediction_path):
    import numpy as np
    from PIL import Image

    transition, _ = sealed(transition_path, "predict_transition")
    prediction, probe = sealed(prediction_path, "predict_external_action")
    if transition["source_episode_id"] != case["source_episode_id"] \
            or transition["first_frame"] != case["first_frame"] or transition["future_frame"] != case["future_frame"] \
            or transition["policy_ckpt_sha256"] != case["policy_ckpt_sha256"] \
            or prediction["transition_sha256"] != sha256(transition_path) \
            or prediction["same_current_state"] is not True or prediction["same_sampling_seed"] is not True \
            or prediction["eligible_for_training"] is not False or prediction["eligible_for_planning"] is not False \
            or prediction["sample_count"] != 1 or prediction["action_branches"] != 2 \
            or prediction["steps"] != 4 or prediction["seed"] != 42 or prediction["batch"] != 1 \
            or prediction["precision"] != "bf16" or prediction["attention"] != "torch SDPA" \
            or prediction["action_horizon"] != 16 or prediction["environment_control_hz"] != 20 \
            or len(prediction["condition_audit"]) != 2 \
            or any(row["max_injection_error"] != 0 or row["clean_condition"] is not True for row in prediction["condition_audit"]):
        raise ValueError("Predict多状态结果的真实源、动作条件或权限不符")
    futures = {row["name"]: row for row in transition["branches"]}
    branches = {row["name"]: row for row in prediction["branches"]}
    if set(futures) != {"executed", "hold"} or set(branches) != set(futures):
        raise ValueError("Predict双动作分支不完整")
    if case["future_frame"] == case["action_count"] and (transition.get("terminal_reference_used") is not True
            or futures["executed"]["environment_success_after_16_steps"] is not case["source_environment_success"]):
        raise ValueError("Predict最后动作窗口没有恢复原回合真实终态")
    rows = []
    for name, branch in branches.items():
        for view in ["agentview", "wrist"]:
            images = {}
            for kind in ["current", "prediction", "truth"]:
                path = bound_file(prediction_path.parent, probe, f"{name}_{view}_{kind}.png")
                with Image.open(path) as image:
                    if image.mode != "RGB" or image.size != (224, 224):
                        raise ValueError("Predict像素误差必须使用原始224图，禁止展示缩放后计算")
                    images[kind] = np.asarray(image).astype(np.float64)
            measured = {key + "_mae_255": float(np.abs(images[label] - images["truth"]).mean())
                        for key, label in [("prediction", "prediction"), ("copy_current", "current")]}
            if any(not math.isfinite(branch["metrics"][view][key])
                   or abs(branch["metrics"][view][key] - value) > 1e-9 for key, value in measured.items()):
                raise ValueError("Predict报告的像素误差与实际图片不符")
            rows.append({**case, "branch": name, "view": view, **measured,
                         "environment_success_after_16_steps": futures[name]["environment_success_after_16_steps"],
                         "predict_decode_wall_s": branch["predict_decode_wall_s"],
                         "prediction_report": str(prediction_path), "prediction_report_sha256": sha256(prediction_path),
                         "transition": str(transition_path), "transition_sha256": sha256(transition_path)})
    return rows


def summarize(rows, plan):
    expected = {(case["case_id"], branch, view) for case in plan["cases"]
                for branch in ["executed", "hold"] for view in ["agentview", "wrist"]}
    actual = {(row["case_id"], row["branch"], row["view"]) for row in rows}
    if len(rows) != len(expected) or actual != expected:
        raise ValueError("Predict汇总缺少、重复或增加了未预声明端点")
    cases = {row["case_id"]: row for row in plan["cases"]}
    groups = {}
    for row in rows:
        if any(row.get(key) != value for key, value in cases[row["case_id"]].items()):
            raise ValueError("Predict端点的源回合、时刻或策略与预声明样本不符")
        for key in ["prediction_mae_255", "copy_current_mae_255"]:
            if not math.isfinite(row[key]) or not 0 <= row[key] <= 255:
                raise ValueError("Predict汇总存在非法像素误差")
        groups.setdefault(row["source_episode_id"], []).append(row)
    episode_means = [{"source_episode_id": identity,
        **{key: sum(row[key] for row in values) / len(values) for key in ["prediction_mae_255", "copy_current_mae_255"]}}
        for identity, values in sorted(groups.items())]
    means = {key: sum(row[key] for row in episode_means) / len(episode_means)
             for key in ["prediction_mae_255", "copy_current_mae_255"]}
    return {"states": len(plan["cases"]), "source_episodes": len(groups), "pixel_comparisons": len(rows),
            "equal_episode_mean": means, "episode_means": episode_means,
            "prediction_beats_copy_on_mean_pixel_error": means["prediction_mae_255"] < means["copy_current_mae_255"],
            "pixel_error_is_task_success": False, "eligible_for_planning": False, "eligible_for_training": False,
            "goal_critical_metrics_and_action_ranking_validated": False}


def export_materials(directory, rows, report):
    import csv
    from PIL import Image, ImageDraw, ImageFont

    output = directory / "materials"
    output.mkdir(exist_ok=False)
    font_path = subprocess.check_output(["fc-match", "-f", "%{file}", "Noto Sans CJK SC"], text=True)
    fonts = {size: ImageFont.truetype(font_path, size) for size in [24, 28, 32, 38, 48, 58]}
    resampling = getattr(Image, "Resampling", Image)
    fields = [key for key in rows[0] if key != "positions"]
    with (output / "endpoint_metrics.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows({key: row[key] for key in fields} for row in rows)
    with (output / "episode_means.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report["summary"]["episode_means"][0]))
        writer.writeheader(); writer.writerows(report["summary"]["episode_means"])
    cases = report["inputs"]["plan"]["cases"]
    by_case = {case["case_id"]: [row for row in rows if row["case_id"] == case["case_id"]] for case in cases}
    sources = {}
    for offset in range(0, len(cases), 2):
        canvas = Image.new("RGB", (3840, 2160), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((90, 55), "Cosmos Predict：固定多状态、双动作与真实未来对照", font=fonts[58], fill="#203541")
        draw.text((90, 145), "Thor / 保持既有功耗 / bf16 / Torch SDPA / batch1 / 4步扩散 / seed42 / 16动作后端点", font=fonts[32], fill="#526D7C")
        draw.text((90, 210), "源为独立采集反馈，按条件/环境成败/回合ID预选；这是离线开发诊断，不是冻结策略收益评测。", font=fonts[32], fill="#526D7C")
        columns = [(620, "current", "agentview", "当前·外部"), (1120, "current", "wrist", "当前·腕部"),
                   (1620, "prediction", "agentview", "预测·外部（模型）"), (2120, "prediction", "wrist", "预测·腕部（模型）"),
                   (2620, "truth", "agentview", "未来·外部（仿真）"), (3120, "truth", "wrist", "未来·腕部（仿真）")]
        for x, _, _, title in columns:
            draw.text((x, 300), title, font=fonts[28], fill="#203541")
        for position, case in enumerate(cases[offset:offset + 2]):
            values = by_case[case["case_id"]]
            prediction_path = pathlib.Path(values[0]["prediction_report"])
            _, probe = sealed(prediction_path, "predict_external_action")
            for branch_index, branch in enumerate(["executed", "hold"]):
                y = 380 + position * 740 + branch_index * 350
                draw.rounded_rectangle((70, y - 15, 3770, y + 330), radius=15, fill="#F1F5F7")
                texts = [case["case_id"] + " / " + case["condition_id"],
                         "原策略动作" if branch == "executed" else "零位姿增量、保持夹爪",
                         f"时刻 {case['first_frame']} → {case['future_frame']}",
                         "源回合环境成功" if case["source_environment_success"] else "源回合环境失败",
                         "源ID尾部：" + case["source_episode_id"][-22:]]
                for index, text in enumerate(texts):
                    draw.text((90, y + 45 + index * 48), text, font=fonts[24], fill="#203541")
                for x, kind, view, _ in columns:
                    relative = f"{branch}_{view}_{kind}.png"
                    path = bound_file(prediction_path.parent, probe, relative)
                    with Image.open(path) as image:
                        canvas.paste(image.resize((320, 320), resampling.NEAREST), (x, y))
                    sources[str(path)] = {"sha256": sha256(path), "kind": "model_prediction" if kind == "prediction" else "simulator_render"}
        draw.text((90, 1900), f"完整诊断：{report['summary']['source_episodes']}条源回合、{report['summary']['states']}个时刻；全部分页展示，时刻和视角不是独立回合。", font=fonts[32], fill="#526D7C")
        draw.text((90, 1970), "原图224×224，本页仅为展示放大；未来预测列由模型生成，当前/真实未来列由仿真渲染。", font=fonts[32], fill="#526D7C")
        draw.text((90, 2040), "像素MAE不能证明任务成败或动作排序；Predict仍不用于规划、训练标签或晋升门禁。", font=fonts[32], fill="#9D641D")
        canvas.save(output / f"endpoint_comparison_{offset // 2 + 1:03d}.png")
    # 四组均先在源回合内平均多个时刻；不把图片数量当作独立样本量或置信区间。
    canvas = Image.new("RGB", (3840, 2160), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((100, 70), "Predict 多状态端点误差：与复制当前图比较", font=fonts[58], fill="#203541")
    draw.text((100, 165), "Thor / 既有功耗 / bf16 / Torch SDPA / batch1 / 4步扩散 / seed42 / 16步动作后端点", font=fonts[32], fill="#526D7C")
    groups = []
    for branch in ["executed", "hold"]:
        for view in ["agentview", "wrist"]:
            identities = {}
            for row in rows:
                if row["branch"] == branch and row["view"] == view:
                    identities.setdefault(row["source_episode_id"], []).append(row)
            groups.append({"label": ("原策略动作" if branch == "executed" else "保持姿态") + " / " + ("外部" if view == "agentview" else "腕部"),
                **{key: sum(sum(row[key] for row in values) / len(values) for values in identities.values()) / len(identities)
                   for key in ["prediction_mae_255", "copy_current_mae_255"]}})
    highest = max(row[key] for row in groups for key in ["prediction_mae_255", "copy_current_mae_255"]) or 1.
    maximum = highest * 1.2
    chart_top, chart_bottom = 390, 1510
    for index in range(6):
        level = index / 5 * maximum
        y = chart_bottom - index / 5 * (chart_bottom - chart_top)
        draw.line((290, y, 3680, y), fill="#DBE4E9", width=2)
        draw.text((110, y - 18), f"{level:.2f}", font=fonts[28], fill="#526D7C")
    draw.text((110, 300), "原图像素 MAE（0–255）", font=fonts[32], fill="#526D7C")
    for index, group in enumerate(groups):
        x = 500 + index * 820
        for offset, key, color in [(0, "prediction_mae_255", "#217C85"), (250, "copy_current_mae_255", "#9FAEB8")]:
            y = chart_bottom - group[key] / maximum * (chart_bottom - chart_top)
            draw.rectangle((x + offset, y, x + offset + 190, chart_bottom), fill=color)
            draw.text((x + offset, y - 60), f"{group[key]:.3f}", font=fonts[32], fill="#203541")
        draw.text((x - 20, 1550), group["label"], font=fonts[32], fill="#203541")
    draw.rectangle((600, 1685, 645, 1730), fill="#217C85")
    draw.text((670, 1680), "动作条件预测", font=fonts[38], fill="#203541")
    draw.rectangle((1580, 1685, 1625, 1730), fill="#9FAEB8")
    draw.text((1650, 1680), "复制当前图基准", font=fonts[38], fill="#203541")
    draw.text((100, 1830), f"来源：{report['summary']['source_episodes']}条回合、{report['summary']['states']}个状态；各源回合等权，多个时刻/条件存在相关性。", font=fonts[32], fill="#526D7C")
    draw.text((100, 1900), "每组先在同一源回合内平均时刻，再平均各源；不把双动作、双视角计成独立试验。", font=fonts[32], fill="#526D7C")
    draw.text((100, 1970), "全图MAE容易受背景影响；没有验证任务物体物理误差、动作排序或真实规划收益。", font=fonts[32], fill="#9D641D")
    draw.text((100, 2040), "预测误差更低也不会授予规划或训练资格；原始逐端点和逐回合指标全部保存在CSV。", font=fonts[32], fill="#9D641D")
    canvas.save(output / "predict_pixel_errors.png")
    manifest = {"schema": "thor-predict-validation-materials-v1", "summary": report["summary"],
        "selection": "全部预声明样本按case_id分页，无事后挑选", "all_endpoint_comparisons_exported": True,
        "prediction_video_created": False, "prediction_columns_model_generated": True,
        "truth_columns_model_generated": False, "sources": sources,
        "files_sha256": {path.name: sha256(path) for path in output.iterdir() if path.is_file()}}
    dump(output / "materials_manifest.json", manifest)
    return {"directory": str(output), "manifest_sha256": sha256(output / "materials_manifest.json")}


def validate(directory, bindings):
    for item in bindings["files"].values():
        if sha256(pathlib.Path(item["path"])) != item["sha256"]:
            raise ValueError("Predict排队期间输入变化，须新建明确实验")
    files = {key: pathlib.Path(value["path"]) for key, value in bindings["files"].items()}
    dump(directory / "selection_plan.json", bindings["plan"])
    transitions = {}
    for case in bindings["plan"]["cases"]:
        transitions[case["case_id"]] = execute_probe(directory, case["case_id"] + "_transition", "sim",
            "p0_predict_transition.py", "predict_transition", "transition.json",
            ["--collection", files["collection"], "--episode-id", case["source_episode_id"], "--frame", case["first_frame"]])
    embeddings = execute_probe(directory, "language_embeddings", "policy", "p0_predict_embeddings.py", "predict_embeddings",
                               "embeddings.json", ["--collection", files["collection"]])
    rows = []
    for case in bindings["plan"]["cases"]:
        transition = transitions[case["case_id"]]
        prediction = execute_probe(directory, case["case_id"] + "_prediction", "predict", "p0_predict_smoke.py",
            "predict_external_action", "predict_report.json",
            ["--transition", transition, "--embeddings", embeddings, "--steps", 4, "--seed", 42])
        rows.extend(checked_metrics(case, transition, prediction))
        dump(directory / "endpoint_metrics.json", rows)
    result = {"schema": "thor-predict-validation-v1", "inputs": bindings, "summary": summarize(rows, bindings["plan"]),
              "embeddings": str(embeddings), "embeddings_sha256": sha256(embeddings),
              "diagnostic_complete": True, "independent_policy_gain_experiment": False,
              "complete_rsi": False, "production_promotion_performed": False, "real_robot_commands_sent": False}
    result["materials"] = export_materials(directory, rows, result)
    dump(directory / "predict_validation.json", result)
    dump(directory / "progress.json", {"phase": "completed_offline_diagnostic", "summary": result["summary"]})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--iteration-run", type=pathlib.Path, required=True)
    parser.add_argument("--after-reason-exec", type=pathlib.Path, required=True)
    parser.add_argument("--config", type=pathlib.Path, default=ROOT / "configs/predict_validation.yaml")
    parser.add_argument("--wait-timeout", type=int, default=86400)
    args = parser.parse_args()
    for name in ["iteration_run", "after_reason_exec", "config"]:
        value = getattr(args, name).resolve(strict=True)
        if not value.is_relative_to(ROOT):
            parser.error("Predict全部输入必须在本项目内")
        setattr(args, name, value)
    if ROOT != pathlib.Path("/home/thor/rsi") or args.wait_timeout < 1:
        parser.error("Predict只允许在Thor本项目运行，并使用正等待时长")
    bindings = bind_inputs(args)
    with (ROOT / "runs/.predict_validation.lock").open("a") as queue_lock:
        fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(json.dumps({"phase": "waiting_for_exact_reason_followup", "inputs": bindings}, ensure_ascii=False), flush=True)
        bindings["reason_completion"] = wait_reason(bindings, args.wait_timeout)
        with (ROOT / "runs/.rsi_iteration.lock").open("a") as model_lock:
            deadline = time.monotonic() + 30
            while True:
                try:
                    fcntl.flock(model_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise RuntimeError("其他整圈仍持有共享阶段锁，禁止抢占")
                    time.sleep(1)
            ensure_gpu_idle()
            return run_stage("p1_predict_validation", lambda directory: validate(directory, bindings),
                {"compute": "Thor模型阶段串行；等待不计入实验墙钟", "precision": "bf16", "batch": 1,
                 "diffusion_steps": 4, "sampling_seed": 42, "states": bindings["plan"]["states"],
                 "source_episodes": bindings["plan"]["source_episodes"], "authority": "offline_diagnostic_only"})


if __name__ == "__main__":
    def interrupted(number, _frame):
        raise RuntimeError(f"Predict多状态接续收到信号{number}，清理本次子任务")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    raise SystemExit(main())
