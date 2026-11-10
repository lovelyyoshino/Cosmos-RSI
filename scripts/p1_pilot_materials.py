#!/usr/bin/env python3
"""保守开发试验封存后导出真实配对图表和视频，不授予正式评测或晋升资格。

@author pony
@date 2026-11-09
@version v1.0.0
@last_modified 2026-11-09
@changelog
    - v1.0.0 (2026-11-09): 绑定确切试验、完整训练与同初态开发回合，导出固定三条件展示。
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import os
import pathlib
import subprocess
import time

from p0_common import ROOT, run_stage
from p0_feedback_judge import checked_collection
from p0_lerobot_handoff import dump
from p0_video_materials import video_info
from p1_conservative_pilot import CONDITIONS, INITIAL_SEED, POLICY_SEED, validate_development
from p1_eval_gate import bound_file, sealed, sha256
from p1_iteration_materials import ROLES, training_trace, write_csv
from p1_workflow_status import logged_run, read_process


TRAINING_ROLES = ["candidate", "control", "analytic_ablation"]
LABELS = {"incumbent": "原策略", "candidate": "保守候选", "control": "D0复制对照",
          "analytic_ablation": "解析外观消融"}


def project(value):
    path = (ROOT / value).resolve(strict=True)
    if not path.is_relative_to(ROOT):
        raise ValueError("开发素材输入必须在Thor本项目内")
    return path


def check_authority(report):
    expected = {"development_only": True, "formal_gate_executed": False,
                "production_promotion": False, "real_robot_enabled": False}
    if report.get("schema") != "thor-conservative-pilot-v1" or report.get("authority") != expected \
            or report.get("complete_rsi") is not False or report.get("causal_attribution_established") is not False \
            or set(report["training"]) != set(TRAINING_ROLES) or set(report["development"]) != set(ROLES):
        raise ValueError("只导出完整的保守开发试验，不能将它当作正式门禁或RSI收益")


def paired_cases(episodes):
    """按预声明条件和首个初态选片；不读取成败标签来挑选素材。"""
    indexes = {}
    for role in ROLES:
        rows = episodes[role]
        index = {(row["condition_id"], row["init_seed"], row["policy_seed"]): row for row in rows}
        if not index or len(index) != len(rows):
            raise ValueError("开发回合缺失或包含重复条件与种子")
        indexes[role] = index
    expected = set(indexes["incumbent"])
    if any(set(indexes[role]) != expected for role in ROLES):
        raise ValueError("三策略开发回合不是同一组条件与种子")
    for key in expected:
        if len({indexes[role][key]["initial_scene_sha256"] for role in ROLES}) != 1:
            raise ValueError("配对开发回合的实际初始物理场景不同")
    result = []
    for condition in CONDITIONS:
        key = (condition, INITIAL_SEED, POLICY_SEED)
        if key not in expected:
            raise ValueError("预声明的首个展示初态不存在，禁止用其他回合替代")
        result.append((condition, [indexes[role][key] for role in ROLES]))
    return result


def wait_for_pilot(execution, timeout):
    launch_path = project(execution / "launch.json")
    launch = json.loads(launch_path.read_text())
    digest = sha256(launch_path)
    command = launch["command"]
    if type(launch.get("pid")) is not int or launch["pid"] < 1 \
            or not any(value.endswith("scripts/p1_conservative_pilot.py") for value in command):
        raise ValueError("素材必须等待确切保守试验的正整数PID与实际命令")
    log = project(launch["log"])
    if not log.is_relative_to(execution):
        raise ValueError("启动日志不属于绑定的确切执行目录")
    print(json.dumps({"phase": "waiting_for_exact_pilot_cpu_materials", "execution": str(execution),
        "launch_sha256": digest, "pid": launch["pid"], "model_loaded": False,
        "inputs": {"files": {"pilot_launch": {"path": str(launch_path), "sha256": digest}}}},
        ensure_ascii=False), flush=True)
    deadline = time.monotonic() + timeout
    while True:
        if sha256(launch_path) != digest:
            raise ValueError("等待期间保守试验的启动记录发生变化")
        run = logged_run(ROOT, log)
        if run and (run / "probe.json").exists():
            report, _ = sealed(run / "pilot.json", "p1_conservative_pilot")
            check_authority(report)
            return run
        if read_process(launch["pid"]) != command:
            raise RuntimeError("确切保守试验已退出或PID身份变化，保留失败，不按最新目录猜测")
        if time.monotonic() >= deadline:
            raise TimeoutError("等待保守试验超时，未停止或重启任何模型任务")
        time.sleep(min(5., max(.01, deadline - time.monotonic())))


def checked_training(path, role):
    report, probe = sealed(path, "p1_policy_train")
    if report.get("schema") != "thor-feedback-policy-update-v1" or report.get("role") != role \
            or report.get("development_only") is not True or report.get("training_performed") is not True \
            or report.get("eligible_for_evaluation") is not False or report.get("eligible_for_promotion") is not False \
            or not 0 <= report["reload_max_abs_diff"] <= 1e-5:
        raise ValueError("开发训练的角色、重载或权限边界不符")
    checkpoint = project(report["output_ckpt"])
    relative = str(checkpoint.relative_to(path.parent))
    metadata = json.loads(bound_file(path.parent, probe, relative + "/checkpoint.json").read_text())
    weights = bound_file(path.parent, probe, relative + "/state.pt")
    if metadata.get("development_only") is not True or metadata.get("formal_policy_update") is not False \
            or metadata.get("eligible_for_promotion") is not False \
            or metadata["state_sha256"] != report["output_ckpt_sha256"] \
            or sha256(weights) != report["output_ckpt_sha256"] \
            or any(metadata[key] != report[key] for key in ["recipe", "role", "parent_ckpt_sha256", "paired_plan_sha256"]):
        raise ValueError("开发训练与实际检查点不一致")
    trace = path.parent / "logs/steps.jsonl"
    return report, probe, training_trace(trace, report), trace


def render_clip(directory, condition, rows, sources):
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    font_path = subprocess.check_output(["fc-match", "-f", "%{file}", "Noto Sans CJK SC"], text=True)
    fonts = {size: ImageFont.truetype(font_path, size) for size in [20, 24, 30]}
    inputs = []
    for role, row in zip(ROLES, rows):
        path, probe = sources[role]
        for view in ["agentview", "wrist"]:
            video = project(pathlib.Path(row["artifacts"]) / (view + ".mp4"))
            video = bound_file(path.parent, probe, str(video.relative_to(path.parent)))
            info = video_info(video)
            if info["frame_count"] != row["metrics"]["steps"] or (info["width"], info["height"]) != (256, 256):
                raise ValueError("开发视频与真实步数或相机尺寸不同")
            inputs.append({"role": role, "view": view, "path": str(video), "sha256": sha256(video), **info})
    output = directory / ("pilot_" + condition.removeprefix("t00-") + ".mp4")
    length = max(row["metrics"]["steps"] for row in rows)
    screenshots = []
    with ExitStack() as stack:
        readers = [stack.enter_context(imageio.get_reader(item["path"])) for item in inputs]
        writer = stack.enter_context(imageio.get_writer(output, fps=20, codec="libx264", pixelformat="yuv420p",
            macro_block_size=1, ffmpeg_params=["-crf", "18", "-threads", "2", "-movflags", "+faststart"]))
        for frame in range(length):
            canvas = Image.new("RGB", (1920, 560), "#F4F7F7")
            draw = ImageDraw.Draw(canvas)
            draw.text((15, 8), f"保守开发试验 / {condition} / 同初态{INITIAL_SEED} / 策略种子{POLICY_SEED}", font=fonts[30], fill="#20313C")
            draw.text((15, 51), f"t={frame / 20:.2f}s（仿真时钟） / 固定首回合，不按成败选片 / 这是开发诊断，尚未正式验收", font=fonts[20], fill="#52616B")
            for column, (role, row) in enumerate(zip(ROLES, rows)):
                x = column * 640
                outcome = "成功" if row["outcome"]["success"] else "失败"
                draw.text((x + 12, 96), f"{LABELS[role]} / 环境：{outcome}", font=fonts[24], fill="#20313C")
                if frame < row["metrics"]["steps"]:
                    for view in range(2):
                        pixels = Image.fromarray(readers[column * 2 + view].get_data(frame))
                        canvas.paste(pixels.resize((320, 320), Image.Resampling.LANCZOS), (x + view * 320, 136))
                else:
                    draw.rectangle((x, 136, x + 639, 455), fill="#E3E9E8")
                    draw.text((x + 20, 245), f"回合已结束：{outcome}，实际{row['metrics']['steps']}步", font=fonts[24], fill="#20313C")
                    draw.text((x + 20, 286), "结束卡不表示后续机器人观测或动作", font=fonts[20], fill="#52616B")
                draw.text((x + 10, 464), "外部相机                       腕部相机", font=fonts[20], fill="#52616B")
            draw.text((15, 500), "Thor / 既有功耗见来源 / VLM bf16、LoRA与动作头fp32 / sdpa / 推理batch1 / 动作块8", font=fonts[20], fill="#52616B")
            draw.text((15, 530), "父初训5000步；开发各250步、batch8、seed42；小样本不证明增益，未执行正式门禁或晋升。", font=fonts[20], fill="#98602C")
            if frame in {0, min(80, length - 1), length - 1}:
                screenshot = output.with_name(output.stem + f"_{frame:03d}.png")
                canvas.save(screenshot)
                screenshots.append({"file": screenshot.name, "sha256": sha256(screenshot), "frame": frame})
            writer.append_data(np.asarray(canvas))
    info = video_info(output)
    if info["frame_count"] != length or (info["width"], info["height"]) != (1920, 560) \
            or info["codec_name"] != "h264" or info["pix_fmt"] != "yuv420p":
        raise ValueError("导出的开发视频帧数、尺寸或编码不符")
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(output), "-f", "null", "-"],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=60)
    return {"file": output.name, "condition": condition, "sha256": sha256(output), "inputs": inputs,
            "output": info, "screenshots": screenshots, "eligible_for_training": False,
            "selection": "按预声明条件及首个初态/策略种子选片，不按成败选择"}


def plot(directory, traces, counts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties

    font = subprocess.check_output(["fc-match", "-f", "%{file}", "Noto Sans CJK SC"], text=True)
    plt.rcParams.update({"font.family": FontProperties(fname=font).get_name(), "font.size": 12})
    figure, axes = plt.subplots(1, 2, figsize=(16, 8))
    colors = ["#217C85", "#697C91", "#BC7A2C"]
    for role, color in zip(TRAINING_ROLES, colors):
        axes[0].plot([row["step"] for row in traces[role]], [row["loss"] for row in traces[role]],
                     lw=.8, color=color, label=LABELS[role])
    axes[0].set(title="保守续训：完整原始损失", xlabel="开发增量训练步", ylabel="动作损失")
    for index, (role, color) in enumerate(zip(ROLES, colors)):
        selected = [next(row for row in counts if row["role"] == role and row["condition"] == condition) for condition in CONDITIONS]
        x = [offset + (index - 1) * .25 for offset in range(3)]
        bars = axes[1].bar(x, [100 * row["success"] / row["n"] for row in selected], .24, color=color, label=LABELS[role])
        axes[1].bar_label(bars, labels=[f"{row['success']}/{row['n']}" for row in selected], padding=3, fontsize=10)
    axes[1].set(title="独立开发回合：环境真值，尚非正式门禁", ylabel="成功率（%）", ylim=(0, 115),
                xticks=range(3), xticklabels=["标准", "光照", "背景"])
    for axis in axes:
        axis.legend(fontsize=10)
        axis.grid(axis="y", alpha=.16)
        axis.set_axisbelow(True)
    figure.suptitle("Cosmos × Thor：恢复优化器、降低学习率与增加D0重放的开发试验", fontsize=20)
    figure.subplots_adjust(top=.84, bottom=.31, wspace=.3)
    notes = ["Thor / 既有功耗见来源 / VLM bf16、LoRA与动作头fp32 / sdpa / 无DeepSpeed",
        "同一5000步父策略；各250步 / batch8 / seed42 / 每角色2000个有放回槽位；不是2000条独立示范。",
        "开发推理batch1 / 动作块8 / 每回合最多520步；每条件3条、跨条件复用3个初态，各策略共9条回合。",
        f"初态{INITIAL_SEED}–{INITIAL_SEED + 2} / 策略种子{POLICY_SEED}–{POLICY_SEED + 2}；环境成败以实际仿真为准。",
        "多个训练因素同时改变；单训练种子、小样本开发诊断不能证明因果、Cosmos独立贡献或连续RSI收益。",
        "未执行冻结门禁、生产晋升或真机动作；保留旧正式拒绝结果。解析消融只训练，未采集成功率。"]
    for index, note in enumerate(notes):
        figure.text(.06, .235 - index * .032, note, fontsize=9.5, color="#52616B")
    for suffix in ["png", "svg"]:
        figure.savefig(directory / ("conservative_pilot." + suffix), dpi=240)
    plt.close(figure)


def collect(directory, run):
    report, _ = sealed(run / "pilot.json", "p1_conservative_pilot")
    check_authority(report)
    sources, traces, training, summaries, policies = {}, {}, {}, [], {}
    for role in TRAINING_ROLES:
        binding = report["training"][role]
        path = project(binding["path"])
        if sha256(path) != binding["sha256"]:
            raise ValueError("保守开发试验绑定的真实训练已经改变")
        train, probe, traces[role], log = checked_training(path, role)
        if train["steps"] != 250 or train["batch"] != 8 or train["seed"] != 42 \
                or train["paired_plan_sha256"] != report["inputs"]["paired_plan_sha256"] \
                or train["parent_ckpt_sha256"] != report["inputs"]["parent_ckpt_sha256"]:
            raise ValueError("保守开发试验的实际训练预算、父权重或抽样计划不符")
        training[role] = train
        policies[role] = train["output_ckpt_sha256"]
        sources[role] = {**binding, "probe_sha256": sha256(path.parent / "probe.json"),
                         "steps_log": str(log), "steps_log_sha256": sha256(log)}
        summaries.append({"role": role, "steps": train["steps"], "batch": train["batch"], "seed": train["seed"],
            "training_wall_s": train["training_wall_s"], "full_stage_wall_s": probe["wall_s"],
            "whole_host_peak_ram_gb": probe["peak_mem_gb"], "precision": train["precision"],
            "hardware": json.dumps(probe["hardware"], ensure_ascii=False), "recipe": json.dumps(train["recipe"], ensure_ascii=False)})
    for key in ["recipe", "precision", "attention", "parent_ckpt_sha256", "paired_plan_sha256"]:
        if any(training[role][key] != training["candidate"][key] for role in TRAINING_ROLES):
            raise ValueError("开发三角色的训练条件不匹配")
    policies["incumbent"] = report["inputs"]["parent_ckpt_sha256"]
    episodes, collections, counts = {}, {}, []
    for role in ROLES:
        binding = report["development"][role]
        path = project(binding["path"])
        if sha256(path) != binding["sha256"]:
            raise ValueError("保守开发试验绑定的真实采集已经改变")
        collection, rows, probe = checked_collection(path)
        actual = validate_development(rows, policies[role])
        if actual != binding["counts"] or collection["policy_ckpt_sha256"] != policies[role]:
            raise ValueError("开发成功计数、隔离边界或策略与实际回合不符")
        episodes[role], collections[role] = rows, (path, probe)
        sources["development_" + role] = {**binding, "probe_sha256": sha256(path.parent / "probe.json")}
        counts.extend({"role": role, "condition": condition, **value} for condition, value in actual.items())
    cases = paired_cases(episodes)
    write_csv(directory / "training_steps.csv", [{"role": role, **row} for role in TRAINING_ROLES for row in traces[role]])
    write_csv(directory / "training_summary.csv", summaries)
    write_csv(directory / "development_counts.csv", counts)
    write_csv(directory / "development_episodes.csv", [{"role": role, **{key: row[key] for key in
        ["episode_id", "condition_id", "init_seed", "policy_seed", "initial_scene_sha256", "policy_ckpt_sha256"]},
        "success": row["outcome"]["success"], "steps": row["metrics"]["steps"]} for role in ROLES for row in episodes[role]])
    plot(directory, traces, counts)
    clips = [render_clip(directory, condition, rows, collections) for condition, rows in cases]
    result = {"schema": "thor-conservative-pilot-materials-v1", "pilot_run": str(run),
        "pilot_sha256": sha256(run / "pilot.json"), "sources": sources, "counts": counts, "clips": clips,
        "authority": report["authority"], "new_model_inference_performed": False, "eligible_for_training": False,
        "causal_attribution_established": False, "complete_rsi": False, "presentation_created": False,
        "AI_generated_experiment_results": False,
        "files_sha256": {p.name: sha256(p) for p in directory.iterdir() if p.is_file()}}
    dump(directory / "materials_manifest.json", result)
    return {"development_export_complete": True, "clips": len(clips), "complete_rsi": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pilot-run", type=pathlib.Path)
    group.add_argument("--after-pilot-exec", type=pathlib.Path)
    parser.add_argument("--wait-timeout", type=int, default=86400)
    args = parser.parse_args()
    if ROOT != pathlib.Path("/home/thor/rsi") or os.environ.get("CUDA_VISIBLE_DEVICES") != "" or args.wait_timeout < 1:
        parser.error("只在Thor本项目用CPU导出，须显式禁用CUDA且等待时长为正")
    implementation = sha256(pathlib.Path(__file__))
    run = project(args.pilot_run) if args.pilot_run else wait_for_pilot(project(args.after_pilot_exec), args.wait_timeout)
    if sha256(pathlib.Path(__file__)) != implementation:
        raise ValueError("等待期间素材实现发生变化，须另建明确导出任务")
    # 沿用现有video_materials备份规则，只导出固定少量展示视频，不复制原始回合数据。
    return run_stage("video_materials_conservative_pilot", lambda directory: collect(directory, run),
                     {"compute": "Thor CPU导出完整保守开发试验，CUDA禁用，无新训练推理或运动",
                      "precision": "展示导出不适用；源实验精度、batch、步数、种子及功耗另存CSV",
                      "batch": None, "seed": None})


if __name__ == "__main__":
    raise SystemExit(main())
