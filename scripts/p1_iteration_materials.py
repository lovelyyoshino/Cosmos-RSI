#!/usr/bin/env python3
"""完整首圈结束后导出真实训练、冻结对比和固定回合视频，拒绝用未完成结果制作收益素材。

@author pony
@date 2026-11-09
@version v1.0.0
@last_modified 2026-11-09
@changelog
    - v1.0.0 (2026-11-09): 绑定整圈封存记录，保留门禁拒绝；固定ID选片并明确回合结束卡。
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
import json
import math
import os
import pathlib
import subprocess
import sys
import time

from p0_analyze_evaluation import wilson95
from p0_common import ROOT, run_stage
from p0_lerobot_handoff import dump
from p0_video_materials import video_info
from p1_eval_gate import bound_file, checked_evaluation, checked_update, sealed, sha256


ROLES = ["incumbent", "candidate", "control"]
LABELS = {"incumbent": "当前策略", "candidate": "Cosmos与自身成功候选", "control": "D0复制对照",
          "analytic_ablation": "解析纯色消融", "standard": "标准", "target_perturbed": "目标扰动",
          "nontarget_perturbed": "非目标扰动"}


def training_trace(path, report):
    """日志不在阶段封存范围；导出时另存实际字节摘要，并拒绝缺步、重复步和非有限值。"""
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if len(rows) != report["steps"] or [r["step"] for r in rows] != list(range(1, report["steps"] + 1)):
        raise ValueError("训练日志没有完整、连续且唯一覆盖正式步数")
    for row in rows:
        if row["sample_slots"] != report["batch"] or any(type(row[key]) not in {int, float}
                or not math.isfinite(row[key]) or row[key] < 0 for key in ["loss", "grad_norm", "wall_s"]):
            raise ValueError("训练日志的数值或样本量不符")
    return rows


def fixed_video_cases(episodes, task="t00"):
    """每条件按共同冻结ID取首条；不读取成功标签，避免以选片替代完整对比。"""
    indexes = {role: {row["eval_id"]: row for row in episodes[role]} for role in ROLES}
    expected = set(indexes["incumbent"])
    if not expected or any(len(indexes[r]) != len(episodes[r]) or set(indexes[r]) != expected for r in ROLES):
        raise ValueError("三组回合必须有完整且相同的唯一冻结ID")
    pairs = []
    for condition in ["standard", "lighting", "background"]:
        choices = sorted(identity for identity, row in indexes["incumbent"].items()
                         if row["condition_id"] == f"{task}-{condition}")
        if not choices:
            raise ValueError("固定任务的预声明展示条件缺失")
        rows = [indexes[role][choices[0]] for role in ROLES]
        for key in ["initial_state_sha256", "init_seed", "policy_seed", "condition_id", "eval_set_sha256"]:
            if len({row[key] for row in rows}) != 1:
                raise ValueError("展示回合的冻结初态、条件或策略噪声错绑")
        pairs.append((condition, rows))
    return pairs


def write_csv(path, rows):
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_comparison(directory, condition, rows, evaluation_probes):
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    lengths = [row["metrics"]["steps"] for row in rows]
    videos, inputs = [], []
    for role, row in zip(ROLES, rows):
        report_path, probe = evaluation_probes[role]
        for view in ["agentview", "wrist"]:
            path = pathlib.Path(row["artifacts"]) / f"{view}.mp4"
            path = bound_file(report_path.parent, probe, str(path.relative_to(report_path.parent)))
            info = video_info(path)
            if info["frame_count"] != row["metrics"]["steps"] or (info["width"], info["height"]) != (256, 256):
                raise ValueError("原始视频帧数或尺寸偏离实际评测")
            videos.append(path)
            inputs.append({"role": role, "path": str(path), "sha256": sha256(path), **info})
    font_path = subprocess.check_output(["fc-match", "-f", "%{file}", "Noto Sans CJK SC"], text=True)
    fonts = {size: ImageFont.truetype(font_path, size) for size in [20, 24, 30]}
    path = directory / f"iteration_{condition}.mp4"
    count = max(lengths)
    screenshots = []
    with ExitStack() as stack:
        readers = [stack.enter_context(imageio.get_reader(video)) for video in videos]
        writer = stack.enter_context(imageio.get_writer(path, fps=20, codec="libx264", pixelformat="yuv420p",
            macro_block_size=1, ffmpeg_params=["-crf", "18", "-threads", "2", "-movflags", "+faststart"]))
        for frame in range(count):
            canvas = Image.new("RGB", (1920, 560), "#f5f4ef")
            draw = ImageDraw.Draw(canvas)
            draw.text((15, 10), f"正式策略对比 / {condition} / 固定回合 {rows[0]['eval_id']}", font=fonts[30], fill="#20313c")
            draw.text((15, 53), f"同初态 {rows[0]['init_seed']} / 策略种子 {rows[0]['policy_seed']} / t={frame/20:.2f}s / 20Hz仿真时钟", font=fonts[20], fill="#45545d")
            for column, (role, row, length) in enumerate(zip(ROLES, rows, lengths)):
                x = column * 640
                outcome = "成功" if row["outcome"]["success"] else "失败"
                draw.text((x + 12, 96), f"{LABELS[role]} / 环境：{outcome}", font=fonts[24], fill="#20313c")
                if frame < length:
                    for view in range(2):
                        image = Image.fromarray(readers[column * 2 + view].get_data(frame))
                        canvas.paste(image.resize((320, 320), Image.Resampling.LANCZOS), (x + view * 320, 136))
                else:
                    draw.rectangle((x, 136, x + 639, 455), fill="#e6e8e4")
                    draw.text((x + 25, 244), f"回合已结束：{outcome}，实际{length}步", font=fonts[24], fill="#20313c")
                    draw.text((x + 25, 285), "结束卡不是后续机器人观测或动作", font=fonts[20], fill="#45545d")
                draw.text((x + 10, 465), "外部相机                       腕部相机", font=fonts[20], fill="#45545d")
            draw.text((15, 501), "每条件按ID取首回合，不按成败选片；全量结果见CSV和冻结门禁。展示视频禁止入训。", font=fonts[20], fill="#6c4537")
            draw.text((15, 530), "Thor / 既有MAXN / VLM bf16、LoRA与动作头fp32 / sdpa / 推理batch1 / 动作块8", font=fonts[20], fill="#45545d")
            if frame in {0, min(80, count - 1), count - 1}:
                screenshot = directory / f"iteration_{condition}_{frame:03d}.png"
                canvas.save(screenshot)
                screenshots.append({"file": screenshot.name, "frame": frame, "sha256": sha256(screenshot)})
            writer.append_data(np.asarray(canvas))
    output = video_info(path)
    if output["frame_count"] != count or (output["width"], output["height"]) != (1920, 560) \
            or output["codec_name"] != "h264" or output["pix_fmt"] != "yuv420p":
        raise ValueError("正式对比视频的实际解码、尺寸或时序不符")
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=60)
    return {"file": path.name, "eval_id": rows[0]["eval_id"], "inputs": inputs, "output": output,
            "source_lengths": lengths, "screenshots": screenshots, "sha256": sha256(path),
            "selection": "固定t00，每条件按ID字典序首回合，不读取成败选片",
            "frame_mapping": "每列来自同一实际输入帧；回合结束后改为明确结束卡，不循环或生成动作",
            "eligible_for_training": False}


def collect(directory, args):
    iteration, _ = sealed(args.iteration_run / "iteration.json", "rsi_iteration")
    gate_path = pathlib.Path(iteration["gate"])
    gate, _ = sealed(gate_path, "p1_evaluation_gate")
    if sha256(gate_path) != iteration["gate_sha256"] or gate["iteration"] != iteration["iteration"] \
            or gate["decision"] != iteration["decision"]:
        raise ValueError("整圈与真实冻结门禁不一致")

    def stage(name, expected):
        binding = iteration["stages"][name]
        path = pathlib.Path(binding["path"])
        if binding["stage"] != expected or sha256(path) != binding["sha256"]:
            raise ValueError("整圈阶段身份或封存摘要改变")
        return path

    traces, training, sources = {}, {}, {}
    for role in ["candidate", "control", "analytic_ablation"]:
        path = stage("train_" + role, "p1_policy_train")
        report, _ = checked_update(path, role)
        if role != "analytic_ablation" and gate["inputs"][role + "_training"]["sha256"] != sha256(path):
            raise ValueError("展示训练不是正式门禁比较的同一权重")
        log = path.parent / "logs/steps.jsonl"
        traces[role] = training_trace(log, report)
        training[role] = report
        sources[role] = {"training": str(path), "sha256": sha256(path), "log": str(log), "log_sha256": sha256(log)}
    if len({r["paired_plan_sha256"] for r in training.values()}) != 1:
        raise ValueError("三角色训练计划不一致")
    write_csv(directory / "training_steps.csv", [{"role": role, **row} for role, rows in traces.items() for row in rows])
    write_csv(directory / "training_summary.csv", [{"role": role, **{key: report[key] for key in
        ["steps", "batch", "seed", "training_wall_s", "reload_max_abs_diff", "output_ckpt_sha256"]}}
        for role, report in training.items()])

    import yaml
    budget_path = pathlib.Path(iteration["inputs"]["budget"]["path"])
    if sha256(budget_path) != iteration["inputs"]["budget"]["sha256"]:
        raise ValueError("整圈运行后的预声明预算改变")
    budget = yaml.safe_load(budget_path.read_text())
    frozen_path = ROOT / budget["evaluation"]["definition"]
    frozen = {**json.loads(frozen_path.read_text()), "sha256": sha256(frozen_path)}
    episodes, evaluations, probes = {}, {}, {}
    for role in ROLES:
        path = pathlib.Path(gate["inputs"][role + "_evaluation"]["path"])
        if sha256(path) != gate["inputs"][role + "_evaluation"]["sha256"]:
            raise ValueError("门禁使用的实际评测改变")
        expected = gate["parent_ckpt_sha256"] if role == "incumbent" else training[role]["output_ckpt_sha256"]
        report, _ = checked_evaluation(path, expected, frozen)
        _, probe = sealed(path, "policy_evaluation")
        rows_path = bound_file(path.parent, probe, "episodes.jsonl")
        episodes[role] = [json.loads(line) for line in rows_path.read_text().splitlines()]
        evaluations[role] = report
        probes[role] = (path, probe)
    comparison = [{"role": role, "split": split, **counts,
                   "wilson95_low": wilson95(counts["success"], counts["n"])[0],
                   "wilson95_high": wilson95(counts["success"], counts["n"])[1]}
                  for role in ROLES for split, counts in evaluations[role]["splits"].items()]
    write_csv(directory / "frozen_comparison.csv", comparison)
    write_csv(directory / "frozen_episodes.csv", [{"role": role, **{key: row[key] for key in
        ["eval_id", "condition_id", "split", "init_seed", "policy_seed"]}, "success": row["outcome"]["success"]}
        for role in ROLES for row in episodes[role]])
    plot(directory, traces, training, comparison, gate)
    clips = [render_comparison(directory, condition, rows, probes) for condition, rows in fixed_video_cases(episodes)]
    result = {"schema": "thor-p1-iteration-materials-v1", "iteration": str(args.iteration_run),
        "iteration_sha256": sha256(args.iteration_run / "iteration.json"), "gate": str(gate_path),
        "gate_sha256": sha256(gate_path), "decision": gate["decision"], "training_sources": sources,
        "clips": clips, "AI_generated_experiment_results": False, "presentation_created": False,
        "complete_rsi": False, "production_promotion_performed": False,
        "scope": "真实首圈训练和完整冻结对比；解析消融只训练未评测；不能据此声称Cosmos独立贡献或连续多圈RSI",
        "files_sha256": {p.name: sha256(p) for p in directory.iterdir() if p.is_file()}}
    dump(directory / "materials_manifest.json", result)
    return {"decision": gate["decision"], "clips": len(clips), "complete_rsi": False}


def plot(directory, traces, training, comparison, gate):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties

    font = subprocess.check_output(["fc-match", "-f", "%{file}", "Noto Sans CJK SC"], text=True)
    plt.rcParams.update({"font.family": FontProperties(fname=font).get_name(), "font.size": 10})
    figure, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    for role, rows in traces.items():
        axes[0].plot([r["step"] for r in rows], [r["loss"] for r in rows], lw=.7, alpha=.8, label=LABELS[role])
    axes[0].set(title="正式同预算训练：完整原始曲线", xlabel="增量训练步", ylabel="动作损失")
    splits = ["standard", "target_perturbed", "nontarget_perturbed"]
    for offset, role in enumerate(ROLES):
        values = [next(r for r in comparison if r["role"] == role and r["split"] == split) for split in splits]
        centers = [i + (offset - 1) * .24 for i in range(3)]
        heights = [100 * r["p"] for r in values]
        errors = [[100 * (r["p"] - r["wilson95_low"]) for r in values],
                  [100 * (r["wilson95_high"] - r["p"]) for r in values]]
        axes[1].bar(centers, heights, width=.23, yerr=errors, capsize=3, label=LABELS[role])
        for x, height, row in zip(centers, heights, values):
            axes[1].text(x, height + 1, f"{row['success']}/{row['n']}", ha="center", fontsize=8)
    axes[1].set(title=f"完整冻结评测 / 门禁：{gate['decision']}", ylabel="环境成功率（%）",
                xticks=range(3), xticklabels=[LABELS[s] for s in splits], ylim=(0, 105))
    for axis in axes:
        axis.legend(fontsize=9)
        axis.grid(axis="y", alpha=.18)
        axis.set_axisbelow(True)
    conditions = training["candidate"]
    figure.suptitle(f"Thor / 既有MAXN / VLM bf16、LoRA与动作头fp32 / sdpa / 各{conditions['steps']}步 / batch{conditions['batch']} / seed{conditions['seed']}")
    figure.supxlabel("误差线为回合Wilson95%描述性区间，同初态回合相关；聚类分析另存。损失不证明收益，解析消融尚无成功率评测。", fontsize=9)
    for extension in ["png", "svg"]:
        figure.savefig(directory / ("iteration_comparison." + extension), dpi=240)
    plt.close(figure)


def wait_for_iteration(path, timeout):
    started = time.monotonic()
    while not (path / "probe.json").exists():
        if time.monotonic() - started >= timeout:
            raise TimeoutError("等待指定整圈结束超时；没有停止或重复启动实验")
        time.sleep(min(30, max(.01, timeout - (time.monotonic() - started))))
    probe = json.loads((path / "probe.json").read_text())
    if probe["stage"] != "rsi_iteration" or probe["status"] != "passed":
        raise ValueError("整圈工程阶段未通过；保留原始失败，不以部分回合生成完整对比")


if __name__ == "__main__":
    material_env = ROOT / "envs/sim"
    if ROOT == pathlib.Path("/home/thor/rsi") and pathlib.Path(sys.prefix).resolve() != material_env.resolve():
        executable = str(material_env / "bin/python")
        print("CPU素材阶段使用envs/sim；禁用CUDA，不启动策略服务。", flush=True)
        os.execve(executable, [executable, "-u", str(pathlib.Path(__file__).resolve()), *sys.argv[1:]],
                  {**os.environ, "CUDA_VISIBLE_DEVICES": "", "MPLBACKEND": "Agg", "OPENBLAS_NUM_THREADS": "1"})
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--iteration-run", type=pathlib.Path, required=True)
    parser.add_argument("--wait-timeout", type=int, default=1)
    args = parser.parse_args()
    args.iteration_run = args.iteration_run.resolve(strict=True)
    if not args.iteration_run.is_relative_to(ROOT / "runs/p0") or args.wait_timeout < 1:
        parser.error("仅接受本项目实际整圈目录及正等待时长")
    wait_for_iteration(args.iteration_run, args.wait_timeout)
    raise SystemExit(run_stage("p1_iteration_materials", lambda directory: collect(directory, args),
        {"compute": "Thor CPU/Agg/Pillow/ffmpeg，编码threads2；无模型运行", "sample_count": None}))
