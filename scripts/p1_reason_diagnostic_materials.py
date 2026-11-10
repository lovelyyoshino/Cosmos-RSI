#!/usr/bin/env python3
"""从封存独立反馈导出Reason校准诊断，保留全部误判而不运行新模型。

@author pony
@date 2026-11-09
@version v1.0.1
@last_modified 2026-11-09
@changelog
    - v1.0.1 (2026-11-09): 诊断页用中文判定标签，英文证据按词换行。
    - v1.0.0 (2026-11-09): 核对完整回合、真值与六图绑定，导出混淆矩阵及固定成败输入。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import subprocess
import sys
import textwrap

from p0_background_preview_materials import checked_artifact
from p0_common import ROOT, run_stage
from p0_feedback_judge import calibration_metrics, checked_collection
from p0_lerobot_handoff import dump
from p1_eval_gate import sha256


def audit_rows(episodes, judgments, report):
    """每个完整采集回合只计一次，拒绝错绑真值或仅挑模型答对的子集。"""
    expected = {row["episode_id"]: row for row in episodes}
    actual = {row["episode_id"]: row for row in judgments}
    if len(expected) != len(episodes) or len(actual) != len(judgments) or set(actual) != set(expected):
        raise ValueError("判官回合不等于完整独立采集")
    for identity, row in actual.items():
        episode = expected[identity]
        if type(row["environment_truth"]) is not bool \
                or row["environment_truth"] != episode["outcome"]["success"] \
                or episode["outcome"]["source"] != "env" \
                or row["policy_ckpt_sha256"] != episode["policy_ckpt_sha256"] \
                or row["is_eval"] is not False or row["truth_hidden_from_model"] is not True:
            raise ValueError("判官真值或策略身份与采集不一致")
        phases = [(item["phase"], item["view"]) for item in row["inputs"]]
        if phases != [(phase, view) for phase in ["initial", "middle", "terminal"]
                      for view in ["agentview", "wrist"]]:
            raise ValueError("实际判官输入不是预声明的六张有序关键图")
    metrics = calibration_metrics(judgments, report["config"])
    if len(judgments) != report["sample_count"] or metrics != report["calibration"]:
        raise ValueError("重新计数与封存校准报告不一致")
    return metrics


def collect(directory, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from PIL import Image, ImageDraw, ImageFont

    probe = checked_artifact(args.judge, "feedback_judge")
    report = json.loads(args.judge.read_text())
    collection = pathlib.Path(report["collection"]).resolve(strict=True)
    if sha256(collection) != report["collection_sha256"]:
        raise ValueError("判官没有绑定当前独立反馈")
    source, episodes, _ = checked_collection(collection)
    verdicts = pathlib.Path(report["verdicts"]).resolve(strict=True)
    if not verdicts.is_relative_to(args.judge.parent) \
            or probe["artifacts_sha256"].get(str(verdicts.relative_to(args.judge.parent))) != sha256(verdicts):
        raise ValueError("实际判官逐回合记录未封存")
    rows = [json.loads(line) for line in verdicts.read_text().splitlines()]
    metrics = audit_rows(episodes, rows, report)
    for row in rows:
        for item in row["inputs"]:
            path = pathlib.Path(item["path"]).resolve(strict=True)
            if not path.is_relative_to(args.judge.parent) or sha256(path) != item["sha256"] \
                    or probe["artifacts_sha256"].get(str(path.relative_to(args.judge.parent))) != item["sha256"]:
                raise ValueError("判官实际输入图像已变化或越界")
    font_file = subprocess.check_output(["fc-match", "-f", "%{file}", "Noto Sans CJK SC"], text=True)
    plt.rcParams.update({"font.family": FontProperties(fname=font_file).get_name(), "font.size": 11})
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    matrix = [[metrics["true_positive"], metrics["false_negative"]],
              [metrics["false_positive"], metrics["true_negative"]]]
    axes[0].imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, len(rows)))
    for y, values in enumerate(matrix):
        for x, value in enumerate(values):
            axes[0].text(x, y, str(value), ha="center", va="center", fontsize=28,
                         color="white" if value > len(rows) / 2 else "#21323b")
    axes[0].set(xticks=[0, 1], xticklabels=["判成功", "判失败"], yticks=[0, 1],
                yticklabels=["环境成功", "环境失败"], title="完整回合混淆矩阵；弃权另计")
    names = ["准确率", "成功召回", "失败召回"]
    values = [metrics[key] for key in ["accuracy", "success_recall", "failure_recall"]]
    heights = [100 * value if value is not None else 0 for value in values]
    axes[1].bar(names, heights, color=["#526e7e", "#497d42", "#b46a32"])
    axes[1].axhline(100 * report["config"]["min_accuracy"], ls="--", color="#8c4141", label="预声明门槛")
    for index, value in enumerate(values):
        axes[1].text(index, heights[index] + 2, "缺类别" if value is None else f"{value:.1%}", ha="center")
    axes[1].set(ylim=(0, 112), ylabel="比例（%）", title="校准通过" if metrics["calibration_passed"] else "校准未通过")
    axes[1].legend()
    figure.suptitle(f"{report['model_name']} / {report['precision']} / CUDA Graph / batch{report['batch']} / seed{report['seed']}")
    figure.supxlabel(f"Thor / 既有功耗模式见来源 / {len(rows)}回合 / {report['unique_init_seeds']}个初态种子 / 最多{report['max_new_tokens']}输出token\n重复初态跨条件相关；自报置信度不是校准概率。没有新模型推理或策略收益实验。", fontsize=9)
    for extension in ["png", "svg"]:
        figure.savefig(directory / ("reason_calibration." + extension), dpi=220)
    plt.close(figure)
    flat = [{"episode_id": row["episode_id"], "environment_success": row["environment_truth"],
             "judge_success": row["verdict"]["success"], "valid": row.get("valid", True),
             "confidence": row["verdict"]["confidence"], "evidence": row["verdict"]["evidence"]} for row in rows]
    with (directory / "reason_verdicts.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)

    # 诊断页显式按环境成败各取最小ID，不能把这种选片当训练前后收益比较。
    selected = [min((row for row in rows if row["environment_truth"] is outcome),
                    key=lambda row: row["episode_id"], default=None) for outcome in [False, True]]
    selected = [row for row in selected if row is not None]
    canvas = Image.new("RGB", (1920, 180 + 690 * len(selected)), "#f6f8f8")
    draw = ImageDraw.Draw(canvas); fonts = {size: ImageFont.truetype(font_file, size) for size in [20, 24, 32]}
    draw.text((32, 22), "Reason实际六图输入：按环境失败/成功各取最小ID", font=fonts[32], fill="#20313b")
    draw.text((32, 78), "仅用于误判诊断；每列为同一时刻的外部及腕部相机，完整统计见CSV。", font=fonts[24], fill="#4d626c")
    for index, row in enumerate(selected):
        y = 140 + 690 * index
        draw.text((32, y), row["episode_id"], font=fonts[20], fill="#20313b")
        answer = row["verdict"]["success"]
        label = "成功" if answer is True else "失败" if answer is False else "弃权"
        draw.text((32, y + 34), f"环境：{'成功' if row['environment_truth'] else '失败'}；Reason：{label}；自报置信度：{row['verdict']['confidence']}", font=fonts[24], fill="#9c542a")
        for phase in range(3):
            x = 32 + phase * 620
            draw.text((x, y + 80), ["初始", "中段", "最终动作后"][phase], font=fonts[24], fill="#20313b")
            for view in range(2):
                item = row["inputs"][phase * 2 + view]
                with Image.open(item["path"]) as image:
                    canvas.paste(image.convert("RGB").resize((280, 280)), (x + view * 290, y + 125))
            draw.text((x, y + 420), "外部相机                   腕部相机", font=fonts[20], fill="#4d626c")
        evidence = row["verdict"]["evidence"]
        for line, value in enumerate(textwrap.wrap(evidence, width=150, break_long_words=False, break_on_hyphens=False)):
            draw.text((32, y + 465 + line * 30), value, font=fonts[20], fill="#20313b")
    canvas.save(directory / "reason_actual_inputs.png")
    result = {"scope": "完整独立反馈的Reason误判诊断；无新模型推理或收益结论", "judge": str(args.judge),
        "judge_sha256": sha256(args.judge), "collection": str(collection), "collection_sha256": sha256(collection),
        "metrics": metrics, "unique_init_seeds": report["unique_init_seeds"], "input_image_count": 6 * len(rows),
        "unique_input_file_sha256": len({item["sha256"] for row in rows for item in row["inputs"]}),
        "selected_diagnostic_ids": [row["episode_id"] for row in selected],
        "selection": "环境失败和成功各取ID最小项；诊断选择，不能用于证明策略收益",
        "hardware": probe["hardware"], "model_config": report["config"],
        "policy_ckpt_sha256": source["policy_ckpt_sha256"], "AI_generated_experiment_results": False,
        "new_inference_performed": False, "complete_rsi": False,
        "files_sha256": {p.name: sha256(p) for p in directory.iterdir() if p.is_file()}}
    dump(directory / "materials_manifest.json", result)
    return {"sample_count": len(rows), "calibration_passed": metrics["calibration_passed"], "complete_rsi": False}


if __name__ == "__main__":
    material_env = ROOT / "envs/sim"
    if ROOT == pathlib.Path("/home/thor/rsi") and pathlib.Path(sys.prefix).resolve() != material_env.resolve():
        executable = str(material_env / "bin/python")
        os.execve(executable, [executable, "-u", str(pathlib.Path(__file__).resolve()), *sys.argv[1:]],
                  {**os.environ, "CUDA_VISIBLE_DEVICES": "", "MPLBACKEND": "Agg", "OPENBLAS_NUM_THREADS": "1"})
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--judge", type=pathlib.Path, required=True)
    args = parser.parse_args(); args.judge = args.judge.resolve(strict=True)
    if not args.judge.is_relative_to(ROOT / "runs/p0"):
        parser.error("只读取本项目实际独立判官报告")
    raise SystemExit(run_stage("reason_diagnostic_materials", lambda directory: collect(directory, args),
        {"compute": "Thor CPU/Agg/Pillow；禁用CUDA，无新模型运行", "samples": "来自绑定的完整独立反馈"}))
