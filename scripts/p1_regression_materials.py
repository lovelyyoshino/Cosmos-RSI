#!/usr/bin/env python3
"""将真实权重审计和零更新动作诊断转成可追溯图表，不加载模型或声明任务收益。

@author pony
@date 2026-11-09
@version v1.0.0
@last_modified 2026-11-09
@changelog
    - v1.0.0 (2026-11-09): 导出参数漂移与独立示范动作误差；可等待确切诊断任务。
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
import pathlib
import subprocess

from p0_common import ROOT, run_stage
from p0_lerobot_handoff import dump
from p1_conservative_pilot import bind_probe, wait_probe
from p1_eval_gate import bound_file, sealed, sha256
from p1_iteration_materials import write_csv
from p1_policy_regression_probe import CONFIG, ROLES, action_metrics, project


LABELS = {"incumbent": "原策略", "candidate": "本圈候选", "control": "D0复制对照",
          "analytic_ablation": "解析外观消融"}
METRICS = ["action_mse", "arm_mse", "gripper_mse", "drift_vs_incumbent_rms"]


def checked_audit(path):
    report, probe = sealed(path, "p1_policy_weight_reader_audit")
    if report.get("schema") != CONFIG["schema"] or report.get("authority") != CONFIG["authority"] \
            or report.get("causal_attribution_established") is not False \
            or set(report["weight_deltas"]) != set(ROLES[1:]):
        raise ValueError("素材只接受真实零更新审计，不能借此授予收益或训练资格")
    for binding in report["inputs"]["files"].values():
        if sha256(project(binding["path"])) != binding["sha256"]:
            raise ValueError("权重审计绑定的实际来源已经改变")
    for groups in report["weight_deltas"].values():
        if set(groups) != {"vlm_lora", "action_head"}:
            raise ValueError("参数分组缺失")
        for row in groups.values():
            for key in ["parent_l2", "delta_l2", "relative_delta_l2", "max_abs_delta"]:
                value = row[key]
                if type(value) not in {int, float} or not math.isfinite(value) or value < 0:
                    raise ValueError("真实参数距离不是有限非负数")
            if row["parent_l2"] <= 0 or not math.isclose(row["relative_delta_l2"],
                    row["delta_l2"] / row["parent_l2"], rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError("参数相对距离与实际范数不一致")
    if report["reader_audit"].get("source_reading_passed") is not True \
            or not report["reader_audit"]["checked_blocks"]:
        raise ValueError("原生读取审计尚未通过")
    return report, probe


def summarize_predictions(rows, locations, seeds):
    """重算每块误差；先平均种子和时刻、再等权平均示范，避免把相关块当独立回合。"""
    import numpy as np

    locations_by_key = {(r["episode_id"], r["frame"]): r for r in locations}
    if len(locations_by_key) != len(locations) or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("动作素材的预声明位置或种子有重复")
    expected = {(role, *key, seed) for role in ROLES for key in locations_by_key for seed in seeds}
    indexed = {}
    for row in rows:
        key = (row["role"], row["episode_id"], row["frame"], row["policy_seed"])
        location = locations_by_key.get((row["episode_id"], row["frame"]))
        if key in indexed or key not in expected or location is None \
                or any(row[k] != location[k] for k in location) or row["split"] != "demo_validation":
            raise ValueError("动作素材缺块、重复、位置错绑或混入训练来源")
        prediction, target = [np.asarray(row[k], dtype=np.float64)
                              for k in ["normalized_actions", "target_actions"]]
        if prediction.shape != (8, 7) or target.shape != (8, 7) \
                or not np.isfinite(prediction).all() or not np.isfinite(target).all():
            raise ValueError("动作素材必须保留完整有限的8×7原始动作与真值")
        indexed[key] = (row, prediction, target)
    if set(indexed) != expected:
        raise ValueError("动作素材没有完整覆盖全部角色、动作块和种子")
    per_episode = defaultdict(list)
    for key, (row, prediction, target) in indexed.items():
        old_row, old, old_target = indexed[("incumbent", *key[1:])]
        if not np.array_equal(target, old_target) or row["instruction"] != old_row["instruction"]:
            raise ValueError("配对策略使用了不同动作真值或任务")
        actual = action_metrics(prediction, target, old)
        if any(type(row[k]) not in {int, float} or not math.isfinite(row[k])
                or not math.isclose(row[k], actual[k], rel_tol=1e-10, abs_tol=1e-12) for k in METRICS):
            raise ValueError("动作素材记录的误差与实际预测不一致")
        per_episode[(row["role"], row["instruction"], row["episode_id"])].append(actual)
    per_task = defaultdict(list)
    for (role, instruction, _), values in per_episode.items():
        per_task[(role, instruction)].append({k: sum(v[k] for v in values) / len(values) for k in METRICS})
    return [{"role": role, "instruction": instruction, "source_episodes": len(values),
             **{k: sum(v[k] for v in values) / len(values) for k in METRICS}}
            for (role, instruction), values in sorted(per_task.items())]


def checked_actions(path, audit):
    report, probe = sealed(path, "p1_policy_action_probe")
    if report.get("schema") != CONFIG["schema"] or report.get("authority") != CONFIG["authority"] \
            or report.get("causal_attribution_established") is not False \
            or report["inputs"]["iteration_run"] != audit["inputs"]["iteration_run"] \
            or report["inputs"]["checkpoints"] != audit["inputs"]["checkpoints"] \
            or report["inputs"]["locations"] != audit["inputs"]["locations"] \
            or set(report["zero_update_reproduction"]) != set(ROLES) \
            or report["inference_seeds"] != CONFIG["inference"]["seeds"]:
        raise ValueError("动作诊断与实际权重审计来源、位置或实验范围不同")
    for key, value in audit["inputs"]["files"].items():
        if report["inputs"]["files"].get(key) != value or sha256(project(value["path"])) != value["sha256"]:
            raise ValueError("动作诊断与原权重审计的实际文件错绑")
    for value in report["zero_update_reproduction"].values():
        error = value["max_abs_diff"]
        if value.get("passed") is not True or value.get("optimization_steps") != 0 \
                or type(error) not in {int, float} or not math.isfinite(error) \
                or not 0 <= error <= CONFIG["inference"]["saved_probe_max_abs_error"]:
            raise ValueError("原保存动作尚未真实复现，不能输出完整动作比较素材")
    predictions = bound_file(path.parent, probe, "validation_predictions.jsonl")
    rows = [json.loads(line) for line in predictions.read_text().splitlines()]
    summary = summarize_predictions(rows, report["inputs"]["locations"], report["inference_seeds"])
    if report["validation_blocks"] != len(report["inputs"]["locations"]):
        raise ValueError("动作诊断实际块数与来源不一致")
    return report, summary, predictions


def render(directory, audit, actions, summaries):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties

    font = subprocess.check_output(["fc-match", "-f", "%{file}", "Noto Sans CJK SC"], text=True)
    plt.rcParams.update({"font.family": FontProperties(fname=font).get_name(), "font.size": 12})
    figure, axes = plt.subplots(1, 2, figsize=(16, 9), gridspec_kw={"width_ratios": [1, 1.4]})
    positions = np.arange(3)
    for group, label, offset, color in [("vlm_lora", "VLM LoRA", -.18, "#247C85"),
                                         ("action_head", "动作头", .18, "#BC7A2C")]:
        values = [100 * audit["weight_deltas"][role][group]["relative_delta_l2"] for role in ROLES[1:]]
        bars = axes[0].bar(positions + offset, values, .36, label=label, color=color)
        axes[0].bar_label(bars, fmt="%.2f%%", padding=3, fontsize=10)
    axes[0].set(xticks=positions, xticklabels=[LABELS[r] for r in ROLES[1:]],
                ylabel="相对父权重的 L2 距离（%）", title="真实张量更新幅度")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=.15)
    axes[0].set_axisbelow(True)
    if actions is None:
        axes[1].axis("off")
        axes[1].text(.05, .72, "零更新动作复现与独立示范误差\n仍在等待确切模型任务", transform=axes[1].transAxes,
                     fontsize=20, linespacing=1.8, color="#52616B")
        axes[1].text(.05, .36, "当前只展示已完成的参数与读取审计。\n不填充动作误差、不推断退化原因。",
                     transform=axes[1].transAxes, fontsize=13, linespacing=1.8, color="#52616B")
    else:
        tasks = sorted({r["instruction"] for r in summaries})
        lookup = {(r["role"], r["instruction"]): r["action_mse"] for r in summaries}
        matrix = np.array([[lookup[(role, task)] for task in tasks] for role in ROLES])
        heatmap = axes[1].imshow(matrix, aspect="auto", cmap="YlOrBr", vmin=0)
        axes[1].set(yticks=range(4), yticklabels=[LABELS[r] for r in ROLES],
                    xticks=range(len(tasks)), xticklabels=[f"T{i + 1:02d}" for i in range(len(tasks))],
                    title="独立示范动作 MSE（越低越好）", xlabel="任务映射与全部原始误差见 CSV")
        for (i, j), value in np.ndenumerate(matrix):
            axes[1].text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=8,
                         color="white" if value > matrix.max() * .65 else "#20313c")
        figure.colorbar(heatmap, ax=axes[1], fraction=.04, pad=.03)
    figure.suptitle("Cosmos × Thor：策略更新退化诊断", fontsize=24, y=.97)
    figure.subplots_adjust(left=.07, right=.97, top=.86, bottom=.34, wspace=.3)
    metadata = {role: json.loads(project(audit["inputs"]["files"]["checkpoint_" + role]["path"]).read_text())
                for role in ROLES}
    recipes = [metadata[role]["recipe"] for role in ROLES[1:]]
    if any(any(row[key] != recipes[0][key] for key in ["steps", "batch", "seed"]) for row in recipes):
        raise ValueError("参数图的三个角色训练条件不一致")
    recipe = recipes[0]
    blocks = len(audit["reader_audit"]["checked_blocks"])
    successes = sum(audit["reader_audit"]["self_success_instructions"].values())
    footer = [f"源实验：Thor / 功耗见原probe / {metadata['incumbent']['precision']} / sdpa",
              f"父{metadata['incumbent']['step']}步、更新各{recipe['steps']}步 / batch{recipe['batch']} / seed{recipe['seed']}",
              "参数图：三角色同一实际父权重的fp32可训练张量；VLM LoRA与动作头分别汇总。",
              f"读取审计：{successes}条自身成功及两种派生源共{blocks}个首中末动作块；不是{blocks}个独立回合。",
              (f"动作图：{len(tasks)}任务 / {actions['validation_blocks']}块 / batch{actions['batch']} / 噪声{actions['inference_seeds']} / 动作块8；按示范等权。"
               if actions else "动作图尚未运行；保留空白状态，不将参数距离当成动作误差或任务成功率。"),
              "本图：Thor CPU重新核算及绘制，CUDA禁用；不执行训练、模型推理、冻结门禁或晋升。",
              "参数距离和离线动作误差仅用于诊断；不能证明退化原因、Cosmos贡献或连续RSI收益。"]
    for index, line in enumerate(footer):
        figure.text(.055, .27 - .033 * index, line, fontsize=9.5, color="#52616B")
    figure.savefig(directory / "policy_regression_diagnostics.png", dpi=240)
    figure.savefig(directory / "policy_regression_diagnostics.svg")
    plt.close(figure)


def collect(directory, audit_path, action_path=None):
    audit, audit_probe = checked_audit(audit_path)
    parameters = [{"role": role, "group": group, **row}
                  for role, groups in audit["weight_deltas"].items() for group, row in groups.items()]
    write_csv(directory / "parameter_drift.csv", parameters)
    action_report, summaries, predictions = None, [], None
    if action_path is not None:
        action_report, summaries, predictions = checked_actions(action_path, audit)
        write_csv(directory / "validation_by_task.csv", summaries)
        tasks = sorted({r["instruction"] for r in summaries})
        write_csv(directory / "task_mapping.csv", [{"task": f"T{i + 1:02d}", "instruction": task}
                                                   for i, task in enumerate(tasks)])
    render(directory, audit, action_report, summaries)
    sources = {"weight_audit": {"path": str(audit_path), "sha256": sha256(audit_path)}}
    if action_path is not None:
        sources.update(action_probe={"path": str(action_path), "sha256": sha256(action_path)},
                       predictions={"path": str(predictions), "sha256": sha256(predictions)})
    result = {"schema": "thor-policy-regression-materials-v1", "sources": sources,
              "action_diagnostics_complete": action_report is not None,
              "new_model_inference_performed": False, "optimization_steps": 0,
              "task_success_evaluated": False, "causal_attribution_established": False,
              "eligible_for_training": False, "complete_rsi": False, "presentation_created": False,
              "source_audit_hardware": audit_probe.get("hardware"),
              "source_audit_conditions": audit_probe.get("conditions"),
              "summary_by_task": summaries,
              "files_sha256": {p.name: sha256(p) for p in directory.iterdir() if p.is_file()}}
    dump(directory / "materials_manifest.json", result)
    return {"action_diagnostics_complete": result["action_diagnostics_complete"], "complete_rsi": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weight-audit", type=pathlib.Path, required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--action-probe", type=pathlib.Path)
    group.add_argument("--after-action-probe-exec", type=pathlib.Path)
    parser.add_argument("--wait-timeout", type=int, default=86400)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or args.wait_timeout < 1:
        parser.error("素材入口必须显式禁用CUDA并使用正等待时长")
    args.weight_audit = project(args.weight_audit)
    audit, _ = checked_audit(args.weight_audit)
    audit_sha = sha256(args.weight_audit)
    if args.after_action_probe_exec:
        binding = bind_probe(project(args.after_action_probe_exec), audit["inputs"])
        print(json.dumps({"phase": "waiting_for_exact_action_probe_cpu_materials", "probe": binding}, ensure_ascii=False), flush=True)
        args.action_probe, _ = wait_probe(binding, args.wait_timeout)
    if sha256(args.weight_audit) != audit_sha:
        raise ValueError("等待期间权重审计发生变化")
    if args.action_probe:
        args.action_probe = project(args.action_probe)
    return run_stage("p1_regression_materials", lambda directory: collect(directory, args.weight_audit, args.action_probe),
                     {"compute": "Thor CPU、CUDA禁用；不加载模型，仅封存审计出图", "optimization_steps": 0,
                      "new_model_inference_performed": False})


if __name__ == "__main__":
    raise SystemExit(main())
