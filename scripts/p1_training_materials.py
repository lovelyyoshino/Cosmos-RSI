#!/usr/bin/env python3
"""从正在评测的确切整圈导出已封存正式训练素材，不提前计算任务收益。

@author pony
@date 2026-11-09
@version v1.0.0
@last_modified 2026-11-09
@changelog
    - v1.0.0 (2026-11-09): 核对三角色权重、同预算和实际stdout后导出曲线与CSV。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

from p0_common import ROOT, run_stage
from p0_lerobot_handoff import dump
from p1_eval_gate import checked_update, sha256
from p1_iteration_materials import LABELS, training_trace, write_csv


def collect(directory, iteration_run):
    state_path = iteration_run / "iteration_state.json"
    state = json.loads(state_path.read_text())
    roles = ["candidate", "control", "analytic_ablation"]
    training, traces, probes, sources = {}, {}, {}, {}
    for role in roles:
        binding = state["completed_stages"]["train_" + role]
        path = pathlib.Path(binding["path"]).resolve(strict=True)
        if not path.is_relative_to(ROOT / "runs/p0") or binding["stage"] != "p1_policy_train" \
                or sha256(path) != binding["sha256"]:
            raise ValueError("素材没有绑定本圈完整正式训练阶段")
        report, _ = checked_update(path, role)
        probe_path = path.parent / "probe.json"
        probes[role] = json.loads(probe_path.read_text())
        trace_path = path.parent / "logs/steps.jsonl"
        rows = training_trace(trace_path, report)
        stdout = iteration_run / "logs" / ("train_" + role + ".log")
        printed = [json.loads(line) for line in stdout.read_text().splitlines() if line.startswith('{"step":')]
        if printed != rows or report["iteration"] != state["iteration"]:
            raise ValueError("正式训练stdout、逐步日志或圈数不一致")
        training[role], traces[role] = report, rows
        sources[role] = {"report": str(path), "report_sha256": sha256(path),
                        "probe": str(probe_path), "probe_sha256": sha256(probe_path),
                        "trace": str(trace_path), "trace_sha256": sha256(trace_path),
                        "stdout": str(stdout), "stdout_sha256": sha256(stdout)}
    for key in ["iteration", "parent_ckpt_sha256", "paired_plan_sha256", "steps", "batch", "seed", "recipe",
                "precision", "attention", "deepSpeed"]:
        if any(training[role][key] != training["candidate"][key] for role in roles):
            raise ValueError("正式三角色的父策略、抽样计划或预算不一致")
    write_csv(directory / "training_steps.csv", [{"role": role, **row} for role in roles for row in traces[role]])
    summary = [{"role": role, **{key: training[role][key] for key in
                   ["steps", "batch", "seed", "training_wall_s", "reload_max_abs_diff", "output_ckpt_sha256"]},
                "sample_slots_with_replacement": training[role]["steps"] * training[role]["batch"],
                "precision": json.dumps(training[role]["precision"], ensure_ascii=False),
                "attention": training[role]["attention"], "deepSpeed": training[role]["deepSpeed"],
                "hardware": json.dumps(probes[role]["hardware"], ensure_ascii=False),
                "full_stage_wall_s": probes[role]["wall_s"],
                "whole_host_peak_ram_gb": probes[role]["peak_mem_gb"]} for role in roles]
    write_csv(directory / "training_summary.csv", summary)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties

    font_path = subprocess.check_output(["fc-match", "-f", "%{file}", "Noto Sans CJK SC"], text=True)
    plt.rcParams.update({"font.family": FontProperties(fname=font_path).get_name(), "font.size": 13})
    figure, axes = plt.subplots(1, 2, figsize=(16, 9), gridspec_kw={"width_ratios": [1.5, 1]})
    colors = ["#217C85", "#697C91", "#BC7A2C"]
    for role, color in zip(roles, colors):
        rows = traces[role]
        axes[0].plot([row["step"] for row in rows], [row["loss"] for row in rows],
                     color=color, linewidth=.75, alpha=.8, label=LABELS[role])
    axes[0].set(title="正式训练：全部原始损失，无平滑或删点", xlabel="增量训练步", ylabel="动作损失")
    axes[0].legend(fontsize=10)
    for index, (row, color) in enumerate(zip(summary, colors)):
        axes[1].barh(index, row["full_stage_wall_s"], color=color, alpha=.3, label="完整阶段" if index == 0 else None)
        axes[1].barh(index, row["training_wall_s"], height=.4, color=color, label="训练循环" if index == 0 else None)
        axes[1].text(row["full_stage_wall_s"] + 12, index,
                     f"{row['full_stage_wall_s']:.1f}s", va="center", fontsize=10)
    axes[1].set(title="每角色实测墙钟", xlabel="秒", yticks=range(3), yticklabels=[LABELS[role] for role in roles])
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, max(row["full_stage_wall_s"] for row in summary) * 1.23)
    axes[1].legend(fontsize=10, loc="upper center", bbox_to_anchor=(.5, -.12), ncol=2, frameon=False)
    for axis in axes:
        axis.grid(axis="y", alpha=.15)
        axis.set_axisbelow(True)
    report = training["candidate"]
    figure.suptitle(f"Cosmos × Thor：第{state['iteration']}圈正式三角色训练", fontsize=23, y=.96)
    figure.subplots_adjust(left=.08, right=.98, top=.85, bottom=.35, wspace=.45)
    footer = [f"Thor / 各训练原probe记录MAXN / VLM bf16、LoRA与动作头fp32 / sdpa / 无DeepSpeed",
              f"同一正式父策略 / 各{report['steps']}步 / batch{report['batch']} / seed{report['seed']} / 每角色{report['steps'] * report['batch']}个有放回槽位",
              "候选含D0、自采成功与既有合格Cosmos纹理窗口；对照用D0补齐；解析消融使用纯色外观。",
              "完整阶段含核验、加载、训练、保存与重载；CSV另列tegrastats全机RAM采样峰值。",
              "仅证明真实训练接入；损失和耗时不能证明任务成功率、Cosmos独立贡献或连续多圈RSI。",
              "本圈未重新调用Cosmos生成；冻结评测另行验收。本图为CPU导出，未运行新模型。"]
    for index, line in enumerate(footer):
        figure.text(.06, .235 - .032 * index, line, fontsize=10, color="#52616B")
    figure.savefig(directory / "formal_training.png", dpi=240)
    figure.savefig(directory / "formal_training.svg")
    plt.close(figure)
    result = {"schema": "thor-p1-training-materials-v1", "iteration_run": str(iteration_run),
              "iteration": state["iteration"], "phase_at_export": state["phase"],
              "state_snapshot": state, "sources": sources, "summary": summary,
              "policy_gain_evaluated_by_this_export": False, "new_model_inference_performed": False,
              "eligible_for_training": False, "complete_rsi": False, "presentation_created": False,
              "files_sha256": {path.name: sha256(path) for path in directory.iterdir() if path.is_file()}}
    dump(directory / "materials_manifest.json", result)
    return {"roles": roles, "steps_per_role": report["steps"], "policy_gain_evaluated": False,
            "manifest": str(directory / "materials_manifest.json")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--iteration-run", type=pathlib.Path, required=True)
    args = parser.parse_args()
    args.iteration_run = args.iteration_run.resolve(strict=True)
    if not args.iteration_run.is_relative_to(ROOT / "runs/p0"):
        parser.error("只接受本项目明确指定的整圈")
    raise SystemExit(run_stage("p1_training_materials", lambda directory: collect(directory, args.iteration_run),
                              {"compute": "Thor CPU导出已封存训练；无新模型运行", "precision": "不适用",
                               "batch": None, "seed": None}))
