#!/usr/bin/env python3
"""核验完整冻结基线，导出分任务指标、真实帧截图及可追溯的中文实验素材。

@author pony
@date 2026-11-07
@version v1.0.1
@last_modified 2026-11-09
@changelog
    - v1.0.1 (2026-11-09): 使用既有sim绘图环境；从真实权重区分基线与正式更新，避免误标π0。
    - v1.0.0 (2026-11-07): 按初态聚类报告区间；评测数据只用于诊断和素材，不进入训练或生成源。
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
import os
import pathlib
import shutil
import sys

from p0_common import ROOT, run_stage
from p0_policy_runtime import sha256


CONDITIONS = ["standard", "lighting", "background", "layout"]
LABELS = {"standard": "标准", "lighting": "光照", "background": "背景纹理", "layout": "布局",
          "target_perturbed": "目标扰动", "nontarget_perturbed": "非目标扰动"}


def wilson95(successes: int, count: int) -> list[float]:
    if type(count) is not int or type(successes) is not int or count < 1 or not 0 <= successes <= count:
        raise ValueError("成功数和回合数必须有效")
    z = 1.959963984540054
    p = successes / count
    scale = 1 + z * z / count
    center = (p + z * z / (2 * count)) / scale
    radius = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / scale
    return [max(0.0, center - radius), min(1.0, center + radius)]


def validate_episodes(rows: list[dict], frozen: dict, summary: dict) -> None:
    expected = {row["eval_id"]: row for row in frozen["episodes"]}
    if len(expected) != len(frozen["episodes"]):
        raise ValueError("冻结文件含重复评测ID")
    ids = [row["eval_id"] for row in rows]
    if not summary.get("full_eval_complete") or len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError("实际回合没有完整且唯一覆盖冻结集")
    for row in rows:
        case = expected[row["eval_id"]]
        if row.get("is_eval") is not True or row.get("eligible_for_training") is not False \
                or type(row["outcome"]["success"]) is not bool or row["outcome"]["source"] != "env":
            raise ValueError("评测用途或环境真值字段不合格")
        for key in ["condition_id", "split", "init_seed", "policy_seed", "initial_state_sha256"]:
            if row[key] != case[key]:
                raise ValueError(f"实际回合偏离冻结定义：{key}")
        if row["policy_ckpt_sha256"] != summary["policy_ckpt_sha256"] \
                or row["eval_set_sha256"] != summary["eval_set_sha256"]:
            raise ValueError("回合模型或冻结集摘要不一致")
    for split, reported in summary["splits"].items():
        selected = [row for row in rows if row["split"] == split]
        successes = sum(row["outcome"]["success"] for row in selected)
        if len(selected) != reported["n"] or successes != reported["success"] \
                or not math.isclose(successes / len(selected), reported["p"], abs_tol=1e-12):
            raise ValueError("实际回合与汇总成功数不符")


def describe(rows: list[dict], bootstrap_seed: int = 42, replicates: int = 2000) -> dict:
    import numpy as np

    if not rows or replicates < 1:
        raise ValueError("统计输入不能为空，bootstrap重复数必须为正")
    clusters = defaultdict(list)
    for row in rows:
        # 同一任务初态的两个策略种子，以及同种子不同条件保守地放在同一簇，避免虚增独立样本。
        clusters[(row["task_id"], row["init_seed"])].append(row)
    strata = defaultdict(list)
    for (task, _), values in clusters.items():
        strata[task].append((sum(value["outcome"]["success"] for value in values), len(values)))
    rng = np.random.default_rng(bootstrap_seed)
    numerator, denominator = np.zeros(replicates), np.zeros(replicates)
    for task in sorted(strata):
        values = np.asarray(strata[task], dtype=np.float64)
        picked = rng.integers(0, len(values), size=(replicates, len(values)))
        numerator += values[picked, 0].sum(axis=1)
        denominator += values[picked, 1].sum(axis=1)
    n = len(rows)
    successes = sum(row["outcome"]["success"] for row in rows)
    return {"n": n, "success": successes, "p": successes / n, "initial_state_clusters": len(clusters),
            "wilson95_episode_descriptive": wilson95(successes, n),
            "cluster_bootstrap95": np.percentile(numerator / denominator, [2.5, 97.5]).tolist(),
            "mean_steps": float(np.mean([row["metrics"]["steps"] for row in rows])),
            "mean_episode_wall_s": float(np.mean([row["metrics"]["duration_s"] for row in rows]))}


def write_csv(path: pathlib.Path, records: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def policy_context(summary, metadata):
    if metadata["state_sha256"] != summary["policy_ckpt_sha256"] or metadata["step"] != summary["training_step"]:
        raise ValueError("素材对应的权重或累计训练步数不符")
    recipe = metadata["recipe"]
    if metadata.get("formal_policy_update") is True:
        role = metadata["role"]
        labels = {"candidate": "正式候选", "control": "D0复制对照", "analytic_ablation": "解析纯色消融"}
        label = f"第{metadata['iteration']}圈{labels[role]}"
    else:
        role, label = "incumbent", "正式基线策略"
    return {"role": role, "label": label, "iteration": metadata.get("iteration", 0),
            "training_batch": recipe["batch"], "training_seed": recipe["seed"]}


def collect(directory: pathlib.Path, args: argparse.Namespace) -> dict:
    import imageio.v2 as imageio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    summary_path = args.evaluation / "evaluation.json"
    episodes_path = args.evaluation / "episodes.jsonl"
    probe_path = args.evaluation / "probe.json"
    summary, probe = json.loads(summary_path.read_text()), json.loads(probe_path.read_text())
    frozen = json.loads(args.frozen_eval.read_text())
    rows = [json.loads(line) for line in episodes_path.read_text().splitlines()]
    if probe["status"] != "passed" or probe["stage"] != "policy_evaluation" \
            or not summary["server_stopped"] or summary["server_forced_kill"] \
            or probe["artifacts_sha256"].get(summary_path.name) != sha256(summary_path) \
            or probe["artifacts_sha256"].get(episodes_path.name) != sha256(episodes_path):
        raise RuntimeError("只分析服务正常退出的完整正式评测")
    if sha256(args.frozen_eval) != summary["eval_set_sha256"]:
        raise RuntimeError("冻结集内容与实际评测不符")
    validate_episodes(rows, frozen, summary)
    conditions = {row["condition_id"]: row for row in frozen["conditions"]}
    metadata_path = pathlib.Path(summary["checkpoint"]) / "checkpoint.json"
    context = policy_context(summary, json.loads(metadata_path.read_text()))
    sources = {summary_path, episodes_path, probe_path, args.frozen_eval, metadata_path}
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    plt.rcParams.update({"font.family": FontProperties(fname=font_path).get_name(), "axes.unicode_minus": False})
    image_font = ImageFont.truetype(font_path, 18, index=2)
    small_font = ImageFont.truetype(font_path, 14, index=2)
    common_caption = (f"Thor / 既有MAXN / VLM bf16、LoRA及动作头fp32 / 累计{summary['training_step']}步、"
                      f"训练batch{context['training_batch']}、seed{context['training_seed']} / 推理batch1")
    splits = {split: describe([row for row in rows if row["split"] == split]) for split in summary["splits"]}
    condition_stats = {condition: describe([row for row in rows if row["condition"] == condition]) for condition in CONDITIONS}
    per_condition = []
    for condition_id, condition in conditions.items():
        selected = [row for row in rows if row["condition_id"] == condition_id]
        per_condition.append({"task": condition_id.split("-", 1)[0], "condition": condition["condition"],
                              "task_id": condition["task_id"], "instruction": condition["instruction"],
                              **describe(selected), "is_eval": True, "eligible_for_training": False})
    write_csv(directory / "task_condition_metrics.csv", per_condition)
    write_csv(directory / "split_metrics.csv", [{"split": split, **value} for split, value in splits.items()])
    write_csv(directory / "condition_metrics.csv", [{"condition": key, **value} for key, value in condition_stats.items()])

    figure, axis = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    for index, condition in enumerate(CONDITIONS):
        value = condition_stats[condition]
        lo, hi = value["cluster_bootstrap95"]
        axis.bar(index, value["p"] * 100, color=["#287b72", "#c89a55", "#b76a55", "#73899d"][index])
        axis.errorbar(index, value["p"] * 100, yerr=[[max(0, value['p'] - lo) * 100], [max(0, hi - value['p']) * 100]],
                      capsize=4, color="#333333")
        axis.text(index, hi * 100 + 2, f"{value['success']}/{value['n']}\n{value['p']:.1%}", ha="center")
    axis.set(xticks=range(4), xticklabels=[LABELS[key] for key in CONDITIONS], ylabel="环境判定成功率（%）", ylim=(0, 105),
             title=f"{context['label']}：不同扰动下的真实成功率\n误差线按任务分层、按初态聚类bootstrap；跨策略结论见整圈门禁")
    axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(common_caption, fontsize=8)
    figure.savefig(directory / "condition_success_rates.png", dpi=180)
    plt.close(figure)

    task_ids = sorted({value["task"] for value in per_condition})
    matrix = np.asarray([[next(value["p"] for value in per_condition if value["task"] == task and value["condition"] == condition)
                          for condition in CONDITIONS] for task in task_ids])
    figure, axis = plt.subplots(figsize=(8.5, 6.6), constrained_layout=True)
    chart = axis.imshow(matrix * 100, vmin=0, vmax=100, cmap="YlGnBu", aspect="auto")
    for y, task in enumerate(task_ids):
        for x, condition in enumerate(CONDITIONS):
            value = next(value for value in per_condition if value["task"] == task and value["condition"] == condition)
            axis.text(x, y, f"{value['success']}/{value['n']}\n{value['p']:.0%}", ha="center", va="center",
                      color="white" if value["p"] >= 0.6 else "#192e35", fontsize=9)
    axis.set(xticks=range(4), xticklabels=[LABELS[key] for key in CONDITIONS], yticks=range(len(task_ids)), yticklabels=task_ids,
             title="十个任务 × 四种条件（1200回合）\n任务原文、场景路径与条件摘要见CSV及来源索引")
    figure.colorbar(chart, ax=axis, label="成功率（%）")
    figure.suptitle(common_caption, fontsize=8)
    figure.savefig(directory / "task_condition_heatmap.png", dpi=180)
    plt.close(figure)

    latencies = np.asarray([latency for row in rows for latency in row["policy_latencies_ms"]])
    http_latencies = np.asarray([latency for row in rows for latency in row["http_request_latencies_ms"]])
    if not np.isfinite(latencies).all() or not np.isfinite(http_latencies).all() or len(latencies) != summary["latency_ms"]["measured_over"]:
        raise RuntimeError("延迟原始记录不完整或非有限")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    for values, label, color in [(latencies, "服务内部", "#287b72"), (http_latencies, "HTTP含排队", "#b4774e")]:
        order = np.sort(values)
        axes[0].plot(order, np.arange(1, len(order) + 1) / len(order), label=label, color=color)
    axes[0].set(xlabel="毫秒", ylabel="累计比例", xlim=(0, float(np.percentile(http_latencies, 99.5))), title="四worker离线评测的每次请求延迟")
    axes[0].legend(frameon=False)
    for success, label, color in [(True, "成功回合", "#287b72"), (False, "失败回合", "#b4774e")]:
        values = np.sort([row["metrics"]["steps"] for row in rows if row["outcome"]["success"] == success])
        axes[1].step(values, np.arange(1, len(values) + 1) / len(values), where="post", label=f"{label} n={len(values)}", color=color)
    axes[1].set(xlabel="实际动作步数（上限520）", ylabel="累计比例", title="成功与失败的动作长度分布")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(common_caption + f" / 请求n={len(latencies)} / 20Hz动作块8", fontsize=8)
    figure.savefig(directory / "latency_and_episode_lengths.png", dpi=180)
    plt.close(figure)
    per_seed = []
    for seed in sorted({row["policy_seed"] for row in rows}):
        for split in splits:
            per_seed.append({"policy_seed": seed, "split": split,
                             **describe([row for row in rows if row["policy_seed"] == seed and row["split"] == split])})
    write_csv(directory / "policy_seed_metrics.csv", per_seed)

    screenshots = directory / "screenshots"
    screenshots.mkdir()
    screenshot_records = []
    sheet = Image.new("RGB", (1120, 72 + len(task_ids) * 284), "#f5f4ef")
    draw = ImageDraw.Draw(sheet)
    draw.text((20, 10), "冻结评测真实初始画面：同任务的四种条件", font=image_font, fill="#24363d")
    draw.text((20, 38), "每格按eval_id排序取首条；不同条件初态分别冻结，此图不是因果配对实验。", font=small_font, fill="#785a42")
    for y, task in enumerate(task_ids):
        for x, condition in enumerate(CONDITIONS):
            row = sorted([row for row in rows if row["condition_id"] == f"{task}-{condition}"], key=lambda row: row["eval_id"])[0]
            source = pathlib.Path(row["artifacts"]) / "agentview_first.png"
            if not source.resolve().is_relative_to(args.evaluation):
                raise RuntimeError("实际截图越过评测目录")
            target = screenshots / f"{row['eval_id']}_first.png"
            shutil.copyfile(source, target)
            sources.add(source)
            left, top = 20 + x * 275, 72 + y * 284
            with Image.open(source) as frame:
                sheet.paste(frame.convert("RGB").resize((250, 250)), (left, top + 20))
            draw.text((left, top), f"{task} / {LABELS[condition]}", font=small_font, fill="#24363d")
            screenshot_records.append({"output": str(target), "source": str(source), "source_sha256": sha256(source),
                                       "eval_id": row["eval_id"], "frame": 0, "condition": condition, "kind": "MuJoCo真实渲染帧"})
    sheet.save(directory / "all_conditions_contact_sheet.png")

    # 每个有成功和失败的标准任务按eval_id各取首个，固定选择规则，避免只挑最漂亮的例子。
    examples = []
    for task in task_ids:
        selected = sorted([row for row in rows if row["condition_id"] == f"{task}-standard"], key=lambda row: row["eval_id"])
        good = next((row for row in selected if row["outcome"]["success"]), None)
        bad = next((row for row in selected if not row["outcome"]["success"]), None)
        if good and bad:
            examples.extend([good, bad])
        if len(examples) >= 6:
            break
    for row in examples:
        artifact = pathlib.Path(row["artifacts"])
        count = row["metrics"]["steps"]
        indices = sorted(set([0, count // 3, 2 * count // 3, count - 1]))
        story = Image.new("RGB", (1100, 675), "#f5f4ef")
        draw = ImageDraw.Draw(story)
        draw.text((16, 10), f"真实策略回合 / {row['condition_id']} / 环境判定：{'成功' if row['outcome']['success'] else '失败'}", font=image_font, fill="#24363d")
        draw.text((16, 38), row["eval_id"], font=small_font, fill="#785a42")
        for view_index, view in enumerate(["agentview", "wrist"]):
            video = artifact / f"{view}.mp4"
            sources.add(video)
            seen = 0
            with imageio.get_reader(video) as reader:
                for index, frame in enumerate(reader):
                    seen += 1
                    if index in indices:
                        slot = indices.index(index)
                        target = screenshots / f"{row['eval_id']}_{view}_{index:03d}.png"
                        Image.fromarray(frame).save(target)
                        left, top = 16 + slot * 270, 78 + view_index * 280
                        story.paste(Image.fromarray(frame).resize((250, 250)), (left, top + 21))
                        draw.text((left, top), f"{'外部' if view_index == 0 else '腕部'} / 动作前第{index}帧", font=small_font, fill="#24363d")
                        screenshot_records.append({"output": str(target), "source": str(video), "eval_id": row["eval_id"],
                                                   "frame": index, "kind": "真实策略视频解码帧", "view": view})
            if seen != count:
                raise RuntimeError("真实视频帧数与动作记录不相等")
        draw.text((16, 644), f"20Hz仿真时间 / {count}步 / {row['metrics']['duration_s']:.2f}秒实际墙钟 / 评测数据禁止入训", font=small_font, fill="#785a42")
        story.save(directory / f"storyboard_{row['eval_id']}.png")
        for name in ["agentview_last.png", "episode.json"]:
            source = artifact / name
            shutil.copyfile(source, screenshots / f"{row['eval_id']}_{name}")
            sources.add(source)

    training_sources = []
    if args.training:
        runs = [json.loads(path.read_text()) for path in args.training]
        metrics = []
        validation = []
        for path, run in zip(args.training, runs):
            metric_path = path.parent / "logs/train_metrics.jsonl"
            metrics.extend(json.loads(line) for line in metric_path.read_text().splitlines())
            validation.extend(run["validation"])
            training_sources.extend([path, metric_path])
        metrics.sort(key=lambda row: row["step"])
        steps = [row["step"] for row in metrics]
        if steps != list(range(1, summary["training_step"] + 1)) or any(run["recipe"] != runs[0]["recipe"] for run in runs):
            raise RuntimeError("分段训练的完整步号或配方不一致")
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
        losses = np.asarray([row["loss"] for row in metrics])
        axes[0].plot(steps, losses, color="#c3d6d2", linewidth=0.4)
        axes[0].plot(steps[49:], np.convolve(losses, np.ones(50) / 50, mode="valid"), color="#287b72", label="训练50步移动均值")
        axes[0].plot([row["step"] for row in validation], [row["mean_loss"] for row in validation], color="#b4774e", marker=".", label="固定离线验证32窗口")
        axes[0].set(xlabel="实际训练步数", ylabel="Flow matching loss", title="完整5000步离线曲线；不能替代成功率")
        axes[0].legend(frameon=False)
        axes[1].plot(steps, [row["samples_per_s"] for row in metrics], linewidth=0.6, color="#287b72")
        axes[1].set(xlabel="实际训练步数", ylabel="samples/s", title="含缓存采样和前后向，不含验证、保存及加载")
        for axis in axes:
            axis.spines[["top", "right"]].set_visible(False)
        figure.suptitle(common_caption, fontsize=8)
        figure.savefig(directory / "full_training_curves.png", dpi=180)
        plt.close(figure)
        write_csv(directory / "full_training_metrics.csv", [{key: row[key] for key in ["step", "loss", "grad_norm", "wall_s", "samples_per_s"]} for row in metrics])
    sources.update(training_sources)
    source_records = [{"path": str(path), "sha256": sha256(path)} for path in sorted(sources)]
    record = {"evaluation": str(summary_path), "iter": context["iteration"], "role": context["role"],
              "policy_label": context["label"], "policy_ckpt_sha256": summary["policy_ckpt_sha256"],
              "eval_set_sha256": summary["eval_set_sha256"], "hardware": probe["hardware"],
              "precision": summary["precision"], "training_step": summary["training_step"], "policy_batch": 1,
              "training_batch": context["training_batch"], "training_seed": context["training_seed"],
              "policy_seeds": sorted({row["policy_seed"] for row in rows}),
              "action_horizon": 8, "control_hz": 20, "episode_max_steps": 520, "environment_workers": 4,
              "splits": splits, "conditions": condition_stats, "per_task_condition": per_condition, "per_policy_seed": per_seed,
              "latency_ms": summary["latency_ms"], "http_request_latency_ms": summary["http_request_latency_ms"],
              "rollout_wall_s": summary["rollout_wall_s"], "peak_mem_gb": probe["peak_mem_gb"],
              "statistics": {"cluster": "task_id+init_seed，同初态全部策略种子/条件保守聚为一簇",
                             "bootstrap": "按task_id分层，簇有放回抽样，保留任务样本量", "seed": 42, "replicates": 2000,
                             "wilson": "回合级描述性区间；同初态策略种子不独立，推断以聚类结果为准",
                             "limitation": "只有一个训练种子；区间不包含训练随机性，零成功任务的bootstrap会退化"},
              "screenshots": screenshot_records, "sources": source_records,
              "is_eval": True, "eligible_for_training": False, "training_or_generation_exported": False,
              "promotion_performed": False, "presentation_created": False,
              "boundary": "评测指标与截图仅用于诊断和展示；不能把失败归因当作因果证明或将评测回合用于训练"}
    (directory / "analysis.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    (directory / "material_sources.json").write_text(json.dumps({"sources": source_records, "screenshots": screenshot_records,
                                                                "conditions": {key: record[key] for key in ["hardware", "precision", "training_step", "training_batch", "training_seed", "policy_batch", "policy_seeds", "action_horizon", "control_hz", "episode_max_steps", "environment_workers"]}},
                                                               ensure_ascii=False, indent=2) + "\n")
    (directory / "captions.md").write_text(
        f"# {context['label']}真实冻结评测素材\n\n" + common_caption + "。策略种子34001/34002，十任务、四条件、1200回合；20Hz、动作块8、每回合上限520步。\n\n"
        "condition_success_rates.png为四条件总体成功率，task_condition_heatmap.png为分任务结果；区间按任务分层、初态聚类。"
        "完整数据在CSV及analysis.json。只有一个训练种子，不能据此宣称RSI增益。\n\n"
        "all_conditions_contact_sheet.png及screenshots内图片来自真实MuJoCo评测，按固定ID规则选取；不是实机照片。"
        "storyboard显示实际双视角视频的动作前帧，环境成功信号在最后动作后判定。不同条件并非严格配对因果实验。\n\n"
        "latency_and_episode_lengths.png中的HTTP延迟包含四worker排队，服务内部延迟单列，不得当作单机器人实时控制延迟。"
        "指定初训日志时另导出full_training_curves.png；正式增量曲线由整圈素材入口导出。成功率应看独立冻结评测。\n\n"
        "源文件、SHA256、帧索引和回合ID在material_sources.json；原视频留在Thor。评测数据禁止入训或作生成源，PPT尚未制作。\n",
        encoding="utf-8")
    return {"materials": str(directory), "sample_count": len(rows), "splits": splits, "conditions": condition_stats,
            "screenshot_count": len(screenshot_records), "policy_ckpt_sha256": summary["policy_ckpt_sha256"],
            "eligible_for_training": False, "presentation_created": False}


if __name__ == "__main__":
    # 编排器使用policy解释器；CPU绘图改由已验证的sim环境执行，不修改训练环境依赖。
    material_env = ROOT / "envs/sim"
    if ROOT == pathlib.Path("/home/thor/rsi") and pathlib.Path(sys.prefix).resolve() != material_env.resolve():
        executable = str(material_env / "bin/python")
        print("CPU素材阶段使用envs/sim；禁用CUDA，不启动策略服务。", flush=True)
        os.execve(executable, [executable, "-u", str(pathlib.Path(__file__).resolve()), *sys.argv[1:]],
                  {**os.environ, "CUDA_VISIBLE_DEVICES": "", "MPLBACKEND": "Agg", "OPENBLAS_NUM_THREADS": "1"})
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--evaluation", type=pathlib.Path, required=True)
    parser.add_argument("--frozen-eval", type=pathlib.Path, default=ROOT / "eval/frozen_eval_set.json")
    parser.add_argument("--training", type=pathlib.Path, action="append", default=[])
    arguments = parser.parse_args()
    arguments.evaluation = arguments.evaluation.resolve(strict=True)
    arguments.frozen_eval = arguments.frozen_eval.resolve(strict=True)
    arguments.training = [path.resolve(strict=True) for path in arguments.training]
    if not arguments.evaluation.is_relative_to(ROOT / "runs/p0") \
            or not arguments.frozen_eval.is_relative_to(ROOT / "eval") \
            or any(not path.is_relative_to(ROOT / "runs/p0") for path in arguments.training):
        parser.error("评测、训练记录和冻结集必须位于Thor项目目录")
    raise SystemExit(run_stage("evaluation_materials", lambda directory: collect(directory, arguments),
                              {"evaluation": str(arguments.evaluation), "compute": "CPU_only_analysis_and_real_frame_export"}))
