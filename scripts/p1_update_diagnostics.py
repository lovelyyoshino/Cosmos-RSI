#!/usr/bin/env python3
"""用已封存首圈产物诊断候选退化；CPU出图，不改训练、冻结集或晋升权限。

@author pony
@date 2026-11-09
@version v1.1.0
@last_modified 2026-11-09
@changelog
    - v1.1.0 (2026-11-09): 可选导出固定回合完整双策略视频，结束后显示结束卡并实际解码核验。
    - v1.0.0 (2026-11-09): 配对统计真实回合、核对采样覆盖并导出固定帧，不把退化归因于单一组件。
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
import json
import pathlib

from p0_common import ROOT, run_stage
from p0_lerobot_handoff import dump
from p0_paired_sampling import paired_plan_sha256
from p1_eval_gate import SPLITS, bound_file, checked_evaluation, checked_update, sealed, sha256
from p1_iteration_materials import training_trace, write_csv


def paired_outcomes(incumbent, candidate):
    """按完整冻结身份配对；两个策略种子或不同条件不能视为额外独立初态。"""
    indexes = [{row["eval_id"]: row for row in rows} for rows in [incumbent, candidate]]
    if not indexes[0] or any(len(index) != len(rows) for index, rows in zip(indexes, [incumbent, candidate])) \
            or set(indexes[0]) != set(indexes[1]):
        raise ValueError("候选和当前策略缺回合、重复ID或冻结计划不一致")
    counts = defaultdict(lambda: Counter(n=0, incumbent_success=0, candidate_success=0,
                                         lost_success=0, gained_success=0))
    pairs = []
    for identity, left in sorted(indexes[0].items()):
        right = indexes[1][identity]
        for row in [left, right]:
            if row["is_eval"] is not True or row["eligible_for_training"] is not False \
                    or row["outcome"].get("source") != "env" \
                    or type(row["outcome"].get("success")) is not bool:
                raise ValueError("退化诊断只接受冻结评测的实际环境成败")
        for key in ["condition_id", "split", "init_seed", "policy_seed", "initial_state_sha256", "eval_set_sha256"]:
            if left[key] != right[key]:
                raise ValueError("配对回合的实际初态、种子或冻结条件错绑")
        old, new = left["outcome"]["success"], right["outcome"]["success"]
        values = dict(n=1, incumbent_success=int(old), candidate_success=int(new),
                      lost_success=int(old and not new), gained_success=int(new and not old))
        for group in ["split:" + left["split"], "condition:" + left["condition_id"]]:
            counts[group].update(values)
        pairs.append({"eval_id": identity, "condition_id": left["condition_id"], "split": left["split"],
                      "init_seed": left["init_seed"], "policy_seed": left["policy_seed"], **values})
    summary = [{"group": key, **value, "delta_percentage_points":
                100 * (value["candidate_success"] - value["incumbent_success"]) / value["n"]}
               for key, value in sorted(counts.items())]
    return summary, pairs


def exposure_counts(plan, role, instructions):
    """统计实际有放回槽位和唯一窗口；不把相同窗口的重复训练计为新轨迹。"""
    if paired_plan_sha256(plan) != plan["sha256"] or role not in {"candidate", "control"}:
        raise ValueError("实际采样计划摘要或角色不符")
    selected = plan[role]
    if len(selected) != plan["steps"] * plan["batch"]:
        raise ValueError("采样计划没有覆盖实际步数和batch")
    descriptions = {row["sample_id"]: row for row in plan["descriptions"]}
    counts, windows, sources = Counter(), defaultdict(set), defaultdict(set)
    for index, row in enumerate(selected):
        descriptor = descriptions[row["sample_id"]]
        expected_kind = "demo" if row["kind"] == "duplicate_fill" else row["kind"]
        if row["kind"] not in {"demo", "duplicate_fill", "self_distill", "generated"} \
                or descriptor["kind"] != expected_kind or row["is_eval"] is not False \
                or any(row[key] != descriptor[key] for key in ["source_episode_id", "window", "source_split"]) \
                or (row["step"], row["slot"]) != (index // plan["batch"] + 1, index % plan["batch"]) \
                or not row["window"]["start_frame"] <= row["source_frame"] \
                or row["source_frame"] + plan["horizon"] > row["window"]["end_frame"]:
            raise ValueError("实际槽位偏离来源、窗口或训练顺序")
        key = (instructions[row["source_episode_id"]], row["kind"])
        counts[key] += 1
        windows[key].add((row["source_episode_id"], row["source_frame"]))
        sources[key].add(row["source_episode_id"])
    return [{"role": role, "instruction": task, "kind": kind, "sample_slots": value,
             "unique_windows": len(windows[(task, kind)]), "source_episodes": len(sources[(task, kind)]),
             "fraction_of_role": value / len(selected)} for (task, kind), value in sorted(counts.items())]


def fixed_frames(directory, episodes, probes):
    from PIL import Image, ImageDraw, ImageFont
    import imageio.v2 as imageio
    import subprocess

    font_path = subprocess.check_output(["fc-match", "-f", "%{file}", "Noto Sans CJK SC"], text=True)
    fonts = {size: ImageFont.truetype(font_path, size) for size in [17, 20, 25]}
    canvas = Image.new("RGB", (1320, 1100), "#f5f4ef")
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 15), "完整候选评测诊断：固定回合、同一仿真时刻", font=fonts[25], fill="#20313c")
    draw.text((20, 55), "每条件按ID取首回合；展示第80步双视角，不按成败选图。", font=fonts[20], fill="#45545d")
    inputs = []
    indexes = {role: {r["eval_id"]: r for r in rows} for role, rows in episodes.items()}
    for line, condition in enumerate(["standard", "lighting", "background"]):
        identity = min(key for key, row in indexes["incumbent"].items()
                       if row["condition_id"] == "t00-" + condition)
        y = 100 + line * 310
        draw.text((15, y + 35), "t00 / " + condition, font=fonts[20], fill="#20313c")
        for column, (role, label) in enumerate([("incumbent", "原策略"), ("candidate", "本圈候选")]):
            row = indexes[role][identity]
            report, probe = probes[role]
            x = 280 + column * 512
            outcome = "成功" if row["outcome"]["success"] else "失败"
            draw.text((x + 5, y), f"{label} / 最终{outcome} / 实际{row['metrics']['steps']}步",
                      font=fonts[17], fill="#20313c")
            if row["metrics"]["steps"] <= 80:
                draw.text((x + 25, y + 135), "第80步前回合已结束；此处为结束卡", font=fonts[17], fill="#6c4537")
                continue
            for view, name in enumerate(["agentview", "wrist"]):
                video = pathlib.Path(row["artifacts"]) / (name + ".mp4")
                video = bound_file(report.parent, probe, str(video.relative_to(report.parent)))
                with imageio.get_reader(video) as reader:
                    image = Image.fromarray(reader.get_data(80))
                if image.size != (256, 256):
                    raise ValueError("真实冻结视频尺寸偏离256×256")
                canvas.paste(image, (x + view * 256, y + 30))
                inputs.append({"role": role, "eval_id": identity, "path": str(video),
                               "sha256": sha256(video), "frame": 80, "view": name})
    draw.text((20, 1040), "Thor / 既有MAXN / VLM bf16、LoRA与动作头fp32 / sdpa / 推理batch1 / 动作块8",
              font=fonts[17], fill="#45545d")
    draw.text((20, 1070), "评测噪声种子34001、34002；最多520步。图片禁止入训，正式门禁仍须完整对照结果。",
              font=fonts[17], fill="#6c4537")
    canvas.save(directory / "fixed_frames.png")
    return inputs


def fixed_videos(directory, episodes, probes):
    """展示同初态、同策略噪声的完整轨迹；不补造结束后的机器人画面。"""
    from PIL import Image, ImageDraw, ImageFont
    import imageio.v2 as imageio
    import numpy as np
    import subprocess
    from p0_video_materials import video_info

    font_path = subprocess.check_output(["fc-match", "-f", "%{file}", "Noto Sans CJK SC"], text=True)
    fonts = {size: ImageFont.truetype(font_path, size) for size in [18, 23, 27]}
    indexes = {role: {r["eval_id"]: r for r in rows} for role, rows in episodes.items()}
    result = []
    for condition in ["standard", "lighting", "background"]:
        identity = min(key for key, row in indexes["incumbent"].items()
                       if row["condition_id"] == "t00-" + condition)
        rows = [indexes[role][identity] for role in ["incumbent", "candidate"]]
        inputs, paths = [], []
        for role, row in zip(["incumbent", "candidate"], rows):
            report, probe = probes[role]
            for view in ["agentview", "wrist"]:
                path = pathlib.Path(row["artifacts"]) / (view + ".mp4")
                path = bound_file(report.parent, probe, str(path.relative_to(report.parent)))
                info = video_info(path)
                if info["frame_count"] != row["metrics"]["steps"] or (info["width"], info["height"]) != (256, 256):
                    raise ValueError("完整展示视频的真实帧数与环境步数或尺寸不符")
                inputs.append({"role": role, "path": str(path), "sha256": sha256(path), **info})
                paths.append(path)
        output = directory / ("candidate_" + condition + ".mp4")
        count = max(row["metrics"]["steps"] for row in rows)
        with ExitStack() as stack:
            readers = [stack.enter_context(imageio.get_reader(path)) for path in paths]
            writer = stack.enter_context(imageio.get_writer(output, fps=20, codec="libx264", pixelformat="yuv420p",
                macro_block_size=1, ffmpeg_params=["-crf", "18", "-threads", "2", "-movflags", "+faststart"]))
            for frame in range(count):
                canvas = Image.new("RGB", (1280, 520), "#f5f4ef")
                draw = ImageDraw.Draw(canvas)
                draw.text((15, 9), f"完整候选退化诊断 / t00-{condition} / 固定回合", font=fonts[27], fill="#20313c")
                draw.text((15, 47), f"初态{rows[0]['init_seed']} / 策略种子{rows[0]['policy_seed']} / t={frame/20:.2f}s / 20Hz仿真时钟",
                          font=fonts[18], fill="#45545d")
                for column, (label, row) in enumerate(zip(["原策略", "本圈候选"], rows)):
                    x = column * 640
                    success = "成功" if row["outcome"]["success"] else "失败"
                    draw.text((x + 10, 80), f"{label} / 最终{success} / 实际{row['metrics']['steps']}步",
                              font=fonts[23], fill="#20313c")
                    if frame < row["metrics"]["steps"]:
                        for view in range(2):
                            image = Image.fromarray(readers[column * 2 + view].get_data(frame))
                            canvas.paste(image.resize((320, 320), Image.Resampling.LANCZOS), (x + view * 320, 116))
                    else:
                        draw.text((x + 35, 235), f"回合已结束：{success}", font=fonts[23], fill="#20313c")
                        draw.text((x + 35, 272), "结束卡不是后续机器人观测或动作", font=fonts[18], fill="#6c4537")
                draw.text((15, 447), "每条件固定ID首回合，不按成败选片；原策略5000步，候选追加1000步；训练batch8、seed42。",
                          font=fonts[18], fill="#45545d")
                draw.text((15, 475), "Thor既有MAXN / bf16 VLM、fp32 LoRA与动作头 / sdpa / 推理batch1 / 动作块8；展示禁止入训。",
                          font=fonts[18], fill="#6c4537")
                writer.append_data(np.asarray(canvas))
        info = video_info(output)
        if info["frame_count"] != count or info["codec_name"] != "h264" or info["pix_fmt"] != "yuv420p" \
                or (info["width"], info["height"]) != (1280, 520):
            raise ValueError("导出视频帧数、编码或尺寸核验失败")
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(output), "-f", "null", "-"], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=60)
        result.append({"file": output.name, "sha256": sha256(output), "eval_id": identity, "inputs": inputs,
                       "output": info, "frame_mapping": "真实输入同一帧；回合结束后显示结束卡，不循环、补帧或变速",
                       "selection": "每条件固定ID首回合，不按成败选片", "eligible_for_training": False})
    return result


def collect(directory, args):
    state_path = args.iteration_run / "iteration_state.json"
    state_bytes = state_path.read_bytes()
    state = json.loads(state_bytes)
    bindings = state["completed_stages"]
    paths = {}
    training, metadata = {}, {}
    for role in ["candidate", "control", "analytic_ablation"]:
        binding = bindings["train_" + role]
        path = pathlib.Path(binding["path"])
        if binding["sha256"] != sha256(path) or binding["stage"] != "p1_policy_train":
            raise ValueError("诊断训练不属于确切整圈")
        training[role], metadata[role] = checked_update(path, role)
        if training[role]["iteration"] != state["iteration"]:
            raise ValueError("诊断训练与当前圈数不符")
        training_trace(path.parent / "logs/steps.jsonl", training[role])
        paths[role] = path
    for key in ["parent_ckpt_sha256", "paired_plan_sha256", "steps", "batch", "seed"]:
        if len({training[role][key] for role in training}) != 1:
            raise ValueError("退化诊断的三角色没有匹配父权重、计划或训练预算")
    candidate = pathlib.Path(bindings["evaluate_candidate"]["path"])
    if bindings["evaluate_candidate"]["sha256"] != sha256(candidate):
        raise ValueError("完整候选评测与整圈摘要不符")
    incumbent = pathlib.Path(state["inputs"]["incumbent_evaluation"]["path"])
    if state["inputs"]["incumbent_evaluation"]["sha256"] != sha256(incumbent):
        raise ValueError("原策略冻结评测已改变")
    frozen_path = ROOT / "eval/frozen_eval_set.json"
    frozen = {**json.loads(frozen_path.read_text()), "sha256": sha256(frozen_path)}
    evaluations, episodes, probes, runtimes = {}, {}, {}, {}
    for role, path in [("incumbent", incumbent), ("candidate", candidate)]:
        expected = training["candidate"]["parent_ckpt_sha256" if role == "incumbent" else "output_ckpt_sha256"]
        evaluations[role], runtimes[role] = checked_evaluation(path, expected, frozen)
        _, probe = sealed(path, "policy_evaluation")
        source = bound_file(path.parent, probe, "episodes.jsonl")
        episodes[role] = [json.loads(line) for line in source.read_text().splitlines()]
        probes[role] = path, probe
    if runtimes["incumbent"] != runtimes["candidate"]:
        raise ValueError("原策略与候选的实际运行条件不一致")
    execution_code = {}
    for name in ["p0_evaluate_policy.py", "p0_policy_runtime.py", "p0_policy_server.py",
                 "p0_eval_catalog.py", "p0_common.py", "p0_freeze_eval.py"]:
        checksums = {probes[role][1]["source_files_sha256"]["scripts/" + name] for role in probes}
        if len(checksums) != 1:
            raise ValueError("候选和原策略的评测执行代码不同，须先排查运行实现变化")
        execution_code[name] = checksums.pop()
    summary, pairs = paired_outcomes(episodes["incumbent"], episodes["candidate"])
    write_csv(directory / "paired_outcomes.csv", pairs)
    write_csv(directory / "paired_summary.csv", summary)
    parent_path = pathlib.Path(state["inputs"]["parent"]["path"]) / "checkpoint.json"
    if sha256(parent_path) != state["inputs"]["parent"]["sha256"]:
        raise ValueError("原策略元数据不属于本圈输入")
    parent = json.loads(parent_path.read_text())
    cache_path = pathlib.Path(parent["cache_manifest"])
    if sha256(cache_path) != parent["cache_manifest_sha256"]:
        raise ValueError("采样覆盖引用的D0清单已改变")
    cache = json.loads(cache_path.read_text())
    proof = training["candidate"]["data_qualification"]
    collection_path = pathlib.Path(proof["collection"])
    collection, collection_probe = sealed(collection_path, "feedback_rollout")
    if sha256(collection_path) != proof["collection_sha256"]:
        raise ValueError("采样覆盖诊断引用了其他反馈")
    journal = bound_file(collection_path.parent, collection_probe,
                         str(pathlib.Path(collection["episodes"]).relative_to(collection_path.parent)))
    instructions = {row["episode_id"]: row["instruction"] for row in cache["episodes"]}
    instructions.update({row["episode_id"]: row["instruction"] for row in
                         map(json.loads, journal.read_text().splitlines())})
    plan_path = paths["candidate"].parent / "paired_plan.json"
    plan = json.loads(plan_path.read_text())
    exposure = [row for role in ["candidate", "control"] for row in exposure_counts(plan, role, instructions)]
    write_csv(directory / "sampling_exposure.csv", exposure)
    screenshots = fixed_frames(directory, episodes, probes)
    videos = fixed_videos(directory, episodes, probes) if args.videos else []

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
    figure, axis = plt.subplots(figsize=(11, 6), layout="constrained")
    labels = {"standard": "标准", "target_perturbed": "目标扰动", "nontarget_perturbed": "非目标扰动"}
    for offset, role, title, color in [(-.18, "incumbent", "原策略", "#477c91"),
                                       (.18, "candidate", "本圈候选", "#b34c42")]:
        values = [evaluations[role]["splits"][split] for split in SPLITS]
        xs = [i + offset for i in range(len(values))]
        axis.bar(xs, [100 * row["p"] for row in values], width=.34, label=title, color=color)
        for x, row in zip(xs, values):
            axis.text(x, 100 * row["p"] + .5, f"{row['success']}/{row['n']}", ha="center")
    axis.set(xticks=range(len(SPLITS)), xticklabels=[labels[s] for s in SPLITS],
             ylabel="环境成功率（%）", ylim=(0, 36), title="完整候选评测出现退化；匹配对照与正式门禁仍待完成")
    axis.legend()
    figure.supxlabel("Thor既有MAXN / bf16 VLM、fp32 LoRA与动作头 / sdpa / 训练batch8、seed42\n"
                     "原策略5000步；候选追加1000步。各1200回合 / 600初态 / 评测种子34001、34002 / 推理batch1", fontsize=9)
    figure.savefig(directory / "candidate_comparison.png", dpi=220)
    plt.close(figure)
    (directory / "iteration_state_snapshot.json").write_bytes(state_bytes)
    result = {"schema": "thor-candidate-diagnostics-v1", "iteration": state["iteration"],
        "iteration_state_sha256": sha256(directory / "iteration_state_snapshot.json"),
        "source_state_phase": state["phase"], "comparison": summary, "exposure": exposure,
        "matching_execution_code_sha256": execution_code,
        "evaluations": {role: {"path": str(path), "sha256": sha256(path)} for role, path in
                        [("incumbent", incumbent), ("candidate", candidate)]},
        "training": {role: {"path": str(path), "sha256": sha256(path), "recipe": training[role]["recipe"]}
                     for role, path in paths.items()}, "fixed_frame_sources": screenshots, "videos": videos,
        "causal_attribution_established": False, "control_evaluation_used": False,
        "formal_promotion_gate_executed": False, "production_deployed": False,
        "eligible_for_training": False, "complete_rsi": False,
        "boundary": "完整冻结评测用于报告退化；不得据测试回合生成训练样本、定向改提示或挑选训练检查点。"
                    "采样偏重和优化器/学习率变化只是待验证假设；后续用独立开发集排查。"}
    dump(directory / "diagnostics.json", result)
    return {"candidate_episodes": len(episodes["candidate"]),
            "comparison": [row for row in summary if row["group"].startswith("split:")],
            "causal_attribution_established": False, "complete_rsi": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--iteration-run", type=pathlib.Path, required=True)
    parser.add_argument("--videos", action="store_true", help="额外用CPU导出三个固定回合的完整双策略对比")
    args = parser.parse_args()
    args.iteration_run = args.iteration_run.resolve(strict=True)
    if not args.iteration_run.is_relative_to(ROOT / "runs/p0"):
        parser.error("诊断只接受Thor本项目的明确整圈目录")
    raise SystemExit(run_stage("p1_update_diagnostics", lambda directory: collect(directory, args),
        {"compute": "Thor CPU，CUDA禁用；只读取已封存结果", "new_model_inference": False}))
