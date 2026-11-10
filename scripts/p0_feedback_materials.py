#!/usr/bin/env python3
"""导出独立采集的真实终帧、逐回合CSV和成败双视角视频，不生成模拟结果。

@author pony
@date 2026-11-09
@version v1.2.0
@last_modified 2026-11-09
@changelog
    - v1.2.0 (2026-11-09): 卡片使用固定任务短号，模型与精度来自实测报告，避免长路径溢出。
    - v1.1.0 (2026-11-09): 十任务采集按八回合分页，不截断样本；保留完整CSV和每页回合ID。
    - v1.0.0 (2026-11-09): 从绑定采集及判官报告导出4K总览和展示视频。
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import subprocess

from p0_common import ROOT, run_stage
from p0_feedback_judge import checked_collection
from p0_lerobot_handoff import dump
from p0_policy_runtime import sha256


def collect(directory, args):
    from PIL import Image, ImageDraw, ImageFont

    rows, inputs, conditions = [], [], []
    for collection, judge_path in zip(args.collection, args.judge):
        source, episodes, _ = checked_collection(collection)
        probe = json.loads((judge_path.parent / "probe.json").read_text())
        report = json.loads(judge_path.read_text())
        verdicts = pathlib.Path(report["verdicts"])
        if probe.get("stage") != "feedback_judge" or probe.get("status") != "passed" \
                or probe["artifacts_sha256"].get(judge_path.name) != sha256(judge_path) \
                or report["collection_sha256"] != sha256(collection) \
                or verdicts.parent != judge_path.parent or probe["artifacts_sha256"].get(verdicts.name) != sha256(verdicts):
            raise ValueError("素材未绑定实际采集和判官输出")
        judgments = {row["episode_id"]: row for row in map(json.loads, verdicts.read_text().splitlines())}
        for episode in episodes:
            judgment = judgments[episode["episode_id"]]
            if judgment["environment_truth"] != episode["outcome"]["success"]:
                raise ValueError("展示真值错绑")
            terminal = {item["view"]: item for item in judgment["inputs"] if item["phase"] == "terminal"}
            for item in terminal.values():
                if sha256(pathlib.Path(item["path"])) != item["sha256"]:
                    raise ValueError("实际终帧被替换")
            rows.append({"episode_id": episode["episode_id"], "task_id": episode["task_id"],
                "condition_id": episode["condition_id"],
                "instruction": episode["instruction"], "condition": episode["condition"],
                "init_seed": episode["init_seed"], "policy_seed": episode["policy_seed"],
                "environment_success": episode["outcome"]["success"], "judge_success": judgment["verdict"]["success"],
                "judge_valid": judgment.get("valid", True),
                "steps": episode["metrics"]["steps"], "duration_s": episode["metrics"]["duration_s"],
                "action_dispersion_p90": episode["metrics"]["action_dispersion_p90"],
                "judge_evidence": judgment["verdict"]["evidence"], "calibration_passed": report["calibration"]["calibration_passed"],
                "terminal": terminal, "artifacts": episode["artifacts"]})
        conditions.append({"collection": str(collection), "collection_sha256": sha256(collection),
            "judge": str(judge_path), "judge_sha256": sha256(judge_path),
            "judge_model": pathlib.Path(report["model"]).name,
            "judge_precision": report["precision"], "judge_response_mode": report.get("response_mode", "structured_json"),
            **{key: source[key] for key in ["policy_ckpt_sha256", "training_steps", "training_seed", "precision",
                                          "batch", "max_steps", "init_seeds", "policy_seeds", "config"]}})
        inputs.append({"collection": str(collection), "judge": str(judge_path)})
    if not rows or len({row["episode_id"] for row in rows}) != len(rows) \
            or len({value["policy_ckpt_sha256"] for value in conditions}) != 1:
        raise ValueError("总览需要同一策略、不重复的实际回合")
    scale = 2
    font_file = subprocess.check_output(["fc-match", "-f", "%{file}", "Noto Sans CJK SC"], text=True)
    fonts = {size: ImageFont.truetype(font_file, size * scale) for size in [17, 18, 20, 24, 36]}
    names = {"standard": "标准", "lighting": "光照", "background": "桌面纹理"}
    pages = []
    for first in range(0, len(rows), 8):
        page_rows = rows[first:first + 8]
        canvas = Image.new("RGB", (3840, 2160), "white")
        drawing = ImageDraw.Draw(canvas)
        def text(value, x, y, size=20, color="#263C48"):
            drawing.text((x * scale, y * scale), value, font=fonts[size], fill=color)
        judge_names = "/".join(sorted({r["judge_model"].removeprefix("Cosmos-") for r in conditions}))
        text(f"独立运行反馈：真实成败 × {judge_names} 关键帧复核", 70, 40, 36)
        text(f"LIBERO 十任务 · Thor / 既有 MAXN · 正式初训 5000 步、batch8、seed42 · 第 {first // 8 + 1} 页", 70, 98, 20)
        precision = "FP8 权重 / bf16 激活" if conditions[0]["judge_precision"].startswith("FP8") else "bf16 权重与激活"
        text(f"运行 batch1、动作块8；离散度按配置抽测，每次3种子；判官 {precision}、batch1、seed42、六图", 70, 130, 18)
        for index, row in enumerate(page_rows):
            x, y = 70 + 445 * (index % 4), 180 + 360 * (index // 4)
            drawing.rounded_rectangle((x * scale, y * scale, (x + 430) * scale, (y + 335) * scale),
                                       radius=12 * scale, fill="#F5F8FA", outline="#D9E2E7", width=2)
            task_short = row["condition_id"].split("-", 1)[0]
            text(f"{task_short} · {names.get(row['condition'], row['condition'])}", x + 15, y + 12, 24)
            truth = "成功" if row["environment_success"] else "失败"
            opinion = "弃权" if not row["judge_valid"] else "成功" if row["judge_success"] else "失败"
            text(f"环境：{truth}    Reason：{opinion}    s{row['init_seed']}", x + 15, y + 49, 18,
                 "#39762C" if row["environment_success"] else "#9B5921")
            for offset, view in enumerate(["agentview", "wrist"]):
                with Image.open(row["terminal"][view]["path"]) as image:
                    resized = image.convert("RGB").resize((192 * scale, 192 * scale), Image.Resampling.LANCZOS)
                canvas.paste(resized, ((x + 15 + offset * 208) * scale, (y + 80) * scale))
            text("外部终帧", x + 15, y + 275, 17)
            text("腕部终帧", x + 223, y + 275, 17)
            text(f"{row['steps']} 步 · 动作离散度 p90={row['action_dispersion_p90']:.3f}", x + 15, y + 307, 17)
        text("这些是独立采集样本，尚未用于证明策略更新收益；判官校准需覆盖足量成功和失败。", 70, 935, 20)
        text("终帧为最后动作后的真实观测；训练存档保留动作前观测。动作离散度是归一化动作标准差，不是成功概率。", 70, 975, 18)
        text("任务原文及回合ID见CSV；原始轨迹与无损LeRobot保存在Thor；展示图片和视频不用于训练。", 70, 1015, 18)
        filename = "feedback_overview.png" if first == 0 else f"feedback_overview_{first // 8 + 1:02d}.png"
        canvas.save(directory / filename, optimize=True)
        pages.append({"file": filename, "episode_ids": [r["episode_id"] for r in page_rows]})
    fields = [key for key in rows[0] if key not in {"terminal", "artifacts"}]
    with (directory / "feedback_episodes.csv").open("x", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)
    clips = []
    for success in [True, False]:
        selected = next((row for row in rows if row["environment_success"] is success), None)
        if selected is None:
            continue
        name = "feedback_success.mp4" if success else "feedback_failure.mp4"
        source = pathlib.Path(selected["artifacts"])
        command = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(source / "agentview.mp4"),
            "-i", str(source / "wrist.mp4"), "-filter_complex", "[0:v][1:v]hstack=inputs=2[v]", "-map", "[v]",
            "-c:v", "libx264", "-crf", "18", "-preset", "fast", "-threads", "2", str(directory / name)]
        subprocess.run(command, check=True, timeout=120, capture_output=True)
        metadata = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,nb_read_frames,avg_frame_rate", "-of", "json", str(directory / name)], text=True))["streams"][0]
        if int(metadata["nb_read_frames"]) != selected["steps"] or metadata["avg_frame_rate"] != "20/1":
            raise ValueError("展示视频帧数或帧率偏离实际轨迹")
        clips.append({"file": name, "episode_id": selected["episode_id"], "source_environment_success": success,
                      "metadata": metadata, "format": "有损双视角拼接展示，不用于训练"})
    manifest = {"schema": "thor-feedback-materials-v1", "sources": inputs, "conditions": conditions,
        "sample_count": len(rows), "successes": sum(row["environment_success"] for row in rows),
        "AI_generated": False, "clips": clips, "overview_pages": pages, "figure_dimensions": [3840, 2160],
        "files_sha256": {path.name: sha256(path) for path in directory.iterdir()
                         if path.is_file() and path.suffix in {".png", ".csv", ".mp4"}},
        "claim": "真实独立采集和判官反馈素材，不是完整闭环或RSI收益证明"}
    dump(directory / "materials_manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--collection", type=pathlib.Path, action="append", required=True)
    parser.add_argument("--judge", type=pathlib.Path, action="append", required=True)
    args = parser.parse_args()
    if len(args.collection) != len(args.judge):
        parser.error("collection与judge必须一一对应")
    args.collection = [path.resolve(strict=True) for path in args.collection]
    args.judge = [path.resolve(strict=True) for path in args.judge]
    if any(not path.is_relative_to(ROOT / "runs/p0") for path in args.collection + args.judge):
        parser.error("素材来源必须在本项目P0目录")
    raise SystemExit(run_stage("feedback_materials", lambda directory: collect(directory, args),
        {"compute": "Thor CPU/Pillow/ffmpeg；无新模型计算", "scope": "独立运行反馈真实素材"}))
