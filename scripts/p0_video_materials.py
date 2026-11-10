#!/usr/bin/env python3
"""把已有真实回合及Cosmos结果编成可追溯的展示视频，保持源帧率与完整时序。

@author pony
@date 2026-11-07
@version v1.0.1
@last_modified 2026-11-07
@changelog
    - v1.0.1 (2026-11-07): 视频叠字使用π0，避免中文字体缺少下标零而显示方框。
    - v1.0.0 (2026-11-07): 导出生成对比、基线双视角与分割提示对比；不声称策略前后收益。
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from fractions import Fraction
import itertools
import json
import pathlib
import subprocess

from p0_common import ROOT, run_stage
from p0_policy_runtime import sha256


def video_info(path: pathlib.Path) -> dict:
    result = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                             "-show_entries", "stream=codec_name,pix_fmt,width,height,r_frame_rate,nb_read_frames",
                             "-of", "json", str(path)], check=True, capture_output=True, text=True, timeout=60)
    streams = json.loads(result.stdout)["streams"]
    if len(streams) != 1:
        raise ValueError("视频必须恰有一条可解码的视频流")
    value = streams[0]
    value["frame_count"] = int(value["nb_read_frames"])
    value["fps"] = float(Fraction(value["r_frame_rate"]))
    if value["frame_count"] < 1 or Fraction(value["r_frame_rate"]) != 20:
        raise ValueError("展示只接收实际20fps且非空的源视频，禁止隐式改速")
    return value


def render(directory: pathlib.Path, filename: str, videos: list[pathlib.Path], labels: list[str],
           title: str, subtitle: str, footer: list[str], count: int, metadata: dict) -> dict:
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    inputs = [{"path": str(path), "sha256": sha256(path), **video_info(path)} for path in videos]
    if len(videos) != len(labels) or any(item["frame_count"] != count for item in inputs):
        raise ValueError("同帧对比必须等长，禁止截断或补齐源视频")
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    fonts = [ImageFont.truetype(font_path, size, index=2) for size in [28, 20, 17]]
    template = Image.new("RGB", (480 * len(videos), 672), "#f5f4ef")
    draw = ImageDraw.Draw(template)
    draw.text((18, 10), title, font=fonts[0], fill="#20313c")
    draw.text((18, 49), subtitle, font=fonts[2], fill="#45545d")
    for column, label in enumerate(labels):
        draw.text((column * 480 + 15, 76), label, font=fonts[1], fill="#20313c")
    for line, text in enumerate(footer):
        draw.text((18, 591 + line * 24), text, font=fonts[2], fill="#6c4537")
    output = directory / filename
    with ExitStack() as stack:
        readers = [stack.enter_context(imageio.get_reader(path)) for path in videos]
        writer = stack.enter_context(imageio.get_writer(output, fps=20, codec="libx264", pixelformat="yuv420p",
                                                        macro_block_size=1, ffmpeg_params=["-crf", "18", "-threads", "2", "-movflags", "+faststart"]))
        seen = 0
        for index, frames in enumerate(itertools.zip_longest(*readers)):
            if any(frame is None for frame in frames) or index >= count:
                raise ValueError("实际解码长度与逐帧对齐约定不一致")
            canvas = template.copy()
            for column, frame in enumerate(frames):
                if frame.shape[0] != frame.shape[1]:
                    raise ValueError("当前展示版式只支持方形源帧，不能拉伸非方形输入")
                canvas.paste(Image.fromarray(frame).resize((480, 480), Image.Resampling.LANCZOS), (column * 480, 104))
            ImageDraw.Draw(canvas).text((canvas.width - 246, 48), f"帧 {index:03d} / {count - 1}  {index / 20:.2f}s",
                                        font=fonts[2], fill="#20313c")
            if index == 0:
                canvas.save(output.with_suffix(".jpg"), quality=95)
            writer.append_data(np.asarray(canvas))
            seen += 1
    actual = video_info(output)
    if seen != count or actual["frame_count"] != count or actual["codec_name"] != "h264" \
            or actual["pix_fmt"] != "yuv420p" or actual["width"] != template.width or actual["height"] != template.height:
        raise RuntimeError("展示输出未通过实际帧数、尺寸和通用播放编码检查")
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(output), "-f", "null", "-"], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=60)
    return {"file": filename, "poster": output.with_suffix(".jpg").name, "title": title, "subtitle": subtitle,
            "inputs": inputs, "output": actual, "sha256": sha256(output), "bytes": output.stat().st_size,
            "duration_s": count / 20, "frame_mapping": "output[t]的每列来自对应输入[t]，不补帧、不截断、不变速",
            "notes": footer, **metadata}


def calibration_video(directory: pathlib.Path, calibration_path: pathlib.Path, source_sha: str) -> dict:
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    report = json.loads(calibration_path.read_text())
    probe = json.loads((calibration_path.parent / "probe.json").read_text())
    if report["source_input_sha256"] != source_sha or report["frames"] != list(range(93)) \
            or not report["balanced_calibration_passed"] or probe["status"] != "passed":
        raise RuntimeError("分割提示视频必须绑定同一成功源和全部已通过校准的帧")
    font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 20, index=2)
    path = directory / "07_segmentation_prompt_before_after.mp4"
    inputs = []
    with imageio.get_writer(path, fps=20, codec="libx264", pixelformat="yuv420p", macro_block_size=1,
                           ffmpeg_params=["-crf", "18", "-threads", "2", "-movflags", "+faststart"]) as writer:
        for index in range(93):
            source = calibration_path.parent / f"calibration_{index:03d}.png"
            digest = sha256(source)
            if digest != probe["artifacts_sha256"][source.name]:
                raise RuntimeError("源校准截图已经改变")
            with Image.open(source) as image:
                if image.size != (1024, 290):
                    raise ValueError("源校准截图版式与记录不一致")
                canvas = Image.new("RGB", (1024, 370), "#f5f4ef")
                canvas.paste(image, (0, 40))
            draw = ImageDraw.Draw(canvas)
            draw.text((12, 7), "分割提示优化前后：同一成功示范，不代表策略收益", font=font, fill="#20313c")
            draw.text((12, 338), "原阈值0.75；左起：源图、仿真真值、原三点提示、12正点+12负点", font=font, fill="#6c4537")
            if index == 0:
                canvas.save(path.with_suffix(".jpg"), quality=95)
            writer.append_data(np.asarray(canvas))
            inputs.append({"path": str(source), "sha256": digest})
    info = video_info(path)
    if info["frame_count"] != 93 or info["width"] != 1024 or info["height"] != 370 \
            or info["codec_name"] != "h264" or info["pix_fmt"] != "yuv420p":
        raise RuntimeError("源校准对比视频的时序或播放编码不合格")
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=60)
    return {"file": path.name, "poster": path.with_suffix(".jpg").name, "title": "分割提示优化前后",
            "inputs": inputs, "output": info, "sha256": sha256(path), "bytes": path.stat().st_size,
            "duration_s": 93 / 20, "calibration": str(calibration_path),
            "calibration_sha256": sha256(calibration_path), "kind": "source_segmentation_diagnostic",
            "hardware": probe["hardware"], "conditions": {key: report[key] for key in ["precision", "batch", "seed", "sample_count"]},
            "frame_mapping": "output[t]来自真实成功示范的calibration_t.png；20fps，不补帧或变速",
            "eligible_for_training": False, "policy_improvement_claimed": False}


def collect(directory: pathlib.Path, args: argparse.Namespace) -> dict:
    probes = [json.loads(path.read_text()) for path in [args.base_probe, args.distilled_probe]]
    if any(probe["stage"] != "transfer_smoke" or probe["status"] != "passed" for probe in probes):
        raise RuntimeError("只允许展示已经实际完成的生成记录")
    base, distilled = [probe["result"] for probe in probes]
    evidence = [json.loads(pathlib.Path(result["source_success_evidence"]["path"]).read_text())
                for result in [base, distilled]]
    if any(not item.get("source_success") or not item.get("replay_validation", {}).get("passed") for item in evidence) \
            or evidence[0]["source_rgb_sha256"] != evidence[1]["source_rgb_sha256"] \
            or evidence[0]["labels_sha256"] != evidence[1]["labels_sha256"]:
        raise RuntimeError("生成对比必须来自同一条已验证成功示范和同一动作/状态窗口")
    for source in evidence:
        for name, digest in source["files_sha256"].items():
            path = pathlib.Path(source["spec"]).parent / name
            if sha256(path) != digest:
                raise RuntimeError("生成对比的源图、edge或掩码摘要已改变")
    spec = json.loads(pathlib.Path(base["source_spec"]).read_text())
    source = pathlib.Path(spec["video_path"])
    generated = pathlib.Path(base["outputs"][0])
    clips = []
    captions = ["源/edge原生256×256，展示放大；生成原生480×480。20fps逐帧对应。",
                "现有生成样本仍拒绝入训；视频不代表策略改善或目标光照已经达成。",
                "4.65秒为93帧播放时长；生成墙钟时间和完整条件见来源JSON。"]
    clips.append(render(directory, "01_cosmos_source_edge_base.mp4", [source, pathlib.Path(spec["edge"]["control_path"]), generated],
                        ["成功示范源图", "RGB边缘控制", f"Cosmos {base['variant']} / {base['num_steps']}步"],
                        "Cosmos生成：同一成功示范的源图、控制与结果", "Thor / 既有MAXN / DiT bf16 / batch1 / seed42 / 单窗口93帧",
                        captions, 93, {"kind": "generation_comparison", "is_eval": False, "eligible_for_training": False,
                                       "generation_probe": str(args.base_probe), "generation_probe_sha256": sha256(args.base_probe)}))
    if not args.smoke:
        clips.append(render(directory, "02_cosmos_source_distilled_base.mp4", [source, pathlib.Path(distilled["outputs"][0]), generated],
                            ["同一成功示范源图", f"{distilled['variant']} / {distilled['num_steps']}步", f"{base['variant']} / {base['num_steps']}步"],
                            "Cosmos生成对照：四步蒸馏与base", "Thor / 既有MAXN / DiT bf16 / batch1 / seed42 / 单窗口93帧",
                            ["同源、同种子；模型、步数和空间引导均有差异，不能只归因于某个因素。", *captions[:2]], 93,
                            {"kind": "generation_comparison", "is_eval": False, "eligible_for_training": False,
                             "generation_probes": [{"path": str(path), "sha256": sha256(path)}
                                                   for path in [args.base_probe, args.distilled_probe]]}))
        summary_path = args.evaluation / "evaluation.json"
        summary = json.loads(summary_path.read_text())
        evaluation_probe = json.loads((args.evaluation / "probe.json").read_text())
        if evaluation_probe["status"] != "passed" or not summary["full_eval_complete"]:
            raise RuntimeError("只收录已经完成的正式基线回合")
        episodes = [json.loads(line) for line in (args.evaluation / "episodes.jsonl").read_text().splitlines()]
        wanted = [("t00", "取书放入收纳盒后格", 100000), ("t02", "汤罐与奶酪盒放进篮子", 102000)]
        for task, task_name, init_seed in wanted:
            selected = sorted([row for row in episodes if row["condition_id"] == task + "-standard"
                               and row["init_seed"] == init_seed], key=lambda row: row["policy_seed"])
            if len(selected) != 2 or {row["outcome"]["success"] for row in selected} != {False, True} \
                    or len({row["initial_state_sha256"] for row in selected}) != 1:
                raise RuntimeError("选定素材不是同初态、不同策略噪声的成功/失败基线对")
            for episode in selected:
                if episode["is_eval"] is not True or episode["eligible_for_training"] is not False \
                        or episode["outcome"]["source"] != "env" or episode["policy_ckpt_sha256"] != summary["policy_ckpt_sha256"]:
                    raise RuntimeError("基线回合身份或环境真值不符合记录")
                truth = "成功" if episode["outcome"]["success"] else "失败"
                filename = f"{len(clips) + 1:02d}_policy_{task}_{'success' if episode['outcome']['success'] else 'failure'}.mp4"
                artifacts = pathlib.Path(episode["artifacts"])
                clips.append(render(directory, filename, [artifacts / "agentview.mp4", artifacts / "wrist.mp4"],
                                    ["外部视角 agentview", "腕部视角 wrist"], f"π0基线{truth}案例：{task_name}",
                                    f"训练5000步 / batch8 / seed42；推理batch1 / policy_seed {episode['policy_seed']}",
                                    ["Thor / 既有MAXN / VLM bf16、LoRA与动作头fp32；同一π0，没有策略训练前后对比。",
                                     f"环境最终真值：{truth}；256×256原画展示放大。评测视频禁止入训或作生成源。",
                                     "20fps仿真时钟回放，不是Thor墙钟实时推理；不剪切、不补帧、不变速。"],
                                    episode["metrics"]["steps"], {"kind": "policy_baseline", "episode": episode,
                                    "episode_record_sha256": sha256(artifacts / "episode.json"), "is_eval": True,
                                    "eligible_for_training": False, "policy_improvement_claimed": False}))
        if args.calibration:
            clips.append(calibration_video(directory, args.calibration,
                                            sha256(pathlib.Path(base["source_success_evidence"]["path"]))))
    report = {"source_probes": [{"path": str(path), "sha256": sha256(path)}
                                for path in [args.base_probe, args.distilled_probe]],
              "clips": clips, "clip_count": len(clips), "compute": "Thor CPU解码、PIL排版和libx264编码；不运行模型",
              "precision": "解码RGB uint8，展示有损H264/yuv420p；实验模型精度另列",
              "sample_scope": "展示样本挑选用于看效果，不用于估计总体成功率",
              "presentation_created": False, "policy_improvement_claimed": False, "eligible_for_training": False}
    (directory / "video_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    lines = ["# 真实实验展示视频", "", "逐帧保留源视频时序，H264/yuv420p，20fps，无音轨。播放时长不是模型推理墙钟。", "",
             "基线成功/失败案例来自同一π₀及同一初态，不是训练前后实验。Cosmos工程样本仍拒绝入训。", ""]
    for clip in clips:
        lines += [f"- [{clip['title']}]({clip['file']})：{clip['output']['frame_count']}帧，{clip['duration_s']:.2f}秒。"]
    (directory / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"directory": str(directory), "clips": [clip["file"] for clip in clips]}, ensure_ascii=False), flush=True)
    return {"materials": str(directory), "clip_count": len(clips), "eligible_for_training": False,
            "presentation_created": False, "policy_improvement_claimed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-probe", type=pathlib.Path, required=True)
    parser.add_argument("--distilled-probe", type=pathlib.Path, required=True)
    parser.add_argument("--evaluation", type=pathlib.Path, required=True)
    parser.add_argument("--calibration", type=pathlib.Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    for name in ["base_probe", "distilled_probe", "evaluation", "calibration"]:
        if getattr(args, name) is None:
            continue
        value = getattr(args, name).resolve(strict=True)
        if not value.is_relative_to(ROOT / "runs/p0"):
            parser.error("视频素材来源必须位于Thor的runs/p0")
        setattr(args, name, value)
    raise SystemExit(run_stage("video_materials", lambda directory: collect(directory, args),
                               {"compute": "CPU_only", "smoke": args.smoke}))
