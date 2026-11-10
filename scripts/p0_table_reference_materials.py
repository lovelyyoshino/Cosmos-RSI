#!/usr/bin/env python3
"""保存材质参考试验的完整预览、逐帧证据和关键展示视频，始终禁止入训。

@author pony
@date 2026-11-08
@version v1.0.0
@last_modified 2026-11-08
@changelog
    - v1.0.0 (2026-11-08): 源/参考/原生/派生同帧对比，保留保护像素、桌面时序与实际来源。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib

from p0_background_preview_materials import checked_artifact
from p0_common import ROOT, ensure_gpu_idle, run_stage
from p0_policy_runtime import sha256


def native_video(probe_path, expected_stage):
    probe = json.loads(probe_path.read_text())
    if probe.get("stage") != expected_stage or probe.get("status") != "passed":
        raise ValueError("只允许查看实际完成的原生生成")
    outputs = probe.get("result", {}).get("outputs", [])
    if len(outputs) != 1:
        raise ValueError("原生阶段必须绑定唯一实际视频")
    output = pathlib.Path(outputs[0]).resolve(strict=True)
    if not output.is_relative_to(probe_path.parent) or sha256(output) != probe["artifacts_sha256"].get(
            str(output.relative_to(probe_path.parent))):
        raise ValueError("原生视频未绑定实际阶段产物")
    return probe, output


def compare_repro(directory, args):
    ensure_gpu_idle()
    import numpy as np
    from p0_lossless_background import read_exact_video

    records, arrays = [], []
    for path, stage in [(args.baseline_probe, "transfer_smoke"), (args.observed_probe, "single_control_trace"),
                        (args.repeat_probe, "transfer_smoke")]:
        probe, video = native_video(path, stage)
        result = probe["result"]
        if result.get("recipe") != "edge/distilled" or result.get("num_steps") != 4 or result.get("seed") != 42:
            raise ValueError("复现比较必须保持同一蒸馏四步/seed42配方")
        binding = result["source_success_evidence"]
        if records and binding != records[0]["source_binding"]:
            raise ValueError("复现比较的源证据或标签不相同")
        frames, _ = read_exact_video(video, size=None)
        if arrays and frames.shape != arrays[0].shape:
            raise ValueError("复现比较改变了实际帧数或视频尺寸")
        arrays.append(frames)
        records.append({"probe": str(path), "probe_sha256": sha256(path), "video": str(video),
                        "video_sha256": sha256(video), "source_binding": binding,
                        "stage_wall_s": probe["wall_s"], "peak_mem_gb": probe["peak_mem_gb"]})
    rows, pairs = [], []
    for left, right, name in [(0, 1, "baseline_vs_observed"), (0, 2, "baseline_vs_repeat"), (1, 2, "observed_vs_repeat")]:
        difference = np.abs(arrays[left].astype(np.int16) - arrays[right].astype(np.int16))
        pair = {"pair": name, "decoded_pixels_identical": bool(not difference.any()),
                "max_abs_rgb_difference": int(difference.max()), "mean_abs_rgb_difference": float(difference.mean())}
        pairs.append(pair)
        rows += [{"pair": name, "frame": index, "max_abs_rgb_difference": int(frame.max()),
                  "mean_abs_rgb_difference": float(frame.mean())} for index, frame in enumerate(difference)]
    with (directory / "repro_all_frames.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    report = {"runs": records, "pairs": pairs, "compute": "Thor CPU uint8完整解码后int16差分；0–255单位",
              "conditions": "既有MAXN/DiT bf16/batch1/四步/seed42/同一成功源93帧20fps/原生480方形",
              "source_windows": 1, "independent_policy_evaluations": 0,
              "eligible_for_training": False, "diagnostic_only": True,
              "caveat": "同一设计窗口的复现性对照；样本量不足以估计随机性分布或孤立hook的因果效应"}
    (directory / "repro_comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return {"pairs": pairs, "eligible_for_training": False}


def diagnose(directory, args):
    ensure_gpu_idle()
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from p0_table_reference_sample import checked_input
    from p0_table_background import checked_table_source, compose_table, pixel_certificate
    from p0_prepare_table_background import write_video
    from p0_lossless_background import read_exact_video
    from p0_single_control_materials import static_table_changes

    probe, native_path = native_video(args.generation_probe, "table_reference_generation")
    result = probe["result"]
    reference_input = pathlib.Path(result["reference_input"])
    if sha256(reference_input) != result["reference_input_sha256"]:
        raise ValueError("材质参考输入发生变化")
    record, _ = checked_input(reference_input)
    source_path = pathlib.Path(record["source_input"])
    evidence, source, masks, editable, _, _ = checked_table_source(source_path)
    reference, _ = read_exact_video(reference_input.parent / "reference_rgb.mp4")
    native, _ = read_exact_video(native_path, size=None)
    resized = np.asarray([cv2.resize(frame, (256, 256), interpolation=cv2.INTER_AREA) for frame in native])
    preview = compose_table(source, resized, editable)
    preview_path = directory / "design_composite_lossless.mp4"
    write_video(preview_path, preview)
    actual, _ = read_exact_video(preview_path)
    certificate = pixel_certificate(source, resized, editable, actual, masks)
    if not certificate["geometry_passed"]:
        raise ValueError("实际设计预览没有保持全部非桌体及原生桌面像素")
    visible = editable.all(axis=0)
    temporal = []
    for kind, frames in [("source", source), ("reference", reference), ("native", resized), ("preview", actual)]:
        temporal += [{"kind": kind, **row} for row in static_table_changes(frames, visible)]
    for name, rows in [("pixel_certificate.csv", certificate["per_frame"]), ("static_table_temporal.csv", temporal)]:
        with (directory / name).open("x", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    font = ImageFont.truetype(font_path, 17, index=2)
    for frame in [0, 46, 92]:
        sheet = Image.new("RGB", (1024, 324), "#f6f4ee")
        draw = ImageDraw.Draw(sheet)
        for column, (label, pixels) in enumerate(zip(
                ["同一成功源", "确定性材质参考", "Cosmos原生", "仅桌面设计预览"],
                [source[frame], reference[frame], resized[frame], actual[frame]])):
            draw.text((column * 256 + 5, 5), label, font=font, fill="#23343e")
            sheet.paste(Image.fromarray(pixels), (column * 256, 32))
        draw.text((6, 296), f"帧{frame}；仅设计诊断，未通过独立方法/全部质量门，禁止入训；无RSI收益。", font=font, fill="#7a4237")
        sheet.save(directory / f"reference_actual_{frame:03d}.png")
    for page in range(8):
        indices = list(range(page * 12, min(93, (page + 1) * 12)))
        sheet = Image.new("RGB", (1024, 32 + len(indices) * 276), "#f6f4ee")
        draw = ImageDraw.Draw(sheet)
        draw.text((5, 5), "全部实际帧：成功源 / 材质参考 / Cosmos原生 / 仅桌面预览；不可入训", font=font, fill="#23343e")
        for row, index in enumerate(indices):
            y = 32 + row * 276
            draw.text((5, y), f"实际帧{index}", font=font, fill="#23343e")
            for column, frames in enumerate([source, reference, resized, actual]):
                sheet.paste(Image.fromarray(frames[index]), (column * 256, y + 20))
        sheet.save(directory / f"review_all_frames_{page:02d}.png")
    fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
    temporal_p95 = {}
    for kind in ["source", "reference", "native", "preview"]:
        rows = [row for row in temporal if row["kind"] == kind]
        values = [row["mean_abs_rgb_change_0_1"] for row in rows]
        temporal_p95[kind] = float(np.percentile(values, 95))
        ax.plot(range(1, 93), values, label=kind, linestyle="--" if kind == "preview" else "-")
    ax.set(xlabel="Actual frame / 20fps", ylabel="Adjacent mean absolute RGB change / 255",
           title="Common visible table / same source / 93 frames; diagnostic only")
    ax.legend(); ax.grid(alpha=.2)
    for extension in ["png", "svg"]:
        fig.savefig(directory / ("reference_temporal." + extension), dpi=180)
    plt.close(fig)
    summary = {"generation_probe": str(args.generation_probe), "generation_probe_sha256": sha256(args.generation_probe),
               "reference_input": str(reference_input), "reference_input_sha256": sha256(reference_input),
               "source_input": str(source_path), "source_input_sha256": sha256(source_path),
               "source_episode_id": evidence["source_episode_id"], "native_video": str(native_path),
               "native_video_sha256": sha256(native_path), "preview_video": str(preview_path),
               "preview_video_sha256": sha256(preview_path), "protected_pixels_preserved": True,
               "pixel_certificate": certificate, "common_visible_table_pixels": int(visible.sum()),
               "adjacent_mean_rgb_change_p95": temporal_p95, "diagnostic_only": True,
               "eligible_for_training": False, "policy_improvement_claimed": False,
               "reason": "设计源完整像素量测；尚无新配方独立方法及全帧身份/材质/时序/边界验收"}
    (directory / "reference_preview.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return {key: summary[key] for key in ["protected_pixels_preserved", "adjacent_mean_rgb_change_p95",
                                         "eligible_for_training", "policy_improvement_claimed"]}


def publish(directory, args):
    ensure_gpu_idle()
    from p0_video_materials import render

    checked_artifact(args.preview_report, "table_reference_preview")
    preview = json.loads(args.preview_report.read_text())
    generation_probe = pathlib.Path(preview["generation_probe"])
    if sha256(generation_probe) != preview["generation_probe_sha256"]:
        raise ValueError("设计预览的原生阶段发生变化")
    generation, native = native_video(generation_probe, "table_reference_generation")
    derived = pathlib.Path(preview["preview_video"])
    if sha256(derived) != preview["preview_video_sha256"]:
        raise ValueError("设计预览的实际视频发生变化")
    previous_probe = checked_artifact(args.previous_manifest, "video_materials")
    previous = json.loads(args.previous_manifest.read_text())
    clips = [clip for clip in previous["clips"] if clip["file"] != "29_cosmos_lossless_background_acceptance.mp4"]
    if len(previous["clips"]) != 8 or len(clips) != 7:
        raise ValueError("只续接已声明8段精选，将早期杯子改色失败移回归档入口")
    for clip in clips:
        for name in [clip["file"], clip["poster"]]:
            if pathlib.PurePosixPath(name).name != name or sha256(args.previous_manifest.parent / name) != previous_probe["artifacts_sha256"].get(name):
                raise ValueError("历史精选文件已改变或越界")
            os.link(args.previous_manifest.parent / name, directory / name)
    source = pathlib.Path(preview["source_input"]).parent / "source.mp4"
    reference = pathlib.Path(preview["reference_input"]).parent / "reference_rgb.mp4"
    clips.append(render(directory, "31_cosmos_table_reference_design.mp4", [source, reference, native, derived],
                        ["成功训练源", "确定性材质参考", "Cosmos原生输出", "仅桌面设计预览"],
                        "桌面材质参考设计试验：不是策略学习前后",
                        "Thor / 既有MAXN / DiT bf16 / batch1 / base35步 / seed42 / 单源93帧 / 20fps",
                        ["全部非桌体像素保持的设计预览；尚无独立方法和完整质量验收，禁止入训。",
                         "参考是确定性派生控制；原生由Cosmos生成；右栏明确为背景派生。",
                         "展示视频为有损H264；不替代无损像素证据，尚无候选/对照或RSI收益。"],
                        93, {"kind": "table_reference_design_preview", "eligible_for_training": False,
                             "preview_report": str(args.preview_report), "preview_report_sha256": sha256(args.preview_report),
                             "generation_conditions": generation["conditions"], "policy_improvement_claimed": False}))
    manifest = {"clips": clips, "clip_count": len(clips), "curated": True,
                "previous_manifest": str(args.previous_manifest), "previous_manifest_sha256": sha256(args.previous_manifest),
                "preview_report": str(args.preview_report), "preview_report_sha256": sha256(args.preview_report),
                "eligible_for_training": False, "policy_improvement_claimed": False, "presentation_created": False}
    (directory / "video_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return {key: manifest[key] for key in ["clip_count", "curated", "eligible_for_training", "presentation_created"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    diagnostic = commands.add_parser("diagnose")
    diagnostic.add_argument("--generation-probe", type=pathlib.Path, required=True)
    publication = commands.add_parser("publish")
    publication.add_argument("--preview-report", type=pathlib.Path, required=True)
    publication.add_argument("--previous-manifest", type=pathlib.Path, required=True)
    reproduction = commands.add_parser("repro")
    for name in ["baseline-probe", "observed-probe", "repeat-probe"]:
        reproduction.add_argument("--" + name, type=pathlib.Path, required=True)
    arguments = parser.parse_args()
    for name in ["generation_probe", "preview_report", "previous_manifest", "baseline_probe", "observed_probe", "repeat_probe"]:
        if getattr(arguments, name, None):
            setattr(arguments, name, getattr(arguments, name).resolve(strict=True))
    callback, stage = {"diagnose": (diagnose, "table_reference_preview"), "publish": (publish, "video_materials"),
                       "repro": (compare_repro, "table_repro_comparison")}[arguments.command]
    raise SystemExit(run_stage(stage, lambda directory: callback(directory, arguments),
                     {"compute": "Thor CPU uint8视频与float32诊断；不运行模型", "source_windows": 1,
                      "frames": 93, "fps": 20, "diagnostic_only": True}))
