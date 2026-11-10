#!/usr/bin/env python3
"""导出前景保护诊断视频与精简素材清单，保留原生输出及不可入训边界。

@author pony
@date 2026-11-07
@version v1.3.0
@last_modified 2026-11-08
@changelog
    - v1.3.0 (2026-11-08): 区分低vis权重与ROI范围对照，保持各自标签和旧来源核验兼容。
    - v1.2.0 (2026-11-07): 验证已绑定全帧基线，导出四列ROI对照并保留前一几何通过样本。
    - v1.1.1 (2026-11-07): 已核验历史媒体用硬链接复用，避免每次精简导出再次复制相同内容。
    - v1.1.0 (2026-11-07): 正式预声明背景样本也可导出，分别标记几何和目标条件，替换旧诊断预览。
    - v1.0.0 (2026-11-07): 绑定成功源、原生生成与全帧诊断，主展示仅收录八段关键视频。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import shutil

from p0_common import ROOT, run_stage
from p0_guidance_materials import checked_run
from p0_media_index import digest
from p0_protected_background_preview import checked_preview_frames
from p0_video_materials import render


def checked_artifact(path: pathlib.Path, stage: str) -> dict:
    record = json.loads((path.parent / "probe.json").read_text())
    if record.get("stage") != stage or record.get("status") != "passed" \
            or record.get("artifacts_sha256", {}).get(path.name) != digest(path):
        raise ValueError("素材输入未绑定实际成功阶段及摘要")
    return record


def copy_key_clips(directory: pathlib.Path, previous_manifest: pathlib.Path,
                   roi_comparison: bool = False) -> list[dict]:
    probe = checked_artifact(previous_manifest, "video_materials")
    previous = json.loads(previous_manifest.read_text())
    wanted = {"03_policy_t00_success.mp4", "04_policy_t00_failure.mp4",
              "13_training_500_vs_5000_standard.mp4", "14_training_500_vs_5000_lighting.mp4",
              "15_training_500_vs_5000_background.mp4", "17_sim_source_depth_seg_controls.mp4",
              "21_cosmos_visual_vs_four_controls.mp4"}
    if roi_comparison:
        wanted.remove("21_cosmos_visual_vs_four_controls.mp4")
        wanted.add("26_cosmos_predeclared_background_sample.mp4")
    clips = [clip for clip in previous["clips"] if clip["file"] in wanted]
    if len(clips) != len(wanted):
        raise ValueError("关键历史视频不完整")
    for clip in clips:
        for name in [clip["file"], clip["poster"]]:
            original = previous_manifest.parent / name
            if pathlib.PurePosixPath(name).name != name or digest(original) != probe["artifacts_sha256"].get(name):
                raise ValueError("关键视频或海报已改变")
            os.link(original, directory / name)
    return clips


def collect_formal(directory: pathlib.Path, args: argparse.Namespace) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw, ImageFont
    from p0_background_composite import decode_frames
    from p0_generated_window_gate import verified_generation_paths

    derived = json.loads(args.composite_probe.read_text())
    if derived.get("stage") != "background_composite":
        raise ValueError("正式派生素材必须来自无损背景合成阶段")
    source, output = verified_generation_paths(derived, args.composite_probe)
    native_path = pathlib.Path(derived["result"]["native_generation_probe"])
    native = checked_run(native_path, args.native_gate)
    checked_artifact(args.composite_gate, "generated_window_gate")
    gate = json.loads(args.composite_gate.read_text())
    if source != native[2] or gate["generation_probe_sha256"] != digest(args.composite_probe) \
            or gate["output_video_sha256"] != digest(output) or gate["source_input_sha256"] != digest(source) \
            or gate["gate_config_sha256"] != digest(ROOT / "configs/gen_gate.yaml") \
            or gate["segmented_frames"] != list(range(93)):
        raise ValueError("正式派生的全帧质量报告与实际来源不一致")
    evidence = json.loads(source.read_text())
    comparison = evidence.get("comparison_baseline")
    baseline_video = None
    if comparison:
        from p0_prepare_background_transfer import bind_comparison_baseline

        bound = bind_comparison_baseline(pathlib.Path(comparison["generation_probe"]),
            pathlib.Path(evidence["parent_verified_input"]), json.loads((source.parent / "spec.json").read_text()), evidence,
            comparison.get("intervention_kind", "native_control_roi_mask"))
        if bound != comparison:
            raise ValueError("素材对照的全帧基线偏离新源生成前绑定的摘要")
        baseline_video = pathlib.Path(bound["output_video"])
    clips = copy_key_clips(directory, args.previous_manifest, roi_comparison=bool(comparison))
    geometry = "通过" if gate["geometry_passed"] else "未通过"
    semantic = "通过" if gate["semantic_review_passed"] else "未通过"
    condition = "通过" if gate["target_condition_passed"] else "未通过"
    sources = [source.parent / "source.mp4", pathlib.Path(derived["result"]["native_video"]), output]
    labels = ["同一成功训练源", "Cosmos原生输出", "无损前景保护派生输出"]
    if baseline_video:
        sources.insert(1, baseline_video)
        labels = ["同一成功训练源", "前次全帧控制原生", "本次ROI控制原生", "本次前景保护派生"]
    filename = "27_cosmos_full_frame_vs_roi_background.mp4" if comparison else "26_cosmos_predeclared_background_sample.mp4"
    low_vis = bool(comparison and comparison.get("intervention_kind") == "vis_control_weight_reduction")
    if low_vis:
        filename = "28_cosmos_full_frame_low_vis_background.mp4"
        labels = ["同一成功训练源", "原全帧vis=0.5", "全帧vis降为0.1", "本次前景保护派生"]
    clips.append(render(directory, filename, sources, labels,
        "同源外观控制对照：仅降低vis权重" if low_vis else
        "全帧/ROI背景同源对照：保留拒绝结果" if comparison else "背景单样本：几何一致性与目标条件分别验收",
        "Thor / 既有MAXN / DiT bf16 / batch1 / 35步 / seed42 / 单源93帧 / 480×480",
        [f"派生原几何门{geometry}；全帧语义{semantic}；预声明背景目标{condition}；入训资格：{'是' if gate['eligible_for_training'] else '否'}。",
         "源与无损派生的保护前景逐像素核对；此展示MP4再次有损编码，不具有像素完全一致保证。",
         "原生输出与派生合成分别留证；没有候选/对照或RSI收益；20fps保留全部实际帧。"],
        93, {"kind": "predeclared_background_composite", "is_eval": False,
             "eligible_for_training": False, "source_training_eligibility": gate["eligible_for_training"],
             "policy_improvement_claimed": False, "composite_probe": str(args.composite_probe),
             "comparison_baseline": comparison,
             "composite_probe_sha256": digest(args.composite_probe), "composite_gate": str(args.composite_gate),
             "composite_gate_sha256": digest(args.composite_gate)}))
    figure, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
    rows = []
    for axis, name in zip(axes, native[3]["roi_names"]):
        for label, report in [("native Cosmos", native[1]), ("protected composite", gate)]:
            values = [row["rois"][name]["generated_iou_vs_sim_gt"] for row in report["per_frame"]]
            axis.plot(range(93), values, label=label)
            rows.extend({"frame": i, "roi": name, "variant": label, "iou": value} for i, value in enumerate(values))
        axis.axhline(.8 if name == "robot" else .75, color="#a04836", linestyle="--")
        axis.set_ylabel(name + " / IoU"); axis.legend(loc="lower right")
    axes[-1].set_xlabel("Actual frame / 20fps / 1 source")
    figure.suptitle("Thor MAXN / SAM2 fp32 + bf16 autocast / paired batch2 / seed42 / background target "
                   + ("accepted" if gate["target_condition_passed"] else "not accepted"), fontsize=9)
    figure.savefig(directory / "formal_background_roi.png", dpi=180); plt.close(figure)
    with (directory / "formal_background_roi.csv").open("x", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    frames = [decode_frames(path) for path in sources]
    font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 18, index=2)
    sheet = Image.new("RGB", (1056, 292 * len(sources)), "#f5f4ef"); draw = ImageDraw.Draw(sheet)
    for slot, (label, video) in enumerate(zip(labels, frames)):
        for column, index in enumerate([0, 30, 61, 92]):
            draw.text((column * 264 + 4, slot * 292 + 3), f"{label}：帧{index}", font=font, fill="#20313c")
            sheet.paste(Image.fromarray(video[index]), (column * 264, slot * 292 + 31))
    sheet.save(directory / "formal_background_contact_sheet.png")
    report = {"clips": clips, "clip_count": len(clips), "curated": True,
              "previous_manifest": str(args.previous_manifest), "previous_manifest_sha256": digest(args.previous_manifest),
              "composite_probe": str(args.composite_probe), "composite_probe_sha256": digest(args.composite_probe),
              "composite_gate": str(args.composite_gate), "composite_gate_sha256": digest(args.composite_gate),
              "comparison_baseline": comparison,
              "geometry_passed": gate["geometry_passed"], "semantic_review_passed": gate["semantic_review_passed"],
              "target_condition_passed": gate["target_condition_passed"], "source_training_eligibility": gate["eligible_for_training"],
              "eligible_for_training": False, "policy_improvement_claimed": False, "presentation_created": False,
              "compute": "Thor CPU素材导出；展示不作为训练输入；完整历史及失败证据保留"}
    (directory / "video_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return {"materials": str(directory), "clip_count": len(clips), "eligible_for_training": False,
            "geometry_passed": gate["geometry_passed"], "target_condition_passed": gate["target_condition_passed"]}


def collect(directory: pathlib.Path, args: argparse.Namespace) -> dict:
    import imageio.v2 as imageio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    preview = json.loads(args.preview_probe.read_text())
    data = preview["result"]
    generation = checked_run(pathlib.Path(data["native_generation_probe"]), args.native_gate)
    source = generation[2]
    frames = checked_preview_frames(args.preview_probe, digest(source), 93)
    if data["native_generation_probe_sha256"] != digest(pathlib.Path(data["native_generation_probe"])) \
            or data["native_video_sha256"] != digest(pathlib.Path(data["native_video"])) \
            or data["foreground_max_abs_diff_all_frames"] != 0:
        raise ValueError("预览必须绑定原生生成且保持源前景")
    controls_probe = checked_artifact(args.controls, "segmentation_controls")
    controls = json.loads(args.controls.read_text())
    if controls["background_preview_probe_sha256"] != digest(args.preview_probe) \
            or controls["source_input_sha256"] != digest(source) \
            or controls["frames"] != list(range(93)) or controls["eligible_for_training"] is not False:
        raise ValueError("诊断缺少同一预览的全部93帧证据")
    clips = copy_key_clips(directory, args.previous_manifest)
    # 展示副本会再次有损编码；像素保持结论仅针对原始256分辨率数组。
    temporary = directory / "videos"
    temporary.mkdir()
    display = temporary / "preview_display_only.mp4"
    with imageio.get_writer(display, fps=20, codec="libx264", pixelformat="yuv420p", macro_block_size=1,
                           ffmpeg_params=["-crf", "18", "-threads", "2"]) as writer:
        for frame in frames:
            writer.append_data(frame)
    clips.append(render(directory, "25_cosmos_protected_background_preview.mp4",
        [source.parent / "source.mp4", pathlib.Path(data["native_video"]), display],
        ["成功训练源", "Cosmos原生输出：几何拒绝", "前景保护派生预览：不可入训"],
        "背景增强路线诊断：保留前景，复用Cosmos背景",
        "Thor / 既有MAXN / 原生DiT bf16 batch1 35步 seed42 / 单源93帧20fps",
        ["右栏是明确的合成预览；仅背景变化，不代表全场光照或RSI收益。",
         "原256数组前景逐像素保持；本展示视频有损编码，不具有像素完全一致保证。",
         "原ROI诊断通过不代表完整几何/语义/目标条件验收；历史原生输出仍拒绝入训。"],
        93, {"kind": "derived_background_diagnostic", "eligible_for_training": False,
             "policy_improvement_claimed": False, "preview_probe": str(args.preview_probe),
             "preview_probe_sha256": digest(args.preview_probe), "controls": str(args.controls),
             "controls_sha256": digest(args.controls)}))
    rows = []
    figure, axes = plt.subplots(len(generation[3]["roi_names"]), 1, figsize=(11, 8), sharex=True,
                               constrained_layout=True)
    for axis, name in zip(np.atleast_1d(axes), generation[3]["roi_names"]):
        native = {row["frame"]: row["rois"][name]["generated_iou_vs_sim_gt"]
                  for row in generation[1]["per_frame"] if name in row["rois"]}
        for label in ["identity", "protected_background_preview"]:
            values = []
            for row in controls["per_frame"]:
                item = row["variants"][label][name]
                values.append(item["iou_vs_unchanged_sim_gt"])
                rows.append({"frame": row["frame"], "roi": name, "variant": label,
                             "iou": values[-1], "threshold": item["threshold"]})
            axis.plot(range(93), values, label=label)
        axis.plot(list(native), list(native.values()), label="native Cosmos")
        axis.axhline(.80 if name == "robot" else .75, color="#a04836", linestyle="--")
        axis.set_ylabel(name + " / IoU")
        axis.legend(loc="lower right", fontsize=8)
    axes[-1].set_xlabel("Actual frame / 20fps / 1 source")
    figure.suptitle("Thor MAXN / SAM2 fp32 + bf16 autocast / paired batch2 / seed42 / diagnostic only", fontsize=9)
    figure.savefig(directory / "protected_background_roi.png", dpi=180)
    plt.close(figure)
    with (directory / "protected_background_roi.csv").open("x", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    for name in ["protected_background_000.png", "protected_background_046.png", "protected_background_092.png",
                 "pixel_preservation.csv", "preview.json"]:
        original = args.preview_probe.parent / name
        if preview["artifacts_sha256"].get(name) != digest(original):
            raise ValueError("像素证据或关键截图偏离原始记录")
        shutil.copyfile(original, directory / name)
    report = {"clips": clips, "clip_count": len(clips), "previous_manifest": str(args.previous_manifest),
              "previous_manifest_sha256": digest(args.previous_manifest), "curated": True,
              "background_preview_all_frames_roi_passed": controls["background_preview_all_frames_roi_passed"],
              "preview_probe": str(args.preview_probe), "preview_probe_sha256": digest(args.preview_probe),
              "controls_sha256": digest(args.controls), "conditions": controls_probe["conditions"],
              "hardware": controls_probe["hardware"], "eligible_for_training": False,
              "policy_improvement_claimed": False, "presentation_created": False,
              "compute": "Thor CPU素材导出；历史视频和测量依据保留在原目录"}
    (directory / "video_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return {"materials": str(directory), "clip_count": len(clips), "eligible_for_training": False,
            "background_preview_all_frames_roi_passed": report["background_preview_all_frames_roi_passed"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ["native-gate", "previous-manifest"]:
        parser.add_argument("--" + name, type=pathlib.Path, required=True)
    for name in ["preview-probe", "controls", "composite-probe", "composite-gate"]:
        parser.add_argument("--" + name, type=pathlib.Path)
    args = parser.parse_args()
    formal = bool(args.composite_probe or args.composite_gate)
    if (formal and (not args.composite_probe or not args.composite_gate or args.preview_probe or args.controls)) \
            or (not formal and (not args.preview_probe or not args.controls)):
        parser.error("需选择完整的正式派生probe/gate或完整的诊断预览probe/controls")
    for key in ["preview_probe", "controls", "native_gate", "previous_manifest", "composite_probe", "composite_gate"]:
        if not getattr(args, key):
            continue
        path = getattr(args, key).resolve(strict=True)
        if not path.is_relative_to(ROOT / "runs/p0"):
            parser.error("素材输入必须位于本项目P0目录")
        setattr(args, key, path)
    raise SystemExit(run_stage("video_materials", lambda directory: (collect_formal if formal else collect)(directory, args),
                               {"compute": "CPU_only", "formal_sample": formal}))
