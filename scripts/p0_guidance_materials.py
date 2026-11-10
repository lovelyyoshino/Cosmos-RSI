#!/usr/bin/env python3
"""归档同源四控制时空引导的真实视频、逐帧质量与实际latent覆盖。

@author pony
@date 2026-11-07
@version v1.0.1
@last_modified 2026-11-07
@changelog
    - v1.0.1 (2026-11-07): 传入真实生成目录，支持核验ROI空间控制数组后导出拒绝样本。
    - v1.0.0 (2026-11-07): 绑定同源单因素对照和完整质量门，保留历史附件及原生引导权重证据。
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
import json
import pathlib
import shutil

from p0_common import ROOT, run_stage
from p0_generation_sample_materials import validate_sample_records
from p0_multicontrol_materials import copy_training_progress_evidence
from p0_policy_runtime import sha256
from p0_video_materials import render


def validate_pair(before: dict, after: dict, before_spec: dict, after_spec: dict) -> None:
    for name in ["source_episode_id", "window", "instruction", "roi_names", "labels_sha256", "cache_manifest_sha256"]:
        if before.get(name) is None or before[name] != after.get(name):
            raise ValueError(f"时空引导对照改变了源或标签：{name}")
    old_files, new_files = before["files_sha256"], after["files_sha256"]
    if set(old_files) != set(new_files) or not {"source.mp4", "mask.mp4", "roi_masks.npy", "edge.mp4", "vis.mp4", "depth.mp4", "seg.mp4"} <= set(old_files):
        raise ValueError("四控制对照源文件集合不同或不完整")
    if old_files["mask.mp4"] == new_files["mask.mp4"] or any(old_files[key] != new_files[key] for key in old_files if key != "mask.mp4"):
        raise ValueError("只允许改变采样器引导mask，RGB、控制及标签必须不变")
    if before.get("mask_transform") is not None or after.get("mask_transform", {}).get("kind") != "sim_roi_spatial_dilation_and_latent_group_union":
        raise ValueError("必须从原始引导建立独立的时空扩展对照")
    excluded = {"name", "video_path", "guided_generation_mask", "image_context_path", "edge", "vis", "depth", "seg"}
    if {k: v for k, v in before_spec.items() if k not in excluded} != {k: v for k, v in after_spec.items() if k not in excluded}:
        raise ValueError("同源消融改变了提示、步数、种子或其他采样参数")
    if bool(before_spec.get("image_context_path")) != bool(after_spec.get("image_context_path")):
        raise ValueError("图像参考条件发生变化")
    for name in ["edge", "vis", "depth", "seg"]:
        if {k: v for k, v in before_spec[name].items() if k != "control_path"} != {k: v for k, v in after_spec[name].items() if k != "control_path"}:
            raise ValueError("原生控制参数发生变化")


def checked_run(probe_path: pathlib.Path, gate_path: pathlib.Path) -> tuple[dict, dict, pathlib.Path, dict, dict]:
    generation = json.loads(probe_path.read_text())
    result = generation["result"]
    source = pathlib.Path(result["source_success_evidence"]["path"]).resolve(strict=True)
    output = pathlib.Path(result["outputs"][0]).resolve(strict=True)
    if not source.is_relative_to(ROOT / "runs/p0") or not output.is_relative_to(probe_path.parent):
        raise RuntimeError("实际生成来源或输出越界")
    evidence, gate = json.loads(source.read_text()), json.loads(gate_path.read_text())
    validate_sample_records(generation, gate, evidence, sha256(probe_path), sha256(source),
                            generation_directory=probe_path.parent)
    for name, expected in {**evidence["files_sha256"], "source_labels.npz": evidence["labels_sha256"]}.items():
        if pathlib.PurePosixPath(name).name != name or sha256(source.parent / name) != expected:
            raise RuntimeError("生成源、控制或监督标签偏离实际摘要")
    gate_probe = json.loads((gate_path.parent / "probe.json").read_text())
    if gate_probe.get("status") != "passed" or gate_probe.get("stage") != "generated_window_gate" \
            or sha256(gate_path) != gate_probe["artifacts_sha256"].get(gate_path.name) \
            or gate["gate_config_sha256"] != sha256(ROOT / "configs/gen_gate.yaml"):
        raise RuntimeError("原质量门报告或配置摘要不一致")
    if sha256(output) != generation["artifacts_sha256"].get(str(output.relative_to(probe_path.parent))):
        raise RuntimeError("原生生成视频偏离实际阶段摘要")
    spec_path = pathlib.Path(result["source_spec"]).resolve(strict=True)
    if spec_path.parent != source.parent or evidence["spec"] != str(spec_path):
        raise RuntimeError("生成配方没有绑定相同成功源")
    return generation, gate, source, evidence, json.loads(spec_path.read_text())


def write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    with path.open("x", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect(directory: pathlib.Path, args: argparse.Namespace) -> dict:
    import cv2
    import imageio.v2 as imageio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    before = checked_run(args.before_probe, args.before_gate)
    after = checked_run(args.after_probe, args.after_gate)
    validate_pair(before[3], after[3], before[4], after[4])
    if after[3].get("parent_verified_input") != str(before[2]) or after[3].get("parent_verified_input_sha256") != sha256(before[2]):
        raise RuntimeError("时空引导对照的直接父来源或摘要错误")
    old = json.loads(args.previous_manifest.read_text())
    old_probe = json.loads((args.previous_manifest.parent / "probe.json").read_text())
    if old_probe.get("status") != "passed" or old_probe.get("stage") != "video_materials" \
            or old_probe["artifacts_sha256"].get("video_manifest.json") != sha256(args.previous_manifest):
        raise RuntimeError("历史视频包未通过实际阶段摘要核验")
    clips = list(old["clips"])
    if old["clip_count"] != len(clips):
        raise RuntimeError("历史视频清单计数不一致")
    copied = []
    # 既有图表、逐帧CSV和截图也随包归档，避免只复制MP4丢失其测量依据。
    for name, expected in old_probe["artifacts_sha256"].items():
        if pathlib.PurePosixPath(name).name != name or name == "video_manifest.json":
            continue
        source = args.previous_manifest.parent / name
        if source.suffix.lower() not in {".mp4", ".png", ".jpg", ".csv", ".json"}:
            continue
        if sha256(source) != expected:
            raise RuntimeError("历史素材偏离清单")
        shutil.copyfile(source, directory / name)
        copied.append({"file": name, "sha256": expected})
    for clip in clips:
        if pathlib.PurePosixPath(clip["file"]).name != clip["file"] or pathlib.PurePosixPath(clip["poster"]).name != clip["poster"] \
                or sha256(directory / clip["file"]) != clip["sha256"] or not (directory / clip["poster"]).is_file():
            raise RuntimeError("历史视频或海报没有完整保留")
    training = copy_training_progress_evidence(directory, args.previous_manifest, clips)
    sources = [before[2].parent / "source.mp4", pathlib.Path(before[0]["result"]["outputs"][0]),
               pathlib.Path(after[0]["result"]["outputs"][0])]
    common = {"kind": "four_control_guidance_comparison", "is_eval": False, "eligible_for_training": False,
              "policy_improvement_claimed": False, "before_probe": str(args.before_probe), "before_probe_sha256": sha256(args.before_probe),
              "after_probe": str(args.after_probe), "after_probe_sha256": sha256(args.after_probe),
              "before_gate": str(args.before_gate), "before_gate_sha256": sha256(args.before_gate),
              "after_gate": str(args.after_gate), "after_gate_sha256": sha256(args.after_gate)}
    clips.append(render(directory, f"{len(clips) + 1:02d}_cosmos_four_conservative_comparison.mp4", sources,
        ["同一成功训练源", "原四控制：逐帧引导", "四控制：扩展时空引导"],
        "Cosmos同源消融：只扩大采样器引导的时空覆盖",
        "Thor / 既有MAXN / DiT bf16 / batch1 / 35步 / seed42 / 单源93帧 / 480×480",
        ["RGB、动作、状态、提示与edge/vis/depth/seg权重不变；无输出像素回填。",
         f"原几何门：原配方{'通过' if before[1]['geometry_passed'] else '未通过'}，扩展引导{'通过' if after[1]['geometry_passed'] else '未通过'}；完整入训资格见质量报告。",
         "保持全部实际帧和20fps；4.65秒为播放时长，不是生成时间；尚无策略或RSI收益。"], 93, common))
    clips.append(render(directory, f"{len(clips) + 1:02d}_cosmos_guidance_masks.mp4",
        [sources[0], before[2].parent / "mask.mp4", after[2].parent / "mask.mp4"],
        ["同帧源RGB", "原采样器引导mask", "空间膨胀与四帧组并集"],
        "采样器引导输入对照：白色区域用于原生latent软引导",
        "Thor CPU控制准备 / 同一成功源 / 93帧 / 20Hz / 源RGB与监督标签不变",
        ["空间半径6个源像素；首帧独立，其后每四帧取并集；不是新生成图像。",
         "只修改原生采样器输入mask；不替换生成像素，不改变质量门。",
         "控制覆盖诊断不保证几何通过；实际模型输出、完整条件和逐帧指标相邻保存。"], 93,
        {**common, "kind": "native_guidance_mask_inputs"}))
    rows, coverage, summaries = [], [], []
    masks = np.load(before[2].parent / "roi_masks.npy", allow_pickle=False).astype(bool)
    figure, axes = plt.subplots(len(before[3]["roi_names"]), 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    for label, data, probe_path in [("original", before, args.before_probe), ("conservative", after, args.after_probe)]:
        gate = data[1]
        trace = data[0]["result"]["spatial_guidance"]["native_weight_trace"]
        if len(trace) != 1 or pathlib.PurePosixPath(trace[0]["file"]).name != trace[0]["file"]:
            raise RuntimeError("实际引导权重缺失或不唯一")
        weights_path = probe_path.parent / trace[0]["file"]
        if sha256(weights_path) != trace[0]["sha256"]:
            raise RuntimeError("实际引导权重摘要变化")
        weights = np.load(weights_path, allow_pickle=False)[0]
        if weights.shape != (24, 60, 60) or not np.isfinite(weights).all():
            raise RuntimeError("实际引导权重形状或有限性不符合480方形93帧约定")
        for frame in gate["per_frame"]:
            index = frame["frame"]
            latent = 0 if index == 0 else 1 + (index - 1) // 4
            projected = cv2.resize(weights[latent], (256, 256), interpolation=cv2.INTER_NEAREST)
            for roi_index, name in enumerate(data[3]["roi_names"]):
                roi = masks[index, roi_index]
                if name in frame["rois"]:
                    rows.append({"recipe": label, "frame": index, "roi": name, **frame["rois"][name]})
                coverage.append({"recipe": label, "frame": index, "roi": name,
                                 "source_visible_pixels": int(roi.sum()),
                                 "latent_weight_ge_0_99_fraction": float((projected[roi] >= 0.99).mean()) if roi.any() else None,
                                 "latent_weight_mean": float(projected[roi].mean()) if roi.any() else None})
        for axis, name in zip(np.atleast_1d(axes), data[3]["roi_names"]):
            selected = [row for row in rows if row["recipe"] == label and row["roi"] == name]
            values = [row["generated_iou_vs_sim_gt"] for row in selected]
            threshold = 0.80 if name == "robot" else 0.75
            axis.plot([row["frame"] for row in selected], values, label=label)
            summaries.append({"recipe": label, "roi": name, "frames": len(values), "threshold": threshold,
                              "below_threshold": sum(value < threshold for value in values), "min_iou": min(values),
                              "mean_iou": float(np.mean(values)), "geometry_passed": gate["geometry_passed"],
                              "eligible_for_training": gate["eligible_for_training"]})
    for axis, name in zip(np.atleast_1d(axes), before[3]["roi_names"]):
        axis.axhline(0.80 if name == "robot" else 0.75, color="#a04836", linestyle="--", label="original threshold")
        axis.set_ylim(0.65, 1.01)
        axis.set_ylabel(f"{name} / IoU")
        axis.legend(ncol=3, loc="lower right")
    axes[-1].set_xlabel("Actual frame / 20 fps")
    figure.suptitle("Thor MAXN / SAM2 fp32 + bf16 autocast / paired batch2 / seed42 / 1 source / all 93 frames", fontsize=9)
    figure.savefig(directory / "guidance_roi_comparison.png", dpi=180)
    plt.close(figure)
    write_csv(directory / "guidance_roi_per_frame.csv", rows)
    write_csv(directory / "guidance_latent_coverage.csv", coverage)
    write_csv(directory / "guidance_roi_summary.csv", summaries)
    font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 18, index=2)
    with ExitStack() as stack:
        readers = [stack.enter_context(imageio.get_reader(path)) for path in sources]
        sheet = Image.new("RGB", (1056, 292 * len(readers)), "#f5f4ef")
        draw = ImageDraw.Draw(sheet)
        for i, reader in enumerate(readers):
            for column, frame in enumerate([0, 30, 61, 92]):
                draw.text((column * 264 + 4, i * 292 + 3), f"{['成功源', '原四控制', '扩展引导'][i]}：帧{frame}", font=font, fill="#20313c")
                sheet.paste(Image.fromarray(reader.get_data(frame)).resize((256, 256)), (column * 264, i * 292 + 31))
        sheet.save(directory / "guidance_contact_sheet.png")
    report = {"clips": clips, "clip_count": len(clips), "training_progress_evidence": training,
              "previous_manifest": {"path": str(args.previous_manifest), "sha256": sha256(args.previous_manifest)},
              "copied_historical_assets": copied, "summary": summaries, "hardware": after[0]["hardware"],
              "generation_wall_s": after[0]["result"]["generate_wall_s"], "peak_mem_gb": after[0]["peak_mem_gb"],
              "comparison": common, "eligible_for_training": False, "policy_improvement_claimed": False,
              "presentation_created": False, "compute": "Thor CPU解码、图表、H264编码，不运行模型",
              "coverage_note": "实际latent权重按最近邻映射到源像素的诊断，不是独立几何真值或通过门槛"}
    (directory / "video_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return {"materials": str(directory), "clip_count": len(clips), "summary": summaries,
            "eligible_for_training": False, "policy_improvement_claimed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ["before-probe", "before-gate", "after-probe", "after-gate", "previous-manifest"]:
        parser.add_argument("--" + name, type=pathlib.Path, required=True)
    arguments = parser.parse_args()
    for key, value in vars(arguments).items():
        path = value.resolve(strict=True)
        if not path.is_relative_to(ROOT / "runs/p0"):
            parser.error("素材来源必须在本项目P0目录")
        setattr(arguments, key, path)
    raise SystemExit(run_stage("video_materials", lambda directory: collect(directory, arguments), {"compute": "CPU_only"}))
