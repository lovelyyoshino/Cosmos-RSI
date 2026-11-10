#!/usr/bin/env python3
"""用几何不变负控与已知平移正控检查原SAM2门，不产生可入训生成样本。

@author pony
@date 2026-11-07
@version v1.4.0
@last_modified 2026-11-08
@changelog
    - v1.4.0 (2026-11-08): 增加独立源核验与前景保持的暗/灰背景负控，保存实际数组和逐ROI误拒。
    - v1.3.1 (2026-11-07): probe独立报告派生预览的全帧结果，避免误读基础负控汇总。
    - v1.3.0 (2026-11-07): 同源前景保护派生预览沿用原ROI门，仅诊断，不改变原生生成资格。
    - v1.2.0 (2026-11-07): 可检查同源已完成VAE重建的全帧ROI，明确区别诊断与生成样本。
    - v1.1.0 (2026-11-07): 可加入已知几何不变的解析暖暗光照负控，区分外观变化与几何误拒。
    - v1.0.0 (2026-11-07): 同源identity、480档编码回环、右移24像素，保留原阈值和源候选编号。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

from p0_common import ROOT, ensure_gpu_idle, run_stage
from p0_generated_window_gate import mask_iou, validate_source_calibration
from p0_policy_runtime import sha256


def check_independent_source(source: dict, reference: dict) -> dict:
    """独立性限定为不同训练示范，不把同源的新窗口或复制文件算作新样本。"""
    for evidence in (source, reference):
        if evidence.get("source_success") is not True or evidence.get("replay_validation", {}).get("passed") is not True \
                or not evidence.get("source_episode_id") or not evidence.get("source_rgb_sha256") \
                or not evidence.get("labels_sha256") or not evidence.get("cache_manifest_sha256"):
            raise ValueError("独立校准需要完整的成功训练源证据")
    if source["cache_manifest_sha256"] != reference["cache_manifest_sha256"] \
            or source["source_episode_id"] == reference["source_episode_id"] \
            or source["source_rgb_sha256"] == reference["source_rgb_sha256"] \
            or source["labels_sha256"] == reference["labels_sha256"]:
        raise ValueError("独立源必须来自同一D0的不同回合，RGB和动作标签不能是旧源复制")
    return {"reference_episode_id": reference["source_episode_id"],
            "independent_episode_id": source["source_episode_id"],
            "same_d0_manifest": True, "different_episode_rgb_and_labels": True,
            "limitation": "不同示范回合不等于统计独立或未用于基线训练，仅用于评价器独立诊断"}


def background_controls(source, masks):
    """固定半径8保护任务前景，负控只改变保护区外背景，不产生训练数据。"""
    import numpy as np
    from p0_protected_background_preview import protected_background

    source, masks = np.asarray(source), np.asarray(masks)
    if masks.dtype != np.bool_:
        raise ValueError("背景负控要求显式bool真值掩码")
    dark = np.rint(source.astype(np.float32) * .5).astype(np.uint8)
    gray = np.full_like(source, 128)
    variants = {name: protected_background(source, value, masks, 8)[0]
                for name, value in [("analytic_background_dim", dark), ("analytic_background_gray", gray)]}
    return variants


def inspect_controls(directory: pathlib.Path, args: argparse.Namespace) -> dict:
    ensure_gpu_idle()
    import cv2
    import imageio.v2 as imageio
    import numpy as np
    import torch
    import yaml
    from PIL import Image, ImageDraw
    from transformers import Sam2Model, Sam2Processor
    from p0_calibrate_segmentation import balanced_prompts

    config = yaml.safe_load(args.config.read_text())
    evidence = json.loads(args.source_input.read_text())
    if not evidence.get("source_success") or not evidence.get("replay_validation", {}).get("passed"):
        raise RuntimeError("负控只允许使用已核验成功训练源")
    for name, digest in {**evidence["files_sha256"], "source_labels.npz": evidence["labels_sha256"]}.items():
        if pathlib.PurePosixPath(name).name != name or sha256(args.source_input.parent / name) != digest:
            raise RuntimeError("成功源文件摘要改变或路径越界")
    cache = pathlib.Path(evidence["cache_manifest"]).resolve(strict=True)
    if not cache.is_relative_to(ROOT / "data") or sha256(cache) != evidence["cache_manifest_sha256"]:
        raise ValueError("D0清单摘要改变或越界")
    source_rows = [row for row in json.loads(cache.read_text())["episodes"]
                   if row.get("episode_id") == evidence["source_episode_id"]]
    if len(source_rows) != 1 or source_rows[0].get("split") != "demo_train" \
            or source_rows[0].get("env_success") is not True or source_rows[0].get("is_eval", False) is not False:
        raise ValueError("评价器校准仅允许成功训练源，禁止冻结评测和验证来源")
    reference_path = getattr(args, "independent_of", None)
    independence = check_independent_source(evidence, json.loads(reference_path.read_text())) if reference_path else None
    if independence:
        independence.update({"reference_input": str(reference_path), "reference_input_sha256": sha256(reference_path)})
    manifests = []
    for path in sorted(ROOT.glob("runs/p0/model_download_*/model_manifest.json")):
        value = json.loads(path.read_text())
        if pathlib.Path(value["path"]).name == "sam2.1-hiera-tiny":
            manifests.append((path, value))
    if not manifests:
        raise RuntimeError("缺少已校验SAM2权重")
    manifest_path, manifest = manifests[-1]
    model_path = pathlib.Path(manifest["path"])
    for item in manifest["files"]:
        if sha256(model_path / item["file"]) != item["sha256"]:
            raise RuntimeError("SAM2权重摘要不一致")
    names, required = evidence["roi_names"], config["required_frames"]
    validate_source_calibration(json.loads(args.calibration.read_text()), sha256(args.source_input),
                                sha256(manifest_path), sha256(args.config), names, required,
                                config["source_segmentation_iou_min"])
    masks = np.load(args.source_input.parent / "roi_masks.npy", allow_pickle=False).astype(bool)
    if masks.shape != (required, len(names), 256, 256):
        raise RuntimeError("成功源ROI尺寸不符合约定")
    with imageio.get_reader(args.source_input.parent / "source.mp4") as reader:
        frames = [cv2.resize(frame, (256, 256), interpolation=cv2.INTER_AREA) for frame in reader]
    if len(frames) != required:
        raise RuntimeError("成功源必须恰好93帧，不补帧")
    codec_path = directory / "diagnostic_codec_roundtrip.mp4"
    with imageio.get_writer(codec_path, fps=20, codec="libx264", pixelformat="yuv420p", macro_block_size=1,
                           ffmpeg_params=["-crf", "18", "-threads", "2"]) as writer:
        for frame in frames:
            writer.append_data(cv2.resize(frame, (480, 480), interpolation=cv2.INTER_LINEAR))
    with imageio.get_reader(codec_path) as reader:
        roundtrip = [cv2.resize(frame, (256, 256), interpolation=cv2.INTER_AREA) for frame in reader]
    if len(roundtrip) != required:
        raise RuntimeError("编码回环帧数不一致")
    vae_frames, vae_evidence = None, None
    if args.vae_probe:
        vae = json.loads(args.vae_probe.read_text())
        if vae["stage"] != "vae_roundtrip" or vae["status"] != "passed" \
                or vae["result"]["source_input_sha256"] != sha256(args.source_input):
            raise RuntimeError("VAE对照缺少同源实际重建证据")
        vae_path = pathlib.Path(vae["result"]["video"]).resolve(strict=True)
        if not vae_path.is_relative_to(ROOT / "runs/p0") or sha256(vae_path) != vae["result"]["video_sha256"]:
            raise RuntimeError("VAE重建视频摘要改变或越界")
        with imageio.get_reader(vae_path) as reader:
            vae_frames = [cv2.resize(frame, (256, 256), interpolation=cv2.INTER_AREA) for frame in reader]
        if len(vae_frames) != required:
            raise RuntimeError("VAE重建不满足93帧逐帧对齐")
        vae_evidence = {"probe": str(args.vae_probe), "probe_sha256": sha256(args.vae_probe),
                        "video": str(vae_path), "video_sha256": sha256(vae_path)}
    preview_frames = None
    preview_path = getattr(args, "background_preview", None)
    if preview_path:
        from p0_protected_background_preview import checked_preview_frames
        preview_frames = checked_preview_frames(preview_path, sha256(args.source_input), required)
    indices = list(range(required)) if args.all_frames else [0]
    torch.manual_seed(args.seed)
    model = Sam2Model.from_pretrained(model_path, local_files_only=True).to("cuda").eval()
    processor = Sam2Processor.from_pretrained(model_path, local_files_only=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    rows = []
    for index in indices:
        source = frames[index]
        shifted = cv2.warpAffine(source, np.float32([[1, 0, 24], [0, 1, 0]]), (256, 256),
                                 flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=(90, 90, 90))
        variants = {"identity": source, "resize_codec": roundtrip[index]}
        if args.include_background_controls:
            variants.update(background_controls(source, masks[index]))
        if args.include_warm_dim:
            variants["analytic_warm_dim"] = np.rint(source.astype(np.float32)
                                                   * np.asarray([0.55, 0.40, 0.30], np.float32)).astype(np.uint8)
        if vae_frames is not None:
            variants["vae_roundtrip"] = vae_frames[index]
        if preview_frames is not None:
            variants["protected_background_preview"] = preview_frames[index]
        variants["shift24"] = shifted
        if args.include_background_controls:
            # 保存实际模型输入与源真值，后续可核验前景保持及完整负控来源。
            np.savez_compressed(directory / f"control_inputs_{index:03d}.npz", source_rois=masks[index], **variants)
        visible = [i for i, mask in enumerate(masks[index]) if mask.sum() >= config["visible_pixels_min"]]
        boxes, points, labels = balanced_prompts(masks[index, visible], 12)
        row = {"frame": index, "variants": {}}
        pictures = [source.copy()]
        for variant, transformed in variants.items():
            inputs = processor(images=[Image.fromarray(source), Image.fromarray(transformed)],
                               input_boxes=[boxes, boxes], input_points=[points, points],
                               input_labels=[labels, labels], return_tensors="pt")
            sizes = inputs["original_sizes"]
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(**inputs.to("cuda"), multimask_output=True)
            predicted = processor.post_process_masks(outputs.pred_masks.cpu(), sizes.cpu())
            picture = transformed.copy()
            per_roi = {}
            for position, group in enumerate(visible):
                reference = masks[index, group]
                candidate = max(range(predicted[0].shape[1]),
                                key=lambda i: mask_iou(reference, predicted[0][position, i].numpy()))
                prediction = predicted[1][position, candidate].numpy().astype(bool)
                threshold = config["robot_mask_iou_min" if names[group] == "robot" else "object_mask_iou_min"]
                iou = mask_iou(reference, prediction)
                per_roi[names[group]] = {"source_selected_candidate": candidate,
                                         "source_calibration_iou": mask_iou(reference, predicted[0][position, candidate].numpy()),
                                         "iou_vs_unchanged_sim_gt": iou, "threshold": threshold,
                                         "passes_original_roi_threshold": iou >= threshold}
                color = [(20, 220, 70), (255, 140, 30), (50, 140, 255)][group % 3]
                for image, mask in [(picture, prediction), (pictures[0], reference)]:
                    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(image, contours, -1, color, 1)
            row["variants"][variant] = per_roi
            pictures.append(picture)
        canvas = Image.new("RGB", (256 * len(pictures), 290), "white")
        draw = ImageDraw.Draw(canvas)
        variant_labels = ["Source GT", "Identity", "Resize + codec"]
        if args.include_background_controls:
            variant_labels.extend(["Background dim (analytic)", "Background gray (analytic)"])
        if args.include_warm_dim:
            variant_labels.append("Analytic warm dim")
        if vae_frames is not None:
            variant_labels.append("VAE roundtrip")
        if preview_frames is not None:
            variant_labels.append("Derived background PREVIEW")
        variant_labels.append("Shift right 24px")
        for slot, (label, picture) in enumerate(zip(variant_labels, pictures)):
            draw.text((slot * 256 + 3, 5), f"{index:03d}: {label}", fill="black")
            canvas.paste(Image.fromarray(picture), (slot * 256, 30))
        canvas.save(directory / f"controls_{index:03d}.png")
        rows.append(row)
        print(json.dumps(row), flush=True)
    summaries = {}
    variant_names = ["identity", "resize_codec",
                     *(["analytic_background_dim", "analytic_background_gray"] if args.include_background_controls else []),
                     *(["analytic_warm_dim"] if args.include_warm_dim else []),
                     *(["vae_roundtrip"] if vae_frames is not None else []),
                     *(["protected_background_preview"] if preview_frames is not None else []), "shift24"]
    for variant in variant_names:
        summaries[variant] = {}
        for name in names:
            values = [(row["frame"], row["variants"][variant][name]) for row in rows if name in row["variants"][variant]]
            summaries[variant][name] = {
                "visible_frames": len(values),
                "min_iou": min(value["iou_vs_unchanged_sim_gt"] for _, value in values),
                "mean_iou": float(np.mean([value["iou_vs_unchanged_sim_gt"] for _, value in values])),
                "rejected_frames": [frame for frame, value in values if not value["passes_original_roi_threshold"]]}
    negative_pass = all(not summary["rejected_frames"] for variant in variant_names if variant not in ["shift24", "vae_roundtrip", "protected_background_preview"]
                        for summary in summaries[variant].values())
    positive_detected = any(summary["rejected_frames"] for summary in summaries["shift24"].values())
    report = {"source_input": str(args.source_input), "source_input_sha256": sha256(args.source_input),
              "calibration": str(args.calibration), "calibration_sha256": sha256(args.calibration),
              "gate_config_sha256": sha256(args.config), "sam_manifest_sha256": sha256(manifest_path),
              "sample_count": 1, "batch": 2, "seed": args.seed, "frames": indices,
              "precision": "SAM2 fp32参数/bf16 autocast；RGB uint8；CPU IoU",
              "steps": None, "all_frames": len(indices) == required, "summary": summaries, "per_frame": rows,
              "negative_controls_all_roi_passed": negative_pass,
              "positive_control_detected": positive_detected,
              "wall_s_without_model_loading": time.monotonic() - started,
              "torch_peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
              "codec": "原256解码RGB->OpenCV linear 480->libx264 CRF18 yuv420p->area256；20fps93帧",
              "codec_video_sha256": sha256(codec_path),
              "source_independence": independence,
              "analytic_background_controls": {"enabled": args.include_background_controls,
                                               "protected_foreground_radius_pixels": 8,
                                               "dim_rgb_scale": .5, "gray_rgb": [128, 128, 128],
                                               "foreground_pixels_preserved": True if args.include_background_controls else None,
                                               "eligible_for_training": False},
              "vae_reconstruction_evidence": vae_evidence,
              "background_preview_probe": str(preview_path) if preview_path else None,
              "background_preview_probe_sha256": sha256(preview_path) if preview_path else None,
              "background_preview_all_frames_roi_passed": (
                  len(indices) == required and all(not item["rejected_frames"]
                                                  for item in summaries["protected_background_preview"].values())
              ) if preview_frames is not None else None,
              "analytic_warm_dim": {"enabled": args.include_warm_dim, "rgb_scale": [0.55, 0.40, 0.30],
                                    "definition": "逐像素sRGB uint8乘法并四舍五入，不改变几何；不等同于物理渲染或Cosmos生成"},
              "limitation": "仅诊断原ROI分割门的误拒/检出；平移仍沿用源提示。不是生成、语义检查或策略评测。",
              "eligible_for_training": False, "generation_performed": False}
    (directory / "controls.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return {key: report[key] for key in ["source_input", "frames", "all_frames", "summary", "batch", "seed",
                                        "negative_controls_all_roi_passed", "positive_control_detected", "eligible_for_training",
                                        "background_preview_all_frames_roi_passed", "source_independence",
                                        "analytic_background_controls"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-input", type=pathlib.Path, required=True)
    parser.add_argument("--calibration", type=pathlib.Path, required=True)
    parser.add_argument("--config", type=pathlib.Path, default=ROOT / "configs/gen_gate.yaml")
    parser.add_argument("--all-frames", action="store_true")
    parser.add_argument("--include-warm-dim", action="store_true")
    parser.add_argument("--include-background-controls", action="store_true", help="前景保持的解析暗背景与灰背景负控")
    parser.add_argument("--independent-of", type=pathlib.Path, help="旧源证据；必须为不同训练回合及不同RGB/标签")
    parser.add_argument("--vae-probe", type=pathlib.Path)
    parser.add_argument("--background-preview", type=pathlib.Path, help="同源且明确不可入训的前景保护预览probe")
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()
    for name in ["source_input", "calibration", "config"]:
        path = getattr(arguments, name).resolve(strict=True)
        if not path.is_relative_to(ROOT):
            parser.error("诊断输入必须位于本项目目录")
        setattr(arguments, name, path)
    if arguments.vae_probe:
        arguments.vae_probe = arguments.vae_probe.resolve(strict=True)
        if not arguments.vae_probe.is_relative_to(ROOT / "runs/p0"):
            parser.error("VAE诊断记录必须位于本项目P0目录")
    if arguments.background_preview:
        arguments.background_preview = arguments.background_preview.resolve(strict=True)
        if not arguments.background_preview.is_relative_to(ROOT / "runs/p0"):
            parser.error("派生预览记录必须位于本项目P0目录")
    if arguments.independent_of:
        arguments.independent_of = arguments.independent_of.resolve(strict=True)
        if not arguments.independent_of.is_relative_to(ROOT / "runs/p0"):
            parser.error("独立性参照必须位于本项目P0目录")
    raise SystemExit(run_stage("segmentation_controls", lambda directory: inspect_controls(directory, arguments),
                               {"sample_count": 1, "batch": 2, "seed": arguments.seed,
                                "all_frames": arguments.all_frames, "include_warm_dim": arguments.include_warm_dim,
                                "include_background_controls": arguments.include_background_controls,
                                "independent_of": str(arguments.independent_of) if arguments.independent_of else None,
                                "vae_probe": str(arguments.vae_probe) if arguments.vae_probe else None,
                                "background_preview": str(arguments.background_preview) if arguments.background_preview else None,
                                "generation_performed": False}))
