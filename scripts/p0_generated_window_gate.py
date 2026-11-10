#!/usr/bin/env python3
"""用已知仿真ROI校准SAM2后检查Cosmos窗口，保留全部拒绝原因和逐帧指标。

@author pony
@date 2026-11-06
@version v1.5.1
@last_modified 2026-11-08
@changelog
    - v1.5.1 (2026-11-08): 原SAM2入口对所有背景派生仅出测量，训练资格统一交由独立派生门。
    - v1.5.0 (2026-11-08): 单独核验桌面v2派生来源与完整像素后复用原SAM2测量，原阈值不变。
    - v1.4.0 (2026-11-07): 明确接收预声明背景派生数据，逐像素核验完整来源后仍使用原质量门。
    - v1.3.0 (2026-11-07): 在加载评价模型前绑定实际源和输出摘要；独立验收目标外观，拒绝未声明条件或未复核区域。
    - v1.2.0 (2026-11-07): 可使用绑定源图、模型和原阈值的全帧校准提示，不放宽质量门。
    - v1.1.0 (2026-11-06): 用源图真值确定SAM2候选粒度，生成图沿用同一候选编号，不按生成IoU挑mask。
    - v1.0.0 (2026-11-06): 独立分割、ROI边缘及帧间残差检查；部分帧或缺少语义复核禁止入训。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

from p0_common import ROOT, ensure_gpu_idle, run_stage
from p0_policy_runtime import sha256


def mask_iou(first, second) -> float:
    import numpy as np

    first, second = np.asarray(first, dtype=bool), np.asarray(second, dtype=bool)
    union = np.logical_or(first, second).sum()
    return float(np.logical_and(first, second).sum() / union) if union else 0.0


def prompted_regions(masks):
    import cv2
    import numpy as np

    boxes, points = [], []
    for mask in masks:
        ys, xs = np.nonzero(mask)
        if not len(xs):
            raise RuntimeError("不能给不可见物体构造分割提示")
        boxes.append([max(0, int(xs.min()) - 4), max(0, int(ys.min()) - 4),
                      min(mask.shape[1] - 1, int(xs.max()) + 4), min(mask.shape[0] - 1, int(ys.max()) + 4)])
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
        components = sorted(range(1, count), key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True)[:3]
        locations = []
        for component in components:
            distance = cv2.distanceTransform((labels == component).astype(np.uint8), cv2.DIST_L2, 3)
            y, x = np.unravel_index(distance.argmax(), distance.shape)
            locations.append([int(x), int(y)])
        locations += [locations[0]] * (3 - len(locations))
        points.append(locations)
    return boxes, points


def roi_edge_f1(source, generated, mask, tolerance: int) -> float:
    import cv2
    import numpy as np

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    edges = [cv2.Canny(clahe.apply(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)), 80, 160) > 0
             for image in [source, generated]]
    kernel = np.ones((2 * tolerance + 1, 2 * tolerance + 1), dtype=np.uint8)
    roi = cv2.dilate(mask.astype(np.uint8), kernel) > 0
    first, second = [edge & roi for edge in edges]
    precision = (second & (cv2.dilate(first.astype(np.uint8), kernel) > 0)).sum() / max(1, second.sum())
    recall = (first & (cv2.dilate(second.astype(np.uint8), kernel) > 0)).sum() / max(1, first.sum())
    return float(2 * precision * recall / max(1e-12, precision + recall))


def validate_source_calibration(report: dict, source_sha: str, model_sha: str, config_sha: str,
                                roi_names: list[str], required_frames: int, threshold: float) -> None:
    if report.get("source_input_sha256") != source_sha or report.get("sam_model_manifest_sha256") != model_sha \
            or report.get("gate_config_sha256") != config_sha or report.get("threshold") != threshold:
        raise ValueError("分割校准没有绑定同一源图、SAM2和原质量门配置")
    if report.get("all_frames") is not True or report.get("frames") != list(range(required_frames)) \
            or report.get("balanced_calibration_passed") is not True:
        raise ValueError("分割提示必须先完成全部源帧校准")
    summary = report.get("summary", {}).get("balanced12", {})
    if set(summary) != set(roi_names) or any(value.get("below_threshold_frames")
            or value.get("visible_frames", 0) < 1 or not threshold <= value.get("min_iou", -1) <= 1
            for value in summary.values()):
        raise ValueError("校准缺少目标ROI或仍有未达标源帧")


def verified_generation_paths(generation: dict, probe_path: pathlib.Path,
                              root: pathlib.Path | None = None) -> tuple[pathlib.Path, pathlib.Path]:
    root = ROOT if root is None else root
    if generation.get("stage") == "table_background_composite":
        from p0_table_background import verified_composite_paths
        return verified_composite_paths(generation, probe_path, root)
    if generation.get("stage") not in {"transfer_smoke", "background_composite"} or generation.get("status") != "passed":
        raise ValueError("生成检查只接收实际完成的Transfer运行")
    result = generation["result"]
    source_record = result.get("source_success_evidence", {})
    if not source_record.get("path") or not source_record.get("sha256") or len(result.get("outputs", [])) != 1:
        raise ValueError("生成源或单窗口输出缺少可核验的摘要")
    source = pathlib.Path(source_record["path"]).resolve(strict=True)
    output = pathlib.Path(result["outputs"][0]).resolve(strict=True)
    if not source.is_relative_to(root / "runs/p0") or sha256(source) != source_record["sha256"]:
        raise ValueError("实际成功源证据越界或偏离生成时摘要")
    if not output.is_relative_to(probe_path.parent):
        raise ValueError("生成视频越过本次生成目录")
    expected = generation.get("artifacts_sha256", {}).get(str(output.relative_to(probe_path.parent)))
    if not expected or sha256(output) != expected:
        raise ValueError("实际生成视频偏离原probe摘要，禁止借用旧复核放行")
    if result.get("guardrails_enabled") is not True:
        raise ValueError("生成没有启用安全检查")
    if generation["stage"] == "background_composite":
        import numpy as np
        from p0_background_composite import check_declaration, decode_frames, verify_composite_pixels

        evidence = json.loads(source.read_text())
        check_declaration(evidence)
        native_probe = pathlib.Path(result["native_generation_probe"]).resolve(strict=True)
        if not native_probe.is_relative_to(root / "runs/p0") \
                or sha256(native_probe) != result.get("native_generation_probe_sha256"):
            raise ValueError("派生数据没有绑定真实原生生成摘要")
        native = json.loads(native_probe.read_text())
        if native.get("stage") != "transfer_smoke":
            raise ValueError("禁止递归合成或伪装原生生成")
        native_source, native_video = verified_generation_paths(native, native_probe, root)
        preparation = json.loads((source.parent / "probe.json").read_text())
        if native_source != source or sha256(native_video) != result.get("native_video_sha256") \
                or pathlib.Path(result["native_video"]).resolve(strict=True) != native_video \
                or result.get("composite_recipe") != evidence["composite_recipe"] \
                or result.get("derived_kind") != evidence["composite_recipe"]["kind"] \
                or result.get("target_condition") != evidence["target_condition"] \
                or result.get("lossless_pixels_verified") is not True \
                or preparation.get("stage") != "background_transfer_input" or preparation.get("status") != "passed" \
                or preparation.get("artifacts_sha256", {}).get(source.name) != sha256(source):
            raise ValueError("派生输出偏离生成前声明或同一成功源")
        for name, checksum in {**evidence["files_sha256"], "source_labels.npz": evidence["labels_sha256"]}.items():
            if pathlib.PurePosixPath(name).name != name or sha256(source.parent / name) != checksum:
                raise ValueError("派生数据的源RGB、ROI或标签摘要改变")
        verify_composite_pixels(decode_frames(source.parent / "source.mp4"), decode_frames(native_video),
                                np.load(source.parent / "roi_masks.npy", allow_pickle=False), decode_frames(output),
                                evidence["composite_recipe"]["radius"])
    return source, output


def legacy_gate_decision(stage: str, reasons: list[str]) -> dict:
    """原生沿用原门；派生不能借原门的绿色结果绕过独立方法及背景复核。"""
    authorities = {"transfer_smoke": "thor-roi-sam2-v1", "background_composite": "thor-lossless-background-v1",
                   "table_background_composite": "thor-table-background-v2.1"}
    if stage not in authorities:
        raise ValueError("未知输出类型不能取得训练资格")
    passed = not reasons
    return {"passed": passed, "eligible_for_training": stage == "transfer_smoke" and passed,
            "training_eligibility_authority": authorities[stage],
            "training_eligibility_notice": "原生使用本门" if stage == "transfer_smoke" else "本报告仅记录原SAM2测量，派生资格须由独立门授予"}


def review_bound_to_video(review: dict | None, generation_sha: str, video_sha: str,
                           required_frames: int) -> bool:
    return bool(isinstance(review, dict) and review.get("source_probe_sha256") == generation_sha
                and review.get("output_video_sha256") == video_sha
                and review.get("reviewed_frames") == list(range(required_frames))
                and isinstance(review.get("reviewer"), str) and review["reviewer"].strip())


def target_condition_review_passed(review: dict | None, target: dict | None,
                                    generation_sha: str, video_sha: str, required_frames: int) -> bool:
    allowed = {"light", "background", "texture", "sensor_noise"}
    if not isinstance(target, dict) or not target or not set(target) <= allowed \
            or any(not isinstance(value, str) or not value.strip() for value in target.values()):
        return False
    if not review_bound_to_video(review, generation_sha, video_sha, required_frames) \
            or review.get("target_condition") != target or review.get("passed") is not True:
        return False
    # 全场光照和噪声不能只看背景均值；背景纹理也必须确认前景仍受保护。
    regions = {"background"}
    if set(target) & {"light", "sensor_noise"}:
        regions.update({"robot", "target_objects"})
    if set(target) & {"background", "texture"}:
        regions.add("protected_foreground")
    checks = review.get("region_checks", {})
    if not isinstance(checks, dict):
        return False
    return all(isinstance(checks.get(name), dict) and checks[name].get("passed") is True
               and isinstance(checks[name].get("criterion"), str) and checks[name]["criterion"].strip()
               and isinstance(checks[name].get("observations"), str) and checks[name]["observations"].strip()
               for name in regions)


def evaluate(directory: pathlib.Path, args: argparse.Namespace) -> dict:
    ensure_gpu_idle()
    import cv2
    import imageio.v2 as imageio
    import numpy as np
    import torch
    import yaml
    from PIL import Image, ImageDraw
    from transformers import Sam2Model, Sam2Processor

    config = yaml.safe_load(args.config.read_text())
    generation = json.loads(args.probe.read_text())
    evidence_path, output_path = verified_generation_paths(generation, args.probe)
    result = generation["result"]
    generation_sha, output_sha = sha256(args.probe), sha256(output_path)
    evidence = json.loads(evidence_path.read_text())
    if not evidence.get("replay_validation", {}).get("passed") or not evidence.get("source_success"):
        raise RuntimeError("缺少本机成功源及逐帧复现证据")
    for name, expected in evidence["files_sha256"].items():
        source_file = (evidence_path.parent / name).resolve(strict=True)
        if pathlib.PurePosixPath(name).name != name or not source_file.is_relative_to(evidence_path.parent) \
                or sha256(source_file) != expected:
            raise RuntimeError("生成源视频或ROI已经改变")
    if sha256(evidence_path.parent / "source_labels.npz") != evidence["labels_sha256"]:
        raise RuntimeError("源动作与状态标签摘要不一致")
    manifests = []
    for path in sorted(ROOT.glob("runs/p0/model_download_*/model_manifest.json")):
        value = json.loads(path.read_text())
        if pathlib.Path(value["path"]).name == "sam2.1-hiera-tiny":
            manifests.append((path, value))
    if not manifests:
        raise RuntimeError("SAM2必须先下载并逐文件核验")
    model_manifest_path, model_manifest = manifests[-1]
    model_path = pathlib.Path(model_manifest["path"])
    for item in model_manifest["files"]:
        if sha256(model_path / item["file"]) != item["sha256"]:
            raise RuntimeError("SAM2权重或配置偏离已校验版本")
    calibration_path = getattr(args, "calibration", None)
    if calibration_path:
        validate_source_calibration(json.loads(calibration_path.read_text()), sha256(evidence_path),
                                    sha256(model_manifest_path), sha256(args.config), evidence["roi_names"],
                                    config["required_frames"], config["source_segmentation_iou_min"])
    sources = [evidence_path.parent / "source.mp4", output_path]
    videos = []
    for path in sources:
        frames = []
        with imageio.get_reader(path) as reader:
            for frame in reader:
                frames.append(cv2.resize(frame, (256, 256), interpolation=cv2.INTER_AREA))
                if len(frames) > config["required_frames"]:
                    raise RuntimeError("生成窗口超过约定帧数")
        if len(frames) != config["required_frames"]:
            raise RuntimeError("源与生成窗口必须逐帧等长，禁止补帧")
        videos.append(np.asarray(frames))
    masks = np.load(evidence_path.parent / "roi_masks.npy", allow_pickle=False).astype(bool)
    names = evidence["roi_names"]
    if masks.shape != (93, len(names), 256, 256) or names[0] != "robot":
        raise RuntimeError("源ROI形状或机器人命名不符合约定")
    indices = sorted(set([*range(0, 93, args.stride), 92]))
    torch.manual_seed(args.seed)
    model = Sam2Model.from_pretrained(model_path, local_files_only=True).to("cuda").eval()
    processor = Sam2Processor.from_pretrained(model_path, local_files_only=True)
    torch.cuda.reset_peak_memory_stats()
    rows = []
    before = time.monotonic()
    for frame_index in indices:
        source, generated = videos[0][frame_index], videos[1][frame_index]
        visible = [i for i, mask in enumerate(masks[frame_index]) if mask.sum() >= config["visible_pixels_min"]]
        if calibration_path:
            from p0_calibrate_segmentation import balanced_prompts

            boxes, points, labels = balanced_prompts(masks[frame_index, visible], 12)
        else:
            boxes, points = prompted_regions(masks[frame_index, visible])
            labels = [[1, 1, 1] for _ in visible]
        inputs = processor(images=[Image.fromarray(source), Image.fromarray(generated)],
                           input_boxes=[boxes, boxes], input_points=[points, points],
                           input_labels=[labels, labels], return_tensors="pt")
        original_sizes = inputs["original_sizes"]
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(**inputs.to("cuda"), multimask_output=True)
        predicted = processor.post_process_masks(outputs.pred_masks.cpu(), original_sizes.cpu())
        per_roi = {}
        visual = np.concatenate([source, generated], axis=1).copy()
        for index, group in enumerate(visible):
            reference = masks[frame_index, group]
            # 只在已知真值的源图校准候选粒度，禁止用生成图IoU挑最有利的候选。
            candidate = max(range(predicted[0].shape[1]),
                            key=lambda i: mask_iou(reference, predicted[0][index, i].numpy()))
            source_prediction = predicted[0][index, candidate].numpy().astype(bool)
            generated_prediction = predicted[1][index, candidate].numpy().astype(bool)
            colors = []
            for frame, mask in [(source, reference), (generated, generated_prediction)]:
                pixels = frame[mask].astype(np.float32)
                pixels = pixels[pixels.sum(axis=1) > 30]
                colors.append(np.mean(pixels / np.maximum(1, pixels.sum(axis=1, keepdims=True)), axis=0)
                              if len(pixels) else np.full(3, np.nan))
            per_roi[names[group]] = {
                "source_selected_candidate": candidate,
                "source_calibration_iou": mask_iou(reference, source_prediction),
                "generated_iou_vs_sim_gt": mask_iou(reference, generated_prediction),
                "edge_f1": roi_edge_f1(source, generated, reference, config["edge_tolerance_pixels"]),
                "chromaticity_error": float(np.mean(np.abs(colors[0] - colors[1])))
                                      if np.isfinite(colors).all() else None,
                "source_visible_pixels": int(reference.sum()), "generated_mask_pixels": int(generated_prediction.sum())}
            for offset, mask in [(0, reference), (256, generated_prediction)]:
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(visual[:, offset:offset + 256], contours, -1, (50, 240, 120), 1)
        imageio.imwrite(directory / f"roi_{frame_index:03d}.png", visual)
        row = {"frame": frame_index, "rois": per_roi}
        rows.append(row)
        print(json.dumps(row), flush=True)
    temporal = []
    for index in range(1, 93):
        delta = []
        roi = masks[index].any(axis=0) | masks[index - 1].any(axis=0)
        for video in videos:
            delta.append(cv2.cvtColor(video[index], cv2.COLOR_RGB2GRAY).astype(np.float32)
                         - cv2.cvtColor(video[index - 1], cv2.COLOR_RGB2GRAY).astype(np.float32))
        temporal.append(float(np.abs(delta[1] - delta[0])[roi].mean() / 255) if roi.any() else 1.0)
    checks = [item for row in rows for item in row["rois"].values()]
    reasons = []
    if len(indices) != 93:
        reasons.append("仅抽样分割，尚未检查全部93帧")
    if not checks or min(item["source_calibration_iou"] for item in checks) < config["source_segmentation_iou_min"]:
        reasons.append("源图分割未达到仿真真值校准阈值，不能信任生成图IoU")
    for row in rows:
        for name, item in row["rois"].items():
            threshold = config["robot_mask_iou_min" if name == "robot" else "object_mask_iou_min"]
            if item["generated_iou_vs_sim_gt"] < threshold:
                reasons.append(f"{name}生成分割与源真值不一致")
            if name != "robot" and (item["chromaticity_error"] is None
                    or item["chromaticity_error"] > config["object_chromaticity_error_max"]):
                reasons.append(f"{name}颜色比例改变或无有效像素")
    edge_values = [item["edge_f1"] for item in checks]
    if not edge_values or np.mean(edge_values) < config["roi_edge_f1_mean_min"] \
            or np.percentile(edge_values, 10) < config["roi_edge_f1_p10_min"]:
        reasons.append("ROI边缘一致性不足")
    flicker_ok = np.percentile(temporal, 95) <= config["temporal_residual_p95_max"]
    if not flicker_ok:
        reasons.append("ROI帧间亮度差分残差过大")
    geometry_passed = not reasons
    semantic = None
    if args.semantic_review:
        semantic = json.loads(args.semantic_review.read_text())
    semantic_ok = bool(review_bound_to_video(semantic, generation_sha, output_sha, config["required_frames"])
                       and semantic.get("object_identity_preserved") is True
                       and semantic.get("task_semantics_preserved") is True
                       and semantic.get("unwanted_objects") is False)
    if config["semantic_review_required"] and not semantic_ok:
        reasons.append("缺少覆盖全部93帧、绑定生成摘要的物体身份与任务语义复核")
    condition_review_path = getattr(args, "condition_review", None)
    condition_review = json.loads(condition_review_path.read_text()) if condition_review_path else None
    condition_ok = target_condition_review_passed(condition_review, evidence.get("target_condition"),
                                                   generation_sha, output_sha, config["required_frames"])
    if not condition_ok:
        reasons.append("缺少明确目标外观及绑定实际视频的全帧分区域条件验收")
    report = {"generation_probe": str(args.probe), "generation_probe_sha256": generation_sha,
              "output_kind": "native_cosmos" if generation["stage"] == "transfer_smoke" else result["derived_kind"],
              "native_generation_probe": result.get("native_generation_probe"),
              "native_generation_probe_sha256": result.get("native_generation_probe_sha256"),
              "output_video": str(output_path), "output_video_sha256": output_sha,
              "source_input": str(evidence_path), "source_input_sha256": sha256(evidence_path),
              "gate_config": str(args.config), "gate_config_sha256": sha256(args.config),
              "sam_model_manifest": str(model_manifest_path), "sam_model_manifest_sha256": sha256(model_manifest_path),
              "method": (("仿真掩码导出box和12个正点/12个负点；已独立校准全部源帧；" if calibration_path
                           else "仿真掩码导出box和最多3个连通区域点提示；")
                          + "源图按真值选候选粒度；生成图沿用同一候选编号"),
              "source_calibration_report": ({"path": str(calibration_path), "sha256": sha256(calibration_path)}
                                            if calibration_path else None),
              "precision": "SAM2 fp32参数/bf16 autocast，RGB uint8，CPU指标计算fp32/fp64",
              "seed": args.seed, "batch": 2, "sample_count": 1, "segmented_frames": indices,
              "all_frames_segmented": len(indices) == 93, "per_frame": rows,
              "roi_edge_f1_mean": float(np.mean(edge_values)) if edge_values else None,
              "temporal_residual_p95": float(np.percentile(temporal, 95)), "temporal_residual_per_frame": temporal,
              "geometry_passed": geometry_passed, "semantic_review_passed": semantic_ok,
              "target_condition": evidence.get("target_condition"), "target_condition_passed": condition_ok,
              "target_condition_review": condition_review,
              "semantic_review": semantic, **legacy_gate_decision(generation["stage"], reasons),
              "reject_reasons": sorted(set(reasons)), "wall_s": time.monotonic() - before,
              "torch_peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
              "limitation": "提示依赖源位置；分割IoU和颜色比例是几何与颜色检查，不是独立语义准确率评估"}
    (directory / "quality_gate.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    # 每张总览均覆盖真实帧；所有逐帧图片保留，后续语义复核必须显式列出覆盖帧。
    for start in range(0, len(indices), 12):
        group = indices[start:start + 12]
        sheet = Image.new("RGB", (1024, 6 * 280), "white")
        draw = ImageDraw.Draw(sheet)
        for slot, index in enumerate(group):
            left, top = (slot % 2) * 512, (slot // 2) * 280
            draw.text((left + 6, top + 3), f"Frame {index}: source | generated", fill="black")
            sheet.paste(Image.open(directory / f"roi_{index:03d}.png"), (left, top + 22))
        sheet.save(directory / f"contact_sheet_{start:03d}.png")
    return {key: report[key] for key in ["generation_probe", "sample_count", "segmented_frames", "passed",
                                        "eligible_for_training", "reject_reasons", "precision", "seed", "batch"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--probe", type=pathlib.Path, required=True)
    parser.add_argument("--config", type=pathlib.Path, default=ROOT / "configs/gen_gate.yaml")
    parser.add_argument("--stride", type=int, default=1, help="1检查全部帧；大于1仅探针，不放行训练")
    parser.add_argument("--semantic-review", type=pathlib.Path)
    parser.add_argument("--condition-review", type=pathlib.Path, help="绑定实际视频的全帧目标外观验收；光照必须分机器人、目标物和背景")
    parser.add_argument("--calibration", type=pathlib.Path, help="已通过全部源帧的分割提示校准报告")
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()
    arguments.probe = arguments.probe.resolve(strict=True)
    arguments.config = arguments.config.resolve(strict=True)
    for key in ["semantic_review", "condition_review"]:
        value = getattr(arguments, key)
        if value:
            value = value.resolve(strict=True)
            if not value.is_relative_to(ROOT / "runs/p0"):
                parser.error("语义及目标条件复核必须位于项目runs/p0")
            setattr(arguments, key, value)
    if arguments.calibration:
        arguments.calibration = arguments.calibration.resolve(strict=True)
        if not arguments.calibration.is_relative_to(ROOT / "runs/p0"):
            parser.error("校准记录必须位于项目runs/p0")
    if not arguments.probe.is_relative_to(ROOT / "runs/p0") or not 1 <= arguments.stride <= 92:
        parser.error("生成记录必须位于项目runs/p0；stride在1至92之间")
    raise SystemExit(run_stage("generated_window_gate", lambda directory: evaluate(directory, arguments),
                              {key: str(value) if isinstance(value, pathlib.Path) else value
                               for key, value in vars(arguments).items()}))
