#!/usr/bin/env python3
"""隔离源任务ROI邻域外的背景，独立检查误拒与形变检出；仅诊断，不放行数据。

@author pony
@date 2026-11-08
@version v1.1.0
@last_modified 2026-11-08
@changelog
    - v1.1.0 (2026-11-08): 新增独立预声明的逐ROI上下文消融及问题帧先行检查，保留原union方法。
    - v1.0.0 (2026-11-08): 固定半径8及中性背景，对比背景负控、全局/局部位移和机器人遮挡。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import time

from p0_background_preview_materials import checked_artifact
from p0_common import ROOT, ensure_gpu_idle, run_stage
from p0_generated_window_gate import mask_iou
from p0_policy_runtime import sha256


BASE_VARIANTS = ["identity", "resize_codec", "analytic_background_dim", "analytic_background_gray", "shift24"]
NEGATIVES = BASE_VARIANTS[:4]
POSITIVES = ["shift24", "book_shift12", "robot_box_occluded"]


def normalized_context(source, candidate, rois, radius: int = 8):
    """只中和远处上下文；邻域内必须保留实际候选像素，不能用源图修补错误。"""
    import cv2
    import numpy as np

    source, candidate, rois = np.asarray(source), np.asarray(candidate), np.asarray(rois)
    if source.dtype != np.uint8 or candidate.dtype != np.uint8 or source.ndim != 3 or source.shape[-1] != 3 \
            or source.shape != candidate.shape or rois.dtype != np.bool_ or rois.ndim != 3 \
            or rois.shape[1:] != source.shape[:2] or not rois.any() or type(radius) is not int or radius < 0:
        raise ValueError("背景隔离要求同尺寸uint8 RGB、非空bool源ROI及非负整数半径")
    support = cv2.dilate(rois.any(axis=0).astype(np.uint8),
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))) > 0
    first, second = np.full_like(source, 128), np.full_like(candidate, 128)
    first[support], second[support] = source[support], candidate[support]
    return first, second, support


def roi_context_pairs(source, candidate, rois, radius: int = 8):
    """每个ROI独立隔离背景，保留候选实际像素；不把其他目标的源像素写回候选。"""
    import numpy as np

    rois = np.asarray(rois)
    if rois.ndim != 3 or not len(rois) or rois.dtype != np.bool_ or not all(mask.any() for mask in rois):
        raise ValueError("逐ROI上下文要求每个目标均可见且为bool真值")
    return [normalized_context(source, candidate, rois[index:index + 1], radius) for index in range(len(rois))]


def object_shift(source, mask, pixels: int = 12):
    """移除原位置再粘贴同一物体RGB，保存已知平移真值；不构造训练标签。"""
    import cv2
    import numpy as np

    if source.dtype != np.uint8 or mask.dtype != np.bool_ or mask.shape != source.shape[:2] \
            or not mask.any() or type(pixels) is not int or pixels <= 0:
        raise ValueError("局部位移要求可见bool物体与正整数位移")
    matrix = np.float32([[1, 0, pixels], [0, 1, 0]])
    size = (source.shape[1], source.shape[0])
    shifted_mask = cv2.warpAffine(mask.astype(np.uint8), matrix, size, flags=cv2.INTER_NEAREST) > 0
    shifted_rgb = cv2.warpAffine(source, matrix, size, flags=cv2.INTER_NEAREST)
    output = cv2.inpaint(source, mask.astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA)
    output[shifted_mask] = shifted_rgb[shifted_mask]
    return output, shifted_mask


def box_occlusion(source, mask):
    """用矩形遮挡机器人及8px余量，避免沿原轮廓填色反而泄露正确形状。"""
    import numpy as np

    if source.dtype != np.uint8 or mask.dtype != np.bool_ or mask.shape != source.shape[:2] or not mask.any():
        raise ValueError("遮挡需要源RGB及非空bool目标")
    ys, xs = np.nonzero(mask)
    region = np.zeros_like(mask)
    region[max(0, int(ys.min()) - 8):min(mask.shape[0], int(ys.max()) + 9),
           max(0, int(xs.min()) - 8):min(mask.shape[1], int(xs.max()) + 9)] = True
    output = source.copy(); output[region] = 128
    return output, region


def summarize(rows: list[dict], names: list[str], required: int = 93) -> dict:
    """部分帧不产生验证通过；正控按事先声明的目标ROI判定，不能用别处误拒凑数。"""
    variants = BASE_VARIANTS + POSITIVES[1:]
    if not rows or len(set(names)) != len(names) or not {"robot", "black_book_1"} <= set(names) \
            or any(list(row.get("variants", {})) != variants for row in rows):
        raise ValueError("诊断缺少固定干预或任务ROI")
    for row in rows:
        if not row.get("visible_rois") or any(set(metrics) != set(row["visible_rois"])
                                              for metrics in row["variants"].values()):
            raise ValueError("正负控不能省略源图可见ROI")
        for metrics in row["variants"].values():
            for name, metric in metrics.items():
                value, threshold = metric.get("iou", float("nan")), .8 if name == "robot" else .75
                if name not in names or not math.isfinite(value) or not 0 <= value <= 1 \
                        or metric.get("threshold") != threshold \
                        or metric.get("passed_original_roi") is not (value >= threshold):
                    raise ValueError("诊断指标或布尔判定偏离原阈值")
    complete = [row["frame"] for row in rows] == list(range(required))
    summary = {}
    for variant in variants:
        summary[variant] = {}
        for name in names:
            values = [(row["frame"], row["variants"][variant][name]) for row in rows
                      if name in row["variants"][variant]]
            if not values:
                raise ValueError("诊断ROI没有可见测量")
            summary[variant][name] = {"visible_frames": len(values), "min_iou": min(v["iou"] for _, v in values),
                                    "rejected_frames": [frame for frame, value in values if not value["passed_original_roi"]]}
    negative_pass = all(not item["rejected_frames"] for variant in NEGATIVES for item in summary[variant].values())
    positive = {}
    for variant, target in [("shift24", "robot"), ("book_shift12", "black_book_1"), ("robot_box_occluded", "robot")]:
        metric = summary[variant][target]
        positive[variant] = {"target_roi": target, "visible_frames": metric["visible_frames"],
                             "detected_frames": len(metric["rejected_frames"]),
                             "all_visible_frames_detected": len(metric["rejected_frames"]) == metric["visible_frames"]}
    return {"summary": summary, "all_frames": complete, "negative_controls_all_roi_passed": negative_pass,
            "positive_controls": positive, "diagnostic_validation_passed": complete and negative_pass
            and all(item["all_visible_frames_detected"] for item in positive.values())}


def inspect(directory: pathlib.Path, args: argparse.Namespace) -> dict:
    ensure_gpu_idle()
    import cv2
    import numpy as np
    import torch
    import yaml
    from PIL import Image, ImageDraw
    from transformers import Sam2Model, Sam2Processor
    from p0_calibrate_segmentation import balanced_prompts

    config = yaml.safe_load(args.config.read_text())
    expected = {"version": "source-roi-context-neutral-v1", "diagnostic_only": True, "protected_radius_pixels": 8,
                "neutral_rgb": [128, 128, 128], "object_shift_pixels": 12, "target_roi": "black_book_1",
                "robot_roi": "robot", "candidate_selection": "source_only_balanced12", "required_frames": 93,
                "design_episode_suffix": "caddy-demo_13", "heldout_episode_suffix": "caddy-demo_16"}
    individual = args.context_scope == "per_roi"
    if individual:
        expected.update(version="source-roi-individual-context-v1", design_episode_suffix="caddy-demo_16",
                        heldout_episode_suffix="caddy-demo_19", context_scope="per_roi",
                        previously_observed_episode_suffixes=["caddy-demo_13", "caddy-demo_16"])
    if config != expected:
        raise ValueError("诊断方法必须匹配预声明配置，不能在同次试验扫描参数")
    original_probe = checked_artifact(args.controls, "segmentation_controls")
    original = json.loads(args.controls.read_text())
    if original.get("frames") != list(range(93)) or original.get("all_frames") is not True \
            or original.get("eligible_for_training") is not False:
        raise ValueError("上下文诊断只接受既有完整且不可入训的背景负控")
    source_path = pathlib.Path(original["source_input"]).resolve(strict=True)
    if not source_path.is_relative_to(ROOT / "runs/p0") or sha256(source_path) != original["source_input_sha256"]:
        raise ValueError("实际训练源越界或摘要改变")
    evidence = json.loads(source_path.read_text())
    suffix = config["heldout_episode_suffix" if args.role == "heldout" else "design_episode_suffix"]
    if not evidence["source_episode_id"].endswith(suffix):
        raise ValueError("实际源不符合预声明的设计/独立验证划分")
    cache = pathlib.Path(evidence["cache_manifest"]).resolve(strict=True)
    if not cache.is_relative_to(ROOT / "data") or sha256(cache) != evidence["cache_manifest_sha256"]:
        raise ValueError("D0来源改变或越界")
    registered = [row for row in json.loads(cache.read_text())["episodes"] if row["episode_id"] == evidence["source_episode_id"]]
    if len(registered) != 1 or registered[0]["split"] != "demo_train" or registered[0]["env_success"] is not True:
        raise ValueError("诊断禁止使用验证集或冻结评测来源")
    gate_config_path = ROOT / "configs/gen_gate.yaml"
    if original["gate_config_sha256"] != sha256(gate_config_path):
        raise ValueError("原质量门配置改变")
    gate = yaml.safe_load(gate_config_path.read_text())
    design_binding = None
    if args.role == "heldout":
        if args.design_report is None:
            raise ValueError("独立验证必须绑定已通过的完整设计检查")
        design_probe = checked_artifact(args.design_report, "context_normalized_segmentation")
        design = json.loads(args.design_report.read_text())
        dependencies = ["scripts/p0_context_normalized_segmentation.py", "scripts/p0_calibrate_segmentation.py",
                        "scripts/p0_generated_window_gate.py", str(args.config.relative_to(ROOT))]
        if design.get("role") != "design" or design.get("diagnostic_validation_passed") is not True \
                or design.get("all_frames") is not True or design.get("method_config_sha256") != sha256(args.config) \
                or design.get("sam_manifest_sha256") != original["sam_manifest_sha256"] \
                or design.get("original_gate_config_sha256") != sha256(gate_config_path) \
                or design.get("seed") != args.seed or design.get("batch") != 2 \
                or any(design_probe.get("source_files_sha256", {}).get(name) != sha256(ROOT / name)
                       for name in dependencies):
            raise ValueError("设计检查未通过或方法/模型/配置改变，禁止借用旧设计结果")
        design_binding = {"report": str(args.design_report), "sha256": sha256(args.design_report),
                          "source_episode_id": design["source_episode_id"], "method_and_dependencies_unchanged": True}
    masks_path = source_path.parent / "roi_masks.npy"
    if sha256(masks_path) != evidence["files_sha256"]["roi_masks.npy"]:
        raise ValueError("源ROI改变")
    masks = np.load(masks_path, allow_pickle=False)
    names = evidence["roi_names"]
    if masks.shape != (93, len(names), 256, 256) or not np.isin(masks, [0, 1]).all():
        raise ValueError("源ROI必须为完整二值真值")
    manifest_paths = [p for p in ROOT.glob("runs/p0/model_download_*/model_manifest.json")
                      if sha256(p) == original["sam_manifest_sha256"]]
    if len(manifest_paths) != 1:
        raise ValueError("不能定位原SAM2清单")
    manifest_path = manifest_paths[0]
    manifest = json.loads(manifest_path.read_text())
    model_path = pathlib.Path(manifest["path"]).resolve(strict=True)
    if model_path != ROOT / "checkpoints/base/sam2.1-hiera-tiny":
        raise ValueError("模型不在本项目已验证目录")
    for entry in manifest["files"]:
        path = (model_path / entry["file"]).resolve(strict=True)
        if not path.is_relative_to(model_path) or sha256(path) != entry["sha256"]:
            raise ValueError("SAM2文件内容改变或越界")
    torch.manual_seed(args.seed)
    model = Sam2Model.from_pretrained(model_path, local_files_only=True).to("cuda").eval()
    processor = Sam2Processor.from_pretrained(model_path, local_files_only=True)
    torch.cuda.reset_peak_memory_stats()
    indices = list(range(93)) if args.all_frames else sorted(set(args.frames))
    rows = []
    started = time.monotonic()
    for frame in indices:
        input_path = args.controls.parent / f"control_inputs_{frame:03d}.npz"
        if sha256(input_path) != original_probe["artifacts_sha256"].get(input_path.name):
            raise ValueError("原负控的实际输入数组改变")
        with np.load(input_path, allow_pickle=False) as archive:
            variants = {key: archive[key] for key in BASE_VARIANTS}
            source = variants["identity"]
            np.testing.assert_array_equal(archive["source_rois"], masks[frame].astype(bool))
        if hashlib.sha256(source.tobytes()).hexdigest() != evidence["source_rgb_sha256"][frame]:
            raise ValueError("实际源RGB偏离训练源")
        book, robot = names.index("black_book_1"), names.index("robot")
        variants["book_shift12"], shifted_book = object_shift(source, masks[frame, book].astype(bool), 12)
        variants["robot_box_occluded"], occlusion = box_occlusion(source, masks[frame, robot].astype(bool))
        rois = masks[frame].astype(bool)
        visible = [i for i, mask in enumerate(rois) if mask.sum() >= gate["visible_pixels_min"]]
        boxes, points, labels = balanced_prompts(rois[visible], 12)
        row = {"frame": frame, "visible_rois": [names[i] for i in visible], "variants": {}}
        raw = {"source_rois": rois, "shifted_book_gt": shifted_book, "robot_occlusion_region": occlusion}
        panels = []
        for variant, actual in variants.items():
            pairs = roi_context_pairs(source, actual, rois[visible], 8) if individual else [
                normalized_context(source, actual, rois, 8)]
            predictions = []
            for position, (first, second, support) in enumerate(pairs):
                box_batch = [boxes[position]] if individual else boxes
                point_batch = [points[position]] if individual else points
                label_batch = [labels[position]] if individual else labels
                inputs = processor(images=[Image.fromarray(first), Image.fromarray(second)],
                                   input_boxes=[box_batch, box_batch], input_points=[point_batch, point_batch],
                                   input_labels=[label_batch, label_batch], return_tensors="pt")
                sizes = inputs["original_sizes"]
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    output = model(**inputs.to("cuda"), multimask_output=True)
                predictions.append(processor.post_process_masks(output.pred_masks.cpu(), sizes.cpu()))
            predicted = [torch.cat([prediction[side] for prediction in predictions], dim=0) for side in range(2)]
            values, chosen = {}, []
            picture = actual.copy() if individual else second.copy()
            for position, group in enumerate(visible):
                reference = rois[group]
                candidate = max(range(predicted[0].shape[1]),
                                key=lambda i: mask_iou(reference, predicted[0][position, i].numpy()))
                mask = predicted[1][position, candidate].numpy().astype(bool)
                threshold = gate["robot_mask_iou_min" if names[group] == "robot" else "object_mask_iou_min"]
                iou = mask_iou(reference, mask)
                values[names[group]] = {"iou": iou, "threshold": threshold, "passed_original_roi": iou >= threshold,
                                       "source_selected_candidate": candidate,
                                       "source_calibration_iou": mask_iou(reference, predicted[0][position, candidate].numpy())}
                chosen.append(mask)
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(picture, contours, -1, [(20, 220, 70), (255, 140, 30), (50, 140, 255)][group % 3], 1)
            row["variants"][variant] = values
            raw.update({variant + "_actual": actual, variant + "_normalized": second,
                        variant + "_predicted": np.asarray(chosen)})
            if individual:
                raw.update({variant + "_roi_normalized_sources": np.asarray([pair[0] for pair in pairs]),
                            variant + "_roi_normalized_candidates": np.asarray([pair[1] for pair in pairs]),
                            variant + "_roi_context_support": np.asarray([pair[2] for pair in pairs])})
            panels.append(picture)
        if not individual:
            raw.update({"normalized_source": first, "context_support": support})
        else:
            # 逐ROI输入是多幅图，不能把最后一幅误标为覆盖全部ROI的单一归一化输入。
            raw = {key: value for key, value in raw.items() if not key.endswith("_normalized")}
        np.savez_compressed(directory / f"context_inputs_{frame:03d}.npz", **raw)
        if frame in (0, 46, 87, 92):
            sheet = Image.new("RGB", (256 * len(panels), 290), "white")
            draw = ImageDraw.Draw(sheet)
            for slot, (variant, picture) in enumerate(zip(variants, panels)):
                draw.text((slot * 256 + 3, 5), f"{frame:03d} {variant}", fill="black")
                sheet.paste(Image.fromarray(picture), (slot * 256, 30))
            sheet.save(directory / f"normalized_{frame:03d}.png")
        rows.append(row)
        print(json.dumps(row), flush=True)
    summary = summarize(rows, names)
    source_calibrated = all(metric["source_calibration_iou"] >= gate["source_segmentation_iou_min"]
                            for row in rows for rois in row["variants"].values() for metric in rois.values())
    if individual and not source_calibrated:
        summary["diagnostic_validation_passed"] = False
    result = {"method": config["version"], "method_config_sha256": sha256(args.config), "role": args.role,
              "source_input": str(source_path), "source_input_sha256": sha256(source_path),
              "source_episode_id": evidence["source_episode_id"], "original_controls": str(args.controls),
              "original_controls_sha256": sha256(args.controls), "original_gate_config_sha256": sha256(gate_config_path),
              "sam_manifest_sha256": sha256(manifest_path), "original_summary": original["summary"],
              "precision": "SAM2 fp32参数/bf16 autocast；RGB uint8；CPU IoU", "batch": 2, "seed": args.seed,
              "sample_count": 1, "frames": indices, "per_frame": rows, "design_binding": design_binding, **summary,
              "context_scope": args.context_scope, "source_segmentation_calibration_passed": source_calibrated,
              "wall_s_without_model_loading": time.monotonic() - started,
              "torch_peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
              "eligible_for_training": False, "original_gate_modified": False,
              "generation_performed": False, "training_performed": False, "promotion_performed": False,
              "limitation": "仅任务ROI及其8px邻域的分割诊断。远处新增物体仍须全帧语义检查；诊断通过不改变历史入训资格。"}
    (directory / "context_normalized.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return {key: result[key] for key in ["method", "role", "source_episode_id", "frames", "summary", "positive_controls",
                                       "diagnostic_validation_passed", "eligible_for_training"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--controls", type=pathlib.Path, required=True)
    parser.add_argument("--config", type=pathlib.Path, default=ROOT / "configs/geometry_context_diagnostic.yaml")
    parser.add_argument("--role", choices=["design", "heldout"], required=True)
    parser.add_argument("--design-report", type=pathlib.Path, help="heldout必须绑定相同方法、模型及原阈值的完整设计报告")
    parser.add_argument("--all-frames", action="store_true")
    parser.add_argument("--frames", type=int, nargs="+", default=[0], help="部分帧仅诊断，不代表完整方法验收")
    parser.add_argument("--context-scope", choices=["union", "per_roi"], default="union")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not args.frames or any(frame < 0 or frame >= 93 for frame in args.frames):
        parser.error("问题帧必须落在预声明窗口的0至92之间")
    for field in ("controls", "config"):
        path = getattr(args, field).resolve(strict=True)
        if not path.is_relative_to(ROOT):
            parser.error("诊断输入必须位于Thor项目目录")
        setattr(args, field, path)
    if args.seed != 42:
        parser.error("本次诊断固定seed42，不扫描种子")
    if args.design_report:
        args.design_report = args.design_report.resolve(strict=True)
        if not args.design_report.is_relative_to(ROOT / "runs/p0"):
            parser.error("设计报告必须位于Thor项目P0目录")
    raise SystemExit(run_stage("context_normalized_segmentation", lambda directory: inspect(directory, args),
                               {"role": args.role, "seed": args.seed, "batch": 2,
                                "all_frames": args.all_frames, "context_scope": args.context_scope,
                                "frames": args.frames, "diagnostic_only": True}))
