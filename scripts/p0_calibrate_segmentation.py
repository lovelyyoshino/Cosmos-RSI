#!/usr/bin/env python3
"""只用成功示范的仿真真值校准分割提示，不读取生成图或冻结评测图。

@author pony
@date 2026-11-07
@version v1.0.0
@last_modified 2026-11-07
@changelog
    - v1.0.0 (2026-11-07): 对比原始三点提示与覆盖ROI内外的提示，保留逐帧校准及截图。
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import time

from p0_common import ROOT, ensure_gpu_idle, run_stage
from p0_generated_window_gate import mask_iou, prompted_regions
from p0_policy_runtime import sha256


def spread_points(mask, count: int):
    """从区域内部向外铺点，避免重复点把细长部件或断连部件遗漏。"""
    import cv2
    import numpy as np

    if count < 1:
        raise ValueError("提示点数必须为正整数")
    mask = np.asarray(mask, dtype=bool)
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return []
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    depth = distance[ys, xs].astype(np.float64)
    selected = [int(depth.argmax())]
    nearest = np.full(len(xs), np.inf)
    for _ in range(1, min(count, len(xs))):
        last = selected[-1]
        nearest = np.minimum(nearest, (xs - xs[last]) ** 2 + (ys - ys[last]) ** 2)
        selected.append(int((nearest * np.sqrt(np.maximum(depth, 0.1))).argmax()))
    points = [[int(xs[i]), int(ys[i])] for i in selected]
    return points + [points[-1]] * (count - len(points))


def balanced_prompts(masks, count: int = 12):
    import cv2
    import numpy as np

    boxes, _ = prompted_regions(masks)
    points, labels = [], []
    for mask, box in zip(masks, boxes):
        mask = np.asarray(mask, dtype=bool)
        positive = spread_points(mask, count)
        if not positive:
            raise ValueError("空ROI没有可校准的正提示")
        region = np.zeros_like(mask)
        x0, y0, x1, y1 = box
        region[y0:y1 + 1, x0:x1 + 1] = True
        # 负点与轮廓隔开一像素，减小压缩边缘对源图校准的影响。
        negative_region = region & ~(cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
        if not negative_region.any():
            negative_region = region & ~mask
        negative = spread_points(negative_region, count)
        # 满屏物体没有背景时，用SAM的忽略标签补齐，不制造错误负点。
        points.append(positive + (negative or [[0, 0]] * count))
        labels.append([1] * count + ([0] * count if negative else [-1] * count))
    return boxes, points, labels


def calibrate(directory: pathlib.Path, args: argparse.Namespace) -> dict:
    ensure_gpu_idle()
    import cv2
    import imageio.v2 as imageio
    import numpy as np
    import torch
    import yaml
    from PIL import Image, ImageDraw
    from transformers import Sam2Model, Sam2Processor

    config = yaml.safe_load(args.config.read_text())
    evidence = json.loads(args.source_input.read_text())
    if evidence.get("source_success") is not True or not evidence.get("replay_validation", {}).get("passed"):
        raise RuntimeError("校准只允许使用已验证成功及逐帧复现的训练示范")
    for name, expected in evidence["files_sha256"].items():
        if sha256(args.source_input.parent / name) != expected:
            raise RuntimeError("成功示范的图像或掩码摘要发生变化")
    if sha256(args.source_input.parent / "source_labels.npz") != evidence["labels_sha256"]:
        raise RuntimeError("成功示范的动作或状态标签发生变化")
    masks = np.load(args.source_input.parent / "roi_masks.npy", allow_pickle=False).astype(bool)
    names = evidence["roi_names"]
    required = config["required_frames"]
    if masks.shape != (required, len(names), 256, 256):
        raise RuntimeError("仿真真值ROI形状不符合已冻结的生成窗口约定")
    frames = []
    with imageio.get_reader(args.source_input.parent / "source.mp4") as reader:
        for frame in reader:
            frames.append(cv2.resize(frame, (256, 256), interpolation=cv2.INTER_AREA))
    if len(frames) != required:
        raise RuntimeError("成功示范窗口帧数错误，禁止插值或补帧")
    indices = list(range(required)) if args.all_frames else sorted(set(args.frames))
    if not indices or any(i < 0 or i >= required for i in indices):
        raise ValueError("校准帧号必须落在成功示范窗口内")
    candidates = []
    for path in sorted(ROOT.glob("runs/p0/model_download_*/model_manifest.json")):
        manifest = json.loads(path.read_text())
        if pathlib.Path(manifest["path"]).name == "sam2.1-hiera-tiny":
            candidates.append((path, manifest))
    if not candidates:
        raise RuntimeError("缺少已经逐文件核验的SAM2模型")
    manifest_path, manifest = candidates[-1]
    model_path = pathlib.Path(manifest["path"])
    for item in manifest["files"]:
        if sha256(model_path / item["file"]) != item["sha256"]:
            raise RuntimeError("SAM2文件摘要不匹配")
    torch.manual_seed(args.seed)
    model = Sam2Model.from_pretrained(model_path, local_files_only=True).to("cuda").eval()
    processor = Sam2Processor.from_pretrained(model_path, local_files_only=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    rows = []
    for index in indices:
        visible = [i for i, mask in enumerate(masks[index]) if mask.sum() >= config["visible_pixels_min"]]
        if not visible:
            raise RuntimeError(f"第{index}帧没有可见ROI")
        boxes, points = prompted_regions(masks[index, visible])
        recipes = {"original3": (boxes, points, [[1, 1, 1] for _ in visible]),
                   "balanced12": balanced_prompts(masks[index, visible], 12)}
        pictures = [frames[index].copy() for _ in range(4)]
        row = {"frame": index, "variants": {}}
        for slot, (recipe, (boxes, points, labels)) in enumerate(recipes.items(), start=2):
            inputs = processor(images=[Image.fromarray(frames[index])], input_boxes=[boxes],
                               input_points=[points], input_labels=[labels], return_tensors="pt")
            sizes = inputs["original_sizes"]
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(**inputs.to("cuda"), multimask_output=True)
            predicted = processor.post_process_masks(outputs.pred_masks.cpu(), sizes.cpu())[0]
            per_roi = {}
            for position, group in enumerate(visible):
                reference = masks[index, group]
                selected = max(range(predicted.shape[1]),
                               key=lambda i: mask_iou(reference, predicted[position, i].numpy()))
                prediction = predicted[position, selected].numpy().astype(bool)
                per_roi[names[group]] = {"iou": mask_iou(reference, prediction), "candidate": selected,
                                         "reference_pixels": int(reference.sum()),
                                         "predicted_pixels": int(prediction.sum())}
                color = [(20, 220, 70), (255, 140, 30), (50, 140, 255)][group % 3]
                for picture, mask in [(pictures[1], reference), (pictures[slot], prediction)]:
                    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                                  cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(picture, contours, -1, color, 1)
            row["variants"][recipe] = per_roi
        canvas = Image.new("RGB", (1024, 290), "white")
        draw = ImageDraw.Draw(canvas)
        for slot, (label, picture) in enumerate(zip(["Source", "Sim GT", "Original 3 positive",
                                                   "12 positive + 12 negative"], pictures)):
            draw.text((slot * 256 + 5, 4), f"{index}: {label}", fill="black")
            canvas.paste(Image.fromarray(picture), (slot * 256, 28))
        canvas.save(directory / f"calibration_{index:03d}.png")
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    summaries = {}
    for recipe in ["original3", "balanced12"]:
        summaries[recipe] = {}
        for name in names:
            items = [(row["frame"], row["variants"][recipe][name]["iou"]) for row in rows
                     if name in row["variants"][recipe]]
            if not items:
                continue
            values = [value for _, value in items]
            summaries[recipe][name] = {"visible_frames": len(items), "min_iou": min(values),
                                       "mean_iou": float(np.mean(values)),
                                       "below_threshold_frames": [frame for frame, value in items
                                                                  if value < config["source_segmentation_iou_min"]]}
    report = {"source_input": str(args.source_input), "source_input_sha256": sha256(args.source_input),
              "sam_model_manifest": str(manifest_path), "sam_model_manifest_sha256": sha256(manifest_path),
              "gate_config_sha256": sha256(args.config), "threshold": config["source_segmentation_iou_min"],
              "precision": "SAM2 fp32参数/bf16 autocast，RGB uint8，IoU在CPU计算", "batch": 1,
              "seed": args.seed, "sample_count": 1, "frames": indices, "all_frames": len(indices) == required,
              "summary": summaries, "per_frame": rows, "wall_s_without_model_loading": time.monotonic() - started,
              "torch_peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
              "balanced_calibration_passed": len(indices) == required and bool(summaries["balanced12"])
              and all(not value["below_threshold_frames"] for value in summaries["balanced12"].values()),
              "eligible_for_training": False,
              "limitation": "仅校准成功训练示范源图；不检查生成图，不替代几何/目标条件/全帧语义门，也不代表策略收益"}
    (directory / "calibration.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    with (directory / "calibration.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame", "recipe", "roi", "iou", "candidate", "reference_pixels", "predicted_pixels"])
        for row in rows:
            for recipe, values in row["variants"].items():
                for name, value in values.items():
                    writer.writerow([row["frame"], recipe, name, value["iou"], value["candidate"],
                                     value["reference_pixels"], value["predicted_pixels"]])
    return {key: report[key] for key in ["source_input", "batch", "seed", "sample_count", "frames", "summary",
                                        "balanced_calibration_passed", "eligible_for_training"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-input", type=pathlib.Path, required=True)
    parser.add_argument("--config", type=pathlib.Path, default=ROOT / "configs/gen_gate.yaml")
    parser.add_argument("--frames", type=int, nargs="+", default=[0])
    parser.add_argument("--all-frames", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    raise SystemExit(run_stage("segmentation_calibration", lambda directory: calibrate(directory, args),
                               {"purpose": "仅成功D0源图的分割校准", "seed": args.seed,
                                "batch": 1, "all_frames": args.all_frames, "frames": args.frames}))
