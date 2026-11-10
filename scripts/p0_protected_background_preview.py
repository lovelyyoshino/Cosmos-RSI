#!/usr/bin/env python3
"""预览Cosmos背景与源前景的显式合成，只诊断，不生成可入训来源。

@author pony
@date 2026-11-07
@version v1.0.0
@last_modified 2026-11-07
@changelog
    - v1.0.0 (2026-11-07): 保存原生/派生区别、逐帧像素保护证据及只读来源，资格固定为false。
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib

from p0_common import ROOT, run_stage
from p0_generated_window_gate import verified_generation_paths
from p0_policy_runtime import sha256


def protected_background(source, generated, rois, radius: int = 8):
    import cv2
    import numpy as np

    source, generated, rois = np.asarray(source), np.asarray(generated), np.asarray(rois)
    if source.dtype != np.uint8 or generated.dtype != np.uint8 or source.ndim != 3 or source.shape[-1] != 3 \
            or source.shape != generated.shape or rois.ndim != 3 or rois.shape[1:] != source.shape[:2] \
            or rois.shape[0] < 1 or not np.isin(rois, [0, 1]).all() \
            or type(radius) is not int or radius < 0 or not rois.any():
        raise ValueError("要求同尺寸uint8 RGB、非空二值ROI和非负整数保护半径")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    protection = cv2.dilate(rois.any(axis=0).astype(np.uint8), kernel) > 0
    derived = generated.copy()
    # 源前景像素直接保留，背景取真实生成；这是明确的合成操作，不能称为原生模型输出。
    derived[protection] = source[protection]
    return derived, protection


def checked_preview_frames(probe_path: pathlib.Path, source_sha: str, required_frames: int):
    import numpy as np

    probe = json.loads(probe_path.read_text())
    result = probe.get("result", {})
    if probe.get("stage") != "protected_background_preview" or probe.get("status") != "passed" \
            or result.get("source_input_sha256") != source_sha or result.get("eligible_for_training") is not False \
            or result.get("derived_kind") != "cosmos_background_source_foreground_composite":
        raise ValueError("预览必须绑定同源、已完成且明确不可入训的派生诊断")
    path = pathlib.Path(result["frames_file"]).resolve(strict=True)
    if path.parent != probe_path.parent or sha256(path) != result["frames_sha256"] \
            or probe.get("artifacts_sha256", {}).get(path.name) != result["frames_sha256"]:
        raise ValueError("预览数组越界或偏离实际摘要")
    frames = np.load(path, mmap_mode="r", allow_pickle=False)
    if frames.dtype != np.uint8 or frames.shape != (required_frames, 256, 256, 3):
        raise ValueError("预览数组尺寸、类型或逐帧长度错误")
    return frames


def preview(directory: pathlib.Path, args: argparse.Namespace) -> dict:
    import cv2
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    generation = json.loads(args.probe.read_text())
    source_path, native_path = verified_generation_paths(generation, args.probe)
    evidence = json.loads(source_path.read_text())
    if not evidence.get("source_success") or not evidence.get("replay_validation", {}).get("passed"):
        raise RuntimeError("预览只接受实际成功并精确复现的训练源")
    for name, expected in {**evidence["files_sha256"], "source_labels.npz": evidence["labels_sha256"]}.items():
        if pathlib.PurePosixPath(name).name != name or sha256(source_path.parent / name) != expected:
            raise RuntimeError("成功源文件或动作标签已改变")
    rois = np.load(source_path.parent / "roi_masks.npy", allow_pickle=False)
    if rois.shape != (93, len(evidence["roi_names"]), 256, 256):
        raise RuntimeError("源真值ROI尺寸错误")
    videos = []
    for path in [source_path.parent / "source.mp4", native_path]:
        with imageio.get_reader(path) as reader:
            videos.append(np.asarray([cv2.resize(frame, (256, 256), interpolation=cv2.INTER_AREA) for frame in reader]))
    if any(video.shape != (93, 256, 256, 3) for video in videos):
        raise RuntimeError("只接受同帧93帧，不补帧或移动时间窗")
    font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 16, index=2)
    derived, rows = [], []
    for index, (source, generated, masks) in enumerate(zip(*videos, rois)):
        output, protection = protected_background(source, generated, masks, args.radius)
        foreground = masks.any(axis=0)
        maximum = int(np.abs(output[foreground].astype(np.int16) - source[foreground].astype(np.int16)).max())
        if maximum != 0:
            raise RuntimeError("派生预览改变了需要保留的前景像素")
        derived.append(output)
        rows.append({"frame": index, "source_roi_pixels": int(foreground.sum()),
                     "protected_pixels": int(protection.sum()), "foreground_max_abs_diff": maximum,
                     "background_from_native_pixels": int((~protection).sum()),
                     "background_mean_abs_diff_vs_source": float(np.abs(
                         output[~protection].astype(np.float32) - source[~protection].astype(np.float32)).mean())})
        if index in [0, 46, 92]:
            sheet = Image.new("RGB", (768, 328), "#f5f4ef")
            draw = ImageDraw.Draw(sheet)
            for column, (label, picture) in enumerate(zip(["成功训练源", "Cosmos原生输出", "前景保护派生预览"], [source, generated, output])):
                draw.text((column * 256 + 4, 4), f"帧{index}：{label}", font=font, fill="#20313c")
                sheet.paste(Image.fromarray(picture), (column * 256, 28))
            draw.text((4, 288), "仅背景改写预览；派生合成，不是原生输出、全场光照或RSI收益。", font=font, fill="#6c4032")
            draw.text((4, 307), "未通过完整几何/语义/目标条件验收，不可入训。", font=font, fill="#6c4032")
            sheet.save(directory / f"protected_background_{index:03d}.png")
    frames_path = directory / "preview_frames.npy"
    np.save(frames_path, np.asarray(derived))
    with (directory / "pixel_preservation.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    result = {"derived_kind": "cosmos_background_source_foreground_composite",
              "source_input": str(source_path), "source_input_sha256": sha256(source_path),
              "native_generation_probe": str(args.probe), "native_generation_probe_sha256": sha256(args.probe),
              "native_video": str(native_path), "native_video_sha256": sha256(native_path),
              "source_labels_sha256": evidence["labels_sha256"], "roi_masks_sha256": evidence["files_sha256"]["roi_masks.npy"],
              "protection_radius_source_pixels": args.radius, "feathering": False,
              "frames_file": str(frames_path), "frames_sha256": sha256(frames_path),
              "foreground_max_abs_diff_all_frames": max(row["foreground_max_abs_diff"] for row in rows),
              "frames": 93, "fps": 20, "sample_count": 1, "batch": None, "seed": None,
              "precision": "CPU uint8图像与布尔真值掩码；无模型推理", "generation_performed": False,
              "eligible_for_training": False, "target_condition": None,
              "limitation": "仅复用历史原生输出验证显式背景合成；不向旧源补目标，不声称全场暖暗或完整语义通过"}
    (directory / "preview.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--probe", type=pathlib.Path, required=True)
    parser.add_argument("--radius", type=int, default=8)
    arguments = parser.parse_args()
    arguments.probe = arguments.probe.resolve(strict=True)
    if not arguments.probe.is_relative_to(ROOT / "runs/p0") or not 0 <= arguments.radius <= 32:
        parser.error("只接受项目原生生成与0至32源像素的保护半径")
    raise SystemExit(run_stage("protected_background_preview", lambda directory: preview(directory, arguments),
                               {"compute": "Thor CPU only", "eligible_for_training": False, "radius": arguments.radius}))
