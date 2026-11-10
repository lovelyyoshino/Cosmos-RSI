#!/usr/bin/env python3
"""在固定设计/独立成功源上验证无损背景像素门的正负控，全部输出只诊断。

@author pony
@date 2026-11-08
@version v1.1.0
@last_modified 2026-11-08
@changelog
    - v1.1.0 (2026-11-08): 控制使用实际无损视频及正式文件核验入口，补机器人局部位移和原生错绑。
    - v1.0.0 (2026-11-08): 实际视频覆盖像素、位移、遮挡、乱序、尺寸/帧率/缺帧及标签篡改。
"""
from __future__ import annotations

import argparse
import json
import pathlib

from p0_background_preview_materials import checked_artifact
from p0_common import ROOT, ensure_gpu_idle, run_stage
from p0_context_normalized_segmentation import box_occlusion, object_shift
from p0_lossless_background import (
    METHOD_DEPENDENCIES, checked_bound_file, checked_config, checked_training_source,
    pixel_certificate, read_exact_video, validate_controls_summary, validate_labels,
)
from p0_policy_runtime import sha256
from p0_protected_background_preview import protected_background
from p0_segmentation_controls import check_independent_source


def write_video(path, frames, fps=20):
    import imageio.v2 as imageio

    with imageio.get_writer(path, fps=fps, codec="libx264rgb", pixelformat="rgb24", macro_block_size=1,
                           ffmpeg_params=["-crf", "0", "-preset", "veryfast", "-threads", "2"]) as writer:
        for frame in frames:
            writer.append_data(frame)


def controls(directory: pathlib.Path, args: argparse.Namespace) -> dict:
    ensure_gpu_idle()
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw

    config = checked_config(args.config)
    evidence, source, masks, source_proof = checked_training_source(args.source_input)
    suffix = config["design_episode_suffix" if args.role == "design" else "heldout_episode_suffix"]
    if not evidence["source_episode_id"].endswith(suffix):
        raise ValueError("诊断源不符合事先固定的设计/未见划分")
    dependencies = METHOD_DEPENDENCIES
    binding = None
    if args.role == "heldout":
        if args.design_report is None:
            raise ValueError("独立验证必须绑定已通过的完整设计检查")
        probe = checked_artifact(args.design_report, "lossless_background_controls")
        design = json.loads(args.design_report.read_text())
        validate_controls_summary(design, "design")
        if design.get("role") != "design" or design.get("validation_passed") is not True \
                or design.get("method_config_sha256") != sha256(args.config) \
                or design.get("frames") != list(range(93)) \
                or any(probe.get("source_files_sha256", {}).get(name) != sha256(ROOT / name) for name in dependencies):
            raise ValueError("完整设计未通过或方法/依赖改变，禁止借用旧结果")
        previous_path = pathlib.Path(design["source_input"]).resolve(strict=True)
        if not previous_path.is_relative_to(ROOT / "runs/p0") or sha256(previous_path) != design["source_input_sha256"]:
            raise ValueError("设计源绑定改变或越界")
        independence = check_independent_source(evidence, json.loads(previous_path.read_text()))
        binding = {"design_report": str(args.design_report), "design_report_sha256": sha256(args.design_report),
                   "same_method_and_dependencies": True, "source_independence": independence}
    names = evidence["roi_names"]
    native_dim = (source.astype(np.float32) * .5).astype(np.uint8)
    native_gray = np.full_like(source, 128)
    dim = np.asarray([protected_background(a, b, m, 8)[0] for a, b, m in zip(source, native_dim, masks)])
    gray = np.asarray([protected_background(a, b, m, 8)[0] for a, b, m in zip(source, native_gray, masks)])
    write_video(directory / "roundtrip_lossless.mp4", gray)
    roundtrip, roundtrip_metadata = read_exact_video(directory / "roundtrip_lossless.mp4")
    write_video(directory / "analytic_background_dim.mp4", dim)
    decoded_dim, dim_metadata = read_exact_video(directory / "analytic_background_dim.mp4")
    inputs = {"identity": (source, source, args.source_input.parent / "source.mp4"),
              "lossless_roundtrip": (native_gray, roundtrip, directory / "roundtrip_lossless.mp4"),
              "analytic_background_dim": (native_dim, decoded_dim, directory / "analytic_background_dim.mp4"),
              "analytic_background_gray": (native_gray, roundtrip, directory / "roundtrip_lossless.mp4")}
    negatives = {}
    for name, (native, actual, video_path) in inputs.items():
        proof = pixel_certificate(source, native, masks, actual, names)
        negatives[name] = {"passed": proof["geometry_passed"], "per_frame": proof["per_frame"],
                           "decoded_video_checked": True, "video": str(video_path), "video_sha256": sha256(video_path)}
    pixel_changed = source.copy()
    for frame in range(93):
        ys, xs = np.nonzero(masks[frame, 0])
        if not len(xs):
            raise ValueError("预声明单像素正控要求每帧机器人可见")
        pixel_changed[frame, ys[0], xs[0], 0] ^= 1
    shifted_book = np.asarray([object_shift(frame, mask.astype(bool), 12)[0]
                               for frame, mask in zip(source, masks[:, names.index("black_book_1")])])
    shifted_robot = np.asarray([object_shift(frame, mask.astype(bool), 12)[0]
                                for frame, mask in zip(source, masks[:, 0])])
    occluded = np.asarray([box_occlusion(frame, mask.astype(bool))[0] for frame, mask in zip(source, masks[:, 0])])
    reordered = gray.copy(); reordered[[0, 92]] = reordered[[92, 0]]
    positives = {}
    cases = {"protected_pixel_changed": (source, pixel_changed, "robot"),
             "book_shift12": (source, shifted_book, "black_book_1"),
             "robot_shift12": (source, shifted_robot, "robot"),
             "global_shift24": (source, np.roll(source, 24, axis=2), "robot"),
             "robot_box_occluded": (source, occluded, "robot"),
             "swap_first_last_frames": (native_gray, reordered, None)}
    for name, (native, actual, target) in cases.items():
        video_path = directory / (name + ".mp4")
        write_video(video_path, actual)
        decoded_actual, metadata = read_exact_video(video_path)
        proof = pixel_certificate(source, native, masks, decoded_actual, names)
        detected = [row["frame"] for row in proof["per_frame"]
                    if row["roi_max_abs_diff"][target] > 0] if target else []
        positives[name] = {"detected": not proof["geometry_passed"] and (target is None or detected == list(range(93))),
                           "target_roi": target, "target_error_frames": detected, "per_frame": proof["per_frame"],
                           "decoded_video_checked": True, "video_sha256": sha256(video_path), "video_metadata": metadata}
    for name, frames, fps in [("wrong_fps", gray, 24), ("missing_frame", gray[:-1], 20),
                              ("wrong_size", np.asarray([cv2.resize(f, (128, 128)) for f in gray]), 20)]:
        video = directory / (name + ".mp4")
        write_video(video, frames, fps)
        try:
            read_exact_video(video)
        except ValueError as exc:
            positives[name] = {"detected": True, "error": str(exc), "video_sha256": sha256(video)}
        else:
            positives[name] = {"detected": False, "video_sha256": sha256(video)}
    with np.load(args.source_input.parent / "source_labels.npz", allow_pickle=False) as archive:
        expected = {key: archive[key] for key in archive.files}
    for key in ["action", "state", "frame_index", "timestamp"]:
        altered = {name: value.copy() for name, value in expected.items()}
        altered[key].flat[0] += 1
        np.savez_compressed(directory / ("wrong_" + key + ".npz"), **altered)
        try:
            with np.load(directory / ("wrong_" + key + ".npz"), allow_pickle=False) as archive:
                validate_labels(dict(archive), expected)
        except ValueError as exc:
            positives["wrong_" + key] = {"detected": True, "error": str(exc)}
        else:
            positives["wrong_" + key] = {"detected": False}
    bad_masks = masks.copy(); bad_masks[0, 0, 0, 0] ^= 1
    np.save(directory / "wrong_roi.npy", bad_masks, allow_pickle=False)
    # 使用正式核验函数实际执行拒绝；仅记录两个摘要不相等不能证明门禁会拒绝。
    bindings = [
        ("wrong_roi_sha", directory / "wrong_roi.npy", evidence["files_sha256"]["roi_masks.npy"]),
        ("wrong_source_binding", directory / "roundtrip_lossless.mp4", evidence["files_sha256"]["source.mp4"]),
        ("wrong_native_binding", directory / "analytic_background_dim.mp4", sha256(directory / "roundtrip_lossless.mp4")),
    ]
    for name, path, checksum in bindings:
        try:
            checked_bound_file(path, checksum, directory)
        except ValueError as exc:
            positives[name] = {"detected": True, "error": str(exc), "actual_file_sha256": sha256(path),
                               "expected_file_sha256": checksum, "validator": "checked_bound_file"}
        else:
            positives[name] = {"detected": False}
    np.savez_compressed(directory / "actual_control_inputs.npz", source=source, masks=masks,
                        native_dim=native_dim, native_gray=native_gray, dim=dim, gray=gray, roundtrip=roundtrip,
                        pixel_changed=pixel_changed, shifted_book=shifted_book, shifted_robot=shifted_robot,
                        occluded=occluded, reordered=reordered)
    for frame in [0, 46, 92]:
        panels = [source[frame], dim[frame], gray[frame], shifted_book[frame], occluded[frame]]
        sheet = Image.new("RGB", (256 * len(panels), 290), "white"); draw = ImageDraw.Draw(sheet)
        for slot, (label, panel) in enumerate(zip(["source", "dim BG", "gray BG", "book +12px", "robot occluded"], panels)):
            draw.text((slot * 256 + 3, 5), f"frame {frame}: {label}", fill="black")
            sheet.paste(Image.fromarray(panel), (slot * 256, 30))
        sheet.save(directory / f"lossless_controls_{frame:03d}.png")
    result = {"method": config["version"], "method_config_sha256": sha256(args.config), "role": args.role,
              "source_input": str(args.source_input), "source_input_sha256": sha256(args.source_input),
              "source_episode_id": evidence["source_episode_id"], "source_proof": source_proof,
              "frames": list(range(93)), "negative_controls": negatives, "positive_controls": positives,
              "validation_passed": all(r["passed"] for r in negatives.values()) and all(r["detected"] for r in positives.values()),
              "design_binding": binding, "roundtrip_video": roundtrip_metadata,
              "precision": "CPU uint8 RGB；整数像素差；libx264rgb CRF0；非模型推理", "batch": None, "seed": None,
              "steps": None, "sample_count": 1, "eligible_for_training": False, "original_sam_gate_modified": False,
              "generation_performed": False, "training_performed": False, "promotion_performed": False,
              "limitation": "解析控制验证像素证据的方法，不是Cosmos生成或策略收益；仍须单独验收真实派生视频和全部复核"}
    (directory / "lossless_controls.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return {key: result[key] for key in ["method", "role", "source_episode_id", "validation_passed", "eligible_for_training"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-input", type=pathlib.Path, required=True)
    parser.add_argument("--role", choices=["design", "heldout"], required=True)
    parser.add_argument("--design-report", type=pathlib.Path)
    parser.add_argument("--config", type=pathlib.Path, default=ROOT / "configs/lossless_background_gate.yaml")
    args = parser.parse_args()
    for field in ["source_input", "design_report", "config"]:
        value = getattr(args, field)
        if value is not None:
            value = value.resolve(strict=True)
            if not value.is_relative_to(ROOT):
                parser.error("诊断输入必须位于Thor本项目目录")
            setattr(args, field, value)
    raise SystemExit(run_stage("lossless_background_controls", lambda directory: controls(directory, args),
                               {"compute": "CPU_only", "role": args.role, "diagnostic_only": True}))
