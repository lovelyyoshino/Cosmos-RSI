#!/usr/bin/env python3
"""固定设计与独立源的真实视频错误控制，验证所有非桌体像素都受保护。

@author pony
@date 2026-11-08
@version v2.1.0
@last_modified 2026-11-08
@changelog
    - v2.1.0 (2026-11-08): 实际检查旧门不能授予派生资格，并统一检查全部判决/控制依赖。
    - v2.0.0 (2026-11-08): 增加杯子改色和桌面掩码篡改控制，身份/时序/标签仍逐文件验证。
"""
from __future__ import annotations

import argparse
import json
import pathlib

from p0_background_preview_materials import checked_artifact
from p0_common import ROOT, ensure_gpu_idle, run_stage
from p0_context_normalized_segmentation import box_occlusion, object_shift
from p0_generated_window_gate import legacy_gate_decision
from p0_lossless_background import checked_bound_file, read_exact_video, validate_labels
from p0_policy_runtime import sha256
from p0_prepare_table_background import write_video
from p0_segmentation_controls import check_independent_source
from p0_table_background import (
    CONFIG, CONFIG_PATH, checked_config, checked_dependencies, checked_table_source, compose_table, pixel_certificate,
)


NEGATIVES = {"identity", "lossless_roundtrip", "analytic_table_dim", "analytic_table_gray"}
ERRORS = {"protected_pixel_changed", "book_shift12", "robot_shift12", "mug_recolored", "global_shift24",
          "robot_box_occluded", "swap_first_last_frames", "wrong_fps", "missing_frame", "wrong_size",
          "wrong_action", "wrong_state", "wrong_frame_index", "wrong_timestamp", "wrong_roi_sha",
          "wrong_editable_sha", "wrong_source_binding", "wrong_native_binding"}
ERRORS.add("legacy_eligibility_bypass")


def validate_report(report: dict, role: str) -> None:
    if report.get("method") != CONFIG["version"] or report.get("role") != role \
            or report.get("validation_passed") is not True or report.get("eligible_for_training") is not False \
            or report.get("frames") != list(range(93)) or set(report.get("negative_controls", {})) != NEGATIVES \
            or set(report.get("positive_controls", {})) != ERRORS \
            or any(row.get("passed") is not True or row.get("decoded_video_checked") is not True
                   for row in report["negative_controls"].values()) \
            or any(row.get("detected") is not True for row in report["positive_controls"].values()):
        raise ValueError("桌面方法的完整实际视频负控或错误控制未通过，不能据此启用新门")


def checked_design(path: pathlib.Path) -> tuple[dict, dict]:
    probe = checked_artifact(path, "table_background_controls")
    report = json.loads(path.read_text())
    validate_report(report, "design")
    if report.get("method_config_sha256") != sha256(CONFIG_PATH):
        raise ValueError("设计源之后方法改变，禁止准备独立源或借用设计结果")
    checked_dependencies(probe)
    source_path = checked_bound_file(pathlib.Path(report["source_input"]), report["source_input_sha256"], ROOT / "runs/p0")
    evidence, *_ = checked_table_source(source_path)
    return report, evidence


def controls(directory: pathlib.Path, args: argparse.Namespace) -> dict:
    ensure_gpu_idle()
    checked_config()
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw

    binding = None
    if args.role == "heldout":
        if args.design_report is None:
            raise ValueError("独立检查必须先绑定通过的完整设计源")
        _, design_source = checked_design(args.design_report)
    evidence, source, masks, editable, elements, proof = checked_table_source(args.source_input)
    if not evidence["source_episode_id"].endswith(CONFIG[args.role + "_episode_suffix"]):
        raise ValueError("控制源与预声明的设计/独立回合不符")
    if args.role == "heldout":
        binding = {"report": str(args.design_report), "report_sha256": sha256(args.design_report),
                   "source_independence": check_independent_source(evidence, design_source)}
    dim_native = (source.astype(np.float32) * .5).astype(np.uint8)
    gray_native = np.full_like(source, 128)
    dim, gray = (compose_table(source, native, editable) for native in [dim_native, gray_native])
    negatives = {}
    for name, native, actual in [("identity", source, source), ("lossless_roundtrip", gray_native, gray),
                                  ("analytic_table_dim", dim_native, dim), ("analytic_table_gray", gray_native, gray)]:
        video = directory / (name + ".mp4")
        write_video(video, actual)
        decoded, _ = read_exact_video(video)
        certificate = pixel_certificate(source, native, editable, decoded, masks)
        negatives[name] = {"passed": certificate["geometry_passed"], "decoded_video_checked": True,
                           "video_sha256": sha256(video), "per_frame": certificate["per_frame"]}
    changed = source.copy()
    for frame in range(93):
        y, x = np.argwhere(~editable[frame])[0]
        changed[frame, y, x, 0] ^= 1
    mug_ids = [row["id"] for row in evidence["table_geometry"]["geometries"]
               if any(word in str(row.get("instance", "")).lower() for word in ["mug", "cup"])]
    mug_mask = np.isin(elements, mug_ids)
    visible_mug_frames = np.flatnonzero(mug_mask.any(axis=(1, 2))).tolist()
    if not visible_mug_frames or np.any(mug_mask & editable):
        raise ValueError("杯子错误控制需要实际可见、完整受保护的非任务杯子")
    mug_changed = source.copy()
    mug_changed[mug_mask, 1] ^= 64
    swapped = gray.copy()
    swapped[[0, 92]] = swapped[[92, 0]]
    cases = {
        "protected_pixel_changed": (source, changed),
        "book_shift12": (source, np.asarray([object_shift(f, m.astype(bool), 12)[0] for f, m in zip(source, masks[:, 1])])),
        "robot_shift12": (source, np.asarray([object_shift(f, m.astype(bool), 12)[0] for f, m in zip(source, masks[:, 0])])),
        "mug_recolored": (source, mug_changed), "global_shift24": (source, np.roll(source, 24, axis=2)),
        "robot_box_occluded": (source, np.asarray([box_occlusion(f, m.astype(bool))[0] for f, m in zip(source, masks[:, 0])])),
        "swap_first_last_frames": (gray_native, swapped),
    }
    positives = {}
    positives["legacy_eligibility_bypass"] = {
        "detected": legacy_gate_decision("table_background_composite", [])["eligible_for_training"] is False
                    and legacy_gate_decision("background_composite", [])["eligible_for_training"] is False
                    and legacy_gate_decision("transfer_smoke", [])["eligible_for_training"] is True,
        "validator": "legacy_gate_decision", "legacy_measurements_can_pass": True,
        "limitation": "实际资格决策函数检查，不是新的模型或图像样本"}
    for name, (native, actual) in cases.items():
        path = directory / (name + ".mp4")
        write_video(path, actual)
        decoded, _ = read_exact_video(path)
        certificate = pixel_certificate(source, native, editable, decoded, masks)
        error_frames = [r["frame"] for r in certificate["per_frame"] if not r["expected_composite_equal"]]
        required = [0, 92] if name == "swap_first_last_frames" else visible_mug_frames if name == "mug_recolored" else list(range(93))
        positives[name] = {"detected": not certificate["geometry_passed"] and set(required) <= set(error_frames),
                           "required_error_frames": required, "error_frames": error_frames,
                           "decoded_video_checked": True, "video_sha256": sha256(path), "per_frame": certificate["per_frame"]}
    for name, frames, fps in [("wrong_fps", gray, 24), ("missing_frame", gray[:-1], 20),
                              ("wrong_size", np.asarray([cv2.resize(f, (128, 128)) for f in gray]), 20)]:
        path = directory / (name + ".mp4")
        import imageio.v2 as imageio
        with imageio.get_writer(path, fps=fps, codec="libx264rgb", pixelformat="rgb24", macro_block_size=1,
                               ffmpeg_params=["-crf", "0", "-threads", "2"]) as writer:
            for frame in frames:
                writer.append_data(frame)
        try:
            read_exact_video(path)
        except ValueError as exc:
            positives[name] = {"detected": True, "error": str(exc), "actual_video_sha256": sha256(path)}
        else:
            positives[name] = {"detected": False}
    with np.load(args.source_input.parent / "source_labels.npz", allow_pickle=False) as archive:
        expected = dict(archive)
    for key in ["action", "state", "frame_index", "timestamp"]:
        altered = {name: value.copy() for name, value in expected.items()}
        altered[key].flat[0] += 1
        path = directory / ("wrong_" + key + ".npz")
        np.savez_compressed(path, **altered)
        try:
            with np.load(path, allow_pickle=False) as archive:
                validate_labels(dict(archive), expected)
        except ValueError as exc:
            positives["wrong_" + key] = {"detected": True, "error": str(exc)}
        else:
            positives["wrong_" + key] = {"detected": False}
    altered_roi = masks.copy()
    altered_roi[0, 0, 0, 0] ^= 1
    altered_editable = editable.copy()
    altered_editable[0, 0, 0] ^= True
    np.save(directory / "wrong_roi.npy", altered_roi, allow_pickle=False)
    np.save(directory / "wrong_editable.npy", altered_editable, allow_pickle=False)
    for name, path, expected_sha in [
        ("wrong_roi_sha", directory / "wrong_roi.npy", evidence["files_sha256"]["roi_masks.npy"]),
        ("wrong_editable_sha", directory / "wrong_editable.npy", evidence["files_sha256"]["editable_table.npy"]),
        ("wrong_source_binding", directory / "analytic_table_dim.mp4", evidence["files_sha256"]["source.mp4"]),
        ("wrong_native_binding", directory / "analytic_table_dim.mp4", sha256(directory / "analytic_table_gray.mp4")),
    ]:
        try:
            checked_bound_file(path, expected_sha, directory)
        except ValueError as exc:
            positives[name] = {"detected": True, "error": str(exc), "validator": "checked_bound_file"}
        else:
            positives[name] = {"detected": False}
    for index in [0, 46, 92]:
        sheet = Image.new("RGB", (1280, 282), "white")
        draw = ImageDraw.Draw(sheet)
        for slot, (label, panel) in enumerate(zip(["source", "editable table", "dim table", "gray table", "mug recolored"],
                [source[index], np.repeat(editable[index, ..., None], 3, axis=-1).astype(np.uint8) * 255,
                 dim[index], gray[index], mug_changed[index]])):
            draw.text((slot * 256 + 2, 2), f"{index}: {label}", fill="black")
            sheet.paste(Image.fromarray(panel), (slot * 256, 24))
        sheet.save(directory / f"table_controls_{index:03d}.png")
    result = {"method": CONFIG["version"], "method_config_sha256": sha256(CONFIG_PATH), "role": args.role,
        "source_input": str(args.source_input), "source_input_sha256": sha256(args.source_input),
        "source_episode_id": evidence["source_episode_id"], "source_proof": proof, "frames": list(range(93)),
        "negative_controls": negatives, "positive_controls": positives, "design_binding": binding,
        "validation_passed": all(r["passed"] for r in negatives.values()) and all(r["detected"] for r in positives.values()),
        "eligible_for_training": False, "precision": "Thor CPU uint8 RGB/整数差分/CRF0实际编码解码",
        "batch": None, "seed": None, "steps": None, "sample_count": 1,
        "generation_performed": False, "training_performed": False, "promotion_performed": False,
        "limitation": "仅方法控制，不是Cosmos生成样本、独立策略评测或RSI收益"}
    (directory / "table_controls.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return {key: result[key] for key in ["method", "role", "validation_passed", "eligible_for_training", "source_episode_id"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-input", type=pathlib.Path, required=True)
    parser.add_argument("--role", choices=["design", "heldout"], required=True)
    parser.add_argument("--design-report", type=pathlib.Path)
    args = parser.parse_args()
    for name in ["source_input", "design_report"]:
        path = getattr(args, name)
        if path is not None:
            path = path.resolve(strict=True)
            if not path.is_relative_to(ROOT / "runs/p0"):
                parser.error("全部诊断输入必须在本项目P0目录")
            setattr(args, name, path)
    raise SystemExit(run_stage("table_background_controls", lambda directory: controls(directory, args),
                               {"role": args.role, "method": CONFIG["version"], "diagnostic_only": True}))
