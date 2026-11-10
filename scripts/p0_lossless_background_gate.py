#!/usr/bin/env python3
"""独立验收预声明的仿真无损背景派生，保留原SAM2拒绝和全部非几何门。

@author pony
@date 2026-11-08
@version v1.0.1
@last_modified 2026-11-08
@changelog
    - v1.0.1 (2026-11-08): 拒绝说明区分复核缺失与复核明确失败，不把已有失败复核误写为缺少。
    - v1.0.0 (2026-11-08): 绑定设计/独立验证、实际视频、源标签及完整语义/目标/时序复核。
"""
from __future__ import annotations

import argparse
import json
import pathlib

from p0_background_preview_materials import checked_artifact
from p0_common import ROOT, ensure_gpu_idle, run_stage
from p0_generated_window_gate import review_bound_to_video, target_condition_review_passed, verified_generation_paths
from p0_lossless_background import (
    checked_bound_file, checked_config, checked_method_validation, checked_training_source,
    pixel_certificate, read_exact_video,
)
from p0_policy_runtime import sha256
from p0_prepare_background_transfer import COMPOSITE_RECIPE


def load_composite(probe_path: pathlib.Path):
    """只接受单级预声明派生；原生输出不能借用新几何证据。"""
    from p0_background_composite import check_declaration

    probe = json.loads(probe_path.read_text())
    result = probe.get("result", {})
    if probe.get("stage") != "background_composite" or probe.get("status") != "passed" \
            or result.get("derived_kind") != COMPOSITE_RECIPE["kind"] \
            or result.get("composite_recipe") != COMPOSITE_RECIPE or len(result.get("outputs", [])) != 1:
        raise ValueError("新路线仅接受预声明的单级背景无损派生")
    source_record = result.get("source_success_evidence", {})
    source_path = checked_bound_file(pathlib.Path(source_record["path"]), source_record["sha256"], ROOT / "runs/p0")
    evidence, source, masks, source_proof = checked_training_source(source_path)
    check_declaration(evidence)
    if result.get("target_condition") != evidence["target_condition"] \
            or result.get("source_labels_sha256") != evidence["labels_sha256"] \
            or result.get("lossless_pixels_verified") is not True:
        raise ValueError("派生声明没有保持同一背景目标和源标签")
    preparation = json.loads((source_path.parent / "probe.json").read_text())
    if preparation.get("stage") != "background_transfer_input":
        raise ValueError("背景目标未在生成前绑定成功源")
    spec_path = checked_bound_file(pathlib.Path(evidence["spec"]),
                                  preparation.get("artifacts_sha256", {}).get("spec.json"), source_path.parent,
                                  source_path.parent / "spec.json")
    spec = json.loads(spec_path.read_text())
    checked_bound_file(pathlib.Path(spec["video_path"]), evidence["files_sha256"]["source.mp4"],
                       source_path.parent, source_path.parent / "source.mp4")
    native_probe = checked_bound_file(pathlib.Path(result["native_generation_probe"]),
                                     result["native_generation_probe_sha256"], ROOT / "runs/p0")
    native = json.loads(native_probe.read_text())
    if native.get("stage") != "transfer_smoke" or native.get("result", {}).get("source_spec") != str(spec_path):
        raise ValueError("派生没有绑定同一源配方的实际原生生成")
    native_source, native_video = verified_generation_paths(native, native_probe)
    if native_source != source_path or result.get("guardrails_enabled") is not True:
        raise ValueError("派生和原生成功源不一致或安全检查未启用")
    checked_bound_file(pathlib.Path(result["native_video"]), result["native_video_sha256"], native_probe.parent, native_video)
    output = pathlib.Path(result["outputs"][0]).resolve(strict=True)
    if not output.is_relative_to(probe_path.parent):
        raise ValueError("派生输出越过本次运行目录")
    relative = str(output.relative_to(probe_path.parent))
    checked_bound_file(output, probe.get("artifacts_sha256", {}).get(relative), probe_path.parent)
    checked_bound_file(output, result.get("output_video_sha256"), probe_path.parent)
    return source_path, evidence, source, masks, source_proof, native_probe, native_video, output


def historical_gate(path: pathlib.Path, generation_probe: pathlib.Path, video: pathlib.Path) -> dict:
    checked_artifact(path, "generated_window_gate")
    gate = json.loads(path.read_text())
    if gate.get("generation_probe_sha256") != sha256(generation_probe) \
            or gate.get("output_video_sha256") != sha256(video) \
            or gate.get("gate_config_sha256") != sha256(ROOT / "configs/gen_gate.yaml") \
            or gate.get("segmented_frames") != list(range(93)):
        raise ValueError("原SAM2历史报告没有绑定同一实际视频和全部帧")
    return {"path": str(path), "sha256": sha256(path), "geometry_passed": gate["geometry_passed"],
            "passed": gate["passed"], "eligible_for_training": gate["eligible_for_training"],
            "target_condition_review": gate.get("target_condition_review"),
            "reject_reasons": gate["reject_reasons"], "original_report_modified": False}


def roi_temporal_residual(source, output, masks) -> list[float]:
    import cv2
    import numpy as np

    residuals = []
    for index in range(1, 93):
        roi = masks[index].any(axis=0) | masks[index - 1].any(axis=0)
        delta = [cv2.cvtColor(video[index], cv2.COLOR_RGB2GRAY).astype(np.float32)
                 - cv2.cvtColor(video[index - 1], cv2.COLOR_RGB2GRAY).astype(np.float32)
                 for video in [source, output]]
        residuals.append(float(np.abs(delta[1] - delta[0])[roi].mean() / 255) if roi.any() else 1.0)
    return residuals


def acceptance_reasons(*, geometry_ok: bool, method_ok: bool, roi_temporal_ok: bool, semantic_ok: bool,
                       condition_ok: bool, background_temporal_ok: bool, prior_target_rejected: bool) -> list[str]:
    checks = [(geometry_ok, "完整像素/视频时间证明未通过"), (method_ok, "设计或独立方法验证未通过"),
              (roi_temporal_ok, "原ROI时序残差超出阈值"), (semantic_ok, "完整物体身份/任务语义/无新增障碍复核缺失或未通过"),
              (condition_ok, "预声明目标背景未通过完整分区域复核"),
              (background_temporal_ok, "背景时序与闪烁未通过完整复核"),
              (not prior_target_rejected, "同一视频的预声明背景目标已明确失败，历史样本继续拒绝")]
    return [reason for passed, reason in checks if not passed]


def gate(directory: pathlib.Path, args: argparse.Namespace) -> dict:
    ensure_gpu_idle()
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw

    config = checked_config(args.config)
    method = checked_method_validation(args.design_report, args.heldout_report, args.config)
    source_path, evidence, source, masks, source_proof, native_probe, native_path, output_path = load_composite(args.probe)
    native, native_metadata = read_exact_video(native_path, size=None)
    if min(native_metadata["width"], native_metadata["height"]) < 256:
        raise ValueError("原生背景必须按固定方法下采样，不能借用上采样结果")
    generated = np.asarray([cv2.resize(frame, (256, 256), interpolation=cv2.INTER_AREA) for frame in native])
    output, output_metadata = read_exact_video(output_path)
    pixel_proof = pixel_certificate(source, generated, masks, output, evidence["roi_names"])
    history = {"native": historical_gate(args.native_sam_gate, native_probe, native_path),
               "composite": historical_gate(args.composite_sam_gate, args.probe, output_path)}
    generation_sha, video_sha = sha256(args.probe), sha256(output_path)
    reviews = {key: json.loads(path.read_text()) if (path := getattr(args, key)) else None
               for key in ["semantic_review", "condition_review", "temporal_review"]}
    semantic = reviews["semantic_review"]
    semantic_ok = bool(review_bound_to_video(semantic, generation_sha, video_sha, 93)
                       and semantic.get("object_identity_preserved") is True
                       and semantic.get("task_semantics_preserved") is True
                       and semantic.get("unwanted_objects") is False and semantic.get("no_new_obstacles") is True
                       and isinstance(semantic.get("observations"), str) and semantic["observations"].strip())
    condition_ok = target_condition_review_passed(reviews["condition_review"], evidence["target_condition"],
                                                 generation_sha, video_sha, 93)
    temporal_review = reviews["temporal_review"]
    background_ok = bool(review_bound_to_video(temporal_review, generation_sha, video_sha, 93)
                         and temporal_review.get("passed") is True
                         and temporal_review.get("background_consistent") is True
                         and temporal_review.get("no_flicker") is True
                         and isinstance(temporal_review.get("observations"), str) and temporal_review["observations"].strip())
    residuals = roi_temporal_residual(source, output, masks)
    residual_p95 = float(np.percentile(residuals, 95))
    original_review = history["composite"]["target_condition_review"]
    prior_target_rejected = bool(review_bound_to_video(original_review, generation_sha, video_sha, 93)
                                 and original_review.get("passed") is False)
    reasons = acceptance_reasons(geometry_ok=pixel_proof["geometry_passed"], method_ok=method["passed"],
                                roi_temporal_ok=residual_p95 <= config["temporal_residual_p95_max"],
                                semantic_ok=semantic_ok, condition_ok=condition_ok,
                                background_temporal_ok=background_ok, prior_target_rejected=prior_target_rejected)
    for start in range(0, 93, 12):
        sheet = Image.new("RGB", (1024, 6 * 280), "white")
        draw = ImageDraw.Draw(sheet)
        for slot, index in enumerate(range(start, min(93, start + 12))):
            x, y = slot % 2 * 512, slot // 2 * 280
            draw.text((x + 3, y + 3), f"Frame {index}: source | lossless composite", fill="black")
            sheet.paste(Image.fromarray(source[index]), (x, y + 22))
            sheet.paste(Image.fromarray(output[index]), (x + 256, y + 22))
        sheet.save(directory / f"contact_sheet_{start:03d}.png")
    report = {"method": config["version"], "geometry_evidence_type": config["geometry_evidence_type"],
              "output_kind": "background_composite", "generation_probe": str(args.probe),
              "generation_probe_sha256": generation_sha, "output_video": str(output_path),
              "output_video_sha256": video_sha, "native_generation_probe": str(native_probe),
              "native_generation_probe_sha256": sha256(native_probe), "native_video_sha256": sha256(native_path),
              "source_input": str(source_path), "source_input_sha256": sha256(source_path), "source_proof": source_proof,
              "source_labels_sha256": evidence["labels_sha256"], "label_binding_mode": "unchanged_source_reference",
              "method_validation": method, "gate_config_sha256": sha256(args.config), "historical_sam_gate": history,
              "native_video_metadata": native_metadata, "output_video_metadata": output_metadata,
              "native_resize": config["native_resize"], "geometry_passed": pixel_proof["geometry_passed"],
              "pixel_certificate": pixel_proof, "temporal_residual_p95": residual_p95,
              "temporal_residual_per_frame": residuals, "temporal_residual_p95_max": config["temporal_residual_p95_max"],
              "semantic_review_passed": semantic_ok, "target_condition": evidence["target_condition"],
              "target_condition_passed": condition_ok, "background_temporal_passed": background_ok,
              "reviews": reviews, "review_files_sha256": {key: sha256(path) if (path := getattr(args, key)) else None
                                                           for key in reviews},
              "prior_target_rejected": prior_target_rejected, "passed": not reasons,
              "eligible_for_training": not reasons, "reject_reasons": reasons,
              "precision": "Thor CPU uint8 RGB；整数像素差；ROI时序fp32；无模型推理", "batch": None,
              "seed": None, "steps": None, "sample_count": 1, "frames": list(range(93)),
              "generation_performed": False, "training_performed": False, "promotion_performed": False,
              "limitation": "派生视频资格独立记录，旧SAM2报告不变；不是全场光照增强或策略收益；接入训练还须独立验证动作块边界"}
    (directory / "quality_gate.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return {key: report[key] for key in ["method", "output_kind", "geometry_passed", "semantic_review_passed",
                                        "target_condition_passed", "background_temporal_passed", "eligible_for_training",
                                        "reject_reasons", "precision", "batch", "seed", "sample_count"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ["probe", "design-report", "heldout-report", "native-sam-gate", "composite-sam-gate"]:
        parser.add_argument("--" + name, type=pathlib.Path, required=True)
    for name in ["semantic-review", "condition-review", "temporal-review"]:
        parser.add_argument("--" + name, type=pathlib.Path)
    parser.add_argument("--config", type=pathlib.Path, default=ROOT / "configs/lossless_background_gate.yaml")
    arguments = parser.parse_args()
    for key, value in vars(arguments).items():
        if value is not None:
            value = value.resolve(strict=True)
            allowed = ROOT / ("configs" if key == "config" else "runs/p0")
            if not value.is_relative_to(allowed):
                parser.error("所有验收输入必须位于本项目指定目录")
            setattr(arguments, key, value)
    raise SystemExit(run_stage("lossless_background_gate", lambda directory: gate(directory, arguments),
                               {"compute": "CPU_only", "method": "thor-lossless-background-v1",
                                **{key: str(value) if value else None for key, value in vars(arguments).items()}}))
