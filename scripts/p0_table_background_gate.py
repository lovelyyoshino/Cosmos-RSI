#!/usr/bin/env python3
"""合成并完整验收仅桌面派生，保留原SAM2测量与全部非几何拒绝条件。

@author pony
@date 2026-11-08
@version v2.1.1
@last_modified 2026-11-08
@changelog
    - v2.1.1 (2026-11-08): 扫描全部同视频历史报告，部分帧明确失败同样保持拒绝。
    - v2.1.0 (2026-11-08): 同视频明确语义/目标/时序失败保持拒绝，原SAM2只作为独立测量。
    - v2.0.0 (2026-11-08): 原生来源→完整非桌体无损合成→独立方法/像素/语义/目标/时序门。
"""
from __future__ import annotations

import argparse
import json
import pathlib

from p0_common import ROOT, ensure_gpu_idle, run_stage
from p0_background_preview_materials import checked_artifact
from p0_generated_window_gate import review_bound_to_video, target_condition_review_passed, verified_generation_paths
from p0_lossless_background import read_exact_video
from p0_lossless_background_gate import acceptance_reasons, historical_gate, roi_temporal_residual
from p0_policy_runtime import sha256
from p0_prepare_table_background import write_video
from p0_table_background import (
    CONFIG, RECIPE, checked_config, checked_method, checked_table_source, compose_table,
    pixel_certificate, verified_composite_paths,
)


def native_background(probe_path: pathlib.Path):
    import cv2
    import numpy as np

    generation = json.loads(probe_path.read_text())
    if generation.get("stage") != "transfer_smoke":
        raise ValueError("桌面派生仅接受原生Cosmos输出，不递归合成")
    source, video = verified_generation_paths(generation, probe_path)
    result = generation["result"]
    if result.get("source_spec") != str(source.parent / "spec.json") \
            or any(result.get(key) != value for key, value in
                   {"num_steps": 4, "seed": 42, "batch": 1, "resolution_requested": "480", "recipe": "edge/distilled"}.items()):
        raise ValueError("实际Cosmos运行改变了预声明四步/seed42/480/单样本配方")
    native, metadata = read_exact_video(video, size=None)
    if min(metadata["width"], metadata["height"]) < 256:
        raise ValueError("原生背景不能通过上采样替代")
    resized = np.asarray([cv2.resize(frame, (256, 256), interpolation=cv2.INTER_AREA) for frame in native])
    return generation, source, video, resized


def composite(directory: pathlib.Path, args: argparse.Namespace) -> dict:
    ensure_gpu_idle()
    checked_config()
    generation, source_path, native_path, background = native_background(args.probe)
    evidence, source, masks, editable, _, _ = checked_table_source(source_path)
    output = compose_table(source, background, editable)
    video = directory / "table_background_lossless.mp4"
    write_video(video, output)
    decoded, _ = read_exact_video(video)
    certificate = pixel_certificate(source, background, editable, decoded, masks)
    if not certificate["geometry_passed"]:
        raise ValueError("实际编码改变了预声明桌面合成关系")
    result = {"derived_kind": RECIPE["kind"], "composite_recipe": RECIPE,
        "native_generation_probe": str(args.probe), "native_generation_probe_sha256": sha256(args.probe),
        "native_video": str(native_path), "native_video_sha256": sha256(native_path),
        "source_success_evidence": generation["result"]["source_success_evidence"],
        "source_labels_sha256": evidence["labels_sha256"], "outputs": [str(video)],
        "output_video_sha256": sha256(video), "guardrails_enabled": True,
        "guardrails_scope": "原生Cosmos文本/视频安全检查；派生只回填成功训练源的不可改写区域",
        "target_condition": CONFIG["target_condition"], "target_condition_scope": evidence["target_condition_scope"],
        "lossless_pixels_verified": True, "pixel_certificate": certificate, "eligible_for_training": False,
        "precision": "Thor CPU uint8 RGB/INTER_AREA下采样/CRF0无损编码解码", "batch": None, "seed": None,
        "steps": None, "sample_count": 1, "frames": 93, "fps": 20,
        "generation_performed": False, "training_performed": False, "promotion_performed": False,
        "limitation": "只完成合成与像素核验，完整方法、语义、目标、时序仍待独立验收"}
    (directory / "composite.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def explicit_review_failures(reviews: dict, generation_sha: str, video_sha: str) -> list[str]:
    """缺少复核可以后补；同一实际视频已明确失败的复核不能改名为通过。"""
    def bound_failure(review):
        if not isinstance(review, dict) or review.get("source_probe_sha256") != generation_sha \
                or review.get("output_video_sha256") != video_sha \
                or not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip():
            return False
        frames = review.get("reviewed_frames")
        # 放行须覆盖完整窗口；拒绝只需在有效帧子集中找到一个明确错误。
        return bool(isinstance(frames, list) and frames
                    and all(type(frame) is int and 0 <= frame < 93 for frame in frames)
                    and len(set(frames)) == len(frames))

    failed = []
    semantic = reviews.get("semantic_review")
    if bound_failure(semantic) and (
            semantic.get("passed") is False or semantic.get("object_identity_preserved") is False
            or semantic.get("task_semantics_preserved") is False
            or semantic.get("unwanted_objects") is True or semantic.get("no_new_obstacles") is False):
        failed.append("semantic_review")
    condition = reviews.get("condition_review")
    regions = condition.get("region_checks") if isinstance(condition, dict) else None
    if bound_failure(condition) and (
            condition.get("passed") is False or any(isinstance(check, dict) and check.get("passed") is False
                                                    for check in (regions.values() if isinstance(regions, dict) else []))):
        failed.append("condition_review")
    temporal = reviews.get("temporal_review")
    if bound_failure(temporal) and any(
            temporal.get(key) is False for key in ["passed", "background_consistent", "no_flicker", "boundary_seams_acceptable"]):
        failed.append("temporal_review")
    return failed


def prior_rejections(sam_path: pathlib.Path, generation_sha: str, video_sha: str) -> list[dict]:
    candidates = {sam_path.resolve(): "generated_window_gate"}
    for stage in ["generated_window_gate", "lossless_background_gate", "table_background_gate"]:
        for path in (ROOT / "runs/p0").glob(f"{stage}_*/quality_gate.json"):
            candidates[path.resolve()] = stage
    rejections = []
    for path, stage in sorted(candidates.items()):
        report = json.loads(path.read_text())
        if report.get("generation_probe_sha256") != generation_sha or report.get("output_video_sha256") != video_sha:
            continue
        checked_artifact(path, stage)
        reviews = report.get("reviews", {}) if stage != "generated_window_gate" else {
            "semantic_review": report.get("semantic_review"), "condition_review": report.get("target_condition_review")}
        failures = explicit_review_failures(reviews, generation_sha, video_sha)
        if failures:
            rejections.append({"path": str(path), "sha256": sha256(path), "failed_reviews": failures})
    return rejections


def gate(directory: pathlib.Path, args: argparse.Namespace) -> dict:
    ensure_gpu_idle()
    checked_config()
    import numpy as np
    from PIL import Image, ImageDraw

    if any(getattr(args, key) is None for key in ["design_report", "heldout_report", "native_sam_gate", "composite_sam_gate"]):
        raise ValueError("完整验收必须具备设计/独立方法控制及原生/派生历史SAM2报告")
    method = checked_method(args.design_report, args.heldout_report)
    record = json.loads(args.probe.read_text())
    source_path, output_path = verified_composite_paths(record, args.probe, ROOT)
    evidence, source, masks, editable, _, source_proof = checked_table_source(source_path)
    native_probe = pathlib.Path(record["result"]["native_generation_probe"])
    _, native_source, native_path, background = native_background(native_probe)
    if native_source != source_path:
        raise ValueError("新门、派生和原生来源必须是同一成功窗口")
    output, metadata = read_exact_video(output_path)
    certificate = pixel_certificate(source, background, editable, output, masks)
    history = {"native": historical_gate(args.native_sam_gate, native_probe, native_path),
               "composite": historical_gate(args.composite_sam_gate, args.probe, output_path)}
    generation_sha, video_sha = sha256(args.probe), sha256(output_path)
    reviews = {key: json.loads(path.read_text()) if (path := getattr(args, key)) else None
               for key in ["semantic_review", "condition_review", "temporal_review"]}
    semantic = reviews["semantic_review"]
    semantic_ok = bool(review_bound_to_video(semantic, generation_sha, video_sha, 93)
        and semantic.get("object_identity_preserved") is True and semantic.get("task_semantics_preserved") is True
        and semantic.get("unwanted_objects") is False and semantic.get("no_new_obstacles") is True
        and isinstance(semantic.get("observations"), str) and semantic["observations"].strip())
    condition_ok = target_condition_review_passed(reviews["condition_review"], CONFIG["target_condition"], generation_sha, video_sha, 93)
    temporal = reviews["temporal_review"]
    temporal_ok = bool(review_bound_to_video(temporal, generation_sha, video_sha, 93) and temporal.get("passed") is True
        and temporal.get("background_consistent") is True and temporal.get("no_flicker") is True
        and temporal.get("boundary_seams_acceptable") is True
        and isinstance(temporal.get("observations"), str) and temporal["observations"].strip())
    residuals = roi_temporal_residual(source, output, masks)
    p95 = float(np.percentile(residuals, 95))
    previous_condition = history["composite"].get("target_condition_review")
    prior_reject = bool(review_bound_to_video(previous_condition, generation_sha, video_sha, 93)
                        and previous_condition.get("passed") is False)
    reasons = acceptance_reasons(geometry_ok=certificate["geometry_passed"], method_ok=method["passed"],
        roi_temporal_ok=p95 <= CONFIG["temporal_residual_p95_max"], semantic_ok=semantic_ok,
        condition_ok=condition_ok, background_temporal_ok=temporal_ok, prior_target_rejected=prior_reject)
    prior_failures = prior_rejections(args.composite_sam_gate, generation_sha, video_sha)
    if prior_failures:
        reasons.append("同一实际派生视频已有明确语义、目标或背景时序失败，历史拒绝不能覆盖")
    for start in range(0, 93, 12):
        sheet = Image.new("RGB", (1024, 1680), "white")
        draw = ImageDraw.Draw(sheet)
        for slot, index in enumerate(range(start, min(start + 12, 93))):
            x, y = slot % 2 * 512, slot // 2 * 280
            draw.text((x + 2, y + 2), f"Frame {index}: source | table-only composite", fill="black")
            sheet.paste(Image.fromarray(source[index]), (x, y + 22))
            sheet.paste(Image.fromarray(output[index]), (x + 256, y + 22))
        sheet.save(directory / f"contact_sheet_{start:03d}.png")
    result = {"method": CONFIG["version"], "geometry_evidence_type": CONFIG["geometry_evidence_type"],
        "output_kind": "background_composite", "derived_kind": RECIPE["kind"], "composite_recipe": RECIPE,
        "generation_probe": str(args.probe), "generation_probe_sha256": generation_sha,
        "output_video": str(output_path), "output_video_sha256": video_sha, "output_video_metadata": metadata,
        "native_generation_probe": str(native_probe), "native_generation_probe_sha256": sha256(native_probe),
        "native_video_sha256": sha256(native_path), "source_input": str(source_path),
        "source_input_sha256": sha256(source_path), "source_proof": source_proof,
        "source_labels_sha256": evidence["labels_sha256"], "label_binding_mode": "unchanged_source_reference",
        "method_validation": method, "historical_sam_gate": history, "geometry_passed": certificate["geometry_passed"],
        "pixel_certificate": certificate, "temporal_residual_p95": p95, "temporal_residual_per_frame": residuals,
        "semantic_review_passed": semantic_ok, "target_condition": CONFIG["target_condition"],
        "target_condition_scope": evidence["target_condition_scope"], "target_condition_passed": condition_ok,
        "background_temporal_passed": temporal_ok, "reviews": reviews,
        "review_files_sha256": {key: sha256(path) if (path := getattr(args, key)) else None for key in reviews},
        "prior_target_rejected": prior_reject, "prior_non_geometry_rejections": prior_failures,
        "passed": not reasons, "eligible_for_training": not reasons,
        "reject_reasons": reasons, "precision": "Thor CPU uint8 RGB/整数差分；ROI时序fp32；无模型推理",
        "batch": None, "seed": None, "steps": None, "sample_count": 1, "frames": list(range(93)),
        "generation_performed": False, "training_performed": False, "promotion_performed": False,
        "limitation": "仅桌面派生资格；原SAM2报告不变；不是光照增强、候选/对照或多圈RSI收益"}
    (directory / "quality_gate.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return {key: result[key] for key in ["method", "geometry_passed", "semantic_review_passed", "target_condition_passed",
                                        "background_temporal_passed", "eligible_for_training", "reject_reasons"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=["composite", "gate"])
    parser.add_argument("--probe", type=pathlib.Path, required=True)
    for option in ["design-report", "heldout-report", "native-sam-gate", "composite-sam-gate",
                   "semantic-review", "condition-review", "temporal-review"]:
        parser.add_argument("--" + option, type=pathlib.Path)
    args = parser.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, pathlib.Path):
            value = value.resolve(strict=True)
            if not value.is_relative_to(ROOT / "runs/p0"):
                parser.error("全部合成/验收输入必须在本项目P0目录")
            setattr(args, key, value)
    callback, stage = (composite, "table_background_composite") if args.action == "composite" else (gate, "table_background_gate")
    raise SystemExit(run_stage(stage, lambda directory: callback(directory, args),
        {"method": CONFIG["version"], **{key: str(value) if value is not None else None for key, value in vars(args).items()}}))
