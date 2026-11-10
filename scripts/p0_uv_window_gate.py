#!/usr/bin/env python3
"""以完整双源控制、同帧像素和绑定实际视频的复核验收UV桌面窗口。

@author pony
@date 2026-11-08
@version v1.0.0
@last_modified 2026-11-08
@changelog
    - v1.0.0 (2026-11-08): 独立准入已固定的图集方法，保留原门和历史拒绝。
"""
from __future__ import annotations

import argparse
import json
import pathlib

from p0_background_preview_materials import checked_artifact
from p0_common import ROOT, ensure_gpu_idle, run_stage
from p0_generated_window_gate import review_bound_to_video, target_condition_review_passed
from p0_lossless_background import read_exact_video
from p0_lossless_background_gate import acceptance_reasons, roi_temporal_residual
from p0_policy_runtime import sha256
from p0_table_background import pixel_certificate
from p0_table_background_gate import explicit_review_failures
from p0_table_uv_validation import CONFIG, CONFIG_PATH, checked_controls, checked_render


def review_checks(reviews: dict, probe_sha: str, video_sha: str) -> dict:
    """复核只能用于它实际看过的完整视频；几何通过不能补齐缺少的语义结论。"""
    semantic = reviews.get("semantic_review")
    semantic_ok = bool(review_bound_to_video(semantic, probe_sha, video_sha, 93)
        and semantic.get("passed") is True
        and semantic.get("object_identity_preserved") is True
        and semantic.get("task_semantics_preserved") is True
        and semantic.get("unwanted_objects") is False and semantic.get("no_new_obstacles") is True
        and isinstance(semantic.get("observations"), str) and semantic["observations"].strip())
    temporal = reviews.get("temporal_review")
    temporal_ok = bool(review_bound_to_video(temporal, probe_sha, video_sha, 93)
        and all(temporal.get(key) is True for key in [
            "passed", "background_consistent", "no_flicker", "boundary_seams_acceptable"])
        and isinstance(temporal.get("observations"), str) and temporal["observations"].strip())
    return {"semantic_ok": semantic_ok, "background_temporal_ok": temporal_ok,
            "condition_ok": target_condition_review_passed(
                reviews.get("condition_review"), CONFIG["target_condition"], probe_sha, video_sha, 93)}


def previous_failures(probe_sha: str, video_sha: str) -> list[dict]:
    failures = []
    for stage in ["uv_window_gate", "table_background_gate", "lossless_background_gate", "generated_window_gate"]:
        for path in sorted((ROOT / "runs/p0").glob(stage + "_*/quality_gate.json")):
            data = json.loads(path.read_text())
            if data.get("generation_probe_sha256") != probe_sha or data.get("output_video_sha256") != video_sha:
                continue
            checked_artifact(path, stage)
            reviews = data.get("reviews", {}) if stage != "generated_window_gate" else {
                "semantic_review": data.get("semantic_review"), "condition_review": data.get("target_condition_review")}
            rejected = explicit_review_failures(reviews, probe_sha, video_sha)
            if rejected:
                failures.append({"path": str(path), "sha256": sha256(path), "failed_reviews": rejected})
    return failures


def checked_gate(path: pathlib.Path):
    """训练消费时重新验证绑定与准入条件，不信任一个可单独改写的true字段。"""
    checked_artifact(path, "uv_window_gate")
    data = json.loads(path.read_text())
    if data.get("method") != CONFIG["version"] or data.get("passed") is not True \
            or data.get("eligible_for_training") is not True or data.get("reject_reasons") != [] \
            or data.get("config_sha256") != sha256(CONFIG_PATH):
        raise ValueError("UV窗口尚未通过完整独立准入")
    role = data["role"]
    render_path = pathlib.Path(data["render_report"])
    if sha256(render_path) != data["render_report_sha256"] \
            or sha256(render_path.parent / "probe.json") != data["generation_probe_sha256"] \
            or sha256(render_path.parent / "cosmos.mp4") != data["output_video_sha256"]:
        raise ValueError("准入没有绑定当前实际渲染与视频")
    report, evidence, source, masks, proof = checked_render(render_path, role)
    for current_role, binding in data["method_validation"].items():
        control_path = pathlib.Path(binding["path"])
        if sha256(control_path) != binding["sha256"]:
            raise ValueError("方法控制文件已改变")
        checked_controls(control_path, current_role)
    if set(data["method_validation"]) != {"design", "heldout"} \
            or not all(review_checks(data["reviews"], data["generation_probe_sha256"], data["output_video_sha256"]).values()) \
            or data.get("geometry_passed") is not True or data.get("roi_temporal_passed") is not True \
            or data.get("source_episode_id") != evidence["source_episode_id"]:
        raise ValueError("准入缺少完整方法、来源或非几何证据")
    if previous_failures(data["generation_probe_sha256"], data["output_video_sha256"]):
        raise ValueError("同一实际视频已有明确非几何拒绝，禁止覆盖")
    return data, report, evidence, source, masks, proof


def gate(directory: pathlib.Path, args):
    ensure_gpu_idle()
    import numpy as np
    import yaml

    controls = {}
    for role in ["design", "heldout"]:
        path = getattr(args, role + "_report")
        value, _ = checked_controls(path, role)
        controls[role] = {"path": str(path), "sha256": sha256(path), "render_report": value["render_report"]}
    report, evidence, source, masks, proof = checked_render(args.render_report, args.role)
    if controls[args.role]["render_report"] != str(args.render_report):
        raise ValueError("不能把另一段窗口的方法控制借给当前视频")
    heldout_control, _ = checked_controls(args.heldout_report, "heldout")
    if heldout_control["design_binding"]["path"] != str(args.design_report) \
            or heldout_control["design_binding"]["sha256"] != sha256(args.design_report):
        raise ValueError("双源方法报告不属于同一条设计验证链")
    declaration_path = args.render_report.parent / "declaration.json"
    checked_artifact(declaration_path, "table_uv_replay")
    declaration = json.loads(declaration_path.read_text())
    # 原始edge准备器有它自己的光照prompt；本新派生的目标在渲染前单独固定，绝不修改旧来源。
    if declaration["method"] != CONFIG or declaration["config_sha256"] != sha256(CONFIG_PATH) \
            or declaration["source_input_sha256"] != report["source_input_sha256"]:
        raise ValueError("派生目标和完整配方未在实际重渲染前声明")
    video = args.render_report.parent / "cosmos.mp4"
    output, metadata = read_exact_video(video)
    raw, _ = read_exact_video(args.render_report.parent / "raw_cosmos.mp4")
    elements = np.load(args.render_report.parent / "elements.npy", allow_pickle=False)
    certificate = pixel_certificate(source, raw, elements == report["table_id"], output, masks)
    residual = roi_temporal_residual(source, output, masks)
    thresholds = yaml.safe_load((ROOT / "configs/gen_gate.yaml").read_text())
    p95 = float(np.percentile(residual, 95))
    probe_path = args.render_report.parent / "probe.json"
    probe_sha, video_sha = sha256(probe_path), sha256(video)
    review_values = json.loads(args.reviews.read_text())
    checks = review_checks(review_values, probe_sha, video_sha)
    previous = previous_failures(probe_sha, video_sha)
    reasons = acceptance_reasons(geometry_ok=certificate["geometry_passed"], method_ok=True,
        roi_temporal_ok=p95 <= thresholds["temporal_residual_p95_max"], prior_target_rejected=bool(previous), **checks)
    result = {"method": CONFIG["version"], "config_sha256": sha256(CONFIG_PATH), "role": args.role,
              "render_report": str(args.render_report), "render_report_sha256": sha256(args.render_report),
              "generation_probe": str(probe_path), "generation_probe_sha256": probe_sha,
              "output_video": str(video), "output_video_sha256": video_sha,
              "source_input": report["source_input"], "source_input_sha256": report["source_input_sha256"],
              "source_episode_id": evidence["source_episode_id"], "source_proof": proof,
              "window": evidence["window"], "view": "agentview", "source_kind": "demo",
              "instruction": evidence["instruction"], "target_condition": CONFIG["target_condition"],
              "target_condition_declaration": {"path": str(declaration_path), "sha256": sha256(declaration_path)},
              "method_validation": controls, "texture_binding": report["texture_binding"],
              "output_kind": "background_composite", "derived_kind": report["derived_kind"],
              "geometry_evidence_type": report["geometry_evidence_type"], "pixel_certificate": certificate,
              "geometry_passed": certificate["geometry_passed"], "video_metadata": metadata,
              "roi_temporal_residual_p95": p95, "roi_temporal_residual_per_frame": residual,
              "roi_temporal_threshold": thresholds["temporal_residual_p95_max"],
              "roi_temporal_passed": p95 <= thresholds["temporal_residual_p95_max"],
              "review_file": str(args.reviews), "review_file_sha256": sha256(args.reviews),
              "reviews": review_values, "review_checks": checks, "prior_failures": previous,
              "passed": not reasons, "eligible_for_training": not reasons, "reject_reasons": reasons,
              "is_eval": False, "policy_improvement_claimed": False,
              "limitation": "仅两条成功训练示范验证的桌体视觉派生；尚无策略训练或RSI收益，非原生整段Cosmos生成"}
    (directory / "quality_gate.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return {key: result[key] for key in ["role", "source_episode_id", "passed", "eligible_for_training", "reject_reasons"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--role", choices=["design", "heldout"], required=True)
    for name in ["render-report", "design-report", "heldout-report", "reviews"]:
        parser.add_argument("--" + name, type=pathlib.Path, required=True)
    arguments = parser.parse_args()
    for name, value in vars(arguments).items():
        if isinstance(value, pathlib.Path):
            value = value.resolve(strict=True)
            if not value.is_relative_to(ROOT / "runs/p0"):
                parser.error("全部输入须位于本项目P0目录")
            setattr(arguments, name, value)
    raise SystemExit(run_stage("uv_window_gate", lambda directory: gate(directory, arguments),
        {"method": CONFIG["version"], "compute": "Thor CPU完整解码、像素和来源核验，无模型或训练",
         "precision": "RGB uint8/CPU统计fp32及fp64", "batch": None, "seed": None,
         "frames": 93, "fps": 20, "source_windows": 1, "training_steps": 0}))
