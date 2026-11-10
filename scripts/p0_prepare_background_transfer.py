#!/usr/bin/env python3
"""生成前声明背景纹理目标与前景保护配方，不修改任何历史来源。

@author pony
@date 2026-11-07
@version v1.2.0
@last_modified 2026-11-08
@changelog
    - v1.2.0 (2026-11-08): 预声明仅降低全帧vis权重的同源对照，拒绝同时改变ROI范围。
    - v1.1.0 (2026-11-07): 新源可固定四分支同帧ROI控制，保持背景目标、提示词和标量权重不变。
    - v1.0.0 (2026-11-07): 新建独立背景试验，保持成功源标签及原四控制权重。
"""
from __future__ import annotations

import argparse
import copy
import json
import pathlib
import shutil

from p0_common import ROOT, ensure_gpu_idle, run_stage
from p0_policy_runtime import sha256
from p0_transfer_smoke import MULTICONTROL_ORDER, validate_generation_spec

TARGET = {"background": "matte_dark_wood"}
COMPOSITE_RECIPE = {"kind": "cosmos_background_source_foreground_composite", "radius": 8,
                    "frames": 93, "lossless_rgb_required": True}


def background_spec(parent: dict, directory: pathlib.Path, instruction: str,
                    control_scope: str = "full-frame", appearance_weight: float = .5) -> dict:
    validate_generation_spec(parent, "multicontrol-vis-edge", 35)
    if parent.get("num_steps") != 35 or parent.get("seed") != 42 or str(parent.get("resolution")) != "480" \
            or not instruction.strip():
        raise ValueError("背景单样本保持原35步、seed42、480档，并要求真实任务指令")
    if control_scope not in {"full-frame", "roi"}:
        raise ValueError("背景控制范围只能是full-frame或roi")
    if appearance_weight not in {.5, .1} or (control_scope == "roi" and appearance_weight != .5):
        raise ValueError("本次只允许全帧vis由0.5降至0.1，禁止叠加ROI控制干预")
    spec = copy.deepcopy(parent)
    spec.update({"name": "p0_predeclared_matte_dark_wood_background",
                 "video_path": str(directory / "source.mp4"),
                 "guided_generation_mask": str(directory / "mask.mp4"),
                 "prompt": "A fixed camera records the same Panda robot arm performing the same task: "
                 + instruction + ". Change only the table and background surface texture to matte dark wood. "
                 "Preserve the original illumination, robot, gripper and every task object, their identity, "
                 "color, geometry, positions, motion and visibility. Do not add or remove objects. "
                 "Keep the camera fixed. No colored light patches, no text or overlays."})
    for name in MULTICONTROL_ORDER:
        spec[name]["control_path"] = str(directory / f"{name}.mp4")
        if control_scope == "roi":
            spec[name]["mask_path"] = str(directory / "mask.mp4")
    if control_scope == "roi":
        spec["name"] += "_roi_controls"
    if appearance_weight == .1:
        spec["name"] += "_low_vis"
        spec["vis"]["control_weight"] = appearance_weight
    recipe = "multicontrol-low-vis-background" if appearance_weight == .1 else \
        "multicontrol-roi-background" if control_scope == "roi" else "multicontrol-vis-edge"
    validate_generation_spec(spec, recipe, 35)
    return spec


def bind_comparison_baseline(probe_path: pathlib.Path | None, parent: pathlib.Path,
                             spec: dict, evidence: dict, intervention_kind: str = "native_control_roi_mask") -> dict:
    from p0_generated_window_gate import verified_generation_paths

    if intervention_kind not in {"native_control_roi_mask", "vis_control_weight_reduction"}:
        raise ValueError("不支持未预声明的背景生成干预")
    if probe_path is None or not probe_path.resolve(strict=True).is_relative_to(ROOT / "runs/p0"):
        raise ValueError("单因素对照必须显式绑定已实际完成的全帧背景生成基线")
    probe_path = probe_path.resolve(strict=True)
    generation = json.loads(probe_path.read_text())
    source, output = verified_generation_paths(generation, probe_path, ROOT)
    baseline = json.loads(source.read_text())
    preparation = json.loads((source.parent / "probe.json").read_text())
    baseline_spec_path = source.parent / "spec.json"
    before = json.loads(baseline_spec_path.read_text())
    if generation.get("stage") != "transfer_smoke" or generation["result"].get("recipe") != "multicontrol-vis-edge" \
            or any(generation["result"].get(key) != value for key, value in {"num_steps": 35, "seed": 42, "batch": 1,
                                                                         "resolution_requested": "480"}.items()) \
            or baseline.get("target_condition") != TARGET or baseline.get("composite_recipe") != COMPOSITE_RECIPE \
            or baseline.get("control_recipe") != "multicontrol-vis-edge" \
            or baseline.get("parent_verified_input") != str(parent) \
            or baseline.get("parent_verified_input_sha256") != sha256(parent) \
            or baseline.get("files_sha256") != evidence["files_sha256"] \
            or baseline.get("labels_sha256") != evidence["labels_sha256"] \
            or preparation.get("stage") != "background_transfer_input" or preparation.get("status") != "passed" \
            or any(preparation.get("artifacts_sha256", {}).get(path.name) != sha256(path)
                   for path in [source, baseline_spec_path]):
        raise ValueError("比较基线不是同一源、同一预声明背景目标的实际全帧实验")
    paths = {"name", "video_path", "guided_generation_mask", *MULTICONTROL_ORDER}
    if {key: value for key, value in before.items() if key not in paths} != \
            {key: value for key, value in spec.items() if key not in paths}:
        raise ValueError("ROI对照改变了基线提示词或采样参数")
    for name in MULTICONTROL_ORDER:
        original_weight = .5 if intervention_kind == "vis_control_weight_reduction" and name == "vis" \
            else spec[name]["control_weight"]
        if before[name] != {"control_path": str(source.parent / f"{name}.mp4"),
                            "control_weight": original_weight}:
            raise ValueError("比较基线的控制内容或标量权重不一致")
    if intervention_kind == "vis_control_weight_reduction" and (
            spec["vis"]["control_weight"] != .1 or any(
                set(spec[name]) != {"control_path", "control_weight"} for name in MULTICONTROL_ORDER)):
        raise ValueError("降低vis的对照必须保持全帧控制且不能额外改变控制参数")
    comparison = {"generation_probe": str(probe_path), "generation_probe_sha256": sha256(probe_path),
            "source_input": str(source), "source_input_sha256": sha256(source),
            "source_spec": str(baseline_spec_path), "source_spec_sha256": sha256(baseline_spec_path),
            "source_probe_sha256": sha256(source.parent / "probe.json"),
            "output_video": str(output), "output_video_sha256": sha256(output),
            "only_intervention": "四分支增加同帧原生mask_path；提示词、标量权重、源及采样配置保持"}
    if intervention_kind == "vis_control_weight_reduction":
        comparison.update({"intervention_kind": intervention_kind,
            "only_intervention": "全帧vis标量权重由0.5降至0.1；提示词、其他权重、源、控制区域及采样配置保持"})
    return comparison


def prepare(directory: pathlib.Path, args: argparse.Namespace) -> dict:
    ensure_gpu_idle()
    parent = args.source_input
    evidence = json.loads(parent.read_text())
    probe = json.loads((parent.parent / "probe.json").read_text())
    if probe.get("stage") != "edge_anchor_input" or probe.get("status") != "passed" \
            or probe.get("artifacts_sha256", {}).get(parent.name) != sha256(parent) \
            or evidence.get("source_success") is not True or not evidence.get("replay_validation", {}).get("passed") \
            or not evidence.get("control_replay_validation", {}).get("passed") \
            or evidence.get("mask_transform") is not None:
        raise ValueError("只接受已完整重放成功的原四控制训练源")
    cache = pathlib.Path(evidence["cache_manifest"]).resolve(strict=True)
    if not cache.is_relative_to(ROOT / "data") or sha256(cache) != evidence["cache_manifest_sha256"]:
        raise ValueError("源缓存清单改变或越界")
    episode = next(row for row in json.loads(cache.read_text())["episodes"]
                   if row["episode_id"] == evidence["source_episode_id"])
    if episode.get("split") != "demo_train" or episode.get("env_success") is not True:
        raise ValueError("背景增强不接收验证或冻结评测源")
    spec_path = pathlib.Path(evidence["spec"]).resolve(strict=True)
    if spec_path.parent != parent.parent or probe["artifacts_sha256"].get(spec_path.name) != sha256(spec_path):
        raise ValueError("父配方来源错误")
    control_scope = getattr(args, "control_scope", "full-frame")
    appearance_weight = getattr(args, "appearance_weight", .5)
    spec = background_spec(json.loads(spec_path.read_text()), directory, evidence["instruction"], control_scope,
                           appearance_weight)
    low_vis = appearance_weight == .1
    intervention_kind = "vis_control_weight_reduction" if low_vis else "native_control_roi_mask"
    baseline = bind_comparison_baseline(getattr(args, "baseline_probe", None), parent, spec, evidence, intervention_kind) \
        if control_scope == "roi" or low_vis else None
    if low_vis:
        if evidence.get("vis_control", {}).get("control_weight") != .5:
            raise ValueError("vis单因素对照必须从已核验0.5权重的外观控制源开始")
        evidence["vis_control"] = {**evidence["vis_control"], "parent_control_weight": .5, "control_weight": .1}
    files = {**evidence["files_sha256"], "source_labels.npz": evidence["labels_sha256"]}
    for name, checksum in files.items():
        path = (parent.parent / name).resolve(strict=True)
        if pathlib.PurePosixPath(name).name != name or not path.is_relative_to(parent.parent) or sha256(path) != checksum:
            raise ValueError("源图、ROI或监督标签改变")
        shutil.copyfile(path, directory / name)
        if sha256(directory / name) != checksum:
            raise ValueError("复制后的源文件不一致")
    # 新目标及派生方法在首次生成之前固定；旧暖暗试验不会因此取得任何资格。
    evidence.update({"spec": str(directory / "spec.json"), "parent_verified_input": str(parent),
                     "parent_verified_input_sha256": sha256(parent), "target_condition": TARGET,
                     "target_condition_scope": "背景纹理变化；机械臂、书本及收纳盒前景保留，不宣称全场光照变化",
                     "composite_recipe": COMPOSITE_RECIPE,
                     "control_recipe": "multicontrol-low-vis-background" if low_vis else
                     "multicontrol-roi-background" if control_scope == "roi" else "multicontrol-vis-edge",
                     "control_mask": "四分支仅约束同帧前景ROI，背景不受控制" if control_scope == "roi"
                     else "四分支全帧控制，另有同帧ROI空间引导",
                     "intervention": {"kind": "vis_control_weight_reduction", "branch": "vis", "before": .5, "after": .1,
                                      "prompt_and_geometry_controls_unchanged": True} if low_vis else
                     {"kind": "native_control_roi_mask", "branches": MULTICONTROL_ORDER,
                                      "mask_file": "mask.mp4", "mask_sha256": files["mask.mp4"],
                                      "prompt_and_scalar_weights_unchanged": True} if control_scope == "roi" else None,
                     "comparison_baseline": baseline,
                     "eligible_for_training": False, "reason": "新原生生成、派生数据及完整质量门均待验收"})
    for name, value in [("spec.json", spec), ("verified_input.json", evidence)]:
        (directory / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    return {"source_input": str(directory / "verified_input.json"), "spec": str(directory / "spec.json"),
            "target_condition": TARGET, "composite_recipe": COMPOSITE_RECIPE,
            "sample_count": 1, "eligible_for_training": False, "compute": "Thor CPU source verification"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-input", type=pathlib.Path, required=True)
    parser.add_argument("--control-scope", choices=["full-frame", "roi"], default="full-frame")
    parser.add_argument("--appearance-weight", type=float, choices=[.5, .1], default=.5)
    parser.add_argument("--baseline-probe", type=pathlib.Path, help="ROI对照必须绑定已有全帧背景原生生成")
    args = parser.parse_args()
    args.source_input = args.source_input.resolve(strict=True)
    if not args.source_input.is_relative_to(ROOT / "runs/p0"):
        parser.error("源必须位于Thor项目P0目录")
    raise SystemExit(run_stage("background_transfer_input", lambda directory: prepare(directory, args),
                               {"target_condition": TARGET, "composite_recipe": COMPOSITE_RECIPE,
                                "control_scope": args.control_scope, "appearance_weight": args.appearance_weight}))
