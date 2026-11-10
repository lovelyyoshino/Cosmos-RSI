#!/usr/bin/env python3
"""桌体分割限定的无损派生基础检查；不把像素证明等同于完整训练资格。

@author pony
@date 2026-11-08
@version v2.1.0
@last_modified 2026-11-08
@changelog
    - v2.1.0 (2026-11-08): 纳入判决/控制直接依赖，绑定准备入口与实际代码，固定新独立源。
    - v2.0.0 (2026-11-08): 固定可改写桌体、完整非桌体保护、真实视频及来源/方法绑定。
"""
from __future__ import annotations

import json
import pathlib

from p0_common import ROOT
from p0_lossless_background import checked_bound_file, checked_training_source, read_exact_video
from p0_policy_runtime import sha256


CONFIG_PATH = ROOT / "configs/table_background_gate.yaml"
CONFIG = {
    "version": "thor-table-background-v2.1", "geometry_evidence_type": "lossless_table_pixels_v2",
    "frames": 93, "fps": 20, "width": 256, "height": 256, "table_body": "study_table",
    "table_mesh": "desk_vis", "table_visual_group": 1, "protected_radius_pixels": 8,
    "minimum_editable_pixels_per_frame": 1024, "native_resize": "opencv_inter_area",
    "target_condition": {"background": "matte_dark_wood"}, "design_episode_suffix": "caddy-demo_16",
    "heldout_episode_suffix": "caddy-demo_22", "temporal_residual_p95_max": .08,
}
RECIPE = {"kind": "cosmos_table_source_scene_composite", "method": CONFIG["version"],
          "radius": 8, "frames": 93, "lossless_rgb_required": True,
          "editable_scope": "same_observation_study_table_desk_vis_eroded8", "native_resize": "opencv_inter_area"}
METHOD_FILES = [
    "scripts/p0_table_background.py", "scripts/p0_prepare_table_background.py",
    "scripts/p0_table_background_controls.py", "scripts/p0_table_background_gate.py",
    "scripts/p0_lossless_background.py", "scripts/p0_demo_replay.py", "scripts/p0_common.py",
    "scripts/p0_generated_window_gate.py", "scripts/p0_transfer_smoke.py", "scripts/p0_policy_runtime.py",
    "scripts/p0_lossless_background_gate.py", "scripts/p0_context_normalized_segmentation.py",
    "scripts/p0_segmentation_controls.py", "scripts/p0_background_preview_materials.py", "scripts/p0_sim_data.py",
    "configs/table_background_gate.yaml",
]


def checked_config() -> dict:
    import yaml

    if yaml.safe_load(CONFIG_PATH.read_text()) != CONFIG:
        raise ValueError("桌面派生方法偏离运行前固定配置，须新版本及独立方法验证")
    return CONFIG


def checked_dependencies(probe: dict) -> dict:
    actual = {name: sha256(ROOT / name) for name in METHOD_FILES}
    if any(probe.get("source_files_sha256", {}).get(name) != checksum for name, checksum in actual.items()):
        raise ValueError("判决/控制方法依赖已经改变，不能借用旧设计或独立报告")
    return actual


def select_table_geometries(geometries: list[dict]) -> list[int]:
    """只识别固定桌体网格，不把没有实例名称的墙面或地板当成可改写背景。"""
    selected = [row["id"] for row in geometries if row.get("body") == CONFIG["table_body"]
                and row.get("mesh") == CONFIG["table_mesh"] and row.get("group") == CONFIG["table_visual_group"]]
    if len(selected) != 1 or type(selected[0]) is not int or selected[0] < 0:
        raise ValueError("源场景没有唯一预声明的study_table/desk_vis可见桌体")
    return selected


def editable_table(elements, table_ids: list[int], task_masks):
    import cv2
    import numpy as np

    elements, task_masks = np.asarray(elements), np.asarray(task_masks)
    if elements.shape != (93, 256, 256) or not np.issubdtype(elements.dtype, np.integer) \
            or task_masks.shape != (93, 3, 256, 256) or not np.isin(task_masks, [0, 1]).all() \
            or len(table_ids) != 1 or type(table_ids[0]) is not int or table_ids[0] < 0:
        raise ValueError("桌面分割与固定三任务ROI必须完整逐帧对齐")
    raw = np.isin(elements, table_ids)
    if np.any(raw & task_masks.any(axis=1)):
        raise ValueError("桌体分割不能覆盖机器人或任务物体")
    kernel = np.ones((17, 17), np.uint8)
    editable = np.asarray([cv2.erode(frame.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT,
                                     borderValue=0).astype(bool) for frame in raw])
    if any(frame.sum() < CONFIG["minimum_editable_pixels_per_frame"] for frame in editable):
        raise ValueError("保护全部非桌体及8像素边界后，可改写桌面面积不足；禁止缩小保护范围")
    return editable


def compose_table(source, native, editable):
    import numpy as np

    if any(np.asarray(value).shape != (93, 256, 256, 3) or np.asarray(value).dtype != np.uint8
           for value in [source, native]) or np.asarray(editable).shape != (93, 256, 256) \
            or np.asarray(editable).dtype != np.bool_:
        raise ValueError("桌面合成只接收完整uint8源/原生图和bool可改写掩码")
    return np.where(editable[..., None], native, source)


def pixel_certificate(source, native, editable, output, masks) -> dict:
    import numpy as np

    expected = compose_table(source, native, editable)
    if np.asarray(output).shape != expected.shape or np.asarray(output).dtype != np.uint8 \
            or masks.shape != (93, 3, 256, 256) or not np.isin(masks, [0, 1]).all() \
            or np.any(editable & masks.any(axis=1)):
        raise ValueError("输出/任务ROI不完整或可改写区包含任务前景")
    rows = []
    for index in range(93):
        selected = editable[index]
        if selected.all() or selected.sum() < CONFIG["minimum_editable_pixels_per_frame"]:
            raise ValueError("每帧必须同时保留非桌体和足量可改写桌面")
        src_diff = np.abs(output[index].astype(np.int16) - source[index].astype(np.int16))
        native_diff = np.abs(output[index].astype(np.int16) - native[index].astype(np.int16))
        rows.append({"frame": index, "protected_max_abs_diff": int(src_diff[~selected].max()),
                     "table_max_abs_diff": int(native_diff[selected].max()),
                     "expected_composite_equal": bool(np.array_equal(output[index], expected[index])),
                     "editable_pixels": int(selected.sum()), "protected_pixels": int((~selected).sum()),
                     "changed_table_pixels": int(np.count_nonzero(np.any(output[index] != source[index], axis=-1) & selected)),
                     "roi_max_abs_diff": {name: int(src_diff[mask.astype(bool)].max()) if mask.any() else 0
                                           for name, mask in zip(["robot", "black_book_1", "desk_caddy_1"], masks[index])}})
    return {"geometry_evidence_type": CONFIG["geometry_evidence_type"], "frames": list(range(93)),
            "geometry_passed": all(row["expected_composite_equal"] and row["protected_max_abs_diff"] == 0
                                   and row["table_max_abs_diff"] == 0 for row in rows), "per_frame": rows,
            "limitation": "证明所有不可改写像素及桌面来源；不证明木材目标、背景语义或策略收益"}


def checked_table_source(path: pathlib.Path):
    import numpy as np

    checked_config()
    evidence, source, masks, proof = checked_training_source(path)
    table = evidence.get("table_geometry", {})
    if evidence.get("composite_recipe") != RECIPE or evidence.get("target_condition") != CONFIG["target_condition"] \
            or table.get("method_config_sha256") != sha256(CONFIG_PATH) \
            or table.get("same_observation_as_source_rgb") is not True or table.get("replay_passed") is not True:
        raise ValueError("源缺少生成前绑定的完整桌面定义与同帧重放")
    ids = select_table_geometries(table.get("geometries", []))
    if ids != table.get("table_geom_ids"):
        raise ValueError("桌体身份与预声明的几何选择不一致")
    if not {"table_elements.npy", "editable_table.npy", "scene_mask.mp4"} <= evidence["files_sha256"].keys():
        raise ValueError("源没有实际同帧分割、可改写桌面或控制掩码")
    elements = np.load(path.parent / "table_elements.npy", allow_pickle=False)
    editable = np.load(path.parent / "editable_table.npy", allow_pickle=False)
    expected = editable_table(elements, ids, masks)
    if editable.dtype != np.bool_ or not np.array_equal(editable, expected):
        raise ValueError("实际可改写掩码偏离固定桌体分割/8像素边界")
    scene_mask, _ = read_exact_video(path.parent / "scene_mask.mp4")
    if not np.array_equal(scene_mask, np.repeat((~editable)[..., None], 3, axis=-1).astype(np.uint8) * 255):
        raise ValueError("实际生成控制掩码与完整非桌体保护区不一致")
    spec_path = path.parent / "spec.json"
    preparation = json.loads((path.parent / "probe.json").read_text())
    preparation_command = preparation.get("command", [])
    if preparation.get("stage") != "background_transfer_input" or len(preparation_command) < 2 \
            or (ROOT / preparation_command[1]).resolve() != ROOT / "scripts/p0_prepare_table_background.py" \
            or preparation.get("conditions", {}).get("method") != CONFIG["version"]:
        raise ValueError("桌面定义必须由固定的完整仿真准备入口产生")
    for name in ["scripts/p0_table_background.py", "scripts/p0_prepare_table_background.py", "scripts/p0_lossless_background.py",
                 "scripts/p0_demo_replay.py", "scripts/p0_sim_data.py", "configs/table_background_gate.yaml"]:
        if preparation.get("source_files_sha256", {}).get(name) != sha256(ROOT / name):
            raise ValueError("桌面源重放代码或配置已经改变，须重新准备源")
    checked_bound_file(spec_path, preparation.get("artifacts_sha256", {}).get("spec.json"), path.parent)
    spec = json.loads(spec_path.read_text())
    if spec.get("video_path") != str(path.parent / "source.mp4") \
            or spec.get("edge") != {"control_path": str(path.parent / "edge.mp4"), "control_weight": 1.0,
                                     "mask_path": str(path.parent / "scene_mask.mp4")} \
            or spec.get("guided_generation_mask") is not None \
            or any(spec.get(key) != value for key, value in
                   {"max_frames": 93, "num_video_frames_per_chunk": 93, "num_steps": 4, "seed": 42, "resolution": "480"}.items()):
        raise ValueError("原生RGB/edge/非桌体掩码没有绑定实际源文件")
    return evidence, source, masks, editable, elements, proof


def checked_method(design_path: pathlib.Path, heldout_path: pathlib.Path) -> dict:
    from p0_background_preview_materials import checked_artifact
    from p0_segmentation_controls import check_independent_source
    from p0_table_background_controls import validate_report

    reports, sources = [], []
    for path, role in [(design_path, "design"), (heldout_path, "heldout")]:
        probe = checked_artifact(path, "table_background_controls")
        data = json.loads(path.read_text())
        validate_report(data, role)
        if data.get("method_config_sha256") != sha256(CONFIG_PATH):
            raise ValueError("方法配置或代码改变，不能借用旧设计/独立控制")
        checked_dependencies(probe)
        source = checked_bound_file(pathlib.Path(data["source_input"]), data["source_input_sha256"], ROOT / "runs/p0")
        evidence, *_ = checked_table_source(source)
        if not evidence["source_episode_id"].endswith(CONFIG[role + "_episode_suffix"]):
            raise ValueError("方法检查源不符合事先固定划分")
        reports.append(data)
        sources.append(evidence)
    binding = reports[1].get("design_binding", {})
    if binding.get("report_sha256") != sha256(design_path) or binding.get("report") != str(design_path):
        raise ValueError("独立验证未绑定完整设计结果")
    return {"passed": True, "design": {"path": str(design_path), "sha256": sha256(design_path)},
            "heldout": {"path": str(heldout_path), "sha256": sha256(heldout_path)},
            "source_independence": check_independent_source(sources[1], sources[0]),
            "dependencies_sha256": {name: sha256(ROOT / name) for name in METHOD_FILES}}


def verified_composite_paths(generation: dict, probe_path: pathlib.Path, root: pathlib.Path):
    """供原SAM2门复用同一新派生；验证实际像素，禁止借新类型伪装原生输出。"""
    import cv2
    import numpy as np
    from p0_generated_window_gate import verified_generation_paths

    if root != ROOT or generation.get("stage") != "table_background_composite" or generation.get("status") != "passed":
        raise ValueError("桌面派生只接受本项目已完成的单级运行")
    result = generation.get("result", {})
    record = result.get("source_success_evidence", {})
    source_path = checked_bound_file(pathlib.Path(record["path"]), record["sha256"], ROOT / "runs/p0")
    evidence, source, masks, editable, _, _ = checked_table_source(source_path)
    native_probe = checked_bound_file(pathlib.Path(result["native_generation_probe"]),
                                     result["native_generation_probe_sha256"], ROOT / "runs/p0")
    native_record = json.loads(native_probe.read_text())
    if native_record.get("stage") != "transfer_smoke":
        raise ValueError("禁止递归合成桌面派生")
    native_source, native_path = verified_generation_paths(native_record, native_probe, root)
    if native_source != source_path or result.get("native_video_sha256") != sha256(native_path) \
            or result.get("composite_recipe") != RECIPE or result.get("derived_kind") != RECIPE["kind"] \
            or result.get("target_condition") != CONFIG["target_condition"] \
            or result.get("source_labels_sha256") != evidence["labels_sha256"] \
            or result.get("guardrails_enabled") is not True or len(result.get("outputs", [])) != 1:
        raise ValueError("派生未绑定同一原生输出、标签或运行前配方")
    output = pathlib.Path(result["outputs"][0]).resolve(strict=True)
    checked_bound_file(output, result.get("output_video_sha256"), probe_path.parent)
    checked_bound_file(output, generation.get("artifacts_sha256", {}).get(str(output.relative_to(probe_path.parent))),
                       probe_path.parent)
    native, metadata = read_exact_video(native_path, size=None)
    if min(metadata["width"], metadata["height"]) < 256:
        raise ValueError("原生背景只能下采样，不得上采样替代")
    background = np.asarray([cv2.resize(f, (256, 256), interpolation=cv2.INTER_AREA) for f in native])
    actual, _ = read_exact_video(output)
    if not pixel_certificate(source, background, editable, actual, masks)["geometry_passed"]:
        raise ValueError("实际桌面派生没有保持完整非桌体像素及原生桌面来源")
    return source_path, output
