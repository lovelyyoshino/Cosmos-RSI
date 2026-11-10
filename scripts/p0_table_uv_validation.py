#!/usr/bin/env python3
"""固定镜像UV图集的独立方法验证；保留旧方法原字节，控制通过不授予入训资格。

@author pony
@date 2026-11-08
@version v1.0.0
@last_modified 2026-11-08
@changelog
    - v1.0.0 (2026-11-08): 独立冻结原验证语义；加入图集、demo_26及新设计逐像素复现。
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import shutil

from p0_background_preview_materials import checked_artifact
from p0_common import ROOT, ensure_gpu_idle, run_stage
from p0_lossless_background import checked_bound_file, checked_training_source, read_exact_video, validate_labels
from p0_policy_runtime import sha256
from p0_prepare_table_background import write_video
from p0_segmentation_controls import check_independent_source
from p0_table_background import pixel_certificate, select_table_geometries
from p0_table_renderer_design import CONFIG as RENDER_CONFIG, install_renderer, protected_composite
from p0_table_uv_atlas_design import CONFIG as ATLAS_CONFIG, mirrored_atlas

CONFIG_PATH = ROOT / "configs/table_uv_validation.yaml"
CONFIG = {
    "version": "thor-table-uv-validation-v1", "purpose": "method_validation_only",
    "design_episode_suffix": "caddy-demo_16", "heldout_episode_suffix": "caddy-demo_26",
    "design_reference_run": "table_uv_atlas_design_20261108T172056_719969+0800",
    "texture_design_run": "table_texture_design_20261108T132043_578426+0800",
    "texture_sha256": "1da3f6cf62bf0447dceaf7ef694252d53b1bb452eb89840becff985dc209db97",
    "target_condition": {"background": "matte_dark_wood"}, "frames": 93, "fps": 20,
    "width": 256, "height": 256, "table_body": "study_table", "table_mesh": "desk_vis",
    "table_visual_group": 1, "texture_resize": "opencv_inter_linear",
    "atlas_side": 1024, "repeats_per_axis": 8, "tiling": "alternating_mirror_edge_continuous",
    "atlas_resize": "opencv_inter_area", "analytic_albedo_rgb": [88, 50, 30],
    "matte_reflectance": 0., "matte_shininess": 0., "matte_specular": 0.,
    "composite_scope": "exact_same_observation_visible_table", "protected_pixels": "all_non_table_source_rgb",
    "eligible_for_training": False,
}
METHOD_FILES = [
    "configs/table_uv_validation.yaml", "scripts/p0_table_uv_validation.py",
    "scripts/p0_table_renderer_design.py", "scripts/p0_prepare_verified_transfer.py",
    "scripts/p0_prepare_table_background.py", "scripts/p0_table_background.py",
    "scripts/p0_lossless_background.py", "scripts/p0_background_preview_materials.py",
    "scripts/p0_demo_replay.py", "scripts/p0_sim_data.py", "scripts/p0_policy_runtime.py",
    "scripts/p0_common.py", "scripts/p0_segmentation_controls.py", "scripts/p0_single_control_materials.py",
    "scripts/p0_context_normalized_segmentation.py", "scripts/p0_table_texture_design.py",
    "scripts/p0_table_reference_materials.py", "configs/table_renderer_design.yaml",
    "configs/table_texture_design.yaml", "configs/table_background_gate.yaml",
    "scripts/p0_run_uv_validation.sh", "scripts/p0_table_uv_atlas_design.py", "configs/table_uv_atlas_design.yaml",
]
NEGATIVES = {"identity", "analytic_render", "cosmos_render"}
ERRORS = {"protected_pixel_changed", "table_recipe_changed", "robot_shift12", "book_shift12", "mug_recolored",
          "global_shift24", "swap_first_last_frames", "wrong_fps", "missing_frame", "wrong_size",
          "wrong_action", "wrong_state", "wrong_frame_index", "wrong_timestamp", "wrong_physics",
          "wrong_roi_binding", "wrong_elements_binding", "wrong_source_binding", "wrong_texture_binding",
          "wrong_render_binding", "wrong_atlas_binding", "changed_method_dependency", "training_eligibility_bypass"}


def checked_config():
    import yaml

    if yaml.safe_load(CONFIG_PATH.read_text()) != CONFIG or any(RENDER_CONFIG.get(key) != CONFIG[key] for key in [
            "frames", "fps", "width", "height", "table_body", "table_mesh", "table_visual_group",
            "texture_resize", "analytic_albedo_rgb", "matte_reflectance", "matte_shininess", "matte_specular",
            "composite_scope", "protected_pixels"]):
        raise ValueError("重渲染方法偏离独立源运行前声明，须新版本及新独立验证")
    for key in ["texture_sha256", "atlas_side", "repeats_per_axis", "tiling"]:
        if CONFIG[key] != ATLAS_CONFIG[key]:
            raise ValueError("图集偏离已见设计配方")
    if CONFIG["atlas_resize"] != ATLAS_CONFIG["resize"]:
        raise ValueError("图集缩块方法偏离设计")


def dependency_hashes():
    return {name: sha256(ROOT / name) for name in METHOD_FILES}


def checked_dependencies(probe):
    expected = dependency_hashes()
    if any(probe.get("source_files_sha256", {}).get(key) != value for key, value in expected.items()):
        raise ValueError("重渲染、重放或验收依赖已改变，不能借用旧方法验证")


def validate_report(report, role):
    if report.get("method") != CONFIG["version"] or report.get("role") != role \
            or report.get("frames") != list(range(93)) or report.get("validation_passed") is not True \
            or report.get("eligible_for_training") is not False \
            or set(report.get("negative_controls", {})) != NEGATIVES \
            or set(report.get("positive_controls", {})) != ERRORS \
            or any(row.get("passed") is not True or row.get("decoded_video_checked") is not True
                   for row in report["negative_controls"].values()) \
            or any(row.get("detected") is not True for row in report["positive_controls"].values()):
        raise ValueError("重渲染完整方法控制未通过；方法通过也不能等同于入训资格")


def checked_render(path, role):
    checked_config()
    probe = checked_artifact(path, "table_uv_replay")
    checked_dependencies(probe)
    report = json.loads(path.read_text())
    if report.get("method") != CONFIG["version"] or report.get("role") != role \
            or report.get("config_sha256") != sha256(CONFIG_PATH) or report.get("eligible_for_training") is not False:
        raise ValueError("渲染报告没有绑定固定方法、角色或配置")
    source = checked_bound_file(pathlib.Path(report["source_input"]), report["source_input_sha256"], ROOT / "runs/p0")
    evidence, frames, masks, proof = checked_training_source(source)
    if not evidence["source_episode_id"].endswith(CONFIG[role + "_episode_suffix"]):
        raise ValueError("成功源不符合运行前固定的设计/独立角色")
    if report["texture_binding"].get("atlas_sha256") != report["files_sha256"].get("cosmos_uv_atlas.png"):
        raise ValueError("实际图集没有绑定渲染来源")
    for name, checksum in report["files_sha256"].items():
        checked_bound_file(path.parent / name, checksum, path.parent)
    if max(report["replay_max_differences"].values()) != 0 \
            or report["render_state"]["restore_max_abs_diff"] != 0 \
            or report["render_state"]["integration_equal"] is not True \
            or report["render_state"]["texture_has_effect"] is not True:
        raise ValueError("源重放、视觉恢复或纹理实效未通过")
    return report, evidence, frames, masks, proof


def checked_controls(path, role):
    probe = checked_artifact(path, "table_uv_controls")
    checked_dependencies(probe)
    report = json.loads(path.read_text())
    validate_report(report, role)
    if report.get("config_sha256") != sha256(CONFIG_PATH):
        raise ValueError("控制配置已改变")
    render_path = checked_bound_file(pathlib.Path(report["render_report"]), report["render_report_sha256"], ROOT / "runs/p0")
    rendered, evidence, *_ = checked_render(render_path, role)
    if report["source_episode_id"] != rendered["source_episode_id"]:
        raise ValueError("控制源与实际渲染源不同")
    return report, evidence


def texture_source():
    directory = ROOT / "runs/p0" / CONFIG["texture_design_run"]
    report_path = directory / "texture_design.json"
    checked_artifact(report_path, "table_texture_design")
    report = json.loads(report_path.read_text())
    path = checked_bound_file(directory / "normalized_cosmos_texture.png", CONFIG["texture_sha256"], directory)
    checked_artifact(path, "table_texture_design")
    from p0_table_reference_materials import native_video
    native_probe = pathlib.Path(report["generation_probe"])
    checked_bound_file(native_probe, report["generation_probe_sha256"], ROOT / "runs/p0")
    probe, video = native_video(native_probe, "table_reference_generation")
    if probe["result"].get("guardrails_enabled") is not True:
        raise ValueError("纹理的原生Cosmos来源没有保留安全检查")
    return path, {"texture": str(path), "texture_sha256": sha256(path), "texture_report": str(report_path),
                  "texture_report_sha256": sha256(report_path), "native_probe": str(native_probe),
                  "native_probe_sha256": sha256(native_probe), "native_video": str(video), "native_video_sha256": sha256(video),
                  "derivation": "原生第0帧预定区域；局部亮度归一化、解析深棕基色、固定镜像图集及原场景UV重渲染"}


def render(directory, args):
    ensure_gpu_idle()
    checked_config()
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from p0_demo_replay import initialize_demo
    from p0_sim_data import configure_libero, observation_state
    from p0_single_control_materials import static_table_changes

    binding = None
    # 先检验设计，才读取独立源的第一张图像；不能用独立源调整纹理和边界。
    if args.role == "heldout":
        if args.design_report is None:
            raise ValueError("独立源需要已完成的设计控制报告")
        _, design = checked_controls(args.design_report, "design")
        binding = {"path": str(args.design_report), "sha256": sha256(args.design_report)}
    evidence, source, masks, proof = checked_training_source(args.source_input)
    if not evidence["source_episode_id"].endswith(CONFIG[args.role + "_episode_suffix"]):
        raise ValueError("禁止替换事先固定的设计/独立回合")
    if args.role == "heldout":
        binding["source_independence"] = check_independent_source(evidence, design)
    texture_path, texture_binding = texture_source()
    texture = mirrored_atlas(np.asarray(Image.open(texture_path).convert("RGB")),
                             CONFIG["atlas_side"], CONFIG["repeats_per_axis"])
    Image.fromarray(texture).save(directory / "cosmos_uv_atlas.png")
    texture_binding["atlas_sha256"] = sha256(directory / "cosmos_uv_atlas.png")
    declaration = {"method": CONFIG, "config_sha256": sha256(CONFIG_PATH), "dependencies": dependency_hashes(),
                   "source_input": str(args.source_input), "source_input_sha256": sha256(args.source_input),
                   "texture_binding": texture_binding, "role": args.role, "design_binding": binding}
    (directory / "declaration.json").write_text(json.dumps(declaration, ensure_ascii=False, indent=2) + "\n")
    cache_path = pathlib.Path(evidence["cache_manifest"])
    row = next(r for r in json.loads(cache_path.read_text())["episodes"] if r["episode_id"] == evidence["source_episode_id"])
    arrays = {name: np.load(cache_path.parent / (name + ".npy"), mmap_mode="r", allow_pickle=False)
              for name in ["images", "actions", "states"]}
    benchmark = configure_libero("LIBERO", directory / "libero_config")
    from libero.libero.envs import OffScreenRenderEnv
    env = OffScreenRenderEnv(bddl_file_name=str(benchmark / "bddl_files/libero_10" / row["bddl_file"]),
                             camera_heights=256, camera_widths=256, camera_segmentations="element")
    original_reset, attachment = env.reset, {}

    def observed_reset(*reset_args, **reset_kwargs):
        obs = original_reset(*reset_args, **reset_kwargs)
        model = env.sim.model
        geometries = [{"id": geom, "body": model.body_id2name(int(model.geom_bodyid[geom])),
                       "mesh": model.mesh_id2name(int(model.geom_dataid[geom])) if int(model.geom_type[geom]) == 7 else None,
                       "group": int(model.geom_group[geom]), "instance": env.env.model.geom_ids_to_instances.get(geom)}
                      for geom in range(model.ngeom)]
        table_id, = select_table_geometries(geometries)
        state, original_render = install_renderer(env, table_id, texture)
        attachment.update(state=state, original_render=original_render, sim=env.sim, table_id=table_id, geometries=geometries)
        return obs

    env.reset = observed_reset
    frames = {name: [] for name in ["analytic", "cosmos", "raw_cosmos"]}
    elements, physics, supervised = [], [], []
    errors = {"physics": 0., "state": 0., "rgb": 0, "wrist": 0, "task_roi_pixels": 0}
    first, stop = evidence["window"]["start_frame"], evidence["window"]["end_frame"]
    try:
        with np.load(pathlib.Path(row["directory"]) / "physical_states.npz", allow_pickle=False) as original_physics, \
                np.load(args.source_input.parent / "source_reset_snapshot.npz", allow_pickle=False) as snapshot:
            obs, _, _ = initialize_demo(env, row, original_physics, snapshot)
            groups = {"robot": [i for i, name in env.env.model.geom_ids_to_instances.items()
                                if "panda" in name.lower() or "gripper" in name.lower()],
                      **{name: [i for i, instance in env.env.model.geom_ids_to_instances.items() if instance == name]
                         for name in env.obj_of_interest}}
            if list(groups) != evidence["roi_names"] or env.sim is not attachment["sim"]:
                raise ValueError("重渲染改变了任务ROI或hard_reset语义")
            for index in range(stop):
                flat = row["offset"] + index
                physical = env.get_sim_state().copy()
                state_obs = observation_state(obs)
                errors["physics"] = max(errors["physics"], float(np.abs(physical - original_physics["states"][index]).max()))
                errors["state"] = max(errors["state"], float(np.abs(state_obs - arrays["states"][flat]).max()))
                wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                errors["wrist"] = max(errors["wrist"], int(np.abs(wrist.astype(np.int16) - arrays["images"][flat, 1].astype(np.int16)).max()))
                if index >= first:
                    at = index - first
                    rgb = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    last = attachment["state"]["last"]
                    if last is None or not np.array_equal(last["source"][::-1, ::-1], rgb):
                        raise ValueError("重渲染未绑定同一次RGB传感器观测")
                    errors["rgb"] = max(errors["rgb"], int(np.abs(rgb.astype(np.int16) - source[at].astype(np.int16)).max()))
                    segmentation = np.ascontiguousarray(obs["agentview_segmentation_element"][::-1, ::-1, 0]).astype(np.int32)
                    actual_roi = np.asarray([np.isin(segmentation, ids) for ids in groups.values()])
                    errors["task_roi_pixels"] += int(np.count_nonzero(actual_roi != masks[at]))
                    table = segmentation == attachment["table_id"]
                    for name in ["analytic", "cosmos"]:
                        rendered = np.ascontiguousarray(last[name][::-1, ::-1])
                        frames[name].append(protected_composite(rgb, rendered, table, masks[at]))
                        if name == "cosmos":
                            frames["raw_cosmos"].append(rendered)
                    elements.append(segmentation); physics.append(physical); supervised.append(state_obs)
                obs, _, _, _ = env.step(np.array(arrays["actions"][flat]))
    finally:
        env.reset = original_reset
        if attachment:
            attachment["sim"].render = attachment["original_render"]
        env.close()
    if max(errors.values()) != 0 or not attachment["state"]["texture_has_effect"]:
        raise ValueError(f"成功源物理、标签、双视角或纹理实效未通过：{errors}")
    decoded = {}
    for name, values in frames.items():
        path = directory / (name + ".mp4")
        write_video(path, values)
        decoded[name], _ = read_exact_video(path)
        if not np.array_equal(decoded[name], values):
            raise ValueError("无损输出编码没有保持全部实际像素")
    table = np.asarray(elements) == attachment["table_id"]
    for name, values in [("elements", elements), ("physics", physics), ("supervised", supervised)]:
        np.save(directory / (name + ".npy"), values, allow_pickle=False)
    shutil.copyfile(args.source_input.parent / "source_labels.npz", directory / "source_labels.npz")
    if sha256(directory / "source_labels.npz") != evidence["labels_sha256"]:
        raise ValueError("实际复制标签没有保持源字节")
    certificate = pixel_certificate(source, decoded["raw_cosmos"], table, decoded["cosmos"], masks)
    if not certificate["geometry_passed"]:
        raise ValueError("实际视频的非桌体保护或桌体同帧配方失败")
    reproduced = None
    if args.role == "design":
        reference_path = ROOT / "runs/p0" / CONFIG["design_reference_run"] / "atlas_design.json"
        checked_artifact(reference_path, "table_uv_atlas_design")
        reference = json.loads(reference_path.read_text())
        old_path = reference_path.parent / "cosmos.mp4"
        checked_artifact(old_path, "table_uv_atlas_design")
        old, _ = read_exact_video(old_path)
        if sha256(reference_path.parent / "cosmos_uv_atlas.png") != texture_binding["atlas_sha256"]:
            raise ValueError("固定设计图集未逐字节复现")
        reproduced = bool(reference["source_episode_id"] == evidence["source_episode_id"] and np.array_equal(old, decoded["cosmos"]))
        if not reproduced:
            raise ValueError("固定设计未逐像素复现，不能凭旧总览放行独立源")
    temporal = [{"kind": name, **value} for name, video in [("source", source), ("analytic", decoded["analytic"]), ("cosmos", decoded["cosmos"])]
                for value in static_table_changes(video, table.all(axis=0))]
    for filename, values in [("pixel_certificate.csv", certificate["per_frame"]), ("temporal.csv", temporal)]:
        with (directory / filename).open("x", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0])); writer.writeheader(); writer.writerows(values)
    font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 17, index=2)
    for page in range(8):
        indices = list(range(page * 12, min(93, page * 12 + 12)))
        sheet = Image.new("RGB", (768, 32 + len(indices) * 276), "#f6f4ee"); draw = ImageDraw.Draw(sheet)
        draw.text((4, 4), "成功源 / 解析纯色对照 / Cosmos图集重渲染；方法验证，不可入训", font=font, fill="#23343e")
        for slot, index in enumerate(indices):
            y = 32 + slot * 276
            draw.text((4, y), f"实际帧{index}", font=font, fill="#23343e")
            for column, video in enumerate([source, decoded["analytic"], decoded["cosmos"]]):
                sheet.paste(Image.fromarray(video[index]), (column * 256, y + 20))
        sheet.save(directory / f"review_{page:02d}.png")
    files = ["cosmos_uv_atlas.png", "analytic.mp4", "cosmos.mp4", "raw_cosmos.mp4", "elements.npy", "physics.npy", "supervised.npy", "source_labels.npz", "declaration.json"]
    report = {"method": CONFIG["version"], "config_sha256": sha256(CONFIG_PATH), "role": args.role,
              "source_input": str(args.source_input), "source_input_sha256": sha256(args.source_input),
              "source_episode_id": evidence["source_episode_id"], "source_proof": proof, "texture_binding": texture_binding,
              "design_binding": binding, "replay_max_differences": errors,
              "render_state": {key: value for key, value in attachment["state"].items() if key != "last"},
              "table_id": attachment["table_id"], "geometries": attachment["geometries"],
              "files_sha256": {name: sha256(directory / name) for name in files}, "design_reference_equal": reproduced,
              "pixel_certificate": certificate, "common_table_pixels": int(table.all(axis=0).sum()),
              "adjacent_mean_rgb_change_p95": {name: float(np.percentile([r["mean_abs_rgb_change_0_1"] for r in temporal if r["kind"] == name], 95))
                                               for name in ["source", "analytic", "cosmos"]},
              "output_kind": "background_composite", "derived_kind": "cosmos_mirrored_uv_atlas_sim_rerender",
              "geometry_evidence_type": "same_sensor_table_uv_render_pixels_v1", "eligible_for_training": False,
              "limitations": ["不同训练示范的方法验证，不是冻结策略评测或收益", "完整实际视频控制与视觉质量验收仍须分开", "原生Cosmos只提供纹理；颜色及重渲染为显式派生"]}
    (directory / "renderer_replay.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return {key: report[key] for key in ["role", "source_episode_id", "replay_max_differences", "design_reference_equal", "adjacent_mean_rgb_change_p95", "eligible_for_training"]}


def controls(directory, args):
    ensure_gpu_idle()
    checked_config()
    import cv2
    import imageio.v2 as imageio
    import numpy as np
    from p0_context_normalized_segmentation import object_shift

    report, evidence, source, masks, _ = checked_render(args.render_report, args.role)
    render_dir = args.render_report.parent
    elements = np.load(render_dir / "elements.npy", allow_pickle=False)
    table = elements == report["table_id"]
    actual, _ = read_exact_video(render_dir / "cosmos.mp4")
    raw, _ = read_exact_video(render_dir / "raw_cosmos.mp4")
    analytic, _ = read_exact_video(render_dir / "analytic.mp4")
    negatives, positives = {}, {}
    for name, native, output in [("identity", source, source), ("analytic_render", analytic, analytic), ("cosmos_render", raw, actual)]:
        path = directory / (name + ".mp4")
        write_video(path, output)
        decoded, _ = read_exact_video(path)
        certificate = pixel_certificate(source, native, table, decoded, masks)
        negatives[name] = {"passed": certificate["geometry_passed"], "decoded_video_checked": True, "video_sha256": sha256(path)}
    bad_protected, bad_table = actual.copy(), actual.copy()
    for at in range(93):
        for selected, video in [(~table[at], bad_protected), (table[at], bad_table)]:
            y, x = np.argwhere(selected)[0]; video[at, y, x, 0] ^= 1
    mug_ids = [r["id"] for r in report["geometries"] if any(w in str(r.get("instance", "")).lower() for w in ["mug", "cup"])]
    mug = np.isin(elements, mug_ids)
    if not mug.any() or np.any(mug & table):
        raise ValueError("杯子错误控制需要真实可见且受保护的杯子")
    bad_mug = actual.copy(); bad_mug[mug, 1] ^= 64
    swapped = actual.copy(); swapped[[0, 92]] = swapped[[92, 0]]
    cases = {"protected_pixel_changed": bad_protected, "table_recipe_changed": bad_table, "mug_recolored": bad_mug,
             "robot_shift12": np.asarray([object_shift(f, m.astype(bool), 12)[0] for f, m in zip(actual, masks[:, 0])]),
             "book_shift12": np.asarray([object_shift(f, m.astype(bool), 12)[0] for f, m in zip(actual, masks[:, 1])]),
             "global_shift24": np.roll(actual, 24, axis=2), "swap_first_last_frames": swapped}
    for name, output in cases.items():
        path = directory / (name + ".mp4"); write_video(path, output)
        decoded, _ = read_exact_video(path)
        certificate = pixel_certificate(source, raw, table, decoded, masks)
        errors = [r["frame"] for r in certificate["per_frame"] if not r["expected_composite_equal"]]
        required = [0, 92] if name == "swap_first_last_frames" else np.flatnonzero(mug.any(axis=(1, 2))).tolist() if name == "mug_recolored" else list(range(93))
        positives[name] = {"detected": not certificate["geometry_passed"] and set(required) <= set(errors),
                           "decoded_video_checked": True, "required_error_frames": required, "error_frames": errors,
                           "video_sha256": sha256(path)}
    for name, output, fps in [("wrong_fps", actual, 24), ("missing_frame", actual[:-1], 20),
                               ("wrong_size", np.asarray([cv2.resize(f, (128, 128)) for f in actual]), 20)]:
        path = directory / (name + ".mp4")
        with imageio.get_writer(path, fps=fps, codec="libx264rgb", pixelformat="rgb24", macro_block_size=1,
                                ffmpeg_params=["-crf", "0", "-threads", "2"]) as writer:
            for frame in output:
                writer.append_data(frame)
        try:
            read_exact_video(path)
        except ValueError as exc:
            positives[name] = {"detected": True, "error": str(exc), "video_sha256": sha256(path)}
        else:
            positives[name] = {"detected": False}
    with np.load(render_dir / "source_labels.npz", allow_pickle=False) as archive:
        labels = dict(archive)
    for key in ["action", "state", "frame_index", "timestamp"]:
        changed = {name: values.copy() for name, values in labels.items()}; changed[key].flat[0] += 1
        path = directory / ("wrong_" + key + ".npz"); np.savez_compressed(path, **changed)
        try:
            with np.load(path, allow_pickle=False) as archive:
                validate_labels(dict(archive), labels)
        except ValueError as exc:
            positives["wrong_" + key] = {"detected": True, "error": str(exc)}
        else:
            positives["wrong_" + key] = {"detected": False}
    physics = np.load(render_dir / "physics.npy", allow_pickle=False)
    changed = physics.copy(); changed.flat[0] += 1
    np.save(directory / "wrong_physics.npy", changed, allow_pickle=False)
    positives["wrong_physics"] = {"detected": not np.array_equal(np.load(directory / "wrong_physics.npy", allow_pickle=False), physics),
                                   "validator": "与真实源完整物理逐值比较，覆盖渲染实际重放中的比较语义"}
    for name, path, checksum in [
            ("wrong_roi_binding", render_dir / "elements.npy", evidence["files_sha256"]["roi_masks.npy"]),
            ("wrong_elements_binding", render_dir / "supervised.npy", report["files_sha256"]["elements.npy"]),
            ("wrong_source_binding", render_dir / "cosmos.mp4", evidence["files_sha256"]["source.mp4"]),
            ("wrong_texture_binding", render_dir / "cosmos.mp4", CONFIG["texture_sha256"]),
            ("wrong_atlas_binding", render_dir / "analytic.mp4", report["texture_binding"]["atlas_sha256"]),
            ("wrong_render_binding", render_dir / "analytic.mp4", report["files_sha256"]["raw_cosmos.mp4"])]:
        try:
            checked_bound_file(path, checksum, ROOT / "runs/p0")
        except ValueError as exc:
            positives[name] = {"detected": True, "error": str(exc)}
        else:
            positives[name] = {"detected": False}
    altered_probe = {"source_files_sha256": dependency_hashes()}
    altered_probe["source_files_sha256"][METHOD_FILES[0]] = "0" * 64
    try:
        checked_dependencies(altered_probe)
    except ValueError as exc:
        positives["changed_method_dependency"] = {"detected": True, "error": str(exc)}
    else:
        positives["changed_method_dependency"] = {"detected": False}
    positives["training_eligibility_bypass"] = {"detected": report["eligible_for_training"] is False and CONFIG["eligible_for_training"] is False,
                                               "limitation": "本阶段没有训练准入接口；不能用控制通过推断资格"}
    result = {"method": CONFIG["version"], "role": args.role, "config_sha256": sha256(CONFIG_PATH),
              "render_report": str(args.render_report), "render_report_sha256": sha256(args.render_report),
              "source_episode_id": evidence["source_episode_id"], "design_binding": report["design_binding"],
              "frames": list(range(93)), "negative_controls": negatives, "positive_controls": positives,
              "validation_passed": all(r["passed"] for r in negatives.values()) and all(r["detected"] for r in positives.values()),
              "eligible_for_training": False, "independent_policy_evaluations": 0,
              "limitation": "完整方法控制，不是合格训练窗口或策略收益；实际质量须独立全帧复核"}
    (directory / "renderer_controls.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    validate_report(result, args.role)
    return {key: result[key] for key in ["role", "source_episode_id", "validation_passed", "eligible_for_training"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=["render", "controls"])
    parser.add_argument("--role", choices=["design", "heldout"], required=True)
    parser.add_argument("--source-input", type=pathlib.Path)
    parser.add_argument("--render-report", type=pathlib.Path)
    parser.add_argument("--design-report", type=pathlib.Path)
    args = parser.parse_args()
    if (args.action == "render" and args.source_input is None) or (args.action == "controls" and args.render_report is None):
        parser.error("render需要--source-input；controls需要--render-report")
    for name, value in vars(args).items():
        if isinstance(value, pathlib.Path):
            value = value.resolve(strict=True)
            if not value.is_relative_to(ROOT / "runs/p0"):
                parser.error("所有输入必须位于Thor项目P0目录")
            setattr(args, name, value)
    stage, callback = ("table_uv_replay", render) if args.action == "render" else ("table_uv_controls", controls)
    raise SystemExit(run_stage(stage, lambda directory: callback(directory, args),
        {"method": CONFIG["version"], "role": args.role, "compute": "Thor CPU物理及EGL重渲染；无模型推理或训练",
         "precision": "RGB uint8/物理fp64/监督标签fp32", "batch": 1, "seed": 0, "frames": 93,
         "fps": 20, "training_steps": 0, "source_windows": 1}))
