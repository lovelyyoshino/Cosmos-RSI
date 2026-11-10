#!/usr/bin/env python3
"""完整重放一个成功训练源，预声明仅桌体可见区域可改写的四步Cosmos试验。

@author pony
@date 2026-11-08
@version v2.1.0
@last_modified 2026-11-08
@changelog
    - v2.1.0 (2026-11-08): 准备入口记录新版依赖；修复后另用预声明独立源，不追用旧控制。
    - v2.0.0 (2026-11-08): 同次观测重建全部element分割，保护非任务物体并固定目标与控制掩码。
"""
from __future__ import annotations

import argparse
import copy
import json
import pathlib
import shutil

from p0_common import ROOT, ensure_gpu_idle, run_stage
from p0_demo_replay import initialize_demo
from p0_lossless_background import checked_training_source, read_exact_video
from p0_policy_runtime import sha256
from p0_sim_data import configure_libero, observation_state
from p0_table_background import CONFIG, CONFIG_PATH, RECIPE, checked_config, editable_table, select_table_geometries
from p0_transfer_smoke import validate_generation_spec


def write_video(path, frames):
    import imageio.v2 as imageio

    with imageio.get_writer(path, fps=20, codec="libx264rgb", pixelformat="rgb24", macro_block_size=1,
                           ffmpeg_params=["-crf", "0", "-preset", "veryfast", "-threads", "2"]) as writer:
        for frame in frames:
            writer.append_data(frame)


def prepare(directory: pathlib.Path, args: argparse.Namespace) -> dict:
    ensure_gpu_idle()
    checked_config()
    import numpy as np
    from PIL import Image

    if args.role == "heldout":
        from p0_table_background_controls import checked_design
        if args.design_report is None:
            raise ValueError("独立源的新桌面准备必须先绑定已通过的完整设计结果")
        checked_design(args.design_report)
    evidence, source, masks, _ = checked_training_source(args.source_input)
    parent = json.loads((args.source_input.parent / "probe.json").read_text())
    if parent.get("stage") != "verified_transfer_input" or not evidence["source_episode_id"].endswith(
            CONFIG[args.role + "_episode_suffix"]):
        raise ValueError("只接受预声明设计/独立成功训练源的完整原始重放，不复用派生源")
    cache_path = pathlib.Path(evidence["cache_manifest"])
    cache = json.loads(cache_path.read_text())
    row = next(r for r in cache["episodes"] if r["episode_id"] == evidence["source_episode_id"])
    for name, checksum in {**evidence["files_sha256"], "source_labels.npz": evidence["labels_sha256"]}.items():
        shutil.copyfile(args.source_input.parent / name, directory / name)
        if sha256(directory / name) != checksum:
            raise ValueError("复制后的成功源字节不一致")
    benchmark = configure_libero("LIBERO", directory / "libero_config")
    from libero.libero.envs import OffScreenRenderEnv

    env = OffScreenRenderEnv(bddl_file_name=str(benchmark / "bddl_files/libero_10" / row["bddl_file"]),
                             camera_heights=256, camera_widths=256, camera_segmentations="element")
    elements = []
    errors = {"physics": 0., "state": 0., "rgb": 0, "wrist": 0, "task_roi_pixels": 0}
    images = np.load(cache_path.parent / "images.npy", mmap_mode="r", allow_pickle=False)
    actions = np.load(cache_path.parent / "actions.npy", mmap_mode="r", allow_pickle=False)
    states = np.load(cache_path.parent / "states.npy", mmap_mode="r", allow_pickle=False)
    first, stop = evidence["window"]["start_frame"], evidence["window"]["end_frame"]
    try:
        with np.load(pathlib.Path(row["directory"]) / "physical_states.npz", allow_pickle=False) as physical, \
             np.load(directory / "source_reset_snapshot.npz", allow_pickle=False) as snapshot:
            obs, _, _ = initialize_demo(env, row, physical, snapshot)
            model = env.sim.model
            instances = env.env.model.geom_ids_to_instances
            geometries = []
            for geom in range(model.ngeom):
                mesh = int(model.geom_dataid[geom])
                geometries.append({"id": geom, "name": model.geom_id2name(geom),
                    "body": model.body_id2name(int(model.geom_bodyid[geom])), "group": int(model.geom_group[geom]),
                    "mesh": model.mesh_id2name(mesh) if int(model.geom_type[geom]) == 7 and mesh >= 0 else None,
                    "instance": instances.get(geom)})
            table_ids = select_table_geometries(geometries)
            groups = {"robot": [i for i, name in instances.items() if "panda" in name.lower() or "gripper" in name.lower()],
                      **{name: [i for i, instance in instances.items() if instance == name] for name in env.obj_of_interest}}
            if list(groups) != evidence["roi_names"]:
                raise ValueError("桌面新重放改变了原任务ROI定义")
            for index in range(stop):
                flat = row["offset"] + index
                errors["physics"] = max(errors["physics"], float(np.max(np.abs(env.get_sim_state() - physical["states"][index]))))
                errors["state"] = max(errors["state"], float(np.max(np.abs(observation_state(obs) - states[flat]))))
                wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                errors["wrist"] = max(errors["wrist"], int(np.max(np.abs(wrist.astype(np.int16) - images[flat, 1].astype(np.int16)))))
                if index >= first:
                    at = index - first
                    rgb = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    errors["rgb"] = max(errors["rgb"], int(np.max(np.abs(rgb.astype(np.int16) - source[at].astype(np.int16)))))
                    segmentation = np.ascontiguousarray(obs["agentview_segmentation_element"][::-1, ::-1, 0])
                    actual_roi = np.asarray([np.isin(segmentation, ids) for ids in groups.values()])
                    errors["task_roi_pixels"] += int(np.count_nonzero(actual_roi != masks[at]))
                    elements.append(segmentation.astype(np.int32))
                obs, _, _, _ = env.step(np.array(actions[flat]))
    finally:
        env.close()
    if errors["physics"] > 1e-9 or errors["state"] > 1e-6 or max(errors["rgb"], errors["wrist"]) > 0 \
            or errors["task_roi_pixels"]:
        raise ValueError(f"桌面分割未复现同一源RGB、ROI、物理或监督标签：{errors}")
    elements = np.asarray(elements)
    editable = editable_table(elements, table_ids, masks)
    np.save(directory / "table_elements.npy", elements, allow_pickle=False)
    np.save(directory / "editable_table.npy", editable, allow_pickle=False)
    controls = np.repeat((~editable)[..., None], 3, axis=-1).astype(np.uint8) * 255
    write_video(directory / "scene_mask.mp4", controls)
    decoded, _ = read_exact_video(directory / "scene_mask.mp4")
    if not np.array_equal(decoded, controls):
        raise ValueError("实际非桌体控制视频不是无损掩码")
    spec = copy.deepcopy(json.loads((args.source_input.parent / "spec.json").read_text()))
    validate_generation_spec(spec, "edge/distilled", 4)
    spec.update({"name": "p0_table_only_matte_dark_wood_v2", "video_path": str(directory / "source.mp4"),
        "num_steps": 4, "seed": 42, "resolution": "480",
        "edge": {"control_path": str(directory / "edge.mp4"), "control_weight": 1.0,
                 "mask_path": str(directory / "scene_mask.mp4")},
        "prompt": "A fixed camera records the same robot performing the same task: " + evidence["instruction"]
        + ". Only the tabletop surface is dark brown matte wood with subtle straight wooden grain. "
        "The tabletop is uniformly dark brown, flat, non-reflective, without colored patterns. "
        "Keep the original lighting, every object, the mug, the plant, the robot and the walls unchanged. "
        "No new objects, no obstacles, no ripples, no metallic reflections, no text. The camera stays fixed."})
    validate_generation_spec(spec, "edge/distilled", 4)
    evidence.update({"spec": str(directory / "spec.json"), "parent_verified_input": str(args.source_input),
        "parent_verified_input_sha256": sha256(args.source_input), "target_condition": CONFIG["target_condition"],
        "target_condition_scope": "仅同帧study_table/desk_vis可见桌体内部；墙面及全部其他物体保持源RGB，不是全场光照增强",
        "composite_recipe": RECIPE, "control_recipe": "table-only-edge-distilled-v2",
        "table_geometry": {"method_config_sha256": sha256(CONFIG_PATH), "table_geom_ids": table_ids,
            "geometries": geometries, "same_observation_as_source_rgb": True, "replay_passed": True,
            "replay_max_differences": errors, "editable_pixels_per_frame": [int(f.sum()) for f in editable]},
        "files_sha256": {**evidence["files_sha256"], **{name: sha256(directory / name)
            for name in ["table_elements.npy", "editable_table.npy", "scene_mask.mp4"]}},
        "predeclared_experiment": {"model": "edge/distilled", "num_steps": 4, "seed": 42,
            "resolution": "480", "batch": 1, "frames": 93, "fps": 20, "role": args.role,
            "comparison_claim": "新配方单窗口工程验证，非与v1的单因素比较，非策略收益"},
        "eligible_for_training": False, "reason": "仅源与新桌面范围通过；方法控制及真实生成的完整质量门待验收"})
    for name, data in [("spec.json", spec), ("verified_input.json", evidence)]:
        (directory / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    for index in [0, 46, 92]:
        preview = source[index].copy()
        preview[editable[index]] = (preview[editable[index]].astype(np.uint16) // 2 + np.array([0, 100, 0])).clip(0, 255).astype(np.uint8)
        Image.fromarray(preview).save(directory / f"editable_table_{index:03d}.png")
    return {"source_input": str(directory / "verified_input.json"), "spec": str(directory / "spec.json"),
            "source_episode_id": evidence["source_episode_id"], "role": args.role, "table_geom_ids": table_ids,
            "replay_max_differences": errors, "editable_pixels_min": min(int(f.sum()) for f in editable),
            "eligible_for_training": False, "precision": "MuJoCo CPU物理；EGL同帧RGB uint8/element int32；动作状态fp32",
            "batch": 1, "seed": 0, "steps": stop, "sample_count": 1}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-input", type=pathlib.Path, required=True)
    parser.add_argument("--role", choices=["design", "heldout"], required=True)
    parser.add_argument("--design-report", type=pathlib.Path)
    args = parser.parse_args()
    args.source_input = args.source_input.resolve(strict=True)
    if not args.source_input.is_relative_to(ROOT / "runs/p0"):
        parser.error("源必须位于本项目P0目录")
    if args.design_report is not None:
        args.design_report = args.design_report.resolve(strict=True)
        if not args.design_report.is_relative_to(ROOT / "runs/p0"):
            parser.error("设计报告必须位于本项目P0目录")
    raise SystemExit(run_stage("background_transfer_input", lambda directory: prepare(directory, args),
                               {"method": CONFIG["version"], "source_input": str(args.source_input), "role": args.role}))
