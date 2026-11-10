#!/usr/bin/env python3
"""从本机成功D0重建机器人/目标物真值掩码，准备保留动作标签的Cosmos窗口。

@author pony
@date 2026-11-06
@version v1.5.0
@last_modified 2026-11-08
@changelog
    - v1.5.0 (2026-11-08): 支持精确选择独立成功训练回合，拒绝验证集、失败源和窗口越界。
    - v1.4.0 (2026-11-07): 在新生成之前显式声明全场暖暗目标条件，供独立分区域验收；不改历史来源。
    - v1.3.0 (2026-11-07): 恢复原始前序示范，保存固定物体与积分重置快照，严格核对双相机RGB。
    - v1.2.0 (2026-11-06): 局部mask实测产生背景幻觉后，支持全帧edge控制；ROI仍保留作独立检查。
    - v1.1.0 (2026-11-06): 使用与RGB同次传感器采样的element分割，避免动作期间的掩码时间错位。
    - v1.0.0 (2026-11-06): 使用原生mask_path约束机器人及目标物，保存源动作、状态与逐物体掩码。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

from p0_common import ROOT, ensure_gpu_idle, run_stage
from p0_demo_replay import initialize_demo
from p0_policy_runtime import sha256
from p0_sim_data import configure_libero, observation_state


def select_training_source(data: dict, task: str, start_frame: int, episode_id: str | None = None) -> dict:
    """显式指定回合时禁止退回首个示范，避免独立校准实际复用旧源。"""
    if data.get("all_accepted_episodes_env_success") is not True:
        raise ValueError("必须使用逐条本机回放成功的D0")
    if not isinstance(task, str) or not task.strip() or type(start_frame) is not int or start_frame < 0 \
            or (episode_id is not None and (not isinstance(episode_id, str) or not episode_id.strip())):
        raise ValueError("任务、回合ID或窗口起点无效")
    selected = [row for row in data["episodes"]
                if row.get("split") == "demo_train" and row.get("env_success") is True
                and row.get("is_eval", False) is False and task in row.get("bddl_file", "")
                and type(row.get("length")) is int and row["length"] >= start_frame + 93
                and (episode_id is None or row.get("episode_id") == episode_id)]
    if not selected or (episode_id is not None and len(selected) != 1):
        raise ValueError("没有唯一符合条件的指定成功训练源；禁止改用其他回合")
    return selected[0]


def prepare(run_dir: pathlib.Path, args: argparse.Namespace) -> dict:
    ensure_gpu_idle()
    import cv2
    import imageio.v2 as imageio
    import numpy as np

    manifest_path = args.cache / "manifest.json"
    data = json.loads(manifest_path.read_text())
    row = select_training_source(data, args.task, args.start_frame, getattr(args, "episode_id", None))
    provenance = pathlib.Path(row["directory"])
    physical = np.load(provenance / "physical_states.npz")
    images = np.load(args.cache / "images.npy", mmap_mode="r")
    states = np.load(args.cache / "states.npy", mmap_mode="r")
    actions = np.load(args.cache / "actions.npy", mmap_mode="r")
    benchmark = configure_libero("LIBERO", run_dir / "libero_config")
    from libero.libero.envs import OffScreenRenderEnv

    env = OffScreenRenderEnv(bddl_file_name=str(benchmark / "bddl_files/libero_10" / row["bddl_file"]),
                             camera_heights=256, camera_widths=256, camera_segmentations="element")
    try:
        obs, reset_snapshot, reset_metadata = initialize_demo(env, row, physical)
        np.savez_compressed(run_dir / "source_reset_snapshot.npz", **reset_snapshot)
        mapping = env.env.model.geom_ids_to_instances
        robot_ids = [key for key, name in mapping.items() if "panda" in name.lower() or "gripper" in name.lower()]
        interests = list(env.obj_of_interest)
        groups = {"robot": robot_ids, **{name: [key for key, instance in mapping.items() if instance == name] for name in interests}}
        if not interests or any(not values for values in groups.values()):
            raise RuntimeError(f"目标或机器人没有对应geom：{groups}")
        masks = np.empty((93, len(groups), 256, 256), dtype=np.uint8)
        max_physics_difference = max_rgb_difference = max_wrist_difference = max_label_difference = 0
        source_hashes = []
        first, last = args.start_frame, args.start_frame + 93
        with imageio.get_writer(run_dir / "source.mp4", fps=20, codec="libx264rgb", pixelformat="rgb24", ffmpeg_params=["-crf", "0"]) as source, \
             imageio.get_writer(run_dir / "edge.mp4", fps=20, codec="libx264rgb", pixelformat="rgb24", ffmpeg_params=["-crf", "0"]) as edge_video, \
             imageio.get_writer(run_dir / "mask.mp4", fps=20, codec="libx264rgb", pixelformat="rgb24", ffmpeg_params=["-crf", "0"]) as mask_video:
            for index in range(last):
                flat = row["offset"] + index
                actual = env.get_sim_state()
                max_physics_difference = max(max_physics_difference, float(np.max(np.abs(actual - physical["states"][index]))))
                max_label_difference = max(max_label_difference, float(np.max(np.abs(observation_state(obs) - states[flat]))))
                wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                max_wrist_difference = max(max_wrist_difference, int(np.max(np.abs(
                    wrist.astype(np.int16) - images[flat, 1].astype(np.int16)))))
                if index >= first:
                    image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    difference = int(np.max(np.abs(image.astype(np.int16) - images[flat, 0].astype(np.int16))))
                    max_rgb_difference = max(max_rgb_difference, difference)
                    # robosuite相机在仿真子步采样；即时render可能晚于缓存RGB，必须取同一观测里的分割。
                    segmentation = obs["agentview_segmentation_element"][::-1, ::-1, 0]
                    for group_index, ids in enumerate(groups.values()):
                        masks[index - first, group_index] = np.isin(segmentation, ids)
                    combined = cv2.dilate(masks[index - first].any(axis=0).astype(np.uint8), np.ones((5, 5), np.uint8))
                    frame = np.array(images[flat, 0])
                    edge = cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), 80, 160)
                    source.append_data(frame)
                    edge_video.append_data(np.repeat(edge[..., None], 3, axis=-1))
                    mask_video.append_data(np.repeat((combined * 255)[..., None], 3, axis=-1))
                    source_hashes.append(hashlib.sha256(frame.tobytes()).hexdigest())
                    if index - first in [0, 46, 92]:
                        imageio.imwrite(run_dir / f"source_{index-first:03d}.png", frame)
                        imageio.imwrite(run_dir / f"mask_{index-first:03d}.png", combined * 255)
                obs, _, _, _ = env.step(np.array(actions[flat]))
        if max_physics_difference > 1e-9 or max(max_rgb_difference, max_wrist_difference) > 1 or max_label_difference > 1e-6:
            raise RuntimeError(f"生成源重放未精确复现：physics={max_physics_difference}, RGB={max_rgb_difference}, wrist={max_wrist_difference}, state={max_label_difference}")
        if any(not masks[:, index].any() for index in range(len(groups))):
            raise RuntimeError("目标或机器人在整个窗口没有可见掩码")
        np.save(run_dir / "roi_masks.npy", masks)
        start = row["offset"] + first
        stop = row["offset"] + last
        np.savez_compressed(run_dir / "source_labels.npz", action=np.array(actions[start:stop]), state=np.array(states[start:stop]),
                            frame_index=np.arange(first, last), timestamp=np.arange(first, last) / 20)
        prompt = ("A fixed camera records the same Panda robot arm performing the same manipulation task. "
                  f"The task is: {row['instruction']}. "
                  "Preserve the exact robot, gripper, every object identity, color, shape, position, geometry and motion. "
                  "Change only the illumination to warm dim indoor lighting. Keep the objects clearly visible. "
                  "Do not add, remove, replace or recolor any object. The camera remains fixed.")
        spec = {"name": "p0_verified_full_edge_warm_dim" if args.full_frame_edge else "p0_verified_masked_warm_dim",
                "prompt": prompt, "video_path": str(run_dir / "source.mp4"),
                "max_frames": 93, "num_video_frames_per_chunk": 93, "num_steps": 4, "seed": 42,
                "resolution": "480", "keep_input_resolution": False,
                "edge": {"control_path": str(run_dir / "edge.mp4"), "control_weight": 1.0,
                         **({} if args.full_frame_edge else {"mask_path": str(run_dir / "mask.mp4")})}}
        (run_dir / "spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n")
        result = {"kind": "thor_verified_demo_window", "source_episode_id": row["episode_id"], "source_kind": "demo",
                  "source_selection": {"requested_episode_id": getattr(args, "episode_id", None),
                                       "selected_episode_id": row["episode_id"], "split": row["split"]},
                  "source_success": True, "source_success_source": row["source_success"],
                  "cache_manifest": str(manifest_path), "cache_manifest_sha256": sha256(manifest_path),
                  "instruction": row["instruction"], "window": {"start_frame": first, "end_frame": last, "phase": "pre_grasp"},
                  "target_condition": {"light": "warm_dim_indoor"},
                  "target_condition_scope": "全场光照：机器人、任务目标物及背景均须独立验收；不是仅背景改写",
                  "view": "agentview", "fps": 20, "frames": 93, "temporal_resampling": "none",
                  "roi_names": list(groups), "mask_source": "sim_seg",
                  "control_mask": "全帧edge，无控制mask；真值ROI仅用于质量检查" if args.full_frame_edge
                                  else "robot+target_objects，向外膨胀2px",
                  "mask_temporal_alignment": "RGB与element分割来自同次20Hz观测采样，无额外render或时间平移",
                  "replay_validation": {"passed": True, "physics_max_abs_diff": max_physics_difference,
                                        "rgb_max_abs_diff": max_rgb_difference, "wrist_max_abs_diff": max_wrist_difference,
                                        "state_max_abs_diff": max_label_difference},
                  "reset_protocol": reset_metadata, "source_reset_snapshot": "source_reset_snapshot.npz",
                  "source_rgb_sha256": source_hashes, "spec": str(run_dir / "spec.json"),
                  "labels_sha256": sha256(run_dir / "source_labels.npz"),
                  "files_sha256": {name: sha256(run_dir / name) for name in ["source.mp4", "edge.mp4", "mask.mp4", "roi_masks.npy", "source_reset_snapshot.npz"]},
                  "eligible_for_training": False, "reason": "仅源轨迹和掩码通过，生成结果须另做ROI/时序/语义检查",
                  "precision": "MuJoCo CPU、RGB uint8、geom真值掩码uint8、动作/状态fp32", "seed": 0, "batch": 1,
                  "sample_count": 1, "steps": last, "real_robot_commands_sent": False}
        (run_dir / "verified_input.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        return result
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", type=pathlib.Path, required=True)
    parser.add_argument("--task", default="STUDY_SCENE1")
    parser.add_argument("--episode-id", help="精确选择成功demo_train源，不匹配时立即退出")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--full-frame-edge", action="store_true", help="用全帧边缘锚定场景，仍保存逐目标真值ROI")
    arguments = parser.parse_args()
    arguments.cache = arguments.cache.resolve(strict=True)
    if not arguments.cache.is_relative_to(ROOT / "data") or arguments.start_frame < 0:
        parser.error("缓存必须位于项目data目录，帧起点不能为负")
    raise SystemExit(run_stage("verified_transfer_input", lambda directory: prepare(directory, arguments),
                              {key: str(value) if isinstance(value, pathlib.Path) else value for key, value in vars(arguments).items()}))
