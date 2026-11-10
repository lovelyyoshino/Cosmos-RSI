#!/usr/bin/env python3
"""把映射出的原始完整状态恢复到MuJoCo，核对缓存观测的时间偏移。

@author pony
@date 2026-11-06
@version v1.3.0
@last_modified 2026-11-06
@changelog
    - v1.3.0 (2026-11-06): 复刻OFT的seed0、BDDL、十步稳定与no-op过滤，核对整段缓存观测。
    - v1.2.0 (2026-11-06): 可选完整示范动作回放，记录本机新观测和环境成功真值。
    - v1.1.0 (2026-11-06): 解析旧版chiliocosm资源路径，仅映射到项目已固定的assets。
    - v1.0.0 (2026-11-06): 最小单回合、多偏移的物理状态恢复，不发送真机动作。
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import xml.etree.ElementTree as ET

from p0_common import ROOT, ensure_gpu_idle, run_stage


def localize_assets(xml: str, asset_root: pathlib.Path) -> str:
    document = ET.fromstring(xml)
    for element in document.iter():
        original = element.get("file", "")
        for prefix in ["/chiliocosm/assets/", "/libero/assets/"]:
            if prefix not in original:
                continue
            relative = original.split(prefix, 1)[1]
            target = (asset_root / relative).resolve(strict=True)
            if not target.is_relative_to(asset_root.resolve()):
                raise RuntimeError("旧XML资源路径越过项目assets边界")
            element.set("file", str(target))
            break
    return ET.tostring(document, encoding="unicode")


def align(run_dir: pathlib.Path, mapping_path: pathlib.Path, replay_actions: bool = False, official_regeneration: bool = False) -> dict:
    ensure_gpu_idle()
    import h5py
    import imageio.v2 as imageio
    import numpy as np
    from scipy.spatial.transform import Rotation

    mapping = json.loads(mapping_path.read_text())
    item = mapping["mapped"][0]
    cache_path = pathlib.Path(mapping["cache_manifest"]).parent
    cache = json.loads((cache_path / "manifest.json").read_text())
    episode = next(e for e in cache["episodes"] if e["episode_id"] == item["episode_id"])
    states = np.load(cache_path / "states.npy", mmap_mode="r")
    images = np.load(cache_path / "images.npy", mmap_mode="r")
    repo = ROOT / "third_party/LIBERO"
    benchmark_root = repo / "libero/libero"
    config_dir = run_dir / "libero_config"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text(json.dumps({
        "benchmark_root": str(benchmark_root), "bddl_files": str(benchmark_root / "bddl_files"),
        "init_states": str(benchmark_root / "init_files"), "datasets": str(ROOT / "data/LIBERO"),
        "assets": str(benchmark_root / "assets")}) + "\n")
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    sys.path.insert(0, str(repo))
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero.utils.utils import postprocess_model_xml
    from robosuite.utils.transform_utils import quat2axisangle

    bddl = benchmark_root / "bddl_files/libero_10" / cache["tasks"][item["instruction"]]
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    observations = []
    try:
        env.seed(42)
        env.reset()
        with h5py.File(item["raw_file"], "r") as handle:
            demo = handle["data"][item["raw_demo_id"]]
            xml = localize_assets(postprocess_model_xml(demo.attrs["model_file"], {}), benchmark_root / "assets")
            (run_dir / "localized_model.xml").write_text(xml)
            env.reset_from_xml_string(xml)
            env.sim.reset()
            raw_states = demo["states"][:]
            raw_actions = demo["actions"][:]
        for shift in [-1, 0, 1]:
            xyz, angles, grip, rgb = [], [], [], []
            for index in range(8):
                raw_index = item["raw_start_frame"] + index + shift
                if not 0 <= raw_index < len(raw_states):
                    continue
                obs = env.regenerate_obs_from_state(raw_states[raw_index])
                actual = np.concatenate([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"].copy()),
                                         obs["robot0_gripper_qpos"]])
                target = states[episode["offset"] + index]
                image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                xyz.append(float(np.linalg.norm(actual[:3] - target[:3])))
                angles.append(float((Rotation.from_rotvec(actual[3:6]).inv() * Rotation.from_rotvec(target[3:6])).magnitude()))
                grip.append(float(np.max(np.abs(actual[6:] - target[6:]))))
                rgb.append(float(np.sqrt(np.mean((image.astype(np.float32) - images[episode["offset"] + index, 0]) ** 2))))
                if index in [0, 7]:
                    imageio.imwrite(run_dir / f"restored_shift{shift}_frame{index}.png", image)
                    imageio.imwrite(run_dir / f"source_frame{index}.png", images[episode["offset"] + index, 0])
            observations.append({"raw_frame_shift": shift, "sampled_frames": len(xyz),
                                 "xyz_max_error_m": max(xyz), "xyz_mean_error_m": float(np.mean(xyz)),
                                 "rotation_max_error_rad": max(angles), "gripper_max_error": max(grip),
                                 "rgb_rmse_mean": float(np.mean(rgb))})
        best = min(observations, key=lambda row: row["xyz_mean_error_m"])
        replay = {"performed": False}
        if replay_actions:
            import cv2

            if official_regeneration:
                # 官方再生按BDDL重建，不能沿用前面对原始XML做的恢复。
                env.close()
                env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
                env.seed(0)
                env.reset()
                obs = env.set_init_state(raw_states[0])
                for _ in range(10):
                    obs, _, _, _ = env.step(np.asarray([0, 0, 0, 0, 0, 0, -1]))
                action_start = 0
            else:
                env.reset_from_xml_string(xml)
                env.sim.reset()
                obs = env.regenerate_obs_from_state(raw_states[item["raw_start_frame"]])
                action_start = item["raw_start_frame"]
            labels, physical_states = [], []
            executed_actions, raw_indices = [], []
            with imageio.get_writer(run_dir / "replay_agentview.mp4", fps=20, codec="libx264") as writer, \
                 imageio.get_writer(run_dir / "source.mp4", fps=20, codec="libx264") as source_writer, \
                 imageio.get_writer(run_dir / "edge.mp4", fps=20, codec="libx264rgb", pixelformat="rgb24", ffmpeg_params=["-crf", "0"]) as edge_writer:
                for raw_index in range(action_start, len(raw_actions)):
                    action = raw_actions[raw_index]
                    if official_regeneration and np.linalg.norm(action[:-1]) < 1e-4:
                        if not executed_actions or action[-1] == executed_actions[-1][-1]:
                            continue
                    index = len(labels)
                    image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    state = np.concatenate([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"].copy()), obs["robot0_gripper_qpos"]])
                    labels.append(state)
                    physical_states.append(env.get_sim_state())
                    executed_actions.append(action)
                    raw_indices.append(raw_index)
                    writer.append_data(image)
                    if index < 93:
                        source_writer.append_data(image)
                        edge = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), 80, 160)
                        edge_writer.append_data(np.repeat(edge[..., None], 3, axis=-1))
                    obs, _, _, _ = env.step(action)
            success = bool(env.check_success())
            imageio.imwrite(run_dir / "replay_last.png", np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
            np.savez_compressed(run_dir / "replay_labels.npz", state=np.asarray(labels, dtype=np.float32),
                                action=np.asarray(executed_actions, dtype=np.float32), physical_state=np.asarray(physical_states),
                                raw_action_index=np.asarray(raw_indices))
            expected_actions = np.asarray(np.load(cache_path / "actions.npy", mmap_mode="r")[episode["offset"]:episode["offset"] + episode["length"]])
            observed_actions = np.asarray(executed_actions, dtype=np.float32)
            action_match = np.array_equal(observed_actions, expected_actions)
            state_comparison = {"performed": False}
            if action_match:
                target = np.asarray(states[episode["offset"]:episode["offset"] + episode["length"]])
                actual = np.asarray(labels)
                distances = np.linalg.norm(actual[:, :3] - target[:, :3], axis=1)
                rotations = (Rotation.from_rotvec(actual[:, 3:6]).inv() * Rotation.from_rotvec(target[:, 3:6])).magnitude()
                state_comparison = {"performed": True, "frames": len(labels), "xyz_max_error_m": float(distances.max()),
                                    "xyz_mean_error_m": float(distances.mean()), "rotation_max_error_rad": float(rotations.max()),
                                    "gripper_max_abs_error": float(np.max(np.abs(actual[:, 6:] - target[:, 6:]))),
                                    "passed": bool(distances.max() < 1e-4 and rotations.max() < 1e-3
                                                   and np.max(np.abs(actual[:, 6:] - target[:, 6:])) < 1e-4)}
            replay = {"performed": True, "source": "官方示范动作在本机重新执行", "env_success": success,
                      "steps": len(labels), "control_hz": 20, "action_repeat": 1, "settle_steps": 10 if official_regeneration else 0,
                      "seed": 0 if official_regeneration else 42, "official_regeneration": official_regeneration,
                      "official_reference": "openvla-oft@e4287e94541f459edc4feabc4e181f537cd569a8/experiments/robot/libero/regenerate_libero_dataset.py" if official_regeneration else None,
                      "full_action_match": action_match, "cache_state_comparison": state_comparison,
                      "labels": str(run_dir / "replay_labels.npz"), "source_window_frames": min(93, len(labels)),
                      "training_eligible": False, "reason": "尚未构建带冻结集校验的正式数据缓存"}
            if success and len(labels) >= 93:
                spec = {"name": "p0_verified_caddy_warm_dim", "prompt": "A fixed camera records a Panda robot arm picking up a book and placing it in the back compartment of a caddy. Preserve the exact robot, gripper, book, caddy, object identities, positions, geometry and motion. Change only the illumination to warm dim indoor lighting. Keep all objects clearly visible.",
                        "video_path": str(run_dir / "source.mp4"), "max_frames": 93, "num_video_frames_per_chunk": 93,
                        "num_steps": 4, "seed": 42, "resolution": "480", "keep_input_resolution": False,
                        "edge": {"control_path": str(run_dir / "edge.mp4"), "control_weight": 1.0}}
                (run_dir / "spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n")
                replay["generation_spec"] = str(run_dir / "spec.json")
        result = {"mapping": str(mapping_path), "episode_id": item["episode_id"], "raw_demo_id": item["raw_demo_id"],
                  "comparisons": observations, "best_xyz_shift": best["raw_frame_shift"],
                  "replay": replay,
                  "physical_alignment_passed": replay.get("cache_state_comparison", {}).get("passed", False) if official_regeneration else best["xyz_max_error_m"] < 1e-4 and best["rotation_max_error_rad"] < 1e-3,
                  "eligible_for_training": False, "evaluation_gate_passed": False,
                  "boundary": "仅一条示范前8帧的时间对齐诊断，不代表全部数据或冻结初态验收",
                  "precision": "MuJoCo CPU、完整状态fp64、RGB uint8", "seed": 42, "batch": 1,
                  "steps": replay.get("steps", 0), "sample_count": 1, "real_robot_commands_sent": False}
        (run_dir / "alignment.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        return result
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mapping", type=pathlib.Path, required=True)
    parser.add_argument("--replay-actions", action="store_true")
    parser.add_argument("--official-regeneration", action="store_true")
    args = parser.parse_args()
    args.mapping = args.mapping.resolve(strict=True)
    raise SystemExit(run_stage("raw_libero_alignment", lambda directory: align(directory, args.mapping, args.replay_actions or args.official_regeneration, args.official_regeneration),
                              {"mapping": str(args.mapping), "replay_actions": args.replay_actions, "official_regeneration": args.official_regeneration,
                               "scope": "仅MuJoCo状态恢复及示范回放"}))
