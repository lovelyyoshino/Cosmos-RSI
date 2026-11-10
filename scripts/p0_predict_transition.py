#!/usr/bin/env python3
"""从独立真实运行重放同一状态的动作分支，为Predict准备有物理真值的输入。

@author pony
@date 2026-11-09
@version v1.2.0
@last_modified 2026-11-09
@changelog
    - v1.2.0 (2026-11-09): 压缩轨迹的必要数组仅解码一次，保持全部逐帧及终态等价检查。
    - v1.1.0 (2026-11-09): 最后十六步使用封存终帧及完整仿真状态核对，不伪造缺失终态。
    - v1.0.0 (2026-11-09): 连续十六步、原始九维四元数状态及停止动作对照；不赋予教师数据资格。
"""
from __future__ import annotations

import argparse
import pathlib

from p0_common import ROOT, ensure_gpu_idle, run_stage
from p0_feedback_judge import checked_collection
from p0_lerobot_handoff import dump
from p0_policy_runtime import sha256
from p0_sim_data import configure_libero, observation_state, restore_fixture_state, restore_integration_state


def action_window(first, count):
    """允许最后一个动作后的真实终态；它从封存终帧读取，不能用末帧重复补齐。"""
    if type(first) is not int or type(count) is not int or first < 0 or first + 16 > count:
        raise ValueError("需要完整十六步动作及执行后的真实观测，不允许补帧")
    return first, first + 16


def replay_arrays(archive):
    """NPZ按成员懒解压；逐帧索引前缓存必要数组，避免反复解压整段双相机视频。"""
    return {name: archive[name] for name in ["action", "agentview", "wrist", "state", "final_sim_state"]}


def terminal_reference(row, probe, collection_root):
    import numpy as np
    from PIL import Image

    images = []
    for view in ["agentview", "wrist"]:
        path = (pathlib.Path(row["artifacts"]) / f"{view}_last.png").resolve(strict=True)
        if not path.is_relative_to(collection_root) \
                or probe["artifacts_sha256"].get(str(path.relative_to(collection_root))) != sha256(path):
            raise ValueError("Predict终态参考图未绑定原始采集")
        with Image.open(path) as image:
            if image.mode != "RGB" or image.size != (256, 256):
                raise ValueError("Predict终态参考图格式不符")
            images.append(np.asarray(image).copy())
    return images


def replay(directory, args):
    ensure_gpu_idle()
    import numpy as np
    from PIL import Image

    source, episodes, probe = checked_collection(args.collection)
    candidates = [row for row in episodes if row["episode_id"] == args.episode_id] if args.episode_id else \
        sorted((row for row in episodes if row["implementation"] == "LIBERO"), key=lambda row: row["episode_id"])
    if not candidates:
        raise ValueError("找不到指定的独立采集回合")
    row = candidates[0]
    trajectory = pathlib.Path(row["trajectory"]).resolve(strict=True)
    if not trajectory.is_relative_to(args.collection.parent) or sha256(trajectory) != row["trajectory_sha256"]:
        raise ValueError("Predict转移源未绑定实际独立采集轨迹")
    initial = pathlib.Path(row["artifacts"]) / "initial_state.npz"
    relative = str(initial.relative_to(args.collection.parent))
    if probe["artifacts_sha256"].get(relative) != sha256(initial):
        raise ValueError("转移初态快照已改变")
    spec_path = args.collection.parent / row["implementation"] / "worker_spec.json"
    import json
    spec = json.loads(spec_path.read_text())
    condition = next(item for item in spec["conditions"] if item["condition_id"] == row["condition_id"])
    if sha256(pathlib.Path(condition["bddl_path"])) != condition["bddl_sha256"]:
        raise ValueError("转移场景已改变")
    configure_libero(row["implementation"], directory / "libero_config")
    from libero.libero.envs import OffScreenRenderEnv

    branches = []
    with np.load(trajectory, allow_pickle=False) as archive, np.load(initial, allow_pickle=False) as snapshot:
        stored = replay_arrays(archive)
        count = len(stored["action"])
        first, stop = action_window(args.frame, count)
        terminal_images = terminal_reference(row, probe, args.collection.parent) if stop == count else None
        actual = stored["action"][first:stop].copy()
        held = np.zeros_like(actual)
        held[:, 6] = actual[0, 6]
        for name, actions in [("executed", actual), ("hold", held)]:
            env = OffScreenRenderEnv(bddl_file_name=condition["bddl_path"], camera_heights=256, camera_widths=256)
            try:
                env.seed(row["init_seed"])
                env.reset()
                restore_fixture_state(env, snapshot)
                restore_integration_state(env, snapshot)
                obs = env.set_init_state(snapshot["pre_settle_state"])
                for _ in range(source["config"]["settle_steps"]):
                    obs, _, _, _ = env.step(np.asarray([0, 0, 0, 0, 0, 0, -1], dtype=np.float32))
                max_rgb_error, max_state_error = 0, 0.
                frames, wrists, proprios = [], [], []
                for index in range(stop + 1):
                    images = [np.ascontiguousarray(obs[key][::-1, ::-1])
                              for key in ["agentview_image", "robot0_eye_in_hand_image"]]
                    if name == "executed" or index <= first:
                        expected_images = terminal_images if index == count else [stored[view][index] for view in ["agentview", "wrist"]]
                        for image, reference in zip(images, expected_images):
                            max_rgb_error = max(max_rgb_error, int(np.abs(image.astype(np.int16) -
                                reference.astype(np.int16)).max()))
                        state, reference_state = (env.get_sim_state(), stored["final_sim_state"]) if index == count \
                            else (observation_state(obs), stored["state"][index])
                        max_state_error = max(max_state_error, float(np.abs(state - reference_state).max()))
                    if index >= first:
                        frames.append(images[0]); wrists.append(images[1])
                        proprios.append(np.concatenate([obs["robot0_gripper_qpos"], obs["robot0_eef_pos"],
                                                         obs["robot0_eef_quat"]]).astype(np.float32))
                    if index < stop:
                        action = stored["action"][index] if index < first else actions[index - first]
                        obs, _, _, _ = env.step(action)
                if max_rgb_error > 1 or max_state_error != 0:
                    raise ValueError(f"独立运行重放不等价：RGB={max_rgb_error}，state={max_state_error}")
                path = directory / (name + ".npz")
                np.savez_compressed(path, actions=actions, agentview=np.asarray(frames), wrist=np.asarray(wrists),
                                    proprio=np.asarray(proprios))
                for view, frames_for_view in [("agentview", frames), ("wrist", wrists)]:
                    for label, frame in [("initial", frames_for_view[0]), ("future", frames_for_view[-1])]:
                        Image.fromarray(frame).save(directory / f"{name}_{view}_{label}.png")
                branches.append({"name": name, "packet": str(path), "sha256": sha256(path),
                    "environment_success_after_16_steps": bool(env.check_success()),
                    "replay_max_rgb_error": max_rgb_error, "replay_max_state_error": max_state_error})
            finally:
                env.close()
    report = {"schema": "thor-predict-transition-v1", "collection": str(args.collection),
        "collection_sha256": sha256(args.collection), "source_episode_id": row["episode_id"],
        "policy_ckpt_sha256": row["policy_ckpt_sha256"], "instruction": row["instruction"],
        "task_id": row["task_id"], "init_seed": row["init_seed"], "policy_seed": row["policy_seed"],
        "first_frame": first, "future_frame": stop, "control_hz": 20, "action_horizon": 16,
        "terminal_reference_used": stop == count,
        "terminal_reference_source": "sealed_last_png_and_final_sim_state" if stop == count else "stored_pre_action_observations",
        "branches": branches, "proprio_order": "gripper_qpos(2),eef_pos(3),eef_quat_xyzw(4)",
        "image_orientation": "canonical_rotate180_from_raw_mujoco; Predictor另转为官方vertical_flip方向",
        "is_eval": False, "eligible_for_training": False, "real_robot_commands_sent": False,
        "claim": "同一真实仿真状态的动作条件测试；不代表预测正确、规划收益或成功教师轨迹"}
    dump(directory / "transition.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--collection", type=pathlib.Path, required=True)
    parser.add_argument("--episode-id")
    parser.add_argument("--frame", type=int, default=80)
    args = parser.parse_args()
    args.collection = args.collection.resolve(strict=True)
    raise SystemExit(run_stage("predict_transition", lambda directory: replay(directory, args),
                     {key: str(value) for key, value in vars(args).items()}))
