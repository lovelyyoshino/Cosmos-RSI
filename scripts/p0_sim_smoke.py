#!/usr/bin/env python3
"""单环境 EGL 渲染和动作接口验证；零动作轨迹禁止用于策略训练。

@author pony
@date 2026-11-06
@version v1.3.0
@last_modified 2026-11-06
@changelog
    - v1.3.0 (2026-11-06): 给硬编码资源位置建立项目内软链接，不覆盖任何已有资源。
    - v1.2.0 (2026-11-06): 按扰动后缀选择最小样本，使用BDDL任务语言避免条件名混入指令。
    - v1.1.0 (2026-11-06): 支持外部资源目录，保留逐步耗时和环境成功真值。
    - v1.0.0 (2026-11-06): 保存双视角图像、视频和动作状态，明确不代表策略成功。
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

from p0_common import ROOT, ensure_gpu_idle, run_stage


def smoke(run_dir: pathlib.Path, args: argparse.Namespace) -> dict:
    ensure_gpu_idle()
    repo = ROOT / "third_party" / args.repo
    benchmark_root = repo / "libero" / "libero"
    if args.assets_dir:
        assets_link = benchmark_root / "assets"
        assets_target = args.assets_dir.resolve(strict=True)
        if assets_link.exists() and assets_link.resolve() != assets_target:
            raise RuntimeError("上游固定assets位置已有其他资源，禁止覆盖")
        if not assets_link.exists():
            if assets_link.is_symlink():
                raise RuntimeError("上游assets位置存在失效软链接，需要先核查")
            assets_link.symlink_to(assets_target, target_is_directory=True)
    config_dir = run_dir / "libero_config"
    config_dir.mkdir()
    config = {"benchmark_root": str(benchmark_root), "bddl_files": str(benchmark_root / "bddl_files"),
              "init_states": str(benchmark_root / "init_files"), "datasets": str(ROOT / "data" / args.repo),
              "assets": str(args.assets_dir.resolve() if args.assets_dir else benchmark_root / "assets")}
    pathlib.Path(config["datasets"]).mkdir(parents=True, exist_ok=True)
    # JSON 是 YAML 的子集；预写配置可避免上游 import 时交互询问或写入 ~/.libero。
    (config_dir / "config.yaml").write_text(json.dumps(config, indent=2) + "\n")
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    sys.path.insert(0, str(repo))

    import imageio.v2 as imageio
    import mujoco
    import numpy as np
    import robosuite
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    np.random.seed(args.seed)
    suite = benchmark.get_benchmark_dict()[args.suite]()
    task_index = args.task
    if args.task_suffix:
        options = [index for index, task in enumerate(suite.tasks) if task.name.endswith(args.task_suffix)]
        if not options:
            raise RuntimeError(f"没有找到指定扰动后缀：{args.task_suffix}")
        task_index = options[0]
    task = suite.get_task(task_index)
    bddl_path = benchmark_root / "bddl_files" / task.problem_folder / task.bddl_file
    env = None
    external_writer = wrist_writer = None
    actions, eef_pos, eef_quat, gripper_qpos = [], [], [], []
    timings = []
    try:
        env = OffScreenRenderEnv(bddl_file_name=str(bddl_path), camera_heights=args.resolution,
                                 camera_widths=args.resolution)
        env.seed(args.seed)
        obs = env.reset()
        external_writer = imageio.get_writer(run_dir / "agentview.mp4", fps=20, codec="libx264")
        wrist_writer = imageio.get_writer(run_dir / "wrist.mp4", fps=20, codec="libx264")
        first_frame_std = None
        done = False
        for step in range(args.steps):
            # 保存与上游评测相同的180度图像旋转；原始状态不做隐式拼接或归一化。
            external = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            assert external.shape == (args.resolution, args.resolution, 3) and external.dtype == np.uint8
            assert wrist.shape == external.shape and wrist.dtype == np.uint8
            if step == 0:
                first_frame_std = float(external.std())
                if first_frame_std < 3:
                    raise RuntimeError("渲染图像接近纯色，EGL 冒烟不通过")
                imageio.imwrite(run_dir / "agentview_first.png", external)
                imageio.imwrite(run_dir / "wrist_first.png", wrist)
            external_writer.append_data(external)
            wrist_writer.append_data(wrist)
            eef_pos.append(np.asarray(obs["robot0_eef_pos"]).copy())
            eef_quat.append(np.asarray(obs["robot0_eef_quat"]).copy())
            gripper_qpos.append(np.asarray(obs["robot0_gripper_qpos"]).copy())
            action = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
            actions.append(action)
            before = time.perf_counter()
            obs, reward, done, info = env.step(action)
            timings.append(time.perf_counter() - before)
            if done:
                break
        imageio.imwrite(run_dir / "agentview_last.png", np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
        np.savez_compressed(run_dir / "actions_states.npz", action=np.asarray(actions),
                            eef_pos=np.asarray(eef_pos), eef_quat=np.asarray(eef_quat),
                            gripper_qpos=np.asarray(gripper_qpos))
        result = {
            "kind": "zero_action_simulator_smoke", "eligible_for_training": False,
            "policy": None, "repo": args.repo, "suite": args.suite, "task_index": task_index,
            "task_name": task.name, "instruction": env.language_instruction, "benchmark_language": task.language,
            "bddl_path": str(bddl_path),
            "assets_path": config["assets"], "assets_runtime_link": str(benchmark_root / "assets"),
            "seed": args.seed, "sample_count": 1, "batch": 1, "steps": len(actions),
            "precision": "MuJoCo CPU physics；RGB uint8；发送动作 fp32", "renderer": "EGL",
            "resolution": [args.resolution, args.resolution], "fps": 20,
            "image_transform": "rotate180，与上游LIBERO评测一致", "first_frame_std": first_frame_std,
            "environment_done": bool(done), "final_reward": float(reward),
            "task_success": bool(env.check_success()), "warmup_steps": 0,
            "step_wall_ms_samples": [float(value * 1000) for value in timings],
            "step_wall_ms": {"mean": float(np.mean(timings) * 1000),
                             "p50": float(np.percentile(timings, 50) * 1000),
                             "p95": float(np.percentile(timings, 95) * 1000)},
            "step_timing_includes": "env.step的CPU物理和双视角渲染，不含模型推理或视频编码",
            "mujoco": mujoco.__version__, "robosuite": robosuite.__version__, "numpy": np.__version__,
            "observation_shapes": {k: list(v.shape) for k, v in obs.items() if hasattr(v, "shape")},
        }
        (run_dir / "episode.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        return result
    finally:
        if external_writer is not None:
            external_writer.close()
        if wrist_writer is not None:
            wrist_writer.close()
        if env is not None:
            env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", choices=["LIBERO", "LIBERO-plus"], default="LIBERO")
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--task-suffix", help="仅用于P0接口测试，选择第一个匹配后缀的任务；正式评测必须冻结具体ID")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--assets-dir", type=pathlib.Path)
    args = parser.parse_args()
    if args.steps < 1 or args.resolution < 16:
        parser.error("steps 必须为正数，resolution 至少16")
    conditions = {k: str(v) if isinstance(v, pathlib.Path) else v for k, v in vars(args).items()}
    return run_stage("sim_smoke", lambda run_dir: smoke(run_dir, args), conditions)


if __name__ == "__main__":
    raise SystemExit(main())
