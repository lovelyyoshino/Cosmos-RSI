#!/usr/bin/env python3
"""在Thor回放官方原始动作，构建有完整初态和成功真值的双视角D0缓存。

@author pony
@date 2026-11-06
@version v1.1.0
@last_modified 2026-11-06
@changelog
    - v1.1.0 (2026-11-06): 按历史运行SHA256复用已完成回合，仅重跑尚未完成任务，不修改历史文件。
    - v1.0.0 (2026-11-06): 按官方BDDL/seed0/十步稳定再生，保留失败及完整状态，禁止复用未知时序标签。
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
import os
import pathlib
import shutil
import time

from p0_common import ROOT, command, ensure_gpu_idle, run_stage
from p0_policy_runtime import sha256
from p0_sim_data import configure_libero, kept_action_indices, observation_state, scene_digest, scene_joints

REGENERATION_REFERENCE = "openvla-oft@e4287e94541f459edc4feabc4e181f537cd569a8/experiments/robot/libero/regenerate_libero_dataset.py"


def regenerate_task(job: dict) -> list[dict]:
    import h5py
    import imageio.v2 as imageio
    import numpy as np

    output = pathlib.Path(job["output"])
    benchmark = configure_libero("LIBERO", output / "config")
    from libero.libero.envs import OffScreenRenderEnv

    env = OffScreenRenderEnv(bddl_file_name=str(benchmark / "bddl_files/libero_10" / job["bddl"]),
                             camera_heights=256, camera_widths=256)
    results = []
    try:
        with h5py.File(job["file"], "r") as handle:
            group = handle["data"]
            demo_ids = sorted(group.keys(), key=lambda key: int(key.removeprefix("demo_")))[:job["per_task"]]
            for demo_id in demo_ids:
                started = time.monotonic()
                demo = group[demo_id]
                actions = demo["actions"][:].astype(np.float32)
                indices = kept_action_indices(actions)
                if len(indices) < 8:
                    raise RuntimeError("原始示范过滤后短于动作窗口")
                episode_id = "thor-regen-v1-" + pathlib.Path(job["bddl"]).stem + "-" + demo_id
                directory = output / episode_id
                directory.mkdir(parents=True, exist_ok=False)
                previous = pathlib.Path(job["reuse_run"]) / "episodes" / episode_id if job.get("reuse_run") else None
                if previous and (previous / "episode.json").is_file():
                    names = ["images.npy", "states.npy", "actions.npy", "physical_states.npz", "initial_sim_state.npy",
                             "agentview_first.png", "agentview_last.png", "episode.json"]
                    for name in names:
                        expected = job["reuse_hashes"].get(f"episodes/{episode_id}/{name}")
                        if not expected or sha256(previous / name) != expected:
                            raise RuntimeError("已完成回合不符合历史运行SHA256，拒绝复用")
                    record = json.loads((previous / "episode.json").read_text())
                    if record["source_file_sha256"] != job["sha256"] or record["source_demo_id"] != demo_id \
                            or record["init_seed"] != 0 or record["settle_steps"] != 10 \
                            or not np.array_equal(np.load(previous / "actions.npy"), actions[indices]):
                        raise RuntimeError("已完成回合的来源、动作或再生配方不同")
                    for name in names[:-1]:
                        os.link(previous / name, directory / name)
                    record = {**record, "directory": str(directory), "reused_from": str(previous),
                              "original_wall_s": record["wall_s"], "wall_s": time.monotonic() - started}
                    (directory / "episode.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
                    results.append(record)
                    print(json.dumps({"episode": episode_id, "success": record["env_success"], "frames": len(indices),
                                      "reused": True, "wall_s": record["wall_s"]}), flush=True)
                    continue
                env.seed(0)
                env.reset()
                raw_initial = demo["states"][0]
                obs = env.set_init_state(raw_initial)
                raw_scene = scene_joints(env)
                for _ in range(10):
                    obs, _, _, _ = env.step(np.asarray([0, 0, 0, 0, 0, 0, -1], dtype=np.float32))
                initial_state = env.get_sim_state()
                initial_scene = scene_joints(env)
                np.save(directory / "initial_sim_state.npy", initial_state)
                images = np.lib.format.open_memmap(directory / "images.npy", mode="w+", dtype=np.uint8,
                                                   shape=(len(indices), 2, 256, 256, 3))
                states = np.empty((len(indices), 8), dtype=np.float32)
                physical = []
                for frame, raw_index in enumerate(indices):
                    for view, key in enumerate(["agentview_image", "robot0_eye_in_hand_image"]):
                        images[frame, view] = obs[key][::-1, ::-1]
                    states[frame] = observation_state(obs)
                    physical.append(env.get_sim_state())
                    obs, _, _, _ = env.step(actions[raw_index])
                success = bool(env.check_success())
                images.flush()
                imageio.imwrite(directory / "agentview_first.png", images[0, 0])
                imageio.imwrite(directory / "agentview_last.png", np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
                np.save(directory / "states.npy", states)
                np.save(directory / "actions.npy", actions[indices])
                np.savez_compressed(directory / "physical_states.npz", states=np.asarray(physical),
                                    raw_initial_state=raw_initial, initial_state=initial_state,
                                    final_state=env.get_sim_state(), raw_action_indices=indices)
                record = {"episode_id": episode_id, "instruction": job["instruction"], "bddl_file": job["bddl"],
                          "length": len(indices), "source_raw_frames": len(actions),
                          "source_file": job["file"], "source_file_sha256": job["sha256"], "source_demo_id": demo_id,
                          "env_success": success, "source_success": "本机MuJoCo完整动作回放的最终环境真值",
                          "init_seed": 0, "seed_meaning": "本机再生环境seed，原始示范的采集seed未知",
                          "raw_initial_scene": raw_scene, "initial_scene": initial_scene,
                          "raw_initial_scene_sha256": scene_digest(raw_scene),
                          "initial_scene_sha256": scene_digest(initial_scene), "control_hz": 20, "settle_steps": 10,
                          "frame_semantics": "先记录obs_t、state_t和完整物理状态，再执行action_t一次",
                          "directory": str(directory), "wall_s": time.monotonic() - started}
                (directory / "episode.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
                results.append(record)
                del images
                print(json.dumps({"episode": episode_id, "success": success, "frames": len(indices),
                                  "wall_s": record["wall_s"]}), flush=True)
    finally:
        env.close()
    return results


def regenerate(run_dir: pathlib.Path, args: argparse.Namespace) -> dict:
    ensure_gpu_idle()
    import h5py
    import numpy as np

    source = json.loads(args.raw_manifest.read_text())
    reuse_probe = json.loads((args.reuse_run / "probe.json").read_text()) if args.reuse_run else None
    if reuse_probe:
        current = command(["git", "-C", str(ROOT / "third_party/LIBERO"), "rev-parse", "HEAD"])
        if reuse_probe["stage"] != "regenerate_d0" or current.get("returncode") != 0 or reuse_probe["upstream_sha"]["LIBERO"] != current["stdout"]:
            raise RuntimeError("不能跨LIBERO版本复用回合")
    jobs = []
    for item in source["files"]:
        path = pathlib.Path(item["file"]).resolve(strict=True)
        if not path.is_relative_to(ROOT / "data") or path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise RuntimeError("原始示范未通过下载清单核验")
        bddl = path.stem.removesuffix("_demo") + ".bddl"
        if args.task != "all" and args.task not in bddl:
            continue
        with h5py.File(path, "r") as handle:
            instruction = json.loads(handle["data"].attrs["problem_info"])["language_instruction"]
        jobs.append({"file": str(path), "sha256": item["sha256"], "bddl": bddl, "instruction": instruction,
                     "output": str(run_dir / "episodes"), "per_task": args.per_task,
                     "reuse_run": str(args.reuse_run) if args.reuse_run else None,
                     "reuse_hashes": reuse_probe["artifacts_sha256"] if reuse_probe else {}})
    if not jobs:
        raise RuntimeError("没有选中原始任务")
    identity = {"algorithm": "thor-regenerate-v1", "source_manifest_sha256": sha256(args.raw_manifest),
                "tasks": [job["bddl"] for job in jobs], "per_task": args.per_task,
                "regeneration_seed": 0, "split_seed": 42, "settle_steps": 10, "horizon": 8,
                "reuse_probe_sha256": sha256(args.reuse_run / "probe.json") if args.reuse_run else None,
                "implementation_sha256": {name: sha256(ROOT / "scripts" / name)
                                          for name in ["p0_regenerate_d0.py", "p0_sim_data.py"]}}
    cache_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    destination = ROOT / "data" / ("libero10_regen_" + cache_id[:12])
    if destination.exists():
        raise RuntimeError(f"再生缓存已存在，请复用而不覆盖：{destination}")
    # 同时容纳逐回合暂存和最终连续数组；不删除旧缓存来腾空间。
    estimate = 0
    for job in jobs:
        with h5py.File(job["file"], "r") as handle:
            keys = sorted(handle["data"], key=lambda key: int(key.removeprefix("demo_")))[:args.per_task]
            estimate += sum(len(handle["data"][key]["actions"]) for key in keys) * 2 * 256 * 256 * 3
    if shutil.disk_usage(ROOT).free < 2 * estimate + 20 * 1024 ** 3:
        raise RuntimeError("再生和聚合后不足20GiB磁盘余量")
    records = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs)),
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(regenerate_task, job) for job in jobs]
        with (run_dir / "logs/regeneration.jsonl").open("w") as journal:
            for future in as_completed(futures):
                for record in future.result():
                    records.append(record)
                    journal.write(json.dumps(record, ensure_ascii=False) + "\n")
                journal.flush()
    accepted = sorted((row for row in records if row["env_success"]), key=lambda row: row["episode_id"])
    rejected = [row for row in records if not row["env_success"]]
    for job in jobs:
        members = [row for row in accepted if row["instruction"] == job["instruction"]]
        if len(members) < 2:
            raise RuntimeError(f"任务成功示范不足以划分训练/验证：{job['instruction']}")
        ranked = sorted(members, key=lambda row: hashlib.sha256(("regen-holdout:42:" + row["episode_id"]).encode()).hexdigest())
        validation_ids = {row["episode_id"] for row in ranked[:max(1, round(len(members) * 0.1))]}
        for row in members:
            row["split"] = "demo_validation" if row["episode_id"] in validation_ids else "demo_train"
    count = sum(row["length"] for row in accepted)
    staging = destination.with_name(destination.name + ".building_" + run_dir.name)
    staging.mkdir()
    arrays = {"images": np.lib.format.open_memmap(staging / "images.npy", mode="w+", dtype=np.uint8,
                                                shape=(count, 2, 256, 256, 3)),
              "states": np.lib.format.open_memmap(staging / "states.npy", mode="w+", dtype=np.float32, shape=(count, 8)),
              "actions": np.lib.format.open_memmap(staging / "actions.npy", mode="w+", dtype=np.float32, shape=(count, 7)),
              "timestamps": np.lib.format.open_memmap(staging / "timestamps.npy", mode="w+", dtype=np.float64, shape=(count,))}
    minima = {"state": np.full(8, np.inf), "action": np.full(7, np.inf)}
    maxima = {"state": np.full(8, -np.inf), "action": np.full(7, -np.inf)}
    offset = 0
    for row in accepted:
        directory = pathlib.Path(row["directory"])
        stop = offset + row["length"]
        for name in ["images", "states", "actions"]:
            values = np.load(directory / (name + ".npy"), mmap_mode="r")
            if name != "images" and not np.isfinite(values).all():
                raise RuntimeError("再生观测或动作包含非有限值")
            arrays[name][offset:stop] = values
        arrays["timestamps"][offset:stop] = np.arange(row["length"]) / 20
        if row["split"] == "demo_train":
            for key, name in [("state", "states"), ("action", "actions")]:
                minima[key] = np.minimum(minima[key], arrays[name][offset:stop].min(axis=0))
                maxima[key] = np.maximum(maxima[key], arrays[name][offset:stop].max(axis=0))
        provenance = staging / "provenance" / row["episode_id"]
        provenance.mkdir(parents=True)
        for name in ["physical_states.npz", "initial_sim_state.npy", "episode.json", "agentview_first.png", "agentview_last.png"]:
            shutil.copy2(directory / name, provenance / name)
        row["offset"] = offset
        row["directory"] = str(destination / "provenance" / row["episode_id"])
        offset = stop
    for array in arrays.values():
        array.flush()
    del arrays
    normalizer = {"mode": "min_max", "fit_split": "demo_train",
                  **{key: {"min": minima[key].tolist(), "max": maxima[key].tolist()} for key in minima}}
    manifest = {"dataset": "LIBERO10_D0_THOR_REGENERATED", "cache_id": cache_id, "identity": identity,
                "source_manifest": str(args.raw_manifest), "source_repo": source["repository"],
                "source_revision": source["revision"], "source_endpoint": source["download_endpoint"],
                "source_reference": REGENERATION_REFERENCE, "source_files": source["files"],
                "frames": count, "episodes": accepted, "rejected_episodes": rejected,
                "tasks": {job["instruction"]: job["bddl"] for job in jobs}, "normalizer": normalizer,
                "horizon": 8, "action_dim": 7, "state_dim": 8, "control_hz": 20,
                "views": {"0": "agentview", "1": "robot0_eye_in_hand"}, "image_transform": "rotate180",
                "state_semantics": ["eef_x", "eef_y", "eef_z", "axis_angle_x", "axis_angle_y", "axis_angle_z",
                                    "gripper_joint0", "gripper_joint1"],
                "frame_semantics": "动作执行前记录；每动作只执行20Hz一拍；跳过的no-op不推进仿真",
                "all_accepted_episodes_env_success": True, "train_validation_episode_overlap": False,
                "eval_overlap_check": {"passed": False, "reason": "后续冻结评测时按实际场景关节去重"},
                "precision": "MuJoCo CPU、RGB uint8、action/state fp32、完整物理状态fp64",
                "array_sha256": {name: sha256(staging / name) for name in ["images.npy", "states.npy", "actions.npy", "timestamps.npy"]}}
    (staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (staging / "COMPLETED").write_text(cache_id + "\n")
    staging.rename(destination)
    # 聚合和哈希成功后仅清理本次任务自己生成的图像暂存，保留完整状态及失败证据。
    for record in records:
        directory = run_dir / "episodes" / record["episode_id"]
        if record["env_success"]:
            for name in ["images.npy", "states.npy", "actions.npy"]:
                (directory / name).unlink()
    result = {"cache": str(destination), "manifest": str(destination / "manifest.json"),
              "manifest_sha256": sha256(destination / "manifest.json"), "tasks": len(jobs), "frames": count,
              "attempted_episodes": len(records), "accepted_episodes": len(accepted), "rejected_episodes": len(rejected),
              "demo_train_episodes": sum(row["split"] == "demo_train" for row in accepted),
              "demo_validation_episodes": sum(row["split"] == "demo_validation" for row in accepted),
              "regeneration_seed": 0, "split_seed": 42, "workers": min(args.workers, len(jobs)),
              "evaluation_gate_passed": False, "real_robot_commands_sent": False,
              "precision": manifest["precision"], "batch": None, "training_steps": 0}
    (run_dir / "regeneration.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--per-task", type=int, default=50)
    parser.add_argument("--task", default="all")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--reuse-run", type=pathlib.Path)
    args = parser.parse_args()
    args.raw_manifest = args.raw_manifest.resolve(strict=True)
    if args.reuse_run:
        args.reuse_run = args.reuse_run.resolve(strict=True)
        if not args.reuse_run.is_relative_to(ROOT / "runs/p0"):
            parser.error("只允许复用项目内已留存完整SHA256的运行")
    if not args.raw_manifest.is_relative_to(ROOT / "runs/p0") or args.per_task < 2 or not 1 <= args.workers <= 4:
        parser.error("清单位于runs/p0，每任务至少2条，worker在1至4之间")
    return run_stage("regenerate_d0", lambda directory: regenerate(directory, args),
                     {key: str(value) if isinstance(value, pathlib.Path) else value for key, value in vars(args).items()})


if __name__ == "__main__":
    raise SystemExit(main())
