#!/usr/bin/env python3
"""捕获并复现真实仿真初态，核查D0物体位姿后一次性冻结正式评测集。

@author pony
@date 2026-11-06
@version v1.3.0
@last_modified 2026-11-06
@changelog
    - v1.3.0 (2026-11-06): 额外保存mjSTATE_INTEGRATION，覆盖布局接触场景的求解器历史；支持多初态回归。
    - v1.2.0 (2026-11-06): 冻结fixture模型位姿，实际复现双相机，避免物理状态相等而画面或碰撞体不同。
    - v1.1.0 (2026-11-06): 捕获和恢复都显式set_init_state，统一相机传感器采样历史并保留差异证据。
    - v1.0.0 (2026-11-06): 固定物理状态、渲染和环境版本，禁止按种子不同就宣称去重。
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
import pathlib

from p0_common import ROOT, command, ensure_gpu_idle, run_stage
from p0_policy_runtime import sha256
from p0_sim_data import (capture_fixture_state, capture_integration_state, configure_libero,
                         restore_fixture_state, restore_integration_state, scene_digest, scene_distance, scene_joints)


def capture_condition(job: dict) -> dict:
    import imageio.v2 as imageio
    import numpy as np

    condition = job["condition"]
    output = pathlib.Path(job["output"]) / condition["condition_id"]
    output.mkdir(parents=True, exist_ok=False)
    configure_libero(condition["repo"], output / "config")
    from libero.libero.envs import OffScreenRenderEnv

    bddl = pathlib.Path(condition["bddl_path"])
    if sha256(bddl) != condition["bddl_sha256"]:
        raise RuntimeError("评测BDDL已经偏离草案")
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    snapshots = []
    noop = np.asarray([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
    try:
        for case in job["initials"]:
            seed = case["init_seed"]
            env.seed(seed)
            obs = env.reset()
            before = env.get_sim_state()
            fixtures = capture_fixture_state(env)
            integration = capture_integration_state(env)
            obs = env.set_init_state(before)
            raw_scene = scene_joints(env)
            for _ in range(case["settle_steps"]):
                obs, _, _, _ = env.step(noop)
            after = env.get_sim_state()
            settled_scene = scene_joints(env)
            image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            if image.std() < 3 or env.check_success():
                raise RuntimeError("评测初态渲染为空或任务已完成")
            distances = [min(scene_distance(raw_scene, row["raw_initial_scene"]),
                             scene_distance(settled_scene, row["initial_scene"])) for row in job["source_scenes"]]
            nearest = float(min(distances))
            if nearest <= 1e-6:
                raise RuntimeError(f"评测与示范存在相同场景初态：{condition['condition_id']}/{seed}")
            snapshot = output / f"s{seed}.npz"
            wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            np.savez_compressed(snapshot, pre_settle_state=before, post_settle_state=after,
                                agentview_image=image, wrist_image=wrist, **fixtures, **integration)
            # 每个初态都实际重新初始化并稳定，验证控制器历史也能重建。
            if not snapshots:
                # 首个初态还跨新环境实例复现，覆盖评测进程与冻结进程分离的调用路径。
                env.close()
                env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
            env.seed(seed)
            env.reset()
            restore_fixture_state(env, fixtures)
            restore_integration_state(env, integration)
            obs2 = env.set_init_state(before)
            for _ in range(case["settle_steps"]):
                obs2, _, _, _ = env.step(noop)
            restored = env.get_sim_state()
            image2 = np.ascontiguousarray(obs2["agentview_image"][::-1, ::-1])
            difference = float(np.max(np.abs(restored - after)))
            rgb_diff = int(np.max(np.abs(image2.astype(np.int16) - image.astype(np.int16))))
            wrist2 = np.ascontiguousarray(obs2["robot0_eye_in_hand_image"][::-1, ::-1])
            wrist_diff = int(np.max(np.abs(wrist2.astype(np.int16) - wrist.astype(np.int16))))
            if difference > 1e-9 or max(rgb_diff, wrist_diff) > 1:
                imageio.imwrite(output / f"s{seed}_capture.png", image)
                imageio.imwrite(output / f"s{seed}_restore.png", image2)
                (output / "replay_diagnostic.json").write_text(json.dumps({
                    "seed": seed, "state_max_abs_diff": difference, "rgb_max_abs_diff": rgb_diff,
                    "wrist_rgb_max_abs_diff": wrist_diff,
                    "rgb_rmse": float(np.sqrt(np.mean((image2.astype(np.float32) - image) ** 2))),
                    "pixels_changed": int(np.any(image2 != image, axis=-1).sum())}, indent=2) + "\n")
                raise RuntimeError(f"固定初态重建不一致：state={difference}, RGB={rgb_diff}, wrist={wrist_diff}")
            if not snapshots:
                imageio.imwrite(output / "agentview_first.png", image)
                imageio.imwrite(output / "wrist_first.png", np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]))
            snapshots.append({"init_seed": seed, "initial_state_snapshot": str(snapshot),
                              "initial_state_sha256": sha256(snapshot), "initial_scene_sha256": scene_digest(settled_scene),
                              "nearest_demo_scene_joint_max_abs_diff": nearest,
                              "scene_duplicate_tolerance": 1e-6, "replay_state_max_abs_diff": difference,
                              "replay_rgb_max_abs_diff_uint8": rgb_diff,
                              "replay_wrist_rgb_max_abs_diff_uint8": wrist_diff,
                              "fixture_model_state_saved": True,
                              "integration_state_saved": True,
                              "fresh_environment_replay_checked": not snapshots,
                              "restore_protocol": "env.seed(init_seed); env.reset(); restore_fixture_state(snapshot); restore_integration_state(snapshot); set_init_state(pre_settle_state); settle_steps次noop",
                              "is_eval": True})
        return {"condition_id": condition["condition_id"], "snapshots": snapshots,
                "sample_count": len(snapshots), "replay_check_passed": True, "source_overlap": False}
    finally:
        env.close()


def freeze(run_dir: pathlib.Path, args: argparse.Namespace) -> dict:
    ensure_gpu_idle()
    canonical = ROOT / "eval/frozen_eval_set.json"
    if canonical.exists() and not args.probe:
        raise RuntimeError("正式评测集已经存在，不允许覆盖；必须另外制定新版本")
    cache_path = args.cache / "manifest.json"
    cache = json.loads(cache_path.read_text())
    if not cache.get("all_accepted_episodes_env_success") or any(not row.get("env_success") for row in cache["episodes"]):
        raise RuntimeError("冻结前要求来源明确且逐条回放成功的本机D0")
    if (args.cache / "COMPLETED").read_text().strip() != cache["cache_id"]:
        raise RuntimeError("D0缓存尚未完成")
    draft = json.loads(args.draft.read_text())
    conditions = [row for row in draft["conditions"] if row["instruction"] in cache["tasks"]]
    if args.probe:
        # 只验证接口时每个实现各选一种条件，不生成正式冻结文件。
        if args.probe_conditions:
            selected = set(args.probe_conditions)
            conditions = [row for row in conditions if row["condition_id"] in selected]
            if {row["condition_id"] for row in conditions} != selected:
                raise RuntimeError("指定的探针条件不在当前草案或D0中")
        else:
            standard = next(row for row in conditions if row["repo"] == "LIBERO")
            plus = next(row for row in conditions if row["repo"] == "LIBERO-plus")
            conditions = [standard, plus]
    jobs = []
    for condition in conditions:
        cases = [row for row in draft["episodes"] if row["condition_id"] == condition["condition_id"]]
        initials = list({row["init_seed"]: row for row in cases}.values())
        if args.probe:
            initials = initials[:args.probe_initials]
        scenes = [row for row in cache["episodes"] if row["instruction"] == condition["instruction"]]
        jobs.append({"condition": condition, "initials": initials, "source_scenes": scenes,
                     "output": str(run_dir / "state_snapshots")})
    results = []
    # 同名libero包按实现分组启动全新进程，防止standard误用plus或反过来。
    for repo in ["LIBERO", "LIBERO-plus"]:
        selected = [job for job in jobs if job["condition"]["repo"] == repo]
        if not selected:
            continue
        with ProcessPoolExecutor(max_workers=min(args.workers, len(selected)),
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = [pool.submit(capture_condition, job) for job in selected]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(json.dumps({"captured": result["condition_id"], "initials": result["sample_count"]}), flush=True)
    snapshots = {(row["condition_id"], snap["init_seed"]): snap for row in results for snap in row["snapshots"]}
    cases = []
    for row in draft["episodes"]:
        key = (row["condition_id"], row["init_seed"])
        if key in snapshots:
            cases.append({**row, **snapshots[key]})
    counts = dict(Counter(row["split"] for row in cases))
    if not args.probe and counts != {"standard": 200, "target_perturbed": 800, "nontarget_perturbed": 200}:
        raise RuntimeError(f"正式评测数量不符合协议：{counts}")
    training_seeds = {row["init_seed"] for row in cache["episodes"]}
    if training_seeds & {row["init_seed"] for row in cases}:
        raise RuntimeError("D0再生seed与评测seed重叠")
    result = {"version": "thor-libero10-eval-v1", "freeze_status": "probe_only" if args.probe else "frozen",
              "conditions": conditions, "episodes": cases, "counts": counts,
              "unique_initial_states": len(snapshots), "source_cache_manifest": str(cache_path),
              "source_cache_manifest_sha256": sha256(cache_path), "source_draft_sha256": sha256(args.draft),
              "eval_overlap_check": {"passed": True, "episode_ids_disjoint": True, "seeds_disjoint": True,
                                     "all_source_episodes": len(cache["episodes"]), "source_scene_duplicate_tolerance": 1e-6,
                                     "scope": "对所选D0全部成功示范的原始和稳定后场景关节比较；不宣称与整个预训练语料去重"},
              "all_initial_states_replayed": True, "runtime_environment_validated": True,
              "simulation": {"resolution": [256, 256], "control_hz": 20, "image_transform": "rotate180",
                             "precision": "MuJoCo CPU、完整状态fp64、RGB uint8", "batch": 1,
                             "policy_seeds": draft["policy_seeds"], "max_steps": 520, "settle_steps": 10,
                             "training_steps": 0, "policy_evaluated": False},
              "initial_snapshot_protocol": "MjSimState+fixture_body_pos+fixture_body_quat+mjSTATE_INTEGRATION-v2",
              "conditions_sha256": {row["condition_id"]: row["bddl_sha256"] for row in conditions},
              "upstream_sha": {repo: command(["git", "-C", str(ROOT / "third_party" / repo), "rev-parse", "HEAD"])["stdout"]
                               for repo in ["LIBERO", "LIBERO-plus"]},
              "real_robot_commands_sent": False}
    # JSON内部摘要覆盖其余字段；文件摘要另存sidecar，避免自引用哈希。
    result["content_sha256"] = hashlib.sha256(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    output = run_dir / ("eval_probe.json" if args.probe else "frozen_eval_set.json")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    if not args.probe:
        canonical.parent.mkdir(exist_ok=True)
        with canonical.open("x") as stream:
            stream.write(output.read_text())
        canonical.with_suffix(".sha256").write_text(sha256(canonical) + "\n")
    summary = {"eval_set": str(output if args.probe else canonical), "sha256": sha256(output),
               "freeze_status": result["freeze_status"], "counts": counts, "unique_initial_states": len(snapshots),
               "eval_overlap_check": result["eval_overlap_check"], "all_initial_states_replayed": True,
               "sample_count": len(cases), "precision": result["simulation"]["precision"], "batch": 1,
               "steps_per_initial": 20, "policy_evaluated": False}
    (run_dir / "freeze_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", type=pathlib.Path, required=True)
    parser.add_argument("--draft", type=pathlib.Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--probe-conditions", nargs="+", default=[], help="仅probe时限定需要回归的真实场景")
    parser.add_argument("--probe-initials", type=int, default=1, help="每个探针条件实际复现的初态数量")
    args = parser.parse_args()
    args.cache = args.cache.resolve(strict=True)
    args.draft = args.draft.resolve(strict=True)
    if args.probe_conditions and not args.probe:
        parser.error("probe-conditions必须与probe同时使用")
    if not 1 <= args.probe_initials <= 20:
        parser.error("probe-initials必须在1至20之间")
    if not args.cache.is_relative_to(ROOT / "data") or not args.draft.is_relative_to(ROOT / "runs/p0") or not 1 <= args.workers <= 4:
        parser.error("缓存和草案必须在项目内，worker在1至4之间")
    return run_stage("freeze_eval", lambda directory: freeze(directory, args),
                     {key: str(value) if isinstance(value, pathlib.Path) else value for key, value in vars(args).items()})


if __name__ == "__main__":
    raise SystemExit(main())
