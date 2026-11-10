#!/usr/bin/env python3
"""正式策略在独立初态采集反馈，记录困难信号并交接无损运行存档。

@author pony
@date 2026-11-08
@version v1.2.0
@last_modified 2026-11-09
@changelog
    - v1.2.0 (2026-11-09): 后续圈采集接纳具体已审批晋升，绑定当前链接及真实父权重。
    - v1.1.0 (2026-11-09): 独立进程组并行采集、失败统一回收及按预声明顺序交接。
    - v1.0.1 (2026-11-09): worker已退出时也清理本次进程组的残留后代。
    - v1.0.0 (2026-11-08): 保留冻结评测实现；新增独立采集、真实动作离散度和LeRobot运行存档。
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import pathlib
import signal
import subprocess
import time
import urllib.error

from p0_common import ROOT, command, ensure_gpu_idle, run_stage, stop_process_group
from p0_evaluate_policy import (POLICY_NOISE_SCHEDULE, policy_request_seed, request_json,
                                validate_loopback_port_available)
from p0_lerobot_handoff import audit_dataset, dump, write_dataset
from p0_policy_runtime import sha256, verify_eval_isolation
from p0_sim_data import (capture_fixture_state, capture_integration_state, configure_libero,
                         observation_state, scene_digest, scene_distance, scene_joints)


def checked_policy_checkpoint(checkpoint, promotion_path=None):
    checkpoint = pathlib.Path(checkpoint).resolve(strict=True)
    if not checkpoint.is_relative_to(ROOT / "runs/p0") \
            and not checkpoint.is_relative_to(ROOT / "checkpoints/promoted"):
        raise ValueError("反馈采集只接受本项目运行权重或经过核验的晋升权重")
    metadata = json.loads((checkpoint / "checkpoint.json").read_text())
    if sha256(checkpoint / "state.pt") != metadata["state_sha256"]:
        raise ValueError("采集策略权重摘要不符")
    promotion = None
    if checkpoint.is_relative_to(ROOT / "checkpoints/promoted"):
        from p1_eval_gate import checked_current_promotion

        _, promotion = checked_current_promotion(checkpoint, metadata["state_sha256"], promotion_path,
                                                 preceding_iteration=metadata["iteration"])
    elif promotion_path is not None:
        raise ValueError("未晋升运行权重不能借用晋升记录")
    return metadata, promotion


def action_dispersion(samples):
    """同一观测、独立噪声动作的总体标准差；不与实际执行动作混为一谈。"""
    import numpy as np

    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] < 2 or values.shape[1:] != (8, 7) \
            or not np.isfinite(values).all():
        raise ValueError("离散度需要至少两个有限的8×7归一化动作样本")
    return float(np.sqrt(np.var(values, axis=0, ddof=0).mean()))


def build_cases(conditions, frozen, cache, *, init_seed, policy_seed, count, max_steps):
    for key, value in [("init_seed", init_seed), ("policy_seed", policy_seed), ("count", count), ("max_steps", max_steps)]:
        if type(value) is not int or value < (1 if key in {"count", "max_steps"} else 0):
            raise ValueError(f"{key}取值非法")
    if max(init_seed, policy_seed) + count > 2 ** 32:
        raise ValueError("初态与策略种子必须为uint32")
    if not conditions or len({row["condition_id"] for row in conditions}) != len(conditions):
        raise ValueError("采集条件为空或重复")
    forbidden_initials = {row["init_seed"] for row in frozen["episodes"] + cache["episodes"]}
    forbidden_policy = {row["policy_seed"] for row in frozen["episodes"]}
    if set(range(init_seed, init_seed + count)) & forbidden_initials \
            or set(range(policy_seed, policy_seed + count)) & forbidden_policy:
        raise ValueError("运行采集与D0/冻结评测的种子相交")
    return [{"case_id": f"{condition['condition_id']}-s{init_seed + index}-p{policy_seed + index}",
             "condition_id": condition["condition_id"], "init_seed": init_seed + index,
             "policy_seed": policy_seed + index, "max_steps": max_steps, "is_eval": False}
            for condition in conditions for index in range(count)]


def partition_feedback_cases(cases, conditions, workers):
    """按原计划轮转分区；worker编号不参与初态、策略或请求噪声种子。"""
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("独立采集worker必须在1至4之间")
    known = {row["condition_id"] for row in conditions}
    ids = [row["case_id"] for row in cases]
    if not cases or len(known) != len(conditions) or len(ids) != len(set(ids)) \
            or any(row["condition_id"] not in known for row in cases):
        raise ValueError("采集分区出现空计划、重复ID或未知条件")
    count = min(workers, len(cases))
    partitions = []
    for index in range(count):
        selected = cases[index::count]
        used = {row["condition_id"] for row in selected}
        partitions.append({"cases": selected,
                           "conditions": [row for row in conditions if row["condition_id"] in used]})
    return partitions


def order_feedback_rows(rows, cases, run_id):
    """真实回合须逐项绑定计划；完成顺序不改变LeRobot episode索引。"""
    expected = {run_id + "-" + case["case_id"]: case for case in cases}
    actual = {row["episode_id"]: row for row in rows}
    if len(expected) != len(cases) or len(actual) != len(rows) or set(actual) != set(expected):
        raise ValueError("真实采集ID与预声明计划不符")
    for episode_id, case in expected.items():
        if any(actual[episode_id][key] != case[key]
               for key in ["condition_id", "init_seed", "policy_seed", "max_steps", "is_eval"]):
            raise ValueError("真实采集条件或种子偏离预声明计划")
    return [actual[episode_id] for episode_id in expected]


def run_collection_workers(directory, repo, spec, partitions, server_process, timeout_s):
    """同一repo并行，失败先终止全部组再逐组等待，避免后续阶段遇到遗留后代。"""
    workers, rows = [], []
    started = time.monotonic()
    try:
        for index, partition in enumerate(partitions):
            output = directory / repo if len(partitions) == 1 else directory / repo / f"worker_{index}"
            output.mkdir(parents=True, exist_ok=False)
            current = {**spec, **partition, "output": str(output),
                       "worker_index": index, "worker_count": len(partitions)}
            dump(output / "worker_spec.json", current)
            suffix = "" if len(partitions) == 1 else str(index)
            log_path = directory / "logs" / f"{repo}_worker{suffix}.log"
            log = log_path.open("x")
            try:
                process = subprocess.Popen([str(ROOT / "envs/sim/bin/python"), "-u",
                    str(ROOT / "scripts/p0_feedback_rollout.py"), "--worker-spec", str(output / "worker_spec.json")],
                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            except BaseException:
                log.close()
                raise
            workers.append({"process": process, "output": output, "log": log, "log_path": log_path,
                            "spec": current, "started": time.monotonic(), "wall_s": None})
        while True:
            now = time.monotonic()
            codes = [worker["process"].poll() for worker in workers]
            for worker, code in zip(workers, codes):
                if code is not None and worker["wall_s"] is None:
                    worker["wall_s"] = now - worker["started"]
            if any(code not in [None, 0] for code in codes):
                raise RuntimeError(f"独立采集{repo}有worker失败，见对应worker日志")
            if server_process.poll() is not None:
                raise RuntimeError("反馈采集策略服务在worker运行期间退出")
            if all(code == 0 for code in codes):
                break
            if now - started >= timeout_s:
                raise TimeoutError(f"独立采集{repo}超过{timeout_s}s，停止本次全部worker")
            time.sleep(.1)
        for worker in workers:
            stored = [json.loads(line) for line in (worker["output"] / "episodes.jsonl").read_text().splitlines()]
            rows.extend(order_feedback_rows(stored, worker["spec"]["cases"], spec["run_id"]))
    finally:
        # 先通知所有进程组；即使leader已退出，也不能跳过它留下的ffmpeg等后代。
        cleanup_errors = []
        for worker in workers:
            try:
                os.killpg(worker["process"].pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except OSError as exc:
                cleanup_errors.append(str(exc))
        summaries = []
        for worker in workers:
            process = worker["process"]
            summary = {"worker_index": worker["spec"]["worker_index"], "pid": process.pid,
                       "cases": len(worker["spec"]["cases"]), "output": str(worker["output"]),
                       "log": str(worker["log_path"]), "wall_s": worker["wall_s"]}
            try:
                summary["cleanup"] = stop_process_group(process)
                if summary["cleanup"]["forced_kill"]:
                    cleanup_errors.append(f"worker {summary['worker_index']}需要强制退出")
            except Exception as exc:
                summary["cleanup_error"] = str(exc)
                cleanup_errors.append(str(exc))
            finally:
                summary["returncode"] = process.poll()
                summary["wall_until_cleanup_s"] = time.monotonic() - worker["started"]
                worker["log"].close()
                summaries.append(summary)
        dump(directory / f"{repo}_workers.json", {"repo": repo, "workers": summaries,
             "actual_workers": len(workers), "timeout_s": timeout_s, "wall_s": time.monotonic() - started,
             "worker_wall_scope": "Popen返回至首次观察退出；异常时另报包含进程组回收的wall_until_cleanup_s",
             "cleanup_errors": cleanup_errors})
        if cleanup_errors:
            raise RuntimeError("独立采集worker回收未正常通过：" + "; ".join(cleanup_errors))
    return rows


def object_scene(scene):
    return {name: values for name, values in scene.items() if not name.startswith(("robot", "gripper"))}


def scene_from_flat_state(env, flattened):
    """只读取同一BDDL模型的qpos；排除时钟和机器人，不推进冻结物理状态。"""
    import numpy as np

    values = np.asarray(flattened)
    qpos = values[1:1 + env.sim.model.nq]
    if qpos.shape != (env.sim.model.nq,) or not np.isfinite(qpos).all():
        raise ValueError("冻结MjSimState缺少有限qpos")
    result = {}
    for name in env.sim.model.joint_names:
        if name.startswith(("robot", "gripper")):
            continue
        address = env.sim.model.get_joint_qpos_addr(name)
        value = np.atleast_1d(qpos[address[0]:address[1]] if isinstance(address, tuple) else qpos[address]).copy()
        if len(value) == 7:
            nonzero = np.flatnonzero(np.abs(value[3:]) > 1e-12)
            if len(nonzero) and value[3 + nonzero[0]] < 0:
                value[3:] *= -1
        result[name] = value.tolist()
    return result


def check_scene_isolation(env, condition, raw, settled, frozen, cache, tolerance):
    import numpy as np

    raw, settled = object_scene(raw), object_scene(settled)
    sources = [row for row in cache["episodes"] if row["instruction"] == condition["instruction"]]
    if not sources:
        raise ValueError("采集任务不属于已验证D0")
    demo_distance = min(min(scene_distance(raw, object_scene(row["raw_initial_scene"])),
                            scene_distance(settled, object_scene(row["initial_scene"]))) for row in sources)
    snapshots = {row["initial_state_snapshot"]: row["initial_state_sha256"]
                 for row in frozen["episodes"] if row["condition_id"] == condition["condition_id"]}
    if not snapshots:
        raise ValueError("缺少同条件冻结初态的隔离参照")
    distances = []
    for filename, checksum in snapshots.items():
        path = pathlib.Path(filename)
        if sha256(path) != checksum:
            raise ValueError("冻结参照初态摘要已变化")
        with np.load(path) as saved:
            distances.append(min(scene_distance(raw, scene_from_flat_state(env, saved["pre_settle_state"])),
                                 scene_distance(settled, scene_from_flat_state(env, saved["post_settle_state"]))))
    nearest_eval = min(distances)
    if min(demo_distance, nearest_eval) <= tolerance:
        raise ValueError("独立采集实际物体初态与D0或冻结参照重复")
    return {"passed": True, "nearest_demo_object_qpos_max_abs_diff": demo_distance,
            "nearest_same_condition_eval_object_qpos_max_abs_diff": nearest_eval,
            "demo_episodes_checked": len(sources), "eval_snapshots_checked": len(snapshots),
            "scene_duplicate_tolerance": tolerance,
            "scope": "全部同任务D0物体初态、同条件冻结物体初态及全冻结集种子/ID隔离；排除机器人/夹爪"}


def collect_worker(spec_path):
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image

    spec = json.loads(spec_path.read_text())
    output = pathlib.Path(spec["output"])
    configure_libero(spec["repo"], output / "libero_config")
    from libero.libero.envs import OffScreenRenderEnv

    frozen = json.loads(pathlib.Path(spec["frozen_eval"]).read_text())
    cache = json.loads(pathlib.Path(spec["cache_manifest"]).read_text())
    conditions = {row["condition_id"]: row for row in spec["conditions"]}
    config = spec["config"]
    rows = []
    for case in spec["cases"]:
        condition = conditions[case["condition_id"]]
        if sha256(pathlib.Path(condition["bddl_path"])) != condition["bddl_sha256"]:
            raise ValueError("采集BDDL摘要不符")
        directory = output / "episodes" / case["case_id"]
        directory.mkdir(parents=True, exist_ok=False)
        env = OffScreenRenderEnv(bddl_file_name=condition["bddl_path"], camera_heights=256, camera_widths=256)
        try:
            env.seed(case["init_seed"])
            env.reset()
            before = np.array(env.get_sim_state(), copy=True)
            fixtures, integration = capture_fixture_state(env), capture_integration_state(env)
            obs = env.set_init_state(before)
            raw = scene_joints(env)
            for _ in range(config["settle_steps"]):
                obs, _, _, _ = env.step(np.asarray([0, 0, 0, 0, 0, 0, -1], dtype=np.float32))
            if env.check_success():
                raise ValueError("采集初态已成功，不能作为正常策略回合")
            settled = scene_joints(env)
            isolation = check_scene_isolation(env, condition, raw, settled, frozen, cache,
                                              config["scene_duplicate_tolerance"])
            np.savez_compressed(directory / "initial_state.npz", pre_settle_state=before,
                                post_settle_state=env.get_sim_state(), **fixtures, **integration)
            actions, states, views, wrists, rewards, dispersions, latencies = [], [], [], [], [], [], []
            request_seeds, queue, chunks = [], [], 0
            started = time.monotonic()
            success = False
            done = False
            with imageio.get_writer(directory / "agentview.mp4", fps=20, codec="libx264", ffmpeg_params=["-threads", "1"]) as external, \
                 imageio.get_writer(directory / "wrist.mp4", fps=20, codec="libx264", ffmpeg_params=["-threads", "1"]) as wrist:
                for step in range(case["max_steps"]):
                    images = [np.ascontiguousarray(obs[key][::-1, ::-1])
                              for key in ["agentview_image", "robot0_eye_in_hand_image"]]
                    state = observation_state(obs)
                    if not queue:
                        encoded = []
                        for frame in images:
                            stream = io.BytesIO(); Image.fromarray(frame).save(stream, format="PNG")
                            encoded.append(base64.b64encode(stream.getvalue()).decode())
                        samples = []
                        sample_count = config["action_samples"] if chunks % config["dispersion_every_chunks"] == 0 else 1
                        for noise_index in range(sample_count):
                            noise_seed = policy_request_seed(case["policy_seed"], chunks) if noise_index == 0 else \
                                int(np.random.SeedSequence([case["policy_seed"], chunks, noise_index, 902]).generate_state(1)[0])
                            result = request_json(spec["server"] + "/predict", {"instruction": condition["instruction"],
                                "state": state.tolist(), "images_png_base64": encoded, "seed": noise_seed})
                            predicted = np.asarray(result["actions"], dtype=np.float32)
                            normalized = np.asarray(result["normalized_actions"], dtype=np.float32)
                            if predicted.shape != (8, 7) or normalized.shape != (8, 7) \
                                    or not np.isfinite(predicted).all() or not np.isfinite(normalized).all() \
                                    or result["state_sha256"] != spec["checkpoint_sha256"]:
                                raise ValueError("策略动作形状、有限性或在线版本不符")
                            if noise_index == 0:
                                queue = list(predicted)
                            samples.append(normalized); request_seeds.append(noise_seed); latencies.append(result["latency_ms"])
                        if sample_count > 1:
                            dispersions.append({"frame": step, "chunk": chunks, "samples": sample_count,
                                                "value": action_dispersion(samples)})
                        chunks += 1
                    if step in {0, case["max_steps"] // 2}:
                        for view_name, image in zip(["agentview", "wrist"], images):
                            Image.fromarray(image).save(directory / f"{view_name}_{step:04d}.png")
                    external.append_data(images[0]); wrist.append_data(images[1])
                    action = queue.pop(0)
                    actions.append(action); states.append(state); views.append(images[0]); wrists.append(images[1])
                    obs, reward, done, _ = env.step(action)
                    rewards.append(float(reward))
                    success = bool(env.check_success())
                    if success or done:
                        break
            for view_name, key in [("agentview", "agentview_image"), ("wrist", "robot0_eye_in_hand_image")]:
                Image.fromarray(np.ascontiguousarray(obs[key][::-1, ::-1])).save(directory / f"{view_name}_last.png")
            count = len(actions)
            trajectory = directory / "trajectory.npz"
            np.savez_compressed(trajectory, action=np.asarray(actions, dtype=np.float32), state=np.asarray(states, dtype=np.float32),
                agentview=np.asarray(views), wrist=np.asarray(wrists), reward=np.asarray(rewards),
                request_seed=np.asarray(request_seeds, dtype=np.uint32), final_sim_state=env.get_sim_state())
            # 重闭夹爪只是命令事件；接触和提起校验前不把它当真实重抓或空抓。
            closing = np.asarray(actions)[:, 6] > 0
            close_events = int(np.count_nonzero(closing & ~np.r_[False, closing[:-1]]))
            row = {"episode_id": spec["run_id"] + "-" + case["case_id"], "iter": 0,
                "kind": "formal_policy_independent_feedback_rollout",
                "env": "libero_plus" if spec["repo"] == "LIBERO-plus" else "libero", "implementation": spec["repo"],
                "task_id": condition["task_id"], "instruction": condition["instruction"],
                "condition_id": case["condition_id"],
                "conditions": {"light": condition["condition_id"] if condition["condition"] == "lighting" else "default",
                    "background": condition["condition_id"] if condition["condition"] == "background" else "default",
                    "layout": "default", "camera": "default", "init": "independent_seeded_reset"},
                "condition": condition["condition"], "init_seed": case["init_seed"], "policy_seed": case["policy_seed"],
                "is_eval": False, "eligible_for_training": False, "outcome": {"success": success, "source": "env"},
                "policy_ckpt_sha256": spec["checkpoint_sha256"], "git_sha": spec["git_sha"],
                "initial_scene_sha256": scene_digest(object_scene(settled)), "isolation": isolation,
                "metrics": {"steps": count, "duration_s": time.monotonic() - started,
                    "sim_duration_s": count / 20, "policy_calls": len(latencies),
                    "action_dispersion_p90": float(np.percentile([x["value"] for x in dispersions], 90)),
                    "regrasp_count": None, "gripper_reclose_command_count": max(0, close_events - 1)},
                "metric_status": {"action_dispersion_p90": "同观测多噪声实际推理；归一化动作总体标准差RMS的回合p90",
                    "regrasp_count": "未校验接触与物体提起，不用重闭命令冒充空抓/重抓"},
                "dispersion": dispersions, "policy_latencies_ms": latencies, "precision": spec["precision"],
                "batch": 1, "action_samples": config["action_samples"], "max_steps": case["max_steps"],
                "control_hz": 20, "action_horizon": 8, "policy_noise_schedule": POLICY_NOISE_SCHEDULE,
                "terminal_reason": "env_success" if success else ("env_done" if done else "max_steps"),
                "trajectory": str(trajectory), "trajectory_sha256": sha256(trajectory),
                "artifacts": str(directory), "interventions": 0, "real_robot_commands_sent": False,
                "training_boundary": "先存档；判官校准、来源与成功资格核验后才能用作自蒸馏"}
            dump(directory / "episode.json", row)
            rows.append(row)
            print(json.dumps({"episode_id": row["episode_id"], "success": success, "steps": count,
                              "action_dispersion_p90": row["metrics"]["action_dispersion_p90"]}), flush=True)
        finally:
            env.close()
    (output / "episodes.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def collect(directory, args):
    import numpy as np
    import yaml

    ensure_gpu_idle()
    config = yaml.safe_load(args.config.read_text())
    if config["version"] != "thor-feedback-rollout-v1" or config["control_hz"] != 20 or config["action_horizon"] != 8 \
            or type(config["action_samples"]) is not int or not 2 <= config["action_samples"] <= 8 \
            or type(config["dispersion_every_chunks"]) is not int or config["dispersion_every_chunks"] < 1 \
            or type(config["settle_steps"]) is not int or config["settle_steps"] < 0 \
            or config["scene_duplicate_tolerance"] != 1e-6:
        raise ValueError("运行配置不符合当前真实动作与困难信号约定")
    checkpoint = args.checkpoint.resolve(strict=True)
    metadata, parent_promotion = checked_policy_checkpoint(checkpoint, args.parent_promotion)
    cache_path = pathlib.Path(metadata["cache_manifest"])
    isolation = verify_eval_isolation(cache_path.parent, args.frozen_eval)
    if isolation["eval_set_sha256"] != metadata["eval_overlap_check"]["eval_set_sha256"]:
        raise ValueError("策略训练的冻结集版本不符")
    frozen, cache = json.loads(args.frozen_eval.read_text()), json.loads(cache_path.read_text())
    selected = set(args.conditions)
    conditions = [row for row in frozen["conditions"] if row["condition_id"] in selected]
    if selected != {row["condition_id"] for row in conditions}:
        raise ValueError("采集条件不在预声明注册表中")
    cases = build_cases(conditions, frozen, cache, init_seed=args.init_seed, policy_seed=args.policy_seed,
                        count=args.per_condition, max_steps=config["max_steps"])
    partitions = {}
    for repo in ["LIBERO", "LIBERO-plus"]:
        current = [row for row in conditions if row["repo"] == repo]
        if current:
            ids = {row["condition_id"] for row in current}
            partitions[repo] = partition_feedback_cases(
                [row for row in cases if row["condition_id"] in ids], current, args.workers)
    plan = {"schema": config["version"], "cases": cases, "conditions": conditions, "config": config,
            "source_eval_registry_sha256": sha256(args.frozen_eval), "is_eval": False,
            "selection_uses_frozen_outcomes": False, "episodes": len(cases), "workers_requested": args.workers,
            "worker_cases": {repo: [[row["case_id"] for row in part["cases"]] for part in jobs]
                             for repo, jobs in partitions.items()}}
    dump(directory / "collection_plan.json", plan)
    git = command(["git", "-C", str(ROOT), "rev-parse", "HEAD"])
    validate_loopback_port_available(args.port)
    server = f"http://127.0.0.1:{args.port}"
    rows, actual_workers, worker_reports = [], {}, {}
    log = (directory / "logs/policy_server.log").open("x")
    process = subprocess.Popen([str(ROOT / "envs/policy/bin/python"), "-u", str(ROOT / "scripts/p0_policy_server.py"),
        "--checkpoint", str(checkpoint), "--port", str(args.port)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True)
    cleanup = None
    try:
        deadline = time.monotonic() + 180
        while True:
            if process.poll() is not None:
                raise RuntimeError("反馈采集策略服务ready前退出，见日志")
            try:
                health = request_json(server + "/health", timeout=3)
                if health["pid"] != process.pid or health["state_sha256"] != metadata["state_sha256"]:
                    raise ValueError("反馈采集连接到了其他策略服务")
                break
            except (urllib.error.URLError, TimeoutError):
                if time.monotonic() >= deadline:
                    raise RuntimeError("反馈采集策略服务ready超时")
                time.sleep(.5)
        rollout_started = time.monotonic()
        for repo, jobs in partitions.items():
            spec = {"repo": repo, "config": config,
                "frozen_eval": str(args.frozen_eval), "cache_manifest": str(cache_path), "server": server,
                "checkpoint_sha256": metadata["state_sha256"], "precision": metadata["precision"],
                "git_sha": git.get("stdout") if git.get("returncode") == 0 else None,
                "run_id": directory.name}
            # HTTP服务仍串行batch1；总repo预算涵盖多个worker共享服务的排队时间。
            timeout_s = max(300, sum(len(job["cases"]) for job in jobs) * 240)
            rows.extend(run_collection_workers(directory, repo, spec, jobs, process, timeout_s))
            actual_workers[repo] = len(jobs)
            worker_reports[repo] = str(directory / f"{repo}_workers.json")
        rollout_wall_s = time.monotonic() - rollout_started
    finally:
        cleanup = stop_process_group(process)
        log.close()
        dump(directory / "server_shutdown.json", {**cleanup, "returncode": process.returncode})
    if cleanup["forced_kill"] or process.returncode:
        raise RuntimeError("反馈采集服务未正常退出，禁止开始后续模型阶段")
    rows = order_feedback_rows(rows, cases, directory.name)
    if any(row["is_eval"] is not False or row["policy_ckpt_sha256"] != metadata["state_sha256"] for row in rows):
        raise ValueError("真实采集回合与预声明计划不符")
    packets = []
    for row in rows:
        with np.load(row["trajectory"]) as stored:
            count = len(stored["action"])
            packets.append({"source_episode_id": row["episode_id"], "instruction": row["instruction"],
                "source_split": "runtime", "source_success": row["outcome"]["success"], "is_eval": False,
                "window": {"start_frame": 0, "end_frame": count}, "source_frame_index": np.arange(count),
                "source_timestamp": np.arange(count) / 20,
                **{key: np.array(stored[key]) for key in ["agentview", "wrist", "action", "state"]}})
    dataset = directory / "rollouts/lerobot"
    write_dataset(dataset, packets, metadata["normalizer"], packet_kind="runtime_rollout")
    audit = audit_dataset(dataset, packets)
    for index, row in enumerate(rows):
        row["lerobot_index"] = {"dataset": str(dataset), "episode_index": index}
        dump(pathlib.Path(row["artifacts"]) / "episode.json", row)
    (directory / "rollouts/episodes.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    result = {"schema": config["version"], "checkpoint": str(checkpoint), "policy_ckpt_sha256": metadata["state_sha256"],
        "parent_promotion": parent_promotion,
        "training_steps": metadata["step"], "training_seed": metadata["recipe"]["seed"],
        "precision": metadata["precision"], "batch": 1, "max_steps": config["max_steps"],
        "init_seeds": sorted({row["init_seed"] for row in rows}), "policy_seeds": sorted({row["policy_seed"] for row in rows}),
        "sample_count": len(rows), "successes": sum(row["outcome"]["success"] for row in rows),
        "episodes": str(directory / "rollouts/episodes.jsonl"), "dataset": str(dataset), "dataset_audit": audit,
        "config": config, "is_eval": False, "eval_overlap_check": isolation, "server_stopped": True,
        "workers_requested": args.workers, "actual_workers": actual_workers, "worker_reports": worker_reports,
        "rollout_wall_s": rollout_wall_s,
        "rollout_wall_scope": "服务ready后各repo串行的worker启动、采集及回收；不含模型加载和最终LeRobot交接",
        "eligible_for_training": False, "promotion_performed": False, "real_robot_commands_sent": False,
        "claim": "独立反馈与无损存档；不是冻结评测、收益结论或已准入自蒸馏数据"}
    dump(directory / "feedback_rollout.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--worker-spec", type=pathlib.Path)
    parser.add_argument("--checkpoint", type=pathlib.Path)
    parser.add_argument("--parent-promotion", type=pathlib.Path)
    parser.add_argument("--config", type=pathlib.Path, default=ROOT / "configs/rollout.yaml")
    parser.add_argument("--frozen-eval", type=pathlib.Path, default=ROOT / "eval/frozen_eval_set.json")
    parser.add_argument("--conditions", nargs="+", default=["t00-standard"])
    parser.add_argument("--init-seed", type=int, default=61000)
    parser.add_argument("--policy-seed", type=int, default=71000)
    parser.add_argument("--per-condition", type=int, default=1)
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=1,
                        help="每个repo的仿真worker上限；repo串行，策略服务仍为串行batch1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.worker_spec:
        spec = args.worker_spec.resolve(strict=True)
        if not spec.is_relative_to(ROOT / "runs/p0"):
            parser.error("worker输入必须在本项目运行目录")
        collect_worker(spec)
        return 0
    if args.checkpoint is None or not 1024 <= args.port <= 65535:
        parser.error("必须提供项目检查点和非特权端口")
    for name in ["checkpoint", "config", "frozen_eval"]:
        setattr(args, name, getattr(args, name).resolve(strict=True))
    if args.parent_promotion:
        args.parent_promotion = args.parent_promotion.resolve(strict=True)
    def interrupted(number, _frame):
        raise RuntimeError(f"反馈采集收到信号{number}，回收本次服务与worker")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    return run_stage("feedback_rollout", lambda directory: collect(directory, args),
                     {key: str(value) if isinstance(value, pathlib.Path) else value for key, value in vars(args).items()})


if __name__ == "__main__":
    raise SystemExit(main())
