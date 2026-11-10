#!/usr/bin/env python3
"""在冻结初态上运行Thor策略服务，保存真实成功率、动作、视频与延迟。

@author pony
@date 2026-11-06
@version v1.4.1
@last_modified 2026-11-07
@changelog
    - v1.4.1 (2026-11-07): 端口探针采用与HTTPServer相同的安全复用语义，允许上一阶段TIME_WAIT正常退出。
    - v1.4.0 (2026-11-06): 清理终止子进程；保存逐步物理状态；按回合和动作块派生独立噪声流。
    - v1.3.0 (2026-11-06): 分区并行仿真与编码，策略HTTP仍串行batch1，保持逐回合噪声种子独立。
    - v1.2.0 (2026-11-06): 恢复MuJoCo积分状态后稳定，使用与正式冻结一致的求解器历史。
    - v1.1.0 (2026-11-06): 先恢复固定物体的模型位姿，再恢复完整状态及稳定历史。
    - v1.0.0 (2026-11-06): 服务就绪后才执行仿真，两套LIBERO隔离进程，结束时回收本次服务。
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import pathlib
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

from p0_common import ROOT, ensure_gpu_idle, run_stage
from p0_policy_runtime import sha256, verify_eval_isolation
from p0_sim_data import (capture_integration_state, configure_libero, observation_state,
                         restore_fixture_state, restore_integration_state)


POLICY_NOISE_SCHEDULE = "numpy.SeedSequence([policy_seed, chunk_index]).generate_state(1, uint32)-v2"


def policy_request_seed(policy_seed: int, chunk_index: int) -> int:
    import numpy as np

    if type(policy_seed) is not int or not 0 <= policy_seed < 2 ** 32 \
            or type(chunk_index) is not int or chunk_index < 0:
        raise ValueError("策略种子必须为uint32，动作块序号必须非负")
    # 直接相加会让相邻回合种子的整条噪声流平移重叠；派生只依赖回合和块，不能依赖worker编号。
    return int(np.random.SeedSequence([policy_seed, chunk_index]).generate_state(1, dtype=np.uint32)[0])


def request_json(url: str, value: dict | None = None, timeout: int = 90):
    request = urllib.request.Request(url, data=json.dumps(value).encode() if value is not None else None,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def validate_loopback_port_available(port: int) -> None:
    with socket.socket() as available:
        # HTTPServer本身允许SO_REUSEADDR；探针也要允许正常关闭后的TIME_WAIT，仍拒绝现存监听者。
        available.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        available.bind(("127.0.0.1", port))


def partition_eval_cases(cases: list[dict], conditions: list[dict], workers: int) -> list[dict]:
    if not 1 <= workers <= 4:
        raise ValueError("仿真worker必须在1至4之间")
    known = {row["condition_id"] for row in conditions}
    ids = [row["eval_id"] for row in cases]
    if not cases or len(known) != len(conditions) or len(ids) != len(set(ids)) \
            or any(row["condition_id"] not in known for row in cases):
        raise ValueError("评测分区出现重复ID或未知条件")
    count = min(workers, len(cases))
    jobs = []
    for index in range(count):
        selected = cases[index::count]
        used = {row["condition_id"] for row in selected}
        jobs.append({"cases": selected, "conditions": [row for row in conditions if row["condition_id"] in used]})
    return jobs


def evaluate_worker(spec_path: pathlib.Path) -> None:
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image

    spec = json.loads(spec_path.read_text())
    output = pathlib.Path(spec["output"])
    configure_libero(spec["repo"], output / "libero_config")
    from libero.libero.envs import OffScreenRenderEnv

    rows = []
    with (output / "episodes.jsonl").open("w") as journal:
        for condition in spec["conditions"]:
            if sha256(pathlib.Path(condition["bddl_path"])) != condition["bddl_sha256"]:
                raise RuntimeError("评测场景BDDL摘要不符")
            cases = [row for row in spec["cases"] if row["condition_id"] == condition["condition_id"]]
            if not cases:
                continue
            env = OffScreenRenderEnv(bddl_file_name=condition["bddl_path"], camera_heights=256, camera_widths=256)
            try:
                for case in cases:
                    directory = output / "episodes" / case["eval_id"]
                    directory.mkdir(parents=True, exist_ok=False)
                    snapshot_path = pathlib.Path(case["initial_state_snapshot"])
                    if sha256(snapshot_path) != case["initial_state_sha256"]:
                        raise RuntimeError("评测初态文件摘要不符")
                    snapshot = np.load(snapshot_path)
                    env.seed(case["init_seed"])
                    env.reset()
                    restore_fixture_state(env, snapshot)
                    restore_integration_state(env, snapshot)
                    obs = env.set_init_state(snapshot["pre_settle_state"])
                    for _ in range(case["settle_steps"]):
                        obs, _, _, _ = env.step(np.asarray([0, 0, 0, 0, 0, 0, -1], dtype=np.float32))
                    if not np.allclose(env.get_sim_state(), snapshot["post_settle_state"], atol=1e-9, rtol=0):
                        raise RuntimeError("评测初态未复现冻结时的物理状态")
                    initial_rgb_difference = max(int(np.max(np.abs(obs[key][::-1, ::-1].astype(np.int16)
                                                                  - snapshot[target].astype(np.int16))))
                                                 for key, target in [("agentview_image", "agentview_image"),
                                                                     ("robot0_eye_in_hand_image", "wrist_image")])
                    if initial_rgb_difference > 1:
                        raise RuntimeError(f"评测双相机初态未复现冻结画面：max_abs_diff={initial_rgb_difference}")
                    before = time.monotonic()
                    actions, states, rewards, latencies, request_latencies = [], [], [], [], []
                    physical_states, integration_states, observation_hashes, request_seeds = [], [], [], []
                    queue = []
                    success = False
                    with imageio.get_writer(directory / "agentview.mp4", fps=20, codec="libx264", ffmpeg_params=["-threads", "1"]) as external, \
                         imageio.get_writer(directory / "wrist.mp4", fps=20, codec="libx264", ffmpeg_params=["-threads", "1"]) as wrist:
                        for step in range(case["max_steps"]):
                            images = [np.ascontiguousarray(obs[key][::-1, ::-1])
                                      for key in ["agentview_image", "robot0_eye_in_hand_image"]]
                            state = observation_state(obs)
                            physical_states.append(np.array(env.get_sim_state(), copy=True))
                            integration_states.append(capture_integration_state(env)["integration_state"])
                            if step == 0:
                                imageio.imwrite(directory / "agentview_first.png", images[0])
                                imageio.imwrite(directory / "wrist_first.png", images[1])
                            if not queue:
                                observation_hashes.append(hashlib.sha256(images[0].tobytes() + images[1].tobytes()
                                                                         + state.tobytes()).hexdigest())
                                encoded = []
                                for image in images:
                                    buffer = io.BytesIO()
                                    Image.fromarray(image).save(buffer, format="PNG")
                                    encoded.append(base64.b64encode(buffer.getvalue()).decode())
                                started = time.monotonic()
                                request_seeds.append(policy_request_seed(case["policy_seed"], len(latencies)))
                                value = request_json(spec["server"] + "/predict", {
                                    "instruction": condition["instruction"], "state": state.tolist(),
                                    "images_png_base64": encoded, "seed": request_seeds[-1]})
                                request_latencies.append((time.monotonic() - started) * 1000)
                                predicted = np.asarray(value["actions"], dtype=np.float32)
                                if predicted.shape != (8, 7) or not np.isfinite(predicted).all() or value["state_sha256"] != spec["checkpoint_sha256"]:
                                    raise RuntimeError("在线策略版本或动作格式不符")
                                queue = list(predicted)
                                latencies.append(value["latency_ms"])
                            external.append_data(images[0])
                            wrist.append_data(images[1])
                            action = queue.pop(0)
                            actions.append(action)
                            states.append(state)
                            obs, reward, done, _ = env.step(action)
                            rewards.append(float(reward))
                            success = bool(env.check_success())
                            if success or done:
                                break
                    imageio.imwrite(directory / "agentview_last.png", np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
                    np.savez_compressed(directory / "actions_states.npz", action=np.asarray(actions), state=np.asarray(states),
                                        reward=np.asarray(rewards), final_sim_state=env.get_sim_state(),
                                        physical_state=np.asarray(physical_states), integration_state=np.asarray(integration_states),
                                        final_integration_state=capture_integration_state(env)["integration_state"],
                                        request_seed=np.asarray(request_seeds, dtype=np.uint32),
                                        request_observation_sha256=np.asarray(observation_hashes, dtype="U64"))
                    row = {"episode_id": case["eval_id"], "eval_id": case["eval_id"], "iter": 0,
                           "condition_id": case["condition_id"], "split": case["split"], "env": spec["repo"],
                           "task_id": condition["task_id"], "instruction": condition["instruction"],
                           "condition": condition["condition"], "init_seed": case["init_seed"], "policy_seed": case["policy_seed"],
                           "policy_noise_schedule": POLICY_NOISE_SCHEDULE,
                           "is_eval": True, "eligible_for_training": False,
                           "policy_ckpt_sha256": spec["checkpoint_sha256"], "eval_set_sha256": spec["eval_sha256"],
                           "outcome": {"success": success, "source": "env"}, "interventions": 0,
                           "metrics": {"steps": len(actions), "duration_s": time.monotonic() - before,
                                       "policy_calls": len(latencies)}, "policy_latencies_ms": latencies,
                           "http_request_latencies_ms": request_latencies, "precision": spec["precision"],
                           "batch": 1, "control_hz": 20, "action_horizon": 8,
                           "action_processing": "服务返回raw动作；每20Hz拍执行一行；不重复反归一化",
                           "initial_state_sha256": case["initial_state_sha256"], "artifacts": str(directory),
                           "initial_rgb_max_abs_diff_uint8": initial_rgb_difference,
                           "real_robot_commands_sent": False}
                    (directory / "episode.json").write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n")
                    journal.write(json.dumps(row, ensure_ascii=False) + "\n")
                    journal.flush()
                    rows.append(row)
                    print(json.dumps({"eval_id": row["eval_id"], "success": success, "steps": len(actions),
                                      "duration_s": row["metrics"]["duration_s"]}), flush=True)
            finally:
                env.close()
    (output / "worker_summary.json").write_text(json.dumps({"episodes": len(rows), "success": sum(row["outcome"]["success"] for row in rows)}, indent=2) + "\n")


def evaluate(run_dir: pathlib.Path, args: argparse.Namespace) -> dict:
    ensure_gpu_idle()
    import numpy as np

    checkpoint = args.checkpoint.resolve(strict=True)
    if not checkpoint.is_relative_to(ROOT):
        raise RuntimeError("只评测本项目检查点")
    metadata = json.loads((checkpoint / "checkpoint.json").read_text())
    if sha256(checkpoint / "state.pt") != metadata["state_sha256"]:
        raise RuntimeError("待评测检查点摘要不符")
    isolation = verify_eval_isolation(pathlib.Path(metadata["cache_manifest"]).parent, args.frozen_eval)
    if metadata["eval_overlap_check"].get("eval_set_sha256") != isolation["eval_set_sha256"]:
        raise RuntimeError("检查点不是在当前冻结门禁下训练的")
    frozen = json.loads(args.frozen_eval.read_text())
    conditions = [row for row in frozen["conditions"] if args.task == "all" or args.task in row["task_id"]]
    cases = []
    for condition in conditions:
        selected = [row for row in frozen["episodes"] if row["condition_id"] == condition["condition_id"]]
        cases.extend(selected[:args.per_condition] if args.per_condition else selected)
    if not cases:
        raise RuntimeError("没有选中冻结评测回合")
    for repo in sorted({row["repo"] for row in conditions}):
        # 先验证完整仿真导入，避免模型加载和标准回合跑完后才发现plus缺少进程内动态库路径。
        code = ("import pathlib,sys; sys.path.insert(0,sys.argv[1]); "
                "from p0_sim_data import configure_libero; "
                "configure_libero(sys.argv[2],pathlib.Path(sys.argv[3])); "
                "from libero.libero.envs import OffScreenRenderEnv")
        log_path = run_dir / "logs" / f"{repo}_preflight.log"
        with log_path.open("w") as preflight_log:
            preflight = subprocess.run([str(ROOT / "envs/sim/bin/python"), "-c", code, str(ROOT / "scripts"),
                                        repo, str(run_dir / "preflight" / repo)], cwd=ROOT,
                                       stdout=preflight_log, stderr=subprocess.STDOUT, timeout=60)
        if preflight.returncode:
            raise RuntimeError(f"{repo}仿真预检失败，请先source scripts/thor_env.sh；见{log_path}")
    validate_loopback_port_available(args.port)
    server = f"http://127.0.0.1:{args.port}"
    logs = (run_dir / "logs/policy_server.log").open("w")
    process = subprocess.Popen([str(ROOT / "envs/policy/bin/python"), "-u", str(ROOT / "scripts/p0_policy_server.py"),
                                "--checkpoint", str(checkpoint), "--port", str(args.port)], stdout=logs, stderr=subprocess.STDOUT)
    episodes = []
    actual_workers = {}
    try:
        deadline = time.monotonic() + 180
        while True:
            if process.poll() is not None:
                raise RuntimeError("策略服务在ready之前退出，见logs/policy_server.log")
            try:
                health = request_json(server + "/health", timeout=3)
                if health["pid"] != process.pid or health["state_sha256"] != metadata["state_sha256"]:
                    raise RuntimeError("健康检查连到其他服务")
                break
            except (urllib.error.URLError, TimeoutError):
                if time.monotonic() > deadline:
                    raise RuntimeError("策略服务ready超时")
                time.sleep(0.5)
        rollout_started = time.monotonic()
        for repo in ["LIBERO", "LIBERO-plus"]:
            selected = [row for row in conditions if row["repo"] == repo]
            if not selected:
                continue
            used = {row["condition_id"] for row in selected}
            repo_cases = [row for row in cases if row["condition_id"] in used]
            partitions = partition_eval_cases(repo_cases, selected, args.workers)
            actual_workers[repo] = len(partitions)
            workers, worker_logs = [], []
            try:
                for index, partition in enumerate(partitions):
                    directory = run_dir / repo if len(partitions) == 1 else run_dir / repo / f"worker_{index}"
                    directory.mkdir(parents=True)
                    spec = {**partition, "repo": repo, "server": server, "output": str(directory),
                            "checkpoint_sha256": metadata["state_sha256"],
                            "eval_sha256": isolation["eval_set_sha256"], "precision": metadata["precision"]}
                    spec_path = directory / "worker_spec.json"
                    spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n")
                    worker_log = (run_dir / "logs" / f"{repo}_worker{index}.log").open("w")
                    worker_logs.append(worker_log)
                    worker = subprocess.Popen([str(ROOT / "envs/sim/bin/python"), "-u",
                                               str(ROOT / "scripts/p0_evaluate_policy.py"), "--worker-spec", str(spec_path)],
                                              stdout=worker_log, stderr=subprocess.STDOUT)
                    workers.append((worker, directory))
                # 服务仍使用单线程HTTPServer和每请求独立种子；只重叠CPU物理、编码与等待时间。
                progress_signature = None
                last_progress = time.monotonic()
                while any(worker.poll() is None for worker, _ in workers):
                    if any(worker.poll() not in [None, 0] for worker, _ in workers):
                        raise RuntimeError("仿真worker失败，见各worker日志；本次停止其他worker")
                    if process.poll() is not None:
                        raise RuntimeError("策略服务在评测中退出")
                    signature = tuple((directory / "episodes.jsonl").stat().st_size
                                      if (directory / "episodes.jsonl").exists() else 0 for _, directory in workers)
                    if signature != progress_signature:
                        progress_signature, last_progress = signature, time.monotonic()
                    if time.monotonic() - last_progress > args.progress_timeout:
                        raise RuntimeError(f"仿真worker超过{args.progress_timeout}s无完整回合进展，停止本次评测")
                    time.sleep(0.2)
                if any(worker.returncode for worker, _ in workers):
                    raise RuntimeError("仿真worker未正常退出")
                for _, directory in workers:
                    episodes.extend(json.loads(line) for line in (directory / "episodes.jsonl").read_text().splitlines())
            finally:
                for worker, _ in workers:
                    if worker.poll() is None:
                        worker.terminate()
                for worker, _ in workers:
                    try:
                        worker.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        worker.kill()
                        worker.wait()
                for log in worker_logs:
                    log.close()
        rollout_wall_s = time.monotonic() - rollout_started
    finally:
        server_forced_kill = False
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server_forced_kill = True
            process.kill()
            process.wait()
        logs.close()
        (run_dir / "server_shutdown.json").write_text(json.dumps({"pid": process.pid,
                                                                  "returncode": process.returncode,
                                                                  "forced_kill": server_forced_kill}, indent=2) + "\n")
    if server_forced_kill or process.returncode != 0:
        raise RuntimeError("策略服务未正常退出，本次评测不放行；见server_shutdown.json")
    expected_ids = [row["eval_id"] for row in cases]
    actual_ids = [row["eval_id"] for row in episodes]
    if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != set(expected_ids):
        raise RuntimeError("实际评测ID与选定冻结子集不完全一致")
    ordering = {value: index for index, value in enumerate(expected_ids)}
    episodes.sort(key=lambda row: ordering[row["eval_id"]])
    splits = {}
    for split in ["standard", "target_perturbed", "nontarget_perturbed"]:
        selected = [row for row in episodes if row["split"] == split]
        count = len(selected)
        successes = sum(row["outcome"]["success"] for row in selected)
        splits[split] = {"n": count, "success": successes, "p": successes / count if count else None}
    latencies = [value for row in episodes for value in row["policy_latencies_ms"]]
    http_latencies = [value for row in episodes for value in row["http_request_latencies_ms"]]
    result = {"checkpoint": str(checkpoint), "policy_ckpt_sha256": metadata["state_sha256"],
              "training_step": metadata["step"], "eval_set_sha256": isolation["eval_set_sha256"],
              "splits": splits, "completed_episodes": len(episodes), "frozen_eval_episodes": len(frozen["episodes"]),
              "full_eval_complete": len(episodes) == len(frozen["episodes"]),
              "latency_ms": {"p50": float(np.percentile(latencies, 50)), "p95": float(np.percentile(latencies, 95)),
                             "measured_over": len(latencies)}, "precision": metadata["precision"], "batch": 1,
              "latency_includes": "服务内PNG解码、预处理、模型和动作反归一化；不含HTTP排队",
              "http_request_latency_ms": {"p50": float(np.percentile(http_latencies, 50)),
                                          "p95": float(np.percentile(http_latencies, 95)),
                                          "measured_over": len(http_latencies),
                                          "includes": "同机HTTP请求、排队、服务处理和响应；不含客户端PNG编码"},
              "mean_episode_wall_s": float(np.mean([row["metrics"]["duration_s"] for row in episodes])),
              "sample_count": len(episodes), "eval_overlap_check": isolation, "is_eval": True,
              "environment_workers": args.workers, "policy_batch": 1, "server_inference_serial": True,
              "actual_environment_workers": actual_workers, "policy_noise_schedule": POLICY_NOISE_SCHEDULE,
              "rollout_wall_s": rollout_wall_s, "rollout_episodes_per_s": len(episodes) / rollout_wall_s,
              "rollout_timing_includes": "服务ready后的环境构造、初态核验、全部动作、视频编码和worker退出；不含模型加载",
              "server_stopped": process.poll() is not None, "server_returncode": process.returncode,
              "server_forced_kill": server_forced_kill, "real_robot_commands_sent": False,
              "claim": "冻结初态的真实策略评测；部分回合不能代替完整基线，尚无自进化增益结论"}
    (run_dir / "evaluation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    (run_dir / "episodes.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in episodes))
    return result


def main() -> int:
    def interrupted(signum, _frame):
        # 后台任务的SIGTERM也必须展开finally，防止下一阶段遇到遗留的模型服务。
        raise RuntimeError(f"评测收到终止信号{signum}，停止本次策略与仿真进程")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=pathlib.Path)
    parser.add_argument("--frozen-eval", type=pathlib.Path, default=ROOT / "eval/frozen_eval_set.json")
    parser.add_argument("--task", default="all")
    parser.add_argument("--per-condition", type=int, default=0, help="0为完整评测；正数只做固定子集的试跑")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--workers", type=int, default=1, help="并行仿真进程数；策略推理始终串行batch1")
    parser.add_argument("--progress-timeout", type=int, default=600, help="无完整回合进展的最长秒数")
    parser.add_argument("--worker-spec", type=pathlib.Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_spec:
        spec = args.worker_spec.resolve(strict=True)
        if not spec.is_relative_to(ROOT / "runs/p0"):
            parser.error("worker输入必须在本项目运行目录")
        evaluate_worker(spec)
        return 0
    if not args.checkpoint or args.per_condition < 0 or not 1024 <= args.port <= 65535 \
            or not 1 <= args.workers <= 4 or args.progress_timeout < 1:
        parser.error("必须提供checkpoint，per-condition非负，使用非特权端口")
    return run_stage("policy_evaluation", lambda directory: evaluate(directory, args),
                     {key: str(value) if isinstance(value, pathlib.Path) else value for key, value in vars(args).items()})


if __name__ == "__main__":
    raise SystemExit(main())
