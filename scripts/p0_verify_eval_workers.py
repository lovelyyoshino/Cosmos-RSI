#!/usr/bin/env python3
"""在相同冻结回合上比较串行与并行评测的逐步动作、状态和最终物理状态。

@author pony
@date 2026-11-06
@version v1.0.0
@last_modified 2026-11-06
@changelog
    - v1.0.0 (2026-11-06): 后台串行调用真实评测，严格比较轨迹数组；只有等价才报告提速。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import signal
import subprocess
import time

from p0_common import ROOT, ensure_gpu_idle, run_stage, stop_process_group
from p0_policy_runtime import sha256


def compare_trajectories(first: dict, second: dict, first_arrays, second_arrays) -> dict:
    import numpy as np

    fields = ["eval_id", "condition_id", "init_seed", "policy_seed", "policy_ckpt_sha256",
              "eval_set_sha256", "initial_state_sha256", "precision", "batch", "control_hz", "action_horizon",
              "policy_noise_schedule", "outcome", "is_eval"]
    differences = [key for key in fields if first[key] != second[key]]
    arrays = {}
    for name in ["action", "state", "reward", "request_seed", "physical_state", "integration_state",
                 "final_sim_state", "final_integration_state"]:
        left, right = first_arrays[name], second_arrays[name]
        same_shape = left.shape == right.shape
        finite = bool(np.isfinite(left).all() and np.isfinite(right).all())
        exact = bool(same_shape and finite and np.array_equal(left, right))
        arrays[name] = {"same_shape": same_shape, "finite": finite, "exact": exact,
                        "max_abs_diff": float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))
                        if same_shape and finite and left.size else None}
        if not exact:
            differences.append(name)
    if not np.array_equal(first_arrays["request_observation_sha256"], second_arrays["request_observation_sha256"]):
        differences.append("request_observation_sha256")
    return {"eval_id": first["eval_id"], "passed": not differences,
            "differing_fields": differences, "arrays": arrays}


def verify(directory: pathlib.Path, args: argparse.Namespace) -> dict:
    ensure_gpu_idle()
    import numpy as np

    results = []
    progress_path = directory / "progress.json"
    for workers in [1, args.workers]:
        command = [str(ROOT / "envs/policy/bin/python"), "-u", str(ROOT / "scripts/p0_evaluate_policy.py"),
                   "--checkpoint", str(args.checkpoint), "--frozen-eval", str(args.frozen_eval),
                   "--task", args.task, "--per-condition", "4", "--workers", str(workers)]
        log_path = directory / "logs" / f"workers_{workers}.log"
        with log_path.open("x") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True)
            try:
                progress_path.write_text(json.dumps({"workers": workers, "pid": process.pid,
                                                     "log": str(log_path), "command": command}, indent=2) + "\n")
                code = process.wait()
            except BaseException:
                stop_process_group(process)
                raise
        if code:
            stop_process_group(process)
            raise RuntimeError(f"workers={workers}真实评测失败，后续停止；见{log_path}")
        paths = [line.split("：", 1)[1].strip() for line in log_path.read_text().splitlines()
                 if line.startswith("记录目录：")]
        if len(paths) != 1:
            raise RuntimeError("不能唯一定位真实评测产物")
        run = pathlib.Path(paths[0]).resolve(strict=True)
        probe = json.loads((run / "probe.json").read_text())
        if not run.is_relative_to(ROOT / "runs/p0") or probe["status"] != "passed":
            raise RuntimeError("评测产物目录或验收状态不符")
        results.append((run, json.loads((run / "evaluation.json").read_text()), probe))
        ensure_gpu_idle()
    first_run, first_summary, first_probe = results[0]
    second_run, second_summary, second_probe = results[1]

    def load_rows(run):
        rows = [json.loads(line) for line in (run / "episodes.jsonl").read_text().splitlines()]
        if not rows or len(rows) != len({row["eval_id"] for row in rows}):
            raise RuntimeError("实际评测记录为空或包含重复ID")
        return {row["eval_id"]: row for row in rows}

    first_rows, second_rows = load_rows(first_run), load_rows(second_run)
    if first_rows.keys() != second_rows.keys():
        raise RuntimeError("串行与并行评测回合ID不一致")
    comparisons = []
    for eval_id, first in first_rows.items():
        second = second_rows[eval_id]
        paths = [pathlib.Path(row["artifacts"]).resolve(strict=True) for row in [first, second]]
        if not paths[0].is_relative_to(first_run) or not paths[1].is_relative_to(second_run):
            raise RuntimeError("轨迹数组不在对应的真实评测目录")
        with np.load(paths[0] / "actions_states.npz", allow_pickle=False) as left, \
             np.load(paths[1] / "actions_states.npz", allow_pickle=False) as right:
            comparisons.append(compare_trajectories(first, second, left, right))
    evaluator_sha = sha256(ROOT / "scripts/p0_evaluate_policy.py")
    runtime_sources = {name: sha256(ROOT / name) for name in ["scripts/p0_evaluate_policy.py", "scripts/p0_policy_server.py",
                                                            "scripts/p0_policy_runtime.py", "scripts/p0_sim_data.py",
                                                            "scripts/p0_common.py"]}
    source_consistent = all(probe["source_files_sha256"][name] == expected
                            for probe in [first_probe, second_probe] for name, expected in runtime_sources.items())
    passed = source_consistent and all(row["passed"] for row in comparisons)
    report = {"passed": passed, "sample_count": len(comparisons), "workers": args.workers,
              "policy_batch": 1, "training_step": first_summary["training_step"],
              "checkpoint_sha256": first_summary["policy_ckpt_sha256"],
              "eval_set_sha256": first_summary["eval_set_sha256"],
              "evaluator_source_sha256": evaluator_sha, "runtime_source_sha256": runtime_sources,
              "source_consistent": source_consistent,
              "precision": first_summary["precision"], "comparisons": comparisons,
              "serial_run": str(first_run), "parallel_run": str(second_run),
              "serial_rollout_wall_s": first_summary["rollout_wall_s"],
              "parallel_rollout_wall_s": second_summary["rollout_wall_s"],
              "rollout_speedup": first_summary["rollout_wall_s"] / second_summary["rollout_wall_s"] if passed else None,
              "serial_peak_mem_gb": first_probe["peak_mem_gb"], "parallel_peak_mem_gb": second_probe["peak_mem_gb"],
              "scope": "固定任务的全部四种冻结条件，每条件两个初态和两个策略噪声种子；不代表完整基线或RSI增益",
              "measured_at_unix_s": time.time()}
    (directory / "worker_equivalence.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if not passed:
        raise RuntimeError("串行与并行轨迹未严格等价，不启用并行基线；见worker_equivalence.json")
    return {key: value for key, value in report.items() if key != "comparisons"}


if __name__ == "__main__":
    def interrupted(signum, _frame):
        raise RuntimeError(f"评测等价性验证收到终止信号{signum}，清理本次子进程组")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--frozen-eval", type=pathlib.Path, default=ROOT / "eval/frozen_eval_set.json")
    parser.add_argument("--task", default="STUDY_SCENE1")
    parser.add_argument("--workers", type=int, choices=[2, 4], default=4)
    arguments = parser.parse_args()
    arguments.checkpoint = arguments.checkpoint.resolve(strict=True)
    arguments.frozen_eval = arguments.frozen_eval.resolve(strict=True)
    if not arguments.checkpoint.is_relative_to(ROOT / "runs/p0") \
            or not arguments.frozen_eval.is_relative_to(ROOT / "eval"):
        parser.error("检查点与冻结集必须位于本项目目录")
    raise SystemExit(run_stage("eval_workers_verification", lambda directory: verify(directory, arguments),
                              {key: str(value) if isinstance(value, pathlib.Path) else value
                               for key, value in vars(arguments).items()}))
