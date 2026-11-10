#!/usr/bin/env python3
"""串行执行正式初训、固定子集试跑、续训及完整基线；失败即停，不晋升权重。

@author pony
@date 2026-11-06
@version v1.2.0
@last_modified 2026-11-06
@changelog
    - v1.2.0 (2026-11-06): 允许等待指定实测预检通过后续训；最终权重重新验证并行轨迹，失败即停。
    - v1.1.0 (2026-11-06): 通过真实轨迹等价验证后允许并行仿真，完整评测显式传递worker数。
    - v1.0.0 (2026-11-06): 长任务后台串行、记录子进程与真实产物路径，固定训练日程，不按评测挑检查点。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import signal
import subprocess
import time

from p0_common import ROOT, run_stage, stop_process_group
from p0_policy_runtime import sha256


def baseline(directory: pathlib.Path, args: argparse.Namespace) -> dict:
    completed = []
    progress_path = directory / "progress.json"

    def progress(phase, **fields):
        record = {"phase": phase, "updated_unix_s": time.time(), "completed": completed, **fields}
        temporary = progress_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(progress_path)
        print(json.dumps({"phase": phase, **fields}, ensure_ascii=False), flush=True)

    def child(phase, script, parameters):
        command = [str(ROOT / "envs/policy/bin/python"), "-u", str(ROOT / "scripts" / script), *parameters]
        log_path = directory / "logs" / f"{phase}.log"
        with log_path.open("x") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True)
            try:
                progress(phase, child_pid=process.pid, command=command, log_path=str(log_path))
                code = process.wait()
            except BaseException:
                stop_process_group(process)
                raise
        if code:
            # leader可能先退出；仍须清理它留下的本次进程组，不能按poll结果跳过。
            stop_process_group(process)
            progress("failed", failed_phase=phase, returncode=code, log_path=str(log_path))
            raise RuntimeError(f"{phase}失败，已停止后续阶段；日志：{log_path}")
        paths = [line.split("：", 1)[1].strip() for line in log_path.read_text().splitlines()
                 if line.startswith("记录目录：")]
        if len(paths) != 1:
            raise RuntimeError(f"无法唯一定位{phase}产物：{log_path}")
        output = pathlib.Path(paths[0]).resolve(strict=True)
        if not output.is_relative_to(ROOT / "runs/p0"):
            raise RuntimeError("子阶段产物越过项目边界")
        record = json.loads((output / "probe.json").read_text())
        if record["status"] != "passed":
            raise RuntimeError("子进程退出码和实验验收状态不一致")
        completed.append({"phase": phase, "run": str(output), "wall_s": record["wall_s"],
                          "peak_mem_gb": record["peak_mem_gb"]})
        progress("between_stages", last_completed=phase)
        return output, record["result"]

    if args.after_verification:
        progress("waiting_for_verification", prerequisite=str(args.after_verification))
        deadline = time.monotonic() + args.wait_timeout
        preflight = None
        while time.monotonic() < deadline:
            if args.after_verification.exists():
                try:
                    preflight = json.loads(args.after_verification.read_text())
                    break
                except json.JSONDecodeError:
                    pass
            time.sleep(5)
        if not preflight or preflight.get("stage") != "eval_workers_verification" or preflight.get("status") != "passed":
            raise RuntimeError("指定的并行实测预检失败或未在期限内完成，禁止启动训练")
        evidence = preflight["result"]
        checkpoint_sha = json.loads((args.resume / "checkpoint.json").read_text())["state_sha256"]
        if not evidence.get("passed") or evidence.get("checkpoint_sha256") != checkpoint_sha \
                or evidence.get("eval_set_sha256") != sha256(args.frozen_eval):
            raise RuntimeError("预检未绑定本次续训起点或冻结集")
        progress("preflight_passed", prerequisite=str(args.after_verification))

    training = ["--cache", str(args.cache), "--formal-baseline", "--frozen-eval", str(args.frozen_eval),
                "--recipe", "lora", "--batch", str(args.batch), "--steps", str(args.steps),
                "--seed", str(args.seed), "--save-every", "500", "--eval-every", "250", "--verify-reload"]
    if args.resume:
        checkpoint = args.resume
    else:
        _, initial = child("train_initial", "p0_train_policy.py", [*training, "--stop-after", str(args.first_steps)])
        checkpoint = pathlib.Path(initial["last_checkpoint"])
    metadata = json.loads((checkpoint / "checkpoint.json").read_text())
    if metadata["recipe"]["steps"] != args.steps or metadata["recipe"]["batch"] != args.batch \
            or metadata["recipe"]["seed"] != args.seed or not metadata.get("formal_baseline_training"):
        raise RuntimeError("检查点不是本次固定日程和种子的正式初训产物")
    parallel_evidence = None
    _, sanity = child("evaluate_fixed_subset", "p0_evaluate_policy.py",
                       ["--checkpoint", str(checkpoint), "--frozen-eval", str(args.frozen_eval),
                        "--task", "STUDY_SCENE1", "--per-condition", "1"])
    if sanity["completed_episodes"] != 4 or not sanity["server_stopped"]:
        raise RuntimeError("固定子集试跑没有完整结束或服务未退出")
    if args.pilot_only:
        result = {"pilot_complete": True, "formal_training_complete": metadata["step"] == args.steps,
                  "baseline_complete": False, "checkpoint": str(checkpoint), "completed": completed,
                  "claim": "只有固定子集接口试跑；不能代替完整基线或RSI实验"}
    else:
        if metadata["step"] < args.steps:
            _, trained = child("train_remaining", "p0_train_policy.py", [*training, "--resume", str(checkpoint)])
            checkpoint = pathlib.Path(trained["last_checkpoint"])
        if args.evaluation_workers > 1:
            final_metadata = json.loads((checkpoint / "checkpoint.json").read_text())
            if args.parallel_verification:
                parallel_evidence = json.loads(args.parallel_verification.read_text())
            if not parallel_evidence or parallel_evidence.get("checkpoint_sha256") != final_metadata["state_sha256"]:
                evidence_run, _ = child("verify_final_parallel", "p0_verify_eval_workers.py",
                                        ["--checkpoint", str(checkpoint), "--frozen-eval", str(args.frozen_eval),
                                         "--workers", str(args.evaluation_workers)])
                parallel_evidence = json.loads((evidence_run / "worker_equivalence.json").read_text())
            if parallel_evidence.get("passed") is not True or parallel_evidence.get("workers") != args.evaluation_workers \
                    or parallel_evidence.get("sample_count", 0) < 16 \
                    or parallel_evidence.get("eval_set_sha256") != sha256(args.frozen_eval):
                raise RuntimeError("最终权重的并行等价性验证不符合当前冻结集或worker数")
            sources = parallel_evidence.get("runtime_source_sha256", {})
            expected_sources = {"scripts/p0_evaluate_policy.py", "scripts/p0_policy_server.py",
                                "scripts/p0_policy_runtime.py", "scripts/p0_sim_data.py", "scripts/p0_common.py"}
            if set(sources) != expected_sources or any(sha256(ROOT / name) != expected for name, expected in sources.items()):
                raise RuntimeError("并行验证后推理或仿真代码已改变，必须重新验证")
        _, evaluation = child("evaluate_full", "p0_evaluate_policy.py",
                              ["--checkpoint", str(checkpoint), "--frozen-eval", str(args.frozen_eval),
                               "--workers", str(args.evaluation_workers)])
        if not evaluation["full_eval_complete"] or not evaluation["server_stopped"]:
            raise RuntimeError("完整冻结基线未完成或服务未退出")
        result = {"formal_training_complete": True, "baseline_complete": True,
                  "checkpoint": str(checkpoint), "evaluation": evaluation, "completed": completed,
                  "claim": "正式初始策略与冻结基线；尚无候选/对照实验或RSI增益"}
    result.update({"promotion_performed": False, "real_robot_commands_sent": False,
                   "evaluation_workers": args.evaluation_workers, "parallel_verification": parallel_evidence,
                   "checkpoint_selection": "预先固定最终训练步数，不根据冻结子集成功率挑选检查点"})
    (directory / "baseline_pipeline.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    progress("completed", baseline_complete=result["baseline_complete"], checkpoint=str(checkpoint))
    return result


if __name__ == "__main__":
    def interrupted(signum, frame):
        raise RuntimeError(f"基线编排收到终止信号{signum}，停止本次子进程组")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", type=pathlib.Path, required=True)
    parser.add_argument("--frozen-eval", type=pathlib.Path, default=ROOT / "eval/frozen_eval_set.json")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--first-steps", type=int, default=500)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=pathlib.Path)
    parser.add_argument("--evaluation-workers", type=int, choices=[1, 2, 4], default=1)
    parser.add_argument("--parallel-verification", type=pathlib.Path)
    parser.add_argument("--after-verification", type=pathlib.Path, help="等待指定预检probe.json通过后才开始续训")
    parser.add_argument("--wait-timeout", type=int, default=3600)
    parser.add_argument("--pilot-only", action="store_true")
    arguments = parser.parse_args()
    arguments.cache = arguments.cache.resolve(strict=True)
    arguments.frozen_eval = arguments.frozen_eval.resolve(strict=True)
    if arguments.resume:
        arguments.resume = arguments.resume.resolve(strict=True)
    if arguments.parallel_verification:
        arguments.parallel_verification = arguments.parallel_verification.resolve(strict=True)
    if arguments.after_verification:
        arguments.after_verification = arguments.after_verification.resolve()
    if not arguments.cache.is_relative_to(ROOT / "data") or not arguments.frozen_eval.is_relative_to(ROOT / "eval") \
            or (arguments.resume and not arguments.resume.is_relative_to(ROOT / "runs/p0")) \
            or (arguments.parallel_verification and not arguments.parallel_verification.is_relative_to(ROOT / "runs/p0")) \
            or (arguments.after_verification and (not arguments.resume
                or not arguments.after_verification.is_relative_to(ROOT / "runs/p0"))) \
            or arguments.wait_timeout < 1 \
            or not 0 < arguments.first_steps <= arguments.steps or arguments.batch < 1:
        parser.error("数据、冻结集和检查点必须位于项目目录；步数与batch必须有效")
    raise SystemExit(run_stage("baseline_pipeline", lambda directory: baseline(directory, arguments),
                              {key: str(value) if isinstance(value, pathlib.Path) else value
                               for key, value in vars(arguments).items()}))
