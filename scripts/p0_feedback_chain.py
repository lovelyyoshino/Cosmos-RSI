#!/usr/bin/env python3
"""串行执行已验证的独立采集、Reason判官与反馈交接，保留每阶段日志。

@author pony
@date 2026-11-09
@version v1.2.0
@last_modified 2026-11-09
@changelog
    - v1.2.0 (2026-11-09): 将具体父权重晋升记录传给后续圈真实采集，不扫描最新晋升。
    - v1.1.0 (2026-11-09): 显式传递独立采集配置和worker数；不改变冻结评测。
    - v1.0.0 (2026-11-09): 新增真实反馈三阶段入口；不把它标成已完成训练部署或多圈RSI。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import signal
import subprocess

from p0_common import ROOT, ensure_gpu_idle, run_stage, stop_process_group
from p0_lerobot_handoff import dump
from p0_policy_runtime import sha256


def execute_stage(directory, phase, script, expected_stage, artifact, arguments):
    ensure_gpu_idle()
    log_path = directory / "logs" / f"{phase}.log"
    with log_path.open("x") as log:
        process = subprocess.Popen([str(ROOT / "envs/policy/bin/python"), "-u", str(ROOT / "scripts" / script),
            *map(str, arguments)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        dump(directory / "progress.json", {"phase": phase, "pid": process.pid, "log": str(log_path)})
        try:
            process.wait()
        finally:
            cleanup = stop_process_group(process)
        if process.returncode or cleanup["forced_kill"]:
            raise RuntimeError(f"反馈阶段{phase}失败，见{log_path}")
    paths = [pathlib.Path(line.split("：", 1)[1]) for line in log_path.read_text().splitlines()
             if line.startswith("记录目录：")]
    if len(paths) != 1 or not paths[0].is_relative_to(ROOT / "runs/p0"):
        raise ValueError("反馈阶段实际目录不唯一或越界")
    probe = json.loads((paths[0] / "probe.json").read_text())
    result = paths[0] / artifact
    if probe["status"] != "passed" or probe["stage"] != expected_stage \
            or probe["artifacts_sha256"].get(artifact) != sha256(result):
        raise ValueError("反馈阶段产物未通过实际运行校验")
    ensure_gpu_idle()
    return result


def chain(directory, args):
    parent_args = ["--parent-promotion", args.parent_promotion] if args.parent_promotion else []
    collection = execute_stage(directory, "collection", "p0_feedback_rollout.py", "feedback_rollout", "feedback_rollout.json",
        ["--checkpoint", args.checkpoint, "--conditions", *args.conditions,
         "--init-seed", args.init_seed, "--policy-seed", args.policy_seed, "--per-condition", args.per_condition,
         "--config", args.rollout_config, "--workers", args.workers, *parent_args])
    judge = execute_stage(directory, "judge", "p0_feedback_judge.py", "feedback_judge", "judge_report.json",
                          ["--collection", collection, "--config", args.judge_config])
    mining = execute_stage(directory, "mining", "p0_feedback_mine.py", "feedback_mine", "mining_report.json",
                           ["--collection", collection, "--judge", judge, "--config", args.mining_config])
    report = {"scope": "独立运行→Reason关键帧复核→失败矩阵/成功源交接", "completed_feedback_chain": True,
        "collection": str(collection), "judge": str(judge), "mining": str(mining),
        "sample_count": json.loads(collection.read_text())["sample_count"],
        "calibration": json.loads(judge.read_text())["calibration"],
        "training_performed": False, "generation_performed": False, "promotion_performed": False,
        "complete_rsi": False, "real_robot_commands_sent": False,
        "artifacts_sha256": {str(path): sha256(path) for path in [collection, judge, mining]}}
    dump(directory / "feedback_chain.json", report)
    dump(directory / "progress.json", {"phase": "completed_feedback_only", "report": str(directory / "feedback_chain.json")})
    return report


if __name__ == "__main__":
    def interrupted(number, _frame):
        raise RuntimeError(f"反馈编排收到信号{number}，清理本阶段进程组")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--parent-promotion", type=pathlib.Path)
    parser.add_argument("--judge-config", type=pathlib.Path, default=ROOT / "configs/judge_calibration.yaml")
    parser.add_argument("--rollout-config", type=pathlib.Path, default=ROOT / "configs/rollout_budget.yaml")
    parser.add_argument("--mining-config", type=pathlib.Path, default=ROOT / "configs/mining.yaml")
    parser.add_argument("--conditions", nargs="+", default=["t00-standard", "t00-lighting", "t00-background"])
    parser.add_argument("--all-tasks", action="store_true", help="从既定条件注册表取全部十任务的标准/光照/纹理，不读取评测结果")
    parser.add_argument("--per-condition", type=int, default=1)
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=1)
    parser.add_argument("--init-seed", type=int, default=62000)
    parser.add_argument("--policy-seed", type=int, default=72000)
    args = parser.parse_args()
    args.checkpoint = args.checkpoint.resolve(strict=True)
    if args.parent_promotion:
        args.parent_promotion = args.parent_promotion.resolve(strict=True)
    args.judge_config, args.mining_config = args.judge_config.resolve(strict=True), args.mining_config.resolve(strict=True)
    args.rollout_config = args.rollout_config.resolve(strict=True)
    if args.all_tasks:
        registry = json.loads((ROOT / "eval/frozen_eval_set.json").read_text())
        args.conditions = [row["condition_id"] for row in registry["conditions"]
                           if row["condition"] in {"standard", "lighting", "background"}]
        if len(args.conditions) != 30:
            parser.error("全任务条件注册表不符合原有十任务范围")
    if args.per_condition < 1:
        parser.error("每条件回合数必须为正")
    raise SystemExit(run_stage("feedback_chain", lambda directory: chain(directory, args),
        {key: str(value) if isinstance(value, pathlib.Path) else value for key, value in vars(args).items()}))
