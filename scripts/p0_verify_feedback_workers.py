#!/usr/bin/env python3
"""逐值核对独立反馈的并行回合，保持原始轨迹和冻结评测不变。

@author pony
@date 2026-11-09
@version v1.0.0
@last_modified 2026-11-09
@changelog
    - v1.0.0 (2026-11-09): 允许复用既有串行参考的明确子集，不以不同规模的耗时计算提速。
"""
from __future__ import annotations

import argparse
import json
import pathlib

from p0_common import ROOT, run_stage
from p0_feedback_judge import checked_collection
from p0_lerobot_handoff import dump
from p0_policy_runtime import sha256


def case_key(row):
    return row["condition_id"], row["init_seed"], row["policy_seed"]


def compare_case(first, second, left, right):
    import numpy as np

    fields = ["condition_id", "init_seed", "policy_seed", "policy_ckpt_sha256", "instruction",
              "conditions", "initial_scene_sha256", "is_eval", "outcome", "precision", "batch",
              "action_samples", "max_steps", "control_hz", "action_horizon", "policy_noise_schedule",
              "terminal_reason", "dispersion", "isolation"]
    differences = [key for key in fields if first[key] != second[key]]
    for key in ["steps", "sim_duration_s", "policy_calls", "action_dispersion_p90",
                "regrasp_count", "gripper_reclose_command_count"]:
        if first["metrics"][key] != second["metrics"][key]:
            differences.append("metrics." + key)
    arrays = {}
    required = {"action", "state", "agentview", "wrist", "reward", "request_seed", "final_sim_state"}
    if set(left) != required or set(right) != required:
        raise ValueError("反馈原始数组的字段不完整或出现未审计字段")
    for name in sorted(required):
        a, b = left[name], right[name]
        finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
        exact = bool(a.dtype == b.dtype and a.shape == b.shape and finite and np.array_equal(a, b))
        arrays[name] = {"exact": exact, "finite": finite, "same_dtype": a.dtype == b.dtype,
                        "same_shape": a.shape == b.shape,
                        "max_abs_diff": float(np.max(np.abs(a.astype(float) - b.astype(float))))
                        if finite and a.shape == b.shape and a.size else None}
        if not exact:
            differences.append(name)
    return {"case": list(case_key(first)), "passed": not differences,
            "differing_fields": differences, "arrays": arrays}


def verified_trajectory(row, report_path, probe):
    path = pathlib.Path(row["trajectory"]).resolve(strict=True)
    if not path.is_relative_to(report_path.parent):
        raise ValueError("轨迹越出其实际采集目录")
    checksum = sha256(path)
    if checksum != row["trajectory_sha256"] \
            or probe["artifacts_sha256"].get(str(path.relative_to(report_path.parent))) != checksum:
        raise ValueError("采集轨迹与真实probe的摘要不符")
    return path


def verify(directory, args):
    import numpy as np

    serial, first, serial_probe = checked_collection(args.serial)
    parallel, second, parallel_probe = checked_collection(args.parallel)
    if serial.get("workers_requested", 1) != 1 or parallel.get("workers_requested") != 4 \
            or max(parallel.get("actual_workers", {}).values(), default=0) != 4 \
            or serial["config"] != parallel["config"] \
            or serial["policy_ckpt_sha256"] != parallel["policy_ckpt_sha256"]:
        raise ValueError("需要相同配方和策略的串行参考及实际四worker采集")
    reference = {case_key(row): row for row in first}
    if len(reference) != len(first) or len({case_key(row) for row in second}) != len(second) \
            or not {case_key(row) for row in second} <= reference.keys():
        raise ValueError("并行回合没有唯一、同条件同种子的串行参考")
    comparisons = []
    for row in second:
        old = reference[case_key(row)]
        a = verified_trajectory(old, args.serial, serial_probe)
        b = verified_trajectory(row, args.parallel, parallel_probe)
        with np.load(a, allow_pickle=False) as left, np.load(b, allow_pickle=False) as right:
            result = compare_case(old, row, left, right)
        terminal_equal = {}
        for view in ["agentview", "wrist"]:
            checksums = []
            for episode, path, probe in [(old, args.serial, serial_probe), (row, args.parallel, parallel_probe)]:
                terminal = (pathlib.Path(episode["artifacts"]) / f"{view}_last.png").resolve(strict=True)
                if not terminal.is_relative_to(path.parent) \
                        or probe["artifacts_sha256"].get(str(terminal.relative_to(path.parent))) != sha256(terminal):
                    raise ValueError("终帧没有绑定真实采集证据")
                checksums.append(sha256(terminal))
            terminal_equal[view] = checksums[0] == checksums[1]
        result["terminal_png_equal"] = terminal_equal
        result["passed"] = result["passed"] and all(terminal_equal.values())
        comparisons.append(result)
    result = {"schema": "thor-feedback-workers-equivalence-v1", "passed": all(r["passed"] for r in comparisons),
              "serial": str(args.serial), "serial_sha256": sha256(args.serial),
              "parallel": str(args.parallel), "parallel_sha256": sha256(args.parallel),
              "sample_count": len(comparisons), "policy_ckpt_sha256": parallel["policy_ckpt_sha256"],
              "precision": parallel["precision"], "batch": parallel["batch"], "config": parallel["config"],
              "initial_seeds": parallel["init_seeds"], "policy_seeds": parallel["policy_seeds"],
              "actual_workers": parallel["actual_workers"], "comparisons": comparisons,
              "parallel_stage_wall_s": parallel_probe["wall_s"], "parallel_peak_mem_gb": parallel_probe["peak_mem_gb"],
              "speedup_measured": False, "eligible_for_training": False,
              "boundary": "复用串行参考的明确子集；不同规模不计算提速；诊断重复回合不作新增独立样本"}
    dump(directory / "worker_equivalence.json", result)
    if not result["passed"]:
        raise ValueError("并行反馈轨迹与串行参考不一致，禁止开始扩量")
    return {key: value for key, value in result.items() if key != "comparisons"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--serial", type=pathlib.Path, required=True)
    parser.add_argument("--parallel", type=pathlib.Path, required=True)
    args = parser.parse_args()
    args.serial, args.parallel = args.serial.resolve(strict=True), args.parallel.resolve(strict=True)
    raise SystemExit(run_stage("feedback_workers_verification", lambda directory: verify(directory, args),
                              {"compute": "Thor CPU逐值核对真实轨迹，无新策略运行"}))
