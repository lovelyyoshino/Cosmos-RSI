#!/usr/bin/env python3
"""将独立运行和盲判结果交接为失败矩阵、困难记录及环境成功轨迹池。

@author pony
@date 2026-11-09
@version v1.1.0
@last_modified 2026-11-09
@changelog
    - v1.1.0 (2026-11-09): 仿真成功池和条件差异由环境真值决定；未校准Reason只作诊断。
    - v1.0.0 (2026-11-09): 环境真值统计、样本不足保留未知、成功源单独准入且绑定实际文件。
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
import pathlib

from p0_common import ROOT, run_stage
from p0_feedback_judge import checked_collection
from p0_lerobot_handoff import dump
from p0_policy_runtime import sha256


def wilson(successes, count):
    if type(successes) is not int or type(count) is not int or not 0 <= successes <= count or count < 1:
        raise ValueError("Wilson计数非法")
    z, p = 1.959963984540054, successes / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return max(0., center - radius), min(1., center + radius)


def summarize(episodes, verdicts, config, calibrated):
    import numpy as np

    if config.get("decision_authority", "environment_truth") != "environment_truth":
        raise ValueError("本入口只接受仿真环境真值，不授予未校准模型判定权")
    ids = [row["episode_id"] for row in episodes]
    judgments = {row["episode_id"]: row for row in verdicts}
    if len(ids) != len(set(ids)) or len(judgments) != len(verdicts) or set(ids) != set(judgments):
        raise ValueError("判官与运行回合不一一对应")
    groups = defaultdict(list)
    for row in episodes:
        judge = judgments[row["episode_id"]]
        if row["is_eval"] is not False or row["outcome"]["source"] != "env" \
                or type(row["outcome"]["success"]) is not bool \
                or row["outcome"]["success"] != judge["environment_truth"] \
                or row["policy_ckpt_sha256"] != judge["policy_ckpt_sha256"]:
            raise ValueError("独立判官角色、环境真值或策略版本错绑")
        groups[(row["task_id"], row["condition"])].append(row)
    matrix, group_stats, thresholds = [], {}, {}
    for key, members in sorted(groups.items()):
        failures = sum(not row["outcome"]["success"] for row in members)
        lower, upper = wilson(failures, len(members))
        value = {"task_id": key[0], "condition": key[1], "failures": failures, "episodes": len(members),
                 "failure_rate": failures / len(members), "wilson95_lower": lower, "wilson95_upper": upper}
        matrix.append(value); group_stats[key] = value
        successful = [row for row in members if row["outcome"]["success"]]
        if len(successful) >= config["hard_success_min_episodes"]:
            thresholds[key] = {metric: float(np.percentile([row["metrics"][metric] for row in successful],
                                                         config["hard_success_percentile"]))
                               for metric in ["duration_s", "action_dispersion_p90"]}
    records, success_pool = [], []
    for episode in episodes:
        judgment = judgments[episode["episode_id"]]
        verdict = judgment["verdict"]
        success = episode["outcome"]["success"]
        key = (episode["task_id"], episode["condition"])
        # 环境成功信号是仿真教师资格依据；未校准模型的误判不能删除正确轨迹。
        if success and episode["metrics"]["steps"] >= 8:
            success_pool.append({"source_episode_id": episode["episode_id"], "source_kind": "rollout_success",
                "source_split": "rollout_train", "is_eval": False, "eligible_for_training": True,
                "policy_ckpt_sha256": episode["policy_ckpt_sha256"], "task_id": episode["task_id"],
                "instruction": episode["instruction"], "init_seed": episode["init_seed"],
                "window": {"start_frame": 0, "end_frame": episode["metrics"]["steps"]},
                "source_length": episode["metrics"]["steps"], "lerobot_index": episode["lerobot_index"],
                "source_success": True, "environment_truth_used": True,
                "independent_judge_agrees": verdict["success"] is True,
                "judge_calibration_passed": calibrated,
                "qualification": "仿真环境成功真值、初态隔离及逐值无损存档；判官意见单列，不替代真值"})
        hits = [f"{metric}>successful_group_p{config['hard_success_percentile']}"
                for metric, threshold in thresholds.get(key, {}).items() if episode["metrics"][metric] > threshold]
        if success and not hits:
            continue
        current, standard = group_stats[key], group_stats.get((key[0], "standard"))
        # 这里只判断环境成败与视觉条件的统计关联；不把Reason解释当成物理因果证明。
        visual = not success and key[1] in {"lighting", "background"} and standard is not None \
            and min(current["episodes"], standard["episodes"]) >= config["min_condition_episodes"] \
            and current["wilson95_lower"] > standard["wilson95_upper"]
        category = "visual" if visual else "unknown"
        route = "cosmos_restyle" if visual else ("self_distill" if success else "ignore")
        records.append({"episode_id": episode["episode_id"], "iter": episode["iter"],
            "policy_ckpt_sha256": episode["policy_ckpt_sha256"], "git_sha": episode["git_sha"],
            "kind": "hard_success" if success else "failure", "category": category,
            "sub_type": key[1] if visual else "other", "route": route,
            "evidence": {"key_frames": [item["frame"] for item in judgment["inputs"]][::2],
                "judge_text": verdict["evidence"], "judge_stage": verdict["failure_stage"],
                "judge_confidence": verdict["confidence"], "rule_hits": hits,
                "environment_truth": success, "judge_success": verdict["success"],
                "condition_statistics": current, "standard_statistics": standard},
            "routing_authority": "environment_truth_condition_comparison",
            "judge_calibration_passed": calibrated, "judge_advisory_only": not calibrated,
            "target_condition": episode["conditions"] if visual else {}, "weight": .5 if success else 1.,
            "eligible_for_generation": bool(visual), "eligible_for_training": False,
            "reason": "独立运行分布存在有足够样本的条件相关弱点" if visual else
                      "保留环境真值困难成功" if route == "self_distill" else "样本或物理归因证据不足，保留未知",
            "is_eval": False})
    return matrix, records, success_pool


def mine(directory, args):
    import yaml

    source, episodes, collection_probe = checked_collection(args.collection)
    report = json.loads(args.judge.read_text())
    probe = json.loads((args.judge.parent / "probe.json").read_text())
    if probe.get("stage") != "feedback_judge" or probe.get("status") != "passed" \
            or probe["artifacts_sha256"].get(args.judge.name) != sha256(args.judge) \
            or report["collection_sha256"] != sha256(args.collection) \
            or pathlib.Path(report["collection"]).resolve(strict=True) != args.collection \
            or report["role"] != "unmodified_base_judge" or report["truth_hidden_from_model"] is not True:
        raise ValueError("独立判官报告或运行来源错绑")
    verdict_path = pathlib.Path(report["verdicts"]).resolve(strict=True)
    if verdict_path.parent != args.judge.parent or probe["artifacts_sha256"].get(verdict_path.name) != sha256(verdict_path):
        raise ValueError("实际判官逐回合输出被替换")
    verdicts = [json.loads(line) for line in verdict_path.read_text().splitlines()]
    config = yaml.safe_load(args.config.read_text())
    if config["version"] != "thor-feedback-mining-v1":
        raise ValueError("挖掘配置版本不符")
    matrix, records, successes = summarize(episodes, verdicts, config, report["calibration"]["calibration_passed"])
    dataset = pathlib.Path(source["dataset"]).resolve(strict=True)
    if not dataset.is_relative_to(args.collection.parent / "rollouts"):
        raise ValueError("实际运行数据集不在采集目录")
    checksums = {relative: expected for relative, expected in collection_probe["artifacts_sha256"].items()
                 if relative.startswith("rollouts/lerobot/")}
    if not checksums or source["dataset_audit"]["labels_equal"] is not True:
        raise ValueError("没有真实运行数据及标签审计")
    for relative, expected in checksums.items():
        if sha256(args.collection.parent / relative) != expected:
            raise ValueError("真实LeRobot运行存档已改变")
    provenance = json.loads((dataset / "meta/rsi_provenance.json").read_text())
    for success in successes:
        episode = provenance["episodes"][success["lerobot_index"]["episode_index"]]
        if episode["source_episode_id"] != success["source_episode_id"] or episode["source_success"] is not True \
                or episode["source_split"] != "runtime" or episode["source_window"] != success["window"]:
            raise ValueError("成功资格与实际运行存档错绑")
    target = directory / "failures"
    target.mkdir()
    (target / "failure_records.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
    (directory / "success_manifest.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in successes))
    with (target / "failure_matrix.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(matrix[0])); writer.writeheader(); writer.writerows(matrix)
    result = {"schema": config["version"], "collection": str(args.collection), "collection_sha256": sha256(args.collection),
        "judge": str(args.judge), "judge_sha256": sha256(args.judge), "config": config,
        "sample_count": len(episodes), "failure_records": len(records), "qualified_successes": len(successes),
        "generation_routes": sum(row["route"] == "cosmos_restyle" for row in records),
        "decision_authority": "environment_truth", "environment_truth_owns_success_pool": True,
        "judge_advisory_only": not report["calibration"]["calibration_passed"],
        "judge_calibration": report["calibration"], "policy_ckpt_sha256": source["policy_ckpt_sha256"],
        "success_manifest": str(directory / "success_manifest.jsonl"), "dataset_files_sha256": checksums,
        "is_eval": False, "training_performed": False, "promotion_performed": False,
        "claim": "环境真值失败矩阵及成功源交接；条件差异只作关联证据，不把判官解释称为因果或RSI收益"}
    dump(directory / "mining_report.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--collection", type=pathlib.Path, required=True)
    parser.add_argument("--judge", type=pathlib.Path, required=True)
    parser.add_argument("--config", type=pathlib.Path, default=ROOT / "configs/mining.yaml")
    args = parser.parse_args()
    for name in ["collection", "judge", "config"]:
        setattr(args, name, getattr(args, name).resolve(strict=True))
    if not args.judge.is_relative_to(ROOT / "runs/p0"):
        parser.error("判官报告必须位于本项目P0目录")
    raise SystemExit(run_stage("feedback_mine", lambda directory: mine(directory, args),
                     {key: str(value) for key, value in vars(args).items()}))
