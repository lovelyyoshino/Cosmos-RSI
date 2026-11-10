#!/usr/bin/env python3
"""只读核验确切串行队列与在途评测；进程存在和工程完成都不代表策略改善。

@author pony
@date 2026-11-09
@version v1.0.0
@last_modified 2026-11-09
@changelog
    - v1.0.0 (2026-11-09): 核对进程身份、确切上游、输入摘要及封存证据，不扫描最新任务。
"""
from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import hashlib
import json
import os
import pathlib

from p0_common import ROOT, run_stage


def checksum(path):
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project(root, value):
    path = (root / value).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError("队列输入越出本项目")
    return path


def logged_run(root, log):
    paths = [project(root, line.split("：", 1)[1]) for line in log.read_text().splitlines()
             if line.startswith("记录目录：")]
    if len(paths) > 1 or any(not p.is_relative_to(root / "runs/p0") for p in paths):
        raise ValueError("确切任务日志包含多个阶段目录或越界，不能按最新目录猜测")
    return paths[0] if paths else None


def read_process(pid):
    try:
        return pathlib.Path(f"/proc/{pid}/cmdline").read_bytes().rstrip(b"\x00").decode().split("\x00")
    except FileNotFoundError:
        return []


def check_command(root, command, specification, upstream_path):
    if not command or not any(value in {specification["script"], str(root / specification["script"])}
                              for value in command):
        raise ValueError("启动PID的实际脚本身份不符或已经退出")
    flag = specification["upstream_flag"]
    indexes = [index for index, value in enumerate(command) if value == flag]
    if len(indexes) != 1 or indexes[0] + 1 == len(command) \
            or project(root, command[indexes[0] + 1]) != upstream_path:
        raise ValueError("实际进程没有等待声明的确切上游任务")


def check_iteration_process(root, command, specification, state):
    if not command or not any(value in {specification["script"], str(root / specification["script"])}
                              for value in command):
        raise ValueError("整圈尚未封存且原PID身份不符或已经退出")
    for name in ["parent", "handoff", "incumbent_evaluation", "budget"]:
        flag = "--" + name.replace("_", "-")
        indexes = [index for index, value in enumerate(command) if value == flag]
        if len(indexes) != 1 or indexes[0] + 1 == len(command) \
                or project(root, command[indexes[0] + 1]) != project(root, state["inputs"][name]["path"]):
            raise ValueError("整圈PID没有使用本圈绑定的实际输入")
    indexes = [index for index, value in enumerate(command) if value == "--iteration"]
    if len(indexes) != 1 or indexes[0] + 1 == len(command) or command[indexes[0] + 1] != str(state["iteration"]):
        raise ValueError("整圈PID的实际圈数不符")


def waiting_inputs(root, log):
    """只查等待时实际输出的绑定，避免用当前配置反过来覆盖实验原声明。"""
    first = log.read_text().splitlines()[0]
    record = json.loads(first)
    bindings = record.get("inputs", {})
    if not bindings and "input" in record:
        path = project(root, record["input"])
        if checksum(path) != record["input_sha256"]:
            raise ValueError("新生成等待时的输入清单已改变")
        bindings = json.loads(path.read_text())
    checked = 0
    for row in bindings.get("files", {}).values():
        if isinstance(row, dict) and {"path", "sha256"} <= row.keys():
            if checksum(project(root, row["path"])) != row["sha256"]:
                raise ValueError("等待期间实际输入摘要改变")
            checked += 1
    for name, digest in bindings.get("dependencies_sha256", {}).items():
        if checksum(project(root, name)) != digest:
            raise ValueError("等待期间实际源码依赖改变")
        checked += 1
    if checked == 0 or not str(record.get("phase", "")).startswith("waiting_for_exact_"):
        raise ValueError("缺少可核验的确切等待声明与输入摘要")
    return checked


def sealed_result(root, run, specification):
    probe_path = run / "probe.json"
    if not probe_path.exists():
        return None
    probe = json.loads(probe_path.read_text())
    if probe.get("stage") != specification["stage"]:
        raise ValueError("封存阶段与预声明不符")
    if probe.get("status") == "failed":
        return {"status": "failed", "probe": str(probe_path), "error": probe.get("error")}
    if probe.get("status") != "passed":
        raise ValueError("封存状态不合法")
    artifact = project(root, run / specification["artifact"])
    if probe.get("artifacts_sha256", {}).get(artifact.name) != checksum(artifact):
        raise ValueError("封存产物摘要不符")
    result = json.loads(artifact.read_text())
    return {"status": "sealed", "artifact": str(artifact), "artifact_sha256": checksum(artifact),
            "decision": result.get("decision"), "wall_s": probe.get("wall_s"),
            "peak_mem_gb": probe.get("peak_mem_gb"), "probe": str(probe_path)}


def inspect_job(root, specification, upstream_path=None, *, process_reader=read_process):
    row = {"id": specification["id"], "label": specification["label"], "errors": []}
    try:
        if "run" in specification:
            run = project(root, specification["run"])
            result = sealed_result(root, run, specification)
            if result:
                row.update(result)
            else:
                state = json.loads((run / "iteration_state.json").read_text())
                pid = specification["pid"]
                if type(pid) is not int or pid < 1:
                    raise ValueError("在途整圈缺少确切正整数PID")
                check_iteration_process(root, process_reader(pid), specification, state)
                row.update(status="running_unsealed", phase=state["phase"], pid=pid)
            return row
        execution = project(root, specification["execution"])
        launch_path = execution / "launch.json"
        launch = json.loads(launch_path.read_text())
        log = project(root, execution / specification["log"])
        if project(root, launch["log"]) != log or type(launch.get("pid")) is not int or launch["pid"] < 1:
            raise ValueError("确切启动记录的日志或PID不合法")
        run = logged_run(root, log)
        # 完成后允许进程已退出，但完成只按该确切日志指向的封存产物认定。
        result = sealed_result(root, run, specification) if run else None
        if result:
            row.update(result)
            return row
        command = process_reader(launch["pid"])
        check_command(root, command, specification, upstream_path)
        checked = waiting_inputs(root, log)
        row.update(status="running" if run else "waiting", pid=launch["pid"],
                   execution=str(execution), launch_sha256=checksum(launch_path),
                   log=str(log), checked_waiting_bindings=checked, upstream=specification["upstream"])
        if run:
            row["run"] = str(run)
    except (ValueError, KeyError, OSError, TypeError, IndexError) as error:
        row.update(status="invalid", errors=[str(error)])
    return row


def evaluation_progress(root, iteration):
    state = json.loads((iteration / "iteration_state.json").read_text())
    phase = state["phase"]
    if phase not in {"evaluate_candidate", "evaluate_control"}:
        return None
    run = logged_run(root, iteration / "logs" / f"{phase}.log")
    if run is None:
        return None
    frozen = json.loads((root / "eval/frozen_eval_set.json").read_text())
    expected = {row["eval_id"] for row in frozen["episodes"]}
    # 封存前只读worker的唯一回合，不把结束后合并清单再数一遍。
    files = sorted(run.glob("*/worker_*/episodes.jsonl"))
    identities, incomplete_lines = set(), 0
    for path in files:
        content = path.read_bytes()
        for index, line in enumerate(content.splitlines()):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if index == len(content.splitlines()) - 1 and not content.endswith(b"\n"):
                    incomplete_lines += 1
                    continue
                raise ValueError("在途回合日志含非末尾损坏记录")
            identity = row["eval_id"]
            if identity not in expected or identity in identities or row["is_eval"] is not True:
                raise ValueError("在途回合混入非冻结样本或重复ID")
            identities.add(identity)
    return {"phase": phase, "run": str(run), "completed_episode_records": len(identities),
            "planned_episode_records": len(expected), "incomplete_trailing_lines": incomplete_lines,
            "sealed": (run / "probe.json").exists(), "success_rates_reported": False,
            "boundary": "只表示记录进度；完整统计结论以封存门禁为准"}


def audit(directory, args):
    import yaml

    config = yaml.safe_load(args.config.read_text())
    jobs = config["jobs"]
    ids = [row["id"] for row in jobs]
    if config.get("schema") != "thor-rsi-execution-v1" or len(ids) != len(set(ids)) or not jobs:
        raise ValueError("执行清单版本或任务ID不合法")
    paths, rows = {}, []
    for specification in jobs:
        upstream = specification.get("upstream")
        if upstream is not None and upstream not in paths:
            raise ValueError("执行清单上游缺失或依赖顺序错误")
        rows.append(inspect_job(ROOT, specification, paths.get(upstream)))
        paths[specification["id"]] = project(ROOT, specification.get("execution", specification.get("run")))
    iteration = project(ROOT, config["iteration_run"])
    environments = {name: (ROOT / "envs" / name / "bin/python").is_file() for name in config["environments"]}
    result = {"schema": config["schema"], "at": dt.datetime.now().astimezone().isoformat(),
        "config_sha256": checksum(args.config), "compute": "Thor CPU只读；无模型加载、训练或运动",
        "environments_present": environments, "jobs": rows,
        "states": dict(Counter(row["status"] for row in rows)),
        "configuration_and_queue_valid": all(row["status"] not in {"invalid", "failed"} for row in rows)
                                         and all(environments.values()),
        "evaluation_progress": evaluation_progress(ROOT, iteration),
        "tasks_started_or_stopped": 0, "complete_rsi": False,
        "boundary": "配置、等待及工程状态不能代替独立校准、合格生成、正式晋升或连续对照收益"}
    (directory / "workflow_status.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return {key: result[key] for key in ["configuration_and_queue_valid", "states", "evaluation_progress", "complete_rsi"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=pathlib.Path, default=ROOT / "configs/p1_execution.yaml")
    args = parser.parse_args()
    args.config = project(ROOT, args.config)
    if ROOT != pathlib.Path("/home/thor/rsi") or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        parser.error("本入口仅在Thor本项目执行，须显式CUDA_VISIBLE_DEVICES=''，不加载模型")
    raise SystemExit(run_stage("p1_workflow_status", lambda directory: audit(directory, args),
                              {"compute": "Thor CPU只读队列核验，batch/训练步数/种子不适用",
                               "memory": "短时CPU核验可能没有tegrastats采样；peak_mem_gb=null不能解读为零"}))
