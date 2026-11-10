#!/usr/bin/env python3
"""在Thor独立槽位演练正式权重的服务切换、延迟及回滚，不晋升策略。

@author pony
@date 2026-11-07
@version v1.0.0
@last_modified 2026-11-07
@changelog
    - v1.0.0 (2026-11-07): 仅在本次运行目录切换链接，校验HTTP身份、动作和正常退出。
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
import pathlib
import signal
import subprocess
import time
import urllib.error

from p0_common import ROOT, ensure_gpu_idle, run_stage, stop_process_group
from p0_evaluate_policy import request_json, validate_loopback_port_available
from p0_policy_runtime import DemoCache, sha256


def atomic_slot(directory: pathlib.Path, name: str, target: pathlib.Path, root: pathlib.Path) -> None:
    """只替换演练目录内的软链接，拒绝覆盖普通文件或越过本次槽位边界。"""
    root = root.resolve(strict=True)
    directory = directory.resolve(strict=True)
    target = target.resolve(strict=True)
    if name not in {"current", "previous"} or not directory.is_relative_to(root) or not target.is_relative_to(root):
        raise ValueError("槽位名称或路径越界")
    destination = directory / name
    if destination.exists() and not destination.is_symlink():
        raise ValueError("拒绝覆盖非链接槽位")
    temporary = directory / f".{name}.{os.getpid()}.tmp"
    temporary.symlink_to(target, target_is_directory=True)
    try:
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def validate_response(response: dict, expected_hash: str) -> None:
    import numpy as np

    if response.get("state_sha256") != expected_hash:
        raise ValueError("响应权重摘要与当前槽位不符")
    for key in ["actions", "normalized_actions"]:
        actions = np.asarray(response[key], dtype=np.float64)
        if actions.shape != (8, 7) or not np.isfinite(actions).all():
            raise ValueError("服务动作不是有限的8×7数组")
    latency = response.get("latency_ms")
    if isinstance(latency, bool) or not isinstance(latency, (int, float)) or not np.isfinite(latency) or latency <= 0:
        raise ValueError("服务延迟非法")


def latency_summary(rows: list[dict], warmup: int, repeats: int) -> dict:
    import numpy as np

    if warmup < 0 or repeats < 1 or len(rows) != warmup + repeats:
        raise ValueError("预热或实测请求数不符")
    measured = rows[warmup:]
    if any(row["request_index"] != index for index, row in enumerate(rows)):
        raise ValueError("请求序号重复或缺失")
    result = {"warmup": warmup, "measured_over": repeats}
    for key in ["server_ms", "http_ms"]:
        values = np.asarray([row[key] for row in measured], dtype=np.float64)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError("延迟必须是有限正数")
        result[key] = {"p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95)),
                       "min": float(values.min()), "max": float(values.max())}
    return result


def server_phase(directory: pathlib.Path, name: str, slot: pathlib.Path, payload: dict,
                 port: int, warmup: int, repeats: int) -> dict:
    ensure_gpu_idle()
    validate_loopback_port_available(port)
    checkpoint = slot.resolve(strict=True)
    metadata = json.loads((checkpoint / "checkpoint.json").read_text())
    expected_hash = metadata["state_sha256"]
    command = [str(ROOT / "envs/policy/bin/python"), "-u", str(ROOT / "scripts/p0_policy_server.py"),
               "--checkpoint", str(slot), "--port", str(port)]
    log_path = directory / "logs" / f"{name}_server.log"
    rows = []
    response = None
    with log_path.open("x") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
        started = time.monotonic()
        interrupted = False
        cleanup = None
        try:
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"{name}服务在ready前退出，见{log_path}")
                try:
                    health = request_json(f"http://127.0.0.1:{port}/health", timeout=3)
                    if health.get("pid") != process.pid or health.get("state_sha256") != expected_hash \
                            or health.get("checkpoint") != str(checkpoint) or health.get("status") != "ready":
                        raise RuntimeError("健康检查连接到错误服务或权重")
                    break
                except (urllib.error.URLError, TimeoutError):
                    if time.monotonic() - started > 180:
                        raise RuntimeError(f"{name}服务ready超时")
                    time.sleep(0.25)
            ready_wall = time.monotonic() - started
            with (directory / "logs" / f"{name}_requests.jsonl").open("x") as journal:
                for index in range(warmup + repeats):
                    before = time.perf_counter()
                    response = request_json(f"http://127.0.0.1:{port}/predict", payload, timeout=90)
                    http_ms = (time.perf_counter() - before) * 1000
                    validate_response(response, expected_hash)
                    row = {"phase": name, "request_index": index, "warmup": index < warmup,
                           "server_ms": response["latency_ms"], "http_ms": http_ms,
                           "state_sha256": expected_hash, "noise_seed": payload["seed"]}
                    rows.append(row)
                    journal.write(json.dumps(row) + "\n")
                    journal.flush()
                    if index == 0 or (index + 1) % 50 == 0:
                        print(json.dumps({"phase": name, "requests": index + 1,
                                          "total": warmup + repeats}), flush=True)
        except BaseException:
            interrupted = True
            raise
        finally:
            cleanup = stop_process_group(process, grace_s=30)
            shutdown = {"pid": process.pid, "returncode": process.returncode, "interrupted": interrupted, **cleanup}
            (directory / f"{name}_shutdown.json").write_text(json.dumps(shutdown, indent=2) + "\n")
    if process.returncode != 0 or cleanup["forced_kill"]:
        raise RuntimeError(f"{name}服务没有正常退出")
    ensure_gpu_idle()
    locations = [line.split("：", 1)[1] for line in log_path.read_text().splitlines() if line.startswith("记录目录：")]
    if len(locations) != 1:
        raise RuntimeError("不能唯一定位服务运行清单")
    server_run = pathlib.Path(locations[0]).resolve(strict=True)
    if not server_run.is_relative_to(ROOT / "runs/p0"):
        raise RuntimeError("服务清单越过项目边界")
    probe = json.loads((server_run / "probe.json").read_text())
    record = json.loads((server_run / "server_run.json").read_text())
    if probe["status"] != "passed" or record["inferences"] != len(rows) or record["errors"] \
            or record["state_sha256"] != expected_hash:
        raise RuntimeError("服务实际记录没有通过验收")
    return {"name": name, "checkpoint": str(checkpoint), "checkpoint_sha256": expected_hash,
            "training_step": metadata["step"], "ready_wall_s": ready_wall, "command": command,
            "latency_ms": latency_summary(rows, warmup, repeats), "rows": rows,
            "response": response, "server_run": str(server_run), "peak_mem_gb": probe["peak_mem_gb"],
            "normal_exit": True}


def verify(directory: pathlib.Path, args: argparse.Namespace) -> dict:
    ensure_gpu_idle()
    import numpy as np
    from PIL import Image

    metadata = json.loads((args.checkpoint / "checkpoint.json").read_text())
    rollback_meta = json.loads((args.rollback / "checkpoint.json").read_text())
    for checkpoint, meta in [(args.checkpoint, metadata), (args.rollback, rollback_meta)]:
        if sha256(checkpoint / "state.pt") != meta["state_sha256"]:
            raise RuntimeError("检查点内容摘要不符")
    if metadata["state_sha256"] == rollback_meta["state_sha256"]:
        raise RuntimeError("演练必须使用不同权重，不能把重载同一模型当作回滚")
    if not metadata.get("formal_baseline_training") or metadata["step"] != 5000:
        raise RuntimeError("当前入口只验证固定5000步正式基线")
    cache = DemoCache(pathlib.Path(metadata["cache_manifest"]).parent)
    if sha256(cache.manifest_path) != metadata["cache_manifest_sha256"]:
        raise RuntimeError("源示范清单已变化")
    location = metadata["prediction_probe"]["locations"][0]
    episodes = [row for row in cache.manifest["episodes"] if row["episode_id"] == location["episode_id"]]
    if len(episodes) != 1 or episodes[0]["split"] != "demo_validation" \
            or not 0 <= location["frame"] < episodes[0]["length"]:
        raise RuntimeError("服务探针必须来自原固定离线验证示范")
    episode = episodes[0]
    offset = episode["offset"] + location["frame"]
    images = []
    for view, label in enumerate(["agentview", "wrist"]):
        image = Image.fromarray(np.array(cache.images[offset, view]))
        image.save(directory / f"deployment_input_{label}.png")
        stream = io.BytesIO()
        image.save(stream, format="PNG")
        images.append(base64.b64encode(stream.getvalue()).decode())
    payload = {"instruction": episode["instruction"], "state": cache.states[offset].tolist(),
               "images_png_base64": images, "seed": args.seed}
    (directory / "request.json").write_text(json.dumps(payload) + "\n")
    frozen = ROOT / "eval/frozen_eval_set.json"
    frozen_before = sha256(frozen)
    production = {str(path): {"exists": path.exists(), "link": os.readlink(path) if path.is_symlink() else None}
                  for path in [ROOT / "checkpoints/current", ROOT / "checkpoints/previous"]}
    slots = directory / "deployment_slots"
    slots.mkdir()
    atomic_slot(slots, "current", args.rollback, ROOT)
    phases = [server_phase(directory, "initial", slots / "current", payload, args.port, 0, 1)]
    atomic_slot(slots, "previous", (slots / "current").resolve(), ROOT)
    atomic_slot(slots, "current", args.checkpoint, ROOT)
    phases.append(server_phase(directory, "formal", slots / "current", payload, args.port, args.warmup, args.repeats))
    atomic_slot(slots, "current", (slots / "previous").resolve(), ROOT)
    phases.append(server_phase(directory, "rollback", slots / "current", payload, args.port, 0, 1))
    before, after = [np.asarray(phase["response"]["normalized_actions"]) for phase in [phases[0], phases[2]]]
    exact = bool(np.array_equal(before, after))
    for value in phases:
        (directory / f"{value['name']}_phase.json").write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    unchanged = frozen_before == sha256(frozen) and all(
        entry == {"exists": pathlib.Path(path).exists(),
                  "link": os.readlink(path) if pathlib.Path(path).is_symlink() else None}
        for path, entry in production.items())
    report = {"iter": 0, "policy_ckpt_sha256": metadata["state_sha256"], "thor_path": str(args.checkpoint),
              "rollback_to": rollback_meta["state_sha256"], "precision": metadata["precision"],
              "batch": 1, "action_horizon": 8, "training_steps": metadata["step"],
              "training_batch": metadata["recipe"]["batch"], "training_seed": metadata["recipe"]["seed"],
              "noise_seed": args.seed, "sample_count": 1,
              "source": {**location, "split": episode["split"], "instruction": episode["instruction"],
                         "cache_manifest_sha256": sha256(cache.manifest_path)},
              "latency_ms": phases[1]["latency_ms"], "phases": [{k: v for k, v in p.items()
                                                                       if k not in {"rows", "response"}} for p in phases],
              "rollback_action_exact": exact, "rollback_max_abs_diff": float(np.max(np.abs(before - after))),
              "production_slots_and_frozen_eval_unchanged": unchanged, "eval_set_sha256": frozen_before,
              "smoke_test": {"episodes": 0, "requests": sum(len(p["rows"]) for p in phases),
                             "shape_and_finiteness_passed": True},
              "latency_includes": {"server_ms": "PNG解码、状态归一化、双视角预处理、VLM、动作头、D2H、反归一化；不含网络、JSON编解码和加载",
                                   "http_ms": "loopback完整HTTP请求含JSON编解码及全部服务计算；无其他客户端排队"},
              "environment_actions_sent": False, "production_deployed": False, "eligible_for_promotion": False,
              "scope": "独立槽位的P0部署演练；一条固定验证观测重复请求，不是机器人回合或策略提升",
              "passed": exact and unchanged}
    (directory / "deploy_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    rows = [row for phase in phases for row in phase["rows"]]
    with (directory / "latency.csv").open("x", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if not report["passed"]:
        raise RuntimeError("回滚动作或生产槽位/冻结集校验失败，见deploy_manifest.json")
    return report


if __name__ == "__main__":
    def interrupted(signum, _frame):
        raise RuntimeError(f"部署演练收到信号{signum}，回收本次服务组")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--rollback", type=pathlib.Path, required=True)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=200)
    arguments = parser.parse_args()
    for key in ["checkpoint", "rollback"]:
        path = getattr(arguments, key).resolve(strict=True)
        if not path.is_relative_to(ROOT / "runs/p0"):
            parser.error("检查点必须位于项目P0目录")
        setattr(arguments, key, path)
    if not 1024 <= arguments.port <= 65535 or not 0 <= arguments.seed < 2 ** 32 \
            or arguments.warmup < 0 or arguments.repeats < 1:
        parser.error("端口、种子或请求数量非法")
    raise SystemExit(run_stage("deployment_smoke", lambda directory: verify(directory, arguments),
                              {key: str(value) if isinstance(value, pathlib.Path) else value
                               for key, value in vars(arguments).items()}))
