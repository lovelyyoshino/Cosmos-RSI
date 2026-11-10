"""P0 单次运行记录：保存失败、环境条件和全机内存采样。

@author pony
@date 2026-11-06
@version v1.4.0
@last_modified 2026-11-06
@changelog
    - v1.4.0 (2026-11-06): 失败时按本次独立进程组回收后代，leader先退出也不跳过清理。
    - v1.3.0 (2026-11-06): 保存上游实际diff，避免只记录SHA而遗漏本地适配。
    - v1.2.0 (2026-11-06): 将当前环境工具放入子进程PATH，支持FlashInfer本地编译。
    - v1.1.0 (2026-11-06): 保留实际源文件副本，支持无提交仓库的运行复现。
    - v1.0.0 (2026-11-06): 统一最小实验的产物、版本和监控记录。
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import pathlib
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback
from collections.abc import Callable

ROOT = pathlib.Path(__file__).resolve().parents[1]


def command(args: list[str], timeout: int = 20) -> dict:
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return {"returncode": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"returncode": None, "error": str(exc)}


def ensure_gpu_idle() -> None:
    busy = command(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"])
    if busy.get("returncode") != 0 or busy.get("stdout"):
        raise RuntimeError(f"GPU 占用检查未通过：{busy}")


def stop_process_group(process: subprocess.Popen, grace_s: float = 30) -> dict:
    """只清理调用者以start_new_session创建的子任务组，等待仍在运行的后代退出。"""
    if process.pid == os.getpgrp():
        raise RuntimeError("拒绝清理当前主线程进程组")

    def live_members():
        process.poll()
        result = subprocess.run(["ps", "-eo", "pid=,pgid=,stat="], capture_output=True, text=True, check=True)
        return [int(parts[0]) for line in result.stdout.splitlines() if len(parts := line.split()) == 3
                and int(parts[1]) == process.pid and not parts[2].startswith("Z")]

    def send(number):
        try:
            os.killpg(process.pid, number)
        except ProcessLookupError:
            pass

    send(signal.SIGTERM)
    deadline = time.monotonic() + grace_s
    while live_members() and time.monotonic() < deadline:
        time.sleep(0.1)
    forced = bool(live_members())
    if forced:
        send(signal.SIGKILL)
    process.wait(timeout=5)
    deadline = time.monotonic() + 5
    while (remaining := live_members()) and time.monotonic() < deadline:
        time.sleep(0.1)
    if remaining:
        raise RuntimeError(f"本次子任务仍有运行中的后代，禁止进入下一阶段：{remaining}")
    return {"pgid": process.pid, "forced_kill": forced, "remaining_live_pids": remaining}


def run_stage(stage: str, callback: Callable[[pathlib.Path], dict], conditions: dict | None = None) -> int:
    if platform.node() != "thor" or platform.machine() != "aarch64" or ROOT != pathlib.Path("/home/thor/rsi"):
        raise RuntimeError("实验只允许在 Thor 的 /home/thor/rsi 运行")
    # 直接调用venv里的Python不会激活PATH；编译器和推理worker也必须使用该环境的工具。
    os.environ["PATH"] = str(pathlib.Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    started = dt.datetime.now(dt.timezone.utc).astimezone()
    run_dir = ROOT / "runs" / "p0" / f"{stage}_{started.strftime('%Y%m%dT%H%M%S_%f%z')}"
    (run_dir / "logs").mkdir(parents=True, exist_ok=False)
    sha = command(["git", "-C", str(ROOT), "rev-parse", "HEAD"])
    record = {
        "stage": stage, "iter": 0, "policy_ckpt_sha256": None,
        "git_sha": sha.get("stdout") if sha.get("returncode") == 0 else None,
        "git_status": command(["git", "-C", str(ROOT), "status", "--porcelain"]),
        "upstream_sha": {p.name: command(["git", "-C", str(p), "rev-parse", "HEAD"]).get("stdout")
                         for p in (ROOT / "third_party").iterdir() if (p / ".git").is_dir()},
        "hardware": {"host": platform.node(), "arch": platform.machine(),
                     "power_mode": command(["sudo", "-n", "nvpmodel", "-q"])},
        "conditions": conditions or {}, "env_prefix": sys.prefix,
        "command": [sys.executable, *sys.argv], "started_at": started.isoformat(),
        "status": "running", "peak_mem_gb": None,
    }
    # 仓库尚无提交或存在未提交修改时，以实际代码哈希补充 git SHA，避免把工作区误当提交版本。
    source = command(["git", "-C", str(ROOT), "ls-files", "--cached", "--others", "--exclude-standard"])
    record["source_files_sha256"] = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in source.get("stdout", "").splitlines()
        if (ROOT / name).is_file() and not (ROOT / name).is_symlink()
    }
    for name in record["source_files_sha256"]:
        snapshot = run_dir / "source_snapshot" / name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, snapshot)
    record["upstream_worktree_diff"] = {}
    for repo in sorted((ROOT / "third_party").iterdir()):
        if not (repo / ".git").is_dir():
            continue
        diff = command(["git", "-C", str(repo), "diff", "--no-ext-diff", "HEAD"])
        if diff.get("returncode") != 0:
            raise RuntimeError(f"无法记录上游修改：{repo.name}")
        if diff["stdout"]:
            artifact = run_dir / "upstream_diff" / f"{repo.name}.patch"
            artifact.parent.mkdir(exist_ok=True)
            artifact.write_text(diff["stdout"] + "\n")
            record["upstream_worktree_diff"][repo.name] = {
                "file": str(artifact), "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()}
    print(f"记录目录：{run_dir}", flush=True)
    monitor_path = run_dir / "logs" / "tegrastats.log"
    monitor_log = monitor_path.open("w")
    monitor = None
    wall_start = time.monotonic()
    try:
        if shutil.which("tegrastats"):
            monitor = subprocess.Popen(["tegrastats", "--interval", "200"], stdout=monitor_log,
                                       stderr=subprocess.STDOUT)
        record["result"] = callback(run_dir)
        record["status"] = "passed"
    except Exception:
        record["status"] = "failed"
        record["error"] = traceback.format_exc()
        print(record["error"], file=sys.stderr, flush=True)
    finally:
        if monitor is not None:
            monitor.terminate()
            try:
                monitor.wait(timeout=5)
            except subprocess.TimeoutExpired:
                monitor.kill()
                monitor.wait()
        monitor_log.close()
        samples = [int(value) for value in re.findall(r"RAM\s+(\d+)/\d+MB", monitor_path.read_text())]
        record["peak_mem_gb"] = max(samples) * 1024 ** 2 / 1e9 if samples else None
        record["peak_mem_definition"] = "tegrastats每200ms采样的全机已用RAM，MB按MiB换算为十进制GB；包含系统和其他CPU进程"
        record["memory_sample_count"] = len(samples)
        record["wall_s"] = time.monotonic() - wall_start
        record["finished_at"] = dt.datetime.now(dt.timezone.utc).astimezone().isoformat()
        record["artifacts_sha256"] = {
            str(path.relative_to(run_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(run_dir.rglob("*")) if path.is_file() and "logs" not in path.relative_to(run_dir).parts
        }
        (run_dir / "probe.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({k: record[k] for k in ["stage", "status", "wall_s", "peak_mem_gb"]}, ensure_ascii=False), flush=True)
    return 0 if record["status"] == "passed" else 1
