"""等待确切上游退出并释放CUDA上下文；锁与产物封存不能代替进程退出。

@author pony
@date 2026-11-09
@version v1.0.0
@last_modified 2026-11-09
@changelog
    - v1.0.0 (2026-11-09): 有界等待已绑定进程，拒绝其他GPU占用、PID复用及查询失败。
"""
from __future__ import annotations

import pathlib
import time

from p0_common import command


def process_command(pid):
    try:
        return pathlib.Path(f"/proc/{pid}/cmdline").read_bytes().rstrip(b"\x00").decode().split("\x00")
    except FileNotFoundError:
        return []


def wait_stage_teardown(binding, timeout=30.):
    """只等待该前序；不结束进程，也不等待或忽略不相关的GPU任务。"""
    pid, expected = binding["pid"], binding["command"]
    if type(pid) is not int or pid < 1 or not isinstance(expected, list) or not expected \
            or any(not isinstance(item, str) or not item for item in expected) \
            or type(timeout) not in {int, float} or not 0 < timeout <= 60:
        raise ValueError("阶段退出等待必须绑定正整数PID、确切命令和至多60秒时限")
    started = time.monotonic()
    while True:
        actual = process_command(pid)
        # zombie的cmdline为空；它已结束执行，CUDA条目仍须独立确认清空。
        live = bool(actual and actual != [""])
        if live and actual != expected:
            raise RuntimeError("上游PID已被其他进程复用，禁止据旧PID继续")
        query = command(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"])
        if query.get("returncode") != 0:
            raise RuntimeError(f"阶段退出时GPU查询失败：{query}")
        gpu_pids = set()
        for line in query.get("stdout", "").splitlines():
            try:
                value = int(line.split(",", 1)[0].strip())
            except ValueError as error:
                raise RuntimeError("阶段退出时GPU查询返回了不可识别的PID") from error
            if value < 1:
                raise RuntimeError("阶段退出时GPU查询返回了非法PID")
            gpu_pids.add(value)
        if gpu_pids - {pid}:
            raise RuntimeError(f"检测到其他GPU任务，不等待或抢占：{sorted(gpu_pids - {pid})}")
        if not live and not gpu_pids:
            return {"predecessor_pid": pid, "process_exited": True, "gpu_idle": True,
                    "wait_wall_s": time.monotonic() - started, "processes_stopped": 0}
        remaining = timeout - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError(f"确切前序{pid}退出或CUDA释放超时；禁止启动下一模型")
        time.sleep(min(.2, remaining))
