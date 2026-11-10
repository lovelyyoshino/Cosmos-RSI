#!/usr/bin/env python3
"""Thor 环境清单与最小 CUDA 验证；每次执行保留独立记录。

@author pony
@date 2026-11-06
@version v1.0.0
@last_modified 2026-11-06
@changelog
    - v1.0.0 (2026-11-06): 记录依赖、conv3d/sdpa/反向传播和内存监控。
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata as metadata
import json
import os
import pathlib
import platform
import re
import shutil
import subprocess
import sys
import time
import traceback

ROOT = pathlib.Path(__file__).resolve().parents[1]


def command(args: list[str], timeout: int = 20) -> dict:
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return {"returncode": result.returncode, "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"returncode": None, "error": str(exc)}


def versions() -> dict[str, str]:
    return {dist.metadata["Name"]: dist.version for dist in metadata.distributions()
            if dist.metadata.get("Name")}


def inventory() -> dict:
    from huggingface_hub import constants, get_token

    cache = pathlib.Path(constants.HF_HUB_CACHE)
    return {
        "os_release": pathlib.Path("/etc/os-release").read_text(),
        "l4t_release": pathlib.Path("/etc/nv_tegra_release").read_text(),
        "python": sys.version,
        "env_prefix": sys.prefix,
        "packages": versions(),
        "memory": command(["free", "-b"]),
        "disk": command(["df", "-B1", str(ROOT)]),
        "cuda": command(["/usr/local/cuda-13.0/bin/nvcc", "--version"]),
        "gpu": command(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]),
        "hf_auth_available": bool(get_token()),
        "hf_cache": str(cache),
        "cached_models": sorted(p.name for p in cache.glob("models--*")) if cache.is_dir() else [],
    }


def torch_smoke(seed: int) -> dict:
    # 在建立本任务 CUDA context 前检查占用，避免干扰其他工程正在运行的计算。
    busy = command(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"])
    if busy.get("returncode") != 0 or busy.get("stdout"):
        raise RuntimeError(f"GPU 占用检查未通过：{busy}")

    import torch
    import torch.nn.functional as functional
    import torchvision

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    info = torch.cuda.get_device_properties(device)

    x = torch.randn(1, 3, 2, 28, 28, device=device, dtype=dtype, requires_grad=True)
    conv = torch.nn.Conv3d(3, 32, kernel_size=(2, 14, 14), stride=(2, 14, 14)).to(device, dtype)
    embedding = conv(x)
    embedding.float().square().mean().backward()
    assert embedding.shape == (1, 32, 1, 2, 2)
    assert torch.isfinite(embedding).all() and torch.isfinite(x.grad).all()

    q, k, v = [torch.randn(1, 2, 16, 32, device=device, dtype=dtype) for _ in range(3)]
    attention = functional.scaled_dot_product_attention(q, k, v)
    reference = ((q.float() @ k.float().transpose(-2, -1)) / (32 ** 0.5)).softmax(-1) @ v.float()
    torch.testing.assert_close(attention.float(), reference, atol=0.02, rtol=0.02)

    # 判官依赖 CUDA Graph；此处只验证运行时支持，不把小矩阵耗时当作模型吞吐。
    matrix = torch.randn(128, 128, device=device, dtype=dtype)
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(3):
            matrix @ matrix
    torch.cuda.current_stream().wait_stream(warmup)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = matrix @ matrix
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(graph_output, matrix @ matrix)

    boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 9, 9]], device=device, dtype=torch.float32)
    scores = torch.tensor([0.9, 0.8], device=device)
    assert torchvision.ops.nms(boxes, scores, 0.5).tolist() == [0]
    return {
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "torch_cuda": torch.version.cuda, "gpu_name": info.name,
        "compute_capability": [info.major, info.minor],
        "precision": "bf16；torchvision NMS 使用 fp32",
        "batch": 1, "steps": 1, "seed": seed, "sample_count": 1,
        "conv3d_forward_backward": "passed", "sdpa": "passed",
        "cuda_graph": "passed", "torchvision_cuda_nms": "passed",
        "torch_peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["inventory", "torch"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if platform.node() != "thor" or platform.machine() != "aarch64" or ROOT != pathlib.Path("/home/thor/rsi"):
        parser.error("实验只允许在 Thor 的 /home/thor/rsi 运行")

    started = dt.datetime.now(dt.timezone.utc).astimezone()
    run_dir = ROOT / "runs" / "p0" / f"{args.stage}_{started.strftime('%Y%m%dT%H%M%S_%f%z')}"
    (run_dir / "logs").mkdir(parents=True, exist_ok=False)
    power_mode = command(["sudo", "-n", "nvpmodel", "-q"])
    record = {
        "stage": args.stage, "iter": 0, "policy_ckpt_sha256": None,
        "git_sha": command(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
        "upstream_sha": {p.name: command(["git", "-C", str(p), "rev-parse", "HEAD"]).get("stdout")
                         for p in (ROOT / "third_party").iterdir() if (p / ".git").is_dir()},
        "hardware": {"host": platform.node(), "arch": platform.machine(), "power_mode": power_mode},
        "command": [sys.executable, *sys.argv], "started_at": started.isoformat(),
        "status": "running", "peak_mem_gb": None,
    }
    print(f"记录目录：{run_dir}", flush=True)
    monitor_log = (run_dir / "logs" / "tegrastats.log").open("w")
    monitor = None
    wall_start = time.monotonic()
    try:
        if shutil.which("tegrastats"):
            monitor = subprocess.Popen(["tegrastats", "--interval", "200"], stdout=monitor_log,
                                       stderr=subprocess.STDOUT)
        record["result"] = inventory() if args.stage == "inventory" else torch_smoke(args.seed)
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
        samples = [int(value) for value in re.findall(r"RAM\s+(\d+)/\d+MB", (run_dir / "logs" / "tegrastats.log").read_text())]
        record["peak_mem_gb"] = max(samples) * 1024 ** 2 / 1e9 if samples else None
        record["peak_mem_definition"] = "tegrastats 每200ms采样的全机已用 RAM，MB按MiB换算为十进制GB；包含系统和其他CPU进程"
        record["memory_sample_count"] = len(samples)
        record["wall_s"] = time.monotonic() - wall_start
        record["finished_at"] = dt.datetime.now(dt.timezone.utc).astimezone().isoformat()
        (run_dir / "probe.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({k: record[k] for k in ["stage", "status", "wall_s", "peak_mem_gb"]}, ensure_ascii=False), flush=True)
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
