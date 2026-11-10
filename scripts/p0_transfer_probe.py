#!/usr/bin/env python3
"""在生成专用环境验证ARM扩展与bf16算子，失败时保留实际异常。

@author pony
@date 2026-11-06
@version v1.1.0
@last_modified 2026-11-06
@changelog
    - v1.1.0 (2026-11-06): 实际执行所选DiT attention和SDPA，避免把可导入误当内核可运行。
    - v1.0.0 (2026-11-06): 验证官方cu130环境的关键二进制扩展与conv3d。
"""
from __future__ import annotations

import importlib
import importlib.metadata
import os
import pathlib
import sys

from p0_common import ROOT, ensure_gpu_idle, run_stage


def probe(run_dir: pathlib.Path) -> dict:
    ensure_gpu_idle()
    import torch

    versions = {name: importlib.metadata.version(name) for name in
                ["torch", "torchvision", "transformers", "flash-attn", "natten", "transformer-engine", "xformers", "decord"]}
    for module in ["torchvision", "flash_attn", "natten", "transformer_engine.pytorch", "xformers.ops", "decord"]:
        importlib.import_module(module)
    torch.manual_seed(42)
    layer = torch.nn.Conv3d(3, 16, (2, 14, 14), stride=(2, 14, 14)).cuda().bfloat16()
    sample = torch.randn(1, 3, 2, 28, 28, dtype=torch.bfloat16, device="cuda")
    with torch.inference_mode():
        result = layer(sample)
    torch.cuda.synchronize()
    if result.shape != (1, 16, 1, 2, 2) or not torch.isfinite(result).all():
        raise RuntimeError("生成环境bf16 conv3d形状或有限性验证失败")
    repo = ROOT / "third_party/cosmos-transfer2.5"
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    from cosmos_transfer2._src.predict2.networks.minimal_v4_dit import Attention

    attention = Attention(query_dim=128, n_heads=2, head_dim=64, backend="torch").cuda().bfloat16().eval()
    query = torch.randn(1, 16, 128, dtype=torch.bfloat16, device="cuda")
    with torch.inference_mode():
        attended = attention(query)
    torch.cuda.synchronize()
    if attended.shape != query.shape or not torch.isfinite(attended).all():
        raise RuntimeError("实际DiT attention算子验证失败")
    return {"versions": versions, "precision": "bf16", "batch": 1, "seed": 42, "steps": 1, "sample_count": 1,
            "conv3d_output_shape": list(result.shape), "cuda_runtime": torch.version.cuda,
            "dit_attention_backend": "torch", "dit_attention_output_shape": list(attended.shape),
            "device_capability": list(torch.cuda.get_device_capability()),
            "device": torch.cuda.get_device_name(), "inference_performed": "仅最小conv3d与DiT attention算子，非生成模型"}


if __name__ == "__main__":
    raise SystemExit(run_stage("transfer_env", probe))
