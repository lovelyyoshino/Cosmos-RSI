#!/usr/bin/env python3
"""只观察真实四步生成的单分支空间控制，不改变采样、掩码或训练资格。

@author pony
@date 2026-11-08
@version v1.0.1
@last_modified 2026-11-08
@changelog
    - v1.0.1 (2026-11-08): 兼容上游合并权重包装层以**kwargs转发实际控制参数，保留首次失败记录。
    - v1.0.0 (2026-11-08): 记录网络及全部注入层的实际权重，核对原生解码掩码与时序压缩。
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import pathlib
import sys
from unittest.mock import patch

from p0_common import ROOT, ensure_gpu_idle, run_stage
from p0_policy_runtime import sha256


def forwarded_argument(forward, inputs, kwargs, name):
    """上游包装后的签名可能只暴露**kwargs；只读取实际传参，不填充猜测默认值。"""
    if name in kwargs:
        return kwargs[name]
    signature = inspect.signature(forward)
    bound = signature.bind_partial(*inputs, **kwargs).arguments
    if name in bound:
        return bound[name]
    for key, parameter in signature.parameters.items():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD and name in bound.get(key, {}):
            return bound[key][name]
    raise ValueError(f"实际前向调用缺少参数：{name}；签名：{signature}")


def expected_weights(mask, shape):
    """独立CPU参照保留首帧，后续92帧压缩到23帧，不能整段一起插值。"""
    import numpy as np
    import torch
    import torch.nn.functional as functional

    binary = np.asarray(mask)
    if binary.dtype != np.bool_ or binary.ndim != 3 or binary.shape[0] != 93 \
            or binary.all() or not binary.any() or len(shape) != 6 \
            or tuple(shape[:3]) != (1, 1, 24) or shape[-1] != 1 or min(shape[3:5]) <= 1:
        raise ValueError("只诊断93帧非平凡bool掩码及单源/单分支24帧latent")
    tensor = torch.from_numpy(binary.copy()).float()[None, None]
    first = functional.interpolate(tensor[:, :, :1], size=(1, shape[3], shape[4]),
                                   mode="trilinear", align_corners=False)
    rest = functional.interpolate(tensor[:, :, 1:], size=(23, shape[3], shape[4]),
                                  mode="trilinear", align_corners=False)
    return torch.cat([first, rest], dim=2).permute(0, 2, 3, 4, 1).numpy()[None]


def weight_statistics(actual, expected):
    """拒绝全帧标量、反向掩码、错位及跨步变化；容差只覆盖bf16舍入。"""
    import numpy as np

    values, reference = np.asarray(actual), np.asarray(expected)
    if values.shape != reference.shape or values.ndim != 6 or values.shape[:3] != (1, 1, 24) \
            or values.shape[-1] != 1 or not np.isfinite(values).all() \
            or not np.isfinite(reference).all() or values.min() < 0 or values.max() > 1 \
            or not np.allclose(values, reference, rtol=8e-3, atol=2e-4):
        raise ValueError("实际空间权重偏离同帧源掩码的方向、位置或数值")
    if not (values == 0).any() or not (values == 1).any():
        raise ValueError("实际权重没有同时保留完整约束区与完整释放区")
    return {"shape": list(values.shape), "min": float(values.min()), "max": float(values.max()),
            "mean": float(values.mean()), "zero_fraction": float((values == 0).mean()),
            "one_fraction": float((values == 1).mean()),
            "max_abs_error_vs_cpu": float(np.abs(values - reference).max()),
            "matches_source_mask": True}


def validate_trace(trace, block_names, steps=4):
    if len(trace) != steps or not block_names or len(set(block_names)) != len(block_names):
        raise ValueError("实际四步或注入层清单不完整")
    for index, record in enumerate(trace):
        blocks = record.get("blocks", [])
        if record.get("step") != index or not record.get("network", {}).get("matches_source_mask") \
                or [b.get("name") for b in blocks] != block_names \
                or any(not b.get("matches_source_mask") or b.get("max_abs_error_vs_network") != 0 for b in blocks):
            raise ValueError("有去噪步骤或注入层未使用同一实际空间权重")


def diagnose(directory: pathlib.Path, args: argparse.Namespace):
    ensure_gpu_idle()
    from p0_table_background import checked_table_source
    from p0_transfer_smoke import generate

    source_path = args.spec.parent / "verified_input.json"
    checked_table_source(source_path)
    source_probe = args.spec.parent / "probe.json"
    declaration = {"purpose": "同一固定配方的前向观察复跑；不是新配方或独立质量验证",
                   "source_spec": str(args.spec), "source_spec_sha256": sha256(args.spec),
                   "source_probe_sha256": sha256(source_probe), "source_input_sha256": sha256(source_path),
                   "baseline_probe": str(args.baseline_probe), "baseline_probe_sha256": sha256(args.baseline_probe),
                   "expected_steps": 4, "expected_branches": 1, "eligible_for_training": False,
                   "timing_caveat": "前向hook同步及CPU复制带来额外开销，不用于性能比较"}
    baseline = json.loads(args.baseline_probe.read_text())
    previous = baseline.get("result", {})
    if baseline.get("stage") != "transfer_smoke" or baseline.get("status") != "passed" \
            or previous.get("source_spec") != str(args.spec) or previous.get("recipe") != "edge/distilled" \
            or previous.get("num_steps") != 4 or previous.get("seed") != 42:
        raise ValueError("观察复跑必须绑定已记录的同源同配方生成")
    (directory / "declaration.json").write_text(json.dumps(declaration, ensure_ascii=False, indent=2) + "\n")

    # 与正式入口相同的离线注册环境；不改上游文件，不注册无关模型或下载新权重。
    base = ROOT / "checkpoints/base"
    for name, value in {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                        "COSMOS_EXPERIMENTAL_CHECKPOINTS": "1", "RSI_TRANSFER_ONLY": "1",
                        "RSI_QWEN3_GUARD_PATH": str(base / "Qwen3Guard-Gen-0.6B"),
                        "RSI_SIGLIP_PATH": str(base / "siglip-so400m-patch14-384")}.items():
        os.environ[name] = value
    sys.path.insert(0, str(ROOT / "third_party/cosmos-transfer2.5"))
    from cosmos_transfer2.inference import Control2WorldInference
    from cosmos_transfer2._src.transfer2.inference.utils import read_and_resize_input
    import cv2
    import numpy as np

    decoded, fps, _, _ = read_and_resize_input(str(args.spec.parent / "scene_mask.mp4"),
                                              resolution="480", interpolation=cv2.INTER_LINEAR)
    if decoded.shape[1] != 93 or fps != 20:
        raise ValueError("官方读取的掩码改变窗口长度或帧率")
    requested = decoded[0].cpu().numpy() > 127.5
    np.save(directory / "decoded_requested_mask.npy", requested, allow_pickle=False)
    original_init = Control2WorldInference.__init__
    trace, hooks, names, actual_arrays, expected_arrays = [], [], [], [], []
    architecture = {}

    def observed_init(instance, *positional, **keywords):
        original_init(instance, *positional, **keywords)
        net = instance.inference_pipeline.model.net
        architecture.update({"class": type(net).__module__ + "." + type(net).__name__,
                             "num_control_branches": net.num_control_branches,
                             "vace_has_mask": net.vace_has_mask,
                             "use_cuda_graphs": net.use_cuda_graphs})
        if net.num_control_branches != 1 or net.vace_has_mask or net.use_cuda_graphs:
            raise ValueError("实际模型不是已声明的单分支空间权重路径，停止而不套用旧假设")
        controlled = [(name, block) for name, block in net.blocks.named_children()
                      if getattr(block, "block_id", None) is not None]
        names.extend(name for name, _ in controlled)
        if not names:
            raise ValueError("没有发现实际接收控制注入的层")

        def network_hook(module, inputs, kwargs):
            scale = forwarded_argument(module.forward, inputs, kwargs, "control_context_scale")
            if not hasattr(scale, "detach"):
                raise ValueError("网络收到标量而非空间控制")
            actual = scale.detach().float().cpu().numpy().copy()
            reference = expected_weights(requested, actual.shape)
            record = {"step": len(trace), "network": weight_statistics(actual, reference), "blocks": []}
            latent = forwarded_argument(module.forward, inputs, kwargs, "latent_control_input")
            record["control_latent"] = {"shape": list(latent.shape), "dtype": str(latent.dtype)}
            actual_arrays.append(actual)
            expected_arrays.append(reference)
            trace.append(record)

        hooks.append(net.register_forward_pre_hook(network_hook, with_kwargs=True))
        for name, block in controlled:
            def block_hook(module, inputs, kwargs, layer_name=name):
                if not trace:
                    raise ValueError("注入层缺少本次网络前向记录")
                scale = forwarded_argument(module.forward, inputs, kwargs, "control_context_scale")
                scale = scale.detach().float().cpu().numpy().copy()[None]
                record = weight_statistics(scale, expected_arrays[-1])
                record.update({"name": layer_name, "block_id": int(module.block_id),
                               "max_abs_error_vs_network": float(np.abs(scale - actual_arrays[-1]).max())})
                trace[-1]["blocks"].append(record)
            hooks.append(block.register_forward_pre_hook(block_hook, with_kwargs=True))

    try:
        with patch.object(Control2WorldInference, "__init__", observed_init):
            result = generate(directory, argparse.Namespace(spec=args.spec, model="edge/distilled",
                              num_steps=4, resolution="480", seed=42))
        validate_trace(trace, names)
    finally:
        for hook in hooks:
            hook.remove()
        for name, arrays in [("actual_network_weights.npy", actual_arrays), ("expected_cpu_weights.npy", expected_arrays)]:
            if arrays:
                np.save(directory / name, np.stack(arrays), allow_pickle=False)
        report = {"architecture": architecture, "controlled_blocks": names, "calls": trace,
                  "source_mask_sha256": sha256(args.spec.parent / "scene_mask.mp4"),
                  "eligible_for_training": False, "trace_does_not_prove_generation_quality": True}
        (directory / "single_control_trace.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    result.update({"single_control_trace": {"path": str(directory / "single_control_trace.json"),
                   "sha256": sha256(directory / "single_control_trace.json"), "validated": True},
                   "diagnostic_only": True, "eligible_for_training": False})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spec", type=pathlib.Path, required=True)
    parser.add_argument("--baseline-probe", type=pathlib.Path, required=True)
    args = parser.parse_args()
    args.spec, args.baseline_probe = args.spec.resolve(strict=True), args.baseline_probe.resolve(strict=True)
    return run_stage("single_control_trace", lambda directory: diagnose(directory, args),
                     {"model": "edge/distilled", "batch": 1, "steps": 4, "seed": 42,
                      "source_windows": 1, "frames": 93, "fps": 20, "diagnostic_only": True})


if __name__ == "__main__":
    raise SystemExit(main())
