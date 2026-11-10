#!/usr/bin/env python3
"""执行已声明的桌面材质参考双控制生成，仅供设计诊断，不借用旧门入训。

@author pony
@date 2026-11-08
@version v1.0.0
@last_modified 2026-11-08
@changelog
    - v1.0.0 (2026-11-08): 绑定材质参考和成功源，核验每次原生前向的edge/vis空间权重。
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import pathlib
import sys
import time

from p0_common import ROOT, ensure_gpu_idle, run_stage
from p0_policy_runtime import sha256
from p0_table_reference_input import CONFIG, CONFIG_PATH, checked_config, table_reference


def reference_control_weights(mask, shape):
    """edge仅保护非桌体，vis固定覆盖全帧；零分支不能被错误归一化成有效控制。"""
    import numpy as np
    from p0_single_control_trace import expected_weights

    if tuple(shape[:2]) != (4, 1) or len(shape) != 6:
        raise ValueError("必须记录四分支、单源的原生空间权重")
    protected = expected_weights(mask, (1, *shape[1:]))[0]
    result = np.stack([CONFIG["edge_weight"] * protected,
                       np.full_like(protected, CONFIG["vis_weight"]),
                       np.zeros_like(protected), np.zeros_like(protected)])
    return result / np.maximum(1, result.sum(axis=0, keepdims=True))


def check_control_weights(actual, expected):
    import numpy as np

    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or actual.ndim != 6 or actual.shape[:3] != (4, 1, 24) \
            or not np.isfinite(actual).all() or not np.isfinite(expected).all() \
            or actual.min() < 0 or actual.max() > 1 or not np.allclose(actual, expected, rtol=8e-3, atol=2e-4) \
            or np.count_nonzero(actual[2:]) or not (actual[0] == 0).any() or not (actual[0] > 0).any():
        raise ValueError("原生edge/vis实际空间权重与固定源掩码或配方不一致")
    return {"shape": list(actual.shape), "max_abs_error_vs_cpu": float(np.abs(actual - expected).max()),
            "edge_zero_fraction": float((actual[0] == 0).mean()),
            "edge_min": float(actual[0].min()), "edge_max": float(actual[0].max()),
            "vis_min": float(actual[1].min()), "vis_max": float(actual[1].max()),
            "inactive_branches_nonzero": int(np.count_nonzero(actual[2:]))}


def checked_input(path):
    from p0_background_preview_materials import checked_artifact
    from p0_table_background import checked_table_source
    from p0_lossless_background import read_exact_video
    import numpy as np

    checked_config()
    preparation = checked_artifact(path, "table_reference_input")
    record = json.loads(path.read_text())
    if record.get("method") != CONFIG["version"] or record.get("config_sha256") != sha256(CONFIG_PATH) \
            or record.get("diagnostic_only") is not True or record.get("eligible_for_training") is not False:
        raise ValueError("只接受固定新配方的设计诊断输入")
    for name in ["scripts/p0_table_reference_input.py", "configs/table_reference_preview.yaml"]:
        if preparation.get("source_files_sha256", {}).get(name) != sha256(ROOT / name):
            raise ValueError("材质参考准备实现或配置已经改变")
    source_path = pathlib.Path(record["source_input"]).resolve(strict=True)
    if not source_path.is_relative_to(ROOT / "runs/p0") or sha256(source_path) != record["source_input_sha256"]:
        raise ValueError("材质参考绑定的成功源发生变化")
    evidence, source, _, editable, _, _ = checked_table_source(source_path)
    if not evidence["source_episode_id"].endswith(CONFIG["design_episode_suffix"]) \
            or record.get("source_labels_sha256") != evidence["labels_sha256"]:
        raise ValueError("禁止使用未声明的源或错绑动作/状态标签")
    for name, checksum in record["files_sha256"].items():
        actual = (path.parent / name).resolve(strict=True)
        if actual.parent != path.parent or sha256(actual) != checksum:
            raise ValueError("实际参考控制未绑定准备产物")
    reference, _ = table_reference(source, editable)
    decoded, _ = read_exact_video(path.parent / "reference_rgb.mp4")
    if not np.array_equal(reference, decoded):
        raise ValueError("实际材质参考不符合固定桌体/RGB变换")
    spec_path = path.parent / "spec.json"
    if sha256(spec_path) != record["spec_sha256"]:
        raise ValueError("生成配方已改变")
    spec = json.loads(spec_path.read_text())
    original = json.loads((source_path.parent / "spec.json").read_text())
    if spec.get("video_path") != str(source_path.parent / "source.mp4") or spec.get("prompt") != original["prompt"] \
            or spec.get("edge") != {"control_path": str(source_path.parent / "edge.mp4"), "control_weight": .2,
                                    "mask_path": str(source_path.parent / "scene_mask.mp4")} \
            or spec.get("vis") != {"control_path": str(path.parent / "vis.mp4"), "control_weight": .8} \
            or any(spec.get(key) is not None for key in ["depth", "seg", "image_context_path", "guided_generation_mask"]) \
            or any(spec.get(key) != value for key, value in {"num_steps": 35, "seed": 42, "resolution": "480",
                "max_frames": 93, "num_video_frames_per_chunk": 93, "guidance": 3}.items()):
        raise ValueError("生成配方没有绑定同源RGB、实际控制或预声明采样条件")
    return record, spec


def generate(directory, args):
    ensure_gpu_idle()
    record, spec = checked_input(args.reference_input)
    from p0_transfer_smoke import MULTICONTROL_ORDER, checked_components, checked_control_checkpoint

    components = checked_components("edge")
    checkpoints = {key: checked_control_checkpoint(key) for key in MULTICONTROL_ORDER}
    base = ROOT / "checkpoints/base"
    for name, value in {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                        "COSMOS_EXPERIMENTAL_CHECKPOINTS": "1", "RSI_TRANSFER_ONLY": "1",
                        "RSI_QWEN3_GUARD_PATH": str(base / "Qwen3Guard-Gen-0.6B"),
                        "RSI_SIGLIP_PATH": str(base / "siglip-so400m-patch14-384"),
                        "RSI_SIGLIP2_PATH": str(base / "siglip2-so400m-patch16-naflex")}.items():
        os.environ[name] = value
    repo = ROOT / "third_party/cosmos-transfer2.5"
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    import cv2
    import numpy as np
    import torch
    from cosmos_oss.init import init_environment, cleanup_environment
    from cosmos_transfer2.config import InferenceArguments, SetupArguments
    from cosmos_transfer2.inference import Control2WorldInference
    from cosmos_transfer2._src.imaginaire.auxiliary.guardrail.common.core import GUARDRAIL1_CHECKPOINT
    from cosmos_transfer2._src.transfer2.inference.utils import read_and_resize_input
    from p0_single_control_trace import forwarded_argument

    GUARDRAIL1_CHECKPOINT._path = str(base / "Cosmos-Guardrail1")
    sample = InferenceArguments.model_validate(spec)
    setup = SetupArguments(output_dir=directory / "output", model="edge", checkpoint_path=checkpoints["edge"]["path"],
                           disable_guardrails=False, offload_guardrail_models=False, keep_going=False)
    if not setup.enable_guardrails:
        raise ValueError("设计诊断仍要求全部安全检查")
    overrides = [f"++model.config.tokenizer.vae_pth={base / 'Cosmos-Predict2.5-2B/tokenizer.pth'}",
                 f"model.config.text_encoder_config.ckpt_path={base / 'Cosmos-Reason1-7B'}",
                 f"++model.config.text_encoder_config.model_config.tokenizer.cache_dir={base / 'Cosmos-Reason1-7B'}",
                 "++model.config.text_encoder_config.model_config.model_config.attn_implementation=sdpa",
                 "model.config.net.atten_backend=torch"]
    (directory / "setup.json").write_text(setup.model_dump_json(indent=2) + "\n")
    (directory / "sample.json").write_text(sample.model_dump_json(indent=2) + "\n")
    init_environment()
    hooks, trace, weights_saved = [], [], []
    try:
        started = time.perf_counter()
        inference = Control2WorldInference(setup, batch_hint_keys=["edge", "vis"], experiment_opts=overrides,
                                          checkpoint_paths=[checkpoints[key]["path"] for key in MULTICONTROL_ORDER])
        net = inference.inference_pipeline.model.net
        if net.num_control_branches != 4:
            raise ValueError("实际模型没有使用完整原生四分支容器")
        backends = Counter()
        for name, module in net.named_modules():
            if hasattr(module, "backend") and hasattr(module, "attn_op"):
                if module.backend != "torch":
                    raise ValueError(f"实际注意力后端偏离配置：{name}")
                backends[module.backend] += 1
        if not backends:
            raise ValueError("缺少实际注意力后端证据")
        decoded, fps, _, _ = read_and_resize_input(spec["edge"]["mask_path"], resolution="480", interpolation=cv2.INTER_LINEAR)
        if decoded.shape[1] != 93 or fps != 20:
            raise ValueError("实际掩码读取改变窗口时序")
        requested = decoded[0].cpu().numpy() > 127.5

        def observe(module, inputs, kwargs):
            scale = forwarded_argument(module.forward, inputs, kwargs, "control_context_scale")
            actual = scale.detach().float().cpu().numpy().copy()
            expected = reference_control_weights(requested, actual.shape)
            stats = check_control_weights(actual, expected)
            if not weights_saved:
                latent = forwarded_argument(module.forward, inputs, kwargs, "latent_control_input")
                latent_counts = [int(torch.count_nonzero(chunk).item()) for chunk in latent.chunk(4, dim=1)]
                if any(count <= 0 for count in latent_counts[:2]) or any(latent_counts[2:]):
                    raise ValueError("实际控制latent未激活edge/vis或错误激活depth/seg")
                np.save(directory / "actual_control_weights.npy", actual, allow_pickle=False)
                np.save(directory / "expected_control_weights.npy", expected, allow_pickle=False)
                weights_saved.append(latent_counts)
            trace.append({"forward_call": len(trace), **stats})

        hooks.append(net.register_forward_pre_hook(observe, with_kwargs=True))
        torch.cuda.synchronize()
        load_s = time.perf_counter() - started
        started = time.perf_counter()
        outputs = inference.generate([sample], output_dir=setup.output_dir)
        torch.cuda.synchronize()
        generate_s = time.perf_counter() - started
        if len(trace) < 35 or len(outputs) != 1 or not pathlib.Path(outputs[0]).is_file():
            raise ValueError("未取得完整35步生成或唯一通过安全检查的输出")
        from p0_lossless_background import read_exact_video
        _, metadata = read_exact_video(pathlib.Path(outputs[0]), size=None)
        return {"method": CONFIG["version"], "model": "Cosmos-Transfer2.5-2B", "recipe": "base_edge_vis_reference",
                "num_steps": 35, "seed": 42, "batch": 1, "source_windows": 1, "frames": 93, "fps": 20,
                "reference_input": str(args.reference_input), "reference_input_sha256": sha256(args.reference_input),
                "source_input": record["source_input"], "source_input_sha256": record["source_input_sha256"],
                "source_labels_sha256": record["source_labels_sha256"], "target_condition": CONFIG["target_condition"],
                "outputs": outputs, "video_info": metadata, "guardrails_enabled": True, "components": components,
                "control_checkpoints": checkpoints, "attention_backends": dict(backends),
                "dit_parameter_dtypes": dict(Counter(str(p.dtype) for p in net.parameters())),
                "actual_forward_calls": len(trace), "latent_nonzero_per_branch": weights_saved[0],
                "load_wall_s": load_s, "generate_wall_s": generate_s,
                "torch_peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
                "diagnostic_only": True, "eligible_for_training": False,
                "reason": "已见设计源的新材质配方试验；尚无固定方法独立验证及完整视频质量验收"}
    finally:
        for hook in hooks:
            hook.remove()
        (directory / "control_trace.json").write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n")
        cleanup_environment()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reference-input", type=pathlib.Path, required=True)
    arguments = parser.parse_args()
    arguments.reference_input = arguments.reference_input.resolve(strict=True)
    raise SystemExit(run_stage("table_reference_generation", lambda directory: generate(directory, arguments),
                     {"model": "base_edge_vis", "precision": "DiT bf16，实际dtype另录", "batch": 1,
                      "steps": 35, "seed": 42, "frames": 93, "fps": 20, "source_windows": 1, "diagnostic_only": True}))
