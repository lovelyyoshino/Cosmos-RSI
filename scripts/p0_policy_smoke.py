#!/usr/bin/env python3
"""真实Reason2骨干与CosmosGR00T动作头的单样本计算验证。

@author pony
@date 2026-11-06
@version v1.1.0
@last_modified 2026-11-06
@changelog
    - v1.1.0 (2026-11-06): 记录真实参数dtype、预热后延迟分位数和稳态训练吞吐。
    - v1.0.0 (2026-11-06): 直接构建目标框架，验证动作推理和可选单卡训练。
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import pathlib
import sys
import time

from p0_common import ROOT, ensure_gpu_idle, run_stage


def smoke(run_dir: pathlib.Path, args: argparse.Namespace) -> dict:
    ensure_gpu_idle()
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from PIL import Image

    sys.path.insert(0, str(ROOT / "third_party/starVLA"))
    # 直接导入目标类，避免通用工厂扫描并导入所有无关框架和可选训练栈。
    from starVLA.model.framework.VLM4A.CosmosGR00T import Cosmos_GR00T

    fixture = json.loads(args.fixture.read_text())
    data = np.load(fixture["sample_npz"])
    if list(data["normalized_action"].shape) != [8, 7] or list(data["normalized_state"].shape) != [1, 8]:
        raise RuntimeError("样本维度与明确配置不一致")
    example = {"image": [Image.open(item["path"]).convert("RGB") for item in fixture["images"]],
               "lang": fixture["instruction"], "state": data["normalized_state"],
               "action": data["normalized_action"]}
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    config = OmegaConf.create({
        "framework": {"name": "CosmosGR00T", "qwenvl": {"base_vlm": str(args.model), "attn_implementation": "sdpa"},
                      "action_model": {"state_dim": 8, "action_dim": 7, "action_horizon": 8}},
        "datasets": {"vla_data": {"obs_image_size": [256, 256]}},
    })
    (run_dir / "config.yaml").write_text(OmegaConf.to_yaml(config))
    model = Cosmos_GR00T(config).to("cuda")
    model.eval()
    torch.cuda.synchronize()
    before = time.perf_counter()
    prediction = model.predict_action([example])["normalized_actions"]
    torch.cuda.synchronize()
    infer_ms = (time.perf_counter() - before) * 1000
    if prediction.shape != (1, 8, 7) or not np.isfinite(prediction).all():
        raise RuntimeError("预测动作维度或有限性检查未通过")
    np.save(run_dir / "normalized_actions.npy", prediction)
    infer_records = []
    for repeat in range(args.warmup + args.repeats):
        torch.cuda.synchronize()
        before = time.perf_counter()
        repeated_prediction = model.predict_action([example])["normalized_actions"]
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - before) * 1000
        if repeated_prediction.shape != (1, 8, 7) or not np.isfinite(repeated_prediction).all():
            raise RuntimeError("重复推理动作维度或有限性检查未通过")
        if repeat >= args.warmup:
            infer_records.append(wall_ms)
    infer_peak = torch.cuda.max_memory_allocated() / 1e9
    train_records = []
    if args.train_steps:
        if args.recipe == "head":
            model.qwen_vl_interface.requires_grad_(False)
        elif args.recipe == "freeze-vision":
            frozen = 0
            for name, parameter in model.qwen_vl_interface.model.named_parameters():
                if name.startswith(("model.visual.", "visual.")):
                    parameter.requires_grad_(False)
                    frozen += parameter.numel()
            if not frozen:
                raise RuntimeError("未找到视觉塔，禁止静默退化为全量训练")
        elif args.recipe == "lora":
            from peft import LoraConfig, get_peft_model
            model.qwen_vl_interface.model = get_peft_model(
                model.qwen_vl_interface.model,
                LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], bias="none"),
            )
        model.train()
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=1e-5, fused=True)
        torch.cuda.reset_peak_memory_stats()
        for step in range(args.train_steps):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            before = time.perf_counter()
            loss = model([example])["action_loss"]
            if not torch.isfinite(loss):
                raise RuntimeError("训练loss不是有限数")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize()
            train_records.append({"step": step, "loss": float(loss.detach()),
                                  "grad_norm": float(grad_norm), "wall_s": time.perf_counter() - before})
            print(json.dumps(train_records[-1]), flush=True)
    dtype_counts = {}
    for name, module in [("vlm", model.qwen_vl_interface), ("action_head", model.action_model)]:
        counts = Counter()
        for parameter in module.parameters():
            counts[str(parameter.dtype)] += parameter.numel()
        dtype_counts[name] = dict(counts)
    steady_train = train_records[args.train_warmup:]
    result = {
        "framework": "CosmosGR00T", "model_path": str(args.model), "fixture": str(args.fixture),
        "precision": "VLM使用bf16 autocast；动作头为上游默认，参数dtype另列", "attention": "sdpa",
        "parameter_dtypes": dtype_counts,
        "batch": 1, "sample_count": 1, "seed": args.seed, "steps": args.train_steps,
        "action_horizon": 8, "action_shape": list(prediction.shape), "state_dim": 8,
        "cold_infer_ms": infer_ms, "infer_timing_includes": "双视角预处理+VLM+动作头，不含模型加载；未预热",
        "warm_infer": {"warmup": args.warmup, "measured_over": len(infer_records), "latencies_ms": infer_records,
                       "p50_ms": float(np.percentile(infer_records, 50)) if infer_records else None,
                       "p95_ms": float(np.percentile(infer_records, 95)) if infer_records else None,
                       "includes": "双视角预处理+VLM+动作头+D2H，不含网络和模型加载；重复同一真实样本"},
        "recipe": args.recipe if args.train_steps else None, "training": train_records,
        "training_benchmark": {"warmup_steps": args.train_warmup, "measured_steps": len(steady_train),
                               "samples_per_s": len(steady_train) / sum(r["wall_s"] for r in steady_train) if steady_train else None,
                               "includes": "batch1前后向、梯度检查/裁剪、AdamW；同一样本驻留内存，无磁盘加载"},
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "torch_peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
        "infer_peak_allocated_gb": infer_peak,
        "eligible_for_deployment": False, "policy_status": "动作头随机初始化；本次不保存或晋升权重",
        "deepSpeed": False, "flash_attention": False,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fixture", type=pathlib.Path, required=True)
    parser.add_argument("--model", type=pathlib.Path, default=ROOT / "checkpoints/base/Cosmos-Reason2-2B")
    parser.add_argument("--train-steps", type=int, default=0)
    parser.add_argument("--recipe", choices=["head", "freeze-vision", "lora", "full"], default="head")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--train-warmup", type=int, default=1)
    args = parser.parse_args()
    if args.train_steps < 0:
        parser.error("train-steps不能为负数")
    if min(args.warmup, args.repeats, args.train_warmup) < 0:
        parser.error("预热和重复次数不能为负数")
    return run_stage("policy_smoke", lambda run_dir: smoke(run_dir, args),
                     {k: str(v) if isinstance(v, pathlib.Path) else v for k, v in vars(args).items()})


if __name__ == "__main__":
    raise SystemExit(main())
