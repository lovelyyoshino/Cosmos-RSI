#!/usr/bin/env python3
"""Thor单卡CosmosGR00T初训：真实示范、独立验证、可恢复增量检查点。

@author pony
@date 2026-11-06
@version v1.2.0
@last_modified 2026-11-06
@changelog
    - v1.2.0 (2026-11-06): 正式初训先核验冻结集；支持在检查点分段停下并按原配方精确恢复。
    - v1.1.0 (2026-11-06): 记录确定性条件并关闭foreach梯度归约，验证断点续训一致性。
    - v1.0.0 (2026-11-06): 无DeepSpeed，记录逐步吞吐，保存未晋升的P0权重及重载对照。
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
import math
import pathlib
import time

from p0_common import ROOT, ensure_gpu_idle, run_stage
from p0_policy_runtime import DemoCache, build_model, restore_checkpoint, sha256, verify_eval_isolation


def train(run_dir: pathlib.Path, args: argparse.Namespace) -> dict:
    ensure_gpu_idle()
    import numpy as np
    import torch
    from omegaconf import OmegaConf

    cache = DemoCache(args.cache)
    isolation = {"passed": False, "reason": "仅P0计算验证，未提供冻结评测集，不可作为正式基线"}
    if args.formal_baseline and not args.frozen_eval:
        raise RuntimeError("正式初训必须先提供通过初态去重的冻结评测集")
    if args.frozen_eval:
        isolation = verify_eval_isolation(args.cache, args.frozen_eval)
    if args.batch < 1 or args.steps < 1 or args.save_every < 1 or args.eval_every < 1 or args.eval_batches < 1:
        raise RuntimeError("batch、步数和保存/验证间隔必须为正数")
    recipe = {"name": args.recipe, "steps": args.steps, "batch": args.batch, "seed": args.seed,
              "lr_head": args.lr_head, "lr_vlm": args.lr_vlm, "warmup_steps": args.warmup,
              "scheduler": "linear_warmup_cosine_min0.1", "clip_grad_norm": 1.0,
              "lora": {"r": 8, "alpha": 16, "targets": ["q_proj", "v_proj"]} if args.recipe == "lora" else None}
    source_manifest = None
    for path in sorted(ROOT.glob("runs/p0/model_download_*/model_manifest.json")):
        record = json.loads(path.read_text())
        if pathlib.Path(record["path"]).resolve() == args.model.resolve():
            source_manifest = path
    if source_manifest is None:
        raise RuntimeError("缺少基座模型下载校验清单")
    base_record = json.loads(source_manifest.read_text())
    for item in base_record["files"]:
        file = args.model / item["file"]
        if not file.is_file() or file.stat().st_size != item["bytes"] or sha256(file) != item["sha256"]:
            raise RuntimeError("基座模型内容已偏离下载清单")
    if args.resume:
        resume_metadata = json.loads((args.resume / "checkpoint.json").read_text())
        if resume_metadata["recipe"] != recipe or resume_metadata["cache_manifest_sha256"] != sha256(cache.manifest_path):
            raise RuntimeError("续训配方或数据缓存与检查点不一致")
        if resume_metadata["base_manifest_sha256"] != sha256(source_manifest):
            raise RuntimeError("续训基座版本不符")
    model = build_model(args.model, args.recipe, args.seed)
    (run_dir / "config.yaml").write_text(OmegaConf.to_yaml(model.config))
    groups = []
    for prefix, lr in [("qwen_vl_interface.", args.lr_vlm), ("action_model.", args.lr_head)]:
        parameters = [parameter for name, parameter in model.named_parameters() if parameter.requires_grad and name.startswith(prefix)]
        if parameters:
            groups.append({"params": parameters, "lr": lr})
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if sum(len(group["params"]) for group in groups) != len(parameters):
        raise RuntimeError("优化器遗漏可训练参数")
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01, fused=True)

    def schedule(step):
        if args.warmup and step < args.warmup:
            return (step + 1) / args.warmup
        progress = min(1.0, max(0.0, (step - args.warmup) / max(1, args.steps - args.warmup)))
        return 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    rng = np.random.default_rng(args.seed)
    start_step = 0
    if args.resume:
        _, saved = restore_checkpoint(args.resume, model)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        rng.bit_generator.state = saved["numpy_rng"]
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
        start_step = saved["step"]
        del saved
    stop_step = args.steps if args.stop_after is None else min(args.steps, args.stop_after)
    if stop_step <= start_step:
        raise RuntimeError("分段终点必须大于已恢复的步数")
    model.train()
    torch.cuda.reset_peak_memory_stats()
    trainable_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    base_hash = sha256(source_manifest)
    cache_hash = sha256(cache.manifest_path)
    metadata = {"framework": "starVLA CosmosGR00T", "recipe": recipe, "base_model": str(args.model),
                "base_manifest": str(source_manifest), "base_manifest_sha256": base_hash,
                "cache_manifest": str(cache.manifest_path), "cache_manifest_sha256": cache_hash,
                "normalizer": cache.manifest["normalizer"], "model_config": OmegaConf.to_container(model.config, resolve=True),
                "precision": "VLM bf16、LoRA与动作头fp32；无GradScaler", "attention": "sdpa", "deepSpeed": False,
                "determinism": {"torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                                "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                                "cublas_workspace": ":4096:8", "clip_grad_norm_foreach": False},
                "eligible_for_promotion": False,
                "eval_overlap_check": isolation, "formal_baseline_training": args.formal_baseline,
                "trainable_parameters": sum(parameter.numel() for parameter in parameters),
                "parameter_dtypes": dict(Counter({str(dtype): sum(p.numel() for p in model.parameters() if p.dtype == dtype)
                                                   for dtype in {p.dtype for p in model.parameters()}})),
                "sample_count": {split: len(value[0]) for split, value in cache.splits.items()},
                "parent_checkpoint": str(args.resume) if args.resume else None}
    validations = []
    checkpoints = []
    timed_steps = []
    before_train = time.monotonic()

    def prediction_probe(target):
        target.eval()
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(args.seed + 2)
            batch, locations = cache.batch("demo_validation", 1, np.random.default_rng(args.seed + 2))
            prediction = target.predict_action(batch)["normalized_actions"]
        if prediction.shape != (1, cache.horizon, 7) or not np.isfinite(prediction).all():
            raise RuntimeError("保存探针的预测形状或有限性不合格")
        target.train()
        return prediction, locations

    def save(step):
        directory = run_dir / "checkpoints" / f"step_{step:06d}"
        directory.mkdir(parents=True, exist_ok=False)
        prediction, locations = prediction_probe(model)
        saved = {"step": step,
                 "trainable_state": {name: value.detach().cpu() for name, value in model.state_dict().items() if name in trainable_names},
                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "numpy_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}
        torch.save(saved, directory / "state.pt")
        record = {**metadata, "step": step, "state_sha256": sha256(directory / "state.pt"),
                  "prediction_probe": {"seed": args.seed + 2, "locations": locations, "normalized_actions": prediction.tolist()},
                  "wall_s_since_training_start": time.monotonic() - before_train}
        (directory / "checkpoint.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        checkpoints.append(str(directory))
        print(json.dumps({"saved_checkpoint": str(directory), "step": step}), flush=True)

    with (run_dir / "logs/train_metrics.jsonl").open("w") as metrics:
        for step in range(start_step, stop_step):
            started = time.perf_counter()
            batch, locations = cache.batch("demo_train", args.batch, rng)
            optimizer.zero_grad(set_to_none=True)
            loss = model(batch)["action_loss"]
            if not torch.isfinite(loss):
                raise RuntimeError("训练loss非有限值")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True, foreach=False)
            optimizer.step()
            scheduler.step()
            torch.cuda.synchronize()
            duration = time.perf_counter() - started
            timed_steps.append(duration)
            item = {"step": step + 1, "loss": float(loss.detach()), "grad_norm": float(grad_norm), "wall_s": duration,
                    "samples_per_s": args.batch / duration, "lr": scheduler.get_last_lr(), "locations": locations}
            metrics.write(json.dumps(item) + "\n")
            metrics.flush()
            if step == start_step or (step + 1) % 25 == 0:
                print(json.dumps({k: v for k, v in item.items() if k != "locations"}), flush=True)
            if (step + 1) % args.eval_every == 0 or step + 1 == stop_step:
                model.eval()
                validation_losses = []
                val_rng = np.random.default_rng(args.seed + 1)
                with torch.random.fork_rng(devices=[torch.cuda.current_device()]), torch.inference_mode():
                    torch.manual_seed(args.seed + 1)
                    for _ in range(args.eval_batches):
                        validation_batch, _ = cache.batch("demo_validation", args.batch, val_rng)
                        value = model(validation_batch)["action_loss"]
                        validation_losses.append(float(value))
                if not np.isfinite(validation_losses).all():
                    raise RuntimeError("验证loss非有限值")
                validations.append({"step": step + 1, "mean_loss": float(np.mean(validation_losses)),
                                    "windows": args.eval_batches * args.batch, "seed": args.seed + 1,
                                    "meaning": "独立示范的固定采样噪声flow matching loss；不是仿真成功率"})
                print(json.dumps({"validation": validations[-1]}, ensure_ascii=False), flush=True)
                model.train()
            if (step + 1) % args.save_every == 0 or step + 1 == stop_step:
                save(step + 1)
    if not checkpoints:
        raise RuntimeError("没有执行新的训练步骤或保存检查点")
    peak_allocated = torch.cuda.max_memory_allocated() / 1e9
    reload_check = {"performed": False}
    if args.verify_reload:
        config = model.config
        del model, optimizer, scheduler, parameters, groups, loss, grad_norm
        gc.collect()
        torch.cuda.empty_cache()
        restored = build_model(args.model, args.recipe, args.seed, config=config)
        checkpoint_meta, saved = restore_checkpoint(pathlib.Path(checkpoints[-1]), restored)
        del saved
        prediction, locations = prediction_probe(restored)
        reference = np.asarray(checkpoint_meta["prediction_probe"]["normalized_actions"], dtype=np.float32)
        if locations != checkpoint_meta["prediction_probe"]["locations"] or not np.allclose(prediction, reference, atol=1e-5, rtol=1e-4):
            raise RuntimeError("检查点重载后固定种子的动作预测不一致")
        reload_check = {"performed": True, "passed": True, "max_abs_diff": float(np.max(np.abs(prediction - reference))),
                        "atol": 1e-5, "rtol": 1e-4}
    result = {**metadata, "checkpoints": checkpoints, "last_checkpoint": checkpoints[-1], "validation": validations,
              "steps_executed": len(timed_steps), "samples_per_s": args.batch * len(timed_steps) / sum(timed_steps),
              "last_step": stop_step, "training_complete": stop_step == args.steps,
              "throughput_includes": "预解码缓存取样、PIL转换、前后向、梯度检查与AdamW；不含验证与保存",
              "torch_peak_allocated_gb": peak_allocated, "reload_check": reload_check,
              "training_wall_s": time.monotonic() - before_train}
    (run_dir / "train_run.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", type=pathlib.Path, required=True)
    parser.add_argument("--model", type=pathlib.Path, default=ROOT / "checkpoints/base/Cosmos-Reason2-2B")
    parser.add_argument("--recipe", choices=["head", "lora", "freeze-vision", "full"], default="lora")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr-head", type=float, default=1e-4)
    parser.add_argument("--lr-vlm", type=float, default=2e-5)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--resume", type=pathlib.Path)
    parser.add_argument("--verify-reload", action="store_true")
    parser.add_argument("--frozen-eval", type=pathlib.Path)
    parser.add_argument("--formal-baseline", action="store_true")
    parser.add_argument("--stop-after", type=int, help="保留原目标步数和学习率计划，在指定步保存后退出")
    args = parser.parse_args()
    if args.warmup < 0 or args.lr_head <= 0 or args.lr_vlm <= 0:
        parser.error("warmup不能为负，学习率必须为正")
    if args.stop_after is not None and args.stop_after < 1:
        parser.error("stop-after必须为正数")
    return run_stage("policy_train", lambda directory: train(directory, args),
                     {key: str(value) if isinstance(value, pathlib.Path) else value for key, value in vars(args).items()})


if __name__ == "__main__":
    raise SystemExit(main())
