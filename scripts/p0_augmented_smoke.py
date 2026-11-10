#!/usr/bin/env python3
"""从同一正式策略做候选/D0对照/纯色消融的短程数据接入冒烟，不作收益评测。

@author pony
@date 2026-11-08
@version v1.0.0
@last_modified 2026-11-08
@changelog
    - v1.0.0 (2026-11-08): 实际消费已准入部分episode，配对抽样、热启动及保存重载。
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
import pathlib
import time

from p0_common import ROOT, ensure_gpu_idle, run_stage
from p0_paired_sampling import build_paired_plan
from p0_policy_runtime import DemoCache, build_model, normalize, restore_checkpoint, sha256, verify_eval_isolation
from p0_uv_window_gate import checked_gate


def example_at(cache, row, augmented, variant):
    """所有动作块取同一源回合；仅当前agentview可以替换，腕部与监督保持。"""
    import numpy as np
    from PIL import Image

    matches = [r for r in cache.manifest["episodes"] if r["episode_id"] == row["source_episode_id"]]
    if len(matches) != 1 or matches[0]["split"] != "demo_train" or row["is_eval"] is not False:
        raise ValueError("抽样回合不属于唯一的D0训练分区")
    episode = matches[0]
    first, horizon = row["source_frame"], row["horizon"]
    window = row["window"]
    if not window["start_frame"] <= first or first + horizon > min(window["end_frame"], episode["length"]):
        raise ValueError("动作块越过完整回合或生成窗口边界")
    offset = episode["offset"] + first
    agent = np.array(cache.images[offset, 0])
    if row["kind"] == "generated":
        item = augmented[row["sample_id"]]
        if item["source_episode_id"] != episode["episode_id"] or item["window"] != window:
            raise ValueError("生成图像和动作计划不是同一源窗口")
        relative = first - window["start_frame"]
        if relative != row["window_frame"]:
            raise ValueError("生成窗口相对帧与绝对源帧不一致")
        agent = np.array(item[variant][relative])
    return {"image": [Image.fromarray(agent), Image.fromarray(np.array(cache.images[offset, 1]))],
            "lang": episode["instruction"],
            "state": normalize(cache.states[offset:offset + 1], cache.manifest["normalizer"]["state"]),
            "action": normalize(cache.actions[offset:offset + horizon], cache.manifest["normalizer"]["action"])}


def train(directory, args):
    ensure_gpu_idle()
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from p0_lossless_background import read_exact_video, validate_labels

    parent = json.loads((args.parent / "checkpoint.json").read_text())
    if parent.get("formal_baseline_training") is not True or parent["step"] != 5000 \
            or parent["recipe"]["name"] != "lora":
        raise ValueError("本冒烟只能从固定5000步正式LoRA基线热启动")
    manifest_path = pathlib.Path(parent["base_manifest"])
    if sha256(manifest_path) != parent["base_manifest_sha256"]:
        raise ValueError("正式策略的基座清单已改变")
    for item in json.loads(manifest_path.read_text())["files"]:
        if sha256(pathlib.Path(parent["base_model"]) / item["file"]) != item["sha256"]:
            raise ValueError("基座权重或处理器偏离正式初训版本")
    cache_path = pathlib.Path(parent["cache_manifest"]).parent
    isolation = verify_eval_isolation(cache_path, ROOT / "eval/frozen_eval_set.json")
    cache = DemoCache(cache_path)
    augmented, extra, bindings = {}, [], []
    for gate_path in args.gate:
        gate, rendered, evidence, *_ = checked_gate(gate_path)
        if gate["source_proof"]["cache_manifest_sha256"] != sha256(cache.manifest_path):
            raise ValueError("准入窗口与正式训练D0不一致")
        source = next(r for r in cache.manifest["episodes"] if r["episode_id"] == evidence["source_episode_id"])
        if source["split"] != "demo_train":
            raise ValueError("新增数据不能来自示范验证或策略评测")
        window = evidence["window"]
        sample_id = "uv-" + evidence["source_episode_id"]
        if sample_id in augmented:
            raise ValueError("禁止将同一窗口重复描述为不同新增数据")
        render_dir = pathlib.Path(gate["render_report"]).parent
        cosmos = read_exact_video(render_dir / "cosmos.mp4")[0]
        analytic = read_exact_video(render_dir / "analytic.mp4")[0]
        start, stop = source["offset"] + window["start_frame"], source["offset"] + window["end_frame"]
        with np.load(render_dir / "source_labels.npz", allow_pickle=False) as labels:
            validate_labels(dict(labels), {"action": cache.actions[start:stop], "state": cache.states[start:stop],
                "frame_index": np.arange(window["start_frame"], window["end_frame"]),
                "timestamp": np.arange(window["start_frame"], window["end_frame"]) / 20})
        # 纯色只替换同一批已准入源的桌体，作为解析配方消融；不得称为Cosmos贡献。
        augmented[sample_id] = {"source_episode_id": source["episode_id"], "window": window,
                                "cosmos": cosmos, "analytic": analytic}
        extra.append({"kind": "generated", "sample_id": sample_id, "source_episode_id": source["episode_id"],
                      "source_length": source["length"], "source_split": "demo_train", "is_eval": False,
                      "window": {k: window[k] for k in ["start_frame", "end_frame"]}, "eligible_for_training": True})
        augmented[sample_id]["window"] = extra[-1]["window"]
        bindings.append({"path": str(gate_path), "sha256": sha256(gate_path), "source_episode_id": source["episode_id"]})
    demos = [{"kind": "demo", "sample_id": r["episode_id"], "source_episode_id": r["episode_id"],
              "source_length": r["length"], "source_split": "demo_train", "is_eval": False,
              "window": {"start_frame": 0, "end_frame": r["length"]}}
             for r in cache.manifest["episodes"] if r["split"] == "demo_train"]
    plan = build_paired_plan(demos, extra, steps=5, batch=2, horizon=cache.horizon, seed=args.seed,
                             quotas={"generated": 5, "self_distill": 0})
    plan["quality_reports_verified"] = True
    # 原模块摘要只覆盖它实际生成的计划；准入证明另外绑定，避免改摘要字段却不重算。
    from p0_paired_sampling import paired_plan_sha256
    plan["sha256"] = paired_plan_sha256(plan)
    (directory / "paired_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    selected = plan["control" if args.role == "control" else "candidate"]
    variant = "analytic" if args.role == "analytic_ablation" else "cosmos"
    model = build_model(pathlib.Path(parent["base_model"]), "lora", args.seed,
                        OmegaConf.create(parent["model_config"]))
    _, saved = restore_checkpoint(args.parent, model)
    del saved
    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    groups = [{"params": [p for name, p in trainable.items() if name.startswith(prefix)], "lr": lr}
              for prefix, lr in [("qwen_vl_interface.", 1e-5), ("action_model.", 1e-4)]]
    if any(not group["params"] for group in groups) or sum(len(g["params"]) for g in groups) != len(trainable):
        raise ValueError("LoRA或动作头参数没有全部加入优化器")
    optimizer = torch.optim.AdamW(groups, fused=True, weight_decay=.01)
    torch.manual_seed(args.seed)
    model.train(); torch.cuda.reset_peak_memory_stats()
    rows = []
    before = time.monotonic()
    for step in range(5):
        slots = selected[step * 2:step * 2 + 2]
        batch = [example_at(cache, row, augmented, variant) for row in slots]
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = model(batch)["action_loss"]
        if not torch.isfinite(loss):
            raise ValueError("混合数据短程训练loss不是有限值")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(trainable.values(), 1., foreach=False, error_if_nonfinite=True)
        optimizer.step(); torch.cuda.synchronize()
        row = {"step": step + 1, "loss": float(loss.detach()), "grad_norm": float(norm),
               "wall_s": time.perf_counter() - started, "slots": slots}
        rows.append(row); print(json.dumps({k: v for k, v in row.items() if k != "slots"}), flush=True)
    training_wall = time.monotonic() - before
    model.eval()
    probe_batch, locations = cache.batch("demo_validation", 1, np.random.default_rng(args.seed + 2))

    def predict(target):
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(args.seed + 2)
            return np.asarray(target.predict_action(probe_batch)["normalized_actions"]).copy()

    expected = predict(model)
    if expected.shape != (1, cache.horizon, 7) or not np.isfinite(expected).all():
        raise ValueError("微调后推理不是有限动作块")
    checkpoint = directory / "checkpoint"; checkpoint.mkdir()
    weights = {"step": parent["step"] + 5,
               "trainable_state": {name: p.detach().cpu() for name, p in trainable.items()}}
    torch.save(weights, checkpoint / "state.pt")
    metadata = {**parent, "step": parent["step"] + 5, "formal_baseline_training": False,
        "eligible_for_promotion": False, "parent_checkpoint": str(args.parent),
        "parent_ckpt_sha256": parent["state_sha256"], "state_sha256": sha256(checkpoint / "state.pt"),
        "recipe": {**parent["recipe"], "steps": 5, "batch": 2, "seed": args.seed,
                   "lr_head": 1e-4, "lr_vlm": 1e-5, "scheduler": "constant_for_p0_smoke"},
        "role": args.role, "image_variant": variant, "smoke_only": True, "paired_plan_sha256": plan["sha256"],
        "admitted_windows": bindings, "eval_overlap_check": isolation,
        "prediction_probe": {"seed": args.seed + 2, "locations": locations, "normalized_actions": expected.tolist()}}
    (checkpoint / "checkpoint.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    del optimizer, model, trainable, groups, weights, loss, norm
    gc.collect(); torch.cuda.empty_cache()
    reloaded = build_model(pathlib.Path(parent["base_model"]), "lora", args.seed,
                           OmegaConf.create(parent["model_config"]))
    _, state = restore_checkpoint(checkpoint, reloaded)
    del state
    reloaded.eval(); actual = predict(reloaded)
    reload_error = float(np.max(np.abs(expected - actual)))
    if reload_error > 1e-5 or not np.isfinite(actual).all():
        raise ValueError("增量权重保存/重载预测不一致")
    result = {"role": args.role, "image_variant": variant,
        "parent_ckpt_sha256": parent["state_sha256"], "parent_checkpoint": str(args.parent),
        "output_ckpt": str(checkpoint), "output_ckpt_sha256": metadata["state_sha256"],
        "paired_plan_sha256": plan["sha256"], "admitted_windows": bindings,
        "data_mix_slots": dict(Counter(r["kind"] for r in selected)),
        "source_windows": len(extra), "training_steps": 5, "batch": 2, "seed": args.seed,
        "training_wall_s": training_wall, "reload_max_abs_diff": reload_error,
        "precision": parent["precision"], "attention": "sdpa", "deepSpeed": False,
        "eval_overlap_check": isolation, "steps": rows, "eligible_for_promotion": False,
        "scope": "P0真实数据接入与短程微调冒烟；未进行候选/对照冻结评测，不支持效果或极速结论"}
    (directory / "training_smoke.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return {key: result[key] for key in ["role", "output_ckpt", "paired_plan_sha256", "data_mix_slots",
            "training_wall_s", "reload_max_abs_diff", "eligible_for_promotion"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--parent", type=pathlib.Path, required=True)
    parser.add_argument("--gate", type=pathlib.Path, action="append", required=True)
    parser.add_argument("--role", choices=["candidate", "control", "analytic_ablation"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()
    arguments.parent = arguments.parent.resolve(strict=True)
    arguments.gate = [path.resolve(strict=True) for path in arguments.gate]
    if not arguments.parent.is_relative_to(ROOT / "runs/p0") \
            or any(not path.is_relative_to(ROOT / "runs/p0") for path in arguments.gate) \
            or not 0 <= arguments.seed < 2 ** 32:
        parser.error("来源须在本项目P0目录，seed须为uint32")
    raise SystemExit(run_stage("augmented_training_smoke", lambda directory: train(directory, arguments),
        {"role": arguments.role, "framework": "starVLA CosmosGR00T", "steps": 5, "batch": 2,
         "seed": arguments.seed, "precision": "VLM bf16，LoRA与动作头fp32", "attention": "sdpa",
         "deepSpeed": False, "source_windows": len(arguments.gate), "promotion": False}))
