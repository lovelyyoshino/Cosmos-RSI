#!/usr/bin/env python3
"""消费真实LeRobot交接，以固定预算测候选、复制对照或纯色消融；不作策略晋升。

@author pony
@date 2026-11-08
@version v1.0.0
@last_modified 2026-11-08
@changelog
    - v1.0.0 (2026-11-08): 原生数据回读、固定配额、真实训练计时和保存重载。
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
import pathlib
import re
import time

from p0_augmented_smoke import example_at
from p0_common import ROOT, command, ensure_gpu_idle, run_stage
from p0_lerobot_handoff import StrictNativeWindows, dump
from p0_paired_sampling import build_paired_plan, paired_plan_sha256
from p0_policy_runtime import DemoCache, build_model, restore_checkpoint, sha256, verify_eval_isolation


def checked_handoff(path, parent):
    """重新核对全部实际交接文件和原始质量证据，再开放训练读取。"""
    from p0_background_preview_materials import checked_artifact
    from p0_uv_window_gate import checked_gate

    path = path.resolve(strict=True)
    if not path.is_relative_to(ROOT / "runs/p0"):
        raise ValueError("数据交接报告位于项目之外")
    checked_artifact(path, "lerobot_handoff")
    data = json.loads(path.read_text())
    if data["schema"] != "thor-admitted-lerobot-handoff-v1" or data["parent_ckpt_sha256"] != parent["state_sha256"] \
            or data["cache_manifest_sha256"] != parent["cache_manifest_sha256"]:
        raise ValueError("交接不是当前正式基线及D0")
    for relative, checksum in data["files_sha256"].items():
        target = (path.parent / relative).resolve(strict=True)
        if not target.is_relative_to(path.parent / "lerobot") or sha256(target) != checksum:
            raise ValueError("真实LeRobot数据或元数据已改变")
    if set(data["datasets"]) != {"cosmos", "analytic"} or data["source_windows"] != len(data["admitted_windows"]):
        raise ValueError("交接数据不完整")
    for variant, dataset in data["datasets"].items():
        if pathlib.Path(dataset).resolve(strict=True) != path.parent / "lerobot" / variant \
                or data["audits"][variant]["labels_equal"] is not True \
                or data["audits"][variant]["source_time_equal"] is not True:
            raise ValueError("数据回读审计不完整或数据集路径被替换")
    for binding in data["admitted_windows"]:
        gate_path = pathlib.Path(binding["path"]).resolve(strict=True)
        if sha256(gate_path) != binding["sha256"]:
            raise ValueError("原始准入报告已改变")
        gate, *_ = checked_gate(gate_path)
        if gate["source_episode_id"] != binding["source_episode_id"] \
                or gate["output_video_sha256"] != binding["output_video_sha256"]:
            raise ValueError("交接的生成来源错绑")
    return data


def budget_example(cache, row, native, source_indices):
    """增强槽位的实际图像、状态和动作从落盘LeRobot读取，D0槽位沿用已验证缓存。"""
    if row["kind"] != "generated":
        return example_at(cache, row, {}, "cosmos")
    episode_index = source_indices[row["source_episode_id"]]
    source = native.provenance["episodes"][episode_index]
    if source["source_episode_id"] != row["source_episode_id"] or source["source_window"] != row["window"] \
            or row["source_frame"] - row["window"]["start_frame"] != row["window_frame"] \
            or row["is_eval"] is not False or row["horizon"] != 8:
        raise ValueError("配对计划偏离原生部分episode的来源或窗口")
    return native.example_at(episode_index, row["window_frame"])


def train(directory, args):
    ensure_gpu_idle()
    stage_start = time.monotonic()
    import numpy as np
    import torch
    from omegaconf import OmegaConf

    parent = json.loads((args.parent / "checkpoint.json").read_text())
    if parent.get("formal_baseline_training") is not True or parent["step"] != 5000 or parent["recipe"]["name"] != "lora":
        raise ValueError("P0预算测量只能从同一正式5000步基线热启动")
    handoff = checked_handoff(args.handoff, parent)
    manifest_path = pathlib.Path(parent["base_manifest"])
    if sha256(manifest_path) != parent["base_manifest_sha256"]:
        raise ValueError("基座清单偏离正式策略")
    for item in json.loads(manifest_path.read_text())["files"]:
        if sha256(pathlib.Path(parent["base_model"]) / item["file"]) != item["sha256"]:
            raise ValueError("基座模型或处理器发生变化")
    cache_path = pathlib.Path(parent["cache_manifest"]).parent
    isolation = verify_eval_isolation(cache_path, ROOT / "eval/frozen_eval_set.json")
    cache = DemoCache(cache_path)
    if handoff["normalizer"] != cache.manifest["normalizer"]:
        raise ValueError("交接归一化不属于正式D0")
    demos = [{"kind": "demo", "sample_id": r["episode_id"], "source_episode_id": r["episode_id"],
              "source_length": r["length"], "source_split": "demo_train", "is_eval": False,
              "window": {"start_frame": 0, "end_frame": r["length"]}}
             for r in cache.manifest["episodes"] if r["split"] == "demo_train"]
    extra = [{"kind": "generated", "sample_id": "uv-" + b["source_episode_id"],
              "source_episode_id": b["source_episode_id"], "source_length": b["source_length"],
              "source_split": "demo_train", "is_eval": False, "eligible_for_training": True,
              "window": b["window"]} for b in handoff["admitted_windows"]]
    total = args.steps * args.batch
    if total % 2:
        raise ValueError("本P0预算采用精确一半D0与一半增强槽位")
    plan = build_paired_plan(demos, extra, steps=args.steps, batch=args.batch, horizon=8, seed=args.seed,
                             quotas={"generated": total // 2, "self_distill": 0})
    plan["quality_reports_verified"] = True
    plan["sha256"] = paired_plan_sha256(plan)
    dump(directory / "paired_plan.json", plan)
    selected = plan["control" if args.role == "control" else "candidate"]
    variant = "analytic" if args.role == "analytic_ablation" else "cosmos"
    native = StrictNativeWindows(pathlib.Path(handoff["datasets"][variant]), cache.manifest["normalizer"])
    source_indices = {r["source_episode_id"]: i for i, r in enumerate(native.provenance["episodes"])}
    if len(source_indices) != len(extra):
        raise ValueError("原生部分episode数目或来源不唯一")
    model = build_model(pathlib.Path(parent["base_model"]), "lora", args.seed, OmegaConf.create(parent["model_config"]))
    _, restored = restore_checkpoint(args.parent, model)
    del restored
    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    groups = [{"params": [p for name, p in trainable.items() if name.startswith(prefix)], "lr": lr}
              for prefix, lr in [("qwen_vl_interface.", 1e-5), ("action_model.", 1e-4)]]
    if any(not group["params"] for group in groups) or sum(len(g["params"]) for g in groups) != len(trainable):
        raise ValueError("LoRA和动作头优化器分组不完整")
    optimizer = torch.optim.AdamW(groups, fused=True, weight_decay=.01)
    torch.manual_seed(args.seed)
    model.train()
    torch.cuda.reset_peak_memory_stats()
    records = []
    loop_start = time.monotonic()
    with (directory / "logs/steps.jsonl").open("w") as log:
        for step in range(args.steps):
            before = time.perf_counter()
            slots = selected[step * args.batch:(step + 1) * args.batch]
            examples = [budget_example(cache, row, native, source_indices) for row in slots]
            loaded = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss = model(examples)["action_loss"]
            if not torch.isfinite(loss):
                raise ValueError("预算训练出现非有限loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(trainable.values(), 1., foreach=False, error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize()
            after = time.perf_counter()
            row = {"step": step + 1, "loss": float(loss.detach()), "grad_norm": float(norm),
                   "data_wall_s": loaded - before, "model_step_wall_s": after - loaded,
                   "wall_s": after - before, "sample_slots": args.batch}
            records.append(row)
            encoded = json.dumps(row)
            log.write(encoded + "\n"); log.flush()
            print(encoded, flush=True)
    training_wall = time.monotonic() - loop_start
    allocated = torch.cuda.max_memory_allocated() / 1e9
    model.eval()
    probe_batch, locations = cache.batch("demo_validation", 1, np.random.default_rng(args.seed + 2))

    def predict(target):
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(args.seed + 2)
            return np.asarray(target.predict_action(probe_batch)["normalized_actions"]).copy()

    expected = predict(model)
    if expected.shape != (1, 8, 7) or not np.isfinite(expected).all():
        raise ValueError("预算训练后预测动作无效")
    checkpoint = directory / "checkpoint"
    checkpoint.mkdir()
    saved = {"step": parent["step"] + args.steps,
             "trainable_state": {name: p.detach().cpu() for name, p in trainable.items()}}
    torch.save(saved, checkpoint / "state.pt")
    metadata = {**parent, "step": saved["step"], "formal_baseline_training": False,
        "eligible_for_promotion": False, "parent_checkpoint": str(args.parent), "parent_ckpt_sha256": parent["state_sha256"],
        "state_sha256": sha256(checkpoint / "state.pt"), "role": args.role, "image_variant": variant,
        "budget_only": True, "handoff": str(args.handoff), "handoff_sha256": sha256(args.handoff),
        "paired_plan_sha256": plan["sha256"], "eval_overlap_check": isolation,
        "recipe": {**parent["recipe"], "steps": args.steps, "batch": args.batch, "seed": args.seed,
                   "lr_head": 1e-4, "lr_vlm": 1e-5, "scheduler": "constant_for_p0_budget"},
        "prediction_probe": {"seed": args.seed + 2, "locations": locations, "normalized_actions": expected.tolist()}}
    dump(checkpoint / "checkpoint.json", metadata)
    del optimizer, model, trainable, groups, saved, loss, norm
    gc.collect(); torch.cuda.empty_cache()
    model = build_model(pathlib.Path(parent["base_model"]), "lora", args.seed, OmegaConf.create(parent["model_config"]))
    _, restored = restore_checkpoint(checkpoint, model)
    del restored
    model.eval()
    actual = predict(model)
    reload_error = float(np.max(np.abs(actual - expected)))
    if reload_error > 1e-5 or not np.isfinite(actual).all():
        raise ValueError("预算训练检查点保存/重载失败")
    mixture = {}
    for kind in ["demo", "self_distill", "generated", "duplicate_fill"]:
        rows = [r for r in selected if r["kind"] == kind]
        ids_path = directory / (kind + "_ids.txt")
        ids_path.write_text("".join(f"{r['source_episode_id']}@{r['source_frame']}\n" for r in rows))
        mixture[kind] = {"count": len(rows), "unit": "sample_slots_with_replacement", "ids_file": str(ids_path),
                         "unique_source_episodes": len({r["source_episode_id"] for r in rows})}
    measured = records[args.warmup:]
    samples_per_s = len(measured) * args.batch / sum(r["wall_s"] for r in measured)
    ram_values = re.findall(r"RAM\s+(\d+)/\d+MB", (directory / "logs/tegrastats.log").read_text())
    result = {"role": args.role, "parent_ckpt_sha256": parent["state_sha256"], "framework": "starVLA CosmosGR00T",
        "data_mix": mixture, "source_pool": {"demo_episodes": len(demos), "generated_windows": len(extra)},
        "recipe": metadata["recipe"], "eval_overlap_check": isolation,
        "hardware": {"machine": "Jetson AGX Thor", "power_mode": command(["sudo", "-n", "nvpmodel", "-q"])},
        "precision": parent["precision"], "attention": "sdpa", "deepSpeed": False,
        "steps": args.steps, "batch": args.batch, "seed": args.seed, "training_wall_s": training_wall,
        "warmup_excluded_steps": args.warmup, "measured_samples_per_s": samples_per_s,
        "measured_step_wall_s": [r["wall_s"] for r in measured], "steps_log": str(directory / "logs/steps.jsonl"),
        "pytorch_peak_allocated_gb": allocated,
        "peak_mem_gb": max(map(int, ram_values)) * 1024 ** 2 / 1e9 if ram_values else None,
        "peak_mem_scope": "callback结束前tegrastats全机RAM；完整阶段峰值以同目录probe.json为准",
        "wall_s": time.monotonic() - stage_start, "wall_scope": "callback，含验收、加载、训练、保存重载，不含最终probe封存",
        "output_ckpt": str(checkpoint), "output_ckpt_sha256": metadata["state_sha256"],
        "reload_max_abs_diff": reload_error, "paired_plan_sha256": plan["sha256"],
        "handoff": str(args.handoff), "handoff_sha256": sha256(args.handoff), "image_variant": variant,
        "eligible_for_promotion": False, "policy_gain_evaluated": False,
        "scope": "P0格式与计算预算测量；数据仅两条成功训练示范，不代替正式圈训练或策略收益评测"}
    dump(directory / "train_run.json", result)
    return {key: result[key] for key in ["role", "steps", "batch", "seed", "training_wall_s", "measured_samples_per_s",
        "reload_max_abs_diff", "output_ckpt_sha256", "paired_plan_sha256", "eligible_for_promotion"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--parent", type=pathlib.Path, required=True)
    parser.add_argument("--handoff", type=pathlib.Path, required=True)
    parser.add_argument("--role", choices=["candidate", "control", "analytic_ablation"], required=True)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()
    for name in ["parent", "handoff"]:
        path = getattr(arguments, name).resolve(strict=True)
        if not path.is_relative_to(ROOT / "runs/p0"):
            parser.error("P0输入位于项目之外")
        setattr(arguments, name, path)
    if arguments.steps <= 0 or arguments.batch <= 0 or not 0 <= arguments.warmup < arguments.steps \
            or not 0 <= arguments.seed < 2 ** 32:
        parser.error("步数、batch、预热或seed无效")
    raise SystemExit(run_stage("augmented_budget", lambda directory: train(directory, arguments),
        {"role": arguments.role, "steps": arguments.steps, "batch": arguments.batch, "warmup": arguments.warmup,
         "seed": arguments.seed, "precision": "VLM bf16，LoRA/动作头fp32", "attention": "sdpa", "promotion": False}))
