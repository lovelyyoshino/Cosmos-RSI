#!/usr/bin/env python3
"""用当前策略的独立成功反馈与已准入Cosmos数据做同预算策略更新实验。

@author pony
@date 2026-11-09
@version v1.3.0
@last_modified 2026-11-09
@changelog
    - v1.3.0 (2026-11-09): 可选保守续训配方恢复父AdamW状态；显式学习率计划与逐步记录。
    - v1.2.0 (2026-11-09): 后续圈核验旧派生池祖先并复验原质量门；如实记录新优化器与无预热。
    - v1.1.0 (2026-11-09): 后续圈只承接真实已审批晋升；核对采集迭代和生成池规模。
    - v1.0.0 (2026-11-09): 增加真实自身成功池、失败关联选样、精确配额和可追溯权重保存。

本入口复用已经准入的桌体纹理池，不能把选样称为本圈新生成，也不提供光照生成资格。
即使训练完成，仍须独立冻结评测、服务门禁和明确审批才能晋升。
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
import math
import pathlib
import time

from p0_augmented_budget import budget_example, checked_handoff
from p0_background_preview_materials import checked_artifact
from p0_closeout import validate_budget
from p0_common import ROOT, command, ensure_gpu_idle, run_stage
from p0_feedback_judge import checked_collection
from p0_lerobot_handoff import StrictNativeWindows, dump
from p0_paired_sampling import build_paired_plan, paired_plan_sha256
from p0_policy_runtime import DemoCache, build_model, restore_checkpoint, verify_eval_isolation
from p1_eval_gate import sha256


def optimizer_recipe(budget):
    """没有新声明时维持首圈数值路径；保守实验另用预算，不能改写已运行的配方。"""
    options = budget["training"].get("optimizer")
    if options is None:
        return {"lr_vlm": 1e-5, "lr_head": 1e-4, "weight_decay": .01, "warmup_steps": 0,
                "optimizer_state_restored": False, "scheduler": "constant_matched_policy_update_v1"}
    required = {"name", "lr_vlm", "lr_head", "weight_decay", "warmup_steps", "restore_parent_state", "scheduler"}
    if not isinstance(options, dict) or set(options) != required or options["name"] != "adamw" \
            or type(options["restore_parent_state"]) is not bool \
            or options["scheduler"] != "linear_warmup_cosine_min0.1" \
            or type(options["warmup_steps"]) is not int \
            or not 0 <= options["warmup_steps"] < budget["training"]["steps_per_role"]:
        raise ValueError("可选优化器配方、预热或状态恢复声明不合法")
    for key in ["lr_vlm", "lr_head", "weight_decay"]:
        value = options[key]
        if type(value) not in {int, float} or not math.isfinite(value) \
                or value < 0 or (key != "weight_decay" and value == 0):
            raise ValueError("优化器学习率与衰减必须为合法有限数")
    return {key: options[key] for key in ["lr_vlm", "lr_head", "weight_decay", "warmup_steps", "scheduler"]} | \
        {"optimizer_state_restored": options["restore_parent_state"]}


def configure_optimizer(trainable, saved, recipe, steps):
    """恢复矩与步数后重新绑定本次学习率；禁止继承父scheduler的initial_lr峰值。"""
    import torch

    names = [[name for name in trainable if name.startswith(prefix)]
             for prefix in ["qwen_vl_interface.", "action_model."]]
    groups = [{"params": [trainable[name] for name in group_names], "lr": recipe[key]}
              for group_names, key in zip(names, ["lr_vlm", "lr_head"])]
    if any(not group["params"] for group in groups) or sum(len(group["params"]) for group in groups) != len(trainable):
        raise ValueError("策略更新优化器没有完整覆盖LoRA和动作头")
    optimizer = torch.optim.AdamW(groups, fused=True, weight_decay=recipe["weight_decay"])
    if recipe["optimizer_state_restored"]:
        if "optimizer" not in saved or saved.get("optimizer_param_names", names) != names:
            raise ValueError("父权重没有兼容的优化器状态，禁止静默退回新优化器")
        state = saved["optimizer"]
        if len(state["param_groups"]) != len(names) \
                or any(len(group["params"]) != len(group_names) for group, group_names in zip(state["param_groups"], names)):
            raise ValueError("父优化器参数分组与实际模型不符")
        optimizer.load_state_dict(state)
        for group, key in zip(optimizer.param_groups, ["lr_vlm", "lr_head"]):
            group["lr"] = recipe[key]
            group["weight_decay"] = recipe["weight_decay"]
            group.pop("initial_lr", None)
            for parameter in group["params"]:
                moment = optimizer.state.get(parameter)
                if not moment or set(moment) - {"step", "exp_avg", "exp_avg_sq", "max_exp_avg_sq"} \
                        or any(key not in moment or moment[key].shape != parameter.shape
                               or not torch.isfinite(moment[key]).all() for key in ["exp_avg", "exp_avg_sq"]) \
                        or "step" not in moment or not torch.isfinite(moment["step"]).all() \
                        or float(moment["step"]) <= 0:
                    raise ValueError("父优化器实际矩、步数、形状或有限性不合格")

    def schedule(step):
        warmup = recipe["warmup_steps"]
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1., max(0., (step - warmup) / max(1, steps - warmup)))
        return .1 + .9 * (1 + math.cos(math.pi * progress)) / 2

    scheduler = None if recipe["scheduler"] == "constant_matched_policy_update_v1" else \
        torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    return optimizer, scheduler, names


def qualify_successes(episodes, successes, policy_sha):
    """只从同一实际策略且环境成功的完整存档回合建立教师池，不信任单独的资格布尔值。"""
    actual = {row["episode_id"]: row for row in episodes}
    ids = [row["source_episode_id"] for row in successes]
    if len(actual) != len(episodes) or len(ids) != len(set(ids)):
        raise ValueError("反馈或成功池包含重复身份")
    descriptions = []
    for item in successes:
        row = actual.get(item["source_episode_id"])
        if row is None or row["outcome"] != {"success": True, "source": "env"} \
                or row["is_eval"] is not False or row["isolation"]["passed"] is not True \
                or row["policy_ckpt_sha256"] != policy_sha or item["policy_ckpt_sha256"] != policy_sha \
                or item["eligible_for_training"] is not True or item["environment_truth_used"] is not True \
                or item["source_split"] != "rollout_train" or item["is_eval"] is not False \
                or item["window"] != {"start_frame": 0, "end_frame": row["metrics"]["steps"]} \
                or item["source_length"] != row["metrics"]["steps"] \
                or item["lerobot_index"] != row["lerobot_index"] \
                or item["instruction"] != row["instruction"]:
            raise ValueError("自身成功资格与当前策略、环境真值或实际存档错绑")
        descriptions.append({"kind": "self_distill", "sample_id": "self-" + item["source_episode_id"],
            "source_episode_id": item["source_episode_id"], "source_length": item["source_length"],
            "source_split": "rollout_train", "is_eval": False, "eligible_for_training": True,
            "window": dict(item["window"])})
    return descriptions


def check_runtime_slot(row, native, allowed):
    source_id = row["source_episode_id"]
    if source_id not in allowed or row["is_eval"] is not False or row["horizon"] != 8:
        raise ValueError("自身成功抽样不属于已核验资格池")
    index, descriptor = allowed[source_id]
    source = native.provenance["episodes"][index]
    if source["source_episode_id"] != source_id or source["source_success"] is not True \
            or source["is_eval"] is not False or source["source_split"] != "runtime" \
            or source["source_window"] != descriptor["window"] or row["window"] != descriptor["window"] \
            or row["source_frame"] != row["window_frame"] \
            or source["length"] != descriptor["source_length"]:
        raise ValueError("自身成功动作块与实际原生episode不符")
    return native.example_at(index, row["window_frame"])


def check_parent(parent, checkpoint, iteration, promotion_path):
    if type(iteration) is not int or iteration < 1:
        raise ValueError("策略更新圈数必须为正整数")
    if iteration == 1:
        if parent.get("formal_baseline_training") is not True or parent["step"] != 5000 or promotion_path is not None:
            raise ValueError("首圈只能从固定的正式基线开始")
        return None
    if promotion_path is None:
        raise ValueError("后续圈缺少真实晋升证据")
    from p1_eval_gate import checked_promotion

    promotion = checked_promotion(promotion_path)
    if promotion["iteration"] != iteration - 1 or promotion["policy_ckpt_sha256"] != parent["state_sha256"] \
            or pathlib.Path(promotion["promoted_checkpoint"]).resolve(strict=True) != checkpoint \
            or (ROOT / "checkpoints/current").resolve(strict=True) != checkpoint:
        raise ValueError("下一圈的父权重不是上一圈真实晋升的当前策略")
    return {"path": str(promotion_path), "sha256": sha256(promotion_path)}


def checked_recursive_handoff(path, parent, checkpoint, iteration, promotion_path):
    """派生数据绑定初始D0；后续圈验证祖先关系后复验原报告，不改其父权重字段。"""
    check_parent(parent, checkpoint, iteration, promotion_path)
    if iteration == 1:
        return checked_handoff(path, parent), None
    declaration = json.loads(path.read_text())
    anchor_path = pathlib.Path(declaration["parent_checkpoint"]).resolve(strict=True)
    if not anchor_path.is_relative_to(ROOT / "runs/p0"):
        raise ValueError("派生池初始正式来源位于项目之外")
    anchor_meta = anchor_path / "checkpoint.json"
    anchor = json.loads(anchor_meta.read_text())
    if anchor.get("formal_baseline_training") is not True or anchor.get("step") != 5000 \
            or sha256(anchor_path / "state.pt") != anchor["state_sha256"] \
            or anchor["state_sha256"] != declaration["parent_ckpt_sha256"]:
        raise ValueError("后续圈的旧派生池缺少真实正式基线来源")
    compatible = ["cache_manifest_sha256", "normalizer", "base_manifest_sha256", "model_config", "precision", "attention"]
    expected = {key: anchor[key] for key in compatible}
    current_path, current, seen, ancestry = checkpoint, parent, set(), []
    preceding = iteration - 1
    while True:
        current_path = current_path.resolve(strict=True)
        if not current_path.is_relative_to(ROOT) or current_path in seen \
                or sha256(current_path / "state.pt") != current["state_sha256"] \
                or any(current[key] != value for key, value in expected.items()):
            raise ValueError("旧派生池与当前父策略的真实祖先、D0、归一化或模型版本不兼容")
        seen.add(current_path)
        ancestry.append({"checkpoint": str(current_path), "state_sha256": current["state_sha256"],
                         "metadata_sha256": sha256(current_path / "checkpoint.json")})
        if current_path == anchor_path:
            if preceding != 0 or current["state_sha256"] != anchor["state_sha256"]:
                raise ValueError("派生池初始祖先与递归圈数不符")
            break
        if preceding < 1 or current.get("formal_policy_update") is not True \
                or current.get("iteration") != preceding:
            raise ValueError("后续圈的策略祖先存在跳圈或非正式更新")
        previous_path = pathlib.Path(current["parent_checkpoint"]).resolve(strict=True)
        if not previous_path.is_relative_to(ROOT):
            raise ValueError("递归祖先权重路径位于项目之外")
        previous = json.loads((previous_path / "checkpoint.json").read_text())
        if current["parent_ckpt_sha256"] != previous["state_sha256"]:
            raise ValueError("递归父子权重摘要断链")
        current_path, current, preceding = previous_path, previous, preceding - 1
    # 原窗口、实际数据文件、方法控制及独立准入仍按初始正式来源重新验收。
    handoff = checked_handoff(path, anchor)
    return handoff, {"anchor_checkpoint": str(anchor_path), "anchor_metadata_sha256": sha256(anchor_meta),
                     "current_parent_ckpt_sha256": parent["state_sha256"], "ancestry": ancestry,
                     "source_report_rewritten": False, "new_generation_claimed": False}


def prepare(args):
    import yaml

    parent = json.loads((args.parent / "checkpoint.json").read_text())
    if sha256(args.parent / "state.pt") != parent["state_sha256"] or parent["recipe"]["name"] != "lora":
        raise ValueError("策略起点摘要或训练配方不符")
    parent_promotion = check_parent(parent, args.parent, args.iteration, args.parent_promotion)
    budget = yaml.safe_load(args.budget.read_text())
    validate_budget(budget, sha256(ROOT / budget["evaluation"]["definition"]))
    optimizer_recipe(budget)
    if type(budget["training"].get("development_only", False)) is not bool:
        raise ValueError("开发试验范围必须显式使用布尔值")
    if args.seed not in budget["training"]["seeds"]:
        raise ValueError("训练种子未在预算中预声明")
    source, episodes, collection_probe = checked_collection(args.collection)
    expected_init = budget["collection"]["initial_seed"] + (args.iteration - 1) * 1000
    expected_policy = budget["collection"]["policy_seed"] + (args.iteration - 1) * 1000
    expected_cases = {(condition, expected_init + index, expected_policy + index)
                      for condition in budget["collection"]["conditions"]
                      for index in range(budget["collection"]["per_condition"])}
    actual_cases = {(row["condition_id"], row["init_seed"], row["policy_seed"]) for row in episodes}
    if len(episodes) != len(expected_cases) or actual_cases != expected_cases:
        raise ValueError("独立反馈未完整覆盖本圈预声明条件和种子")
    mining_probe = checked_artifact(args.mining, "feedback_mine")
    mining = json.loads(args.mining.read_text())
    if mining["collection_sha256"] != sha256(args.collection) \
            or pathlib.Path(mining["collection"]).resolve(strict=True) != args.collection \
            or mining["policy_ckpt_sha256"] != parent["state_sha256"] \
            or source["policy_ckpt_sha256"] != parent["state_sha256"] \
            or mining.get("decision_authority") != "environment_truth" \
            or mining.get("environment_truth_owns_success_pool") is not True:
        raise ValueError("失败统计或成功池不属于当前策略的环境真值反馈")
    cache = DemoCache(pathlib.Path(parent["cache_manifest"]).parent)
    isolation = verify_eval_isolation(cache.manifest_path.parent, ROOT / budget["evaluation"]["definition"])
    if sha256(cache.manifest_path) != parent["cache_manifest_sha256"] \
            or source["eval_overlap_check"]["eval_set_sha256"] != isolation["eval_set_sha256"]:
        raise ValueError("训练D0或反馈隔离证据偏离冻结版本")
    paths = {"success_manifest": pathlib.Path(mining["success_manifest"]),
             "failure_records": args.mining.parent / "failures/failure_records.jsonl"}
    contents = {}
    for name, path in paths.items():
        path = path.resolve(strict=True)
        if not path.is_relative_to(args.mining.parent) \
                or mining_probe["artifacts_sha256"].get(str(path.relative_to(args.mining.parent))) != sha256(path):
            raise ValueError("成功池或失败记录没有绑定实际挖掘产物")
        contents[name] = [json.loads(line) for line in path.read_text().splitlines()]
    successes = qualify_successes(episodes, contents["success_manifest"], parent["state_sha256"])
    if len(successes) != mining["qualified_successes"] or not successes:
        raise ValueError("缺少真实自身成功池，不补造自蒸馏样本")
    runtime = pathlib.Path(source["dataset"]).resolve(strict=True)
    if not runtime.is_relative_to(args.collection.parent / "rollouts"):
        raise ValueError("运行数据集越界")
    checksums = mining["dataset_files_sha256"]
    if not checksums:
        raise ValueError("缺少运行数据文件证明")
    for relative, expected in checksums.items():
        path = (args.collection.parent / relative).resolve(strict=True)
        if not path.is_relative_to(runtime) or sha256(path) != expected \
                or collection_probe["artifacts_sha256"].get(relative) != expected:
            raise ValueError("自身运行的无损文件或标签已变化")
    handoff, handoff_ancestry = checked_recursive_handoff(
        args.handoff, parent, args.parent, args.iteration, args.parent_promotion)
    conditions = {row["episode_id"]: row for row in episodes}
    background_instructions = {conditions[row["episode_id"]]["instruction"]
        for row in contents["failure_records"] if row["route"] == "cosmos_restyle"
        and row["category"] == "visual" and row["sub_type"] == "background"
        and row["eligible_for_generation"] is True and row["is_eval"] is False}
    generated = []
    selected = []
    demo_lookup = {r["episode_id"]: r for r in cache.manifest["episodes"]}
    for binding in handoff["admitted_windows"]:
        row = demo_lookup[binding["source_episode_id"]]
        if row["instruction"] not in background_instructions:
            continue
        if row["split"] != "demo_train" or row["env_success"] is not True:
            raise ValueError("生成源不是D0训练成功示范")
        selected.append(binding)
        generated.append({"kind": "generated", "sample_id": "uv-" + row["episode_id"],
            "source_episode_id": row["episode_id"], "source_length": row["length"],
            "source_split": "demo_train", "is_eval": False, "eligible_for_training": True,
            "window": dict(binding["window"])})
    if not generated:
        raise ValueError("没有得到统计支持的同任务背景弱条件；不得把预存增强池冒充反馈定向数据")
    if len(generated) > budget["generation"]["maximum_windows"]:
        raise ValueError("合格生成池超过本圈预声明规模，须先固定选样清单")
    demos = [{"kind": "demo", "sample_id": row["episode_id"], "source_episode_id": row["episode_id"],
              "source_length": row["length"], "source_split": "demo_train", "is_eval": False,
              "window": {"start_frame": 0, "end_frame": row["length"]}}
             for row in cache.manifest["episodes"] if row["split"] == "demo_train"]
    quotas = budget["training"]["candidate_quotas"]
    plan = build_paired_plan(demos, generated + successes, steps=budget["training"]["steps_per_role"],
        batch=budget["training"]["batch"], horizon=8, seed=args.seed,
        quotas={key: quotas[key] for key in ["generated", "self_distill"]})
    plan.update(quality_reports_verified=True, iteration=args.iteration,
                feedback_collection_sha256=sha256(args.collection), mining_sha256=sha256(args.mining),
                handoff_sha256=sha256(args.handoff), budget_sha256=sha256(args.budget))
    plan["sha256"] = paired_plan_sha256(plan)
    proof = {"collection": str(args.collection), "collection_sha256": sha256(args.collection),
        "mining": str(args.mining), "mining_sha256": sha256(args.mining),
        "handoff": str(args.handoff), "handoff_sha256": sha256(args.handoff),
        "budget": str(args.budget), "budget_sha256": sha256(args.budget),
        "self_success_episodes": len(successes), "selected_generated_windows": len(generated),
        "selection": "根据当前策略独立背景弱条件选择既有准入Cosmos桌体窗口",
        "new_cosmos_generation_performed": False, "lighting_generation_qualified": False,
        "source_pool_existed_before_feedback": True, "selected_windows": selected,
        "reason_advisory_only": mining["judge_advisory_only"], "eval_overlap_check": isolation}
    proof["parent_promotion"] = parent_promotion
    proof["handoff_ancestry"] = handoff_ancestry
    return parent, budget, cache, plan, proof, handoff, runtime, successes, episodes


def update(directory, args):
    ensure_gpu_idle()
    prepared = prepare(args)
    parent, budget, cache, plan, proof, handoff, runtime, successes, episodes = prepared
    dump(directory / "paired_plan.json", plan)
    dump(directory / "data_qualification.json", proof)
    native_runtime = StrictNativeWindows(runtime, cache.manifest["normalizer"])
    lookup = {row["source_episode_id"]: i for i, row in enumerate(native_runtime.provenance["episodes"])}
    allowed = {row["source_episode_id"]: (lookup[row["source_episode_id"]], row) for row in successes}
    if len(lookup) != len(episodes):
        raise ValueError("运行存档的原生episode身份重复或缺失")
    variant = "analytic" if args.role == "analytic_ablation" else "cosmos"
    native = StrictNativeWindows(pathlib.Path(handoff["datasets"][variant]), cache.manifest["normalizer"])
    generated_lookup = {r["source_episode_id"]: i for i, r in enumerate(native.provenance["episodes"])}
    slots = plan["control" if args.role == "control" else "candidate"]

    def example(row):
        if row["kind"] == "self_distill":
            return check_runtime_slot(row, native_runtime, allowed)
        return budget_example(cache, row, native, generated_lookup)

    # 真实原生读取先覆盖每个新增源的首尾合法动作块，然后才加载策略训练。
    for row in plan["descriptions"]:
        if row["kind"] == "demo":
            continue
        for first in [row["window"]["start_frame"], row["window"]["end_frame"] - 8]:
            example({**row, "source_frame": first, "window_frame": first - row["window"]["start_frame"], "horizon": 8})
    if args.prepare_only:
        result = {"schema": "thor-feedback-policy-data-v1", "iteration": args.iteration,
                  "parent_ckpt_sha256": parent["state_sha256"], "paired_plan_sha256": plan["sha256"],
                  "qualification": proof, "training_performed": False, "eligible_for_promotion": False}
        dump(directory / "prepared_data.json", result)
        return result

    import numpy as np
    import torch
    from omegaconf import OmegaConf

    for row in json.loads(pathlib.Path(parent["base_manifest"]).read_text())["files"]:
        if sha256(pathlib.Path(parent["base_model"]) / row["file"]) != row["sha256"]:
            raise ValueError("策略基座文件已改变")
    if sha256(pathlib.Path(parent["base_manifest"])) != parent["base_manifest_sha256"]:
        raise ValueError("策略基座清单已改变")
    model = build_model(pathlib.Path(parent["base_model"]), "lora", args.seed, OmegaConf.create(parent["model_config"]))
    _, restored = restore_checkpoint(args.parent, model)
    trainable = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    optimization = optimizer_recipe(budget)
    optimizer, scheduler, group_names = configure_optimizer(trainable, restored, optimization, plan["steps"])
    del restored
    torch.manual_seed(args.seed)
    model.train()
    torch.cuda.reset_peak_memory_stats()
    steps, batch = plan["steps"], plan["batch"]
    loop_start = time.monotonic()
    records = []
    with (directory / "logs/steps.jsonl").open("x") as log:
        for step in range(steps):
            before = time.perf_counter()
            learning_rates = [group["lr"] for group in optimizer.param_groups]
            examples = [example(row) for row in slots[step * batch:(step + 1) * batch]]
            optimizer.zero_grad(set_to_none=True)
            loss = model(examples)["action_loss"]
            if not torch.isfinite(loss):
                raise ValueError("正式策略更新出现非有限loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(trainable.values(), 1., foreach=False, error_if_nonfinite=True)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            torch.cuda.synchronize()
            record = {"step": step + 1, "loss": float(loss.detach()), "grad_norm": float(norm),
                      "wall_s": time.perf_counter() - before, "sample_slots": batch,
                      "lr_vlm": learning_rates[0], "lr_head": learning_rates[1]}
            records.append(record)
            line = json.dumps(record)
            log.write(line + "\n"); log.flush()
            print(line, flush=True)
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
        raise ValueError("训练后动作块非法")
    checkpoint = directory / "checkpoint"
    checkpoint.mkdir()
    weights = {"step": parent["step"] + steps,
               "trainable_state": {name: parameter.detach().cpu() for name, parameter in trainable.items()},
               "optimizer": optimizer.state_dict(), "optimizer_param_names": group_names,
               "scheduler": scheduler.state_dict() if scheduler is not None else None}
    torch.save(weights, checkpoint / "state.pt")
    development_only = budget["training"].get("development_only", False)
    metadata = {**parent, "step": weights["step"], "formal_baseline_training": False,
        "formal_policy_update": not development_only, "development_only": development_only,
        "iteration": args.iteration, "eligible_for_promotion": False,
        "parent_checkpoint": str(args.parent), "parent_ckpt_sha256": parent["state_sha256"],
        "state_sha256": sha256(checkpoint / "state.pt"), "role": args.role, "image_variant": variant,
        "budget_only": False, "data_qualification": proof, "paired_plan_sha256": plan["sha256"],
        "eval_overlap_check": proof["eval_overlap_check"],
        "recipe": {**parent["recipe"], "steps": steps, "batch": batch, "seed": args.seed,
                   **optimization},
        "prediction_probe": {"seed": args.seed + 2, "locations": locations, "normalized_actions": expected.tolist()}}
    dump(checkpoint / "checkpoint.json", metadata)
    del model, optimizer, scheduler, trainable, weights, loss, norm
    gc.collect(); torch.cuda.empty_cache()
    model = build_model(pathlib.Path(parent["base_model"]), "lora", args.seed, OmegaConf.create(parent["model_config"]))
    _, restored = restore_checkpoint(checkpoint, model)
    del restored
    model.eval()
    actual = predict(model)
    reload_error = float(np.max(np.abs(actual - expected)))
    if not np.isfinite(actual).all() or reload_error > 1e-5:
        raise ValueError("正式策略更新保存重载不一致")
    mixture = {}
    for kind in ["demo", "self_distill", "generated", "duplicate_fill"]:
        selected = [row for row in slots if row["kind"] == kind]
        path = directory / (kind + "_ids.txt")
        path.write_text("".join(f"{row['source_episode_id']}@{row['source_frame']}\n" for row in selected))
        mixture[kind] = {"count": len(selected), "unit": "sample_slots_with_replacement", "ids_file": str(path),
                         "unique_source_episodes": len({row["source_episode_id"] for row in selected})}
    result = {"schema": "thor-feedback-policy-update-v1", "iteration": args.iteration,
        "role": args.role, "framework": "starVLA CosmosGR00T", "parent_checkpoint": str(args.parent),
        "parent_ckpt_sha256": parent["state_sha256"], "paired_plan_sha256": plan["sha256"],
        "data_qualification": proof, "data_mix": mixture, "recipe": metadata["recipe"],
        "eval_overlap_check": proof["eval_overlap_check"], "training_performed": True,
        "hardware": {"machine": "Jetson AGX Thor", "power_mode": command(["sudo", "-n", "nvpmodel", "-q"])},
        "precision": parent["precision"], "attention": "sdpa", "deepSpeed": False,
        "steps": steps, "batch": batch, "seed": args.seed, "training_wall_s": training_wall,
        "pytorch_peak_allocated_gb": allocated, "reload_max_abs_diff": reload_error,
        "output_ckpt": str(checkpoint), "output_ckpt_sha256": metadata["state_sha256"],
        "image_variant": variant, "development_only": development_only,
        "eligible_for_evaluation": not development_only, "eligible_for_promotion": False,
        "policy_gain_evaluated": False, "new_cosmos_generation_performed": False,
        "scope": "本圈反馈选择既有合格纹理窗口并接入真实自身成功轨迹；新生成、光照资格和策略收益另验"}
    dump(directory / "train_run.json", result)
    return {key: result[key] for key in ["role", "steps", "batch", "seed", "training_wall_s", "reload_max_abs_diff",
                                        "output_ckpt_sha256", "eligible_for_promotion"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ["parent", "collection", "mining", "handoff"]:
        parser.add_argument("--" + name, type=pathlib.Path, required=True)
    parser.add_argument("--budget", type=pathlib.Path, default=ROOT / "configs/iteration_budget.yaml")
    parser.add_argument("--role", choices=["candidate", "control", "analytic_ablation"], required=True)
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--parent-promotion", type=pathlib.Path)
    args = parser.parse_args()
    for name in ["parent", "collection", "mining", "handoff", "budget"]:
        path = getattr(args, name).resolve(strict=True)
        if not path.is_relative_to(ROOT):
            parser.error("策略更新所有输入须位于本项目")
        setattr(args, name, path)
    if args.parent_promotion:
        args.parent_promotion = args.parent_promotion.resolve(strict=True)
        if not args.parent_promotion.is_relative_to(ROOT / "runs/p0"):
            parser.error("父策略晋升记录须位于本项目实际运行目录")
    raise SystemExit(run_stage("p1_policy_data" if args.prepare_only else "p1_policy_train",
        lambda directory: update(directory, args),
        {"iteration": args.iteration, "role": args.role, "seed": args.seed,
         "prepare_only": args.prepare_only, "scope": "当前反馈选样的策略更新，不能计为新Cosmos生成"}))
