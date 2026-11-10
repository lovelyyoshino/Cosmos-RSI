#!/usr/bin/env python3
"""用相同真实状态和噪声验证Predict的外部动作条件，并保存与仿真未来的对照。

@author pony
@date 2026-11-09
@version v1.0.0
@last_modified 2026-11-09
@changelog
    - v1.0.0 (2026-11-09): 显式动作条件、真实分支回放、条件槽审计及复制当前图基准。
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import pathlib
import sys
import time
from types import SimpleNamespace

from p0_common import ROOT, ensure_gpu_idle, run_stage
from p0_lerobot_handoff import dump
from p0_policy_runtime import sha256
from p0_predict_adapter import normalized_actions, predictor_observation


def checked_report(path, stage):
    path = path.resolve(strict=True)
    if not path.is_relative_to(ROOT / "runs/p0"):
        raise ValueError("Predict输入必须位于本项目真实运行目录")
    probe = json.loads((path.parent / "probe.json").read_text())
    if probe["stage"] != stage or probe["status"] != "passed" \
            or probe["artifacts_sha256"].get(path.name) != sha256(path):
        raise ValueError("Predict输入未绑定通过的运行记录")
    return json.loads(path.read_text()), probe


def checked_weight(path):
    """只使用下载清单逐文件绑定的模型；记录镜像来源，不冒称官方摘要已复核。"""
    path = path.resolve(strict=True)
    manifests = []
    for manifest in ROOT.glob("runs/p0/model_download_*/model_manifest.json"):
        data = json.loads(manifest.read_text())
        base = pathlib.Path(data["path"])
        if path.is_relative_to(base):
            entry = next((r for r in data["files"] if base / r["file"] == path), None)
            if entry is not None and sha256(path) == entry["sha256"]:
                manifests.append({"manifest": str(manifest), "manifest_sha256": sha256(manifest),
                    "file": str(path), "sha256": entry["sha256"], "backend": data["backend"],
                    "canonical_repo": data["canonical_repo"], "revision": data["revision"],
                    "canonical_hf_hash_crosscheck": data.get("canonical_hf_hash_crosscheck")})
    if not manifests:
        raise ValueError(f"Predict权重或统计没有匹配的校验清单：{path.name}")
    return sorted(manifests, key=lambda r: r["manifest"])[-1]


def build_batch(packet, embedding, stats, cfg):
    import numpy as np
    import torch
    from cosmos_predict2._src.predict2.cosmos_policy.experiments.robot.cosmos_utils import (
        prepare_images_for_model, rescale_proprio,
    )

    obs = predictor_observation(packet)
    wrist, primary = prepare_images_for_model([obs["wrist_image"], obs["primary_image"]], cfg)
    blank = np.zeros_like(primary)
    # 与官方LIBERO包装逐槽一致；未来图仅放当前图占位，绝不把真实未来输入模型。
    slots = [blank, wrist, primary, blank, blank, wrist, primary, blank]
    frames = np.concatenate([blank[None], *[np.repeat(image[None], 4, axis=0) for image in slots]])
    proprio = rescale_proprio(obs["proprio"], stats).astype(np.float32)
    if frames.shape != (33, 224, 224, 3) or not np.isfinite(proprio).all():
        raise ValueError("Predict模型输入的槽位、分辨率或状态归一化不符")
    batch = {"dataset_name": "video_data",
        "video": torch.from_numpy(np.ascontiguousarray(frames.transpose(3, 0, 1, 2)[None])).cuda(),
        "t5_text_embeddings": embedding.to(device="cuda", dtype=torch.bfloat16),
        "fps": torch.tensor([16], dtype=torch.bfloat16, device="cuda"),
        "padding_mask": torch.zeros((1, 1, 224, 224), dtype=torch.bfloat16, device="cuda"),
        "num_conditional_frames": 5,
        "proprio": torch.from_numpy(proprio[None]).to(device="cuda", dtype=torch.bfloat16),
        "rsi_candidate_actions": torch.from_numpy(normalized_actions(packet["actions"], stats)[None]).cuda()}
    indices = {"current_proprio": 1, "current_wrist_image": 2, "current_wrist_image2": -1,
        "current_image": 3, "current_image2": -1, "action": 4, "future_proprio": 5,
        "future_wrist_image": 6, "future_wrist_image2": -1, "future_image": 7,
        "future_image2": -1, "value": 8}
    for key, index in indices.items():
        batch[key + "_latent_idx"] = torch.tensor([index], dtype=torch.int64, device="cuda")
    return batch, obs, np.stack([wrist, primary])


def predict(directory, args):
    ensure_gpu_idle()
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", DETERMINISTIC="true")
    upstream = ROOT / "third_party/cosmos-predict2.5"
    sys.path.insert(0, str(upstream))
    import numpy as np
    import torch
    from PIL import Image
    from cosmos_predict2._src.predict2.cosmos_policy.experiments.robot.cosmos_utils import (
        get_future_images_from_generated_samples, load_dataset_stats, prepare_images_for_model,
    )
    from cosmos_predict2._src.predict2.utils.model_loader import load_model_from_checkpoint

    transition, transition_probe = checked_report(args.transition, "predict_transition")
    encoded, _ = checked_report(args.embeddings, "predict_embeddings")
    if transition["collection_sha256"] != encoded["collection_sha256"]:
        raise ValueError("语言与动作测试不属于同一实际运行")
    text = next(row for row in encoded["embeddings"] if row["instruction"] == transition["instruction"])
    if sha256(pathlib.Path(text["path"])) != text["sha256"]:
        raise ValueError("实际语言条件已改变")
    embedding = torch.load(text["path"], map_location="cpu", weights_only=True)
    if embedding.shape != (1, 512, 100352) or not torch.isfinite(embedding).all():
        raise ValueError("Predict语言条件不符合Reason1 full-concat约定")
    base = ROOT / "checkpoints/base/Cosmos-Predict2.5-2B"
    checkpoint, tokenizer = base / "robot/policy/libero/model.pt", base / "tokenizer.pth"
    stats_path = ROOT / "checkpoints/base/Cosmos-Policy-LIBERO-Predict2-2B/libero_dataset_statistics.json"
    bindings = [checked_weight(path) for path in [checkpoint, tokenizer, stats_path]]
    stats = load_dataset_stats(str(stats_path))
    cfg = SimpleNamespace(suite="libero", use_wrist_image=True, use_third_person_image=True,
        num_wrist_images=1, num_third_person_images=1, use_jpeg_compression=True, trained_with_image_aug=True)
    before = time.monotonic()
    model, _ = load_model_from_checkpoint(
        experiment_name="cosmos_predict2p5_2b_480p_libero__inference_only_no_s3",
        s3_checkpoint_dir=str(checkpoint),
        config_file="cosmos_predict2/_src/predict2/cosmos_policy/config/config.py",
        load_ema_to_reg=False, instantiate_ema=False, seed=args.seed,
        experiment_opts=[f"model.config.tokenizer.vae_pth={tokenizer}",
                         "model.config.net.atten_backend=torch", "model.config.text_encoder_config=null"])
    model = model.eval().cuda()
    load_wall = time.monotonic() - before
    if model.config.state_t != 9 or model.config.min_num_conditional_frames != 4:
        raise ValueError("Predict权重配置不是登记的LIBERO双视角九槽模型")
    trace, original_velocity = [], model.get_velocity_fn_from_batch

    def audited_velocity(batch, *positional, **keywords):
        velocity = original_velocity(batch, *positional, **keywords)
        condition = inspect.getclosurevars(velocity).nonlocals["condition"]
        actual = condition.gt_frames[:, :, 4]
        values = batch["rsi_candidate_actions"].to(dtype=actual.dtype).reshape(1, -1)
        repeated = values.repeat(1, (actual.numel() + values.numel() - 1) // values.numel())[:, :actual.numel()]
        error = (actual.reshape(1, -1) - repeated).abs().max().item()
        clean = condition.condition_video_input_mask_B_C_T_H_W[:, :, 4]
        if error != 0 or not bool(torch.all(clean == 1)):
            raise ValueError("外部候选动作未进入实际干净条件槽，拒绝生成伪动作对照")
        trace.append({"action_slot": 4, "clean_condition": True, "max_injection_error": error,
            "clean_current_state": condition.gt_frames[:, :, :4].detach().float().cpu()})
        return velocity

    model.get_velocity_fn_from_batch = audited_velocity
    records, predictions, truths = [], {}, {}
    for branch in transition["branches"]:
        path = pathlib.Path(branch["packet"]).resolve(strict=True)
        if path.parent != args.transition.parent or sha256(path) != branch["sha256"] \
                or transition_probe["artifacts_sha256"].get(path.name) != branch["sha256"]:
            raise ValueError("实际动作分支或仿真未来已改变")
        with np.load(path, allow_pickle=False) as loaded:
            packet = {key: loaded[key] for key in loaded.files}
        torch.manual_seed(args.seed)
        batch, obs, current_images = build_batch(packet, embedding, stats, cfg)
        future_images = [np.ascontiguousarray(packet[key][-1, :, ::-1]) for key in ["wrist", "agentview"]]
        actual_future = prepare_images_for_model(future_images, cfg)
        before = time.monotonic()
        with torch.inference_mode():
            sample, clean_latent = model.generate_samples_from_batch(batch, n_sample=1,
                num_steps=args.steps, seed=args.seed, guidance=0, shift=5,
                use_variance_scale=False, return_orig_clean_latent_frames=True)
            output = get_future_images_from_generated_samples(model, sample.clone(), cfg,
                clean_latent, [0, 1, 4, 5], 6, -1, 7, -1)
        torch.cuda.synchronize()
        elapsed = time.monotonic() - before
        metrics, predicted = {}, []
        for index, (view, key) in enumerate([("wrist", "future_wrist_image"), ("agentview", "future_image")]):
            image = output[key][0]
            predicted.append(image)
            truth = actual_future[index]
            metrics[view] = {"prediction_mae_255": float(np.abs(image.astype(float) - truth).mean()),
                "copy_current_mae_255": float(np.abs(current_images[index].astype(float) - truth).mean())}
            for label, value in [("prediction", image), ("truth", truth), ("current", current_images[index])]:
                Image.fromarray(value).save(directory / f"{branch['name']}_{view}_{label}.png")
        predictions[branch["name"]] = np.stack(predicted)
        truths[branch["name"]] = np.asarray(actual_future)
        torch.save(sample.detach().cpu(), directory / (branch["name"] + "_latent.pt"))
        records.append({"name": branch["name"], "packet_sha256": branch["sha256"],
            "predict_decode_wall_s": elapsed, "metrics": metrics,
            "action_normalized_min": float(batch["rsi_candidate_actions"].min()),
            "action_normalized_max": float(batch["rsi_candidate_actions"].max())})
        print(json.dumps(records[-1]), flush=True)
    state_equal = len(trace) == 2 and torch.equal(trace[0]["clean_current_state"], trace[1]["clean_current_state"])
    if not state_equal:
        raise ValueError("动作对照的当前视觉与本体条件不一致")
    sensitivity = float(np.abs(predictions["executed"].astype(float) - predictions["hold"]).mean())
    truth_change = float(np.abs(truths["executed"].astype(float) - truths["hold"]).mean())
    if not sensitivity > 0:
        raise ValueError("不同真实动作未改变预测结果，不能声称动作条件已经生效")
    report = {"schema": "thor-predict-external-action-smoke-v1", "transition": str(args.transition),
        "transition_sha256": sha256(args.transition), "embeddings": str(args.embeddings),
        "embeddings_sha256": sha256(args.embeddings), "model_sources": bindings,
        "model_load_wall_s": load_wall, "precision": "bf16", "attention": "torch SDPA", "batch": 1,
        "steps": args.steps, "seed": args.seed, "sample_count": 1, "action_branches": 2,
        "action_horizon": 16, "model_fps_metadata": 16, "environment_control_hz": transition["control_hz"],
        "output": "t+16的双视角未来图，不是逐控制步视频", "branches": records,
        "condition_audit": [{k: v for k, v in row.items() if k != "clean_current_state"} for row in trace],
        "same_current_state": state_equal, "same_sampling_seed": True,
        "predicted_branch_difference_mae_255": sensitivity, "true_branch_difference_mae_255": truth_change,
        "advisory_only": True, "eligible_for_training": False, "eligible_for_planning": False,
        "promotion_performed": False, "real_robot_commands_sent": False,
        "claim": "动作输入与反事实预测接口验证；单状态两动作不能证明物理可靠性、排序收益或RSI效果"}
    dump(directory / "predict_report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--transition", type=pathlib.Path, required=True)
    parser.add_argument("--embeddings", type=pathlib.Path, required=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.transition, args.embeddings = args.transition.resolve(strict=True), args.embeddings.resolve(strict=True)
    if not 1 <= args.steps <= 35:
        parser.error("Predict诊断采样步数需要在1至35之间")
    raise SystemExit(run_stage("predict_external_action", lambda directory: predict(directory, args),
        {"precision": "bf16", "batch": 1, "steps": args.steps, "seed": args.seed, "sample_count": 1}))
