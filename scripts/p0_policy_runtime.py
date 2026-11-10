#!/usr/bin/env python3
"""P0训练和离线推理共用的数据窗口、模型配方和检查点读取。

@author pony
@date 2026-11-06
@version v1.4.0
@last_modified 2026-11-06
@changelog
    - v1.4.0 (2026-11-06): 正式训练要求冻结集同时保存MuJoCo积分状态。
    - v1.3.0 (2026-11-06): 正式训练要求冻结集包含fixture模型位姿和双相机重建证据。
    - v1.2.0 (2026-11-06): 正式训练强制核验冻结集、初态文件、环境版本与D0数组摘要。
    - v1.1.0 (2026-11-06): 长训前固定确定性算法、cuDNN和cuBLAS设置，避免恢复后数值漂移。
    - v1.0.0 (2026-11-06): 固定双相机和8维状态语义，禁止动作窗口跨回合，校验增量权重。
"""
from __future__ import annotations

import hashlib
from collections import Counter
import json
import os
import pathlib
import sys

from p0_common import ROOT, command


def sha256(path: pathlib.Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_eval_isolation(cache: pathlib.Path, evaluation: pathlib.Path) -> dict:
    cache = cache.resolve(strict=True)
    evaluation = evaluation.resolve(strict=True)
    if not cache.is_relative_to(ROOT / "data") or not evaluation.is_relative_to(ROOT / "eval"):
        raise RuntimeError("正式训练缓存和冻结集必须位于项目约定目录")
    data = json.loads((cache / "manifest.json").read_text())
    frozen = json.loads(evaluation.read_text())
    if frozen.get("freeze_status") != "frozen" or not frozen.get("all_initial_states_replayed"):
        raise RuntimeError("正式训练拒绝尚未冻结或没有实际重建初态的评测集")
    content_hash = frozen.pop("content_sha256")
    actual_content_hash = hashlib.sha256(json.dumps(frozen, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if content_hash != actual_content_hash or evaluation.with_suffix(".sha256").read_text().strip() != sha256(evaluation):
        raise RuntimeError("冻结评测集摘要不一致")
    if frozen.get("initial_snapshot_protocol") != "MjSimState+fixture_body_pos+fixture_body_quat+mjSTATE_INTEGRATION-v2" \
            or any(not row.get("fixture_model_state_saved") or not row.get("integration_state_saved")
                   or row.get("replay_rgb_max_abs_diff_uint8", 256) > 1
                   or row.get("replay_wrist_rgb_max_abs_diff_uint8", 256) > 1 for row in frozen["episodes"]):
        raise RuntimeError("冻结初态缺少fixture位姿或双相机重建证据")
    if not frozen.get("eval_overlap_check", {}).get("passed") or not data.get("all_accepted_episodes_env_success"):
        raise RuntimeError("正式训练的数据来源或评测隔离门禁未通过")
    counts = dict(Counter(row["split"] for row in frozen["episodes"]))
    if counts != frozen["counts"] or counts != {"standard": 200, "target_perturbed": 800, "nontarget_perturbed": 200}:
        raise RuntimeError("正式冻结集的实际回合数不符合评测协议")
    if pathlib.Path(frozen["source_cache_manifest"]) != cache / "manifest.json" or frozen["source_cache_manifest_sha256"] != sha256(cache / "manifest.json"):
        raise RuntimeError("冻结评测核验的不是当前训练数据")
    for name, expected in data["array_sha256"].items():
        path = (cache / name).resolve(strict=True)
        if not path.is_relative_to(cache) or sha256(path) != expected:
            raise RuntimeError("D0数组已经偏离冻结时版本")
    ids = {row["episode_id"] for row in data["episodes"]}
    seeds = {row["init_seed"] for row in data["episodes"]}
    eval_ids = [row["eval_id"] for row in frozen["episodes"]]
    if len(eval_ids) != len(set(eval_ids)) or ids & set(eval_ids) or seeds & {row["init_seed"] for row in frozen["episodes"]}:
        raise RuntimeError("正式训练与评测ID或seed有交集")
    snapshots = {}
    for row in frozen["episodes"]:
        if row.get("is_eval") is not True:
            raise RuntimeError("冻结集中出现可用于训练的回合")
        path = pathlib.Path(row["initial_state_snapshot"]).resolve(strict=True)
        if not path.is_relative_to(ROOT / "runs/p0"):
            raise RuntimeError("冻结初态位于项目产物之外")
        previous = snapshots.setdefault(path, row["initial_state_sha256"])
        if previous != row["initial_state_sha256"]:
            raise RuntimeError("同一初态存在矛盾的摘要")
    for path, expected in snapshots.items():
        if sha256(path) != expected:
            raise RuntimeError("冻结初态文件已改变")
    for row in frozen["conditions"]:
        if sha256(pathlib.Path(row["bddl_path"])) != row["bddl_sha256"]:
            raise RuntimeError("冻结BDDL已改变")
    for repo, expected in frozen["upstream_sha"].items():
        current = command(["git", "-C", str(ROOT / "third_party" / repo), "rev-parse", "HEAD"])
        if current.get("returncode") != 0 or current["stdout"] != expected:
            raise RuntimeError("仿真源码版本已改变，不能复用冻结结果")
    return {"passed": True, "eval_set": str(evaluation), "eval_set_sha256": sha256(evaluation),
            "cache_manifest_sha256": sha256(cache / "manifest.json"), "snapshot_count": len(snapshots),
            "counts": frozen["counts"], "source_episodes": len(ids), "data_arrays_verified": True,
            "scope": frozen["eval_overlap_check"]["scope"]}


def normalize(values, stats, inverse=False):
    import numpy as np

    values = np.asarray(values, dtype=np.float32)
    lower, upper = (np.asarray(stats[key], dtype=np.float32) for key in ["min", "max"])
    span = upper - lower
    valid = span > 1e-8
    if not np.isfinite(values).all() or not np.isfinite(span).all() or (span < 0).any():
        raise RuntimeError("归一化输入或统计值非法")
    if inverse:
        return (lower + (np.clip(values, -1, 1) + 1) * span / 2).astype(np.float32)
    return np.clip(np.where(valid, 2 * (values - lower) / np.where(valid, span, 1) - 1, 0), -1, 1).astype(np.float32)


class DemoCache:
    def __init__(self, path: pathlib.Path):
        import numpy as np

        self.path = path.resolve(strict=True)
        if not self.path.is_relative_to(ROOT / "data"):
            raise RuntimeError("训练缓存必须位于Thor项目data目录")
        self.manifest_path = self.path / "manifest.json"
        self.manifest = json.loads(self.manifest_path.read_text())
        if (self.path / "COMPLETED").read_text().strip() != self.manifest["cache_id"]:
            raise RuntimeError("训练缓存未完成或内容标识不符")
        self.horizon = int(self.manifest["horizon"])
        self.images = np.load(self.path / "images.npy", mmap_mode="r")
        self.states = np.load(self.path / "states.npy", mmap_mode="r")
        self.actions = np.load(self.path / "actions.npy", mmap_mode="r")
        count = self.manifest["frames"]
        if self.images.shape != (count, 2, 256, 256, 3) or self.states.shape != (count, 8) or self.actions.shape != (count, 7):
            raise RuntimeError("预解码缓存形状与约定不符")
        if self.images.dtype != np.uint8 or self.states.dtype != np.float32 or self.actions.dtype != np.float32:
            raise RuntimeError("预解码缓存dtype与约定不符")
        self.splits = {}
        for split in ["demo_train", "demo_validation"]:
            records = [item for item in self.manifest["episodes"] if item["split"] == split]
            if not records or any(item["length"] < self.horizon for item in records):
                raise RuntimeError(f"示范split为空或短于动作窗口：{split}")
            if any(item["offset"] < 0 or item["offset"] + item["length"] > count for item in records):
                raise RuntimeError("回合偏移越过缓存边界")
            cumulative = np.cumsum([item["length"] - self.horizon + 1 for item in records])
            self.splits[split] = (records, cumulative)
        train_ids = {item["episode_id"] for item in self.splits["demo_train"][0]}
        validation_ids = {item["episode_id"] for item in self.splits["demo_validation"][0]}
        if train_ids & validation_ids:
            raise RuntimeError("示范训练集与验证集出现回合交集")

    def batch(self, split: str, batch_size: int, rng):
        import numpy as np
        from PIL import Image

        records, cumulative = self.splits[split]
        examples, locations = [], []
        for window in rng.integers(0, int(cumulative[-1]), size=batch_size):
            episode_index = int(np.searchsorted(cumulative, window, side="right"))
            episode = records[episode_index]
            previous = cumulative[episode_index - 1] if episode_index else 0
            frame = int(window - previous)
            start = episode["offset"] + frame
            examples.append({"image": [Image.fromarray(np.array(self.images[start, view])) for view in range(2)],
                             "lang": episode["instruction"],
                             "state": normalize(self.states[start:start + 1], self.manifest["normalizer"]["state"]),
                             "action": normalize(self.actions[start:start + self.horizon], self.manifest["normalizer"]["action"])})
            locations.append({"episode_id": episode["episode_id"], "frame": frame, "horizon": self.horizon})
        return examples, locations


def build_model(base_model: pathlib.Path, recipe: str, seed: int, config=None):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch
    from omegaconf import OmegaConf

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    np.random.seed(seed)
    sys.path.insert(0, str(ROOT / "third_party/starVLA"))
    from starVLA.model.framework.VLM4A.CosmosGR00T import Cosmos_GR00T

    config = config or OmegaConf.create({
        "framework": {"name": "CosmosGR00T", "qwenvl": {"base_vlm": str(base_model), "attn_implementation": "sdpa"},
                      "action_model": {"state_dim": 8, "action_dim": 7, "action_horizon": 8}},
        "datasets": {"vla_data": {"obs_image_size": [256, 256]}},
    })
    model = Cosmos_GR00T(config).to("cuda")
    if recipe == "head":
        model.qwen_vl_interface.requires_grad_(False)
    elif recipe == "freeze-vision":
        frozen = 0
        for name, parameter in model.qwen_vl_interface.model.named_parameters():
            if name.startswith(("model.visual.", "visual.")):
                parameter.requires_grad_(False)
                frozen += parameter.numel()
        if not frozen:
            raise RuntimeError("未找到视觉塔，拒绝静默全量训练")
    elif recipe == "lora":
        from peft import LoraConfig, get_peft_model

        model.qwen_vl_interface.model = get_peft_model(
            model.qwen_vl_interface.model,
            LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], bias="none"),
        )
    elif recipe != "full":
        raise RuntimeError(f"未知模型配方：{recipe}")
    return model


def restore_checkpoint(directory: pathlib.Path, model):
    import torch

    manifest = json.loads((directory / "checkpoint.json").read_text())
    weights = directory / "state.pt"
    if sha256(weights) != manifest["state_sha256"]:
        raise RuntimeError("检查点SHA256不符")
    saved = torch.load(weights, map_location="cpu", weights_only=True)
    expected = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if set(saved["trainable_state"]) != expected:
        raise RuntimeError("检查点可训练参数集合与模型配方不一致")
    result = model.load_state_dict(saved["trainable_state"], strict=False)
    if result.unexpected_keys or expected.intersection(result.missing_keys):
        raise RuntimeError("检查点未完整恢复可训练参数")
    return manifest, saved
