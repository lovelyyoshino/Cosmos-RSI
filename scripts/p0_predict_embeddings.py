#!/usr/bin/env python3
"""用已有Reason1权重离线计算Predict的语言条件，单独退出后再加载世界模型。

@author pony
@date 2026-11-09
@version v1.1.0
@last_modified 2026-11-09
@changelog
    - v1.1.0 (2026-11-09): 显式请求token列表，兼容新版transformers默认返回BatchEncoding。
    - v1.0.0 (2026-11-09): 官方提示模板、512右填充和逐层归一化full-concat；使用已校验HF格式本地模型。
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import time

from p0_common import ROOT, ensure_gpu_idle, run_stage
from p0_feedback_judge import checked_collection
from p0_lerobot_handoff import dump
from p0_policy_runtime import sha256


def encode(directory, args):
    ensure_gpu_idle()
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    import torch
    from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

    _, episodes, _ = checked_collection(args.collection)
    instructions = sorted({row["instruction"] for row in episodes})
    base = ROOT / "checkpoints/base/Cosmos-Reason1-7B"
    manifests = []
    for candidate in ROOT.glob("runs/p0/model_download_*/model_manifest.json"):
        data = json.loads(candidate.read_text())
        if pathlib.Path(data["path"]) == base:
            manifests.append((candidate, data))
    if not manifests:
        raise ValueError("缺少本地Reason1真实权重清单")
    manifest, binding = sorted(manifests)[-1]
    if not binding["files"]:
        raise ValueError("Reason1模型清单为空")
    for item in binding["files"]:
        if sha256(base / item["file"]) != item["sha256"]:
            raise ValueError("Reason1本地模型已改变")
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(base, torch_dtype=torch.bfloat16,
        device_map="cuda", attn_implementation="sdpa", local_files_only=True).eval()
    results = []
    for index, instruction in enumerate(instructions):
        conversation = [
            {"role": "system", "content": [{"type": "text", "text":
                "You are a helpful assistant who will provide prompts to an image generator."}]},
            {"role": "user", "content": [{"type": "text", "text": instruction}]}]
        ids = tokenizer.apply_chat_template(conversation, tokenize=True, add_generation_prompt=False, return_dict=False)
        if not isinstance(ids, list) or not ids or len(ids) >= 512:
            raise ValueError("语言条件的实际token序列超出既定512预算")
        padded = ids + [tokenizer.pad_token_id] * (512 - len(ids))
        started = time.monotonic()
        with torch.inference_mode():
            output = model(input_ids=torch.tensor([padded], dtype=torch.long, device="cuda"),
                           use_cache=False, output_hidden_states=True, return_dict=True)
            layers = output.hidden_states[1:]
            normalized = [(layer - layer.mean(dim=-1, keepdim=True)) /
                          (layer.std(dim=-1, keepdim=True) + 1e-8) for layer in layers]
            embedding = torch.cat(normalized, dim=-1).to(dtype=torch.bfloat16).cpu()
        if embedding.shape != (1, 512, 100352) or not torch.isfinite(embedding).all():
            raise ValueError("Reason1 full-concat的实际形状或有限性不符")
        path = directory / f"embedding_{index:02d}.pt"
        torch.save(embedding, path)
        results.append({"instruction": instruction, "path": str(path), "sha256": sha256(path),
                        "shape": list(embedding.shape), "tokens_before_padding": len(ids),
                        "forward_wall_s": time.monotonic() - started})
        del output, layers, normalized, embedding
        print(json.dumps({"completed": len(results), "instructions": len(instructions)}), flush=True)
    report = {"schema": "thor-predict-text-embeddings-v1", "collection": str(args.collection),
        "collection_sha256": sha256(args.collection), "model": str(base), "model_manifest": str(manifest),
        "model_manifest_sha256": sha256(manifest), "embeddings": results, "precision": "bf16", "batch": 1,
        "seed": None, "attention": "sdpa", "sequence_length": 512, "padding": "right; no attention mask, per official encoder",
        "backend": "HF-format Reason1; official system prompt and normalized layers 1..28 concatenated",
        "is_eval": False, "generation_performed": False, "promotion_performed": False}
    dump(directory / "embeddings.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--collection", type=pathlib.Path, required=True)
    args = parser.parse_args()
    args.collection = args.collection.resolve(strict=True)
    raise SystemExit(run_stage("predict_embeddings", lambda directory: encode(directory, args),
                     {"collection": str(args.collection), "compute": "Thor单卡独立语言编码阶段"}))
