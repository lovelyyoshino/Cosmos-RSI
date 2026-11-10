#!/usr/bin/env python3
"""原版Reason2的单图判官验证；明确开启CUDA Graph，不报告总体准确率。

@author pony
@date 2026-11-06
@version v1.1.0
@last_modified 2026-11-06
@changelog
    - v1.1.0 (2026-11-06): 将vLLM内部通信限定在本机回环地址。
    - v1.0.0 (2026-11-06): 使用独立原模型、FP8和结构化输出验证判官链路。
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import time

from p0_common import ROOT, ensure_gpu_idle, run_stage


def judge(run_dir: pathlib.Path, args: argparse.Namespace) -> dict:
    ensure_gpu_idle()
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["VLLM_NO_USAGE_STATS"] = "1"
    os.environ["VLLM_HOST_IP"] = "127.0.0.1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from PIL import Image
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    episode_path = args.episode
    if episode_path is None:
        options = [path for path in sorted(ROOT.glob("runs/p0/sim_smoke_*/episode.json"))
                   if "task_success" in json.loads(path.read_text())]
        if not options:
            raise RuntimeError("缺少带环境真值的末帧样本")
        episode_path = options[-1]
    episode = json.loads(episode_path.read_text())
    image_path = episode_path.parent / "agentview_last.png"
    image = Image.open(image_path).convert("RGB")
    model_path = ROOT / "checkpoints/base/Cosmos-Reason2-2B"
    processor = AutoProcessor.from_pretrained(model_path)
    text = ("Evaluate whether the robot task is completed based only on the final image. "
            f"Task: {episode['instruction']}. Return JSON with success (boolean), confidence "
            "(number between 0 and 1), and evidence (one short sentence describing visible evidence).")
    messages = [{"role": "user", "content": [{"type": "image", "image": str(image_path)},
                                                {"type": "text", "text": text}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    schema = {"type": "object", "properties": {"success": {"type": "boolean"},
              "confidence": {"type": "number"}, "evidence": {"type": "string"}},
              "required": ["success", "confidence", "evidence"], "additionalProperties": False}
    llm = None
    try:
        llm = LLM(model=str(model_path), tokenizer=str(model_path), trust_remote_code=False,
                  dtype="bfloat16", quantization="fp8", seed=args.seed, enforce_eager=False,
                  gpu_memory_utilization=0.15, kv_cache_memory_bytes=2 * 1024 ** 3,
                  max_model_len=2048, max_num_seqs=1, max_num_batched_tokens=2048,
                  limit_mm_per_prompt={"image": 1, "video": 0},
                  mm_processor_kwargs={"min_pixels": 65536, "max_pixels": 65536},
                  hf_overrides={"tie_word_embeddings": False},
                  compilation_config={"cudagraph_capture_sizes": [1]})
        graph_mode = str(llm.llm_engine.vllm_config.compilation_config.cudagraph_mode)
        if "NONE" in graph_mode.upper():
            raise RuntimeError("vLLM未启用CUDA Graph")
        params = SamplingParams(temperature=0, max_tokens=192, seed=args.seed,
                                structured_outputs=StructuredOutputsParams(json=schema))
        before = time.perf_counter()
        outputs = llm.generate([{"prompt": prompt, "multi_modal_data": {"image": image}}],
                               sampling_params=params, use_tqdm=False)
        wall_s = time.perf_counter() - before
        response = outputs[0].outputs[0]
        verdict = json.loads(response.text)
        if not 0 <= verdict["confidence"] <= 1:
            raise RuntimeError("判官置信度不在[0,1]范围")
        truth = episode.get("task_success")
        result = {"model": str(model_path), "role": "unmodified_base_judge", "precision": "FP8 weights / bf16 activations",
                  "batch": 1, "sample_count": 1, "steps": "一次生成", "seed": args.seed,
                  "episode": str(episode_path), "image": str(image_path), "verdict": verdict,
                  "environment_truth": truth, "single_case_agreement": verdict["success"] == truth if truth is not None else None,
                  "cudagraph_mode": graph_mode, "enforce_eager": False,
                  "kv_cache_bytes": 2 * 1024 ** 3, "max_new_tokens": 192,
                  "generated_tokens": len(response.token_ids), "generate_wall_s": wall_s,
                  "finish_reason": response.finish_reason,
                  "hf_overrides": {"tie_word_embeddings": False},
                  "accuracy_claim": "无；单个未完成任务末帧仅用于链路验证，不是判官校准",
                  "vllm_version": __import__("vllm").__version__}
        (run_dir / "verdict.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        return result
    finally:
        if llm is not None:
            llm.llm_engine.engine_core.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--episode", type=pathlib.Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return run_stage("judge_smoke", lambda run_dir: judge(run_dir, args),
                     {k: str(v) if isinstance(v, pathlib.Path) else v for k, v in vars(args).items()})


if __name__ == "__main__":
    raise SystemExit(main())
