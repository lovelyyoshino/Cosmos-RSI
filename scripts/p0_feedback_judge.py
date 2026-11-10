#!/usr/bin/env python3
"""独立Reason2读取运行关键帧，逐回合保存判断并对环境真值计算校准指标。

@author pony
@date 2026-11-08
@version v1.3.0
@last_modified 2026-11-09
@changelog
    - v1.3.0 (2026-11-09): 可独立比较既有Reason1判官；保留模型名与全部未达标结果。
    - v1.2.0 (2026-11-09): 可比较bf16与FP8；非法输出显式弃权并计错，不中断整批或伪造失败标签。
    - v1.1.0 (2026-11-09): 增加媒体优先、保留思考的独立提示版本；严格解析最终JSON，保留原实验配置。
    - v1.0.1 (2026-11-09): SIGTERM/SIGINT展开finally，回收已创建的判官引擎。
    - v1.0.0 (2026-11-08): 六关键图、盲判环境结果、CUDA Graph、逐样本存档及有限样本校准。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import re
import signal
import time

from p0_common import ROOT, ensure_gpu_idle, run_stage
from p0_lerobot_handoff import dump
from p0_policy_runtime import sha256


def parse_verdict(text):
    """只读取最终答案；思考中的假设和JSON不得作为任务结论。"""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1].strip()
    elif "<think>" in text:
        raise ValueError("判官思考未结束，拒绝把中间内容作为答案")
    if "<answer>" in text:
        if "</answer>" not in text or text.count("<answer>") != 1:
            raise ValueError("判官最终答案标记不完整")
        text = text.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fenced:
        text = fenced[1]
    verdict = json.loads(text)
    if not isinstance(verdict, dict) or set(verdict) != {"success", "confidence", "evidence", "failure_stage"} \
            or type(verdict["success"]) is not bool or type(verdict["confidence"]) not in {int, float} \
            or not math.isfinite(verdict["confidence"]) or not 0 <= verdict["confidence"] <= 1 \
            or not isinstance(verdict["evidence"], str) or not verdict["evidence"].strip() \
            or verdict["failure_stage"] not in {"approach", "grasp", "transport", "place", "completed", "unknown"}:
        raise ValueError("判官最终JSON字段、标签或置信度非法")
    return verdict


def reasoning_messages(inputs, instruction, with_reasoning=True):
    """按官方媒体优先顺序输入；只提供图像、任务和时刻，环境标签始终不可见。"""
    labels = "; ".join(f"image {index + 1}: {item['phase']} / {item['view']} / t={item['timestamp']}s"
                       for index, item in enumerate(inputs))
    content = [{"type": "image", "image": item["path"]} for item in inputs]
    content.append({"type": "text", "text":
        f"These six images show one robot episode: {labels}. Task: {instruction}. "
        "Assess the observed outcome, not the intended action. Identify the target object and destination. "
        "Compare their initial and terminal locations. Check the exact destination relation, whether the object "
        "is still held, and whether occlusion prevents verification. Motion towards a goal does not prove completion. "
        "Use visible terminal evidence; if completion cannot be verified, return success=false, low confidence "
        "and failure_stage=unknown. Do not invent events between the images. "
        + ("Reason briefly within <think>...</think>, then provide only a JSON object within <answer>...</answer>. "
           if with_reasoning else "Return only the requested JSON object. ") +
        "The JSON must have success (boolean), confidence (number from 0 to 1), evidence (one short sentence "
        "describing observed object/destination relation), and failure_stage "
        "(approach, grasp, transport, place, completed, or unknown)."})
    return [{"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {"role": "user", "content": content}]


def checked_collection(path):
    path = path.resolve(strict=True)
    if not path.is_relative_to(ROOT / "runs/p0"):
        raise ValueError("反馈来源必须在本项目P0目录")
    source = json.loads(path.read_text())
    probe = json.loads((path.parent / "probe.json").read_text())
    if probe.get("status") != "passed" or probe.get("stage") != "feedback_rollout" \
            or probe["artifacts_sha256"].get(path.name) != sha256(path) \
            or source["schema"] != "thor-feedback-rollout-v1" or source["server_stopped"] is not True:
        raise ValueError("来源不是完成且已退出服务的独立反馈采集")
    episodes = pathlib.Path(source["episodes"]).resolve(strict=True)
    relative = str(episodes.relative_to(path.parent))
    if probe["artifacts_sha256"].get(relative) != sha256(episodes):
        raise ValueError("采集回合清单已改变")
    rows = [json.loads(line) for line in episodes.read_text().splitlines()]
    if len(rows) != source["sample_count"] or len({row["episode_id"] for row in rows}) != len(rows):
        raise ValueError("运行回合数或唯一性不符")
    for row in rows:
        if row["is_eval"] is not False or row["outcome"]["source"] != "env" \
                or type(row["outcome"]["success"]) is not bool or row["isolation"]["passed"] is not True \
                or row["policy_ckpt_sha256"] != source["policy_ckpt_sha256"]:
            raise ValueError("判官输入不是有隔离证据的独立环境真值回合")
    return source, rows, probe


def calibration_metrics(rows, config):
    if not rows or len({row["episode_id"] for row in rows}) != len(rows):
        raise ValueError("校准需要不重复的真实回合")
    for row in rows:
        valid = row.get("valid", True)
        if type(valid) is not bool or type(row["environment_truth"]) is not bool \
                or (type(row["verdict"]["success"]) is not bool if valid else row["verdict"]["success"] is not None) \
                or not math.isfinite(row["verdict"]["confidence"]) or not 0 <= row["verdict"]["confidence"] <= 1:
            raise ValueError("判官标签或置信度非法")
    tp = sum(row["environment_truth"] and row["verdict"]["success"] is True for row in rows)
    tn = sum(not row["environment_truth"] and row["verdict"]["success"] is False for row in rows)
    fp = sum(not row["environment_truth"] and row["verdict"]["success"] is True for row in rows)
    fn = sum(row["environment_truth"] and row["verdict"]["success"] is False for row in rows)
    positives = sum(row["environment_truth"] for row in rows)
    negatives = len(rows) - positives
    accuracy = (tp + tn) / len(rows)
    z, n = 1.959963984540054, len(rows)
    center = (accuracy + z * z / (2 * n)) / (1 + z * z / n)
    radius = z * math.sqrt(accuracy * (1 - accuracy) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    enough = n >= config["min_calibration_episodes"] and positives >= config["min_success_episodes"] \
        and negatives >= config["min_failure_episodes"]
    success_recall, failure_recall = (tp / positives if positives else None), (tn / negatives if negatives else None)
    recalls_passed = success_recall is not None and failure_recall is not None \
        and success_recall >= config.get("min_success_recall", config["min_accuracy"]) \
        and failure_recall >= config.get("min_failure_recall", config["min_accuracy"])
    return {"episodes": n, "success_truth_count": positives, "failure_truth_count": negatives,
        "true_positive": tp, "true_negative": tn, "false_positive": fp, "false_negative": fn,
        "abstentions": sum(not row.get("valid", True) for row in rows),
        "accuracy": accuracy, "success_recall": success_recall,
        "failure_recall": failure_recall, "accuracy_wilson95": [center - radius, center + radius],
        "sample_requirements_met": enough,
        "calibration_passed": bool(enough and recalls_passed and accuracy >= config["min_accuracy"]),
        "thresholds": {key: config[key] for key in ["min_calibration_episodes", "min_success_episodes",
                                                     "min_failure_episodes", "min_accuracy"]},
        "boundary": "校准针对本次独立采集分布；区间按回合二项近似，重复初态/条件存在相关性；模型自报置信度未校准"}


def prepare_inputs(directory, row, probe, collection_root):
    import numpy as np
    from PIL import Image

    trajectory = pathlib.Path(row["trajectory"]).resolve(strict=True)
    if not trajectory.is_relative_to(collection_root) or sha256(trajectory) != row["trajectory_sha256"] \
            or probe["artifacts_sha256"].get(str(trajectory.relative_to(collection_root))) != sha256(trajectory):
        raise ValueError("原始无损运行轨迹错绑或已改变")
    destination = directory / "inputs" / row["episode_id"]
    destination.mkdir(parents=True, exist_ok=False)
    inputs, images = [], []
    with np.load(trajectory) as stored:
        count = len(stored["action"])
        for name, frame_index in [("initial", 0), ("middle", count // 2), ("terminal", count)]:
            for view in ["agentview", "wrist"]:
                if name == "terminal":
                    original = pathlib.Path(row["artifacts"]) / f"{view}_last.png"
                    if not original.resolve(strict=True).is_relative_to(collection_root) \
                            or probe["artifacts_sha256"].get(str(original.relative_to(collection_root))) != sha256(original):
                        raise ValueError("终帧未绑定实际采集证据")
                    with Image.open(original) as loaded:
                        image = loaded.convert("RGB").copy()
                else:
                    image = Image.fromarray(stored[view][frame_index])
                if image.size != (256, 256):
                    raise ValueError("判官关键帧分辨率不符")
                path = destination / f"{name}_{view}.png"
                image.save(path)
                inputs.append({"phase": name, "frame": frame_index, "timestamp": frame_index / 20,
                               "view": view, "path": str(path), "sha256": sha256(path)})
                images.append(image)
    return inputs, images


def evaluate(directory, args):
    ensure_gpu_idle()
    import yaml

    config = yaml.safe_load(args.config.read_text())
    if config["version"] != "thor-reason2-feedback-v1" or config["frames"] != ["initial", "middle", "terminal"] \
            or config["views"] != ["agentview", "wrist"]:
        raise ValueError("判官配置版本或关键帧约定不符")
    mode = config.get("response_mode", "structured_json")
    if mode not in {"structured_json", "reasoning_json"}:
        raise ValueError("未知判官输出模式")
    quantization = config.get("quantization", "fp8")
    prompt_variant = config.get("prompt_variant", "media_first" if mode == "reasoning_json" else "text_first")
    if quantization not in {None, "fp8"} or prompt_variant not in {"media_first", "text_first"}:
        raise ValueError("未知判官精度或提示顺序")
    model_length = config.get("max_model_len", 2048)
    pixels = config.get("pixels_per_image", 65536)
    if type(model_length) is not int or not 2048 <= model_length <= 16384 \
            or pixels not in {65536, 262144} or config["max_new_tokens"] >= model_length:
        raise ValueError("判官上下文或图像预算非法")
    source, episodes, probe = checked_collection(args.collection)
    os.environ.update({"VLLM_WORKER_MULTIPROC_METHOD": "spawn", "VLLM_NO_USAGE_STATS": "1",
                       "VLLM_HOST_IP": "127.0.0.1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    model_name = config.get("model_name", "Cosmos-Reason2-2B")
    if model_name not in {"Cosmos-Reason2-2B", "Cosmos-Reason1-7B"}:
        raise ValueError("只允许已经接入的Cosmos独立判官模型")
    model_path = ROOT / "checkpoints/base" / model_name
    processor = AutoProcessor.from_pretrained(model_path)
    schema = {"type": "object", "properties": {"success": {"type": "boolean"},
        "confidence": {"type": "number"}, "evidence": {"type": "string"},
        "failure_stage": {"type": "string", "enum": ["approach", "grasp", "transport", "place", "completed", "unknown"]}},
        "required": ["success", "confidence", "evidence", "failure_stage"], "additionalProperties": False}
    llm, results = None, []
    try:
        llm = LLM(model=str(model_path), tokenizer=str(model_path), trust_remote_code=False,
            dtype="bfloat16", quantization=quantization, seed=config["seed"], enforce_eager=False,
            gpu_memory_utilization=.15, kv_cache_memory_bytes=2 * 1024 ** 3,
            max_model_len=model_length, max_num_seqs=1, max_num_batched_tokens=model_length,
            limit_mm_per_prompt={"image": 6, "video": 0},
            mm_processor_kwargs={"min_pixels": pixels, "max_pixels": pixels},
            hf_overrides={"tie_word_embeddings": False}, compilation_config={"cudagraph_capture_sizes": [1]})
        graph = str(llm.llm_engine.vllm_config.compilation_config.cudagraph_mode)
        if "NONE" in graph.upper():
            raise RuntimeError("反馈判官必须启用CUDA Graph")
        params = SamplingParams(temperature=config["temperature"], max_tokens=config["max_new_tokens"],
            seed=config["seed"], structured_outputs=StructuredOutputsParams(json=schema)
            if mode == "structured_json" else None)
        with (directory / "verdicts.jsonl").open("x") as journal:
            for episode in episodes:
                inputs, images = prepare_inputs(directory, episode, probe, args.collection.parent)
                content = [{"type": "text", "text": "Evaluate a robot task using the six labeled images below. "
                    "The images are initial, middle, and terminal observations from external and wrist cameras. "
                    "Decide success only from visible terminal task evidence. Earlier motion alone does not prove completion. "
                    "If the goal cannot be verified, return success=false and low confidence. "
                    f"Task: {episode['instruction']}. Return JSON: success, confidence in [0,1], "
                    "evidence (one short sentence), and failure_stage (approach, grasp, transport, place, completed, or unknown)."}]
                for item in inputs:
                    content.extend([{"type": "text", "text": f"{item['phase']} / {item['view']} / t={item['timestamp']} seconds"},
                                    {"type": "image", "image": item["path"]}])
                messages = reasoning_messages(inputs, episode["instruction"], mode == "reasoning_json") if prompt_variant == "media_first" \
                    else [{"role": "user", "content": content}]
                prompt = processor.apply_chat_template(messages,
                                                        tokenize=False, add_generation_prompt=True)
                started = time.monotonic()
                output = llm.generate([{"prompt": prompt, "multi_modal_data": {"image": images}}],
                                      sampling_params=params, use_tqdm=False)[0].outputs[0]
                # 同时保存原始模型输出和实际提示，方便定位格式失败与视觉误判。
                dump(directory / "responses" / (episode["episode_id"] + ".json"),
                     {"prompt": prompt, "raw_output": output.text, "response_mode": mode,
                      "finish_reason": output.finish_reason})
                valid, parse_error = True, None
                try:
                    if output.finish_reason != "stop":
                        raise ValueError("判官结构化输出未正常结束")
                    verdict = parse_verdict(output.text)
                except (ValueError, TypeError) as exc:
                    valid, parse_error = False, str(exc)
                    verdict = {"success": None, "confidence": 0., "evidence": "模型输出无法合法解析；原始响应已单独保存",
                               "failure_stage": "unknown"}
                row = {"episode_id": episode["episode_id"], "policy_ckpt_sha256": episode["policy_ckpt_sha256"],
                    "role": "unmodified_base_judge", "environment_truth": episode["outcome"]["success"],
                    "verdict": verdict, "inputs": inputs, "generated_tokens": len(output.token_ids),
                    "generate_wall_s": time.monotonic() - started, "finish_reason": output.finish_reason,
                    "is_eval": False, "truth_hidden_from_model": True, "response_mode": mode,
                    "valid": valid, "parse_error": parse_error}
                calibration_metrics([row], config)
                results.append(row)
                journal.write(json.dumps(row, ensure_ascii=False) + "\n"); journal.flush()
                print(json.dumps({"episode_id": row["episode_id"], "judge_success": verdict["success"],
                                  "env_success": row["environment_truth"], "completed": len(results)}), flush=True)
    finally:
        if llm is not None:
            llm.llm_engine.engine_core.shutdown()
    report = {"schema": config["version"], "collection": str(args.collection), "collection_sha256": sha256(args.collection),
        "model": str(model_path), "model_name": model_name, "role": "unmodified_base_judge",
        "precision": "FP8 weights / bf16 activations" if quantization == "fp8" else "bf16 weights and activations",
        "batch": 1, "seed": config["seed"], "max_new_tokens": config["max_new_tokens"], "sample_count": len(results),
        "policy_ckpt_sha256": source["policy_ckpt_sha256"], "calibration": calibration_metrics(results, config),
        "unique_init_seeds": len({row["init_seed"] for row in episodes}), "cudagraph_mode": graph,
        "config": config, "response_mode": mode, "prompt_variant": prompt_variant,
        "enforce_eager": False, "truth_hidden_from_model": True, "verdicts": str(directory / "verdicts.jsonl"),
        "judgment_used_as_environment_truth": False, "real_robot_commands_sent": False,
        "claim": "独立模型关键帧判断与环境真值对照；未达到样本与准确率门槛时不得宣称已校准"}
    dump(directory / "judge_report.json", report)
    return report


if __name__ == "__main__":
    def interrupted(number, _frame):
        raise RuntimeError(f"反馈判官收到信号{number}，关闭本次模型引擎")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--collection", type=pathlib.Path, required=True)
    parser.add_argument("--config", type=pathlib.Path, default=ROOT / "configs/judge.yaml")
    args = parser.parse_args()
    args.collection = args.collection.resolve(strict=True)
    args.config = args.config.resolve(strict=True)
    raise SystemExit(run_stage("feedback_judge", lambda directory: evaluate(directory, args),
                     {key: str(value) for key, value in vars(args).items()}))
