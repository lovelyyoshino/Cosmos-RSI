#!/usr/bin/env python3
"""在Thor CPU上比较连续训练与断点恢复的最终权重和完整训练状态。

@author pony
@date 2026-11-06
@version v1.0.0
@last_modified 2026-11-06
@changelog
    - v1.0.0 (2026-11-06): 对权重、优化器、调度器及随机状态逐项做精确比较。
"""
from __future__ import annotations

import argparse
import json
import pathlib

from p0_common import ROOT, run_stage
from p0_policy_runtime import sha256


def verify(run_dir: pathlib.Path, args: argparse.Namespace) -> dict:
    import torch

    checkpoints = []
    metadata = []
    for path in [args.original, args.resumed]:
        path = path.resolve(strict=True)
        if not path.is_relative_to(ROOT / "runs/p0"):
            raise RuntimeError("仅接受本项目P0检查点")
        record = json.loads((path / "checkpoint.json").read_text())
        if sha256(path / "state.pt") != record["state_sha256"]:
            raise RuntimeError("检查点文件SHA256不符")
        metadata.append(record)
        checkpoints.append(torch.load(path / "state.pt", map_location="cpu", weights_only=True))
    for key in ["step", "recipe", "base_manifest_sha256", "cache_manifest_sha256", "determinism"]:
        if metadata[0][key] != metadata[1][key]:
            raise RuntimeError(f"比较条件不一致：{key}")
    differences = []
    tensors = 0

    def compare(left, right, path):
        nonlocal tensors
        if isinstance(left, torch.Tensor):
            tensors += 1
            if not isinstance(right, torch.Tensor) or left.dtype != right.dtype or left.shape != right.shape or not torch.equal(left, right):
                differences.append(path)
        elif isinstance(left, dict) and isinstance(right, dict):
            if left.keys() != right.keys():
                differences.append(path + ":keys")
            for key in left.keys() & right.keys():
                compare(left[key], right[key], path + "/" + str(key))
        elif isinstance(left, (list, tuple)) and type(left) is type(right):
            if len(left) != len(right):
                differences.append(path + ":length")
            for index, (a, b) in enumerate(zip(left, right)):
                compare(a, b, path + "/" + str(index))
        elif type(left) is not type(right) or left != right:
            differences.append(path)

    compare(checkpoints[0], checkpoints[1], "training_state")
    compare(metadata[0]["prediction_probe"], metadata[1]["prediction_probe"], "prediction_probe")
    result = {"original": str(args.original), "resumed": str(args.resumed), "step": metadata[0]["step"],
              "recipe": metadata[0]["recipe"], "comparison_device": "CPU，未使用GPU算力",
              "tensor_count": tensors, "difference_count": len(differences), "differences": differences[:30],
              "passed": not differences, "comparison": "精确相等，包含权重、优化器、调度器、随机状态与预测探针"}
    (run_dir / "resume_comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    if differences:
        raise RuntimeError(f"恢复训练与连续训练存在{len(differences)}项差异；详见resume_comparison.json")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--original", type=pathlib.Path, required=True)
    parser.add_argument("--resumed", type=pathlib.Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run_stage("resume_verification", lambda directory: verify(directory, args),
                              {key: str(value) for key, value in vars(args).items()}))
