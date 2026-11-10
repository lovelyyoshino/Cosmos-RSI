#!/usr/bin/env python3
"""用完整动作序列唯一映射原始示范，保存完整状态与未验证的时序边界。

@author pony
@date 2026-11-06
@version v1.0.0
@last_modified 2026-11-06
@changelog
    - v1.0.0 (2026-11-06): 恢复已有缓存的原始来源，不以近邻或首帧猜测替代全序列核对。
"""
from __future__ import annotations

import argparse
import json
import pathlib

from p0_common import ROOT, run_stage
from p0_policy_runtime import sha256


def map_sources(run_dir: pathlib.Path, raw_manifest: pathlib.Path, cache_path: pathlib.Path) -> dict:
    import h5py
    import numpy as np

    source = json.loads(raw_manifest.read_text())
    cache = json.loads((cache_path / "manifest.json").read_text())
    actions = np.load(cache_path / "actions.npy", mmap_mode="r")
    states = np.load(cache_path / "states.npy", mmap_mode="r")
    mapped, unresolved = [], []
    for file_record in source["files"]:
        path = pathlib.Path(file_record["file"])
        if not path.is_relative_to(ROOT / "data") or sha256(path) != file_record["sha256"]:
            raise RuntimeError("原始HDF5没有通过清单哈希核验")
        with h5py.File(path, "r") as handle:
            group = handle["data"]
            instruction = json.loads(group.attrs["problem_info"])["language_instruction"]
            selected = [item for item in cache["episodes"] if item["instruction"] == instruction]
            raw_actions = {key: value["actions"][:].astype(np.float32) for key, value in group.items()}
            for episode in selected:
                offset, length = episode["offset"], episode["length"]
                actual = np.asarray(actions[offset:offset + length])
                matches = []
                for demo_id, reference in raw_actions.items():
                    # 已有转换只允许裁剪开头；整条剩余序列必须逐值一致。
                    start = len(reference) - length
                    if start >= 0 and np.array_equal(reference[start:], actual):
                        matches.append((demo_id, start))
                if len(matches) != 1:
                    unresolved.append({"episode_id": episode["episode_id"], "match_count": len(matches)})
                    continue
                demo_id, start = matches[0]
                demo = group[demo_id]
                aligned_states = demo["states"][start:start + length]
                if len(aligned_states) != length or not np.isfinite(aligned_states).all():
                    raise RuntimeError("原始完整状态长度或有限性不符合动作映射")
                expected_obs = np.concatenate([demo["obs/ee_states"][:], demo["obs/gripper_states"][:]], axis=1)
                source_obs = np.asarray(states[offset:offset + length])
                comparisons = {}
                for shift in [-1, 0, 1]:
                    first, last = max(0, -start - shift), min(length, len(expected_obs) - start - shift)
                    delta = source_obs[first:last, :3] - expected_obs[start + first + shift:start + last + shift, :3]
                    comparisons[str(shift)] = {"xyz_rmse_m": float(np.sqrt(np.mean(delta ** 2))), "frames": last - first}
                directory = run_dir / "mapped" / episode["episode_id"]
                directory.mkdir(parents=True)
                np.savez_compressed(directory / "full_states.npz", states=aligned_states,
                                    raw_init_state=np.asarray(demo.attrs["init_state"]),
                                    raw_first_state=demo["states"][0])
                (directory / "model.xml").write_text(str(demo.attrs["model_file"]))
                mapped.append({"episode_id": episode["episode_id"], "instruction": instruction,
                               "raw_file": str(path), "raw_file_sha256": file_record["sha256"],
                               "raw_demo_id": demo_id, "raw_start_frame": start, "frames": length,
                               "action_match": "全部帧转fp32后逐值精确相等，唯一匹配",
                               "action_max_abs_diff": 0.0, "raw_state_dimension": aligned_states.shape[1],
                               "observation_xyz_offset_diagnostics": comparisons,
                               "raw_states": str(directory / "full_states.npz"),
                               "model_xml": str(directory / "model.xml"),
                               "physical_replay_validated": False})
    complete = len(mapped) == len(cache["episodes"]) and not unresolved
    result = {"raw_manifest": str(raw_manifest), "raw_manifest_sha256": sha256(raw_manifest),
              "cache_manifest": str(cache_path / "manifest.json"),
              "cache_manifest_sha256": sha256(cache_path / "manifest.json"),
              "mapped_count": len(mapped), "total_cache_episodes": len(cache["episodes"]),
              "all_episode_actions_mapped": complete, "mapped": mapped, "unresolved": unresolved,
              "eval_overlap_check": {"passed": False, "reason": "动作来源匹配不等于物理状态/观测时序回放通过；冻结集尚待构建"},
              "precision": "Thor CPU，动作转fp32精确比对、原始完整状态fp64", "batch": None,
              "seed": None, "steps": 0, "sample_count": len(mapped), "eligible_for_training": False}
    (run_dir / "raw_mapping.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return {key: value for key, value in result.items() if key != "mapped"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--cache", type=pathlib.Path, default=ROOT / "data/libero10_cache_25c73d1ad798")
    args = parser.parse_args()
    args.raw_manifest = args.raw_manifest.resolve(strict=True)
    args.cache = args.cache.resolve(strict=True)
    if not args.raw_manifest.is_relative_to(ROOT / "runs/p0") or not args.cache.is_relative_to(ROOT / "data"):
        parser.error("来源清单和缓存必须位于Thor项目目录")
    raise SystemExit(run_stage("raw_libero_mapping", lambda directory: map_sources(directory, args.raw_manifest, args.cache),
                              {"raw_manifest": str(args.raw_manifest), "scope": "CPU来源恢复，非冻结评测验收"}))
