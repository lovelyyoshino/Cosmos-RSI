#!/usr/bin/env python3
"""从实际BDDL和官方分类表构造评测草案；未捕获初态前不宣称已冻结。

@author pony
@date 2026-11-06
@version v1.0.0
@last_modified 2026-11-06
@changelog
    - v1.0.0 (2026-11-06): 校验真实任务、扰动、文件哈希和种子命名空间，避免手写不存在的场景。
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import pathlib
import re

from p0_common import ROOT, run_stage


def digest(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def catalog(run_dir: pathlib.Path) -> dict:
    cache = ROOT / "data/libero10_cache_25c73d1ad798/manifest.json"
    data = json.loads(cache.read_text())
    plus = ROOT / "third_party/LIBERO-plus/libero/libero"
    standard = ROOT / "third_party/LIBERO/libero/libero"
    classifications = plus / "benchmark/task_classification.json"
    categories = json.loads(classifications.read_text())["libero_10"]
    conditions = []
    cases = []
    policy_seeds = [34001, 34002]
    for task_index, (instruction, base_file) in enumerate(sorted(data["tasks"].items())):
        base_name = pathlib.Path(base_file).stem
        path = standard / "bddl_files/libero_10" / base_file
        if not path.is_file():
            raise RuntimeError(f"标准任务BDDL不存在：{path}")
        variants = [("standard", "standard", "LIBERO", path, None)]
        for split, name, category, suffix in [
            ("target_perturbed", "lighting", "Light Conditions", r"_light_\d+"),
            ("target_perturbed", "background", "Background Textures", r"_(?:table|tb)_\d+"),
            ("nontarget_perturbed", "layout", "Objects Layout", r"_add_\d+"),
        ]:
            options = [item for item in categories if item["category"] == category
                       and re.fullmatch(re.escape(base_name) + suffix, item["name"])
                       and (plus / "bddl_files/libero_10" / (item["name"] + ".bddl")).is_file()]
            if not options:
                raise RuntimeError(f"任务没有真实且已分类的{category}场景：{base_name}")
            # 先固定中等难度与文件名排序，不根据候选策略的评测表现挑场景。
            selected = min(options, key=lambda item: (abs(item["difficulty_level"] - 3), item["name"]))
            path = plus / "bddl_files/libero_10" / (selected["name"] + ".bddl")
            variants.append((split, name, "LIBERO-plus", path, selected))
        for split, name, repo, path, classification in variants:
            condition_id = f"t{task_index:02d}-{name}"
            conditions.append({"condition_id": condition_id, "task_id": "libero_10/" + base_name,
                               "instruction": instruction, "repo": repo, "split": split, "condition": name,
                               "bddl_path": str(path), "bddl_sha256": digest(path),
                               "classification": classification})
            for initial in range(20 if split == "target_perturbed" else 10):
                # 正式运行分布使用另一命名空间；D0未给出种子，保留未解决的初态核验门禁。
                init_seed = 100000 + task_index * 1000 + initial
                for policy_seed in policy_seeds:
                    cases.append({"eval_id": f"p0-eval-v1-{condition_id}-s{init_seed}-p{policy_seed}",
                                  "condition_id": condition_id, "split": split,
                                  "init_seed": init_seed, "policy_seed": policy_seed,
                                  "max_steps": 520, "settle_steps": 10, "is_eval": True,
                                  "initial_state_snapshot": None, "initial_state_sha256": None})
    if len({item["eval_id"] for item in cases}) != len(cases):
        raise RuntimeError("评测ID重复")
    counts = dict(Counter(item["split"] for item in cases))
    if counts != {"standard": 200, "target_perturbed": 800, "nontarget_perturbed": 200}:
        raise RuntimeError(f"草案数量不符合当前评测协议：{counts}")
    result = {"version": "p0-eval-v1-draft", "freeze_status": "draft_pending_state_capture_and_leakage_check",
              "conditions": conditions, "episodes": cases, "counts": counts,
              "source_cache_manifest": str(cache), "source_cache_manifest_sha256": digest(cache),
              "classification_sha256": digest(classifications), "policy_seeds": policy_seeds,
              "seed_namespaces": {"evaluation": "100000 + task_index*1000 + state_index",
                                  "runtime_rollout": "200000 + task_index*1000 + state_index"},
              "eval_overlap_check": {"passed": False,
                                     "reason": "D0初态种子缺失；需捕获实际仿真初态并核查源示范，不能仅凭新种子宣称无泄漏"},
              "runtime_environment_validated": False, "canonical_frozen_file_created": False,
              "precision": "CPU文件和元数据核验，无模型计算，无batch或训练步数",
              "selection_seed": None, "selection": "任务文字与实际BDDL对应；扰动按官方中等难度和文件名固定"}
    output = run_dir / "eval_draft.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    summary = {"catalog": str(output), "sha256": digest(output), "condition_count": len(conditions),
               "counts": counts, "frozen": False, "eval_overlap_check_passed": False}
    (run_dir / "catalog_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    raise SystemExit(run_stage("eval_catalog", catalog, {"scope": "实际场景清单草案，不运行评测或冻结正式评测集"}))
