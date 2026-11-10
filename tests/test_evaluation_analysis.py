"""核验评测统计只接收完整冻结结果，并正确保留同初态聚类。

@author pony
@date 2026-11-07
@version v1.0.1
@last_modified 2026-11-09
@changelog
    - v1.0.1 (2026-11-09): 正式增量策略不能被绘图入口误标为初训基线。
    - v1.0.0 (2026-11-07): 覆盖成功计数、重复/缺失ID、用途边界、相关策略种子及Wilson区间。
"""
from __future__ import annotations

import copy
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from p0_analyze_evaluation import describe, policy_context, validate_episodes, wilson95


class EvaluationAnalysisTests(unittest.TestCase):
    def test_formal_update_is_not_labeled_as_initial_baseline(self):
        summary = {"training_step": 6000, "policy_ckpt_sha256": "synthetic-update"}
        metadata = {"step": 6000, "state_sha256": "synthetic-update", "formal_policy_update": True,
                    "iteration": 1, "role": "candidate", "recipe": {"batch": 8, "seed": 43}}
        context = policy_context(summary, metadata)
        self.assertEqual((context["role"], context["iteration"], context["training_seed"]), ("candidate", 1, 43))
        self.assertIn("候选", context["label"])
        metadata["state_sha256"] = "wrong-weight"
        with self.assertRaises(ValueError):
            policy_context(summary, metadata)

    def setUp(self):
        self.rows = []
        for initial, success in [(1, True), (2, False)]:
            for seed in [34001, 34002]:
                self.rows.append({"eval_id": f"fixture-{initial}-{seed}", "condition_id": "t00-standard",
                                  "split": "standard", "init_seed": initial, "policy_seed": seed,
                                  "task_id": "task", "initial_state_sha256": str(initial),
                                  "policy_ckpt_sha256": "checkpoint", "eval_set_sha256": "frozen",
                                  "is_eval": True, "eligible_for_training": False,
                                  "outcome": {"success": success, "source": "env"},
                                  "metrics": {"steps": 100, "duration_s": 1}})
        self.frozen = {"episodes": copy.deepcopy(self.rows)}
        self.summary = {"full_eval_complete": True, "policy_ckpt_sha256": "checkpoint", "eval_set_sha256": "frozen",
                        "splits": {"standard": {"n": 4, "success": 2, "p": 0.5}}}

    def test_complete_counts_and_clustered_seeds(self):
        validate_episodes(self.rows, self.frozen, self.summary)
        result = describe(self.rows, replicates=200)
        self.assertEqual((result["n"], result["success"], result["initial_state_clusters"]), (4, 2, 2))
        self.assertEqual(result["p"], 0.5)
        self.assertEqual(result, describe(self.rows, replicates=200))

    def test_missing_duplicate_or_changed_frozen_identity_fails(self):
        changed = copy.deepcopy(self.rows)
        changed[0]["init_seed"] = 99
        for rows in [self.rows[:-1], [*self.rows, self.rows[0]], changed]:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                validate_episodes(rows, self.frozen, self.summary)

    def test_no_training_eligibility_or_misreported_counts(self):
        changed = copy.deepcopy(self.rows)
        changed[0]["eligible_for_training"] = True
        with self.assertRaises(ValueError):
            validate_episodes(changed, self.frozen, self.summary)
        invalid = copy.deepcopy(self.summary)
        invalid["splits"]["standard"]["success"] = 3
        with self.assertRaises(ValueError):
            validate_episodes(self.rows, self.frozen, invalid)

    def test_wilson_bounds_do_not_hide_zero_success_uncertainty(self):
        lo, hi = wilson95(0, 20)
        self.assertAlmostEqual(lo, 0)
        self.assertGreater(hi, 0.15)
        lo, hi = wilson95(10, 20)
        self.assertAlmostEqual(lo, 1 - hi)
        for arguments in [(0, 0), (21, 20), (-1, 20)]:
            with self.assertRaises(ValueError):
                wilson95(*arguments)


if __name__ == "__main__":
    unittest.main()
