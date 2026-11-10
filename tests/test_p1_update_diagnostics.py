"""用CPU合成记录检查退化统计和实际训练覆盖，避免错读成败或把重复槽位计作新数据。

@author pony
@date 2026-11-09
@version v1.0.0
@last_modified 2026-11-09
@changelog
    - v1.0.0 (2026-11-09): 覆盖配对丢失成功、冻结身份不符、真实采样窗口及重复计数。
"""
import copy
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from p0_paired_sampling import build_paired_plan, paired_plan_sha256
from p1_update_diagnostics import exposure_counts, paired_outcomes


def episode(identity, success):
    return {"eval_id": identity, "condition_id": "t00-standard", "split": "standard",
            "init_seed": 12, "policy_seed": 13, "initial_state_sha256": "synthetic-state",
            "eval_set_sha256": "synthetic-evaluation", "is_eval": True, "eligible_for_training": False,
            "outcome": {"source": "env", "success": success}}


class DiagnosticTests(unittest.TestCase):
    def test_nested_environment_outcomes_and_gained_lost_pairs(self):
        left = [episode("a", True), episode("b", False), episode("c", True)]
        right = [episode("c", False), episode("b", True), episode("a", False)]
        summary, pairs = paired_outcomes(left, right)
        self.assertEqual(len(pairs), 3)
        for row in summary:
            self.assertEqual((row["incumbent_success"], row["candidate_success"]), (2, 1))
            self.assertEqual((row["lost_success"], row["gained_success"]), (2, 1))
            self.assertAlmostEqual(row["delta_percentage_points"], -100 / 3)

    def test_missing_duplicate_or_mismatched_frozen_identity_rejects(self):
        left = [episode("a", True)]
        for right in [[], [episode("a", True)] * 2, [episode("other", True)],
                      [{**episode("a", True), "init_seed": 99}],
                      [{**episode("a", True), "is_eval": False}],
                      [{**episode("a", True), "outcome": {"success": 1, "source": "env"}}]]:
            with self.subTest(right=right), self.assertRaises(ValueError):
                paired_outcomes(left, right)

    def plan(self):
        demo = {"kind": "demo", "sample_id": "d", "source_episode_id": "source",
                "source_length": 8, "source_split": "demo_train", "is_eval": False,
                "window": {"start_frame": 0, "end_frame": 8}}
        return build_paired_plan([demo], [], steps=3, batch=2, horizon=8, seed=42,
                                 quotas={"generated": 0, "self_distill": 0})

    def test_repeated_training_slots_are_one_unique_window(self):
        for role in ["candidate", "control"]:
            rows = exposure_counts(self.plan(), role, {"source": "synthetic task"})
            self.assertEqual(rows[0]["sample_slots"], 6)
            self.assertEqual(rows[0]["unique_windows"], 1)
            self.assertEqual(rows[0]["source_episodes"], 1)

    def test_even_rehashed_wrong_window_source_or_order_rejects(self):
        for key, value in [("source_frame", 1), ("source_episode_id", "wrong"), ("step", 4), ("is_eval", True)]:
            plan = copy.deepcopy(self.plan())
            plan["candidate"][0][key] = value
            plan["sha256"] = paired_plan_sha256(plan)
            with self.subTest(key=key), self.assertRaises(ValueError):
                exposure_counts(plan, "candidate", {"source": "synthetic task"})


if __name__ == "__main__":
    unittest.main()
