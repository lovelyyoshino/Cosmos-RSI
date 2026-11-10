"""检验失败真值、归因样本边界及成功轨迹准入。

@author pony
@date 2026-11-09
@version v1.1.0
@last_modified 2026-11-09
@changelog
    - v1.1.0 (2026-11-09): 环境成功和统计路由不被未校准Reason否决；假阳性仍不能入训。
    - v1.0.0 (2026-11-09): 失败动作不可入成功池；单例不作视觉归因；不使用判官替换环境真值。
"""
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from p0_feedback_mine import summarize, wilson


class FeedbackMiningTests(unittest.TestCase):
    config = {"min_condition_episodes": 10, "hard_success_min_episodes": 10, "hard_success_percentile": 90}

    def samples(self, condition, successes, start=0):
        rows, judges = [], []
        for index, success in enumerate(successes, start):
            row = {"episode_id": str(index), "iter": 0, "git_sha": None, "is_eval": False,
                "task_id": "book", "instruction": "place book", "condition": condition, "conditions": {},
                "policy_ckpt_sha256": "same", "outcome": {"source": "env", "success": success},
                "init_seed": index, "metrics": {"steps": 20, "duration_s": 4., "action_dispersion_p90": .1},
                "lerobot_index": {"dataset": "runtime", "episode_index": index}}
            rows.append(row)
            judges.append({"episode_id": str(index), "policy_ckpt_sha256": "same", "environment_truth": success,
                "verdict": {"success": success, "confidence": .9, "evidence": "visible", "failure_stage": "unknown"},
                "inputs": [{"frame": frame} for frame in [0, 0, 10, 10, 20, 20]]})
        return rows, judges

    def test_small_failure_is_unknown_and_never_a_success_teacher(self):
        rows, judges = self.samples("lighting", [False])
        matrix, records, pool = summarize(rows, judges, self.config, True)
        self.assertEqual(matrix[0]["failure_rate"], 1)
        self.assertEqual(records[0]["category"], "unknown")
        self.assertEqual(records[0]["route"], "ignore")
        self.assertEqual(pool, [])

    def test_environment_success_owns_teacher_pool_despite_uncalibrated_false_negative(self):
        rows, judges = self.samples("standard", [True])
        _, records, pool = summarize(rows, judges, self.config, False)
        self.assertEqual(records, [])
        self.assertTrue(pool[0]["eligible_for_training"])
        self.assertFalse(pool[0]["judge_calibration_passed"])
        self.assertEqual(pool[0]["window"], {"start_frame": 0, "end_frame": 20})
        judges[0]["verdict"]["success"] = False
        pool = summarize(rows, judges, self.config, False)[2]
        self.assertTrue(pool[0]["eligible_for_training"])
        self.assertFalse(pool[0]["independent_judge_agrees"])

    def test_visual_routing_uses_environment_statistics_and_records_uncalibrated_judge(self):
        rows, judges = self.samples("standard", [True] * 10)
        added, opinions = self.samples("lighting", [False] * 10, 10)
        rows += added; judges += opinions
        _, records, _ = summarize(rows, judges, self.config, True)
        self.assertTrue(all(r["route"] == "cosmos_restyle" for r in records))
        _, records, _ = summarize(rows, judges, self.config, False)
        self.assertTrue(all(r["route"] == "cosmos_restyle" for r in records))
        self.assertTrue(all(r["judge_advisory_only"] for r in records))
        self.assertTrue(all(not r["judge_calibration_passed"] for r in records))

    def test_model_authority_or_non_environment_truth_cannot_use_simulation_route(self):
        rows, judges = self.samples("standard", [True])
        with self.assertRaises(ValueError):
            summarize(rows, judges, {**self.config, "decision_authority": "judge"}, False)
        rows[0]["outcome"]["source"] = "judge"
        with self.assertRaises(ValueError):
            summarize(rows, judges, self.config, False)

    def test_false_positive_judge_cannot_turn_failure_into_training_source(self):
        rows, judges = self.samples("standard", [False])
        judges[0]["verdict"]["success"] = True
        self.assertEqual(summarize(rows, judges, self.config, True)[2], [])
        judges[0]["environment_truth"] = True
        with self.assertRaises(ValueError): summarize(rows, judges, self.config, True)

    def test_wilson_extreme_counts_still_have_uncertainty(self):
        low, high = wilson(0, 10)
        self.assertAlmostEqual(low, 0)
        self.assertGreater(high, .2)
        with self.assertRaises(ValueError): wilson(11, 10)


if __name__ == "__main__":
    unittest.main()
