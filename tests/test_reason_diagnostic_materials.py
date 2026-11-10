"""Reason素材拒绝挑选子集、改写真值和错误关键帧，避免把诊断变成虚假校准。

@author pony
@date 2026-11-09
@version v1.0.0
@last_modified 2026-11-09
@changelog
    - v1.0.0 (2026-11-09): 使用显式合成夹具检查完整回合计数及输入绑定边界。
"""
import copy
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from p0_feedback_judge import calibration_metrics
from p1_reason_diagnostic_materials import audit_rows


class ReasonDiagnosticTests(unittest.TestCase):
    def fixture(self):
        config = {"min_calibration_episodes": 20, "min_success_episodes": 5,
                  "min_failure_episodes": 5, "min_accuracy": .9}
        episodes = [{"episode_id": str(i), "outcome": {"success": i < 10, "source": "env"},
                     "policy_ckpt_sha256": "synthetic-parent"} for i in range(20)]
        rows = [{"episode_id": row["episode_id"], "environment_truth": row["outcome"]["success"],
                 "policy_ckpt_sha256": "synthetic-parent", "is_eval": False, "truth_hidden_from_model": True,
                 "verdict": {"success": True, "confidence": .97},
                 "inputs": [{"phase": phase, "view": view} for phase in ["initial", "middle", "terminal"]
                            for view in ["agentview", "wrist"]]} for row in episodes]
        return episodes, rows, {"sample_count": 20, "config": config, "calibration": calibration_metrics(rows, config)}

    def test_all_success_bias_remains_failed_calibration_in_export(self):
        episodes, rows, report = self.fixture()
        metrics = audit_rows(episodes, rows, report)
        self.assertEqual(metrics["failure_recall"], 0)
        self.assertFalse(metrics["calibration_passed"])

    def test_incomplete_duplicate_relabelled_or_reordered_sources_reject(self):
        for mutation in ["subset", "duplicate", "truth", "parent", "hidden_truth", "frames", "summary"]:
            episodes, rows, report = self.fixture()
            if mutation == "subset": rows.pop()
            elif mutation == "duplicate": rows.append(copy.deepcopy(rows[0]))
            elif mutation == "truth": rows[0]["environment_truth"] = False
            elif mutation == "parent": rows[0]["policy_ckpt_sha256"] = "stale"
            elif mutation == "hidden_truth": rows[0]["truth_hidden_from_model"] = False
            elif mutation == "frames": rows[0]["inputs"].reverse()
            else: report["calibration"]["accuracy"] = 1
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                audit_rows(episodes, rows, report)


if __name__ == "__main__":
    unittest.main()
