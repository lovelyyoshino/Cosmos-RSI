"""判官校准计数与样本不足边界；不调用GPU或模型。

@author pony
@date 2026-11-08
@version v1.0.0
@last_modified 2026-11-08
@changelog
    - v1.0.0 (2026-11-08): 覆盖类别缺失、虚假满分、混淆矩阵及重复回合。
"""
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from p0_feedback_judge import calibration_metrics, parse_verdict, reasoning_messages


class FeedbackJudgeTests(unittest.TestCase):
    config = {"min_calibration_episodes": 20, "min_success_episodes": 5, "min_failure_episodes": 5, "min_accuracy": .9}

    def rows(self, truths, predictions):
        return [{"episode_id": str(index), "environment_truth": truth,
                 "verdict": {"success": predicted, "confidence": .8}}
                for index, (truth, predicted) in enumerate(zip(truths, predictions))]

    def test_single_case_agreement_does_not_pass_calibration(self):
        metrics = calibration_metrics(self.rows([False], [False]), self.config)
        self.assertEqual(metrics["accuracy"], 1)
        self.assertIsNone(metrics["success_recall"])
        self.assertFalse(metrics["calibration_passed"])

    def test_one_class_only_cannot_pass_with_high_accuracy(self):
        metrics = calibration_metrics(self.rows([False] * 30, [False] * 30), self.config)
        self.assertFalse(metrics["sample_requirements_met"])

    def test_majority_accuracy_cannot_hide_zero_success_recall(self):
        metrics = calibration_metrics(self.rows([True] * 5 + [False] * 95, [False] * 100), self.config)
        self.assertEqual(metrics["accuracy"], .95)
        self.assertTrue(metrics["sample_requirements_met"])
        self.assertFalse(metrics["calibration_passed"])

    def test_invalid_response_is_an_abstention_and_counts_against_accuracy(self):
        rows = self.rows([True, False, False], [True, False, None])
        rows[-1]["valid"] = False
        metrics = calibration_metrics(rows, self.config)
        self.assertEqual(metrics["abstentions"], 1)
        self.assertEqual(metrics["accuracy"], 2 / 3)
        self.assertEqual(metrics["true_negative"], 1)
        self.assertEqual(metrics["false_positive"], 0)
        rows[-1]["valid"] = True
        with self.assertRaises(ValueError): calibration_metrics(rows, self.config)

    def test_confusion_counts_preserve_both_recalls(self):
        truths, predictions = [True] * 10 + [False] * 10, [True] * 9 + [False] * 10 + [True]
        metrics = calibration_metrics(self.rows(truths, predictions), self.config)
        self.assertEqual(metrics["accuracy"], .9)
        self.assertEqual(metrics["success_recall"], .9)
        self.assertEqual(metrics["failure_recall"], .9)
        self.assertTrue(metrics["calibration_passed"])
        self.assertLess(metrics["accuracy_wilson95"][0], .9)

    def test_duplicates_and_invalid_confidence_are_rejected(self):
        rows = self.rows([True, False], [True, False])
        rows[1]["episode_id"] = rows[0]["episode_id"]
        with self.assertRaises(ValueError): calibration_metrics(rows, self.config)
        rows[1]["episode_id"] = "unique"
        rows[1]["verdict"]["confidence"] = float("nan")
        with self.assertRaises(ValueError): calibration_metrics(rows, self.config)

    def test_only_final_answer_is_parsed(self):
        import json
        verdict = {"success": False, "confidence": .2, "evidence": "The object is still held.",
                   "failure_stage": "unknown"}
        output = '<think>{"success":true} is only a hypothesis.</think><answer>' + json.dumps(verdict) + '</answer>'
        self.assertEqual(parse_verdict(output), verdict)
        for bad in ['<think>' + json.dumps(verdict), '<answer>' + json.dumps(verdict),
                    json.dumps({**verdict, "confidence": True}), json.dumps({**verdict, "success": "false"})]:
            with self.assertRaises((ValueError, TypeError)): parse_verdict(bad)

    def test_reasoning_inputs_are_media_first_without_environment_truth(self):
        inputs = [{"path": f"frame{index}.png", "phase": "terminal", "view": "agentview", "timestamp": index}
                  for index in range(6)]
        messages = reasoning_messages(inputs, "put the book in the caddy")
        content = messages[-1]["content"]
        self.assertEqual([item["type"] for item in content], ["image"] * 6 + ["text"])
        self.assertIn("put the book in the caddy", content[-1]["text"])
        self.assertNotIn("environment_truth", str(messages))


if __name__ == "__main__":
    unittest.main()
