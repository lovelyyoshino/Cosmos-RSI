"""动作诊断图必须完整配对并重算原始误差，不把相关块当成独立回合。

@author pony
@date 2026-11-09
@version v1.0.0
@last_modified 2026-11-09
@changelog
    - v1.0.0 (2026-11-09): 覆盖缺块、真值漂移、虚报误差及按示范等权聚合。
"""
from __future__ import annotations

import copy
import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from p1_policy_regression_probe import ROLES, action_metrics
from p1_regression_materials import summarize_predictions


def fixture():
    locations = [{"episode_id": episode, "instruction": "synthetic-task", "frame": frame,
                  "horizon": 8, "split": "demo_validation", "position": "synthetic"}
                 for episode, frames in [("synthetic-a", [0, 4]), ("synthetic-b", [0])]
                 for frame in frames]
    rows = []
    for role in ROLES:
        for location in locations:
            for seed in [44, 45]:
                prediction = np.ones((8, 7)) * (1 if location["episode_id"] == "synthetic-a" else 2)
                target = np.zeros((8, 7))
                rows.append({"role": role, **location, "policy_seed": seed,
                    **action_metrics(prediction, target, prediction),
                    "normalized_actions": prediction.tolist(), "target_actions": target.tolist()})
    return rows, locations


class RegressionMaterialsTests(unittest.TestCase):
    def test_aggregation_weights_demonstrations_equally(self):
        rows, locations = fixture()
        summary = summarize_predictions(rows, locations, [44, 45])
        self.assertEqual(len(summary), 4)
        for row in summary:
            self.assertEqual(row["source_episodes"], 2)
            self.assertEqual(row["action_mse"], 2.5)
            self.assertEqual(row["drift_vs_incumbent_rms"], 0.)

    def test_missing_duplicate_unexpected_seed_and_training_split_reject(self):
        for mutation in ["missing", "duplicate", "seed", "split", "location"]:
            rows, locations = fixture()
            if mutation == "missing": rows.pop()
            elif mutation == "duplicate": rows.append(copy.deepcopy(rows[0]))
            elif mutation == "seed": rows[0]["policy_seed"] = 46
            elif mutation == "split": rows[0]["split"] = "demo_train"
            else: locations.append(copy.deepcopy(locations[0]))
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                summarize_predictions(rows, locations, [44, 45])

    def test_raw_actions_must_explain_recorded_error_and_drift(self):
        for mutation in ["action_mse", "drift_vs_incumbent_rms", "nonfinite", "shape", "missing_role"]:
            rows, locations = fixture()
            if mutation in {"action_mse", "drift_vs_incumbent_rms"}: rows[-1][mutation] += 1
            elif mutation == "nonfinite": rows[-1]["normalized_actions"][0][0] = float("nan")
            elif mutation == "shape": rows[-1]["normalized_actions"].pop()
            else: rows = [row for row in rows if row["role"] != "control"]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                summarize_predictions(rows, locations, [44, 45])

    def test_paired_truth_and_instruction_changes_are_rejected(self):
        for mutation in ["target", "instruction"]:
            rows, locations = fixture()
            if mutation == "target": rows[-1]["target_actions"][0][0] = 1.
            else: rows[-1]["instruction"] = "other-task"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                summarize_predictions(rows, locations, [44, 45])

    def test_duplicate_seed_cannot_inflate_sample_count(self):
        rows, locations = fixture()
        with self.assertRaises(ValueError):
            summarize_predictions(rows, locations, [44, 44])


if __name__ == "__main__":
    unittest.main()
