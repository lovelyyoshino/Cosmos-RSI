"""拒绝成功率相同但动作、输入、初态或物理轨迹不同的并行验证。

@author pony
@date 2026-11-06
@version v1.0.0
@last_modified 2026-11-06
@changelog
    - v1.0.0 (2026-11-06): 覆盖轨迹数值、非有限值、噪声与图像摘要的等价性门禁。
"""
from __future__ import annotations

import copy
import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from p0_verify_eval_workers import compare_trajectories


class EvaluationEquivalenceTests(unittest.TestCase):
    def setUp(self):
        self.row = {key: "same" for key in ["eval_id", "condition_id", "init_seed", "policy_seed", "policy_ckpt_sha256",
                                          "eval_set_sha256", "initial_state_sha256", "precision", "batch", "control_hz",
                                          "action_horizon", "policy_noise_schedule", "outcome", "is_eval"]}
        self.arrays = {key: np.arange(14, dtype=np.float64).reshape(2, 7)
                       for key in ["action", "state", "reward", "request_seed", "physical_state", "integration_state",
                                   "final_sim_state", "final_integration_state"]}
        self.arrays["request_observation_sha256"] = np.asarray(["observation_a", "observation_b"])

    def test_identical_evaluations_pass(self):
        result = compare_trajectories(self.row, copy.deepcopy(self.row), self.arrays, copy.deepcopy(self.arrays))
        self.assertTrue(result["passed"])
        self.assertTrue(all(row["max_abs_diff"] == 0 for row in result["arrays"].values()))

    def test_same_outcome_cannot_hide_a_different_action_or_physical_path(self):
        for field in ["action", "physical_state", "integration_state"]:
            other = copy.deepcopy(self.arrays)
            other[field][0, 0] += 1e-10
            result = compare_trajectories(self.row, self.row, self.arrays, other)
            self.assertFalse(result["passed"])
            self.assertIn(field, result["differing_fields"])

    def test_same_nonfinite_arrays_do_not_pass(self):
        values = copy.deepcopy(self.arrays)
        values["action"][0, 0] = np.inf
        result = compare_trajectories(self.row, self.row, values, copy.deepcopy(values))
        self.assertFalse(result["passed"])

    def test_changed_noise_seed_or_observation_fails(self):
        row = {**self.row, "policy_seed": 999}
        result = compare_trajectories(self.row, row, self.arrays, self.arrays)
        self.assertIn("policy_seed", result["differing_fields"])
        arrays = {**self.arrays, "request_observation_sha256": np.asarray(["different"])}
        result = compare_trajectories(self.row, self.row, self.arrays, arrays)
        self.assertIn("request_observation_sha256", result["differing_fields"])


if __name__ == "__main__":
    unittest.main()
