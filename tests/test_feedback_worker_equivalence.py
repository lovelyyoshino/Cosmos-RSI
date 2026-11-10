"""并行反馈不能用相同成败掩盖动作、像素或噪声日程的变化。

@author pony
@date 2026-11-09
@version v1.0.0
@last_modified 2026-11-09
@changelog
    - v1.0.0 (2026-11-09): 覆盖相同结果不同轨迹、非有限值及计时差异。
"""
import copy
import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from p0_verify_feedback_workers import compare_case


class FeedbackEquivalenceTests(unittest.TestCase):
    def fixture(self):
        fields = ["condition_id", "init_seed", "policy_seed", "policy_ckpt_sha256", "instruction",
                  "conditions", "initial_scene_sha256", "is_eval", "outcome", "precision", "batch",
                  "action_samples", "max_steps", "control_hz", "action_horizon", "policy_noise_schedule",
                  "terminal_reason", "dispersion", "isolation"]
        row = {key: key for key in fields}
        row["metrics"] = {key: 1 for key in ["steps", "sim_duration_s", "policy_calls", "action_dispersion_p90",
                                             "regrasp_count", "gripper_reclose_command_count"]}
        arrays = {key: np.zeros((2, 7), dtype=np.float32) for key in
                  ["action", "state", "agentview", "wrist", "reward", "request_seed", "final_sim_state"]}
        return row, arrays

    def test_timings_can_change_without_changing_the_trajectory(self):
        row, arrays = self.fixture()
        other = copy.deepcopy(row)
        other["metrics"]["duration_s"] = 999
        self.assertTrue(compare_case(row, other, arrays, arrays)["passed"])

    def test_action_image_state_and_noise_changes_fail_even_with_identical_outcomes(self):
        row, arrays = self.fixture()
        for name in arrays:
            with self.subTest(name=name):
                other = {key: value.copy() for key, value in arrays.items()}
                other[name][0, 0] = 1
                result = compare_case(row, row, arrays, other)
                self.assertFalse(result["passed"])
                self.assertIn(name, result["differing_fields"])

    def test_nan_cannot_pass_by_being_present_in_both_runs(self):
        row, arrays = self.fixture()
        arrays["action"][0, 0] = np.nan
        self.assertFalse(compare_case(row, row, arrays, arrays)["passed"])

    def test_seed_condition_and_difficulty_signal_changes_fail(self):
        row, arrays = self.fixture()
        for name in ["policy_seed", "condition_id", "dispersion"]:
            other = copy.deepcopy(row)
            other[name] = "different"
            self.assertFalse(compare_case(row, other, arrays, arrays)["passed"])

    def test_missing_array_cannot_silently_reduce_the_comparison(self):
        row, arrays = self.fixture()
        other = dict(arrays)
        del other["wrist"]
        with self.assertRaises(ValueError):
            compare_case(row, row, arrays, other)


if __name__ == "__main__":
    unittest.main()
