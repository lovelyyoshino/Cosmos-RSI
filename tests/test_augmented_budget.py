"""预算训练必须消费真实原生窗口，并保持配对计划的绝对源帧。

@author pony
@date 2026-11-08
@version v1.0.0
@last_modified 2026-11-08
@changelog
    - v1.0.0 (2026-11-08): 回归错窗口、源帧错位、评测标志及非八帧动作拒绝。
"""
from __future__ import annotations

import copy
import pathlib
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from p0_augmented_budget import budget_example


class BudgetExampleTests(unittest.TestCase):
    def fixture(self):
        window = {"start_frame": 120, "end_frame": 213}
        native = SimpleNamespace(provenance={"episodes": [{"source_episode_id": "book", "source_window": window}]},
                                 example_at=Mock(return_value={"actual_native": True}))
        row = {"kind": "generated", "source_episode_id": "book", "window": window,
               "source_frame": 205, "window_frame": 85, "is_eval": False, "horizon": 8}
        return native, row

    def test_last_valid_source_frame_reaches_real_partial_episode_reader(self):
        native, row = self.fixture()
        example = budget_example(None, row, native, {"book": 0})
        self.assertEqual(example, {"actual_native": True})
        native.example_at.assert_called_once_with(0, 85)

    def test_misaligned_plan_is_rejected_before_native_read(self):
        for change in [{"is_eval": True}, {"horizon": 7}, {"source_frame": 206},
                       {"window": {"start_frame": 119, "end_frame": 212}}]:
            with self.subTest(change=change):
                native, row = self.fixture()
                with self.assertRaises(ValueError):
                    budget_example(None, {**copy.deepcopy(row), **change}, native, {"book": 0})
                native.example_at.assert_not_called()


if __name__ == "__main__":
    unittest.main()
