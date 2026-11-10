"""实际混合样本读取的窗口边界、双视角标签对齐和评测分区拒绝测试。

@author pony
@date 2026-11-08
@version v1.0.0
@last_modified 2026-11-08
@changelog
    - v1.0.0 (2026-11-08): 覆盖末端合法动作块、越界、错源、相对帧及腕部保持。
"""
from __future__ import annotations

import copy
import pathlib
import sys
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from p0_augmented_smoke import example_at


class AugmentedSmokeTests(unittest.TestCase):
    def fixture(self):
        images = np.zeros((20, 2, 4, 4, 3), np.uint8)
        images[:, 0] = 10; images[:, 1] = 60
        actions = np.arange(20 * 7, dtype=np.float32).reshape(20, 7)
        states = np.arange(20 * 8, dtype=np.float32).reshape(20, 8)
        cache = SimpleNamespace(images=images, actions=actions, states=states,
            manifest={"episodes": [{"episode_id": "book", "split": "demo_train", "length": 14,
                                     "offset": 1, "instruction": "put book in caddy"}],
                      "normalizer": {"action": {"min": [0] * 7, "max": [1000] * 7},
                                     "state": {"min": [0] * 8, "max": [1000] * 8}}})
        row = {"source_episode_id": "book", "source_frame": 5, "horizon": 8, "is_eval": False,
               "kind": "generated", "sample_id": "g", "window_frame": 3,
               "window": {"start_frame": 2, "end_frame": 13}}
        augmented = {"g": {"source_episode_id": "book", "window": row["window"],
                           "cosmos": np.full((11, 4, 4, 3), 199, np.uint8),
                           "analytic": np.full((11, 4, 4, 3), 88, np.uint8)}}
        return cache, row, augmented

    def test_last_legal_block_changes_only_current_agentview_and_preserves_source_labels(self):
        cache, row, extra = self.fixture()
        example = example_at(cache, row, extra, "cosmos")
        np.testing.assert_array_equal(np.asarray(example["image"][0]), 199)
        np.testing.assert_array_equal(np.asarray(example["image"][1]), 60)
        np.testing.assert_allclose(example["action"], 2 * cache.actions[6:14] / 1000 - 1)
        np.testing.assert_allclose(example["state"], 2 * cache.states[6:7] / 1000 - 1)
        self.assertEqual(example["lang"], "put book in caddy")
        row["kind"] = "duplicate_fill"
        np.testing.assert_array_equal(np.asarray(example_at(cache, row, {}, "cosmos")["image"][0]), 10)

    def test_wrong_source_partial_window_and_eval_rows_are_rejected(self):
        cache, base, extra = self.fixture()
        for field, replacement in [("source_frame", 6), ("source_frame", 1), ("source_episode_id", "eval"),
                                   ("is_eval", True), ("window_frame", 4)]:
            row = {**copy.deepcopy(base), field: replacement}
            with self.subTest(field=field, replacement=replacement), self.assertRaises(ValueError):
                example_at(cache, row, extra, "cosmos")
        cache.manifest["episodes"][0]["split"] = "demo_validation"
        with self.assertRaises(ValueError):
            example_at(cache, base, extra, "cosmos")

    def test_analytic_ablation_reuses_identical_source_frame_and_wrist(self):
        cache, row, extra = self.fixture()
        a = example_at(cache, row, extra, "analytic")
        b = example_at(cache, row, extra, "cosmos")
        np.testing.assert_array_equal(np.asarray(a["image"][0]), 88)
        np.testing.assert_array_equal(np.asarray(a["image"][1]), np.asarray(b["image"][1]))
        np.testing.assert_array_equal(a["action"], b["action"])
        np.testing.assert_array_equal(a["state"], b["state"])


if __name__ == "__main__":
    unittest.main()
