"""验证真实示范加载的归一化和动作块边界约束。

@author pony
@date 2026-11-06
@version v1.2.0
@last_modified 2026-11-06
@changelog
    - v1.2.0 (2026-11-06): 已签名但缺少fixture和双相机证据的旧冻结格式仍须拒绝。
    - v1.1.0 (2026-11-06): 正式训练拒绝草案及内容被修改的冻结评测集。
    - v1.0.0 (2026-11-06): 常数维度逆归一化、跨回合边界和训练验证隔离回归。
"""
from __future__ import annotations

import json
import hashlib
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from p0_policy_runtime import DemoCache, normalize, verify_eval_isolation


class PolicyRuntimeTests(unittest.TestCase):
    def test_formal_training_rejects_probe_and_tampered_eval(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            cache = root / "data/cache"
            evaluation = root / "eval/frozen_eval_set.json"
            cache.mkdir(parents=True)
            evaluation.parent.mkdir()
            (cache / "manifest.json").write_text('{}')
            evaluation.write_text(json.dumps({"freeze_status": "probe_only", "all_initial_states_replayed": True}))
            with patch("p0_policy_runtime.ROOT", root), patch("p0_policy_runtime.sha256", side_effect=lambda path:
                    hashlib.sha256(path.read_bytes()).hexdigest()):
                with self.assertRaisesRegex(RuntimeError, "尚未冻结"):
                    verify_eval_isolation(cache, evaluation)
                evaluation.write_text(json.dumps({"freeze_status": "frozen", "all_initial_states_replayed": True,
                                                  "content_sha256": "incorrect"}))
                with self.assertRaisesRegex(RuntimeError, "摘要不一致"):
                    verify_eval_isolation(cache, evaluation)
                legacy = {"freeze_status": "frozen", "all_initial_states_replayed": True, "episodes": []}
                content = json.dumps(legacy, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
                legacy["content_sha256"] = hashlib.sha256(content).hexdigest()
                evaluation.write_text(json.dumps(legacy))
                evaluation.with_suffix(".sha256").write_text(hashlib.sha256(evaluation.read_bytes()).hexdigest())
                with self.assertRaisesRegex(RuntimeError, "fixture"):
                    verify_eval_isolation(cache, evaluation)

    def test_constant_dimension_round_trip(self):
        stats = {"min": [-2, 0.5], "max": [2, 0.5]}
        values = np.asarray([[1, 0.5]], dtype=np.float32)
        normalized = normalize(values, stats)
        np.testing.assert_allclose(normalized, [[0.5, 0]])
        np.testing.assert_allclose(normalize(normalized, stats, inverse=True), values)

    def test_nonfinite_values_fail(self):
        with self.assertRaisesRegex(RuntimeError, "非法"):
            normalize([float("nan")], {"min": [0], "max": [1]})

    def test_action_windows_stay_in_one_episode(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            path = root / "data/cache"
            path.mkdir(parents=True)
            np.save(path / "images.npy", np.zeros((18, 2, 256, 256, 3), dtype=np.uint8))
            np.save(path / "states.npy", np.zeros((18, 8), dtype=np.float32))
            actions = np.concatenate([np.full((9, 7), -1, dtype=np.float32), np.full((9, 7), 1, dtype=np.float32)])
            np.save(path / "actions.npy", actions)
            manifest = {"cache_id": "fixture", "frames": 18, "horizon": 8,
                        "normalizer": {"state": {"min": [-1] * 8, "max": [1] * 8},
                                       "action": {"min": [-1] * 7, "max": [1] * 7}},
                        "episodes": [{"episode_id": "train", "offset": 0, "length": 9, "instruction": "task", "split": "demo_train"},
                                     {"episode_id": "validation", "offset": 9, "length": 9, "instruction": "task", "split": "demo_validation"}]}
            (path / "manifest.json").write_text(json.dumps(manifest))
            (path / "COMPLETED").write_text("fixture\n")
            with patch("p0_policy_runtime.ROOT", root):
                cache = DemoCache(path)
                for split, expected in [("demo_train", -1), ("demo_validation", 1)]:
                    examples, locations = cache.batch(split, 8, np.random.default_rng(42))
                    self.assertTrue(all(example["action"].shape == (8, 7) for example in examples))
                    self.assertTrue(all(np.all(example["action"] == expected) for example in examples))
                    self.assertTrue(all(0 <= location["frame"] <= 1 for location in locations))
                manifest["episodes"][1]["episode_id"] = "train"
                (path / "manifest.json").write_text(json.dumps(manifest))
                with self.assertRaisesRegex(RuntimeError, "交集"):
                    DemoCache(path)


if __name__ == "__main__":
    unittest.main()
