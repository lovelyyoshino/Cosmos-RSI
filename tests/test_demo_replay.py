"""验证再生历史按数字排序、绑定原始初态并在新相机环境恢复相同快照。

@author pony
@date 2026-11-07
@version v1.0.0
@last_modified 2026-11-07
@changelog
    - v1.0.0 (2026-11-07): 防止demo_10排在demo_2前、静默漏回合或无效来源进入重放。
"""
from __future__ import annotations

import pathlib
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from p0_demo_replay import initialize_demo, preceding_demo_ids


class DemoReplayTests(unittest.TestCase):
    def test_history_uses_original_numeric_order_and_excludes_selected_demo(self):
        self.assertEqual(preceding_demo_ids(["demo_10", "demo_2", "demo_0", "demo_1"], "demo_2"),
                         ["demo_0", "demo_1"])
        self.assertEqual(preceding_demo_ids(["demo_0"], "demo_0"), [])
        for keys, selected in [(["demo_0"], "demo_1"), (["demo_a"], "demo_a"), (["0"], "0")]:
            with self.subTest(keys=keys), self.assertRaises(ValueError):
                preceding_demo_ids(keys, selected)

    def test_snapshot_is_restored_before_raw_state_and_settle(self):
        events = []
        env = types.SimpleNamespace(seed=lambda value: events.append(("seed", value)),
            reset=lambda: events.append(("reset",)),
            set_init_state=lambda value: events.append(("raw", int(value[0]))) or {"frame": 0},
            step=lambda action: (events.append(("step", float(action[-1]))) or {"frame": len(events)}, 0, False, {}))
        snapshot = {"fixtures": "snapshot", "integration": "snapshot"}
        with patch("p0_demo_replay.restore_fixture_state", side_effect=lambda *_: events.append(("fixture",))), \
             patch("p0_demo_replay.restore_integration_state", side_effect=lambda *_: events.append(("integration",))), \
             patch("p0_demo_replay.capture_fixture_state", return_value={"fixture_saved": True}), \
             patch("p0_demo_replay.capture_integration_state", return_value={"integration_saved": True}):
            _, captured, meta = initialize_demo(env, {"init_seed": 0, "settle_steps": 10, "source_demo_id": "demo_1"},
                                                {"raw_initial_state": np.asarray([7.0])}, snapshot)
        self.assertEqual(events[:5], [("seed", 0), ("reset",), ("fixture",), ("integration",), ("raw", 7)])
        self.assertEqual(events[5:], [("step", -1.0)] * 10)
        self.assertTrue(meta["reused_reset_snapshot"])
        self.assertEqual(captured, {"fixture_saved": True, "integration_saved": True})

    def test_history_replays_filtered_predecessor_before_initializing_selected_demo(self):
        if sys.version_info < (3, 11):
            self.skipTest("历史HDF5摘要使用项目的file_digest接口，本项在Thor Python3.12执行")
        try:
            import h5py
        except ImportError:
            self.skipTest("完整HDF5历史检查在Thor sim环境执行，policy环境未安装h5py")
        from p0_policy_runtime import sha256

        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            (root / "data").mkdir()
            source = root / "data/raw.hdf5"
            with h5py.File(source, "w") as raw:
                for index in [0, 1, 2, 10]:
                    demo = raw.create_group(f"data/demo_{index}")
                    demo.create_dataset("states", data=[[index]])
                    actions = np.zeros((2, 7), dtype=np.float32)
                    actions[:, -1] = -1
                    actions[1, 0] = index + 1
                    demo.create_dataset("actions", data=actions)
            events = []
            env = types.SimpleNamespace(seed=lambda *_: None, reset=lambda: None,
                set_init_state=lambda value: events.append(("raw", int(value[0]))) or {},
                step=lambda action: (events.append(("action", float(action[0]))) or {}, 0, False, {}))
            row = {"init_seed": 0, "settle_steps": 10, "source_demo_id": "demo_2",
                   "source_file": str(source), "source_file_sha256": sha256(source)}
            with patch("p0_demo_replay.ROOT", root), \
                 patch("p0_demo_replay.capture_fixture_state", return_value={}), \
                 patch("p0_demo_replay.capture_integration_state", return_value={}):
                _, _, meta = initialize_demo(env, row, {"raw_initial_state": np.asarray([2])})
                self.assertEqual([item for item in events if item[0] == "raw"], [("raw", 0), ("raw", 1), ("raw", 2)])
                self.assertEqual([value for kind, value in events if kind == "action" and value], [1.0, 2.0])
                self.assertEqual(meta["history_demo_ids"], ["demo_0", "demo_1"])
                with self.assertRaisesRegex(RuntimeError, "初态"):
                    initialize_demo(env, row, {"raw_initial_state": np.asarray([99])})
                with self.assertRaisesRegex(RuntimeError, "改变"):
                    initialize_demo(env, {**row, "source_file_sha256": "wrong"}, {"raw_initial_state": np.asarray([2])})


if __name__ == "__main__":
    unittest.main()
