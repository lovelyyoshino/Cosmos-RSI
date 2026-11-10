"""再生动作时序和跨场景初态比较的回归测试。

@author pony
@date 2026-11-06
@version v1.3.0
@last_modified 2026-11-06
@changelog
    - v1.3.0 (2026-11-06): 用真实MuJoCo单关节模型验证warmstart、ctrl及外力的积分状态往返。
    - v1.2.0 (2026-11-06): 覆盖不在MjSimState内的fixture位姿快照恢复及异常快照拒绝。
    - v1.1.0 (2026-11-06): 复现并防止namespace package在复用worker时因__file__为空而失败。
    - v1.0.0 (2026-11-06): 覆盖夹爪改变保留、布局添加物体及物体位姿变化。
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
from p0_sim_data import (capture_fixture_state, capture_integration_state, configure_libero, kept_action_indices,
                         restore_fixture_state, restore_integration_state, scene_digest, scene_distance)


class SimulationDataTests(unittest.TestCase):
    def test_integration_snapshot_restores_solver_and_control_inputs(self):
        try:
            import mujoco
        except ImportError:
            self.skipTest("仅Thor仿真环境提供MuJoCo；须在那里执行本项")
        model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="arm">'
            '<joint name="joint" type="hinge"/><geom type="sphere" size="0.1" mass="1"/>'
            '</body></worldbody><actuator><motor joint="joint"/></actuator></mujoco>')
        data = mujoco.MjData(model)
        env = types.SimpleNamespace(sim=types.SimpleNamespace(model=types.SimpleNamespace(_model=model),
                                                              data=types.SimpleNamespace(_data=data)))
        data.qacc_warmstart[:] = 3.5
        data.ctrl[:] = 0.37
        data.qfrc_applied[:] = -0.13
        data.xfrc_applied[1] = np.arange(6)
        snapshot = capture_integration_state(env)
        mujoco.mj_resetData(model, data)
        restore_integration_state(env, snapshot)
        np.testing.assert_array_equal(capture_integration_state(env)["integration_state"], snapshot["integration_state"])
        np.testing.assert_array_equal(data.qacc_warmstart, [3.5])
        np.testing.assert_array_equal(data.ctrl, [0.37])
        np.testing.assert_array_equal(data.qfrc_applied, [-0.13])
        np.testing.assert_array_equal(data.xfrc_applied[1], np.arange(6))
        snapshot["integration_state"][0] = np.nan
        with self.assertRaisesRegex(RuntimeError, "积分状态"):
            restore_integration_state(env, snapshot)

    def test_fixed_fixture_pose_survives_npz_and_changed_reset(self):
        model = types.SimpleNamespace(body_pos=np.zeros((3, 3)), body_quat=np.tile([1., 0, 0, 0], (3, 1)),
                                      body_name2id={"cabinet_main": 1}.__getitem__)
        env = types.SimpleNamespace(env=types.SimpleNamespace(fixtures_dict={
            "cabinet": types.SimpleNamespace(root_body="cabinet_main")}), sim=types.SimpleNamespace(model=model))
        model.body_pos[1] = [0.2, 0.3, 0.8]
        fixture = capture_fixture_state(env)
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "snapshot.npz"
            np.savez_compressed(path, **fixture)
            model.body_pos[1] = [-0.2, 0.1, 0.9]
            model.body_quat[1] = [0, 0, 0, 1]
            model.body_pos[2] = [9, 8, 7]
            with np.load(path, allow_pickle=False) as saved:
                restore_fixture_state(env, saved)
        np.testing.assert_array_equal(model.body_pos[1], [0.2, 0.3, 0.8])
        np.testing.assert_array_equal(model.body_quat[1], [1, 0, 0, 0])
        np.testing.assert_array_equal(model.body_pos[2], [9, 8, 7])

    def test_invalid_fixture_snapshot_is_rejected_before_mutation(self):
        model = types.SimpleNamespace(body_pos=np.zeros((2, 3)), body_quat=np.tile([1., 0, 0, 0], (2, 1)),
                                      body_name2id={"cabinet_main": 1}.__getitem__)
        env = types.SimpleNamespace(env=types.SimpleNamespace(fixtures_dict={
            "cabinet": types.SimpleNamespace(root_body="cabinet_main")}), sim=types.SimpleNamespace(model=model))
        fixture = capture_fixture_state(env)
        fixture["fixture_body_pos"][0] = [1, 2, 3]
        fixture["fixture_body_quat"][0, 0] = np.nan
        with self.assertRaisesRegex(RuntimeError, "fixture"):
            restore_fixture_state(env, fixture)
        np.testing.assert_array_equal(model.body_pos, np.zeros((2, 3)))

    def test_namespace_package_can_be_reused_only_for_same_implementation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            package = types.ModuleType("libero")
            package.__file__ = None
            package.__path__ = [str(root / "third_party/LIBERO/libero")]
            with patch("p0_sim_data.ROOT", root), patch.dict(sys.modules, {"libero": package}), \
                 patch("sys.path", list(sys.path)), patch.dict("os.environ"):
                configure_libero("LIBERO", root / "config1")
                configure_libero("LIBERO", root / "config2")
                with self.assertRaisesRegex(RuntimeError, "混用"):
                    configure_libero("LIBERO-plus", root / "config3")

    def test_noop_filter_preserves_gripper_change_and_action_order(self):
        actions = np.zeros((6, 7), dtype=np.float32)
        actions[:, -1] = [-1, -1, -1, 1, 1, -1]
        actions[1, 0] = 0.5
        np.testing.assert_array_equal(kept_action_indices(actions), [1, 3, 5])

    def test_added_layout_object_does_not_hide_shared_initial_state(self):
        base = {"book_joint": [0.2, 0.1, 0.8, 1, 0, 0, 0]}
        layout = {**base, "distractor_joint": [0.4, 0.1, 0.8, 1, 0, 0, 0]}
        self.assertEqual(scene_distance(base, layout), 0)
        self.assertNotEqual(scene_digest(base), scene_digest(layout))

    def test_position_change_is_measurable(self):
        first = {"book_joint": [0.2, 0.1, 0.8, 1, 0, 0, 0]}
        second = {"book_joint": [0.21, 0.1, 0.8, 1, 0, 0, 0]}
        self.assertAlmostEqual(scene_distance(first, second), 0.01)
        with self.assertRaises(ValueError):
            scene_distance(first, {"other": [0]})

    def test_nonfinite_action_is_rejected(self):
        with self.assertRaises(ValueError):
            kept_action_indices([[0, 0, 0, 0, 0, 0, np.nan]])


if __name__ == "__main__":
    unittest.main()
