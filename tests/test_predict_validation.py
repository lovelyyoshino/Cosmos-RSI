"""Predict多状态诊断须完整绑定源、窗口和原始图片，不用像素改善冒称任务收益。

@author pony
@date 2026-11-09
@version v1.1.0
@last_modified 2026-11-09
@changelog
    - v1.1.0 (2026-11-09): 实际压缩NPZ与双分支模拟器回归，检查一次解码、逐帧等价及封存终态。
    - v1.0.0 (2026-11-09): 真实终帧窗口、确定性选择、像素重算、分层权重及等待身份回归。
"""
from __future__ import annotations

import copy
import hashlib
import json
import pathlib
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import p0_predict_transition as transition
import p1_predict_validation as validation


class PredictValidationTests(unittest.TestCase):
    def test_replay_decodes_each_compressed_array_once_and_keeps_both_action_branches(self):
        # 直接执行重放入口，防止只测试缓存函数却遗漏实际逐帧读取路径。
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            episode = root / "LIBERO/episodes/synthetic"
            episode.mkdir(parents=True)
            bddl = root / "synthetic.bddl"
            bddl.write_text("synthetic simulator fixture; no GPU or robot")
            (root / "collection.json").write_text("synthetic collection fixture")
            digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            (root / "LIBERO/worker_spec.json").write_text(json.dumps({"conditions": [{
                "condition_id": "synthetic-standard", "bddl_path": str(bddl), "bddl_sha256": digest(bddl)}]}))
            images = np.stack([np.full((256, 256, 3), i, dtype=np.uint8) for i in range(20)])
            actions = np.zeros((20, 7), dtype=np.float32)
            actions[:, 0] = 1
            trajectory = episode / "trajectory.npz"
            np.savez_compressed(trajectory, action=actions, agentview=images, wrist=images,
                                state=np.arange(20, dtype=np.float64)[:, None], final_sim_state=np.asarray([20.]))
            initial = episode / "initial_state.npz"
            np.savez_compressed(initial, pre_settle_state=np.asarray([0.]))
            hashes = {str(initial.relative_to(root)): digest(initial)}
            for view in ["agentview", "wrist"]:
                path = episode / (view + "_last.png")
                Image.fromarray(np.full((256, 256, 3), 20, dtype=np.uint8)).save(path)
                hashes[str(path.relative_to(root))] = digest(path)
            row = {"episode_id": "synthetic", "implementation": "LIBERO", "artifacts": str(episode),
                   "trajectory": str(trajectory), "trajectory_sha256": digest(trajectory),
                   "condition_id": "synthetic-standard", "init_seed": 17, "policy_seed": 18,
                   "policy_ckpt_sha256": "synthetic-policy", "instruction": "synthetic task", "task_id": "t00"}

            class Environment:
                def __init__(self, **_kwargs): self.position = 0
                def seed(self, _value): pass
                def reset(self): self.position = 0
                def observation(self):
                    image = np.full((256, 256, 3), self.position, dtype=np.uint8)
                    return {"agentview_image": image, "robot0_eye_in_hand_image": image,
                            "robot0_gripper_qpos": np.zeros(2), "robot0_eef_pos": np.asarray([self.position, 0., 0.]),
                            "robot0_eef_quat": np.asarray([0., 0., 0., 1.])}
                def set_init_state(self, state):
                    self.position = int(state[0])
                    return self.observation()
                def step(self, action):
                    self.position += int(action[0])
                    return self.observation(), 0., False, {}
                def get_sim_state(self): return np.asarray([float(self.position)])
                def check_success(self): return self.position == 20
                def close(self): pass

            modules = {name: ModuleType(name) for name in ["libero", "libero.libero", "libero.libero.envs"]}
            modules["libero.libero.envs"].OffScreenRenderEnv = Environment
            actual_load = np.load
            reads = []

            class CountedArchive:
                def __init__(self, value): self.value = value
                def __enter__(self): return self
                def __exit__(self, *_args): self.value.close()
                def __getitem__(self, key):
                    reads.append(key)
                    return self.value[key]

            def counted_load(path, **kwargs):
                value = actual_load(path, **kwargs)
                return CountedArchive(value) if pathlib.Path(path) == trajectory else value

            with patch.dict(sys.modules, modules), patch.object(transition, "ensure_gpu_idle"), \
                    patch.object(transition, "sha256", digest), patch.object(transition, "configure_libero"), \
                    patch.object(transition, "restore_fixture_state"), patch.object(transition, "restore_integration_state"), \
                    patch.object(transition, "observation_state", lambda obs: obs["robot0_eef_pos"][:1]), \
                    patch.object(transition, "checked_collection", return_value=(
                        {"config": {"settle_steps": 0}}, [row], {"artifacts_sha256": hashes})), \
                    patch("numpy.load", side_effect=counted_load):
                for first in [0, 4]:
                    with self.subTest(first=first):
                        reads.clear()
                        output = root / f"output_{first}"
                        output.mkdir()
                        report = transition.replay(output, SimpleNamespace(
                            collection=root / "collection.json", episode_id="synthetic", frame=first))
                        self.assertCountEqual(reads, ["action", "agentview", "wrist", "state", "final_sim_state"])
                        self.assertEqual(report["terminal_reference_used"], first == 4)
                        self.assertTrue(all(branch["replay_max_rgb_error"] == 0
                                            and branch["replay_max_state_error"] == 0 for branch in report["branches"]))
                        for branch in ["executed", "hold"]:
                            with actual_load(output / (branch + ".npz"), allow_pickle=False) as packet:
                                expected = first + 16 if branch == "executed" else first
                                np.testing.assert_array_equal(packet["agentview"][-1],
                                    np.full((256, 256, 3), expected, dtype=np.uint8))
                                np.testing.assert_array_equal(packet["wrist"][-1], packet["agentview"][-1])
                                self.assertEqual(packet["proprio"].shape, (17, 9))

    def selection(self):
        return {"conditions": ["t00-standard"], "outcomes": [False, True],
                "maximum_episodes_per_stratum": 2, "minimum_episodes_per_stratum": 1,
                "positions": ["initial", "frame80_or_last_valid", "terminal_16"]}

    def episode(self, identity, success, count=120):
        return {"episode_id": identity, "is_eval": False, "outcome": {"source": "env", "success": success},
                "condition_id": "t00-standard", "metrics": {"steps": count}, "init_seed": 63000,
                "policy_seed": 73000, "policy_ckpt_sha256": "synthetic-parent",
                "trajectory": "/synthetic-not-real/trajectory.npz", "trajectory_sha256": "synthetic-trajectory"}

    def test_selection_is_predeclared_and_terminal_window_does_not_repeat_frames(self):
        episodes = [self.episode("synthetic-success-b", True), self.episode("synthetic-failure", False, 16),
                    self.episode("synthetic-success-a", True)]
        plan = validation.select_states(episodes, self.selection())
        self.assertEqual(plan, validation.select_states(list(reversed(episodes)), self.selection()))
        self.assertEqual(plan["source_episodes"], 3)
        short = [row for row in plan["cases"] if row["source_episode_id"] == "synthetic-failure"]
        self.assertEqual(len(short), 1)
        self.assertEqual(short[0]["first_frame"], 0)
        self.assertEqual(short[0]["future_frame"], 16)
        self.assertEqual(short[0]["positions"], self.selection()["positions"])
        self.assertEqual(plan["source_initial_seed_count"], 1)
        self.assertEqual({row["first_frame"] for row in plan["cases"]}, {0, 80, 104})

    def test_missing_stratum_eval_sources_and_duplicate_episode_reject(self):
        source = [self.episode("synthetic-failure", False), self.episode("synthetic-success", True)]
        for problem in ["missing", "duplicate", "eval", "short"]:
            episodes = copy.deepcopy(source)
            if problem == "missing": episodes.pop()
            elif problem == "duplicate": episodes.append(copy.deepcopy(episodes[0]))
            elif problem == "eval": episodes[0]["is_eval"] = True
            else: episodes[1]["metrics"]["steps"] = 15
            with self.subTest(problem=problem), self.assertRaises(ValueError):
                validation.select_states(episodes, self.selection())

    def test_real_terminal_png_is_hash_bound_and_last_action_is_allowed(self):
        self.assertEqual(transition.action_window(104, 120), (104, 120))
        for first, count in [(105, 120), (-1, 120), (True, 120), (0, 15)]:
            with self.subTest(first=first, count=count), self.assertRaises(ValueError):
                transition.action_window(first, count)
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            episode = root / "synthetic-episode"
            episode.mkdir()
            checksums = {}
            for view in ["agentview", "wrist"]:
                path = episode / (view + "_last.png")
                Image.fromarray(np.full((256, 256, 3), 90, dtype=np.uint8)).save(path)
                checksums[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
            with patch.object(transition, "sha256", lambda path: hashlib.sha256(path.read_bytes()).hexdigest()):
                images = transition.terminal_reference({"artifacts": str(episode)}, {"artifacts_sha256": checksums}, root)
                self.assertEqual(int(images[0][0, 0, 0]), 90)
                Image.fromarray(np.full((256, 256, 3), 91, dtype=np.uint8)).save(episode / "wrist_last.png")
                with self.assertRaises(ValueError):
                    transition.terminal_reference({"artifacts": str(episode)}, {"artifacts_sha256": checksums}, root)

    def metric_fixture(self, directory):
        case = validation.select_states([self.episode("synthetic-failure", False),
            self.episode("synthetic-success", True)], self.selection())["cases"][0]
        tr = directory / "transition.json"
        pred = directory / "predict_report.json"
        tr.write_text("synthetic transition, no model")
        pred.write_text("synthetic prediction report, no model")
        transition_report = {"source_episode_id": case["source_episode_id"], "first_frame": case["first_frame"],
            "future_frame": case["future_frame"], "policy_ckpt_sha256": case["policy_ckpt_sha256"],
            "branches": [{"name": branch, "environment_success_after_16_steps": False} for branch in ["executed", "hold"]]}
        report = {"transition_sha256": validation.sha256(tr), "same_current_state": True, "same_sampling_seed": True,
            "eligible_for_training": False, "eligible_for_planning": False, "sample_count": 1,
            "action_branches": 2, "steps": 4, "seed": 42, "batch": 1, "precision": "bf16", "attention": "torch SDPA",
            "action_horizon": 16, "environment_control_hz": 20,
            "condition_audit": [{"max_injection_error": 0, "clean_condition": True} for _ in range(2)], "branches": []}
        hashes = {}
        for branch in ["executed", "hold"]:
            report["branches"].append({"name": branch, "predict_decode_wall_s": 1.,
                "metrics": {view: {"prediction_mae_255": 1., "copy_current_mae_255": 4.} for view in ["agentview", "wrist"]}})
            for view in ["agentview", "wrist"]:
                for kind, value in [("prediction", 127), ("truth", 128), ("current", 124)]:
                    path = directory / f"{branch}_{view}_{kind}.png"
                    Image.fromarray(np.full((224, 224, 3), value, dtype=np.uint8)).save(path)
                    hashes[path.name] = validation.sha256(path)

        def sealed(path, stage):
            return (transition_report, {}) if stage == "predict_transition" else (report, {"artifacts_sha256": hashes})
        return case, tr, pred, report, transition_report, sealed

    def test_metrics_recompute_actual_original_pixels_and_reject_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            case, tr, pred, report, _, sealed = self.metric_fixture(root)
            with patch.object(validation, "sealed", side_effect=sealed):
                rows = validation.checked_metrics(case, tr, pred)
                self.assertEqual(len(rows), 4)
                self.assertTrue(all(row["prediction_mae_255"] == 1 for row in rows))
                report["branches"][0]["metrics"]["agentview"]["prediction_mae_255"] = .1
                with self.assertRaises(ValueError): validation.checked_metrics(case, tr, pred)
                report["branches"][0]["metrics"]["agentview"]["prediction_mae_255"] = 1.
                Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)).save(root / "hold_agentview_truth.png")
                with self.assertRaises(ValueError): validation.checked_metrics(case, tr, pred)

    def test_terminal_endpoint_requires_original_environment_outcome(self):
        with tempfile.TemporaryDirectory() as temporary:
            case, tr, pred, _, report, sealed = self.metric_fixture(pathlib.Path(temporary))
            case["first_frame"] = case["action_count"] - 16
            case["future_frame"] = case["action_count"]
            report.update(first_frame=case["first_frame"], future_frame=case["future_frame"], terminal_reference_used=True)
            with patch.object(validation, "sealed", side_effect=sealed):
                self.assertEqual(len(validation.checked_metrics(case, tr, pred)), 4)
                report["branches"][0]["environment_success_after_16_steps"] = True
                with self.assertRaises(ValueError): validation.checked_metrics(case, tr, pred)

    def test_missing_condition_audit_wrong_batch_and_training_permission_reject(self):
        with tempfile.TemporaryDirectory() as temporary:
            case, tr, pred, report, _, sealed = self.metric_fixture(pathlib.Path(temporary))
            for key, value in [("condition_audit", []), ("batch", 2), ("eligible_for_training", True), ("seed", 43)]:
                original = report[key]
                report[key] = value
                with self.subTest(key=key), patch.object(validation, "sealed", side_effect=sealed), self.assertRaises(ValueError):
                    validation.checked_metrics(case, tr, pred)
                report[key] = original

    def test_all_predeclared_cases_export_to_images_csv_and_bound_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            case, tr, pred, _, _, sealed = self.metric_fixture(directory)
            with patch.object(validation, "sealed", side_effect=sealed):
                rows = validation.checked_metrics(case, tr, pred)
                plan = {"cases": [case]}
                report = {"inputs": {"plan": plan}, "summary": validation.summarize(rows, plan)}
                result = validation.export_materials(directory, rows, report)
            output = pathlib.Path(result["directory"])
            manifest = json.loads((output / "materials_manifest.json").read_text())
            self.assertTrue(manifest["all_endpoint_comparisons_exported"])
            self.assertFalse(manifest["prediction_video_created"])
            self.assertEqual(len(manifest["sources"]), 12)
            for name in ["endpoint_comparison_001.png", "predict_pixel_errors.png"]:
                with Image.open(output / name) as image:self.assertEqual(image.size, (3840, 2160))
                self.assertEqual(validation.sha256(output / name), manifest["files_sha256"][name])

    def test_equal_episode_weight_and_no_pixel_to_success_promotion(self):
        plan = validation.select_states([self.episode("synthetic-failure", False, 16),
            self.episode("synthetic-success", True)], self.selection())
        rows = []
        for case in plan["cases"]:
            for branch in ["executed", "hold"]:
                for view in ["agentview", "wrist"]:
                    rows.append({**case, "branch": branch, "view": view,
                        "prediction_mae_255": 2. if case["source_environment_success"] else 0., "copy_current_mae_255": 3.})
        summary = validation.summarize(rows, plan)
        self.assertEqual(summary["equal_episode_mean"]["prediction_mae_255"], 1.)
        self.assertTrue(summary["prediction_beats_copy_on_mean_pixel_error"])
        self.assertFalse(summary["eligible_for_planning"])
        self.assertFalse(summary["eligible_for_training"])
        self.assertFalse(summary["pixel_error_is_task_success"])
        for mutation in ["missing", "duplicate", "source"]:
            altered = copy.deepcopy(rows)
            if mutation == "missing": altered.pop()
            elif mutation == "duplicate": altered.append(copy.deepcopy(altered[0]))
            else: altered[0]["source_episode_id"] = "different-source"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): validation.summarize(altered, plan)

    def test_wait_for_reason_cannot_start_from_missing_process_handle(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            (root / "logs").mkdir()
            (root / "logs/reason_followup.log").write_text("waiting; no sealed results yet")
            with self.assertRaises(RuntimeError):
                validation.wait_reason({"reason_exec": str(root), "reason_pid": 999999999,
                                        "iteration_run": "/synthetic-iteration"}, 1)


if __name__ == "__main__":
    unittest.main()
