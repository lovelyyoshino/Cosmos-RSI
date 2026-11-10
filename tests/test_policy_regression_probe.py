"""验证零更新诊断不会错用训练分区、掩盖数据错位或接错后台生成。

@author pony
@date 2026-11-09
@version v1.1.0
@last_modified 2026-11-09
@changelog
    - v1.1.0 (2026-11-09): 实际CPU子进程验证角色退出门与错误停止，覆盖动作产物独立重算。
    - v1.0.1 (2026-11-09): 覆盖排队前实际冻结定义与父策略隔离版本一致性。
    - v1.0.0 (2026-11-09): 覆盖独立验证覆盖、标签/腕图漂移、参数异常和确切任务失败。
"""
from __future__ import annotations

import copy
import json
import os
import pathlib
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import p1_policy_regression_probe as probe


class ValidationSelectionTests(unittest.TestCase):
    def manifest(self):
        return {"horizon": 8, "episodes": [
            {"episode_id": f"task{task}-{split}-{i}", "instruction": f"task{task}",
             "split": split, "length": 100 + i, "offset": (task * 8 + i) * 128}
            for task in range(10) for split in ["demo_train", "demo_validation"] for i in range(4)]}

    def test_validation_is_task_stratified_and_order_independent(self):
        manifest = self.manifest()
        locations = probe.validation_locations(manifest, probe.CONFIG["selection"])
        self.assertEqual(len(locations), 60)
        self.assertEqual(len({row["episode_id"] for row in locations}), 20)
        self.assertEqual({row["split"] for row in locations}, {"demo_validation"})
        manifest["episodes"].reverse()
        self.assertEqual(probe.validation_locations(manifest, probe.CONFIG["selection"]), locations)

    def test_missing_task_duplicate_id_or_short_source_rejects(self):
        for mutation in ["missing", "duplicate", "short"]:
            manifest = self.manifest()
            if mutation == "missing":
                manifest["episodes"] = [r for r in manifest["episodes"] if r["instruction"] != "task0"]
            elif mutation == "duplicate": manifest["episodes"].append(copy.deepcopy(manifest["episodes"][0]))
            else: next(r for r in manifest["episodes"] if r["split"] == "demo_validation")["length"] = 9
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                probe.validation_locations(manifest, probe.CONFIG["selection"])

    def test_cached_example_rejects_training_source_and_out_of_bounds(self):
        cache = SimpleNamespace(manifest=self.manifest())
        for location in [{"episode_id": "task0-demo_train-0", "frame": 0, "horizon": 8},
                         {"episode_id": "task0-demo_validation-0", "frame": 93, "horizon": 8}]:
            with self.subTest(location=location), self.assertRaises(ValueError):
                probe.cached_example(cache, location, "demo_validation")


class SourceAlignmentTests(unittest.TestCase):
    def example(self):
        return {"lang": "fixture", "state": np.zeros((1, 8), dtype=np.float32),
                "action": np.zeros((8, 7), dtype=np.float32),
                "image": [Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)) for _ in range(2)]}

    def test_generated_allows_only_agentview_difference(self):
        source = self.example(); changed = self.example()
        changed["image"][0] = Image.fromarray(np.ones((8, 8, 3), dtype=np.uint8))
        probe.assert_example_equal(changed, source, generated=True)
        with self.assertRaises(ValueError): probe.assert_example_equal(changed, source)
        changed["image"][1] = changed["image"][0]
        with self.assertRaises(ValueError): probe.assert_example_equal(changed, source, generated=True)

    def test_generated_cannot_hide_normalization_or_language_drift(self):
        for field in ["lang", "state", "action"]:
            source, changed = self.example(), self.example()
            if field == "lang": changed[field] = "wrong"
            else: changed[field][...] = .1
            with self.subTest(field=field), self.assertRaises(ValueError):
                probe.assert_example_equal(changed, source, generated=True)

    def test_zero_action_drift_and_gripper_errors_are_separate(self):
        source = np.zeros((8, 7)); changed = source.copy(); changed[:, 6] = 1
        values = probe.action_metrics(changed, source, source)
        self.assertEqual(values["arm_mse"], 0)
        self.assertEqual(values["gripper_mse"], 1)
        self.assertAlmostEqual(values["action_mse"], 1 / 7)
        self.assertEqual(probe.action_metrics(source, source, source)["drift_vs_incumbent_rms"], 0)
        for invalid in [np.zeros((7, 7)), np.full((8, 7), np.nan)]:
            with self.assertRaises(ValueError): probe.action_metrics(invalid, source, source)


class WeightAndQueueTests(unittest.TestCase):
    def test_binding_pins_frozen_definition_and_rejects_parent_version_drift(self):
        import yaml

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp); iteration = root / "runs/p0/iteration"; iteration.mkdir(parents=True)
            frozen = root / "eval/frozen_eval_set.json"; frozen.parent.mkdir(); frozen.write_text("{}")
            cache = root / "data/manifest.json"; cache.parent.mkdir()
            cache.write_text(json.dumps(ValidationSelectionTests().manifest()))
            config = root / "config.yaml"; config.write_text(yaml.safe_dump(probe.CONFIG))
            source_files = {}
            for name in ["collection", "handoff"]:
                path = root / (name + ".json"); path.write_text("{}")
                source_files[name] = str(path); source_files[name + "_sha256"] = probe.sha256(path)
            parent = root / "parent"; parent.mkdir(); (parent / "state.pt").write_bytes(b"synthetic parent")
            shared = {"cache_manifest_sha256": probe.sha256(cache), "cache_manifest": str(cache),
                      "base_manifest_sha256": "synthetic base", "normalizer": {}, "model_config": {}, "precision": "fixture",
                      "eval_overlap_check": {"eval_set_sha256": probe.sha256(frozen)}}
            metadata = {**shared, "state_sha256": probe.sha256(parent / "state.pt")}
            (parent / "checkpoint.json").write_text(json.dumps(metadata))
            state = {"iteration": 1, "inputs": {"parent": {"path": str(parent)}}, "completed_stages": {}}
            reports = {}
            for role in probe.ROLES[1:]:
                checkpoint = root / role; checkpoint.mkdir(); (checkpoint / "state.pt").write_bytes(role.encode())
                (checkpoint / "checkpoint.json").write_text(json.dumps({**shared,
                    "state_sha256": probe.sha256(checkpoint / "state.pt"), "parent_ckpt_sha256": metadata["state_sha256"]}))
                report_path = root / (role + ".json"); report_path.write_text("{}")
                state["completed_stages"]["train_" + role] = {"path": str(report_path), "sha256": probe.sha256(report_path)}
                reports[role] = {"output_ckpt": str(checkpoint), "data_qualification": source_files}
            (iteration / "iteration_state.json").write_text(json.dumps(state))
            with patch.object(probe, "ROOT", root), patch.object(probe, "DEPENDENCIES", []), \
                    patch.object(probe, "checked_update", side_effect=lambda path, role: (reports[role], {})):
                bound = probe.bind_inputs(iteration, config)
                self.assertEqual(bound["files"]["frozen_eval"], {"path": str(frozen), "sha256": probe.sha256(frozen)})
                frozen.write_text('{"changed":true}')
                with self.assertRaisesRegex(ValueError, "冻结定义"):
                    probe.bind_inputs(iteration, config)

    def test_weight_deltas_and_invalid_state(self):
        try:
            import torch
        except (ImportError, AttributeError) as error:
            self.skipTest(f"本地CPU torch无法导入，真实参数审计须在Thor执行：{error}")
        before = {"action_model.weight": torch.tensor([3., 4.])}
        after = {"action_model.weight": torch.tensor([6., 8.])}
        self.assertEqual(probe.tensor_deltas(before, after)["action_head"]["relative_delta_l2"], 1.)
        self.assertEqual(probe.tensor_deltas(before, before)["action_head"]["delta_l2"], 0.)
        for wrong in [{}, {"action_model.weight": torch.tensor([float("nan"), 0.])},
                      {"action_model.weight": torch.zeros(1)}, {"action_model.weight": torch.zeros(2, dtype=torch.float64)}]:
            with self.subTest(wrong=str(wrong)), self.assertRaises(ValueError): probe.tensor_deltas(before, wrong)

    def test_own_cuda_context_does_not_allow_other_processes(self):
        with patch("p0_common.command", return_value={"returncode": 0, "stdout": str(os.getpid())}):
            probe.ensure_no_other_gpu_processes()
        with patch("p0_common.command", return_value={"returncode": 0, "stdout": f"{os.getpid()}\n999999"}), self.assertRaises(RuntimeError):
            probe.ensure_no_other_gpu_processes()

    def test_exact_generation_mismatch_or_dead_process_rejects(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp); stage = root / "runs/p0/generation"; stage.mkdir(parents=True)
            launch = root / "launch.json"; launch.write_text("{}")
            log = root / "generation.log"; log.write_text("记录目录：" + str(stage) + "\n")
            (stage / "probe.json").write_text("{}")
            binding = {"launch": str(launch), "launch_sha256": probe.sha256(launch), "log": str(log),
                       "input_sha256": "expected", "input": str(root / "input.json"), "pid": 2147483647,
                       "after_predict_exec": str(root / "predict")}
            with patch.object(probe, "ROOT", root), patch.object(probe, "sealed", return_value=(
                    {"input_sha256": "wrong", "generation_performed": True}, {})), self.assertRaises(ValueError):
                probe.wait_generation(binding, .1)
            log.write_text("")
            with patch.object(probe, "ROOT", root), self.assertRaises(RuntimeError): probe.wait_generation(binding, .1)

    def test_waiting_inputs_are_immutable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp); file = root / "config.yaml"; file.write_text("before")
            binding = {"files": {"config": {"path": str(file), "sha256": probe.sha256(file)}},
                       "dependencies_sha256": {}}
            file.write_text("after")
            with patch.object(probe, "ROOT", root), self.assertRaises(ValueError): probe.verify_bindings(binding)


class IsolatedRoleTests(unittest.TestCase):
    def fixture(self, root):
        checkpoint = root / "checkpoint"; checkpoint.mkdir()
        metadata = {"state_sha256": "synthetic-state", "precision": "synthetic-fp32",
                    "prediction_probe": {"normalized_actions": np.zeros((1, 8, 7)).tolist(),
                                         "locations": [{"frame": 0}], "seed": 44}}
        path = checkpoint / "checkpoint.json"; path.write_text(json.dumps(metadata))
        bindings = {"iteration_run": str(root / "iteration"), "locations": [{"frame": 0}],
                    "checkpoints": {"incumbent": str(checkpoint)},
                    "files": {"checkpoint_incumbent": {"path": str(path), "sha256": probe.sha256(path)}}}
        bindings_path = root / "bindings.json"; bindings_path.write_text(json.dumps(bindings))
        report = {"schema": "thor-isolated-role-action-probe-v1", "role": "incumbent",
                  "bindings_sha256": probe.sha256(bindings_path), "checkpoint": str(checkpoint),
                  "state_sha256": metadata["state_sha256"], "precision": metadata["precision"],
                  "inference_seeds": [44, 45], "optimization_steps": 0,
                  "reproduction": {"max_abs_diff": 0., "passed": True, "locations": [{"frame": 0}],
                                   "seed": 44, "optimization_steps": 0},
                  "observed_saved_probe": np.zeros((1, 8, 7)).tolist(),
                  "predictions": np.zeros((1, 2, 8, 7)).tolist()}
        return bindings, bindings_path, report

    def test_role_output_recomputes_probe_and_rejects_mismatch_or_invalid_arrays(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp); bindings, source, report = self.fixture(root)
            output = root / "role.json"; output.write_text(json.dumps(report))
            _, predictions = probe.checked_role_output(output, bindings, source, "incumbent")
            self.assertEqual(predictions.shape, (1, 2, 8, 7))
            for mutation in ["role", "bindings", "shape", "nan", "observed", "claimed_error"]:
                changed = copy.deepcopy(report)
                if mutation == "role": changed["role"] = "candidate"
                elif mutation == "bindings": changed["bindings_sha256"] = "wrong"
                elif mutation == "shape": changed["predictions"] = []
                elif mutation == "nan": changed["predictions"][0][0][0][0] = float("nan")
                elif mutation == "observed": changed["observed_saved_probe"][0][0][0] = .1
                else: changed["reproduction"]["max_abs_diff"] = .1
                output.write_text(json.dumps(changed))
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    probe.checked_role_output(output, bindings, source, "incumbent")

    def execute_fixture(self, root, script):
        bindings, source, report = self.fixture(root)
        (root / "logs").mkdir(); (root / "scripts").mkdir()
        (root / "fixture.json").write_text(json.dumps(report))
        (root / "scripts/p1_policy_regression_probe.py").write_text(script)
        return bindings, source

    def test_actual_cpu_child_exits_before_teardown_and_result_is_accepted(self):
        from p1_stage_transition import process_command

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            bindings, source = self.execute_fixture(root,
                "import pathlib,sys\n"
                "output=pathlib.Path(sys.argv[sys.argv.index('--role-output')+1])\n"
                "output.write_bytes(pathlib.Path('fixture.json').read_bytes())\n")
            def teardown(binding):
                self.assertEqual(process_command(binding["pid"]), [])
                self.assertIn("role-actions", binding["command"])
                return {"process_exited": True, "gpu_idle": True}
            with patch.object(probe, "ROOT", root), patch.object(probe, "ensure_gpu_idle") as idle, \
                    patch.object(probe, "wait_stage_teardown", side_effect=teardown) as teardown_check:
                report, values, worker = probe.execute_role(root, bindings, source, "incumbent")
            idle.assert_called_once(); teardown_check.assert_called_once()
            self.assertEqual(values.shape, (1, 2, 8, 7))
            self.assertTrue(worker["teardown"]["process_exited"])
            self.assertEqual(worker["output_sha256"], probe.sha256(pathlib.Path(worker["output"])))

    def test_failed_child_or_missing_result_cannot_be_promoted_to_success(self):
        for script, exception in [("raise SystemExit(7)\n", RuntimeError), ("pass\n", FileNotFoundError)]:
            with self.subTest(script=script), tempfile.TemporaryDirectory() as tmp:
                root = pathlib.Path(tmp); bindings, source = self.execute_fixture(root, script)
                with patch.object(probe, "ROOT", root), patch.object(probe, "ensure_gpu_idle"), \
                        patch.object(probe, "wait_stage_teardown", return_value={}), self.assertRaises(exception):
                    probe.execute_role(root, bindings, source, "incumbent")

    def test_uncleared_gpu_stops_before_reading_successful_child_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            bindings, source = self.execute_fixture(root,
                "import pathlib,sys\n"
                "pathlib.Path(sys.argv[sys.argv.index('--role-output')+1]).write_bytes(pathlib.Path('fixture.json').read_bytes())\n")
            with patch.object(probe, "ROOT", root), patch.object(probe, "ensure_gpu_idle"), \
                    patch.object(probe, "wait_stage_teardown", side_effect=RuntimeError("GPU未释放")), \
                    patch.object(probe, "checked_role_output") as reader, self.assertRaisesRegex(RuntimeError, "GPU未释放"):
                probe.execute_role(root, bindings, source, "incumbent")
            reader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
