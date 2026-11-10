"""判官开发失败不能冒称独立验证，排队也不能抢占原始整圈。

@author pony
@date 2026-11-09
@version v1.0.0
@last_modified 2026-11-09
@changelog
    - v1.0.0 (2026-11-09): 覆盖独立验证的执行条件、真实种子组合、固定提示及等待边界。
"""
from __future__ import annotations

import copy
import json
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import p1_reason_followup as followup


class ReasonFollowupTests(unittest.TestCase):
    def fixture(self, directory):
        import yaml

        config = yaml.safe_load((ROOT / "configs/reason_followup.yaml").read_text())
        config["validation"].update(conditions=["t00-standard"], per_condition=1)
        paths = {name: directory / (name + ".json") for name in ["config", "collection", "judge_config", "rollout_config"]}
        for path in paths.values():
            path.write_text("synthetic CPU orchestration fixture; no inference")
        paths["config"].write_text(yaml.safe_dump(config))
        bindings = {"parent": str(directory / "synthetic-parent"),
                    "files": {name: {"path": str(path), "sha256": followup.sha256(path)} for name, path in paths.items()}}
        original = [{"episode_id": "synthetic-dev", "condition_id": "t00-standard", "init_seed": 63000,
                     "policy_seed": 73000, "policy_ckpt_sha256": "synthetic-parent", "is_eval": False}]
        heldout = [{**original[0], "episode_id": "synthetic-validation", "init_seed": 83000, "policy_seed": 93000}]
        return config, paths, bindings, original, heldout

    def test_development_rejection_stops_before_validation_collection(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            _, _, bindings, _, _ = self.fixture(directory)
            seen = []

            def execute(_directory, phase, *_arguments):
                seen.append(phase)
                return directory / (phase + ".json")

            report = {"collection_sha256": bindings["files"]["collection"]["sha256"],
                      "calibration": {"calibration_passed": False}}
            with patch.object(followup, "execute_stage", side_effect=execute), \
                    patch.object(followup, "sealed", return_value=(report, {})):
                result = followup.compare(directory, bindings)
            self.assertEqual(seen, ["development_judge", "development_materials"])
            self.assertFalse(result["heldout_validation_performed"])
            self.assertFalse(result["independent_validation_passed"])
            self.assertFalse(result["judge_authority_changed"])
            self.assertFalse(result["complete_rsi"])

    def test_fixed_config_validates_actual_new_collection_before_calibration(self):
        for calibration_passed in [False, True]:
            with self.subTest(passed=calibration_passed), tempfile.TemporaryDirectory() as temporary:
                directory = pathlib.Path(temporary)
                _, paths, bindings, original, heldout = self.fixture(directory)
                seen = []

                def execute(_directory, phase, _script, _stage, _artifact, arguments):
                    seen.append((phase, list(arguments)))
                    result = directory / (phase + ".json")
                    result.write_text("synthetic result; no model")
                    return result

                def sealed(path, _stage):
                    development = path.name == "development_judge.json"
                    source = paths["collection"] if development else directory / "heldout_collection.json"
                    return {"collection_sha256": followup.sha256(source), "config": {"fixed_prompt": "synthetic"},
                            "calibration": {"calibration_passed": True if development else calibration_passed}}, {}

                def collection(path):
                    return {"policy_ckpt_sha256": "synthetic-parent"}, original if path == paths["collection"] else heldout, {}

                with patch.object(followup, "execute_stage", side_effect=execute), \
                        patch.object(followup, "sealed", side_effect=sealed), \
                        patch.object(followup, "checked_collection", side_effect=collection):
                    result = followup.compare(directory, bindings)
                self.assertEqual([phase for phase, _ in seen], ["development_judge", "development_materials",
                                  "heldout_collection", "heldout_judge", "heldout_materials"])
                self.assertTrue(result["heldout_validation_performed"])
                self.assertEqual(result["independent_validation_passed"], calibration_passed)
                self.assertFalse(result["judge_authority_changed"])
                self.assertFalse(result["training_pool_changed"])
                self.assertEqual(seen[0][1][-1], seen[3][1][-1])

    def test_tampered_queue_inputs_fail_before_any_model_stage(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            _, paths, bindings, _, _ = self.fixture(directory)
            paths["judge_config"].write_text("changed after queueing")
            with patch.object(followup, "execute_stage") as execute, self.assertRaises(ValueError):
                followup.compare(directory, bindings)
            execute.assert_not_called()

    def test_holdout_rejects_overlap_wrong_policy_missing_or_repeated_case(self):
        with tempfile.TemporaryDirectory() as temporary:
            config, _, _, original, heldout = self.fixture(pathlib.Path(temporary))
            for problem in ["seed", "parent", "eval", "missing", "duplicate", "id"]:
                values = copy.deepcopy(heldout)
                if problem == "seed": values[0]["init_seed"] = original[0]["init_seed"]
                elif problem == "parent": values[0]["policy_ckpt_sha256"] = "wrong-parent"
                elif problem == "eval": values[0]["is_eval"] = True
                elif problem == "missing": values.clear()
                elif problem == "duplicate": values.append(copy.deepcopy(values[0]))
                else: values[0]["episode_id"] = original[0]["episode_id"]
                with self.subTest(problem=problem), self.assertRaises(ValueError):
                    followup.validate_holdout(original, values, config["validation"], "synthetic-parent")

    def test_wait_cannot_treat_incomplete_or_changed_iteration_as_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            with self.assertRaises(TimeoutError):
                followup.wait_for_completion(directory, 0, {})
            (directory / "probe.json").write_text("{}")
            with patch.object(followup, "sealed", side_effect=ValueError("synthetic engineering failure")), self.assertRaises(ValueError):
                followup.wait_for_completion(directory, 1, {})
            report = {"production_deployed": False, "complete_rsi": False, "inputs": {"parent": "changed"}}
            with patch.object(followup, "sealed", return_value=(report, {})), self.assertRaises(ValueError):
                followup.wait_for_completion(directory, 1, {"parent": "original"})


if __name__ == "__main__":
    unittest.main()
