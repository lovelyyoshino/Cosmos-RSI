"""CPU回归：反馈路由不得误用成功、冻结来源、错配策略或缺少统计支持的条件。

@author pony
@date 2026-11-09
@version v1.0.0
@last_modified 2026-11-09
@changelog
    - v1.0.0 (2026-11-09): 合成夹具只检查来源、协议及任务等待，不冒充模型实验。
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
import p1_feedback_generation as generation
from p0_transfer_smoke import validate_generation_spec


def fixture():
    episodes = []
    statistics = {}
    for condition, failures in [("standard", 9), ("background", 19)]:
        for index in range(20):
            episodes.append({"episode_id": f"synthetic-{condition}-{index:02d}", "instruction": "fixture task",
                "task_id": "synthetic-task", "condition": condition, "condition_id": f"t00-{condition}",
                "conditions": {"background": f"t00-{condition}"}, "policy_ckpt_sha256": "synthetic-parent",
                "outcome": {"success": index >= failures, "source": "env"}, "is_eval": False,
                "isolation": {"passed": True}})
        lower, upper = generation.wilson(failures, 20)
        statistics[condition] = {"task_id": "synthetic-task", "condition": condition,
            "failures": failures, "episodes": 20, "failure_rate": failures / 20,
            "wilson95_lower": lower, "wilson95_upper": upper}
    records = [{"episode_id": row["episode_id"], "route": "cosmos_restyle", "category": "visual",
        "sub_type": "background", "is_eval": False, "eligible_for_generation": True,
        "routing_authority": "environment_truth_condition_comparison", "policy_ckpt_sha256": "synthetic-parent",
        "target_condition": row["conditions"], "evidence": {"condition_statistics": statistics["background"],
        "standard_statistics": statistics["standard"]}} for row in episodes
        if row["condition"] == "background" and row["outcome"]["success"] is False]
    return episodes, records


class FeedbackGenerationTests(unittest.TestCase):
    def test_valid_routes_have_same_task_and_stable_identity_order(self):
        episodes, records = fixture()
        rows = generation.select_route(episodes, list(reversed(records)), "fixture task", 20)
        self.assertEqual([r["episode_id"] for r in rows], sorted(r["episode_id"] for r in records))
        self.assertEqual(len(rows), 19)

    def test_generation_rejects_failure_identity_and_policy_tampering(self):
        for mutation in ["success", "eval_episode", "eval_record", "isolation", "parent", "condition", "target", "authority"]:
            episodes, records = fixture()
            row = next(r for r in episodes if r["episode_id"] == records[0]["episode_id"])
            if mutation == "success": row["outcome"]["success"] = True
            elif mutation == "eval_episode": row["is_eval"] = True
            elif mutation == "eval_record": records[0]["is_eval"] = True
            elif mutation == "isolation": row["isolation"]["passed"] = False
            elif mutation == "parent": records[0]["policy_ckpt_sha256"] = "other-parent"
            elif mutation == "condition": row["condition"] = "lighting"
            elif mutation == "target": records[0]["target_condition"] = {"background": "other"}
            elif mutation == "authority": records[0]["routing_authority"] = "uncalibrated_judge"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                generation.select_route(episodes, records, "fixture task", 20)

    def test_missing_or_inaccurate_statistical_support_is_rejected(self):
        for mutation in ["missing", "changed_counts", "weak_difference", "wrong_group_parent"]:
            episodes, records = fixture()
            if mutation == "missing": episodes = episodes[1:]
            elif mutation == "changed_counts": records[0]["evidence"]["condition_statistics"]["failures"] = 18
            elif mutation == "weak_difference":
                for row in episodes:
                    if row["condition"] == "standard": row["outcome"]["success"] = False
            elif mutation == "wrong_group_parent": episodes[0]["policy_ckpt_sha256"] = "stale"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                generation.select_route(episodes, records, "fixture task", 20)

    def test_cross_task_unrouted_and_duplicate_records_do_not_fall_back(self):
        episodes, records = fixture()
        for instruction, actual in [("other task", records), ("fixture task", []),
                                     ("fixture task", [*records, copy.deepcopy(records[0])])]:
            with self.subTest(instruction=instruction), self.assertRaises(ValueError):
                generation.select_route(episodes, actual, instruction, 20)
        with self.assertRaises(ValueError):
            generation.select_route([*episodes, copy.deepcopy(episodes[0])], records, "fixture task", 20)

    def test_configuration_keeps_new_data_unqualified_and_recipe_fixed(self):
        config = generation.checked_config(ROOT / "configs/feedback_generation.yaml")
        self.assertFalse(config["authority"]["eligible_for_training"])
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "changed.json"
            changed = copy.deepcopy(config)
            changed["authority"]["eligible_for_training"] = True
            path.write_text(json.dumps(changed))
            with self.assertRaises(ValueError): generation.checked_config(path)

    def test_generated_spec_uses_copied_non_table_mask_and_supported_four_step_recipe(self):
        directory = pathlib.Path("/synthetic-not-an-experiment")
        spec = generation.generation_spec(directory, "fixture task")
        validate_generation_spec(spec, "edge/distilled", 4)
        self.assertEqual(spec["edge"]["mask_path"], str(directory / "scene_mask.mp4"))
        self.assertIn("fixture task", spec["prompt"])
        self.assertNotIn("guided_generation_mask", spec)

    def test_native_result_cannot_claim_generation_when_protocol_or_output_is_wrong(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            video = directory / "synthetic.mp4"
            video.write_bytes(b"synthetic binding fixture, not video or experiment")
            source = directory / "input.json"
            source.write_text("synthetic sealed input stand-in")
            input_report = {"files": {"feedback": {"path": "synthetic", "sha256": "fixture"}},
                "source_episode_id": "synthetic", "sampling": generation.CONFIG["sampling"],
                "target_condition": generation.CONFIG["target_condition"]}
            native = {"guardrails_enabled": True, "recipe": "edge/distilled", "num_steps": 4,
                      "seed": 42, "batch": 1, "outputs": [str(video)]}
            with patch.object(generation, "checked_input", return_value=input_report), \
                    patch.object(generation, "ensure_gpu_idle"), \
                    patch("p0_transfer_smoke.generate", return_value=native):
                for field, invalid in [("guardrails_enabled", False), ("recipe", "edge"), ("num_steps", 35),
                                       ("seed", 44), ("outputs", []), ("batch", 2)]:
                    original = native[field]
                    native[field] = invalid
                    with self.subTest(field=field), self.assertRaises(ValueError):
                        generation.generate(directory, source)
                    native[field] = original

    def test_waiting_missing_exact_predict_process_does_not_restart_anything(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            (root / "logs").mkdir()
            (root / "launch.json").write_text(json.dumps({"pid": 123456789, "iteration_run": "synthetic"}))
            (root / "logs/predict_validation.log").write_text("")
            with patch.object(generation, "ROOT", root), self.assertRaisesRegex(RuntimeError, "已退出"):
                generation.wait_predict(root, {"files": {}}, 1)

    def test_waiting_accepts_only_completed_predict_with_same_feedback(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            result_dir = root / "runs/p0/synthetic-predict"
            result_dir.mkdir(parents=True)
            (result_dir / "probe.json").write_text("synthetic")
            result_path = result_dir / "predict_validation.json"
            result_path.write_text("synthetic")
            execution = root / "runs/p0/synthetic-exec"
            (execution / "logs").mkdir(parents=True)
            (execution / "launch.json").write_text(json.dumps({"pid": 123456789, "iteration_run": "synthetic-iteration"}))
            (execution / "logs/predict_validation.log").write_text("记录目录：" + str(result_dir) + "\n")
            feedback = {"path": "synthetic-feedback", "sha256": "fixture"}
            report = {"diagnostic_complete": True, "inputs": {"iteration_run": "synthetic-iteration",
                      "files": {"feedback": feedback}}}
            with patch.object(generation, "ROOT", root), patch.object(generation, "sealed", return_value=(report, {})):
                self.assertEqual(generation.wait_predict(execution, {"files": {"feedback": feedback}}, 1)["path"], str(result_path))
                for mutation in ["incomplete", "other_feedback", "other_iteration"]:
                    changed = copy.deepcopy(report)
                    if mutation == "incomplete": changed["diagnostic_complete"] = False
                    elif mutation == "other_feedback": changed["inputs"]["files"]["feedback"]["sha256"] = "other"
                    elif mutation == "other_iteration": changed["inputs"]["iteration_run"] = "other"
                    with self.subTest(mutation=mutation), \
                            patch.object(generation, "sealed", return_value=(changed, {})), self.assertRaises(ValueError):
                        generation.wait_predict(execution, {"files": {"feedback": feedback}}, 1)


if __name__ == "__main__":
    unittest.main()
