"""验证保守续训真实恢复AdamW矩、低学习率及独立开发边界，不放宽原冻结门禁。

@author pony
@date 2026-11-09
@version v1.1.0
@last_modified 2026-11-09
@changelog
    - v1.1.0 (2026-11-09): 用实际冻结注册表复现开发种子冲突，检查续用训练的父权重和配方边界。
    - v1.0.0 (2026-11-09): 覆盖父scheduler峰值泄漏、缺失矩、开发种子错绑和确切队列失败。
"""
from __future__ import annotations

import copy
import json
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import p1_policy_update as update
import p1_conservative_pilot as pilot
import p1_eval_gate as gate


class OptimizerRecipeTests(unittest.TestCase):
    def budget(self):
        return {"training": {"steps_per_role": 250, "optimizer": {
            "name": "adamw", "lr_vlm": 2e-6, "lr_head": 1e-5, "weight_decay": .01,
            "warmup_steps": 20, "restore_parent_state": True, "scheduler": "linear_warmup_cosine_min0.1"}}}

    def test_legacy_recipe_keeps_original_constant_rates(self):
        recipe = update.optimizer_recipe({"training": {}})
        self.assertEqual(recipe["lr_head"], 1e-4)
        self.assertEqual(recipe["lr_vlm"], 1e-5)
        self.assertFalse(recipe["optimizer_state_restored"])
        self.assertEqual(recipe["scheduler"], "constant_matched_policy_update_v1")

    def test_invalid_optimizer_controls_reject(self):
        for key, value in [("lr_head", float("nan")), ("lr_vlm", 0), ("warmup_steps", 250),
                           ("restore_parent_state", 1), ("scheduler", "unknown"), ("weight_decay", -1)]:
            budget = self.budget(); budget["training"]["optimizer"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): update.optimizer_recipe(budget)

    def torch(self):
        try:
            import torch
            return torch
        except (ImportError, AttributeError, NameError) as error:
            self.skipTest(f"本地torch不可用；优化器数值验证在Thor CPU进行：{error}")

    def fixture(self):
        torch = self.torch()
        tensors = {"qwen_vl_interface.weight": torch.nn.Parameter(torch.tensor([1., 2.])),
                   "action_model.weight": torch.nn.Parameter(torch.tensor([3., 4.]))}
        optimizer = torch.optim.AdamW([{"params": [tensors[key]], "lr": lr}
            for key, lr in zip(tensors, [2e-5, 1e-4])])
        for parameter in tensors.values(): parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        saved = {"optimizer": copy.deepcopy(optimizer.state_dict()), "optimizer_param_names": [[key] for key in tensors]}
        for group in saved["optimizer"]["param_groups"]: group["initial_lr"] = group["lr"]
        return torch, tensors, saved

    def test_parent_moments_survive_and_scheduler_cannot_restore_old_peak(self):
        torch, tensors, saved = self.fixture()
        recipe = update.optimizer_recipe(self.budget())
        optimizer, scheduler, names = update.configure_optimizer(tensors, saved, recipe, 250)
        self.assertEqual(names, saved["optimizer_param_names"])
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 2e-6 / 20)
        self.assertAlmostEqual(optimizer.param_groups[1]["lr"], 1e-5 / 20)
        self.assertEqual(scheduler.base_lrs, [2e-6, 1e-5])
        for parameter, moment in zip(tensors.values(), saved["optimizer"]["state"].values()):
            self.assertTrue(torch.equal(optimizer.state[parameter]["exp_avg"], moment["exp_avg"]))
            self.assertEqual(float(optimizer.state[parameter]["step"]), 1.)

    def test_missing_moments_wrong_order_shape_and_nonfinite_reject(self):
        torch, tensors, original = self.fixture()
        recipe = update.optimizer_recipe(self.budget())
        for mutation in ["absent", "order", "shape", "nan", "step"]:
            saved = copy.deepcopy(original)
            if mutation == "absent": saved.pop("optimizer")
            elif mutation == "order": saved["optimizer_param_names"].reverse()
            elif mutation == "shape": saved["optimizer"]["state"][0]["exp_avg"] = torch.zeros(3)
            elif mutation == "nan": saved["optimizer"]["state"][0]["exp_avg_sq"][0] = float("nan")
            else: saved["optimizer"]["state"][0]["step"] = torch.tensor(0.)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                update.configure_optimizer(tensors, saved, recipe, 250)


class DevelopmentBoundaryTests(unittest.TestCase):
    def test_preflight_uses_real_frozen_registry_before_training(self):
        frozen = json.loads((pathlib.Path(__file__).resolve().parents[1] / "eval/frozen_eval_set.json").read_text())
        cache = {"episodes": [{"init_seed": 0}]}
        with patch.object(pilot, "INITIAL_SEED", 103000), self.assertRaisesRegex(ValueError, "种子相交"):
            pilot.development_plan(frozen, cache, 520)
        plan = pilot.development_plan(frozen, cache, 520)
        self.assertEqual(len(plan["cases"]), 9)
        self.assertTrue(all(not row["is_eval"] for row in plan["cases"]))
        self.assertFalse({row["init_seed"] for row in plan["cases"]} &
                         {row["init_seed"] for row in frozen["episodes"]})
        with patch.object(pilot, "POLICY_SEED", frozen["episodes"][0]["policy_seed"]), \
                self.assertRaisesRegex(ValueError, "种子相交"):
            pilot.development_plan(frozen, cache, 520)

    def test_reused_training_must_match_actual_parent_plan_recipe_and_qualification(self):
        bindings = {"parent_ckpt_sha256": "parent", "paired_plan_sha256": "plan",
                    "source_qualification": {"collection_sha256": "feedback"},
                    "optimizer": {"lr_head": 1e-5, "optimizer_state_restored": True}}
        report = {"role": "candidate", "steps": 250, "batch": 8, "seed": 42,
                  "parent_ckpt_sha256": "parent", "paired_plan_sha256": "plan",
                  "data_qualification": bindings["source_qualification"],
                  "development_only": True, "training_performed": True,
                  "eligible_for_evaluation": False, "eligible_for_promotion": False,
                  "recipe": {"steps": 250, "batch": 8, "seed": 42, **bindings["optimizer"]}}
        pilot.validate_reused_training(report, bindings, "candidate")
        for key, value in [("role", "control"), ("steps", 1000), ("batch", 1), ("seed", 43),
                           ("parent_ckpt_sha256", "other"), ("paired_plan_sha256", "other"),
                           ("data_qualification", {}), ("recipe", {}), ("development_only", False),
                           ("eligible_for_evaluation", True), ("eligible_for_promotion", True)]:
            changed = copy.deepcopy(report); changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                pilot.validate_reused_training(changed, bindings, "candidate")

    def test_development_weights_cannot_pass_formal_update_qualification(self):
        report = {"schema": "thor-feedback-policy-update-v1", "role": "candidate", "training_performed": True,
                  "eligible_for_evaluation": False, "eligible_for_promotion": False, "development_only": True}
        with patch.object(gate, "sealed", return_value=(report, {})), self.assertRaises(ValueError):
            gate.checked_update(pathlib.Path("synthetic-pilot/train_run.json"), "candidate")

    def rows(self):
        return [{"condition_id": condition, "init_seed": pilot.INITIAL_SEED + i,
            "policy_seed": pilot.POLICY_SEED + i, "is_eval": False, "eligible_for_training": False,
            "policy_ckpt_sha256": "synthetic", "outcome": {"success": i == 0, "source": "env"}}
            for condition in pilot.CONDITIONS for i in range(pilot.PER_CONDITION)]

    def test_development_has_fixed_cases_and_no_training_permission(self):
        self.assertEqual(pilot.validate_development(self.rows(), "synthetic"),
            {key: {"success": 1, "n": 3} for key in pilot.CONDITIONS})
        for key, value in [("is_eval", True), ("eligible_for_training", True),
                           ("init_seed", 0), ("policy_ckpt_sha256", "other")]:
            rows = self.rows(); rows[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): pilot.validate_development(rows, "synthetic")
        with self.assertRaises(ValueError): pilot.validate_development(self.rows()[:-1], "synthetic")

    def test_mutated_budget_or_source_cannot_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "budget.yaml"; path.write_text("before")
            bindings = {"files": {"budget": {"path": str(path), "sha256": pilot.sha256(path)}}, "dependencies_sha256": {}}
            path.write_text("changed")
            with patch.object(pilot, "project", side_effect=lambda path: pathlib.Path(path)), \
                    self.assertRaises(ValueError): pilot.verify_inputs(bindings)

    def test_missing_exact_probe_stops_instead_of_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp); launch = root / "launch.json"; launch.write_text("{}")
            log = root / "actions.log"; log.write_text("")
            binding = {"launch": str(launch), "launch_sha256": pilot.sha256(launch),
                       "log": str(log), "pid": 2147483647, "command": []}
            with patch.object(pilot, "project", side_effect=lambda path: pathlib.Path(path)), \
                    self.assertRaises(RuntimeError): pilot.wait_probe(binding, .1)


if __name__ == "__main__":
    unittest.main()
