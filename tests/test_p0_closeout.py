"""P0收口不能错绑父权重、假冒反馈生成或放宽既有冻结门禁。

@author pony
@date 2026-11-09
@version v1.1.0
@last_modified 2026-11-09
@changelog
    - v1.1.0 (2026-11-09): 覆盖预算整数类型与既有提升、回退和延迟门槛保护。
    - v1.0.0 (2026-11-09): 验证实际产物关系、训练配对、预算和冻结集边界。
"""
import copy
import pathlib
import sys
import unittest

import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from p0_closeout import validate_budget, validate_links


class CloseoutTests(unittest.TestCase):
    def artifacts(self):
        records = {
            "baseline": {"policy_ckpt_sha256": "parent"},
            "feedback": {"policy_ckpt_sha256": "parent"},
            "reason": {"policy_ckpt_sha256": "parent", "collection_sha256": "feedback"},
            "mining": {"policy_ckpt_sha256": "parent", "collection_sha256": "feedback", "judge_sha256": "reason"},
            "generated_data": {},
            "deployment_smoke": {"training_report_sha256": "candidate_budget", "policy_ckpt_sha256": "candidate",
                                 "passed": True, "production_deployed": False},
            "predict_transition": {}, "predict_embeddings": {},
            "predict": {"transition_sha256": "predict_transition", "embeddings_sha256": "predict_embeddings",
                        "same_current_state": True, "same_sampling_seed": True, "eligible_for_planning": False}}
        for name in ["candidate_budget", "control_budget", "analytic_budget"]:
            records[name] = {"parent_ckpt_sha256": "parent", "handoff_sha256": "generated_data",
                "paired_plan_sha256": "plan", "reload_max_abs_diff": 0, "eligible_for_promotion": False,
                "steps": 40, "batch": 8, "seed": 42, "output_ckpt_sha256": "candidate"}
        return {name: {"artifact_verified": True, "latest_valid": {"sha256": name, "data": row}}
                for name, row in records.items()}

    def test_component_evidence_does_not_become_feedback_directed_generation(self):
        result = validate_links(self.artifacts())
        self.assertTrue(result["feedback_chain_linked"])
        self.assertTrue(result["generated_data_training_deployment_linked"])
        self.assertFalse(result["feedback_directed_generation_demonstrated"])

    def test_changed_parent_plan_or_training_seed_is_rejected(self):
        for name, key, value in [("mining", "judge_sha256", "wrong"),
                                 ("control_budget", "paired_plan_sha256", "wrong"),
                                 ("analytic_budget", "seed", 99),
                                 ("candidate_budget", "parent_ckpt_sha256", "wrong"),
                                 ("deployment_smoke", "production_deployed", True),
                                 ("predict", "eligible_for_planning", True)]:
            with self.subTest(name=name, key=key):
                evidence = self.artifacts()
                evidence[name]["latest_valid"]["data"][key] = value
                with self.assertRaises(ValueError):
                    validate_links(evidence)

    def test_missing_or_unverified_component_cannot_close_p0(self):
        evidence = self.artifacts()
        evidence["predict"]["artifact_verified"] = False
        with self.assertRaises(ValueError): validate_links(evidence)

    def test_budget_preserves_frozen_counts_and_matched_training(self):
        path = pathlib.Path(__file__).resolve().parents[1] / "configs/iteration_budget.yaml"
        budget = yaml.safe_load(path.read_text())
        frozen = budget["evaluation"]["sha256"]
        validate_budget(budget, frozen)
        for changed in ["candidate_quota", "eval_counts", "frozen_sha", "missing_pool", "approval"]:
            value = copy.deepcopy(budget)
            if changed == "candidate_quota": value["training"]["candidate_quotas"]["demo"] += 1
            elif changed == "eval_counts": value["evaluation"]["candidate"]["target_perturbed"] = 10
            elif changed == "frozen_sha": value["evaluation"]["sha256"] = "wrong"
            elif changed == "missing_pool": value["training"]["absent_qualified_pool"] = "fill_with_unqualified"
            else: value["promotion"]["human_approval_required"] = False
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                validate_budget(value, frozen)

    def test_float_steps_or_changed_effect_thresholds_are_not_accepted(self):
        path = pathlib.Path(__file__).resolve().parents[1] / "configs/iteration_budget.yaml"
        budget = yaml.safe_load(path.read_text())
        for section, key, value in [("training", "steps_per_role", 1000.0),
            ("promotion", "min_gain_vs_control", "any_positive"), ("promotion", "max_latency_p95_ratio", 2.),
            ("promotion", "require_gain_vs_incumbent", False),
            ("promotion", "max_standard_nontarget_drop", "unlimited"),
            ("promotion", "consecutive_accepted_iterations_required", 1)]:
            changed = copy.deepcopy(budget); changed[section][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_budget(changed, budget["evaluation"]["sha256"])


if __name__ == "__main__":
    unittest.main()
