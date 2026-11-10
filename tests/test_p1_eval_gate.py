"""用明确的CPU合成计数和文件核验门禁，不能将测试夹具当成实验结果。

@author pony
@date 2026-11-09
@version v1.0.0
@last_modified 2026-11-09
@changelog
    - v1.0.0 (2026-11-09): 覆盖阈值、回退、拒绝、1200回合完整性和来源绑定。
"""
from __future__ import annotations

import copy
import json
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import p1_eval_gate as gate


def counts(standard=60, target=80, nontarget=20):
    return {key: {"n": n, "success": success, "p": success / n}
            for (key, n), success in zip(gate.SPLITS.items(), [standard, target, nontarget])}


class GateDecisionTests(unittest.TestCase):
    def test_good_candidate_still_requires_user_approval(self):
        result = gate.decision(counts(target=160), counts(), counts(), 1.1)
        self.assertEqual(result["decision"], "accept_pending_approval")
        self.assertTrue(result["human_approval_required"])
        self.assertFalse(result["production_deployed"])
        self.assertFalse(result["complete_rsi"])

    def test_exact_2_5_percentage_point_floor_is_not_lost_to_float_subtraction(self):
        result = gate.decision(counts(target=24), counts(target=4), counts(target=4), 1.2)
        self.assertTrue(result["statistical_gate_passed"])
        self.assertEqual(result["target_gain_vs_control"], .025)

    def test_insufficient_gain_incumbent_tie_regression_or_latency_rejects(self):
        cases = [(counts(target=90), counts(), counts(), 1.),
                 (counts(target=160), counts(), counts(target=160), 1.),
                 (counts(standard=0, target=160), counts(), counts(), 1.),
                 (counts(target=160, nontarget=0), counts(), counts(), 1.),
                 (counts(target=160), counts(), counts(), 1.20001)]
        for arguments in cases:
            with self.subTest(arguments=arguments):
                result = gate.decision(*arguments)
                self.assertEqual(result["decision"], "reject")
                self.assertFalse(result["eligible_for_promotion"])
                self.assertTrue(result["reject_reasons"])

    def test_missing_subset_short_counts_fake_rate_and_nonfinite_latency_fail(self):
        for key, value in [("n", 799), ("n", True), ("success", 801), ("success", True),
                           ("p", .5), ("p", float("nan"))]:
            changed = counts(target=160)
            changed["target_perturbed"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                gate.decision(changed, counts(), counts(), 1.)
        for ratio in [True, 0, -1, float("inf"), float("nan")]:
            with self.subTest(ratio=ratio), self.assertRaises(ValueError):
                gate.decision(counts(target=160), counts(), counts(), ratio)
        missing = counts(); missing.pop("standard")
        with self.assertRaises(ValueError): gate.rates(missing)


class ActualEvaluationBindingTests(unittest.TestCase):
    def fixture(self, root, name="synthetic_evaluation", policy="synthetic-policy", frozen_sha="synthetic-freeze"):
        directory = root / "runs/p0" / name; directory.mkdir(parents=True)
        summary_path = directory / "evaluation.json"
        frozen, actual = [], []
        for split, n in gate.SPLITS.items():
            for index in range(n):
                fixed = {"eval_id": f"synthetic-{split}-{index}", "condition_id": "fixture-condition",
                         "split": split, "init_seed": 100000 + index, "policy_seed": 34001,
                         "initial_state_sha256": "fixture-initial-state"}
                frozen.append(fixed)
                actual.append({**fixed, "is_eval": True, "eligible_for_training": False,
                    "outcome": {"source": "env", "success": index % 2 == 0},
                    "policy_ckpt_sha256": policy, "eval_set_sha256": frozen_sha,
                    "batch": 1, "control_hz": 20, "action_horizon": 8, "env": "synthetic-libero",
                    "precision": "fixture-fp32", "policy_noise_schedule": "fixture-independent-seeds"})
        summary = {"policy_ckpt_sha256": policy, "eval_set_sha256": frozen_sha,
                   "full_eval_complete": True, "completed_episodes": 1200, "server_stopped": True,
                   "server_forced_kill": False, "server_returncode": 0, "real_robot_commands_sent": False,
                   "splits": {key: {"n": n, "success": n // 2, "p": .5} for key, n in gate.SPLITS.items()},
                   "precision": "fixture-fp32", "batch": 1, "sample_count": 1200,
                   "policy_noise_schedule": "fixture-independent-seeds", "latency_ms": {"p50": 1., "p95": 2.}}
        self.write(directory, summary, actual)
        return summary_path, {"sha256": frozen_sha, "episodes": frozen}, summary, actual

    def write(self, directory, summary, actual):
        (directory / "episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in actual))
        (directory / "evaluation.json").write_text(json.dumps(summary))
        probe = {"stage": "policy_evaluation", "status": "passed",
                 "hardware": {"host": "thor", "arch": "aarch64", "power_mode": "synthetic-fixture"},
                 "upstream_sha": {key: "synthetic-sha" for key in ["starVLA", "LIBERO", "LIBERO-plus"]},
                 "artifacts_sha256": {name: gate.sha256(directory / name) for name in ["evaluation.json", "episodes.jsonl"]}}
        (directory / "probe.json").write_text(json.dumps(probe))

    def test_complete_fixture_is_recounted_and_runtime_is_separately_returned(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary); path, frozen, *_ = self.fixture(root)
            with patch.object(gate, "ROOT", root):
                report, runtime = gate.checked_evaluation(path, "synthetic-policy", frozen)
            self.assertEqual(report["splits"]["target_perturbed"]["success"], 400)
            self.assertEqual(len(runtime), 1202)

    def test_missing_duplicate_wrong_seed_truth_checkpoint_or_summary_is_rejected(self):
        for corruption in ["missing", "duplicate", "seed", "truth", "checkpoint", "summary", "freeze", "batch"]:
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as temporary:
                root = pathlib.Path(temporary); path, frozen, summary, rows = self.fixture(root)
                if corruption == "missing": rows.pop()
                elif corruption == "duplicate": rows[-1] = copy.deepcopy(rows[0])
                elif corruption == "seed": rows[0]["policy_seed"] += 1
                elif corruption == "truth": rows[0]["outcome"]["source"] = "model"
                elif corruption == "checkpoint": rows[0]["policy_ckpt_sha256"] = "other"
                elif corruption == "summary": summary["splits"]["standard"]["success"] += 1
                elif corruption == "freeze": summary["eval_set_sha256"] = "other"
                else: rows[0]["batch"] = 2
                self.write(path.parent, summary, rows)
                with patch.object(gate, "ROOT", root), self.assertRaises(ValueError):
                    gate.checked_evaluation(path, "synthetic-policy", frozen)

    def test_changed_journal_does_not_pass_previously_sealed_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary); path, frozen, *_ = self.fixture(root)
            (path.parent / "episodes.jsonl").write_text("changed after seal")
            with patch.object(gate, "ROOT", root), self.assertRaises(ValueError):
                gate.checked_evaluation(path, "synthetic-policy", frozen)


class CompleteGateContractTests(unittest.TestCase):
    def seal(self, path, stage):
        files = {str(p.relative_to(path.parent)): gate.sha256(p) for p in path.parent.rglob("*")
                 if p.is_file() and p.name != "probe.json"}
        (path.parent / "probe.json").write_text(json.dumps({"stage": stage, "status": "passed",
            "hardware": {"host": "thor", "arch": "aarch64"}, "artifacts_sha256": files,
            "started_at": "2026-11-09T03:04:35+08:00"}))

    def fixture(self, root):
        helper = ActualEvaluationBindingTests()
        _, frozen, *_ = helper.fixture(root, "synthetic_seed_eval")
        frozen.pop("sha256")
        frozen_path = root / "eval/frozen_eval_set.json"; frozen_path.parent.mkdir()
        frozen_path.write_text(json.dumps(frozen)); frozen_sha = gate.sha256(frozen_path)
        budget = yaml.safe_load((pathlib.Path(__file__).resolve().parents[1] / "configs/iteration_budget.yaml").read_text())
        budget["evaluation"]["sha256"] = frozen_sha
        budget_path = root / "budget.yaml"; budget_path.write_text(yaml.safe_dump(budget))
        qualification = {"budget_sha256": gate.sha256(budget_path), "lighting_generation_qualified": False}
        for name, stage in [("collection", "feedback_rollout"), ("mining", "feedback_mine"), ("handoff", "lerobot_handoff")]:
            path = root / "runs/p0" / ("synthetic_" + name) / (name + ".json"); path.parent.mkdir()
            path.write_text("{}"); self.seal(path, stage)
            qualification.update({name: str(path), name + "_sha256": gate.sha256(path)})
        reports = {}
        for role in ["candidate", "control"]:
            run = root / "runs/p0" / ("synthetic_train_" + role); checkpoint = run / "checkpoint"; checkpoint.mkdir(parents=True)
            (checkpoint / "state.pt").write_bytes(("synthetic weights " + role).encode())
            plan = {"synthetic_contract_fixture": True, "iteration": 1}
            plan["sha256"] = gate.paired_plan_sha256(plan)
            (run / "paired_plan.json").write_text(json.dumps(plan))
            recipe = {"steps": 1000, "batch": 8, "seed": 42}
            meta = {"state_sha256": gate.sha256(checkpoint / "state.pt"), "formal_policy_update": True,
                    "budget_only": False, "role": role, "paired_plan_sha256": plan["sha256"],
                    "parent_ckpt_sha256": "synthetic-parent", "recipe": recipe, "iteration": 1,
                    "base_manifest_sha256": "synthetic-base", "cache_manifest_sha256": "synthetic-cache"}
            (checkpoint / "checkpoint.json").write_text(json.dumps(meta))
            report = {"schema": "thor-feedback-policy-update-v1", "role": role, "training_performed": True,
                      "eligible_for_evaluation": True, "eligible_for_promotion": False, "reload_max_abs_diff": 0.,
                      "output_ckpt": str(checkpoint), "output_ckpt_sha256": meta["state_sha256"],
                      "parent_ckpt_sha256": "synthetic-parent", "parent_checkpoint": str(root / "synthetic-parent"),
                      "recipe": recipe, "paired_plan_sha256": plan["sha256"], "iteration": 1, **recipe,
                      "data_qualification": qualification, "new_cosmos_generation_performed": False,
                      "data_mix": {key: {"count": value} for key, value in budget["training"][role + "_quotas"].items()}}
            report_path = run / "train_run.json"; report_path.write_text(json.dumps(report)); self.seal(report_path, "p1_policy_train")
            reports[role] = report_path
        args = SimpleNamespace(budget=budget_path, candidate_training=reports["candidate"], control_training=reports["control"])
        for role in ["candidate", "control", "incumbent"]:
            policy = "synthetic-parent" if role == "incumbent" else json.loads(reports[role].read_text())["output_ckpt_sha256"]
            evaluation, *_ = helper.fixture(root, "synthetic_eval_" + role, policy, frozen_sha)
            setattr(args, role + "_evaluation", evaluation)
        candidate = json.loads(reports["candidate"].read_text())
        smoke = root / "runs/p0/synthetic_smoke/deployment_smoke.json"; smoke.parent.mkdir()
        smoke.write_text(json.dumps({"passed": True, "policy_ckpt_sha256": candidate["output_ckpt_sha256"],
            "parent_ckpt_sha256": "synthetic-parent", "training_report_sha256": gate.sha256(reports["candidate"]),
            "production_deployed": False, "rollback_action_exact": True,
            "production_slots_and_frozen_eval_unchanged": True, "latency_p95_ratio": 1.}))
        self.seal(smoke, "augmented_deploy_smoke"); args.deployment_smoke = smoke
        return args

    def test_entire_contract_preserves_rejection_and_actual_rollout_start(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary); args = self.fixture(root)
            output = root / "synthetic_gate"; output.mkdir()
            with patch.object(gate, "ROOT", root): result = gate.evaluate_gate(output, args)
            self.assertEqual(result["decision"], "reject")
            self.assertEqual(result["rollout_started_at"], "2026-11-09T03:04:35+08:00")
            self.assertFalse(result["new_cosmos_generation_performed"])
            self.assertTrue((output / "iter.json").is_file())

    def test_gate_rejects_feedback_replaced_after_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary); args = self.fixture(root)
            report = json.loads(args.candidate_training.read_text())
            pathlib.Path(report["data_qualification"]["collection"]).write_text("replaced")
            output = root / "synthetic_gate"; output.mkdir()
            with patch.object(gate, "ROOT", root), self.assertRaises(ValueError): gate.evaluate_gate(output, args)


if __name__ == "__main__":
    unittest.main()
