"""CPU核验自身成功来源、后续圈晋升、明确审批和整圈恢复边界。

@author pony
@date 2026-11-09
@version v1.2.0
@last_modified 2026-11-09
@changelog
    - v1.2.0 (2026-11-09): 覆盖第二、三圈旧派生池的真实祖先、来源一致性与原质量门复验。
    - v1.1.0 (2026-11-09): 覆盖晋升权重实际采集资格及第二圈完整证明传递，防止只跑通首圈。
    - v1.0.0 (2026-11-09): 失败动作、预算父权重、错审批和恢复输入变化必须被拒绝。
"""
from __future__ import annotations

import copy
import importlib.util
import json
import pathlib
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import p1_eval_gate
import p1_policy_update as update
import p0_feedback_chain as feedback_chain
import p0_feedback_rollout as feedback_rollout
import p0_augmented_deploy_smoke as deployment
from p1_promote import verify_approval

spec = importlib.util.spec_from_file_location("rsi_loop_test", ROOT / "rsi/loop.py")
loop = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loop)


class SuccessPoolTests(unittest.TestCase):
    def fixture(self):
        episode = {"episode_id": "synthetic-success", "outcome": {"success": True, "source": "env"},
                   "is_eval": False, "isolation": {"passed": True}, "policy_ckpt_sha256": "parent",
                   "metrics": {"steps": 24}, "lerobot_index": {"episode_index": 0}, "instruction": "fixture"}
        item = {"source_episode_id": "synthetic-success", "policy_ckpt_sha256": "parent",
                "eligible_for_training": True, "environment_truth_used": True, "source_split": "rollout_train",
                "is_eval": False, "window": {"start_frame": 0, "end_frame": 24}, "source_length": 24,
                "lerobot_index": {"episode_index": 0}, "instruction": "fixture"}
        return episode, item

    def test_success_has_separate_rollout_training_description(self):
        episode, item = self.fixture()
        result = update.qualify_successes([episode], [item], "parent")
        self.assertEqual(result[0]["kind"], "self_distill")
        self.assertEqual(result[0]["source_split"], "rollout_train")

    def test_failure_eval_stale_policy_shifted_labels_and_duplicates_reject(self):
        for mutation in ["failure", "eval", "parent", "labels", "duplicate", "isolation"]:
            episode, item = self.fixture()
            if mutation == "failure": episode["outcome"]["success"] = False
            elif mutation == "eval": episode["is_eval"] = True
            elif mutation == "parent": item["policy_ckpt_sha256"] = "stale"
            elif mutation == "labels": item["window"]["end_frame"] = 23
            elif mutation == "isolation": episode["isolation"]["passed"] = False
            successes = [item, copy.deepcopy(item)] if mutation == "duplicate" else [item]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                update.qualify_successes([episode], successes, "parent")


class RecursiveHandoffTests(unittest.TestCase):
    def fixture(self, root, generations=1):
        shared = {"cache_manifest_sha256": "synthetic-d0", "normalizer": {"action": "synthetic"},
                  "base_manifest_sha256": "synthetic-base", "model_config": {"name": "synthetic-policy"},
                  "precision": "synthetic-precision", "attention": "sdpa", "recipe": {"name": "lora"}}

        def checkpoint(path, metadata, payload):
            path.mkdir(parents=True)
            (path / "state.pt").write_bytes(payload)
            metadata["state_sha256"] = p1_eval_gate.sha256(path / "state.pt")
            (path / "checkpoint.json").write_text(json.dumps(metadata))
            return path, metadata

        anchor_path, anchor = checkpoint(root / "runs/p0/anchor/checkpoint",
            {**shared, "formal_baseline_training": True, "step": 5000}, b"synthetic baseline weights")
        path, parent = anchor_path, anchor
        for iteration in range(1, generations + 1):
            path, parent = checkpoint(root / "checkpoints/promoted" / f"iter{iteration}",
                {**shared, "formal_baseline_training": False, "formal_policy_update": True,
                 "iteration": iteration, "parent_checkpoint": str(path), "parent_ckpt_sha256": parent["state_sha256"]},
                f"synthetic iteration {iteration} weights".encode())
        handoff = root / "synthetic-handoff.json"
        handoff.write_text(json.dumps({"parent_checkpoint": str(anchor_path), "parent_ckpt_sha256": anchor["state_sha256"]}))
        return handoff, path, parent, anchor

    def test_first_circle_retains_original_parent_bound_quality_check(self):
        parent = {"formal_baseline_training": True, "step": 5000}
        with patch.object(update, "checked_handoff", return_value={"synthetic": True}) as checked:
            result, ancestry = update.checked_recursive_handoff(pathlib.Path("fixture.json"), parent,
                pathlib.Path("fixture-parent"), 1, None)
            checked.assert_called_once_with(pathlib.Path("fixture.json"), parent)
        self.assertTrue(result["synthetic"])
        self.assertIsNone(ancestry)

    def test_second_and_third_circle_revalidate_original_pool_without_rewriting_it(self):
        for generations in [1, 2]:
            with self.subTest(generations=generations), tempfile.TemporaryDirectory() as temporary:
                root = pathlib.Path(temporary)
                handoff, path, parent, anchor = self.fixture(root, generations)
                original = handoff.read_bytes()
                with patch.object(update, "ROOT", root), patch.object(update, "check_parent", return_value={"verified": True}), \
                        patch.object(update, "checked_handoff", return_value={"qualified": True}) as checked:
                    result, proof = update.checked_recursive_handoff(handoff, parent, path, generations + 1,
                                                                     root / "synthetic-promotion.json")
                checked.assert_called_once_with(handoff, anchor)
                self.assertTrue(result["qualified"])
                self.assertEqual(len(proof["ancestry"]), generations + 1)
                self.assertFalse(proof["source_report_rewritten"])
                self.assertEqual(handoff.read_bytes(), original)

    def test_changed_data_normalizer_model_or_parent_chain_rejects(self):
        for key, value in [("cache_manifest_sha256", "other-d0"), ("normalizer", {}),
                           ("base_manifest_sha256", "other-base"), ("model_config", {}),
                           ("parent_ckpt_sha256", "broken-chain"), ("iteration", 2),
                           ("formal_policy_update", False)]:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temporary:
                root = pathlib.Path(temporary)
                handoff, path, parent, _ = self.fixture(root)
                parent[key] = value
                (path / "checkpoint.json").write_text(json.dumps(parent))
                with patch.object(update, "ROOT", root), patch.object(update, "check_parent", return_value={}):
                    with self.assertRaises(ValueError):
                        update.checked_recursive_handoff(handoff, parent, path, 2, root / "synthetic-promotion.json")

    def test_tampered_ancestor_weights_reject_before_quality_check(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            handoff, path, parent, _ = self.fixture(root)
            (root / "runs/p0/anchor/checkpoint/state.pt").write_bytes(b"changed weights")
            with patch.object(update, "ROOT", root), patch.object(update, "check_parent", return_value={}):
                with self.assertRaises(ValueError):
                    update.checked_recursive_handoff(handoff, parent, path, 2, root / "synthetic-promotion.json")

    def test_real_promotion_and_original_quality_gate_remain_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            handoff, path, parent, _ = self.fixture(root)
            with self.assertRaises(ValueError):
                update.checked_recursive_handoff(handoff, parent, path, 2, None)
            with patch.object(update, "ROOT", root), patch.object(update, "check_parent", return_value={}), \
                    patch.object(update, "checked_handoff", side_effect=ValueError("synthetic failed quality")):
                with self.assertRaisesRegex(ValueError, "failed quality"):
                    update.checked_recursive_handoff(handoff, parent, path, 2, root / "synthetic-promotion.json")


class LifecycleTests(unittest.TestCase):
    def test_budget_weights_and_missing_promotion_cannot_become_next_parent(self):
        with self.assertRaises(ValueError): update.check_parent({"step": 5040}, pathlib.Path("fixture"), 1, None)
        with self.assertRaises(ValueError): update.check_parent({"step": 6000}, pathlib.Path("fixture"), 2, None)
        self.assertIsNone(update.check_parent({"formal_baseline_training": True, "step": 5000}, pathlib.Path("fixture"), 1, None))

    def test_next_parent_requires_actual_current_link_and_preceding_iteration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary); checkpoint = root / "checkpoints/promoted/fixture"; checkpoint.mkdir(parents=True)
            (root / "checkpoints/current").symlink_to(checkpoint)
            promotion_file = root / "synthetic-promotion.json"; promotion_file.write_text("synthetic")
            report = {"iteration": 1, "policy_ckpt_sha256": "new", "promoted_checkpoint": str(checkpoint)}
            with patch.object(update, "ROOT", root), patch.object(p1_eval_gate, "checked_promotion", return_value=report):
                self.assertIsNotNone(update.check_parent({"state_sha256": "new"}, checkpoint, 2, promotion_file))
                with self.assertRaises(ValueError): update.check_parent({"state_sha256": "new"}, checkpoint, 3, promotion_file)
                with self.assertRaises(ValueError): update.check_parent({"state_sha256": "stale"}, checkpoint, 2, promotion_file)

    def test_approval_must_identify_this_passed_gate(self):
        passed = {"statistical_gate_passed": True, "eligible_for_promotion": True,
                  "decision": "accept_pending_approval", "human_approval_required": True}
        digest = "a" * 64
        verify_approval(passed, digest, digest, "synthetic explicit approval fixture")
        for changed, approved, reference in [(passed, "b" * 64, "fixture"), (passed, digest, " "),
            ({**passed, "eligible_for_promotion": False}, digest, "fixture"),
            ({**passed, "decision": "reject"}, digest, "fixture")]:
            with self.subTest(changed=changed, approved=approved), self.assertRaises(ValueError):
                verify_approval(changed, digest, approved, reference)

    def test_resume_uses_durable_state_and_cannot_change_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            (root / "progress.json").write_text(json.dumps({"phase": "subprocess", "pid": 123}))
            bindings = {"parent": {"sha256": "synthetic"}}
            (root / "iteration_state.json").write_text(json.dumps({"inputs": bindings,
                "completed_stages": {"candidate": {"path": "fixture", "sha256": "fixture"}}}))
            self.assertIn("candidate", loop.resume_records(root, bindings))
            with self.assertRaises(ValueError): loop.resume_records(root, {"parent": {"sha256": "changed"}})

    def test_orchestrator_keeps_stages_serial_and_stops_at_concrete_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary); run = root / "runs/p0/synthetic_loop"; run.mkdir(parents=True)
            parent = root / "parent"; parent.mkdir(); (parent / "checkpoint.json").write_text("{}")
            frozen = root / "eval/frozen_eval_set.json"; frozen.parent.mkdir(); frozen.write_text("{}")
            budget = root / "budget.yaml"; budget.write_bytes((ROOT / "configs/iteration_budget.yaml").read_bytes())
            handoff = root / "handoff.json"; handoff.write_text("{}")
            incumbent = root / "incumbent.json"; incumbent.write_text("{}")
            chain = root / "feedback.json"; chain.write_text("{}")
            args = SimpleNamespace(parent=parent, budget=budget, handoff=handoff,
                incumbent_evaluation=incumbent, feedback_chain=chain, parent_promotion=None,
                resume_from=None, iteration=1, seed=42, wait_timeout=1)
            seen = []

            def execute(directory, name, script, stage, artifact, arguments):
                seen.append((name, script, stage, list(arguments)))
                target = root / "runs/p0" / ("synthetic_" + name); target.mkdir()
                contents = {"output_ckpt": str(root / "fixture-checkpoint")}
                if artifact == "iter.json": contents.update(decision="reject", eligible_for_promotion=False)
                path = target / artifact; path.write_text(json.dumps(contents))
                # 模拟执行器自己的临时进度，检查整圈恢复证据仍独立保存。
                (directory / "progress.json").write_text(json.dumps({"phase": name, "pid": 123}))
                return path

            with patch.object(loop, "ROOT", root), patch.object(loop, "validate_budget"), \
                    patch.object(loop, "ensure_gpu_idle"), patch.object(loop, "execute_stage", side_effect=execute), \
                    patch.object(loop, "wait_feedback", return_value={"collection": "fixture", "mining": "fixture", "judge": "fixture"}):
                result = loop.iteration(run, args)
            self.assertFalse(result["production_deployed"])
            self.assertEqual(result["decision"], "reject")
            self.assertEqual([row[0] for row in seen], ["feedback_materials", "data_qualification", "train_candidate", "train_control",
                "train_analytic_ablation", "deployment_smoke", "evaluate_candidate", "analysis_candidate",
                "evaluate_control", "analysis_control", "evaluation_gate"])
            analyses = [row for row in seen if row[0].startswith("analysis_")]
            self.assertTrue(all(row[2] == "evaluation_materials" for row in analyses))
            state = json.loads((run / "iteration_state.json").read_text())
            self.assertEqual(state["phase"], "scientific_gate_rejected")
            self.assertEqual(len(state["completed_stages"]), 12)


class RecursiveFeedbackTests(unittest.TestCase):
    def test_later_deployment_checks_the_same_current_parent_promotion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            parent = root / "checkpoints/promoted/synthetic-iter1"
            parent.mkdir(parents=True)
            current = root / "checkpoints/current"
            current.symlink_to(parent)
            promotion = root / "synthetic-promotion.json"
            promotion.write_text("synthetic verified promotion fixture")
            binding = {"path": str(promotion), "sha256": p1_eval_gate.sha256(promotion)}
            verified = {"iteration": 1, "policy_ckpt_sha256": "synthetic-policy-sha",
                        "promoted_checkpoint": str(parent)}
            training = {"iteration": 2, "parent_ckpt_sha256": "synthetic-policy-sha",
                        "data_qualification": {"parent_promotion": binding}}
            with patch.object(deployment, "ROOT", root), patch.object(p1_eval_gate, "ROOT", root), \
                    patch.object(p1_eval_gate, "checked_promotion", return_value=verified):
                deployment.checked_parent_origin(parent, training, "p1_policy_train")
                for mutation in ["missing_binding", "stale_binding", "wrong_iteration", "budget_parent"]:
                    changed = copy.deepcopy(training)
                    stage = "p1_policy_train"
                    if mutation == "missing_binding":
                        changed["data_qualification"] = {}
                    elif mutation == "stale_binding":
                        changed["data_qualification"]["parent_promotion"]["sha256"] = "stale"
                    elif mutation == "wrong_iteration":
                        changed["iteration"] = 3
                    else:
                        stage = "augmented_budget"
                    with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                        deployment.checked_parent_origin(parent, changed, stage)
                current.unlink()
                with self.assertRaises(ValueError):
                    deployment.checked_parent_origin(parent, training, "p1_policy_train")

    def test_promoted_collection_requires_current_link_and_exact_promotion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            checkpoint = root / "checkpoints/promoted/synthetic-iter1"
            checkpoint.mkdir(parents=True)
            (checkpoint / "state.pt").write_bytes(b"synthetic weights; no real model")
            digest = p1_eval_gate.sha256(checkpoint / "state.pt")
            metadata = {"state_sha256": digest, "iteration": 1}
            (checkpoint / "checkpoint.json").write_text(json.dumps(metadata))
            current = root / "checkpoints/current"
            current.symlink_to(checkpoint)
            promotion_file = root / "synthetic-promotion.json"
            promotion_file.write_text("synthetic sealed-promotion fixture")
            report = {"iteration": 1, "policy_ckpt_sha256": digest,
                      "promoted_checkpoint": str(checkpoint)}
            # 保留实际摘要与链接检查；备用哈希实现兼容备份机Python3.10，不改已准入P0源码。
            with patch.object(feedback_rollout, "ROOT", root), patch.object(p1_eval_gate, "ROOT", root), \
                    patch.object(feedback_rollout, "sha256", p1_eval_gate.sha256), \
                    patch.object(p1_eval_gate, "checked_promotion", return_value=report):
                actual, binding = feedback_rollout.checked_policy_checkpoint(checkpoint, promotion_file)
                self.assertEqual(actual, metadata)
                self.assertEqual(binding["sha256"], p1_eval_gate.sha256(promotion_file))
                with self.assertRaises(ValueError):
                    feedback_rollout.checked_policy_checkpoint(checkpoint)
                for key, stale in [("iteration", 2), ("policy_ckpt_sha256", "stale")]:
                    original = report[key]
                    report[key] = stale
                    with self.subTest(stale=key), self.assertRaises(ValueError):
                        feedback_rollout.checked_policy_checkpoint(checkpoint, promotion_file)
                    report[key] = original
                current.unlink()
                with self.assertRaises(ValueError):
                    feedback_rollout.checked_policy_checkpoint(checkpoint, promotion_file)
                current.symlink_to(checkpoint)
                (checkpoint / "state.pt").write_bytes(b"tampered weights")
                with self.assertRaises(ValueError):
                    feedback_rollout.checked_policy_checkpoint(checkpoint, promotion_file)

    def test_feedback_chain_passes_exact_promotion_to_collection(self):
        for promoted in [False, True]:
            with self.subTest(promoted=promoted), tempfile.TemporaryDirectory() as temporary:
                root = pathlib.Path(temporary)
                promotion = root / "synthetic-promotion.json" if promoted else None
                args = SimpleNamespace(checkpoint=root / "checkpoint", parent_promotion=promotion,
                    conditions=["t00-standard"], init_seed=64000, policy_seed=74000, per_condition=20,
                    rollout_config="rollout.yaml", judge_config="judge.yaml", mining_config="mining.yaml", workers=1)
                seen = []

                def execute(directory, phase, script, expected, artifact, arguments):
                    seen.append((phase, list(arguments)))
                    path = root / artifact
                    path.write_text(json.dumps({"sample_count": 20, "calibration": {"passed": False}}))
                    return path

                with patch.object(feedback_chain, "execute_stage", side_effect=execute), \
                        patch.object(feedback_chain, "sha256", p1_eval_gate.sha256):
                    feedback_chain.chain(root, args)
                self.assertEqual([phase for phase, _ in seen], ["collection", "judge", "mining"])
                collection = seen[0][1]
                if promoted:
                    self.assertEqual(collection[collection.index("--parent-promotion") + 1], promotion)
                else:
                    self.assertNotIn("--parent-promotion", collection)

    def test_second_iteration_recollects_from_promoted_parent_with_new_seeds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            run = root / "runs/p0/synthetic-second-iteration"
            run.mkdir(parents=True)
            parent = root / "checkpoints/promoted/synthetic-iter1"
            parent.mkdir(parents=True)
            (parent / "checkpoint.json").write_text("{}")
            frozen = root / "eval/frozen_eval_set.json"
            frozen.parent.mkdir()
            frozen.write_text("{}")
            budget = root / "budget.yaml"
            budget.write_bytes((ROOT / "configs/iteration_budget.yaml").read_bytes())
            handoff, incumbent, promotion = [root / name for name in ["handoff.json", "incumbent.json", "promotion.json"]]
            for path in [handoff, incumbent, promotion]:
                path.write_text("{}")
            args = SimpleNamespace(parent=parent, budget=budget, handoff=handoff,
                incumbent_evaluation=incumbent, feedback_chain=None, parent_promotion=promotion,
                resume_from=None, iteration=2, seed=42, wait_timeout=1)
            seen = []

            def execute(directory, name, script, expected, artifact, arguments):
                seen.append((name, list(arguments)))
                target = root / "runs/p0" / ("synthetic_" + name)
                target.mkdir()
                contents = {"output_ckpt": str(root / "synthetic-checkpoint")}
                if name == "feedback":
                    contents.update(collection="synthetic-collection", judge="synthetic-judge", mining="synthetic-mining")
                elif name == "evaluation_gate":
                    contents.update(decision="reject", eligible_for_promotion=False)
                path = target / artifact
                path.write_text(json.dumps(contents))
                return path

            with patch.object(loop, "ROOT", root), patch.object(loop, "validate_budget"), \
                    patch.object(loop, "ensure_gpu_idle"), patch.object(loop, "execute_stage", side_effect=execute):
                result = loop.iteration(run, args)
            self.assertEqual(seen[0][0], "feedback")
            collection = seen[0][1]
            for key, value in [("--checkpoint", parent), ("--parent-promotion", promotion),
                               ("--init-seed", 64000), ("--policy-seed", 74000)]:
                self.assertEqual(collection[collection.index(key) + 1], value)
            for name, arguments in seen:
                if name == "data_qualification" or name.startswith("train_"):
                    self.assertEqual(arguments[arguments.index("--parent-promotion") + 1], promotion)
            self.assertEqual(result["iteration"], 2)
            self.assertFalse(result["production_deployed"])


if __name__ == "__main__":
    unittest.main()
