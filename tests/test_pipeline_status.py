"""统一状态不能通过配置声明、损坏产物或P0训练冒烟伪造正式RSI完成。

@author pony
@date 2026-11-09
@version v1.3.0
@last_modified 2026-11-09
@changelog
    - v1.3.0 (2026-11-09): 验证当前生产晋升链、历史记录及断链/失效证明的拒绝。
    - v1.2.0 (2026-11-09): 多状态像素诊断完成也不能变成Predict规划资格。
    - v1.1.0 (2026-11-09): 正式训练须三角色匹配；历史门禁或统计拒绝不能显示为本圈通过。
    - v1.0.1 (2026-11-09): CPU备份机仅替换等价字节SHA函数，兼容Python3.10的测试环境。
"""
import hashlib
import json
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from p0_pipeline_status import anchor_feedback, evidence_for, promotion_history, summarize_readiness


class PipelineStatusTests(unittest.TestCase):
    def setUp(self):
        # 生产环境Python3.12不变；测试仍对真实文件字节计算完整SHA256。
        checksum = patch("p0_pipeline_status.sha256", lambda path: hashlib.sha256(path.read_bytes()).hexdigest())
        checksum.start()
        self.addCleanup(checksum.stop)

    def test_only_matching_attested_artifacts_are_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            run = root / "runs/p0/test_stage_20261109"
            run.mkdir(parents=True)
            artifact = run / "report.json"
            artifact.write_text(json.dumps({"role": "candidate"}))
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
            (run / "probe.json").write_text(json.dumps({"stage": "test_stage", "status": "passed",
                "artifacts_sha256": {"report.json": digest}}))
            spec = {"stage": "test_stage", "glob": "test_stage_20*/report.json", "require": {"role": "candidate"}}
            self.assertTrue(evidence_for(root, spec)["artifact_verified"])
            self.assertFalse(evidence_for(root, {**spec, "require": {"role": "control"}})["artifact_verified"])
            artifact.write_text(json.dumps({"role": "candidate", "claim": "tampered"}))
            result = evidence_for(root, spec)
            self.assertFalse(result["artifact_verified"])
            self.assertEqual(len(result["invalid_reports"]), 1)

    def test_p0_budget_and_model_smoke_cannot_become_recursive_improvement(self):
        def evidence(data, digest="abc"):
            return {"latest_valid": {"data": data, "sha256": digest}}
        values = {"feedback": evidence({"sample_count": 30}),
            "reason": evidence({"collection_sha256": "abc", "calibration": {"calibration_passed": False}}),
            "mining": evidence({"collection_sha256": "abc", "judge_sha256": "abc", "generation_routes": 0}),
            "candidate_budget": evidence({"paired_plan_sha256": "plan", "parent_ckpt_sha256": "parent"}),
            "control_budget": evidence({"paired_plan_sha256": "plan", "parent_ckpt_sha256": "parent"}),
            "predict": evidence({"advisory_only": True})}
        result = summarize_readiness(values, "pending")
        self.assertTrue(result["feedback_reports_linked"])
        self.assertTrue(result["matched_budget_artifacts"])
        self.assertTrue(result["predict_interface_verified"])
        self.assertFalse(result["complete_rsi"])
        self.assertFalse(result["formal_gate_completed"])
        self.assertFalse(result["reason_calibration_passed"])
        self.assertEqual(result["promoted_iterations"], 0)
        values["mining"]["latest_valid"]["data"]["judge_sha256"] = "wrong"
        self.assertFalse(summarize_readiness(values, "pending")["feedback_reports_linked"])

    def test_formal_training_requires_matching_roles_and_gate_source(self):
        values = {}
        for role in ["candidate", "control", "analytic_ablation"]:
            values["formal_" + role] = {"latest_valid": {"sha256": role + "-report", "data": {
                "schema": "thor-feedback-policy-update-v1", "training_performed": True,
                "eligible_for_evaluation": True, "parent_ckpt_sha256": "parent", "paired_plan_sha256": "plan",
                "iteration": 1, "steps": 1000, "batch": 8, "seed": 42, "output_ckpt_sha256": role + "-weights"}}}
        self.assertTrue(summarize_readiness(values, "pending")["formal_three_role_training_completed"])
        gate = {"schema": "thor-rsi-evaluation-gate-v1", "iteration": 1,
                "candidate_ckpt_sha256": "candidate-weights", "parent_ckpt_sha256": "parent",
                "inputs": {role + "_training": {"sha256": role + "-report"} for role in ["candidate", "control"]},
                "decision": "reject", "statistical_gate_passed": False, "eligible_for_promotion": False}
        values["formal_gate"] = {"latest_valid": {"data": gate}}
        readiness = summarize_readiness(values, "pending")
        self.assertTrue(readiness["formal_gate_completed"])
        self.assertFalse(readiness["formal_gate_passed"])
        self.assertFalse(readiness["complete_rsi"])
        gate.update(decision="accept_pending_approval", statistical_gate_passed=True, eligible_for_promotion=True)
        self.assertTrue(summarize_readiness(values, "pending")["formal_gate_passed"])
        gate["candidate_ckpt_sha256"] = "historical-weights"
        self.assertFalse(summarize_readiness(values, "pending")["formal_gate_completed"])
        gate["candidate_ckpt_sha256"] = "candidate-weights"
        values["formal_control"]["latest_valid"]["data"]["seed"] = 43
        readiness = summarize_readiness(values, "pending")
        self.assertFalse(readiness["formal_three_role_training_completed"])
        self.assertFalse(readiness["formal_gate_completed"])

    def test_later_research_reports_cannot_replace_main_feedback_chain(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            data, specifications = {"artifacts_sha256": {}}, {}
            for name, key, stage in [("feedback", "collection", "feedback_rollout"),
                                     ("reason", "judge", "feedback_judge"), ("mining", "mining", "feedback_mine")]:
                run = root / "runs/p0" / stage
                run.mkdir(parents=True)
                path = run / "report.json"
                path.write_text(json.dumps({"source": "synthetic-main-feedback"}))
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                (run / "probe.json").write_text(json.dumps({"stage": stage, "status": "passed",
                    "artifacts_sha256": {"report.json": digest}}))
                data[key] = str(path)
                data["artifacts_sha256"][str(path)] = digest
                specifications[name] = {"stage": stage, "glob": "unused"}
            evidence = {"feedback_chain": {"latest_valid": {"data": data}},
                        "reason": {"latest_valid": {"data": {"source": "synthetic-later-research"}}}}
            anchor_feedback(root, evidence, specifications)
            self.assertEqual(evidence["reason"]["latest_valid"]["data"]["source"], "synthetic-main-feedback")
            path.write_text("changed after sealing")
            with self.assertRaises(ValueError):
                anchor_feedback(root, evidence, specifications)

    def test_multistate_pixel_diagnostic_cannot_grant_planning_authority(self):
        values = {"predict_multistate": {"latest_valid": {"data": {
            "schema": "thor-predict-validation-v1", "diagnostic_complete": True,
            "summary": {"states": 3, "eligible_for_planning": False, "eligible_for_training": False,
                        "prediction_beats_copy_on_mean_pixel_error": True}}}}}
        result = summarize_readiness(values, "pending")
        self.assertTrue(result["predict_multistate_diagnostic_completed"])
        self.assertFalse(result["predict_validated_for_planning"])
        self.assertFalse(result["complete_rsi"])
        values["predict_multistate"]["latest_valid"]["data"]["summary"]["eligible_for_planning"] = True
        self.assertFalse(summarize_readiness(values, "pending")["predict_multistate_diagnostic_completed"])

    def synthetic_promotions(self, root):
        """仅为文件契约夹具；已有独立测试负责真实门禁、审批和权重校验。"""
        checksum = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        records, gates, trainings, checkpoints = [], [], [], []
        for iteration in [1, 2]:
            run = root / "runs/p0" / f"p1_promotion_20261109_synthetic_{iteration}"
            run.mkdir(parents=True)
            checkpoint = root / "checkpoints/promoted" / f"synthetic_iter{iteration}"
            checkpoint.mkdir(parents=True)
            training = run / "synthetic_training.json"
            training.write_text(json.dumps({"iteration": iteration, "output_ckpt_sha256": f"weights{iteration}",
                "parent_ckpt_sha256": f"weights{iteration - 1}", "data_qualification": {
                    "parent_promotion": {"path": str(records[-1]), "sha256": checksum(records[-1])} if records else None}}))
            gate = run / "synthetic_gate.json"
            gate.write_text(json.dumps({"statistical_gate_passed": True, "frozen_eval_sha256": "synthetic-frozen",
                "parent_ckpt_sha256": f"weights{iteration - 1}", "inputs": {
                    "candidate_training": {"path": str(training), "sha256": checksum(training)}}}))
            record = run / "promotion.json"
            record.write_text(json.dumps({"iteration": iteration, "gate": str(gate),
                "policy_ckpt_sha256": f"weights{iteration}", "parent_ckpt_sha256": f"weights{iteration - 1}",
                "promoted_checkpoint": str(checkpoint)}))
            records.append(record); gates.append(gate); trainings.append(training); checkpoints.append(checkpoint)
        (root / "checkpoints/current").symlink_to(checkpoints[-1])
        for name, implementation in [
                ("checked_promotion", lambda path: json.loads(path.read_text())),
                ("sealed", lambda path, stage: (json.loads(path.read_text()), {})),
                ("checked_update", lambda path, role: (json.loads(path.read_text()), {}))]:
            mocked = patch("p1_eval_gate." + name, side_effect=implementation)
            mocked.start(); self.addCleanup(mocked.stop)
        return records, gates, trainings, checkpoints

    def test_current_two_round_chain_counts_real_bound_parent_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            self.synthetic_promotions(root)
            history = promotion_history(root)
            self.assertTrue(history["current_production_verified"])
            self.assertEqual(history["promoted_iterations"], 2)
            self.assertEqual([row["iteration"] for row in history["current_chain"]], [1, 2])
            status = summarize_readiness({}, "implemented", promotions=history)
            self.assertEqual(status["promoted_iterations"], 2)
            self.assertFalse(status["complete_rsi"])
            self.assertIn("最后一圈追加训练种子复验与人工介入完整登记", status["remaining"])
            history["current_production_verified"] = False
            self.assertEqual(summarize_readiness({}, "implemented", promotions=history)["promoted_iterations"], 0)

    def test_historical_promotions_do_not_count_as_current_recursive_iterations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            _, _, _, checkpoints = self.synthetic_promotions(root)
            slot = root / "checkpoints/current"
            slot.unlink()
            result = promotion_history(root)
            self.assertEqual(result["verified_promotion_records"], 2)
            self.assertEqual(result["promoted_iterations"], 0)
            slot.symlink_to(checkpoints[0])
            self.assertEqual(promotion_history(root)["promoted_iterations"], 1)
            slot.unlink(); slot.symlink_to(root / "missing-checkpoint")
            self.assertFalse(promotion_history(root)["current_production_verified"])

    def test_changed_inputs_ambiguous_current_and_disconnected_chain_reject(self):
        for issue in ["changed_input", "changed_parent_proof", "wrong_parent", "gap", "changed_frozen",
                      "unpassed_gate", "duplicate_current_record", "rejected_promotion"]:
            with self.subTest(issue=issue), tempfile.TemporaryDirectory() as temporary:
                root = pathlib.Path(temporary)
                records, gates, trainings, _ = self.synthetic_promotions(root)
                if issue == "changed_input":
                    trainings[-1].write_text("tampered synthetic input")
                elif issue == "duplicate_current_record":
                    duplicate = root / "runs/p0/p1_promotion_20261109_synthetic_duplicate"
                    duplicate.mkdir(); (duplicate / "promotion.json").write_bytes(records[-1].read_bytes())
                elif issue == "rejected_promotion":
                    # 保持对未封存/未审批记录的既有正式验证接口，而不是读取JSON就计数。
                    mocked = patch("p1_eval_gate.checked_promotion", side_effect=ValueError("synthetic unsealed"))
                    mocked.start(); self.addCleanup(mocked.stop)
                elif issue in {"changed_frozen", "unpassed_gate"}:
                    gate = json.loads(gates[-1].read_text())
                    gate["frozen_eval_sha256" if issue == "changed_frozen" else "statistical_gate_passed"] = \
                        "other-frozen" if issue == "changed_frozen" else False
                    gates[-1].write_text(json.dumps(gate))
                else:
                    training = json.loads(trainings[-1].read_text())
                    if issue == "changed_parent_proof":
                        training["data_qualification"]["parent_promotion"]["sha256"] = "stale"
                    elif issue == "wrong_parent":
                        training["parent_ckpt_sha256"] = "other-parent"
                    else:
                        training["iteration"] = 3
                    trainings[-1].write_text(json.dumps(training))
                    gate = json.loads(gates[-1].read_text())
                    record = json.loads(records[-1].read_text())
                    if issue == "wrong_parent":
                        gate["parent_ckpt_sha256"] = record["parent_ckpt_sha256"] = "other-parent"
                    elif issue == "gap":
                        record["iteration"] = 3
                    records[-1].write_text(json.dumps(record))
                    gate["inputs"]["candidate_training"]["sha256"] = hashlib.sha256(trainings[-1].read_bytes()).hexdigest()
                    gates[-1].write_text(json.dumps(gate))
                result = promotion_history(root)
                self.assertEqual(result["promoted_iterations"], 0)
                self.assertFalse(result["current_production_verified"])
                self.assertIsNotNone(result["current_chain_error"])


if __name__ == "__main__":
    unittest.main()
