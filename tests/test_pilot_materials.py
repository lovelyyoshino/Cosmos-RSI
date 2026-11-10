"""开发素材必须有完整同初态配对及确切封存证据，不能按成败选片或冒充正式收益。

@author pony
@date 2026-11-09
@version v1.0.0
@last_modified 2026-11-09
@changelog
    - v1.0.0 (2026-11-09): 用显式合成夹具覆盖配对错绑、权限变化、等待失败与启动摘要漂移。
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
import p1_pilot_materials as materials


class PilotMaterialsTests(unittest.TestCase):
    def report(self):
        return {"schema": "thor-conservative-pilot-v1", "authority": {
            "development_only": True, "formal_gate_executed": False,
            "production_promotion": False, "real_robot_enabled": False},
            "complete_rsi": False, "causal_attribution_established": False,
            "training": dict.fromkeys(materials.TRAINING_ROLES),
            "development": dict.fromkeys(materials.ROLES)}

    def test_development_authority_and_all_roles_required(self):
        materials.check_authority(self.report())
        for mutation in ["promotion", "gate", "rsi", "cause", "training", "development"]:
            report = self.report()
            if mutation == "promotion": report["authority"]["production_promotion"] = True
            elif mutation == "gate": report["authority"]["formal_gate_executed"] = True
            elif mutation == "rsi": report["complete_rsi"] = True
            elif mutation == "cause": report["causal_attribution_established"] = True
            elif mutation == "training": report["training"].pop("analytic_ablation")
            else: report["development"].pop("control")
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                materials.check_authority(report)

    def episodes(self):
        rows = [{"episode_id": f"synthetic-{condition}-{i}", "condition_id": condition,
            "init_seed": materials.INITIAL_SEED + i, "policy_seed": materials.POLICY_SEED + i,
            "initial_scene_sha256": f"synthetic-scene-{condition}-{i}",
            "outcome": {"source": "env", "success": i != 0}}
            for condition in materials.CONDITIONS for i in [2, 1, 0]]
        return {role: copy.deepcopy(rows) for role in materials.ROLES}

    def test_selects_predeclared_first_seed_even_when_it_failed(self):
        pairs = materials.paired_cases(self.episodes())
        self.assertEqual([condition for condition, _ in pairs], materials.CONDITIONS)
        self.assertTrue(all(not row["outcome"]["success"] for _, rows in pairs for row in rows))

    def test_missing_duplicate_mismatched_scene_seed_or_absent_first_case_reject(self):
        for mutation in ["missing", "duplicate", "scene", "seed", "first"]:
            rows = self.episodes()
            if mutation == "missing": rows["control"].pop()
            elif mutation == "duplicate": rows["control"].append(copy.deepcopy(rows["control"][0]))
            elif mutation == "scene": rows["candidate"][0]["initial_scene_sha256"] = "wrong"
            elif mutation == "seed": rows["candidate"][0]["policy_seed"] += 10
            else: rows = {role: [r for r in values if r["init_seed"] != materials.INITIAL_SEED]
                          for role, values in rows.items()}
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): materials.paired_cases(rows)

    def wait_fixture(self, root):
        execution = root / "synthetic_exec"; execution.mkdir()
        log = execution / "pilot.log"; log.write_text("")
        command = ["synthetic-python", "scripts/p1_conservative_pilot.py", "--iteration-run", "synthetic-iteration"]
        launch = execution / "launch.json"
        launch.write_text(json.dumps({"pid": 12345, "command": command, "log": str(log)}))
        return execution, launch, log, command

    def test_exited_or_reused_pid_is_not_replaced_by_another_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            execution, _, _, _ = self.wait_fixture(root)
            for actual in [[], ["another-process"]]:
                with patch.object(materials, "ROOT", root), patch.object(materials, "read_process", return_value=actual), \
                        self.assertRaises(RuntimeError): materials.wait_for_pilot(execution, .1)

    def test_launch_mutation_during_wait_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            execution, launch, _, command = self.wait_fixture(root)
            def mutate(_): launch.write_text(launch.read_text() + " ")
            with patch.object(materials, "ROOT", root), patch.object(materials, "read_process", return_value=command), \
                    patch.object(materials.time, "sleep", side_effect=mutate), self.assertRaises(ValueError):
                materials.wait_for_pilot(execution, 1)

    def test_completed_exact_run_can_export_without_live_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            execution, _, log, _ = self.wait_fixture(root)
            run = root / "runs/p0/synthetic_pilot"; run.mkdir(parents=True)
            (run / "probe.json").write_text("{}")
            log.write_text("记录目录：" + str(run) + "\n")
            with patch.object(materials, "ROOT", root), patch.object(materials, "sealed", return_value=(self.report(), {})) \
                    as check, patch.object(materials, "read_process", side_effect=AssertionError("不能要求已完成进程存活")):
                self.assertEqual(materials.wait_for_pilot(execution, .1), run)
                check.assert_called_once_with(run / "pilot.json", "p1_conservative_pilot")

    def test_failed_seal_does_not_export_partial_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            execution, _, log, _ = self.wait_fixture(root)
            run = root / "runs/p0/synthetic_pilot"; run.mkdir(parents=True)
            (run / "probe.json").write_text("{}")
            log.write_text("记录目录：" + str(run) + "\n")
            with patch.object(materials, "ROOT", root), \
                    patch.object(materials, "sealed", side_effect=ValueError("显式合成失败阶段")), self.assertRaises(ValueError):
                materials.wait_for_pilot(execution, .1)


if __name__ == "__main__":
    unittest.main()
