"""部署必须绑定实际训练角色、父权重、步数和封存元数据。

@author pony
@date 2026-11-08
@version v1.0.0
@last_modified 2026-11-08
@changelog
    - v1.0.0 (2026-11-08): 覆盖旧五步记录兼容及预算候选来源/参数篡改拒绝。
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import p0_augmented_deploy_smoke as deployment


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DeploymentLineageTests(unittest.TestCase):
    def fixture(self, root, budget=True):
        run = root / "runs/p0/new_train"; run.mkdir(parents=True)
        parent = root / "runs/p0/parent"; parent.mkdir()
        candidate = run / "checkpoint"; candidate.mkdir()
        (parent / "state.pt").write_bytes(b"parent weights")
        (candidate / "state.pt").write_bytes(b"candidate weights")
        (parent / "checkpoint.json").write_text(json.dumps({"step": 5000}))
        steps = 40 if budget else 5
        recipe = {"steps": steps, "batch": 8 if budget else 2, "seed": 42}
        metadata = {"state_sha256": digest(candidate / "state.pt"), "parent_ckpt_sha256": digest(parent / "state.pt"),
                    "parent_checkpoint": str(parent), "role": "candidate", "eligible_for_promotion": False,
                    "paired_plan_sha256": "plan", "step": 5000 + steps, "recipe": recipe}
        (candidate / "checkpoint.json").write_text(json.dumps(metadata))
        training = {"role": "candidate", "eligible_for_promotion": False, "output_ckpt": str(candidate),
                    "output_ckpt_sha256": metadata["state_sha256"], "parent_ckpt_sha256": metadata["parent_ckpt_sha256"],
                    "paired_plan_sha256": "plan", "batch": recipe["batch"], "seed": 42}
        if budget:
            training["steps"] = steps
        else:
            training.update(parent_checkpoint=str(parent), training_steps=steps, steps=[{"step": i} for i in range(steps)])
        report = run / ("train_run.json" if budget else "training_smoke.json")
        report.write_text(json.dumps(training))
        self.seal(run, report, budget)
        return report, candidate

    def seal(self, run, report, budget=True):
        (run / "probe.json").write_text(json.dumps({"status": "passed",
            "stage": "augmented_budget" if budget else "augmented_training_smoke", "artifacts_sha256": {
                report.name: digest(report), "checkpoint/checkpoint.json": digest(run / "checkpoint/checkpoint.json")}}))

    def test_actual_budget_and_old_smoke_record_have_distinct_valid_step_formats(self):
        for budget in [False, True]:
            with self.subTest(budget=budget), tempfile.TemporaryDirectory() as temporary:
                root = pathlib.Path(temporary); report, _ = self.fixture(root, budget)
                with patch.object(deployment, "ROOT", root), patch.object(deployment, "sha256", digest):
                    *_, conditions = deployment.checked_training(report)
                self.assertEqual(conditions["steps"], 40 if budget else 5)

    def test_mismatched_role_parent_steps_or_sealed_metadata_are_rejected(self):
        for corruption in ["role", "parent", "steps", "unsealed_metadata", "weights"]:
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as temporary:
                root = pathlib.Path(temporary); report, candidate = self.fixture(root)
                data = json.loads(report.read_text())
                if corruption == "role": data["role"] = "control"
                elif corruption == "parent": data["parent_ckpt_sha256"] = "other parent"
                elif corruption == "steps": data["steps"] = 41
                elif corruption == "weights": (candidate / "state.pt").write_bytes(b"replaced")
                else:
                    metadata = json.loads((candidate / "checkpoint.json").read_text())
                    metadata["step"] = 6000
                    (candidate / "checkpoint.json").write_text(json.dumps(metadata))
                report.write_text(json.dumps(data))
                if corruption != "unsealed_metadata": self.seal(report.parent, report)
                with patch.object(deployment, "ROOT", root), patch.object(deployment, "sha256", digest), self.assertRaises(ValueError):
                    deployment.checked_training(report)


if __name__ == "__main__":
    unittest.main()
