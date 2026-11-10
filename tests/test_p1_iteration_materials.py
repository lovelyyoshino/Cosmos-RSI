"""正式素材拒绝缺步训练、不配对评测、失败整圈及按成败选片。

@author pony
@date 2026-11-09
@version v1.0.0
@last_modified 2026-11-09
@changelog
    - v1.0.0 (2026-11-09): 用显式合成夹具核验素材的完整性和选片边界。
"""
from __future__ import annotations

import copy
import json
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from p1_iteration_materials import ROLES, fixed_video_cases, training_trace, wait_for_iteration


class TrainingTraceTests(unittest.TestCase):
    def test_missing_duplicate_nonfinite_or_wrong_batch_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "synthetic_steps.jsonl"
            original = [{"step": i, "loss": .2, "grad_norm": .5, "wall_s": 1., "sample_slots": 8} for i in [1, 2]]
            report = {"steps": 2, "batch": 8}
            path.write_text("\n".join(map(json.dumps, original)))
            self.assertEqual(len(training_trace(path, report)), 2)
            variants = [original[:1], [original[0], original[0]]]
            for key, value in [("loss", float("nan")), ("wall_s", -1), ("sample_slots", 2)]:
                changed = copy.deepcopy(original)
                changed[0][key] = value
                variants.append(changed)
            for rows in variants:
                with self.subTest(rows=rows), self.assertRaises(ValueError):
                    path.write_text("\n".join(map(json.dumps, rows)))
                    training_trace(path, report)


class FixedSelectionTests(unittest.TestCase):
    def fixture(self):
        rows = [{"eval_id": f"synthetic-{condition}-{i}", "condition_id": f"t00-{condition}",
                 "initial_state_sha256": "synthetic-state", "init_seed": 1, "policy_seed": 2,
                 "eval_set_sha256": "synthetic-frozen", "outcome": {"success": bool(i)}}
                for condition in ["standard", "lighting", "background"] for i in [1, 0]]
        return {role: copy.deepcopy(rows) for role in ROLES}

    def test_selects_fixed_first_id_even_when_that_episode_failed(self):
        result = fixed_video_cases(self.fixture())
        self.assertEqual(len(result), 3)
        self.assertTrue(all(not row["outcome"]["success"] for _, rows in result for row in rows))

    def test_id_loss_duplicates_or_mismatched_state_reject(self):
        for mutation in ["missing", "duplicate", "state", "seed"]:
            values = self.fixture()
            if mutation == "missing": values["control"].pop()
            elif mutation == "duplicate": values["control"].append(copy.deepcopy(values["control"][0]))
            elif mutation == "state": values["candidate"][1]["initial_state_sha256"] = "wrong-state"
            else: values["candidate"][1]["policy_seed"] = 3
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                fixed_video_cases(values)


class CompletionBoundaryTests(unittest.TestCase):
    def test_engineering_failure_rejects_but_statistical_rejection_can_be_documented(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            for status in ["failed", "running"]:
                (root / "probe.json").write_text(json.dumps({"stage": "rsi_iteration", "status": status}))
                with self.subTest(status=status), self.assertRaises(ValueError):
                    wait_for_iteration(root, 1)
            (root / "probe.json").write_text(json.dumps({"stage": "rsi_iteration", "status": "passed",
                "result": {"decision": "reject"}}))
            wait_for_iteration(root, 1)


if __name__ == "__main__":
    unittest.main()
