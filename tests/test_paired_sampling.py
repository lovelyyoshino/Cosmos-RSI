"""用纯CPU合成描述核验配对预算、部分episode边界和稳定摘要。

@author pony
@date 2026-11-08
@version v1.0.0
@last_modified 2026-11-08
@changelog
    - v1.0.0 (2026-11-08): 拒绝未准入/验证来源和错边界，检查逐槽补齐与源绝对帧映射。
"""
from __future__ import annotations

import copy
import hashlib
import json
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from p0_paired_sampling import build_paired_plan, paired_plan_sha256, valid_source_starts


def fixture(kind, sample_id, source_id, length, first, stop):
    item = {"kind": kind, "sample_id": sample_id, "source_episode_id": source_id,
            "source_length": length, "source_split": "rollout_train" if kind == "self_distill" else "demo_train",
            "is_eval": False, "window": {"start_frame": first, "end_frame": stop}}
    if kind != "demo":
        item["eligible_for_training"] = True
    return item


class PairedSamplingTests(unittest.TestCase):
    def setUp(self):
        self.demos = [fixture("demo", "fixture-d0-a", "fixture-source-a", 300, 0, 300),
                      fixture("demo", "fixture-d0-b", "fixture-source-b", 150, 10, 140)]
        self.extras = [fixture("generated", "fixture-g-a", "fixture-source-a", 300, 120, 213),
                       fixture("self_distill", "fixture-s-c", "fixture-source-c", 120, 10, 103)]
        self.options = {"steps": 5, "batch": 4, "horizon": 8, "seed": 42,
                        "quotas": {"generated": 5, "self_distill": 3}}

    def build(self, **changes):
        return build_paired_plan(self.demos, self.extras, **{**self.options, **changes})

    def test_exact_quota_and_each_slot_is_auditable(self):
        plan = self.build()
        self.assertEqual(plan["candidate_counts"], {"demo": 12, "generated": 5, "self_distill": 3})
        self.assertEqual(plan["control_counts"], {"demo": 12, "duplicate_fill": 8})
        self.assertEqual(len(plan["candidate"]), 20)
        self.assertEqual(len(plan["control"]), 20)
        self.assertFalse(plan["quality_reports_verified"])
        self.assertFalse(plan["training_performed"])
        for index, (candidate, control) in enumerate(zip(plan["candidate"], plan["control"])):
            self.assertEqual((candidate["step"], candidate["slot"]), (index // 4 + 1, index % 4))
            self.assertEqual((control["step"], control["slot"]), (candidate["step"], candidate["slot"]))
            self.assertEqual(candidate["seed"], 42)
            self.assertEqual(control["seed"], 42)
            if candidate["kind"] == "demo":
                self.assertEqual(candidate, control)
                self.assertIsNot(candidate["window"], control["window"])
            else:
                self.assertEqual(control["kind"], "duplicate_fill")
                self.assertEqual(control["source_split"], "demo_train")
                self.assertEqual(control["matched_candidate_sample_id"], candidate["sample_id"])
                self.assertEqual(control["matched_candidate_kind"], candidate["kind"])
                self.assertIn(control["sample_id"], {row["sample_id"] for row in self.demos})
            for row in [candidate, control]:
                self.assertFalse(row["is_eval"])
                self.assertGreaterEqual(row["source_frame"], row["window"]["start_frame"])
                self.assertEqual(row["source_frame"], row["window"]["start_frame"] + row["window_frame"])
                self.assertLessEqual(row["source_frame"] + row["horizon"], row["window"]["end_frame"])
                self.assertLessEqual(row["window"]["end_frame"], row["source_length"])

    def test_93_frame_window_has_86_starts_and_preserves_nonzero_source_offset(self):
        item = self.extras[0]
        starts = list(valid_source_starts(item, horizon=8))
        self.assertEqual(starts, list(range(120, 206)))
        self.assertEqual([start - 120 for start in starts], list(range(86)))
        plan = self.build(steps=100, batch=1, quotas={"generated": 100, "self_distill": 0})
        for row in plan["candidate"]:
            self.assertIn(row["source_frame"], starts)
            self.assertEqual(row["source_frame"] - row["window_frame"], 120)
        # 源末端刚好等于动作块长度时，只能选择这个起点，不能越入下一个回合。
        single = fixture("demo", "fixture-single", "fixture-end", 19, 11, 19)
        end_plan = build_paired_plan([single], [], steps=1, batch=1, horizon=8, seed=0,
                                     quotas={"generated": 0, "self_distill": 0})
        self.assertEqual(end_plan["candidate"][0]["source_frame"], 11)

    def test_stable_digest_covers_content_and_does_not_mutate_inputs(self):
        before = copy.deepcopy((self.demos, self.extras, self.options))
        first = self.build()
        second = build_paired_plan(list(reversed(self.demos)), list(reversed(self.extras)), **self.options)
        self.assertEqual(first, second)
        self.assertEqual(before, (self.demos, self.extras, self.options))
        content = {key: value for key, value in first.items() if key != "sha256"}
        expected = hashlib.sha256(json.dumps(content, ensure_ascii=False, sort_keys=True,
                                             separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        self.assertEqual(first["sha256"], expected)
        self.assertEqual(first["sha256"], paired_plan_sha256({**first, "sha256": "ignored"}))
        changed = copy.deepcopy(first)
        changed["control"][0]["source_frame"] += 1
        self.assertNotEqual(paired_plan_sha256(changed), first["sha256"])
        self.assertNotEqual(self.build(seed=43)["sha256"], first["sha256"])

    def test_zero_extra_quota_uses_identical_demo_samples(self):
        plan = build_paired_plan(self.demos, [], steps=2, batch=3, horizon=8, seed=0,
                                 quotas={"generated": 0, "self_distill": 0})
        self.assertEqual(plan["candidate"], plan["control"])
        self.assertEqual(plan["candidate_counts"], {"demo": 6})
        self.assertEqual(plan["control_counts"], {"demo": 6})

    def test_rejects_invalid_explicit_budget_and_seed(self):
        for changes in [{"steps": 0}, {"steps": True}, {"batch": 0}, {"horizon": 0}, {"horizon": True},
                        {"seed": True}, {"seed": -1}, {"seed": 2 ** 32}, {"seed": 1.5},
                        {"quotas": {}}, {"quotas": {"generated": 1}},
                        {"quotas": {"generated": 21, "self_distill": 0}},
                        {"quotas": {"generated": True, "self_distill": 0}},
                        {"quotas": {"generated": -1, "self_distill": 0}}]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.build(**changes)
        for demos, extras in [([], self.extras), (self.demos, []),
                              ([self.extras[0]], self.extras[1:]), (self.demos, [self.demos[0]])]:
            with self.subTest(demos=demos, extras=extras), self.assertRaises(ValueError):
                build_paired_plan(demos, extras, **self.options)

    def test_all_descriptions_reject_validation_eval_or_unqualified_sources(self):
        for list_name in ["demos", "extras"]:
            for change in [{"is_eval": True}, {"is_eval": 0}, {"source_split": "demo_validation"},
                           {"source_split": ["demo_train"]},
                           {"source_split": "target_perturbed"}, {"eligible_for_training": False}]:
                demos, extras = copy.deepcopy((self.demos, self.extras))
                selected = demos if list_name == "demos" else extras
                selected[0].update(change)
                with self.subTest(list_name=list_name, change=change), self.assertRaises(ValueError):
                    build_paired_plan(demos, extras, **self.options)
        for eligible in [None, False, 1, "true"]:
            extras = copy.deepcopy(self.extras)
            extras[0]["eligible_for_training"] = eligible
            with self.subTest(eligible=eligible), self.assertRaises(ValueError):
                build_paired_plan(self.demos, extras, **self.options)
        extras = copy.deepcopy(self.extras)
        extras[0].pop("eligible_for_training")
        with self.assertRaises(ValueError):
            build_paired_plan(self.demos, extras, **self.options)
        for key in ["is_eval", "source_split", "source_length", "window"]:
            extras = copy.deepcopy(self.extras)
            extras[0].pop(key)
            with self.subTest(missing=key), self.assertRaises(ValueError):
                build_paired_plan(self.demos, extras, **self.options)

    def test_rejects_invalid_lengths_window_bounds_and_field_types(self):
        mutations = [{"source_length": 0}, {"source_length": True}, {"source_length": 212},
                     {"source_length": 300.0}, {"window": {"start_frame": -1, "end_frame": 92}},
                     {"window": {"start_frame": 213, "end_frame": 120}},
                     {"window": {"start_frame": 120, "end_frame": 127}},
                     {"window": {"start_frame": 120.0, "end_frame": 213}},
                     {"window": {"start_frame": True, "end_frame": 213}},
                     {"window": {"start_frame": 120}}, {"window": [120, 213]},
                     {"sample_id": ""}, {"source_episode_id": "  fixture-source-a"},
                     {"kind": "duplicate_fill"}, {"unknown": True}]
        for mutation in mutations:
            extras = copy.deepcopy(self.extras)
            extras[0].update(mutation)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                build_paired_plan(self.demos, extras, **self.options)

    def test_rejects_duplicate_ids_or_conflicting_source_metadata(self):
        for change in [{"sample_id": self.demos[0]["sample_id"]}, {"source_length": 301},
                       {"source_split": "rollout_train"}]:
            extras = copy.deepcopy(self.extras)
            extras[0].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                build_paired_plan(self.demos, extras, **self.options)
        extras = copy.deepcopy(self.extras)
        extras[1]["source_split"] = "demo_train"
        with self.assertRaises(ValueError):
            build_paired_plan(self.demos, extras, **self.options)


if __name__ == "__main__":
    unittest.main()
