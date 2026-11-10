"""检查生成ROI度量的平移敏感性及分割提示是否落在真实物体上。

@author pony
@date 2026-11-06
@version v1.3.0
@last_modified 2026-11-08
@changelog
    - v1.3.0 (2026-11-08): 原SAM2测量即使全绿，也不能给背景派生绕过独立门授训练资格。
    - v1.2.0 (2026-11-07): 覆盖真实视频/源摘要、路径边界、目标条件分区域及复核帧绑定。
    - v1.1.0 (2026-11-07): 拒绝源图、模型、阈值或覆盖帧不匹配的校准报告。
    - v1.0.0 (2026-11-06): 真实数组测试IoU、断连机器人提示和边缘漂移拒绝。
"""
from __future__ import annotations

import pathlib
import sys
import copy
import hashlib
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from p0_generated_window_gate import (mask_iou, prompted_regions, roi_edge_f1, validate_source_calibration,
                                      verified_generation_paths, review_bound_to_video, target_condition_review_passed,
                                      legacy_gate_decision)


class GenerationGateTests(unittest.TestCase):
    def test_legacy_pass_only_grants_native_eligibility_and_derived_needs_its_own_gate(self):
        self.assertTrue(legacy_gate_decision("transfer_smoke", [])["eligible_for_training"])
        self.assertFalse(legacy_gate_decision("transfer_smoke", ["fixture rejection"])["eligible_for_training"])
        for stage in ["background_composite", "table_background_composite"]:
            with self.subTest(stage=stage):
                decision = legacy_gate_decision(stage, [])
                self.assertTrue(decision["passed"])
                self.assertFalse(decision["eligible_for_training"])
        with self.assertRaises(ValueError):
            legacy_gate_decision("unknown", [])

    def test_source_and_actual_output_are_bound_before_model_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            directory = root / "runs/p0/generation"
            directory.mkdir(parents=True)
            source = root / "runs/p0/source.json"
            output = directory / "output.mp4"
            source.write_bytes(b"verified training source")
            output.write_bytes(b"actual generated video")
            digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            record = {"stage": "transfer_smoke", "status": "passed", "result": {
                "source_success_evidence": {"path": str(source), "sha256": digest(source)},
                "outputs": [str(output)], "guardrails_enabled": True},
                "artifacts_sha256": {"output.mp4": digest(output)}}
            with patch("p0_generated_window_gate.sha256", digest):
                self.assertEqual(verified_generation_paths(record, directory / "probe.json", root), (source, output))
                source.write_bytes(b"replaced source")
                with self.assertRaisesRegex(ValueError, "源证据"):
                    verified_generation_paths(record, directory / "probe.json", root)
                source.write_bytes(b"verified training source")
                output.write_bytes(b"replaced video")
                with self.assertRaisesRegex(ValueError, "视频"):
                    verified_generation_paths(record, directory / "probe.json", root)
                output.write_bytes(b"actual generated video")
                changed = copy.deepcopy(record)
                changed["result"]["guardrails_enabled"] = False
                with self.assertRaisesRegex(ValueError, "安全检查"):
                    verified_generation_paths(changed, directory / "probe.json", root)
                del changed["result"]["source_success_evidence"]["sha256"]
                with self.assertRaises(ValueError):
                    verified_generation_paths(changed, directory / "probe.json", root)

    def test_generation_output_cannot_escape_its_recorded_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            directory = root / "runs/p0/generation"
            directory.mkdir(parents=True)
            source = root / "runs/p0/source.json"
            output = root / "unrelated.mp4"
            source.write_bytes(b"source")
            output.write_bytes(b"other video")
            digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            record = {"stage": "transfer_smoke", "status": "passed", "result": {
                "source_success_evidence": {"path": str(source), "sha256": digest(source)},
                "outputs": [str(output)], "guardrails_enabled": True}, "artifacts_sha256": {}}
            with patch("p0_generated_window_gate.sha256", digest), self.assertRaisesRegex(ValueError, "目录"):
                verified_generation_paths(record, directory / "probe.json", root)

    def test_reviews_must_bind_the_exact_video_and_all_ordered_frames(self):
        record = {"source_probe_sha256": "probe", "output_video_sha256": "video",
                  "reviewed_frames": list(range(93)), "reviewer": "reviewer"}
        self.assertTrue(review_bound_to_video(record, "probe", "video", 93))
        for changed in [{"output_video_sha256": "other"}, {"source_probe_sha256": "other"},
                        {"reviewed_frames": list(range(93)) + [92]}, {"reviewed_frames": [0, 46, 92]},
                        {"reviewer": " "}]:
            with self.subTest(changed=changed):
                self.assertFalse(review_bound_to_video({**record, **changed}, "probe", "video", 93))

    def test_global_light_condition_cannot_pass_from_background_only(self):
        check = {"passed": True, "criterion": "预先声明的目标亮度与色温条件", "observations": "逐帧观测记录"}
        target = {"light": "warm_dim"}
        record = {"source_probe_sha256": "probe", "output_video_sha256": "video", "target_condition": target,
                  "reviewed_frames": list(range(93)), "reviewer": "reviewer", "passed": True,
                  "region_checks": {name: dict(check) for name in ["robot", "target_objects", "background"]}}
        self.assertTrue(target_condition_review_passed(record, target, "probe", "video", 93))
        for key in ["robot", "target_objects", "background"]:
            changed = copy.deepcopy(record)
            del changed["region_checks"][key]
            with self.subTest(key=key):
                self.assertFalse(target_condition_review_passed(changed, target, "probe", "video", 93))
        for changed in [{"target_condition": {"light": "another"}}, {"passed": False}, {"region_checks": None}]:
            self.assertFalse(target_condition_review_passed({**record, **changed}, target, "probe", "video", 93))
        self.assertFalse(target_condition_review_passed(record, None, "probe", "video", 93))

    def test_background_condition_also_requires_protected_foreground_review(self):
        target = {"texture": "dark_matte_stone"}
        check = {"passed": True, "criterion": "目标材质且前景不改", "observations": "全帧源与生成对照"}
        record = {"source_probe_sha256": "probe", "output_video_sha256": "video", "target_condition": target,
                  "reviewed_frames": list(range(93)), "reviewer": "reviewer", "passed": True,
                  "region_checks": {name: dict(check) for name in ["background", "protected_foreground"]}}
        self.assertTrue(target_condition_review_passed(record, target, "probe", "video", 93))
        record["region_checks"]["protected_foreground"]["passed"] = False
        self.assertFalse(target_condition_review_passed(record, target, "probe", "video", 93))

    def test_combined_conditions_require_the_union_of_region_checks(self):
        target = {"light": "warm_dim", "background": "dark_stone"}
        check = {"passed": True, "criterion": "组合条件及非目标身份保持", "observations": "全帧源与生成对照"}
        record = {"source_probe_sha256": "probe", "output_video_sha256": "video", "target_condition": target,
                  "reviewed_frames": list(range(93)), "reviewer": "reviewer", "passed": True,
                  "region_checks": {name: dict(check) for name in ["robot", "target_objects", "background"]}}
        self.assertFalse(target_condition_review_passed(record, target, "probe", "video", 93))
        record["region_checks"]["protected_foreground"] = dict(check)
        self.assertTrue(target_condition_review_passed(record, target, "probe", "video", 93))

    def test_iou_detects_shift_and_empty_masks(self):
        first = np.zeros((64, 64), dtype=bool)
        first[10:30, 10:30] = True
        self.assertEqual(mask_iou(first, first), 1)
        self.assertEqual(mask_iou(first, np.roll(first, 30, axis=1)), 0)
        self.assertEqual(mask_iou(np.zeros_like(first), np.zeros_like(first)), 0)

    def test_positive_prompts_cover_disconnected_robot_parts(self):
        mask = np.zeros((64, 64), dtype=bool)
        mask[5:20, 5:20] = True
        mask[32:42, 40:48] = True
        boxes, points = prompted_regions(np.asarray([mask]))
        self.assertEqual(len(boxes), 1)
        self.assertTrue(all(mask[y, x] for x, y in points[0]))
        self.assertTrue(any(x >= 40 for x, y in points[0]))

    def test_same_edges_pass_and_large_geometry_shift_does_not(self):
        mask = np.zeros((64, 64), dtype=bool)
        mask[10:30, 10:30] = True
        first = np.zeros((64, 64, 3), dtype=np.uint8)
        first[mask] = 220
        self.assertGreater(roi_edge_f1(first, first, mask, 2), 0.99)
        self.assertLess(roi_edge_f1(first, np.roll(first, 25, axis=1), mask, 2), 0.45)

    def test_calibration_is_bound_to_source_model_and_unchanged_threshold(self):
        report = {"source_input_sha256": "source", "sam_model_manifest_sha256": "model",
                  "gate_config_sha256": "config", "threshold": 0.75, "all_frames": True,
                  "frames": list(range(93)), "balanced_calibration_passed": True,
                  "summary": {"balanced12": {"robot": {"visible_frames": 93, "min_iou": 0.81,
                                                         "below_threshold_frames": []}}}}
        validate_source_calibration(report, "source", "model", "config", ["robot"], 93, 0.75)
        for field, bad in [("source_input_sha256", "other"), ("sam_model_manifest_sha256", "other"),
                           ("gate_config_sha256", "other"), ("threshold", 0.70), ("frames", [0]),
                           ("balanced_calibration_passed", False)]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_source_calibration({**report, field: bad}, "source", "model", "config", ["robot"], 93, 0.75)


if __name__ == "__main__":
    unittest.main()
