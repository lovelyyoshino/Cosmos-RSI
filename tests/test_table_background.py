"""桌面v2必须保留全部其他物体与边界，不能用方法负控代替真实训练资格。

@author pony
@date 2026-11-08
@version v2.1.1
@last_modified 2026-11-08
@changelog
    - v2.1.1 (2026-11-08): 用实际历史报告验证另选SAM、部分帧失败及摘要篡改不可绕过。
    - v2.1.0 (2026-11-08): 覆盖真实依赖文件变更拒绝，以及同视频明确失败与缺少复核的区别。
    - v2.0.0 (2026-11-08): 覆盖几何身份、非任务物体、侵入/篡改、时序错位及完整控制门。
"""
from __future__ import annotations

import copy
import hashlib
import json
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from p0_table_background import CONFIG, METHOD_FILES, checked_dependencies, compose_table, editable_table, pixel_certificate, select_table_geometries
from p0_table_background_controls import ERRORS, NEGATIVES, validate_report
from p0_table_background_gate import explicit_review_failures, prior_rejections


class TableBackgroundTests(unittest.TestCase):
    def test_any_direct_control_or_judgment_dependency_change_invalidates_reports(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            for name in METHOD_FILES:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(name)
            digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
            probe = {"source_files_sha256": {name: digest(root / name) for name in METHOD_FILES}}
            with patch("p0_table_background.ROOT", root), patch("p0_table_background.sha256", digest):
                checked_dependencies(probe)
                for name in ["scripts/p0_lossless_background_gate.py", "scripts/p0_context_normalized_segmentation.py",
                             "scripts/p0_segmentation_controls.py", "scripts/p0_background_preview_materials.py"]:
                    path = root / name
                    path.write_text("changed dependency")
                    with self.subTest(name=name), self.assertRaises(ValueError):
                        checked_dependencies(probe)
                    path.write_text(name)

    def test_bound_explicit_failures_cannot_be_rewritten_but_missing_reviews_can_be_added(self):
        binding = {"source_probe_sha256": "probe", "output_video_sha256": "video",
                   "reviewed_frames": list(range(93)), "reviewer": "fixture reviewer"}
        self.assertEqual(explicit_review_failures({}, "probe", "video"), [])
        cases = {"semantic_review": {"object_identity_preserved": False},
                 "condition_review": {"passed": False}, "temporal_review": {"no_flicker": False}}
        for name, values in cases.items():
            reviews = {name: {**binding, **values}}
            with self.subTest(name=name):
                self.assertEqual(explicit_review_failures(reviews, "probe", "video"), [name])
                self.assertEqual(explicit_review_failures(reviews, "other-probe", "video"), [])

    def test_partial_frame_failure_blocks_but_never_counts_as_full_review(self):
        from p0_generated_window_gate import review_bound_to_video

        binding = {"source_probe_sha256": "probe", "output_video_sha256": "video",
                   "reviewed_frames": [46], "reviewer": "fixture reviewer"}
        cases = {"semantic_review": {"object_identity_preserved": False},
                 "condition_review": {"region_checks": {"background": {"passed": False}}},
                 "temporal_review": {"boundary_seams_acceptable": False}}
        for name, values in cases.items():
            review = {**binding, **values}
            with self.subTest(name=name):
                self.assertEqual(explicit_review_failures({name: review}, "probe", "video"), [name])
                self.assertFalse(review_bound_to_video(review, "probe", "video", 93))
                for frames in [[], [-1], [93], [True], [1, 1], "46"]:
                    self.assertEqual(explicit_review_failures({name: {**review, "reviewed_frames": frames}},
                                                              "probe", "video"), [])

    def test_selecting_clean_sam_cannot_hide_other_bound_historical_failures(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            binding = {"source_probe_sha256": "probe", "output_video_sha256": "video",
                       "reviewed_frames": [46], "reviewer": "fixture reviewer"}

            def save(stage, suffix, reviews, video="video"):
                directory = root / "runs/p0" / f"{stage}_{suffix}"
                directory.mkdir(parents=True)
                report = {"generation_probe_sha256": "probe", "output_video_sha256": video}
                if stage == "generated_window_gate":
                    report.update({"semantic_review": reviews.get("semantic_review"),
                                   "target_condition_review": reviews.get("condition_review")})
                else:
                    report["reviews"] = reviews
                path = directory / "quality_gate.json"
                path.write_text(json.dumps(report))
                (directory / "probe.json").write_text(json.dumps({"stage": stage, "status": "passed",
                    "artifacts_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()}}))
                return path

            clean = save("generated_window_gate", "selected_clean", {})
            failed_sam = save("generated_window_gate", "earlier_failure",
                              {"semantic_review": {**binding, "object_identity_preserved": False}})
            failed_lossless = save("lossless_background_gate", "earlier_failure",
                                   {"temporal_review": {**binding, "no_flicker": False}})
            failed_table = save("table_background_gate", "earlier_failure",
                                {"condition_review": {**binding, "passed": False}})
            save("generated_window_gate", "other_video",
                 {"condition_review": {**binding, "output_video_sha256": "other", "passed": False}}, "other")
            with patch("p0_table_background_gate.ROOT", root):
                failures = prior_rejections(clean, "probe", "video")
                self.assertEqual({item["path"] for item in failures},
                                 {str(path) for path in [failed_sam, failed_lossless, failed_table]})
                failed_sam.write_text(failed_sam.read_text() + " ")
                with self.assertRaisesRegex(ValueError, "摘要"):
                    prior_rejections(clean, "probe", "video")

    def fixture(self):
        source = np.full((93, 256, 256, 3), 90, np.uint8)
        for frame in range(93):
            source[frame, :, :, 0] = frame
        elements = np.full((93, 256, 256), 7, np.int32)
        masks = np.zeros((93, 3, 256, 256), np.uint8)
        for i, (y, x) in enumerate([(20, 30), (80, 85), (165, 170)]):
            elements[:, y:y + 20, x:x + 20] = i + 10
            masks[:, i, y:y + 20, x:x + 20] = 1
        elements[:, 145:165, 40:65] = 55
        source[:, 145:165, 40:65] = [120, 110, 90]
        editable = editable_table(elements, [7], masks)
        native = np.full_like(source, 30)
        return source, native, masks, editable

    def test_table_is_selected_by_body_mesh_and_visual_group(self):
        table = {"id": 7, "body": "study_table", "mesh": "desk_vis", "group": 1}
        self.assertEqual(select_table_geometries([table, {**table, "id": 8, "body": "wall"}]), [7])
        for invalid in [[], [table, {**table, "id": 9}], [{**table, "group": 0}], [{**table, "mesh": "other"}]]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                select_table_geometries(invalid)

    def test_all_non_table_pixels_and_eight_pixel_boundaries_are_preserved(self):
        source, native, masks, editable = self.fixture()
        before = [v.copy() for v in [source, native, masks, editable]]
        output = compose_table(source, native, editable)
        certificate = pixel_certificate(source, native, editable, output, masks)
        self.assertTrue(certificate["geometry_passed"])
        self.assertNotIn("eligible_for_training", certificate)
        self.assertFalse(editable[:, 137:173, 32:73].any())
        self.assertFalse(editable[:, :8].any())
        np.testing.assert_array_equal(output[~editable], source[~editable])
        np.testing.assert_array_equal(output[editable], native[editable])
        for original, actual in zip(before, [source, native, masks, editable]):
            np.testing.assert_array_equal(original, actual)

    def test_cup_recolor_protected_pixel_and_wrong_native_are_rejected(self):
        source, native, masks, editable = self.fixture()
        output = compose_table(source, native, editable)
        for y, x in [(150, 50), (20, 30), (0, 0)]:
            bad = output.copy()
            bad[46, y, x, 1] ^= 1
            with self.subTest(pixel=(y, x)):
                self.assertFalse(pixel_certificate(source, native, editable, bad, masks)["geometry_passed"])
        self.assertFalse(pixel_certificate(source, native + 1, editable, output, masks)["geometry_passed"])

    def test_full_window_order_is_required(self):
        source, native, masks, editable = self.fixture()
        output = compose_table(source, native, editable)
        output[[0, 92]] = output[[92, 0]]
        self.assertFalse(pixel_certificate(source, native, editable, output, masks)["geometry_passed"])
        with self.assertRaises(ValueError):
            compose_table(source[:-1], native, editable)

    def test_task_intrusion_small_editable_area_and_non_boolean_mask_fail(self):
        source, native, masks, editable = self.fixture()
        elements = np.full((93, 256, 256), 7, np.int32)
        with self.assertRaisesRegex(ValueError, "覆盖"):
            editable_table(elements, [7], masks)
        with self.assertRaises(ValueError):
            editable_table(np.full_like(elements, 8), [7], masks)
        with self.assertRaises(ValueError):
            compose_table(source, native, editable.astype(np.uint8))
        invaded = editable.copy()
        invaded[:, 20, 30] = True
        with self.assertRaises(ValueError):
            pixel_certificate(source, native, invaded, source, masks)

    def test_complete_controls_include_mug_and_editable_file_rejections(self):
        report = {"method": CONFIG["version"], "role": "design", "validation_passed": True,
            "eligible_for_training": False, "frames": list(range(93)),
            "negative_controls": {name: {"passed": True, "decoded_video_checked": True} for name in NEGATIVES},
            "positive_controls": {name: {"detected": True} for name in ERRORS}}
        validate_report(report, "design")
        for change in ["mug_recolored", "wrong_editable_sha", "partial_window", "array_only", "eligible"]:
            bad = copy.deepcopy(report)
            if change in ERRORS:
                bad["positive_controls"][change]["detected"] = False
            elif change == "partial_window":
                bad["frames"] = [0]
            elif change == "array_only":
                bad["negative_controls"]["analytic_table_dim"]["decoded_video_checked"] = False
            else:
                bad["eligible_for_training"] = True
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_report(bad, "design")


if __name__ == "__main__":
    unittest.main()
