"""UV独立准入的完整帧、实际视频绑定及非几何拒绝回归。

@author pony
@date 2026-11-08
@version v1.0.0
@last_modified 2026-11-08
@changelog
    - v1.0.0 (2026-11-08): 阻止缺帧、错摘要、背景代替前景、语义失败及边界失败被放行。
"""
from __future__ import annotations

import copy
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from p0_uv_window_gate import CONFIG, review_checks


class UVWindowGateTests(unittest.TestCase):
    def fixture(self):
        binding = {"source_probe_sha256": "probe", "output_video_sha256": "video",
                   "reviewed_frames": list(range(93)), "reviewer": "Codex", "passed": True}
        return {"semantic_review": {**copy.deepcopy(binding), "object_identity_preserved": True,
                    "task_semantics_preserved": True, "unwanted_objects": False, "no_new_obstacles": True,
                    "observations": "全帧任务对象与源相同"},
                "condition_review": {**copy.deepcopy(binding), "target_condition": CONFIG["target_condition"],
                    "region_checks": {name: {"criterion": "固定桌面或保护区验收", "observations": "实际全帧检查", "passed": True}
                                      for name in ["background", "protected_foreground"]}},
                "temporal_review": {**copy.deepcopy(binding), "background_consistent": True,
                    "no_flicker": True, "boundary_seams_acceptable": True, "observations": "完整相邻帧时序检查"}}

    def test_complete_reviews_pass_and_input_is_unchanged(self):
        value = self.fixture(); original = copy.deepcopy(value)
        self.assertTrue(all(review_checks(value, "probe", "video").values()))
        self.assertEqual(value, original)

    def test_missing_frames_wrong_probe_and_wrong_actual_video_fail_every_review(self):
        for field, replacement in [("reviewed_frames", [0, 46, 92]), ("source_probe_sha256", "other"),
                                   ("output_video_sha256", "other"), ("reviewer", "")]:
            value = self.fixture()
            for row in value.values():
                row[field] = replacement
            with self.subTest(field=field):
                self.assertFalse(any(review_checks(value, "probe", "video").values()))

    def test_geometry_cannot_replace_identity_target_or_temporal_acceptance(self):
        for group, field, replacement, check in [
            ("semantic_review", "object_identity_preserved", False, "semantic_ok"),
            ("semantic_review", "unwanted_objects", True, "semantic_ok"),
            ("temporal_review", "no_flicker", False, "background_temporal_ok"),
            ("temporal_review", "boundary_seams_acceptable", False, "background_temporal_ok"),
            ("condition_review", "target_condition", {"light": "warm_dim_indoor"}, "condition_ok")]:
            value = self.fixture(); value[group][field] = replacement
            with self.subTest(group=group, field=field):
                self.assertFalse(review_checks(value, "probe", "video")[check])
        value = self.fixture(); del value["condition_review"]["region_checks"]["protected_foreground"]
        self.assertFalse(review_checks(value, "probe", "video")["condition_ok"])


if __name__ == "__main__":
    unittest.main()
