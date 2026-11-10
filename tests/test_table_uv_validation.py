"""独立重渲染方法的完整控制、固定配方及依赖失效测试。

@author pony
@date 2026-11-08
@version v1.0.0
@last_modified 2026-11-08
@changelog
    - v1.0.0 (2026-11-08): 新图集方法全部错误控制、demo_26及来源依赖不可绕过。
"""
from __future__ import annotations

import copy
import pathlib
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from p0_table_uv_validation import CONFIG, ERRORS, METHOD_FILES, NEGATIVES, checked_config, checked_dependencies, validate_report


class TableUVValidationTests(unittest.TestCase):
    def fixture(self):
        return {"method": CONFIG["version"], "role": "design", "frames": list(range(93)),
                "validation_passed": True, "eligible_for_training": False,
                "negative_controls": {name: {"passed": True, "decoded_video_checked": True} for name in NEGATIVES},
                "positive_controls": {name: {"detected": True} for name in ERRORS}}

    def test_fixed_renderer_matches_validation_recipe_and_does_not_grant_training(self):
        checked_config()
        self.assertFalse(CONFIG["eligible_for_training"])
        self.assertEqual(CONFIG["heldout_episode_suffix"], "caddy-demo_26")

    def test_old_method_or_old_heldout_cannot_be_reused(self):
        for key, replacement in [("method", "thor-table-renderer-validation-v1.1"), ("frames", [0, 46, 92])]:
            wrong = {**self.fixture(), key: replacement}
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_report(wrong, "design")
        self.assertIn("wrong_atlas_binding", ERRORS)

    def test_complete_controls_required_even_if_summary_says_passed(self):
        value = self.fixture(); validate_report(value, "design")
        for group in ["negative_controls", "positive_controls"]:
            for name in value[group]:
                wrong = copy.deepcopy(value); wrong[group].pop(name)
                with self.subTest(missing=name), self.assertRaises(ValueError):
                    validate_report(wrong, "design")
        for key, replacement in [("frames", list(range(92))), ("role", "heldout"), ("eligible_for_training", True)]:
            wrong = {**value, key: replacement}
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_report(wrong, "design")
        for name in value["positive_controls"]:
            wrong = copy.deepcopy(value); wrong["positive_controls"][name]["detected"] = False
            with self.subTest(missed=name), self.assertRaises(ValueError):
                validate_report(wrong, "design")

    def test_every_method_dependency_is_bound_before_reading_heldout(self):
        hashes = {name: name for name in METHOD_FILES}
        with patch("p0_table_uv_validation.dependency_hashes", return_value=hashes):
            checked_dependencies({"source_files_sha256": hashes})
            for name in METHOD_FILES:
                wrong = hashes.copy(); wrong[name] = "changed"
                with self.subTest(changed=name), self.assertRaises(ValueError):
                    checked_dependencies({"source_files_sha256": wrong})


if __name__ == "__main__":
    unittest.main()
