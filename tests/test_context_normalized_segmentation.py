"""背景隔离必须保留候选错误像素，完整正负控才可报告诊断通过。

@author pony
@date 2026-11-08
@version v1.1.0
@last_modified 2026-11-08
@changelog
    - v1.1.0 (2026-11-08): 单ROI上下文必须去除其他目标，保留本ROI候选错误且不改变输入。
    - v1.0.0 (2026-11-08): 覆盖输入不可变、位移真值、背景不变性及正控漏检拒绝。
"""
from __future__ import annotations

import copy
import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from p0_context_normalized_segmentation import BASE_VARIANTS, POSITIVES, box_occlusion, normalized_context, object_shift, roi_context_pairs, summarize
from p0_segmentation_controls import background_controls


class ContextNormalizedTests(unittest.TestCase):
    def setUp(self):
        self.source = np.full((64, 64, 3), 90, dtype=np.uint8)
        self.rois = np.zeros((2, 64, 64), dtype=bool)
        self.rois[0, 20:30, 20:26] = True
        self.rois[1, 40:45, 40:45] = True
        self.source[self.rois[0]] = [220, 20, 30]

    def test_context_is_invariant_to_background_but_preserves_actual_foreground_errors(self):
        before = self.source.copy()
        for actual in background_controls(self.source, self.rois).values():
            first, second, support = normalized_context(self.source, actual, self.rois)
            np.testing.assert_array_equal(first, second)
            np.testing.assert_array_equal(second[support], actual[support])
            np.testing.assert_array_equal(first[~support], 128)
        changed = self.source.copy(); changed[23, 22] = [0, 255, 0]
        first, second, _ = normalized_context(self.source, changed, self.rois)
        self.assertFalse(np.array_equal(first, second))
        np.testing.assert_array_equal(second[23, 22], changed[23, 22])
        np.testing.assert_array_equal(self.source, before)
        for bad in [self.rois.astype(float), np.zeros_like(self.rois)]:
            with self.assertRaises(ValueError):
                normalized_context(self.source, self.source, bad)

    def test_object_shift_uses_actual_rgb_and_saves_geometric_truth_without_mutation(self):
        before, mask = self.source.copy(), self.rois[0].copy()
        moved, truth = object_shift(self.source, mask, 12)
        expected = np.zeros_like(mask); expected[20:30, 32:38] = True
        np.testing.assert_array_equal(truth, expected)
        np.testing.assert_array_equal(moved[truth], np.tile([220, 20, 30], (int(truth.sum()), 1)))
        self.assertFalse(np.array_equal(moved[mask], self.source[mask]))
        np.testing.assert_array_equal(self.source, before)
        np.testing.assert_array_equal(self.rois[0], mask)

    def test_individual_context_does_not_reintroduce_other_rois_or_repair_candidate(self):
        original = self.source.copy()
        actual = self.source.copy(); actual[22, 22] = [10, 240, 30]
        pairs = roi_context_pairs(self.source, actual, self.rois, 2)
        self.assertEqual(len(pairs), 2)
        first, second, support = pairs[0]
        self.assertTrue(support[self.rois[0]].all())
        self.assertFalse(support[self.rois[1]].any())
        np.testing.assert_array_equal(first[self.rois[1]], 128)
        np.testing.assert_array_equal(second[22, 22], actual[22, 22])
        self.assertFalse(np.array_equal(first, second))
        np.testing.assert_array_equal(self.source, original)
        with self.assertRaises(ValueError):
            roi_context_pairs(self.source, actual, np.zeros_like(self.rois))

    def fixture_rows(self):
        variants = BASE_VARIANTS + POSITIVES[1:]
        names = ["robot", "black_book_1"]
        return [{"frame": i, "visible_rois": names, "variants": {variant: {name: {"iou": .9 if variant not in POSITIVES else .2,
                    "threshold": .8 if name == "robot" else .75,
                    "passed_original_roi": variant not in POSITIVES} for name in names} for variant in variants}}
                for i in range(93)]

    def test_occlusion_uses_a_rectangle_instead_of_revealing_the_old_silhouette(self):
        before = self.source.copy()
        output, region = box_occlusion(self.source, self.rois[0])
        self.assertTrue(region[12:38, 12:34].all())
        np.testing.assert_array_equal(output[region], 128)
        np.testing.assert_array_equal(output[~region], self.source[~region])
        np.testing.assert_array_equal(self.source, before)

    def test_partial_run_and_targeted_positive_miss_cannot_validate_method(self):
        rows = self.fixture_rows()
        self.assertTrue(summarize(rows, ["robot", "black_book_1"])["diagnostic_validation_passed"])
        self.assertFalse(summarize(rows[:1], ["robot", "black_book_1"])["diagnostic_validation_passed"])
        changed = copy.deepcopy(rows)
        changed[15]["variants"]["book_shift12"]["black_book_1"] = {"iou": .9, "threshold": .75, "passed_original_roi": True}
        self.assertFalse(summarize(changed, ["robot", "black_book_1"])["diagnostic_validation_passed"])
        changed = copy.deepcopy(rows); changed[15]["variants"]["book_shift12"].pop("black_book_1")
        with self.assertRaises(ValueError):
            summarize(changed, ["robot", "black_book_1"])
        changed = copy.deepcopy(rows)
        changed[15]["variants"]["analytic_background_gray"]["robot"] = {"iou": .7, "threshold": .8, "passed_original_roi": False}
        self.assertFalse(summarize(changed, ["robot", "black_book_1"])["diagnostic_validation_passed"])
        for value in [float("nan"), 1.1, -.1]:
            changed = copy.deepcopy(rows); changed[15]["variants"]["identity"]["robot"]["iou"] = value
            with self.assertRaises(ValueError):
                summarize(changed, ["robot", "black_book_1"])


if __name__ == "__main__":
    unittest.main()
