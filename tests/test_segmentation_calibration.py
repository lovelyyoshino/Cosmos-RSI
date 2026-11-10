"""核对校准提示的标签和几何边界，防止把背景标成正点。

@author pony
@date 2026-11-07
@version v1.0.0
@last_modified 2026-11-07
@changelog
    - v1.0.0 (2026-11-07): 覆盖断连细长物体、全屏物体和空ROI。
"""
from __future__ import annotations

import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from p0_calibrate_segmentation import balanced_prompts, spread_points


class SegmentationCalibrationTests(unittest.TestCase):
    def test_disconnected_parts_have_positive_and_negative_points(self):
        mask = np.zeros((64, 64), dtype=bool)
        mask[3:22, 5:25] = True
        mask[49:52, 43:60] = True
        _, points, labels = balanced_prompts(np.asarray([mask]))
        self.assertEqual(len(points[0]), 24)
        self.assertEqual(labels[0].count(1), 12)
        self.assertEqual(labels[0].count(0), 12)
        for (x, y), label in zip(points[0], labels[0]):
            self.assertEqual(bool(mask[y, x]), label == 1)
        self.assertTrue(any(y >= 49 and label == 1 for (_, y), label in zip(points[0], labels[0])))

    def test_full_frame_roi_pads_ignored_points_instead_of_false_negatives(self):
        mask = np.ones((32, 32), dtype=bool)
        _, points, labels = balanced_prompts(np.asarray([mask]))
        self.assertEqual(labels[0], [1] * 12 + [-1] * 12)
        self.assertEqual(len(points[0]), 24)

    def test_single_pixel_and_empty_regions(self):
        mask = np.zeros((32, 32), dtype=bool)
        mask[8, 9] = True
        self.assertEqual(spread_points(mask, 12), [[9, 8]] * 12)
        self.assertEqual(spread_points(np.zeros_like(mask), 12), [])
        with self.assertRaises((ValueError, RuntimeError)):
            balanced_prompts(np.asarray([np.zeros_like(mask)]))


if __name__ == "__main__":
    unittest.main()
