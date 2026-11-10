"""固定桌面纹理诊断的边界、时序、错误控制与配方回归检查。

@author pony
@date 2026-11-08
@version v1.0.0
@last_modified 2026-11-08
@changelog
    - v1.0.0 (2026-11-08): 覆盖画面边缘、真实非桌体保护、静态纹理及视频像素错误。
"""
from __future__ import annotations

import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from p0_table_texture_design import checked_config, compose_texture, editable_region, largest_rectangle, pixel_report, static_texture


class TableTextureDesignTests(unittest.TestCase):
    def fixture(self):
        source = np.full((3, 80, 80, 3), [160, 100, 70], dtype=np.uint8)
        raw = np.ones((3, 80, 80), dtype=bool)
        raw[:, :12] = False
        raw[:, 35:45, 35:45] = False
        source[~raw] = [220, 50, 10]
        yy, xx = np.indices((80, 80))
        native = np.repeat(((xx % 7) * 15 + yy % 3)[..., None], 3, axis=-1).astype(np.uint8)
        return source, native, raw

    def test_predeclared_config_never_grants_training(self):
        self.assertFalse(checked_config()["eligible_for_training"])

    def test_largest_rectangle_is_inside_irregular_mask_and_ties_are_stable(self):
        mask = np.zeros((7, 10), dtype=bool)
        mask[1:4, 1:4] = True; mask[4:7, 6:9] = True
        self.assertEqual(largest_rectangle(mask), (1, 1, 4, 4))
        mask[1:4, 1:5] = True
        self.assertEqual(largest_rectangle(mask), (1, 1, 4, 5))
        for bad in [np.zeros((5, 5), bool), mask.astype(np.uint8), mask[None]]:
            with self.assertRaises(ValueError):
                largest_rectangle(bad)

    def test_real_occlusion_halo_is_preserved_but_image_edge_is_editable(self):
        _, _, raw = self.fixture()
        editable, alpha = editable_region(raw)
        self.assertFalse(editable[:, 27:53, 27:53].any())
        self.assertTrue(editable[:, 70:, :8].all())
        self.assertTrue((alpha[:, 70:, :8] == 1).all())
        self.assertTrue((alpha[~editable] == 0).all())
        self.assertTrue((alpha >= 0).all() and (alpha <= 1).all())

    def test_whole_protected_region_and_input_bytes_are_preserved(self):
        source, native, raw = self.fixture()
        original = source.copy()
        output, editable, _, factor, extraction = compose_texture(source, native, raw)
        np.testing.assert_array_equal(source, original)
        np.testing.assert_array_equal(output[~editable], source[~editable])
        self.assertTrue(np.any(output[editable] != source[editable]))
        self.assertGreaterEqual(factor.min(), .84 - 1e-6)
        self.assertLessEqual(factor.max(), 1.16 + 1e-6)
        self.assertGreater(extraction["highpass_rms"], 0)

    def test_static_source_and_mask_produce_identical_derived_frames(self):
        source, native, raw = self.fixture()
        output, *_ = compose_texture(source, native, raw)
        np.testing.assert_array_equal(output[0], output[1])
        np.testing.assert_array_equal(output[1], output[2])

    def test_cosmos_texture_has_actual_effect_and_constant_texture_stays_finite(self):
        source, native, raw = self.fixture()
        actual, *_ = compose_texture(source, native, raw)
        neutral, *_ = compose_texture(source, np.full_like(native, 100), raw)
        self.assertTrue(np.any(actual != neutral))
        factor, _ = static_texture(np.full_like(native, 100), editable_region(raw)[0].all(axis=0))
        np.testing.assert_allclose(factor, 1)

    def test_rgb_changes_and_geometry_corruption_fail_pixel_certificate(self):
        source, native, raw = self.fixture()
        output, editable, *_ = compose_texture(source, native, raw)
        masks = np.zeros((3, 3, 80, 80), dtype=np.uint8)
        masks[:, 0, 36:44, 36:44] = 1
        passed = pixel_report(source, output, output, editable, masks)
        self.assertTrue(passed["formula_and_protection_passed"])
        self.assertFalse(passed["training_eligibility_proved"])
        for y, x in [(0, 0), (38, 38), (72, 72)]:
            wrong = output.copy(); wrong[1, y, x] ^= 1
            self.assertFalse(pixel_report(source, output, wrong, editable, masks)["formula_and_protection_passed"])
        with self.assertRaises(ValueError):
            pixel_report(source, output, output[:-1], editable, masks)

    def test_invalid_regions_or_rgb_are_rejected(self):
        source, native, raw = self.fixture()
        for images, region in [(source.astype(np.float32), raw), (source, raw.astype(np.uint8)),
                               (source, raw[:1]), (source, np.zeros_like(raw)), (source, np.ones_like(raw))]:
            with self.assertRaises(ValueError):
                compose_texture(images, native, region)
        with self.assertRaises(ValueError):
            compose_texture(source, native.astype(np.float32), raw)


if __name__ == "__main__":
    unittest.main()
