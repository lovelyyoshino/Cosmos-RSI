"""桌面小UV面片的纹理覆盖、平铺连续性及固定来源边界回归。

@author pony
@date 2026-11-08
@version v1.0.0
@last_modified 2026-11-08
@changelog
    - v1.0.0 (2026-11-08): 小面片细节覆盖、纯色消融、平铺接缝和无效输入拒绝。
"""
from __future__ import annotations

import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from p0_table_uv_atlas_design import checked_config, mirrored_atlas


class UVAtlasTests(unittest.TestCase):
    def test_constant_control_stays_constant_and_method_is_diagnostic(self):
        original = np.full((256, 256, 3), [88, 50, 30], np.uint8)
        actual = mirrored_atlas(original)
        self.assertEqual(actual.shape, (1024, 1024, 3))
        np.testing.assert_array_equal(actual, np.broadcast_to(original[0, 0], actual.shape))
        self.assertFalse(checked_config()["eligible_for_training"])

    def test_small_uv_panel_recovers_visible_texture_range(self):
        # 上游桌板只采样整张图集的一小块；用色阶检查其覆盖，不依赖主观画质分数。
        texture = np.broadcast_to(np.arange(256, dtype=np.uint8)[None, :, None], (256, 256, 3)).copy()
        before = texture.copy()
        atlas = mirrored_atlas(texture)
        panel = atlas[763:878, 31:105]
        self.assertGreater(int(np.ptp(panel[..., 0])), 140)
        np.testing.assert_array_equal(texture, before)

    def test_every_mirrored_tile_boundary_has_equal_rgb(self):
        texture = np.random.default_rng(7).integers(0, 256, (256, 256, 3), dtype=np.uint8)
        atlas = mirrored_atlas(texture)
        for boundary in range(128, 1024, 128):
            np.testing.assert_array_equal(atlas[:, boundary - 1], atlas[:, boundary])
            np.testing.assert_array_equal(atlas[boundary - 1], atlas[boundary])

    def test_wrong_format_and_unfixed_density_are_rejected(self):
        texture = np.zeros((256, 256, 3), np.uint8)
        for wrong in [texture.astype(np.float32), texture[:128], texture[..., 0]]:
            with self.assertRaises(ValueError):
                mirrored_atlas(wrong)
        with self.assertRaises(ValueError):
            mirrored_atlas(texture, repeats=16)


if __name__ == "__main__":
    unittest.main()
