"""桌体重渲染设计的资源独占性和完整非桌体像素保护检查。

@author pony
@date 2026-11-08
@version v1.0.0
@last_modified 2026-11-08
@changelog
    - v1.0.0 (2026-11-08): 拒绝共享材质/纹理、错误掩码和前景改写。
"""
from __future__ import annotations

import pathlib
import sys
import types
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from p0_table_renderer_design import checked_config, protected_composite, texture_plan


class TableRendererDesignTests(unittest.TestCase):
    def model(self):
        return types.SimpleNamespace(geom_matid=np.array([0, 1, 2]),
            mat_texid=np.array([[0, -1], [1, -1], [2, -1]]), tex_type=np.zeros(3, int),
            tex_width=np.full(3, 16), tex_height=np.full(3, 32), tex_adr=np.array([0, 1536, 3072]),
            tex_nchannel=np.full(3, 3))

    def test_predeclared_diagnostic_does_not_grant_training(self):
        self.assertFalse(checked_config()["eligible_for_training"])

    def test_table_material_and_texture_are_exclusive(self):
        plan = texture_plan(self.model(), 1)
        self.assertEqual((plan["material_id"], plan["texture_id"], plan["bytes"]), (1, 1, 1536))
        for kind in ["shared_material", "shared_texture", "multiple_textures", "cube_texture"]:
            model = self.model()
            if kind == "shared_material":
                model.geom_matid[2] = 1
            elif kind == "shared_texture":
                model.mat_texid[2, 0] = 1
            elif kind == "multiple_textures":
                model.mat_texid[1, 1] = 2
            else:
                model.tex_type[1] = 2
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                texture_plan(model, 1)

    def fixture(self):
        source = np.full((20, 20, 3), [200, 100, 70], np.uint8)
        render = np.full_like(source, 10)
        table = np.ones((20, 20), bool); table[:6] = False
        masks = np.zeros((3, 20, 20), bool); masks[0, :4, :4] = True
        return source, render, table, masks

    def test_every_non_table_pixel_is_kept_and_every_table_pixel_comes_from_render(self):
        source, render, table, masks = self.fixture()
        result = protected_composite(source, render, table, masks)
        np.testing.assert_array_equal(result[~table], source[~table])
        np.testing.assert_array_equal(result[table], render[table])
        np.testing.assert_array_equal(result[masks.any(axis=0)], source[masks.any(axis=0)])

    def test_empty_wrong_dtype_and_foreground_overlapping_masks_are_rejected(self):
        source, render, table, masks = self.fixture()
        for wrong in [np.zeros_like(table), np.ones_like(table), table.astype(np.uint8), np.roll(table, -6, axis=0)]:
            with self.subTest(dtype=str(wrong.dtype)), self.assertRaises(ValueError):
                protected_composite(source, render, wrong, masks)


if __name__ == "__main__":
    unittest.main()
