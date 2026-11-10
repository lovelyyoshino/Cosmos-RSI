"""验证材质参考只改桌体及网络双控制的方向、权重和来源约束。

@author pony
@date 2026-11-08
@version v1.0.0
@last_modified 2026-11-08
@changelog
    - v1.0.0 (2026-11-08): 覆盖渐变范围、RGB增益、非法输入与预声明配置一致性。
"""
from __future__ import annotations

import pathlib
import sys
import hashlib
import json
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from p0_table_reference_input import checked_config, table_reference
from p0_table_reference_sample import check_control_weights, reference_control_weights
from p0_table_reference_materials import native_video


class TableReferenceTests(unittest.TestCase):
    def test_configuration_is_exactly_the_predeclared_recipe(self):
        self.assertFalse(checked_config()["eligible_for_training"])

    def test_reference_changes_only_editable_table_and_uses_fixed_rgb_gain(self):
        source = np.full((2, 40, 40, 3), [160, 100, 80], dtype=np.uint8)
        editable = np.zeros((2, 40, 40), dtype=bool)
        editable[:, 4:36, 4:36] = True
        original = source.copy()
        reference, alpha = table_reference(source, editable)
        np.testing.assert_array_equal(reference[~editable], source[~editable])
        np.testing.assert_array_equal(source, original)
        np.testing.assert_array_equal(reference[0, 20, 20], [88, 42, 28])
        self.assertEqual(alpha[0, 20, 20], 1)
        self.assertGreater(alpha[0, 4, 4], 0)
        self.assertLess(alpha[0, 4, 4], alpha[0, 5, 5])
        self.assertTrue(np.all(alpha[~editable] == 0))

    def test_static_visible_table_reference_is_static_across_frames(self):
        source = np.full((3, 32, 32, 3), 120, dtype=np.uint8)
        editable = np.zeros((3, 32, 32), dtype=bool)
        editable[:, 4:28, 4:28] = True
        result, _ = table_reference(source, editable)
        np.testing.assert_array_equal(result[0], result[1])
        np.testing.assert_array_equal(result[1], result[2])

    def test_empty_full_wrong_dtype_or_misaligned_regions_are_rejected(self):
        source = np.ones((2, 16, 16, 3), dtype=np.uint8)
        valid = np.zeros((2, 16, 16), dtype=bool)
        valid[:, 4:12, 4:12] = True
        missing_frame = valid.copy()
        missing_frame[1] = False
        for images, mask in [(source, np.zeros_like(valid)), (source, np.ones_like(valid)),
                             (source, valid.astype(np.uint8)), (source, valid[:1]),
                             (source.astype(np.float32), valid), (source, missing_frame)]:
            with self.subTest(shape=mask.shape), self.assertRaises(ValueError):
                table_reference(images, mask)

    def test_edge_is_released_only_on_table_and_vis_stays_at_declared_weight(self):
        mask = np.zeros((93, 16, 16), dtype=bool)
        mask[:, :, :8] = True
        actual = reference_control_weights(mask, (4, 1, 24, 4, 4, 1))
        report = check_control_weights(actual, actual)
        self.assertEqual(report["edge_zero_fraction"], .5)
        np.testing.assert_array_equal(actual[2:], 0)
        np.testing.assert_allclose(actual[1], .8)
        np.testing.assert_allclose(actual[0, :, :, :, :2], .2)
        np.testing.assert_allclose(actual[0, :, :, :, 2:], 0)
        self.assertTrue((actual.sum(axis=0) <= 1).all())

    def test_reversed_mask_changed_vis_or_active_unused_branch_is_rejected(self):
        mask = np.zeros((93, 16, 16), dtype=bool)
        mask[:, :, :8] = True
        expected = reference_control_weights(mask, (4, 1, 24, 4, 4, 1))
        reverse = reference_control_weights(~mask, expected.shape)
        vis = expected.copy(); vis[1] = .5
        unwanted = expected.copy(); unwanted[2] = .1
        for actual in [reverse, vis, unwanted, np.ones_like(expected), np.full_like(expected, np.nan)]:
            with self.assertRaises(ValueError):
                check_control_weights(actual, expected)

    def test_preview_refuses_changed_video_or_multiple_native_outputs(self):
        # 这里仅测试产物绑定，不把字节夹具当成可播放视频或模型实验。
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            video = directory / "output.mp4"
            video.write_bytes(b"artifact-binding fixture")
            probe = directory / "probe.json"
            record = {"stage": "table_reference_generation", "status": "passed", "result": {"outputs": [str(video)]},
                      "artifacts_sha256": {video.name: hashlib.sha256(video.read_bytes()).hexdigest()}}
            probe.write_text(json.dumps(record))
            self.assertEqual(native_video(probe, "table_reference_generation")[1], video)
            record["result"]["outputs"].append(str(video))
            probe.write_text(json.dumps(record))
            with self.assertRaises(ValueError):
                native_video(probe, "table_reference_generation")
            record["result"]["outputs"].pop()
            probe.write_text(json.dumps(record))
            video.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                native_video(probe, "table_reference_generation")


if __name__ == "__main__":
    unittest.main()
