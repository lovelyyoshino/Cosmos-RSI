"""空间控制诊断须检出掩码反向、错位和缺失的去噪步骤或注入层。

@author pony
@date 2026-11-08
@version v1.1.0
@last_modified 2026-11-08
@changelog
    - v1.1.0 (2026-11-08): 增加共同可见桌体的静态、真实闪变及空区域拒绝检查。
    - v1.0.0 (2026-11-08): 覆盖首帧独立压缩、全帧/反向/错位及不完整实际trace。
"""
from __future__ import annotations

import copy
import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from p0_single_control_trace import expected_weights, forwarded_argument, validate_trace, weight_statistics
from p0_single_control_materials import static_table_changes


class SingleControlTraceTests(unittest.TestCase):
    def setUp(self):
        self.mask = np.zeros((93, 16, 16), dtype=bool)
        self.mask[:, :, :8] = True
        self.mask[0, :8] = True
        self.shape = (1, 1, 24, 4, 4, 1)

    def test_wrapped_forward_parameters_are_read_from_actual_call(self):
        def wrapped(*args, **kwargs):
            pass

        def explicit(image, control_context_scale):
            pass

        self.assertEqual(forwarded_argument(wrapped, (0,), {"control_context_scale": .4}, "control_context_scale"), .4)
        self.assertEqual(forwarded_argument(explicit, (0, .5), {}, "control_context_scale"), .5)
        with self.assertRaises(ValueError):
            forwarded_argument(wrapped, (0,), {}, "control_context_scale")

    def test_first_frame_is_independent_and_rest_is_temporally_compressed(self):
        actual = expected_weights(self.mask, self.shape)
        self.assertEqual(actual.shape, self.shape)
        self.assertEqual(float(actual[0, 0, 0].mean()), .75)
        self.assertEqual(float(actual[0, 0, 1:].mean()), .5)
        self.assertTrue(weight_statistics(actual, actual)["matches_source_mask"])

    def test_wrong_direction_shift_full_frame_or_nonfinite_is_rejected(self):
        expected = expected_weights(self.mask, self.shape)
        for invalid in [1 - expected, np.roll(expected, 1, axis=4), np.ones_like(expected),
                        np.full_like(expected, np.nan), np.array(1.), expected[..., 0]]:
            with self.subTest(shape=invalid.shape), self.assertRaises(ValueError):
                weight_statistics(invalid, expected)

    def test_invalid_mask_or_latent_shape_is_rejected(self):
        for mask, shape in [(self.mask.astype(np.uint8), self.shape), (self.mask[:92], self.shape),
                            (np.ones_like(self.mask), self.shape), (self.mask, (1, 1, 23, 4, 4, 1))]:
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                expected_weights(mask, shape)

    def test_missing_step_missing_layer_or_changed_layer_scale_is_rejected(self):
        stats = weight_statistics(expected_weights(self.mask, self.shape), expected_weights(self.mask, self.shape))
        trace = [{"step": index, "network": stats, "blocks": [
            {**stats, "name": name, "max_abs_error_vs_network": 0} for name in ["0", "4"]]} for index in range(4)]
        validate_trace(trace, ["0", "4"])
        missing = copy.deepcopy(trace)
        missing[2]["blocks"].pop()
        changed = copy.deepcopy(trace)
        changed[3]["blocks"][0]["max_abs_error_vs_network"] = .01
        for invalid in [trace[:3], missing, changed, trace[::-1]]:
            with self.assertRaises(ValueError):
                validate_trace(invalid, ["0", "4"])

    def test_static_table_changes_ignore_moving_protected_foreground(self):
        frames = np.zeros((3, 4, 4, 3), dtype=np.uint8)
        frames[1, 0] = 255
        visible = np.ones((4, 4), dtype=bool)
        visible[0] = False
        self.assertTrue(all(row["mean_abs_rgb_change_0_1"] == 0 for row in static_table_changes(frames, visible)))
        frames[2, 1:] = 51
        rows = static_table_changes(frames, visible)
        self.assertAlmostEqual(rows[1]["mean_abs_rgb_change_0_1"], .2)
        with self.assertRaises(ValueError):
            static_table_changes(frames, np.zeros_like(visible))


if __name__ == "__main__":
    unittest.main()
