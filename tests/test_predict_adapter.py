"""Predict接口回归：拒绝伪造动作长度、错用八维状态及错误动作尺度。"""
import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from p0_predict_adapter import normalized_actions, predictor_observation


class PredictAdapterTests(unittest.TestCase):
    def test_official_minmax_roundtrip_and_reject_short_chunk(self):
        low = np.arange(7, dtype=np.float32) - 4
        high = low + 2
        raw = np.tile(low + .5, (16, 1))
        stats = {"actions_min": low, "actions_max": high}
        result = normalized_actions(raw, stats)
        np.testing.assert_allclose(result, -.5)
        np.testing.assert_allclose((result + 1) / 2 * (high - low) + low, raw)
        with self.assertRaises(ValueError): normalized_actions(raw[:8], stats)

    def test_camera_conversion_keeps_views_and_true_quaternion(self):
        image = np.arange(256, dtype=np.uint8)[None, :, None] * np.ones((256, 1, 3), dtype=np.uint8)
        proprio = np.zeros((17, 9), dtype=np.float32)
        proprio[:, 8] = 1
        packet = {"agentview": np.tile(image[None], (17, 1, 1, 1)),
                  "wrist": np.zeros((17, 256, 256, 3), dtype=np.uint8), "proprio": proprio}
        obs = predictor_observation(packet)
        self.assertEqual(int(obs["primary_image"][0, 0, 0]), 255)
        self.assertEqual(int(obs["wrist_image"].max()), 0)
        np.testing.assert_array_equal(obs["proprio"], proprio[0])
        packet["proprio"] = proprio[:, :8]
        with self.assertRaises(ValueError): predictor_observation(packet)


if __name__ == "__main__":
    unittest.main()
