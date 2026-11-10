"""验证派生预览的前景保护及来源边界，禁止冒充原生可入训数据。

@author pony
@date 2026-11-07
@version v1.0.0
@last_modified 2026-11-07
@changelog
    - v1.0.0 (2026-11-07): 前景像素不变、背景来自生成、输入不可变、错误来源及篡改数组拒绝。
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from p0_protected_background_preview import protected_background, checked_preview_frames


class ProtectedBackgroundTests(unittest.TestCase):
    def test_preserves_all_disconnected_foreground_and_uses_generated_background(self):
        source = np.zeros((32, 32, 3), dtype=np.uint8)
        source[:] = [40, 70, 90]
        generated = np.full_like(source, 200)
        rois = np.zeros((2, 32, 32), dtype=bool)
        rois[0, 2:6, 2:6] = True
        rois[1, 20:23, 20:23] = True
        originals = [x.copy() for x in [source, generated, rois]]
        output, protection = protected_background(source, generated, rois, 2)
        self.assertTrue(np.all(protection[rois.any(axis=0)]))
        np.testing.assert_array_equal(output[protection], source[protection])
        np.testing.assert_array_equal(output[~protection], generated[~protection])
        self.assertTrue((~protection).any())
        for actual, before in zip([source, generated, rois], originals):
            np.testing.assert_array_equal(actual, before)

    def test_invalid_input_cannot_silently_resample_or_broadcast(self):
        source = np.zeros((8, 8, 3), dtype=np.uint8)
        rois = np.ones((1, 8, 8), dtype=bool)
        for image, mask, radius in [(source.astype(np.float32), rois, 2), (source[:4], rois, 2),
                                    (source, rois.astype(float) * .5, 2),
                                    (source, np.zeros_like(rois), 2), (source, rois, -1), (source, rois, True)]:
            with self.subTest(radius=radius), self.assertRaises(ValueError):
                protected_background(source, image, mask, radius)

    def test_preview_arrays_require_same_source_exact_hash_and_nontraining_role(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            frames = root / "preview_frames.npy"
            np.save(frames, np.zeros((2, 256, 256, 3), dtype=np.uint8))
            digest = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
            record = {"stage": "protected_background_preview", "status": "passed", "result": {
                "derived_kind": "cosmos_background_source_foreground_composite",
                "source_input_sha256": "source", "eligible_for_training": False,
                "frames_file": str(frames), "frames_sha256": digest(frames)},
                "artifacts_sha256": {frames.name: digest(frames)}}
            probe = root / "probe.json"
            probe.write_text(json.dumps(record))
            with patch("p0_protected_background_preview.sha256", digest):
                self.assertEqual(checked_preview_frames(probe, "source", 2).shape, (2, 256, 256, 3))
                with self.assertRaises(ValueError):
                    checked_preview_frames(probe, "other-source", 2)
                record["result"]["eligible_for_training"] = True
                probe.write_text(json.dumps(record))
                with self.assertRaises(ValueError):
                    checked_preview_frames(probe, "source", 2)
                record["result"]["eligible_for_training"] = False
                probe.write_text(json.dumps(record))
                np.save(frames, np.ones((2, 256, 256, 3), dtype=np.uint8))
                with self.assertRaises(ValueError):
                    checked_preview_frames(probe, "source", 2)


if __name__ == "__main__":
    unittest.main()
