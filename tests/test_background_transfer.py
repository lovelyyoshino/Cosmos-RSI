"""核验背景路线不会借用旧暖暗来源，且保留全部前景与源标签约束。

@author pony
@date 2026-11-07
@version v1.0.1
@last_modified 2026-11-07
@changelog
    - v1.0.1 (2026-11-07): 证明旧诊断和未预声明数据会在图像解码及模型加载前拒绝。
    - v1.0.0 (2026-11-07): 覆盖预声明、配方参数不变、全帧像素篡改及错误数组拒绝。
"""
from __future__ import annotations

import copy
import hashlib
import json
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from p0_background_composite import check_declaration, verify_composite_pixels
from p0_prepare_background_transfer import COMPOSITE_RECIPE, TARGET, background_spec
from p0_protected_background_preview import protected_background
from p0_generated_window_gate import verified_generation_paths


class BackgroundTransferTests(unittest.TestCase):
    def test_diagnostic_preview_is_not_accepted_as_a_formal_generation(self):
        with self.assertRaises(ValueError):
            verified_generation_paths({"stage": "protected_background_preview", "status": "passed"},
                                      pathlib.Path("/not/a/real/probe.json"))

    def test_missing_predeclaration_rejected_before_decode_or_models(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            directory = root / "runs/p0/derived"
            directory.mkdir(parents=True)
            source = root / "runs/p0/source.json"
            source.write_text(json.dumps({"target_condition": {"light": "warm_dim"}}))
            output = directory / "output.mp4"
            output.write_bytes(b"content is deliberately not a real video")
            digest = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
            record = {"stage": "background_composite", "status": "passed", "result": {
                "source_success_evidence": {"path": str(source), "sha256": digest(source)},
                "outputs": [str(output)], "guardrails_enabled": True},
                "artifacts_sha256": {output.name: digest(output)}}
            with patch("p0_generated_window_gate.sha256", digest), \
                    patch("p0_background_composite.decode_frames") as decode:
                with self.assertRaisesRegex(ValueError, "声明"):
                    verified_generation_paths(record, directory / "probe.json", root)
                decode.assert_not_called()

    def test_previous_light_record_cannot_be_used_as_background_generation(self):
        valid = {"target_condition": TARGET, "composite_recipe": COMPOSITE_RECIPE,
                 "source_success": True, "replay_validation": {"passed": True}}
        check_declaration(valid)
        for changed in [{"target_condition": {"light": "warm_dim_indoor"}}, {"composite_recipe": None},
                        {"source_success": False}, {"replay_validation": {"passed": False}}]:
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                check_declaration({**valid, **changed})

    def test_only_prompt_name_and_paths_change_from_original_recipe(self):
        parent = {"name": "old", "prompt": "old light", "video_path": "old/source.mp4", "max_frames": 93,
                  "num_video_frames_per_chunk": 93, "num_steps": 35, "seed": 42, "resolution": "480",
                  "keep_input_resolution": False, "guided_generation_mask": "old/mask.mp4",
                  "guided_generation_step_threshold": 34,
                  **{k: {"control_path": f"old/{k}.mp4", "control_weight": w}
                     for k, w in {"edge": .2, "vis": .5, "depth": 1., "seg": 1.}.items()}}
        before = copy.deepcopy(parent)
        spec = background_spec(parent, pathlib.Path("/new"), "pick up the book")
        self.assertEqual(parent, before)
        excluded = {"name", "prompt", "video_path", "guided_generation_mask", "edge", "vis", "depth", "seg"}
        self.assertEqual({k: v for k, v in spec.items() if k not in excluded},
                         {k: v for k, v in parent.items() if k not in excluded})
        for name in ["edge", "vis", "depth", "seg"]:
            self.assertEqual(spec[name]["control_weight"], parent[name]["control_weight"])
        self.assertIn("matte dark wood", spec["prompt"])

    def test_actual_pixels_must_match_predeclared_transform_in_every_region(self):
        source = np.zeros((93, 256, 256, 3), np.uint8)
        generated = np.full_like(source, 197)
        masks = np.zeros((93, 1, 256, 256), bool)
        masks[:, :, 80:110, 80:110] = True
        output = np.asarray([protected_background(a, b, m, 8)[0] for a, b, m in zip(source, generated, masks)])
        verify_composite_pixels(source, generated, masks, output, 8)
        for frame, y, x in [(0, 90, 90), (92, 0, 0)]:
            changed = output.copy(); changed[frame, y, x, 0] ^= 1
            with self.subTest(frame=frame), self.assertRaises(ValueError):
                verify_composite_pixels(source, generated, masks, changed, 8)
        with self.assertRaises(ValueError):
            verify_composite_pixels(source[:-1], generated, masks, output, 8)


if __name__ == "__main__":
    unittest.main()
