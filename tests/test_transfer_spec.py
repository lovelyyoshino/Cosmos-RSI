"""验证Transfer参数不会在固定上游实现中静默失效。

@author pony
@date 2026-11-07
@version v1.2.1
@last_modified 2026-11-07
@changelog
    - v1.2.1 (2026-11-07): 区分图像条件分支与CFG无条件零编码，拒绝缺失或非有限的实际参考。
    - v1.2.0 (2026-11-07): 源首帧参考仅允许base，拒绝视频帧索引静默覆盖参考PNG。
    - v1.1.0 (2026-11-07): base组件清单必须包含独立的SigLIP2；蒸馏入口不额外下载此模型。
    - v1.0.0 (2026-11-07): 覆盖蒸馏无效参数、base空间引导及窗口/控制边界。
"""
from __future__ import annotations

import pathlib
import sys
import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from p0_transfer_smoke import (TRANSFER_WEIGHTS, checked_components, validate_generation_spec,
                               validate_image_context_trace)


class TransferSpecTests(unittest.TestCase):
    def setUp(self):
        self.spec = {"max_frames": 93, "num_video_frames_per_chunk": 93,
                     "edge": {"control_path": "edge.mp4", "control_weight": 1.0}}

    def test_existing_distilled_recipe_remains_valid(self):
        validate_generation_spec(self.spec, "edge/distilled", 4)

    def test_ignored_distilled_options_fail_before_model_load(self):
        for extra in [{"sigma_max": "5"}, {"guided_generation_mask": "mask.mp4"}, {"guidance": 1}]:
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                validate_generation_spec({**self.spec, **extra}, "edge/distilled", 4)
        with self.assertRaises(ValueError):
            validate_generation_spec(self.spec, "edge/distilled", 8)

    def test_base_supports_guidance_but_not_ignored_sigma(self):
        guided = {**self.spec, "guided_generation_mask": "mask.mp4", "guided_generation_step_threshold": 34}
        validate_generation_spec(guided, "edge", 35)
        with self.assertRaises(ValueError):
            validate_generation_spec({**guided, "sigma_max": "5"}, "edge", 35)
        with self.assertRaises(ValueError):
            validate_generation_spec({**guided, "guided_generation_step_threshold": 35}, "edge", 35)

    def test_window_and_control_must_match_executed_model(self):
        for extra in [{"max_frames": 100}, {"num_video_frames_per_chunk": 1}, {"vis": {"control_weight": 0.5}}]:
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                validate_generation_spec({**self.spec, **extra}, "edge/distilled", 4)

    def test_reference_image_cannot_be_ignored_or_overridden(self):
        reference = {**self.spec, "image_context_path": "source_first_frame.png"}
        validate_generation_spec(reference, "edge", 35)
        with self.assertRaises(ValueError):
            validate_generation_spec(reference, "edge/distilled", 4)
        for index in [0, 1]:
            with self.subTest(index=index), self.assertRaises(ValueError):
                validate_generation_spec({**reference, "context_frame_index": index}, "edge", 35)

    def test_cfg_zero_embedding_does_not_reject_a_valid_conditional_reference(self):
        active = {"input_present": True, "embedding_finite": True, "embedding_nonzero": 294912}
        inactive = {"input_present": False, "embedding_finite": True, "embedding_nonzero": 0}
        validate_image_context_trace([active, inactive, active])
        for invalid in [[], [inactive], [{**active, "embedding_nonzero": 0}],
                        [{**active, "embedding_finite": False}], [active, {**inactive, "embedding_nonzero": 1}]]:
            with self.subTest(invalid=invalid), self.assertRaises(RuntimeError):
                validate_image_context_trace(invalid)

    def test_separate_variant_downloads_keep_the_correct_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            names = ["Cosmos-Transfer2.5-2B", "Cosmos-Predict2.5-2B", "Cosmos-Reason1-7B",
                     "Cosmos-Guardrail1", "Qwen3Guard-Gen-0.6B", "siglip-so400m-patch14-384",
                     "siglip2-so400m-patch16-naflex"]

            def write_manifest(index, name, filename, content):
                destination = root / "checkpoints/base" / name
                target = destination / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
                path = root / f"runs/p0/model_download_{index:02d}/model_manifest.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({"path": str(destination), "revision": str(index), "repo": name,
                                            "files": [{"file": filename, "bytes": len(content),
                                                       "sha256": hashlib.sha256(content).hexdigest()}]}))
                return path

            first = write_manifest(0, names[0], TRANSFER_WEIGHTS["edge/distilled"], b"distilled")
            for index, name in enumerate(names[1:], 1):
                write_manifest(index, name, "component.bin", name.encode())
            latest = write_manifest(7, names[0], TRANSFER_WEIGHTS["edge"], b"base")
            with patch("p0_transfer_smoke.ROOT", root):
                self.assertEqual(checked_components()[names[0]]["manifest"], str(first))
                self.assertEqual(checked_components("edge")[names[0]]["manifest"], str(latest))
                (root / "checkpoints/base" / names[0] / TRANSFER_WEIGHTS["edge"]).write_bytes(b"bad!")
                with self.assertRaisesRegex(RuntimeError, "清单|SHA256"):
                    checked_components("edge")


if __name__ == "__main__":
    unittest.main()
