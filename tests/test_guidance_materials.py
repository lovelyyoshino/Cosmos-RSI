"""验证时空引导视频中的同源消融声明，拒绝改变动作或其他生成因素。

@author pony
@date 2026-11-07
@version v1.0.0
@last_modified 2026-11-07
@changelog
    - v1.0.0 (2026-11-07): 验证配对源、控制摘要、提示、种子和原生权重的不变量。
"""
from __future__ import annotations

import copy
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from p0_guidance_materials import validate_pair


class GuidanceMaterialsTests(unittest.TestCase):
    def setUp(self):
        self.before = {"source_episode_id": "demo-train", "window": {"start_frame": 0, "end_frame": 93},
                       "instruction": "pick book", "roi_names": ["robot", "book"], "labels_sha256": "labels",
                       "cache_manifest_sha256": "cache", "files_sha256": {name: name for name in
                       ["source.mp4", "mask.mp4", "roi_masks.npy", "edge.mp4", "vis.mp4", "depth.mp4", "seg.mp4"]}}
        self.after = copy.deepcopy(self.before)
        self.after["files_sha256"]["mask.mp4"] = "expanded mask"
        self.after["mask_transform"] = {"kind": "sim_roi_spatial_dilation_and_latent_group_union"}
        self.spec = {"prompt": "same prompt", "negative_prompt": "same negative prompt", "num_steps": 35,
                     "seed": 42, "resolution": "480", "guided_generation_step_threshold": 34,
                     "video_path": "/parent/source.mp4", "guided_generation_mask": "/parent/mask.mp4",
                     **{key: {"control_path": f"/parent/{key}.mp4", "control_weight": weight}
                        for key, weight in {"edge": 0.2, "vis": 0.5, "depth": 1.0, "seg": 1.0}.items()}}
        self.new_spec = copy.deepcopy(self.spec)
        self.new_spec.update({"name": "derived", "video_path": "/derived/source.mp4", "guided_generation_mask": "/derived/mask.mp4"})
        for key in ["edge", "vis", "depth", "seg"]:
            self.new_spec[key]["control_path"] = f"/derived/{key}.mp4"

    def test_same_content_at_derived_paths_is_valid(self):
        validate_pair(self.before, self.after, self.spec, self.new_spec)

    def test_changed_source_labels_or_window_are_rejected(self):
        for key, value in [("labels_sha256", "changed"), ("source_episode_id", "another"),
                           ("window", {"start_frame": 1, "end_frame": 94}), ("instruction", "another")]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_pair(self.before, {**self.after, key: value}, self.spec, self.new_spec)

    def test_control_pixels_and_same_mask_are_rejected(self):
        for name in ["source.mp4", "roi_masks.npy", "edge.mp4", "vis.mp4", "depth.mp4", "seg.mp4"]:
            changed = copy.deepcopy(self.after)
            changed["files_sha256"][name] = "changed"
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_pair(self.before, changed, self.spec, self.new_spec)
        with self.assertRaises(ValueError):
            validate_pair(self.before, {**self.after, "files_sha256": self.before["files_sha256"]}, self.spec, self.new_spec)

    def test_other_sampling_changes_are_not_single_factor_comparisons(self):
        for key, value in [("seed", 7), ("num_steps", 36), ("prompt", "another"),
                           ("negative_prompt", "another"), ("guided_generation_step_threshold", 33),
                           ("image_context_path", "/reference.png")]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_pair(self.before, self.after, self.spec, {**self.new_spec, key: value})
        changed = copy.deepcopy(self.new_spec)
        changed["edge"]["control_weight"] = 1.0
        with self.assertRaises(ValueError):
            validate_pair(self.before, self.after, self.spec, changed)

    def test_missing_control_or_repeated_mask_expansion_is_rejected(self):
        changed = copy.deepcopy(self.after)
        del changed["files_sha256"]["seg.mp4"]
        with self.assertRaises(ValueError):
            validate_pair(self.before, changed, self.spec, self.new_spec)
        with self.assertRaises(ValueError):
            validate_pair({**self.before, "mask_transform": self.after["mask_transform"]}, self.after, self.spec, self.new_spec)


if __name__ == "__main__":
    unittest.main()
