"""精选目录保留关键前后对照，且不会复制已改变的历史媒体。

@author pony
@date 2026-11-07
@version v1.0.0
@last_modified 2026-11-07
@changelog
    - v1.0.0 (2026-11-07): 验证旧/新精选选择、来源摘要及历史媒体硬链接复用。
"""
import json
import os
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from p0_background_preview_materials import copy_key_clips
from p0_media_index import digest


class CuratedBackgroundMediaTests(unittest.TestCase):
    def fixture(self, root):
        old = root / "original"
        old.mkdir()
        names = ["03_policy_t00_success.mp4", "04_policy_t00_failure.mp4",
                 "13_training_500_vs_5000_standard.mp4", "14_training_500_vs_5000_lighting.mp4",
                 "15_training_500_vs_5000_background.mp4", "17_sim_source_depth_seg_controls.mp4",
                 "21_cosmos_visual_vs_four_controls.mp4", "26_cosmos_predeclared_background_sample.mp4"]
        clips = []
        for name in names:
            poster = pathlib.Path(name).with_suffix(".jpg").name
            for filename in [name, poster]:
                (old / filename).write_bytes(("CPU contract fixture " + filename).encode())
            clips.append({"file": name, "poster": poster})
        manifest = old / "video_manifest.json"
        manifest.write_text(json.dumps({"clips": clips, "clip_count": len(clips)}))
        checksums = {path.name: digest(path) for path in old.iterdir()}
        (old / "probe.json").write_text(json.dumps({"stage": "video_materials", "status": "passed",
                                                   "artifacts_sha256": checksums}))
        return manifest

    def test_roi_export_retains_prior_geometry_example_without_growing_old_comparisons(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            manifest = self.fixture(root)
            for roi in [False, True]:
                destination = root / str(roi)
                destination.mkdir()
                clips = copy_key_clips(destination, manifest, roi_comparison=roi)
                self.assertEqual(len(clips), 7)
                files = {clip["file"] for clip in clips}
                self.assertEqual("26_cosmos_predeclared_background_sample.mp4" in files, roi)
                self.assertEqual("21_cosmos_visual_vs_four_controls.mp4" in files, not roi)
                for clip in clips:
                    self.assertTrue(os.path.samefile(destination / clip["file"], manifest.parent / clip["file"]))

    def test_changed_historical_video_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            manifest = self.fixture(root)
            (manifest.parent / "03_policy_t00_success.mp4").write_bytes(b"changed")
            destination = root / "new"
            destination.mkdir()
            with self.assertRaises(ValueError):
                copy_key_clips(destination, manifest, roi_comparison=True)


if __name__ == "__main__":
    unittest.main()
