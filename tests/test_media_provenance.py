"""确保视频归档同时携带经原始摘要核验的配对CSV与截图，拒绝静默丢失来源。

@author pony
@date 2026-11-07
@version v1.1.0
@last_modified 2026-11-08
@changelog
    - v1.1.0 (2026-11-08): 覆盖实际新清单的路径字符串/独立摘要父链，缺摘要仍拒绝。
    - v1.0.0 (2026-11-07): 覆盖历史来源恢复、截图/CSV篡改、父清单错摘要及循环/越界。
"""
from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from p0_multicontrol_materials import copy_training_progress_evidence
from p0_policy_runtime import sha256


class MediaProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = pathlib.Path(self.temporary.name)
        self.original = self.root / "runs/p0/original"
        self.previous = self.root / "runs/p0/previous"
        self.output = self.root / "runs/p0/output"
        for path in [self.original, self.previous, self.output]:
            path.mkdir(parents=True)
        self.addCleanup(patch.stopall)
        patch("p0_multicontrol_materials.ROOT", self.root).start()
        picture = self.original / "training_frame_000.png"
        picture.write_bytes(b"recorded screenshot")
        csv_path = self.original / "training_progress_pairs.csv"
        csv_path.write_text("episode_id,before_success,after_success\nfixed-id,False,True\n")
        self.clips = [{"kind": "initial_training_progress", "file": "training.mp4", "sha256": "video-digest",
                       "screenshots": [{"file": picture.name, "frame": 0, "sha256": sha256(picture)}]}]
        manifest = {"clips": self.clips, "paired_episodes": [{"episode_id": "fixed-id"}],
                    "subset_summary": [{"n_pairs": 1}], "training_conditions": "fixed training",
                    "evaluation_conditions": "fixed evaluation", "claim": "初训子集，不是RSI"}
        self.source_manifest = self.original / "video_manifest.json"
        self.source_manifest.write_text(json.dumps(manifest))
        probe = {"stage": "video_materials", "status": "passed", "artifacts_sha256": {
            "video_manifest.json": sha256(self.source_manifest), "training_progress_pairs.csv": sha256(csv_path)}}
        (self.original / "probe.json").write_text(json.dumps(probe))
        self.previous_manifest = self.previous / "video_manifest.json"
        self.previous_manifest.write_text(json.dumps({"previous_manifest": {
            "path": str(self.source_manifest), "sha256": sha256(self.source_manifest)}}))

    def archive(self):
        return copy_training_progress_evidence(self.output, self.previous_manifest, self.clips)

    def test_recovers_missing_sidecars_through_verified_history(self):
        result = self.archive()
        self.assertEqual(len(result["files"]), 2)
        self.assertEqual(sha256(self.output / "training_frame_000.png"), sha256(self.original / "training_frame_000.png"))
        self.assertEqual(sha256(self.output / "training_progress_pairs.csv"), sha256(self.original / "training_progress_pairs.csv"))
        self.assertEqual(sha256(self.output / result["archived_manifest"]), sha256(self.source_manifest))
        self.assertEqual(result["training_conditions"], "fixed training")

    def test_changed_screenshot_is_rejected(self):
        (self.original / "training_frame_000.png").write_bytes(b"changed screenshot")
        with self.assertRaises(RuntimeError):
            self.archive()

    def test_string_parent_with_separate_digest_restores_the_same_verified_evidence(self):
        self.previous_manifest.write_text(json.dumps({"previous_manifest": str(self.source_manifest),
            "previous_manifest_sha256": sha256(self.source_manifest)}))
        result = self.archive()
        self.assertEqual(result["source_manifest_sha256"], sha256(self.source_manifest))

    def test_string_parent_without_digest_is_rejected(self):
        self.previous_manifest.write_text(json.dumps({"previous_manifest": str(self.source_manifest)}))
        with self.assertRaises(RuntimeError):
            self.archive()

    def test_changed_csv_is_rejected(self):
        (self.original / "training_progress_pairs.csv").write_text("changed count\n")
        with self.assertRaises(RuntimeError):
            self.archive()

    def test_wrong_parent_digest_is_rejected(self):
        value = json.loads(self.previous_manifest.read_text())
        value["previous_manifest"]["sha256"] = "wrong"
        self.previous_manifest.write_text(json.dumps(value))
        with self.assertRaises(RuntimeError):
            self.archive()

    def test_source_chain_cannot_escape_project(self):
        other = self.root / "outside.json"
        other.write_text(self.source_manifest.read_text())
        self.previous_manifest.write_text(json.dumps({"previous_manifest": {"path": str(other), "sha256": sha256(other)}}))
        with self.assertRaises(RuntimeError):
            self.archive()

    def test_cycle_is_rejected(self):
        self.previous_manifest.write_text(json.dumps({"previous_manifest": {
            "path": str(self.previous_manifest), "sha256": "untrusted-cycle"}}))
        with self.assertRaises(RuntimeError):
            self.archive()

    def test_without_training_clips_does_not_require_training_files(self):
        self.assertIsNone(copy_training_progress_evidence(self.output, self.previous_manifest, []))


if __name__ == "__main__":
    unittest.main()
