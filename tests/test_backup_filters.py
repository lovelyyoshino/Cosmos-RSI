"""用真实rsync临时目录核验精选备份、原始数据隔离和硬链接保留。

@author pony
@date 2026-11-07
@version v1.2.0
@last_modified 2026-11-09
@changelog
    - v1.2.0 (2026-11-09): 核验候选退化展示视频可备份，嵌套原始回合仍排除。
    - v1.1.0 (2026-11-09): 正式整圈展示视频可备份，嵌套原始回合视频仍排除。
    - v1.0.0 (2026-11-07): 无网络/GPU的实际备份回归，避免新阶段过滤带回原始视频或数据。
"""
import os
import pathlib
import shutil
import subprocess
import tempfile
import unittest


SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts/backup_to_lenovo.sh"


@unittest.skipUnless(shutil.which("rsync"), "需要rsync执行真实过滤契约")
class BackupFilterTests(unittest.TestCase):
    def test_selective_backup_excludes_raw_data_preserves_hardlinks_and_existing_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source, dest = root / "source", root / "dest"
            for name in ["scripts/a.py", "docs/report.md", "runs/p0/selected/a.png",
                         "runs/p0/selected/raw.mp4", "runs/p0/selected/raw.npy",
                         "runs/p0/unselected/b.png", "data/episode.parquet",
                         "checkpoints/base/model.pt", "runs/p0/video_materials_selected/clip.mp4",
                         "runs/p0/feedback_materials_selected/feedback_success.mp4",
                         "runs/p0/p1_iteration_materials_selected/iteration_standard.mp4",
                         "runs/p0/p1_iteration_materials_selected/videos/raw.mp4",
                         "runs/p0/p1_update_diagnostics_selected/candidate_standard.mp4",
                         "runs/p0/p1_update_diagnostics_selected/episodes/raw.mp4"]:
                path = source / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(name.encode())
            os.link(source / "runs/p0/selected/a.png", source / "runs/p0/selected/copy.png")
            dest.mkdir(); (dest / "user_note.txt").write_text("must survive")
            env = {**os.environ, "RSI_THOR_SOURCE": str(source), "RSI_BACKUP_DEST": str(dest)}
            result = subprocess.run(["bash", str(SCRIPT), "--pull-from-thor", "--p0-run", "selected",
                                     "--p0-run", "video_materials_selected", "--p0-run", "feedback_materials_selected",
                                     "--p0-run", "p1_iteration_materials_selected",
                                     "--p0-run", "p1_update_diagnostics_selected"], env=env,
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            for name in ["scripts/a.py", "docs/report.md", "runs/p0/selected/a.png",
                         "runs/p0/selected/copy.png", "runs/p0/video_materials_selected/clip.mp4",
                         "runs/p0/feedback_materials_selected/feedback_success.mp4",
                         "runs/p0/p1_iteration_materials_selected/iteration_standard.mp4",
                         "runs/p0/p1_update_diagnostics_selected/candidate_standard.mp4", "user_note.txt"]:
                self.assertTrue((dest / name).is_file(), name)
            for name in ["runs/p0/selected/raw.mp4", "runs/p0/selected/raw.npy",
                         "runs/p0/unselected/b.png", "data/episode.parquet", "checkpoints/base/model.pt",
                         "runs/p0/p1_iteration_materials_selected/videos/raw.mp4",
                         "runs/p0/p1_update_diagnostics_selected/episodes/raw.mp4"]:
                self.assertFalse((dest / name).exists(), name)
            self.assertTrue(os.path.samefile(dest / "runs/p0/selected/a.png", dest / "runs/p0/selected/copy.png"))

    def test_invalid_stage_argument_cannot_start_backup(self):
        for arguments in [["--p0-run"], ["--p0-run", ".."], ["--p0-run", "../data"], ["--p0-run", "/etc"]]:
            with self.subTest(arguments=arguments):
                result = subprocess.run(["bash", str(SCRIPT), *arguments], capture_output=True, text=True, timeout=5)
                self.assertEqual(result.returncode, 2, result.stderr)


if __name__ == "__main__":
    unittest.main()
