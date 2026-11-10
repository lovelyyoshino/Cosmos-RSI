"""发布同步只输出必要文本，并保护独立文档、源工作区和本地实验数据。

@author pony
@date 2026-11-10
@version v1.0.0
@changelog
    - v1.0.0 (2026-11-10): 验证素材排除、过程文档清理、预览与重复同步。
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from release import sync_from_workspace as sync


class ReleaseSyncTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.workspace = self.root / "workspace"
        self.destination = self.root / "public"
        self.destination.mkdir()
        self.write(self.workspace, "rsi/loop.py", "from p0_common import run\n")
        self.write(self.workspace, "scripts/p0_common.py", "CONFIG = 'minimal.yaml'\n")
        self.write(self.workspace, "scripts/thor_env.sh", "#!/bin/bash\n")
        self.write(self.workspace, "scripts/unused.py", "pass\n")
        self.write(self.workspace, "configs/minimal.yaml", "enabled: true\n")
        self.write(self.workspace, "configs/eval_frozen.hash", "旧说明\n")
        self.write(self.workspace, "scripts/p0_common.py", "CONFIGS = ['minimal.yaml', 'eval_frozen.hash']\n")
        self.write(self.workspace, "docs/contracts.md", "# 数据契约\n")
        self.write(self.workspace, "docs/eval_protocol.md", "# 评测协议\n")
        self.write(self.workspace, "docs/runbook.md", "工作区旧用法\n")
        self.write(self.workspace, "eval/frozen_eval_set.json", "{}\n")
        self.write(self.workspace, "patches/example/0001-fix.patch", "patch content\n")
        self.write(self.destination, "release/.keep", "local release tooling\n")
        for name in ["README.md", "AGENTS.md", "PLAN.md", "docs/runbook.md", "docs/installation.md",
                     "docs/contracts.md", "docs/eval_protocol.md", "configs/eval_frozen.hash",
                     "tests/test_release_sync.py", "third_party/example/upstream.py", "runs/example/report.json",
                     "data/example.txt", "checkpoints/state.pt"]:
            self.write(self.destination, name, "必须保留\n")
        self.addCleanup(patch.stopall)
        patch.object(sync, "REPO", self.destination).start()
        patch.object(sync, "SCRIPT_SEEDS", ["loop.py", "p0_common.py", "thor_env.sh"]).start()

    @staticmethod
    def write(root, relative, content):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def invoke(self, *arguments):
        with (patch("sys.argv", ["sync", "--workspace", str(self.workspace), *arguments]),
              contextlib.redirect_stdout(io.StringIO())):
            sync.main()

    def test_recursive_inputs_exclude_images_slides_and_process_records(self):
        for name in ["eval/preview.PNG", "eval/slides.ppt", "eval/slides.PPTX", "eval/report.pdf",
                     "patches/example/diagram.svg", "patches/example/view.drawio",
                     "tests/assets/sample.jpg", "tests/archive/intermediate.json", "docs/notes/progress.md",
                     "docs/figures/framework.png"]:
            self.write(self.workspace, name, "不应上传\n")
        files, report = sync.select(self.workspace)
        self.assertIn("configs/minimal.yaml", files)
        self.assertIn("patches/example/0001-fix.patch", files)
        self.assertNotIn("scripts/unused.py", files)
        self.assertFalse(any(Path(name).suffix.lower() in {".png", ".jpg", ".svg", ".drawio", ".pdf", ".ppt", ".pptx"}
                             for name in files))
        self.assertNotIn("docs/notes/progress.md", files)
        self.assertNotIn("tests/archive/intermediate.json", files)
        self.assertEqual(report["notes_kept"], 0)

    def test_dry_run_changes_neither_source_nor_destination(self):
        old = self.write(self.destination, "docs/notes/old.md", "旧过程记录\n")
        before = {str(p.relative_to(self.root)): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        self.invoke("--dry-run")
        after = {str(p.relative_to(self.root)): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        self.assertTrue(old.exists())

    def test_sync_removes_stale_docs_preserves_local_work_and_is_idempotent(self):
        for name in ["docs/notes/old.md", "docs/figures/old.png", "docs/CURRENT_STATUS.md", "docs/p0_report.md"]:
            self.write(self.destination, name, "旧过程记录\n")
        source_before = {str(p.relative_to(self.workspace)): p.read_bytes()
                         for p in self.workspace.rglob("*") if p.is_file()}
        self.invoke()
        self.assertFalse((self.destination / "docs/notes/old.md").exists())
        self.assertFalse((self.destination / "docs/figures/old.png").exists())
        self.assertFalse((self.destination / "docs/CURRENT_STATUS.md").exists())
        self.assertFalse((self.destination / "docs/p0_report.md").exists())
        for name in ["README.md", "AGENTS.md", "PLAN.md", "docs/runbook.md", "docs/installation.md",
                     "docs/contracts.md", "docs/eval_protocol.md", "configs/eval_frozen.hash",
                     "tests/test_release_sync.py", "third_party/example/upstream.py", "runs/example/report.json",
                     "data/example.txt", "checkpoints/state.pt"]:
            self.assertEqual((self.destination / name).read_text(), "必须保留\n")
        source_after = {str(p.relative_to(self.workspace)): p.read_bytes()
                        for p in self.workspace.rglob("*") if p.is_file()}
        self.assertEqual(source_before, source_after)
        manifest = json.loads((self.destination / "release/sync_manifest.json").read_text())
        for name, checksum in manifest["files"].items():
            self.assertEqual(hashlib.sha256((self.destination / name).read_bytes()).hexdigest(), checksum)
        mtimes = {p: p.stat().st_mtime_ns for p in self.destination.rglob("*")
                  if p.is_file() and p.name != "sync_manifest.json"}
        self.invoke()
        self.assertEqual(mtimes, {p: p.stat().st_mtime_ns for p in mtimes})

    def test_refuses_to_use_the_public_repository_as_source(self):
        patch.object(sync, "REPO", self.workspace).start()
        with self.assertRaisesRegex(SystemExit, "不能是同一目录"):
            self.invoke()

    def test_refuses_a_source_nested_inside_the_public_repository(self):
        patch.object(sync, "REPO", self.root).start()
        with self.assertRaisesRegex(SystemExit, "不能位于发布仓库内部"):
            self.invoke()

    def test_private_content_blocks_mutation_without_echoing_the_secret(self):
        secret = "hf_" + "x" * 32
        self.write(self.destination, "docs/runbook.md", secret)
        old = self.write(self.destination, "docs/notes/old.md", "未完成检查前不得删除\n")
        with self.assertRaises(SystemExit) as caught:
            self.invoke()
        self.assertNotIn(secret, str(caught.exception))
        self.assertIn("docs/runbook.md", str(caught.exception))
        self.assertTrue(old.exists())
        self.assertFalse((self.destination / "eval/frozen_eval_set.json").exists())


if __name__ == "__main__":
    unittest.main()
