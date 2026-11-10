"""回归验证镜像重分片与Range被忽略时的失败边界。

@author pony
@date 2026-11-06
@version v1.0.0
@last_modified 2026-11-06
@changelog
    - v1.0.0 (2026-11-06): 覆盖实际页脚选片、缺回合、破损页脚和拒绝整文件响应。
"""
from __future__ import annotations

import io
import pathlib
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from p0_dataset_index import footer_record, read_range, selected_files


class DatasetIndexTests(unittest.TestCase):
    def test_actual_footer_selects_resplit_file(self):
        import pyarrow as arrow
        import pyarrow.parquet as parquet

        entries = []
        for file_index, episode_ids in enumerate([[0, 0, 1], [2, 2, 3]]):
            payload = io.BytesIO()
            parquet.write_table(arrow.table({"episode_index": episode_ids, "index": list(range(3 * file_index, 3 * file_index + 3))}), payload)
            content = payload.getvalue()
            entries.append(footer_record(content, {"Path": f"data/file-{file_index}.parquet", "Size": len(content), "Sha256": "fixture"}))
        # 旧元数据把episode2指向file0，实际页脚必须选择file1。
        self.assertEqual(selected_files(entries, {2}), ["data/file-1.parquet"])

    def test_missing_episode_fails(self):
        with self.assertRaisesRegex(RuntimeError, "缺少所选回合"):
            selected_files([], {42})

    def test_bad_footer_fails(self):
        with self.assertRaisesRegex(RuntimeError, "PAR1"):
            footer_record(b"invalid!", {"Path": "bad", "Size": 8, "Sha256": "fixture"})

    def test_ignored_range_never_reads_body(self):
        response = MagicMock(status_code=200)
        response.__enter__.return_value = response
        with patch("requests.get", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "未遵守"):
                read_range("https://example.invalid/file", 92, 99, 100)
        response.raw.read.assert_not_called()

    def test_short_range_fails(self):
        response = MagicMock(status_code=206, headers={"Content-Range": "bytes 92-99/100"})
        response.__enter__.return_value = response
        response.raw.read.return_value = b"short"
        with patch("requests.get", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "长度不符"):
                read_range("https://example.invalid/file", 92, 99, 100)


if __name__ == "__main__":
    unittest.main()
