"""验证部署槽位的路径边界、权重身份和延迟计数，不加载模型。

@author pony
@date 2026-11-07
@version v1.0.0
@last_modified 2026-11-07
@changelog
    - v1.0.0 (2026-11-07): 拒绝槽位越界、普通文件覆盖、错误动作及预热混入统计。
"""
from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from p0_deploy_smoke import atomic_slot, latency_summary, validate_response


class DeploySmokeTests(unittest.TestCase):
    def test_switch_and_rollback_do_not_overwrite_real_directories(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            slots, old, new = [root / name for name in ["slots", "old", "new"]]
            for path in [slots, old, new]:
                path.mkdir()
            atomic_slot(slots, "current", old, root)
            atomic_slot(slots, "previous", (slots / "current").resolve(), root)
            atomic_slot(slots, "current", new, root)
            self.assertEqual((slots / "current").resolve(), new)
            atomic_slot(slots, "current", (slots / "previous").resolve(), root)
            self.assertEqual((slots / "current").resolve(), old)
            with self.assertRaisesRegex(ValueError, "越界"):
                atomic_slot(slots, "../../unsafe", new, root)
            with self.assertRaisesRegex(ValueError, "越界"):
                atomic_slot(slots, "current", root.parent, root)
            (slots / "previous").unlink()
            (slots / "previous").write_text("retain")
            with self.assertRaisesRegex(ValueError, "非链接"):
                atomic_slot(slots, "previous", new, root)
            self.assertEqual((slots / "previous").read_text(), "retain")

    def test_latency_excludes_warmup_and_requires_complete_requests(self):
        rows = [{"request_index": i, "server_ms": value, "http_ms": value + 1}
                for i, value in enumerate([1000, 10, 20, 30])]
        summary = latency_summary(rows, 1, 3)
        self.assertEqual(summary["server_ms"]["p50"], 20)
        self.assertEqual(summary["http_ms"]["p50"], 21)
        for changed in [rows[:-1], [rows[0], rows[1], rows[1], rows[3]]]:
            with self.assertRaises(ValueError):
                latency_summary(changed, 1, 3)
        with self.assertRaises(ValueError):
            latency_summary([{"request_index": 0, "server_ms": float("nan"), "http_ms": 1}], 0, 1)

    def test_response_requires_expected_weights_and_finite_action_shape(self):
        valid = {"state_sha256": "expected", "actions": [[0] * 7] * 8,
                 "normalized_actions": [[0] * 7] * 8, "latency_ms": 50}
        validate_response(valid, "expected")
        for changed in [{"state_sha256": "stale"}, {"actions": [[0] * 7]},
                        {"normalized_actions": [[float("inf")] * 7] * 8}, {"latency_ms": -1}]:
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                validate_response({**valid, **changed}, "expected")


if __name__ == "__main__":
    unittest.main()
