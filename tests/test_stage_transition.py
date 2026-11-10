"""阶段封存早于进程/CUDA退出时必须等待；其他占用、PID复用和失败查询继续阻断。

@author pony
@date 2026-11-09
@version v1.0.0
@last_modified 2026-11-09
@changelog
    - v1.0.0 (2026-11-09): 复现生成封存后的退出竞态，并验证不绕过GPU检查。
"""
from __future__ import annotations

import pathlib
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import p1_stage_transition as transition


class StageTransitionTests(unittest.TestCase):
    binding = {"pid": 727420, "command": ["synthetic-python", "synthetic-generation.py"]}

    def test_sealed_predecessor_and_stale_cuda_entry_are_both_waited(self):
        queries = [{"returncode": 0, "stdout": text} for text in ["727420, [No data]", "727420, [No data]", ""]]
        with patch.object(transition, "process_command", side_effect=[self.binding["command"], [], []]), \
                patch.object(transition, "command", side_effect=queries), patch.object(transition.time, "sleep") as sleep:
            result = transition.wait_stage_teardown(self.binding)
            self.assertTrue(result["process_exited"])
            self.assertTrue(result["gpu_idle"])
            self.assertEqual(result["processes_stopped"], 0)
            self.assertEqual(sleep.call_count, 2)

    def test_another_gpu_process_is_never_waited_or_ignored(self):
        with patch.object(transition, "process_command", return_value=[]), patch.object(transition, "command",
                return_value={"returncode": 0, "stdout": "999999, other"}), \
                patch.object(transition.time, "sleep") as sleep, self.assertRaises(RuntimeError):
            transition.wait_stage_teardown(self.binding)
        sleep.assert_not_called()

    def test_pid_reuse_queries_failures_or_unparseable_processes_block(self):
        for mutation in ["reuse", "query", "parse", "negative"]:
            actual = ["another-process"] if mutation == "reuse" else []
            query = {"returncode": 1, "stdout": ""} if mutation == "query" else \
                {"returncode": 0, "stdout": "unknown" if mutation == "parse" else "-1, invalid"}
            with self.subTest(mutation=mutation), patch.object(transition, "process_command", return_value=actual), \
                    patch.object(transition, "command", return_value=query), self.assertRaises(RuntimeError):
                transition.wait_stage_teardown(self.binding)

    def test_live_predecessor_cannot_pass_on_empty_gpu_query_and_times_out(self):
        with patch.object(transition, "process_command", return_value=self.binding["command"]), \
                patch.object(transition, "command", return_value={"returncode": 0, "stdout": ""}), \
                patch.object(transition.time, "monotonic", side_effect=[0., 2.]), self.assertRaises(TimeoutError):
            transition.wait_stage_teardown(self.binding, 1.)

    def test_already_exited_or_zombie_can_pass_only_with_empty_gpu(self):
        for actual in [[], [""]]:
            with self.subTest(actual=actual), patch.object(transition, "process_command", return_value=actual), \
                    patch.object(transition, "command", return_value={"returncode": 0, "stdout": ""}):
                self.assertTrue(transition.wait_stage_teardown(self.binding)["gpu_idle"])


if __name__ == "__main__":
    unittest.main()
