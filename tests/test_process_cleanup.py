"""验证后台任务leader先退出时，其遗留后代仍会被本次进程组清理。

@author pony
@date 2026-11-06
@version v1.0.0
@last_modified 2026-11-06
@changelog
    - v1.0.0 (2026-11-06): 实际子进程验证孤立后代回收，不触碰其他进程组。
"""
from __future__ import annotations

import pathlib
import signal
import subprocess
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from p0_common import stop_process_group


class ProcessCleanupTests(unittest.TestCase):
    def test_descendant_is_stopped_even_after_its_leader_exited(self):
        code = ("import subprocess,sys; "
                "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)'], "
                "stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); "
                "print(child.pid,flush=True)")
        process = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True, start_new_session=True)
        try:
            child_pid = int(process.stdout.readline())
            process.wait(timeout=5)
            result = stop_process_group(process, grace_s=1)
            self.assertEqual(result["remaining_live_pids"], [])
            stat_path = pathlib.Path(f"/proc/{child_pid}/stat")
            if stat_path.exists():
                self.assertEqual(stat_path.read_text().rsplit(")", 1)[1].split()[0], "Z")
        finally:
            stop_process_group(process, grace_s=0)
            process.stdout.close()


if __name__ == "__main__":
    unittest.main()
