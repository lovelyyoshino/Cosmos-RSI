"""保证仿真并行分区不漏回合、不重复回合、不改变逐回合种子。

@author pony
@date 2026-11-06
@version v1.2.0
@last_modified 2026-11-07
@changelog
    - v1.2.0 (2026-11-07): 真实TCP连接覆盖TIME_WAIT快速重启及现存监听拒绝。
    - v1.1.0 (2026-11-06): 空分区与重复条件在启动仿真前拒绝。
    - v1.0.0 (2026-11-06): 覆盖1/2/4进程的回合完整性与未知条件拒绝。
"""
from __future__ import annotations

import copy
import pathlib
import socket
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from p0_evaluate_policy import partition_eval_cases, policy_request_seed, validate_loopback_port_available


class EvaluationPartitionTests(unittest.TestCase):
    def test_existing_listener_is_not_treated_as_a_free_port(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            with self.assertRaises(OSError):
                validate_loopback_port_available(listener.getsockname()[1])

    def test_normal_server_time_wait_allows_a_safe_restart(self):
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            port = listener.getsockname()[1]
            with socket.create_connection(("127.0.0.1", port), timeout=3) as client:
                connection, _ = listener.accept()
                with connection:
                    connection.shutdown(socket.SHUT_WR)
                    self.assertEqual(client.recv(1), b"")
        validate_loopback_port_available(port)

    def test_noise_streams_do_not_shift_into_the_next_episode_seed(self):
        first = [policy_request_seed(34001, index) for index in range(128)]
        second = [policy_request_seed(34002, index) for index in range(128)]
        self.assertFalse(set(first) & set(second))
        self.assertEqual(first, [policy_request_seed(34001, index) for index in range(128)])
        for seed, index in [(-1, 0), (34001, -1), (2 ** 32, 0)]:
            with self.assertRaises(ValueError):
                policy_request_seed(seed, index)

    def test_all_cases_and_seeds_survive_parallel_partition(self):
        conditions = [{"condition_id": f"c{i}"} for i in range(3)]
        cases = [{"eval_id": f"e{i}", "condition_id": f"c{i % 3}", "init_seed": 100 + i,
                  "policy_seed": 34001 + i % 2} for i in range(11)]
        before = copy.deepcopy(cases)
        for workers in [1, 2, 4]:
            jobs = partition_eval_cases(cases, conditions, workers)
            flattened = [row for job in jobs for row in job["cases"]]
            self.assertEqual(sorted(flattened, key=lambda x: x["eval_id"]), sorted(cases, key=lambda x: x["eval_id"]))
            self.assertEqual(len(flattened), len({x["eval_id"] for x in flattened}))
            for job in jobs:
                self.assertEqual({x["condition_id"] for x in job["cases"]},
                                 {x["condition_id"] for x in job["conditions"]})
        self.assertEqual(cases, before)

    def test_invalid_cases_fail_before_workers_start(self):
        case = {"eval_id": "e1", "condition_id": "c1"}
        with self.assertRaises(ValueError):
            partition_eval_cases([case, case], [{"condition_id": "c1"}], 2)
        with self.assertRaises(ValueError):
            partition_eval_cases([case], [{"condition_id": "c2"}], 2)
        with self.assertRaises(ValueError):
            partition_eval_cases([case], [{"condition_id": "c1"}], 0)
        with self.assertRaises(ValueError):
            partition_eval_cases([], [{"condition_id": "c1"}], 1)
        with self.assertRaises(ValueError):
            partition_eval_cases([case], [{"condition_id": "c1"}, {"condition_id": "c1"}], 1)


if __name__ == "__main__":
    unittest.main()
