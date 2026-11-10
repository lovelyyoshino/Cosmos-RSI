"""核查独立采集隔离和真实困难信号，不加载模型或仿真。

@author pony
@date 2026-11-08
@version v1.1.0
@last_modified 2026-11-09
@changelog
    - v1.1.0 (2026-11-09): 确定分区、顺序归并及真实CPU进程组失败/超时回收。
    - v1.0.0 (2026-11-08): 覆盖种子交集、观测内动作离散度、非法样本和配对采集预算。
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
import pathlib
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from p0_common import stop_process_group
from p0_feedback_rollout import (action_dispersion, build_cases, object_scene, order_feedback_rows,
                                 partition_feedback_cases, run_collection_workers)


class FeedbackRolloutTests(unittest.TestCase):
    def cases(self, **kwargs):
        return build_cases([{"condition_id": name} for name in ["standard", "lighting"]],
            {"episodes": [{"init_seed": 21, "policy_seed": 31}]},
            {"episodes": [{"init_seed": 0}]},
            **{"init_seed": 61000, "policy_seed": 71000, "count": 2, "max_steps": 520, **kwargs})

    def test_pairs_share_declared_seeds_without_using_frozen_cases(self):
        result = self.cases()
        self.assertEqual(len(result), 4)
        self.assertEqual(result, self.cases())
        self.assertEqual([r["init_seed"] for r in result], [61000, 61001, 61000, 61001])
        self.assertTrue(all(r["is_eval"] is False and r["max_steps"] == 520 for r in result))
        self.assertEqual(len({r["case_id"] for r in result}), 4)

    def test_seed_intersections_and_invalid_budgets_are_rejected(self):
        for change in [{"init_seed": 0}, {"init_seed": 20}, {"policy_seed": 30},
                       {"count": 0}, {"max_steps": 0}, {"init_seed": True},
                       {"policy_seed": 2 ** 32 - 1}]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.cases(**change)

    def test_dispersion_is_population_rms_std_for_one_observation(self):
        samples = np.zeros((3, 8, 7))
        self.assertEqual(action_dispersion(samples), 0)
        samples[:, :, :] = np.arange(3)[:, None, None]
        self.assertAlmostEqual(action_dispersion(samples), np.sqrt(2 / 3))
        self.assertAlmostEqual(action_dispersion(samples + 100), np.sqrt(2 / 3))

    def test_missing_or_nonfinite_action_samples_do_not_become_zero(self):
        for samples in [np.zeros((1, 8, 7)), np.zeros((3, 7, 7)), np.full((3, 8, 7), np.nan)]:
            with self.subTest(shape=samples.shape), self.assertRaises(ValueError):
                action_dispersion(samples)

    def test_scene_isolation_excludes_robot_and_gripper_joints(self):
        scene = {"robot0_joint": [1], "gripper0_finger": [2], "book_joint": [3]}
        self.assertEqual(object_scene(scene), {"book_joint": [3]})


FAKE_WORKER = """
import json, os, pathlib, subprocess, sys, time
spec = json.loads(pathlib.Path(sys.argv[1]).read_text())
output = pathlib.Path(spec['output'])
mode = spec['test_mode']
processes = {'pid': os.getpid(), 'pgid': os.getpgrp()}
if mode != 'success':
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    processes['child_pid'] = child.pid
(output / 'processes.json').write_text(json.dumps(processes))
if mode == 'failure' and spec['worker_index'] == 1:
    peer = output.parent / 'worker_0/processes.json'
    deadline = time.monotonic() + 5
    while not peer.exists() and time.monotonic() < deadline:
        time.sleep(.01)
    if not peer.exists():
        sys.exit(8)
    sys.exit(7)
if mode != 'success':
    time.sleep(60)
else:
    rows = [{**case, 'episode_id': spec['run_id'] + '-' + case['case_id']}
            for case in reversed(spec['cases'])]
    time.sleep(.02 * (spec['worker_count'] - spec['worker_index']))
    (output / 'episodes.jsonl').write_text(''.join(json.dumps(row) + '\\n' for row in rows))
"""


class FeedbackParallelTests(unittest.TestCase):
    def fixture(self, count=3):
        conditions = [{"condition_id": name} for name in ["standard", "lighting", "background"]]
        cases = build_cases(conditions, {"episodes": []}, {"episodes": []},
                            init_seed=61000, policy_seed=71000, count=count, max_steps=520)
        return conditions, cases

    def test_partitions_preserve_plan_values_seeds_and_condition_order(self):
        conditions, cases = self.fixture()
        for count in range(1, 5):
            jobs = partition_feedback_cases(cases, conditions, count)
            self.assertEqual(len(jobs), count)
            self.assertEqual(jobs, partition_feedback_cases(cases, conditions, count))
            for index, job in enumerate(jobs):
                self.assertEqual(job["cases"], cases[index::count])
                used = {row["condition_id"] for row in job["cases"]}
                self.assertEqual(job["conditions"], [row for row in conditions if row["condition_id"] in used])
            flattened = [row for job in jobs for row in job["cases"]]
            self.assertEqual(len({row["case_id"] for row in flattened}), len(cases))
        self.assertEqual(partition_feedback_cases(cases, conditions, 1),
                         [{"cases": cases, "conditions": conditions}])

    def test_worker_limit_is_capped_by_cases_and_invalid_plans_are_rejected(self):
        conditions, cases = self.fixture(1)
        self.assertEqual(len(partition_feedback_cases(cases[:1], conditions, 4)), 1)
        for value in [0, 5, True, 1.5]:
            with self.subTest(workers=value), self.assertRaises(ValueError):
                partition_feedback_cases(cases, conditions, value)
        for values in [([], conditions), (cases + cases[:1], conditions),
                       (cases, conditions + conditions[:1]),
                       ([{**cases[0], "condition_id": "unknown"}], conditions)]:
            with self.subTest(plan=values), self.assertRaises(ValueError):
                partition_feedback_cases(*values, 4)

    def test_merge_uses_declared_case_order_and_rejects_duplicate_or_rebound_rows(self):
        _, cases = self.fixture()
        rows = [{**case, "episode_id": "run-" + case["case_id"]} for case in cases]
        self.assertEqual(order_feedback_rows(rows[::-1], cases, "run"), rows)
        for bad in [rows[:-1], rows + rows[:1], [{**rows[0], "episode_id": "foreign"}] + rows[1:],
                    [{**rows[0], "policy_seed": 123}] + rows[1:]]:
            with self.subTest(rows=bad), self.assertRaises(ValueError):
                order_feedback_rows(bad, cases, "run")

    @contextmanager
    def workers(self, mode, workers=2, fail_spawn=False):
        conditions, cases = self.fixture(2)
        jobs = partition_feedback_cases(cases, conditions, workers)
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            (directory / "logs").mkdir()
            processes, logs = [], []

            def launch(arguments, **kwargs):
                self.assertIs(kwargs["start_new_session"], True)
                logs.append(kwargs["stdout"])
                if fail_spawn and len(logs) == 2:
                    raise OSError("test spawn failure")
                process = subprocess.Popen([sys.executable, "-u", "-c", FAKE_WORKER, arguments[-1]], **kwargs)
                processes.append(process)
                return process

            fake_subprocess = SimpleNamespace(Popen=launch, STDOUT=subprocess.STDOUT)
            try:
                with patch("p0_feedback_rollout.subprocess", fake_subprocess):
                    yield directory, jobs, {"run_id": directory.name, "test_mode": mode}, processes, logs
            finally:
                for process in processes:
                    stop_process_group(process, grace_s=1)

    def assert_groups_reclaimed(self, directory, processes):
        report = json.loads((directory / "LIBERO_workers.json").read_text())
        self.assertEqual(report["actual_workers"], len(processes))
        for row in report["workers"]:
            self.assertEqual(row["cleanup"]["remaining_live_pids"], [])
            self.assertEqual(row["cleanup"]["pgid"], row["pid"])
            self.assertIsNotNone(row["returncode"])
        for path in directory.rglob("processes.json"):
            recorded = json.loads(path.read_text())
            self.assertEqual(recorded["pid"], recorded["pgid"])
            self.assertNotEqual(recorded["pgid"], os.getpgrp())
            if "child_pid" in recorded:
                stat = pathlib.Path(f"/proc/{recorded['child_pid']}/stat")
                if stat.exists():
                    self.assertEqual(stat.read_text().rpartition(")")[2].split()[0], "Z")
        return report

    def test_real_cpu_workers_preserve_single_worker_layout_and_merge_order(self):
        for count in [1, 4]:
            with self.subTest(workers=count), self.workers("success", count) as (directory, jobs, spec, processes, logs):
                rows = run_collection_workers(directory, "LIBERO", spec, jobs, SimpleNamespace(poll=lambda: None), 5)
                _, cases = self.fixture(2)
                ordered = order_feedback_rows(rows, cases, spec["run_id"])
                self.assertEqual([row["case_id"] for row in ordered], [case["case_id"] for case in cases])
                report = self.assert_groups_reclaimed(directory, processes)
                self.assertTrue(all(row["wall_s"] > 0 for row in report["workers"]))
                self.assertTrue(all(log.closed for log in logs))
                if count == 1:
                    self.assertTrue((directory / "LIBERO/worker_spec.json").is_file())
                    self.assertTrue((directory / "logs/LIBERO_worker.log").is_file())

    def test_failure_reclaims_live_peer_and_orphan_after_leader_exit(self):
        with self.workers("failure") as (directory, jobs, spec, processes, logs):
            with self.assertRaisesRegex(RuntimeError, "worker失败"):
                run_collection_workers(directory, "LIBERO", spec, jobs, SimpleNamespace(poll=lambda: None), 5)
            report = self.assert_groups_reclaimed(directory, processes)
            self.assertEqual(report["workers"][1]["returncode"], 7)
            self.assertTrue(all(log.closed for log in logs))

    def test_timeout_reclaims_all_worker_process_groups(self):
        with self.workers("timeout") as (directory, jobs, spec, processes, logs):
            with self.assertRaisesRegex(TimeoutError, "停止本次全部worker"):
                run_collection_workers(directory, "LIBERO", spec, jobs, SimpleNamespace(poll=lambda: None), .4)
            self.assert_groups_reclaimed(directory, processes)
            self.assertTrue(all(log.closed for log in logs))

    def test_spawn_failure_reclaims_already_started_worker_and_closes_logs(self):
        with self.workers("timeout", fail_spawn=True) as (directory, jobs, spec, processes, logs):
            with self.assertRaisesRegex(OSError, "test spawn failure"):
                run_collection_workers(directory, "LIBERO", spec, jobs, SimpleNamespace(poll=lambda: None), 5)
            self.assertEqual(len(processes), 1)
            self.assert_groups_reclaimed(directory, processes)
            self.assertTrue(all(log.closed for log in logs))

    def test_policy_server_failure_reclaims_all_workers(self):
        with self.workers("timeout") as (directory, jobs, spec, processes, logs):
            with self.assertRaisesRegex(RuntimeError, "策略服务在worker运行期间退出"):
                run_collection_workers(directory, "LIBERO", spec, jobs, SimpleNamespace(poll=lambda: 3), 5)
            self.assert_groups_reclaimed(directory, processes)
            self.assertTrue(all(log.closed for log in logs))


if __name__ == "__main__":
    unittest.main()
