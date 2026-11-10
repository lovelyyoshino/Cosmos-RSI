"""队列状态只接受确切进程与封存来源；人工夹具不构成真实模型实验。

@author pony
@date 2026-11-09
@version v1.0.0
@last_modified 2026-11-09
@changelog
    - v1.0.0 (2026-11-09): 覆盖PID错绑、上游漂移、源码变化、统计拒绝与部分日志边界。
"""
import json
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from p1_workflow_status import checksum, evaluation_progress, inspect_job, logged_run


class WorkflowStatusTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = pathlib.Path(temporary.name).resolve()
        self.upstream = self.root / "runs/p0/synthetic_upstream"
        self.upstream.mkdir(parents=True)
        self.execution = self.root / "runs/p0/synthetic_exec"
        (self.execution / "logs").mkdir(parents=True)
        self.dependency = self.root / "scripts/synthetic.py"
        self.dependency.parent.mkdir()
        self.dependency.write_text("synthetic source")
        self.log = self.execution / "logs/status.log"
        self.log.write_text(json.dumps({"phase": "waiting_for_exact_upstream", "inputs": {
            "files": {"source": {"path": str(self.dependency), "sha256": checksum(self.dependency)}}}}) + "\n")
        (self.execution / "launch.json").write_text(json.dumps({"pid": 123, "log": str(self.log)}))
        self.specification = {"id": "test", "label": "人工夹具", "script": "scripts/synthetic.py",
            "execution": str(self.execution.relative_to(self.root)), "log": "logs/status.log",
            "stage": "synthetic_stage", "artifact": "report.json", "upstream": "upstream",
            "upstream_flag": "--after-upstream"}
        self.command = ["python", str(self.dependency), "--after-upstream", str(self.upstream)]

    def inspect(self, command=None):
        return inspect_job(self.root, self.specification, self.upstream,
                           process_reader=lambda pid: self.command if command is None else command)

    def seal(self, *, failed=False):
        run = self.root / "runs/p0/synthetic_result"
        run.mkdir()
        report = run / "report.json"
        report.write_text(json.dumps({"decision": "reject", "complete_rsi": False}))
        (run / "probe.json").write_text(json.dumps({"stage": "synthetic_stage",
            "status": "failed" if failed else "passed", "error": "synthetic failure" if failed else None,
            "artifacts_sha256": {"report.json": checksum(report)}}))
        with self.log.open("a") as stream:
            stream.write(f"记录目录：{run}\n")
        return run, report

    def test_waiting_requires_actual_script_upstream_and_input_bindings(self):
        row = self.inspect()
        self.assertEqual(row["status"], "waiting")
        self.assertEqual(row["checked_waiting_bindings"], 1)
        self.assertEqual(self.inspect([])["status"], "invalid")
        self.assertEqual(self.inspect(["python", "unrelated.py", *self.command[-2:]])["status"], "invalid")
        self.assertEqual(self.inspect([*self.command[:-1], str(self.execution)])["status"], "invalid")

    def test_changed_source_and_missing_waiting_bindings_are_invalid(self):
        self.dependency.write_text("changed synthetic source")
        self.assertEqual(self.inspect()["status"], "invalid")
        self.log.write_text('{"phase":"waiting_for_exact_upstream","inputs":{}}\n')
        self.assertEqual(self.inspect()["status"], "invalid")

    def test_scientific_rejection_is_sealed_without_claiming_improvement(self):
        _, report = self.seal()
        row = self.inspect([])
        self.assertEqual(row["status"], "sealed")
        self.assertEqual(row["decision"], "reject")
        self.assertNotIn("complete_rsi", row)
        report.write_text('{"decision":"accept_pending_approval"}')
        self.assertEqual(self.inspect([])["status"], "invalid")

    def test_failed_probe_is_failure_even_if_process_exited(self):
        self.seal(failed=True)
        row = self.inspect([])
        self.assertEqual(row["status"], "failed")
        self.assertIn("synthetic failure", row["error"])

    def test_log_cannot_select_latest_among_multiple_run_directories(self):
        self.seal()
        with self.log.open("a") as stream:
            stream.write(f"记录目录：{self.upstream}\n")
        with self.assertRaises(ValueError):
            logged_run(self.root, self.log)

    def test_unsealed_iteration_requires_live_exact_pid_and_all_bound_inputs(self):
        state = {"phase": "evaluate_control", "iteration": 1, "inputs": {
            key: {"path": str(self.dependency)} for key in ["parent", "handoff", "incumbent_evaluation", "budget"]}}
        (self.upstream / "iteration_state.json").write_text(json.dumps(state))
        spec = {"id": "iteration", "label": "人工整圈夹具", "run": str(self.upstream),
                "script": "scripts/synthetic.py", "pid": 456, "stage": "synthetic_stage", "artifact": "report.json"}
        command = ["python", str(self.dependency), "--iteration", "1"]
        for name in state["inputs"]:
            command.extend(["--" + name.replace("_", "-"), str(self.dependency)])
        row = inspect_job(self.root, spec, process_reader=lambda pid: command)
        self.assertEqual(row["status"], "running_unsealed")
        self.assertEqual(inspect_job(self.root, spec, process_reader=lambda pid: [])["status"], "invalid")
        command[-1] = str(self.upstream)
        self.assertEqual(inspect_job(self.root, spec, process_reader=lambda pid: command)["status"], "invalid")

    def test_progress_counts_unique_frozen_records_and_tolerates_only_unfinished_tail(self):
        iteration = self.root / "runs/p0/synthetic_iteration"
        (iteration / "logs").mkdir(parents=True)
        (iteration / "iteration_state.json").write_text('{"phase":"evaluate_control"}')
        (iteration / "logs/evaluate_control.log").write_text(f"记录目录：{self.upstream}\n")
        evaluation = self.root / "eval/frozen_eval_set.json"
        evaluation.parent.mkdir()
        evaluation.write_text(json.dumps({"episodes": [{"eval_id": "synthetic-a"}, {"eval_id": "synthetic-b"}]}))
        worker = self.upstream / "LIBERO/worker_0/episodes.jsonl"
        worker.parent.mkdir(parents=True)
        first = json.dumps({"eval_id": "synthetic-a", "is_eval": True}) + "\n"
        worker.write_text(first + '{"eval_id":')
        result = evaluation_progress(self.root, iteration)
        self.assertEqual(result["completed_episode_records"], 1)
        self.assertEqual(result["planned_episode_records"], 2)
        self.assertEqual(result["incomplete_trailing_lines"], 1)
        self.assertFalse(result["success_rates_reported"])
        worker.write_text(first + first)
        with self.assertRaises(ValueError):
            evaluation_progress(self.root, iteration)
        worker.write_text(first + '{"eval_id":\n')
        with self.assertRaises(ValueError):
            evaluation_progress(self.root, iteration)


if __name__ == "__main__":
    unittest.main()
