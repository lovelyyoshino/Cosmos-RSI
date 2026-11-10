"""实际PNG/Parquet交接的标签、时间、任务与双视角损坏回归。

@author pony
@date 2026-11-08
@version v1.1.0
@last_modified 2026-11-08
@changelog
    - v1.1.0 (2026-11-08): 失败运行可显式存档，不能通过默认成功示范入口或变更环境真值。
    - v1.0.0 (2026-11-08): 跨部分回合来源、错帧、动作及腕部像素损坏均须被检出。
"""
from __future__ import annotations

import copy
import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from p0_lerobot_handoff import StrictNativeWindows, audit_dataset, validate_packet, write_dataset


def packet(index):
    first, count = 23 + index, 10
    return {"source_episode_id": f"train-{index}", "instruction": f"instruction {index}",
            "source_split": "demo_train", "source_success": True, "is_eval": False,
            "window": {"start_frame": first, "end_frame": first + count},
            "source_frame_index": np.arange(first, first + count),
            "source_timestamp": np.arange(first, first + count) / 20,
            "action": np.arange(count * 7, dtype=np.float32).reshape(count, 7) + index * 100,
            "state": np.arange(count * 8, dtype=np.float32).reshape(count, 8) + index * 100,
            "agentview": np.full((count, 256, 256, 3), 71 + index, np.uint8),
            "wrist": np.full((count, 256, 256, 3), 153 + index, np.uint8)}


class HandoffValidationTests(unittest.TestCase):
    def test_failed_runtime_requires_explicit_archive_mode(self):
        runtime = {**packet(0), "source_split": "runtime", "source_success": False,
                   "window": {"start_frame": 0, "end_frame": 10},
                   "source_frame_index": np.arange(10), "source_timestamp": np.arange(10) / 20}
        validate_packet(runtime, "runtime_rollout")
        with self.assertRaises(ValueError):
            validate_packet(runtime)
        with self.assertRaises(ValueError):
            validate_packet({**runtime, "is_eval": True}, "runtime_rollout")

    def test_rejects_eval_wrong_labels_and_lost_source_offset(self):
        for key, value in [("is_eval", True), ("source_success", False), ("source_split", "demo_validation"),
                           ("source_timestamp", np.arange(10) / 20), ("source_frame_index", np.arange(10)),
                           ("action", np.zeros((10, 7), np.float16))]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_packet({**packet(0), key: value})

    def test_native_adapter_rejects_padding_before_calling_reader(self):
        adapter = object.__new__(StrictNativeWindows)
        adapter.provenance = {"episodes": [{"length": 93}]}
        adapter.reader = None
        for episode, frame in [(0, 86), (0, -1), (0, True), (-1, 0), (1, 0), (True, 0)]:
            with self.subTest(episode=episode, frame=frame), self.assertRaises(ValueError):
                adapter.raw_at(episode, frame)


@unittest.skipUnless(importlib.util.find_spec("pyarrow"), "需要Thor已有的pyarrow，lenovo不安装实验依赖")
class HandoffFileTests(unittest.TestCase):
    def test_failed_runtime_stays_failed_and_out_of_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "runtime"
            value = {**packet(0), "source_split": "runtime", "source_success": False,
                     "window": {"start_frame": 0, "end_frame": 10},
                     "source_frame_index": np.arange(10), "source_timestamp": np.arange(10) / 20}
            write_dataset(path, [value], {}, packet_kind="runtime_rollout")
            self.assertEqual(audit_dataset(path, [value])["frames"], 10)
            provenance = json.loads((path / "meta/rsi_provenance.json").read_text())
            self.assertFalse(provenance["eligible_for_training"])
            self.assertFalse(provenance["episodes"][0]["source_success"])
            with self.assertRaises(ValueError):
                audit_dataset(path, [{**value, "source_success": True}])

    def test_roundtrip_two_partial_episodes_keeps_absolute_source_time(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "lerobot"
            packets = [packet(0), packet(1)]
            write_dataset(path, packets, {"fixture": True})
            audit = audit_dataset(path, packets)
            self.assertEqual(audit["frames"], 20)
            self.assertEqual(audit["decoded_images"], 40)
            self.assertTrue(audit["source_time_equal"])
            with self.assertRaises(FileExistsError):
                write_dataset(path, packets, {"fixture": True})

    def test_actual_parquet_corruptions_cannot_pass_full_audit(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        def wrong_action(rows):
            rows[2]["action"][4] += .125

        def swapped_wrist(rows):
            rows[2]["observation.images.wrist"] = rows[2]["observation.images.agentview"]

        changes = {"action": wrong_action, "wrist": swapped_wrist,
                   "task": lambda rows: rows[2].update(task_index=1),
                   "source_time": lambda rows: rows[2].update(source_timestamp=0.),
                   "episode": lambda rows: rows[2].update(episode_index=1),
                   "local_frame": lambda rows: rows[2].update(frame_index=3)}
        for name, change in changes.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                path = pathlib.Path(temporary) / "lerobot"
                packets = [packet(0), packet(1)]
                write_dataset(path, packets, {})
                target = path / "data/chunk-000/file-000.parquet"
                table = pq.read_table(target)
                rows = copy.deepcopy(table.to_pylist())
                change(rows)
                pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), target)
                with self.assertRaisesRegex(ValueError, "偏离|错|落盘"):
                    audit_dataset(path, packets)

    def test_metadata_false_source_and_missing_frames_are_rejected(self):
        import pyarrow.parquet as pq

        for corruption in ["source", "frames"]:
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as temporary:
                path = pathlib.Path(temporary) / "lerobot"
                packets = [packet(0)]
                write_dataset(path, packets, {})
                if corruption == "source":
                    target = path / "meta/rsi_provenance.json"
                    value = json.loads(target.read_text())
                    value["episodes"][0]["source_episode_id"] = "wrong-source"
                    target.write_text(json.dumps(value))
                else:
                    target = path / "data/chunk-000/file-000.parquet"
                    pq.write_table(pq.read_table(target).slice(1), target)
                with self.assertRaises(ValueError):
                    audit_dataset(path, packets)


if __name__ == "__main__":
    unittest.main()
