#!/usr/bin/env python3
"""把独立准入窗口写为无损 LeRobot v3，并用 starVLA 原生读取器逐窗口回读。

@author pony
@date 2026-11-08
@version v1.1.0
@last_modified 2026-11-08
@changelog
    - v1.1.0 (2026-11-08): 显式区分已准入成功示范与独立运行存档，失败运行不冒充训练示范。
    - v1.0.0 (2026-11-08): PNG/Parquet交接、全标签审计及禁止补帧的原生读取适配。
"""
from __future__ import annotations

import argparse
import io
import json
import pathlib
import sys

from p0_common import ROOT, ensure_gpu_idle, run_stage
from p0_policy_runtime import DemoCache, normalize, sha256, verify_eval_isolation


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def validate_packet(packet, packet_kind="admitted_demo_window"):
    """部分回合保持原始源时刻，局部帧号仅用于新的存储索引。"""
    import numpy as np

    first, stop = packet["window"]["start_frame"], packet["window"]["end_frame"]
    count = stop - first
    if packet_kind not in {"admitted_demo_window", "runtime_rollout"}:
        raise ValueError("未知LeRobot记录类别")
    runtime = packet_kind == "runtime_rollout"
    source_ok = (packet["source_split"] == "runtime" and first == 0 and type(packet["source_success"]) is bool) \
        if runtime else (packet["source_split"] == "demo_train" and packet["source_success"] is True)
    if type(first) is not int or type(stop) is not int or first < 0 or count < (1 if runtime else 8) \
            or packet["is_eval"] is not False or not source_ok or not packet["instruction"].strip():
        raise ValueError("窗口边界、成功来源或训练分区不合法")
    for key, shape, dtype in [("agentview", (count, 256, 256, 3), np.uint8),
                              ("wrist", (count, 256, 256, 3), np.uint8),
                              ("action", (count, 7), np.float32), ("state", (count, 8), np.float32)]:
        array = packet[key]
        if array.shape != shape or array.dtype != dtype or not np.isfinite(array).all():
            raise ValueError(f"{key}的形状、dtype或有限性不合法")
    if not np.array_equal(packet["source_frame_index"], np.arange(first, stop)) \
            or not np.array_equal(packet["source_timestamp"], np.arange(first, stop) / 20):
        raise ValueError("原始源帧号或时间戳错位")


def statistics(array):
    import numpy as np

    return {"min": array.min(axis=0).tolist(), "max": array.max(axis=0).tolist(),
            "mean": array.mean(axis=0).tolist(), "std": array.std(axis=0).tolist(),
            "q01": np.quantile(array, .01, axis=0).tolist(), "q99": np.quantile(array, .99, axis=0).tolist()}


def write_dataset(destination, packets, normalizer, packet_kind="admitted_demo_window"):
    """一源窗口一个部分episode；PNG内嵌于Parquet，避免展示视频的有损编码。"""
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image

    if not packets or len({p["source_episode_id"] for p in packets}) != len(packets):
        raise ValueError("窗口列表为空或源窗口重复")
    for packet in packets:
        validate_packet(packet, packet_kind)
    destination.mkdir(parents=True, exist_ok=False)
    tasks = sorted({p["instruction"] for p in packets})
    indices = {task: index for index, task in enumerate(tasks)}
    image_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    fields = [("observation.images.agentview", image_type), ("observation.images.wrist", image_type),
              ("observation.state", pa.list_(pa.float32(), 8)), ("action", pa.list_(pa.float32(), 7)),
              ("timestamp", pa.float64()), ("frame_index", pa.int64()), ("episode_index", pa.int64()),
              ("index", pa.int64()), ("task_index", pa.int64()),
              ("source_frame_index", pa.int64()), ("source_timestamp", pa.float64())]
    schema = pa.schema(fields)
    episodes, offset = [], 0

    def png(array):
        stream = io.BytesIO()
        Image.fromarray(array).save(stream, format="PNG", compress_level=3)
        return {"bytes": stream.getvalue(), "path": None}

    for index, packet in enumerate(packets):
        count = len(packet["action"])
        columns = {"observation.images.agentview": [png(x) for x in packet["agentview"]],
                   "observation.images.wrist": [png(x) for x in packet["wrist"]],
                   "observation.state": packet["state"].tolist(), "action": packet["action"].tolist(),
                   "timestamp": (np.arange(count) / 20).tolist(), "frame_index": list(range(count)),
                   "episode_index": [index] * count, "index": list(range(offset, offset + count)),
                   "task_index": [indices[packet["instruction"]]] * count,
                   "source_frame_index": packet["source_frame_index"].tolist(),
                   "source_timestamp": packet["source_timestamp"].tolist()}
        path = destination / f"data/chunk-000/file-{index:03d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pydict(columns, schema=schema), path, compression="zstd")
        episodes.append({"episode_index": index, "tasks": [packet["instruction"]], "length": count,
                         "data/chunk_index": 0, "data/file_index": index,
                         "dataset_from_index": offset, "dataset_to_index": offset + count,
                         "source_episode_id": packet["source_episode_id"], "is_eval": False,
                         "source_split": packet["source_split"], "source_window": packet["window"],
                         "source_success": packet["source_success"]})
        offset += count
    metadata = destination / "meta"
    (metadata / "episodes/chunk-000").mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(episodes), metadata / "episodes/chunk-000/file-000.parquet")
    pd.DataFrame({"task_index": range(len(tasks))}, index=tasks).to_parquet(metadata / "tasks.parquet")
    features = {key: {"dtype": "image", "shape": [256, 256, 3], "names": ["height", "width", "channel"]}
                for key in ["observation.images.agentview", "observation.images.wrist"]}
    for key, width in [("observation.state", 8), ("action", 7)]:
        features[key] = {"dtype": "float32", "shape": [width], "names": None}
    for key in ["timestamp", "frame_index", "episode_index", "index", "task_index", "source_frame_index", "source_timestamp"]:
        features[key] = {"dtype": "float64" if "timestamp" in key else "int64", "shape": [1], "names": None}
    dump(metadata / "info.json", {"codebase_version": "v3.0", "robot_type": "panda", "fps": 20,
        "total_episodes": len(packets), "total_frames": offset, "total_tasks": len(tasks),
        "total_videos": 0, "total_chunks": 1, "chunks_size": 1000,
        "splits": {"runtime" if packet_kind == "runtime_rollout" else "train": f"0:{len(packets)}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4", "features": features})
    dump(metadata / "modality.json", {
        "video": {"agentview": {"original_key": "observation.images.agentview"},
                  "wrist": {"original_key": "observation.images.wrist"}},
        "state": {"proprio": {"start": 0, "end": 8, "original_key": "observation.state", "dtype": "float32"}},
        "action": {"raw": {"start": 0, "end": 7, "original_key": "action", "dtype": "float32"}},
        "annotation": {"human.task_description": {"original_key": "task_index"}}})
    stats = {key: statistics(np.concatenate([p[name] for p in packets]))
             for key, name in [("observation.state", "state"), ("action", "action")]}
    dump(metadata / "stats.json", stats)
    dump(metadata / "stats_gr00t.json", {"__format_version": 2, "__cache_config": {"mode": "abs"},
                                         "statistics": stats})
    # 文件自身统计仅供通用读取；训练仍使用正式D0的同一归一化参数。
    # 运行存档可以含失败；分类与成功判定门通过之前不能直接作为教师数据。
    dump(metadata / "rsi_provenance.json", {"schema": "rsi-partial-lerobot-v1", "packet_kind": packet_kind,
        "eligible_for_training": packet_kind == "admitted_demo_window", "episodes": episodes,
        "normalizer": normalizer, "action_horizon": 8, "source_fps": 20,
        "image_encoding": "lossless_png_in_parquet", "training_normalizer_source": "frozen_parent_D0",
        "source_time_rule": "timestamp为局部存储时间；source_timestamp保持源绝对时间",
        "raw_action_semantics": "不变换原始控制动作；GR00T action_mode=abs表示不作额外差分，并非物理绝对坐标"})
    return episodes


def audit_dataset(destination, packets):
    """对实际落盘数据全帧解码，逐值比较双视角、监督、任务和两个时间坐标。"""
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    from PIL import Image

    info = json.loads((destination / "meta/info.json").read_text())
    provenance = json.loads((destination / "meta/rsi_provenance.json").read_text())
    episodes = pq.read_table(destination / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
    tasks = pd.read_parquet(destination / "meta/tasks.parquet")
    if info["codebase_version"] != "v3.0" or info["fps"] != 20 or info["total_episodes"] != len(packets) \
            or len(episodes) != len(packets) or episodes != provenance["episodes"] \
            or info["total_frames"] != sum(len(p["action"]) for p in packets):
        raise ValueError("LeRobot元数据、源记录或帧数错位")
    offset = 0
    for index, packet in enumerate(packets):
        validate_packet(packet, provenance.get("packet_kind", "admitted_demo_window"))
        episode = episodes[index]
        if any(episode[key] != packet[key] for key in ["source_episode_id", "source_split", "source_success", "is_eval"]) \
                or episode["source_window"] != packet["window"] or episode["length"] != len(packet["action"]) \
                or episode["episode_index"] != index or episode["dataset_from_index"] != offset:
            raise ValueError("部分episode来源或偏移发生变化")
        path = destination / f"data/chunk-000/file-{index:03d}.parquet"
        rows = pq.read_table(path).to_pylist()
        count = len(packet["action"])
        if len(rows) != count or episode["dataset_to_index"] != offset + count:
            raise ValueError("实际Parquet帧数错误")
        for name, column in [("action", "action"), ("state", "observation.state")]:
            if not np.array_equal(np.asarray([r[column] for r in rows], dtype=np.float32), packet[name]):
                raise ValueError("落盘动作或状态偏离源标签")
        for frame, row in enumerate(rows):
            if row["frame_index"] != frame or row["episode_index"] != index or row["index"] != offset + frame \
                    or row["timestamp"] != frame / 20 or row["source_frame_index"] != packet["source_frame_index"][frame] \
                    or row["source_timestamp"] != packet["source_timestamp"][frame] \
                    or tasks.index[row["task_index"]] != packet["instruction"]:
                raise ValueError("帧号、任务或时间戳偏离源窗口")
            for name in ["agentview", "wrist"]:
                entry = row[f"observation.images.{name}"]
                if entry["path"] is not None or not entry["bytes"]:
                    raise ValueError("图像没有完整内嵌PNG")
                image = Image.open(io.BytesIO(entry["bytes"]))
                if image.mode != "RGB" or image.size != (256, 256) or image.format != "PNG" \
                        or not np.array_equal(np.asarray(image), packet[name][frame]):
                    raise ValueError("落盘双视角RGB偏离无损源")
        offset += count
    return {"episodes": len(packets), "frames": offset, "decoded_images": 2 * offset,
            "labels_equal": True, "wrist_equal": True, "source_time_equal": True}


class StrictNativeWindows:
    """使用原生get_step_data读真实文件；在其补帧和训练打包之前限制合法动作块。"""
    def __init__(self, path, normalizer):
        sys.path.insert(0, str(ROOT / "third_party/starVLA"))
        from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset, ModalityConfig

        self.provenance = json.loads((path / "meta/rsi_provenance.json").read_text())
        if self.provenance["normalizer"] != normalizer or self.provenance["action_horizon"] != 8:
            raise ValueError("增强数据不能重新拟合归一化或改变动作块长度")
        self.normalizer = normalizer
        self.reader = LeRobotSingleDataset(path, modality_configs={
            "video": ModalityConfig(delta_indices=[0], modality_keys=["video.agentview", "video.wrist"]),
            "state": ModalityConfig(delta_indices=[0], modality_keys=["state.proprio"]),
            "action": ModalityConfig(delta_indices=list(range(8)), modality_keys=["action.raw"]),
            "language": ModalityConfig(delta_indices=[0], modality_keys=["annotation.human.task_description"])},
            embodiment_tag="new_embodiment", video_backend="pyav",
            data_cfg={"lerobot_version": "v3.0", "include_state": True, "action_mode": "abs"})

    def raw_at(self, episode_index, frame):
        episodes = self.provenance["episodes"]
        if type(episode_index) is not int or not 0 <= episode_index < len(episodes):
            raise ValueError("部分episode索引无效")
        episode = episodes[episode_index]
        if type(frame) is not int or frame < 0 or frame + 8 > episode["length"]:
            raise ValueError("动作块越过部分episode，禁止调用上游补帧")
        return self.reader.get_step_data(episode_index, frame)

    def example_at(self, episode_index, frame):
        import numpy as np
        from PIL import Image

        raw = self.raw_at(episode_index, frame)
        return {"image": [Image.fromarray(raw[key][0]) for key in ["video.agentview", "video.wrist"]],
                "lang": raw["annotation.human.task_description"][0],
                "state": normalize(np.asarray(raw["state.proprio"], dtype=np.float32), self.normalizer["state"]),
                "action": normalize(np.asarray(raw["action.raw"], dtype=np.float32), self.normalizer["action"])}


def export(directory, args):
    ensure_gpu_idle()
    import numpy as np
    from p0_lossless_background import read_exact_video, validate_labels
    from p0_uv_window_gate import checked_gate

    parent = json.loads((args.parent / "checkpoint.json").read_text())
    if sha256(args.parent / "state.pt") != parent["state_sha256"] or not parent.get("formal_baseline_training"):
        raise ValueError("只接受已验证正式基线作为P0交接起点")
    cache_path = pathlib.Path(parent["cache_manifest"]).parent
    isolation = verify_eval_isolation(cache_path, ROOT / "eval/frozen_eval_set.json")
    cache = DemoCache(cache_path)
    packets, analytic, bindings = [], [], []
    for gate_path in args.gate:
        gate, render, evidence, *_ = checked_gate(gate_path)
        if gate["source_proof"]["cache_manifest_sha256"] != sha256(cache.manifest_path):
            raise ValueError("准入窗口不是正式基线的训练D0")
        source = next(row for row in cache.manifest["episodes"] if row["episode_id"] == evidence["source_episode_id"])
        first, stop = (evidence["window"][key] for key in ["start_frame", "end_frame"])
        start, end = source["offset"] + first, source["offset"] + stop
        render_dir = pathlib.Path(gate["render_report"]).parent
        with np.load(render_dir / "source_labels.npz", allow_pickle=False) as labels:
            expected = {"action": cache.actions[start:end], "state": cache.states[start:end],
                        "frame_index": np.arange(first, stop), "timestamp": np.arange(first, stop) / 20}
            validate_labels(dict(labels), expected)
        packet = {"source_episode_id": source["episode_id"], "instruction": source["instruction"],
                  "source_split": source["split"], "source_success": source["env_success"], "is_eval": False,
                  "window": {"start_frame": first, "end_frame": stop}, "source_frame_index": expected["frame_index"],
                  "source_timestamp": expected["timestamp"], "action": np.array(cache.actions[start:end]),
                  "state": np.array(cache.states[start:end]), "wrist": np.array(cache.images[start:end, 1]),
                  "agentview": read_exact_video(render_dir / "cosmos.mp4")[0]}
        packets.append(packet)
        analytic.append({**packet, "agentview": read_exact_video(render_dir / "analytic.mp4")[0]})
        bindings.append({"path": str(gate_path), "sha256": sha256(gate_path), "source_episode_id": source["episode_id"],
                         "source_length": source["length"], "window": packet["window"],
                         "target_condition": gate["target_condition"], "output_kind": gate["output_kind"],
                         "derived_kind": gate["derived_kind"], "output_video_sha256": gate["output_video_sha256"]})
    audits = {}
    for variant, values in [("cosmos", packets), ("analytic", analytic)]:
        path = directory / "lerobot" / variant
        write_dataset(path, values, cache.manifest["normalizer"])
        audits[variant] = audit_dataset(path, values)
        native = StrictNativeWindows(path, cache.manifest["normalizer"])
        windows = 0
        for index, packet in enumerate(values):
            for frame in range(len(packet["action"]) - 7):
                raw = native.raw_at(index, frame)
                for key, expected in [("video.agentview", packet["agentview"][frame:frame + 1]),
                                      ("video.wrist", packet["wrist"][frame:frame + 1]),
                                      ("action.raw", packet["action"][frame:frame + 8]),
                                      ("state.proprio", packet["state"][frame:frame + 1])]:
                    if not np.array_equal(raw[key], expected):
                        raise ValueError("starVLA原生读取发生图像或标签漂移")
                if raw["annotation.human.task_description"] != [packet["instruction"]]:
                    raise ValueError("原生任务语言错绑")
                windows += 1
            try:
                native.raw_at(index, len(packet["action"]) - 7)
            except ValueError:
                pass
            else:
                raise ValueError("适配器未拒绝末尾越界动作块")
        audits[variant]["native_raw_windows_checked"] = windows
        example = native.example_at(0, 0)
        if example["action"].dtype != np.float32 or example["state"].dtype != np.float32 \
                or any(image.size != (256, 256) for image in example["image"]):
            raise ValueError("训练适配器擅自改变精度或分辨率")
    payload = {"schema": "thor-admitted-lerobot-handoff-v1", "parent_checkpoint": str(args.parent),
        "parent_ckpt_sha256": parent["state_sha256"], "eval_overlap_check": isolation,
        "cache_manifest": str(cache.manifest_path), "cache_manifest_sha256": sha256(cache.manifest_path),
        "normalizer": cache.manifest["normalizer"], "admitted_windows": bindings, "audits": audits,
        "datasets": {variant: str(directory / "lerobot" / variant) for variant in audits},
        "files_sha256": {str(p.relative_to(directory)): sha256(p) for p in sorted((directory / "lerobot").rglob("*")) if p.is_file()},
        "source_windows": len(packets), "training_performed": False, "policy_gain_evaluated": False,
        "scope": "P0部分episode真实格式交接；只覆盖已准入桌体派生，不代表批量生成或闭环收益"}
    dump(directory / "handoff.json", payload)
    return {key: payload[key] for key in ["source_windows", "audits", "training_performed", "policy_gain_evaluated"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--parent", type=pathlib.Path, required=True)
    parser.add_argument("--gate", type=pathlib.Path, action="append", required=True)
    arguments = parser.parse_args()
    arguments.parent = arguments.parent.resolve(strict=True)
    arguments.gate = [p.resolve(strict=True) for p in arguments.gate]
    if not arguments.parent.is_relative_to(ROOT / "runs/p0") or any(not p.is_relative_to(ROOT / "runs/p0") for p in arguments.gate):
        parser.error("交接来源须位于Thor项目P0目录")
    raise SystemExit(run_stage("lerobot_handoff", lambda directory: export(directory, arguments),
        {"compute": "Thor CPU无损PNG/Parquet及starVLA原生读取，无训练", "precision": "RGB uint8、标签fp32、时间fp64",
         "batch": None, "steps": 0, "seed": None, "source_windows": len(arguments.gate), "fps": 20, "horizon": 8}))
