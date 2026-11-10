#!/usr/bin/env python3
"""核验完整LIBERO10示范并生成只读预解码缓存，不接触仿真评测回合。

@author pony
@date 2026-11-06
@version v1.1.0
@last_modified 2026-11-06
@changelog
    - v1.1.0 (2026-11-06): 大文件分配前校验全部回合帧号覆盖，图像解码按小批次读取。
    - v1.0.0 (2026-11-06): 逐帧核验回合、任务与边界，训练归一化只使用训练示范。
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import pathlib
import shutil

from p0_common import ROOT, run_stage


def checksum(path: pathlib.Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(run_dir: pathlib.Path, args: argparse.Namespace) -> dict:
    import numpy as np
    import pyarrow as arrow
    import pyarrow.compute as pc
    import pyarrow.parquet as parquet
    from PIL import Image

    selection = json.loads(args.selection.read_text())
    source_record = json.loads((args.selection.parent / "dataset_manifest.json").read_text())
    selection_sha256 = checksum(args.selection)
    source_root = pathlib.Path(source_record["path"])
    files = source_record["files"]
    snapshot = hashlib.sha256(json.dumps(sorted((f["file"], f["bytes"], f["sha256"]) for f in files),
                                         separators=(",", ":")).encode()).hexdigest()
    for item in files:
        path = source_root / item["file"]
        if not path.resolve().is_relative_to(ROOT / "data") or path.stat().st_size != item["bytes"] or checksum(path) != item["sha256"]:
            raise RuntimeError(f"源示范内容与已校验下载清单不符：{item['file']}")
    episodes = sorted(selection["episodes"], key=lambda row: row["episode_index"])
    if len({episode["episode_index"] for episode in episodes}) != len(episodes):
        raise RuntimeError("选择清单包含重复回合ID")
    # 每个任务独立留出示范；该集合用于离线验证，与冻结的仿真成功率评测集不同。
    validation_ids = set()
    for task in selection["tasks"]:
        members = [episode for episode in episodes if episode["tasks"] == [task]]
        members.sort(key=lambda episode: hashlib.sha256(f"p0-demo-holdout-v1:42:{episode['episode_index']}".encode()).hexdigest())
        validation_ids.update(episode["episode_index"] for episode in members[:max(1, round(len(members) * 0.1))])
    count = sum(episode["length"] for episode in episodes)
    if count != selection["expected_frames"]:
        raise RuntimeError("选择清单的总帧数不一致")
    # 元数据曾因上游重分片而失效；完整性必须先用实际行验证，再分配图像缓存。
    expected = {episode["episode_index"]: episode for episode in episodes}
    coverage = {index: set() for index in expected}
    for file in selection["data_files"]:
        rows = parquet.read_table(source_root / file, columns=["episode_index", "frame_index", "index"]).to_pylist()
        for row in rows:
            index = row["episode_index"]
            if index not in expected:
                continue
            frame = row["frame_index"]
            episode = expected[index]
            if not 0 <= frame < episode["length"] or frame in coverage[index]:
                raise RuntimeError(f"实际分片中回合帧越界或重复：{index}/{frame}")
            if "dataset_from_index" in episode and row["index"] != episode["dataset_from_index"] + frame:
                raise RuntimeError(f"全局帧号与回合偏移不一致：{index}/{frame}")
            coverage[index].add(frame)
    missing = {index: expected[index]["length"] - len(frames) for index, frames in coverage.items()
               if len(frames) != expected[index]["length"]}
    if missing:
        raise RuntimeError(f"实际分片缺少{sum(missing.values())}帧、涉及{len(missing)}回合，拒绝分配训练缓存")
    required_bytes = count * (2 * 256 * 256 * 3 + (8 + 7) * 4 + 8)
    if shutil.disk_usage(ROOT).free < required_bytes + 20 * 1024 ** 3:
        raise RuntimeError("预解码缓存后将不足20GiB磁盘工作空间")
    cache_id = hashlib.sha256(f"p0-cache-v2:{snapshot}:{selection_sha256}".encode()).hexdigest()
    destination = ROOT / "data" / f"libero10_cache_{cache_id[:12]}"
    if destination.exists():
        raise RuntimeError(f"完整缓存目录已存在，请核对并复用，不覆盖：{destination}")
    staging = destination.with_name(destination.name + ".building_" + run_dir.name)
    staging.mkdir()
    images = np.lib.format.open_memmap(staging / "images.npy", mode="w+", dtype=np.uint8, shape=(count, 2, 256, 256, 3))
    states = np.lib.format.open_memmap(staging / "states.npy", mode="w+", dtype=np.float32, shape=(count, 8))
    actions = np.lib.format.open_memmap(staging / "actions.npy", mode="w+", dtype=np.float32, shape=(count, 7))
    timestamps = np.lib.format.open_memmap(staging / "timestamps.npy", mode="w+", dtype=np.float64, shape=(count,))
    filled = np.zeros(count, dtype=bool)
    positions = {}
    offset = 0
    for episode in episodes:
        positions[episode["episode_index"]] = (offset, episode)
        offset += episode["length"]
    task_rows = parquet.read_table(source_root / "meta/tasks.parquet").to_pylist()
    task_text = {row["task_index"]: row.get("task") or row.get("__index_level_0__") for row in task_rows}
    checked = 0
    for file in selection["data_files"]:
        for batch in parquet.ParquetFile(source_root / file).iter_batches(batch_size=128):
            table = arrow.Table.from_batches([batch])
            table = table.filter(pc.is_in(table["episode_index"], value_set=arrow.array(sorted(positions))))
            for row in table.to_pylist():
                start, episode = positions[row["episode_index"]]
                frame = row["frame_index"]
                if frame < 0 or frame >= episode["length"] or task_text[row["task_index"]] != episode["tasks"][0]:
                    raise RuntimeError("帧号或任务标签与所选回合不一致")
                flat = start + frame
                if filled[flat]:
                    raise RuntimeError("同一回合帧出现多次，禁止静默覆盖")
                action = np.asarray(row["action"], dtype=np.float32)
                state = np.asarray(row["observation.state"], dtype=np.float32)
                if action.shape != (7,) or state.shape != (8,) or not np.isfinite(action).all() or not np.isfinite(state).all():
                    raise RuntimeError("动作或状态的维度/有限性不合格")
                for view, key in enumerate(["observation.images.image", "observation.images.image2"]):
                    value = row[key]
                    if not value or not value.get("bytes"):
                        raise RuntimeError("缺少内嵌图像字节")
                    image = np.asarray(Image.open(io.BytesIO(value["bytes"])).convert("RGB"))
                    if image.shape != (256, 256, 3):
                        raise RuntimeError("相机图像尺寸与已验证格式不同")
                    images[flat, view] = image
                actions[flat], states[flat], timestamps[flat] = action, state, row["timestamp"]
                filled[flat] = True
                checked += 1
                if checked % 2000 == 0:
                    print(json.dumps({"checked_frames": checked, "total_frames": count}), flush=True)
    if not filled.all():
        raise RuntimeError(f"缺少{int((~filled).sum())}帧；可能跨分片，禁止把残缺回合交给训练")
    state_min, state_max = np.full(8, np.inf), np.full(8, -np.inf)
    action_min, action_max = np.full(7, np.inf), np.full(7, -np.inf)
    records = []
    for index, episode in positions.items():
        start, meta = episode
        stop = start + meta["length"]
        times = timestamps[start:stop]
        if len(times) < 8 or not np.allclose(np.diff(times), 0.1, atol=1e-4):
            raise RuntimeError("回合不足8帧，或时间戳与源数据10fps不一致")
        split = "demo_validation" if index in validation_ids else "demo_train"
        if split == "demo_train":
            state_min = np.minimum(state_min, states[start:stop].min(axis=0))
            state_max = np.maximum(state_max, states[start:stop].max(axis=0))
            action_min = np.minimum(action_min, actions[start:stop].min(axis=0))
            action_max = np.maximum(action_max, actions[start:stop].max(axis=0))
        records.append({"episode_id": f"HuggingFaceVLA-libero-v3-{index}", "episode_index": index,
                        "instruction": meta["tasks"][0], "offset": start, "length": meta["length"], "split": split,
                        "init_seed": None, "init_seed_provenance": "源数据未提供，不声称已与冻结评测初态交叉核验",
                        "source_success": "发布示范来源；未逐条重建原始仿真状态核验success"})
    for array in [images, states, actions, timestamps]:
        array.flush()
    normalizer = {"mode": "min_max", "fit_split": "demo_train",
                  "state": {"min": state_min.tolist(), "max": state_max.tolist()},
                  "action": {"min": action_min.tolist(), "max": action_max.tolist()}}
    manifest = {"dataset": "LIBERO10_D0", "source_repo": source_record["repo"], "source_backend": "modelscope",
                "scope": selection.get("scope", "完整LIBERO10所选示范"), "tasks": selection["tasks"],
                "cache_id": cache_id, "selection_sha256": selection_sha256,
                "source_revision": source_record.get("revision") or None, "content_snapshot_sha256": snapshot,
                "revision_note": "注册表未提供可用git revision，逐文件SHA256已核验" if not source_record.get("revision") else "固定注册表revision",
                "source_manifest": str(args.selection.parent / "dataset_manifest.json"), "source_files": files,
                "frames": count, "episodes": records, "normalizer": normalizer,
                "views": {"0": "agentview/image", "1": "wrist/image2"}, "image_transform": "none，已对照上游180度仿真旋转",
                "state_semantics": ["eef_x", "eef_y", "eef_z", "axis_angle_x", "axis_angle_y", "axis_angle_z", "gripper_joint0", "gripper_joint1"],
                "action_dim": 7, "state_dim": 8, "horizon": 8,
                "source_declared_fps": 10, "control_hz": 20,
                "control_hz_evidence": "单条首回合前33帧相对xyz回放误差支持每20Hz拍执行1个动作；尚需跨任务核验",
                "action_temporal_resampling": "none", "eval_seed_overlap_check": "源示范缺少init_seed，尚未证明",
                "train_validation_episode_overlap": False, "precision": "RGB uint8、action/state fp32、timestamp fp64",
                "cache_bytes_estimate": required_bytes}
    (staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (staging / "COMPLETED").write_text(cache_id + "\n")
    staging.rename(destination)
    result = {"cache": str(destination), "manifest": str(destination / "manifest.json"), "frames": count,
              "episodes": len(episodes), "demo_train_episodes": len(episodes) - len(validation_ids),
              "demo_validation_episodes": len(validation_ids), "source_content_snapshot_sha256": snapshot,
              "source_manifest_sha256": checksum(args.selection.parent / "dataset_manifest.json"),
              "status": "数据格式和边界通过；不等于源初始状态去重或策略训练通过"}
    (run_dir / "d0_cache.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--selection", type=pathlib.Path, required=True)
    args = parser.parse_args()
    return run_stage("d0_cache", lambda directory: prepare(directory, args), {"selection": str(args.selection), "seed": 42})


if __name__ == "__main__":
    raise SystemExit(main())
