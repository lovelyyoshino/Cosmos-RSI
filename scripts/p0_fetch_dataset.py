#!/usr/bin/env python3
"""先下载最小示范分片或仿真资源，冻结文件哈希并记录可获取的版本。

@author pony
@date 2026-11-06
@version v1.4.0
@last_modified 2026-11-06
@changelog
    - v1.4.0 (2026-11-06): 按实际Parquet回合统计选择文件，修复镜像重分片后的旧元数据映射。
    - v1.3.0 (2026-11-06): 数据注册表无revision时显式标注，以逐文件哈希冻结内容快照。
    - v1.2.0 (2026-11-06): 按官方BDDL任务语言选择LIBERO10的完整示范分片。
    - v1.1.0 (2026-11-06): 识别LIBERO-plus归档里嵌套的原服务器路径，不重复下载或解压。
    - v1.0.0 (2026-11-06): 使用ModelScope官方SDK，默认只取一个数据分片。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import shutil
import stat
import zipfile

from p0_common import ROOT, run_stage


def fetch(run_dir: pathlib.Path, kind: str) -> dict:
    from modelscope.hub.api import HubApi
    from modelscope.hub.snapshot_download import dataset_snapshot_download

    is_demo = kind.startswith("demo")
    repo = "HuggingFaceVLA/libero" if is_demo else "Sylvest/LIBERO-plus"
    api = HubApi(endpoint="https://www.modelscope.cn", timeout=120, max_retries=0)
    listing = api.get_dataset_files(repo, recursive=True, page_size=1000)
    blobs = [item for item in listing if item["Type"] == "blob"]
    (run_dir / "source_file_listing.json").write_text(json.dumps(blobs, ensure_ascii=False, indent=2) + "\n")
    revision = max(blobs, key=lambda item: item["CommittedDate"])["Revision"] or None
    requested_revision = revision or "master"
    selection = None
    if kind == "demo-libero10":
        import pyarrow.parquet as parquet
        data_root = ROOT / "data/libero_v3_p0"
        # 选择分片前必须核对既有元数据与本次镜像版本一致，防止混用两个revision。
        for item in [item for item in blobs if item["Path"].startswith("meta/")]:
            path = data_root / item["Path"]
            with path.open("rb") as source:
                actual = hashlib.file_digest(source, "sha256").hexdigest()
            if actual != item["Sha256"]:
                raise RuntimeError("现有示范元数据版本变化，请重新验证最小样本后再扩大下载")
        tasks = {}
        for path in (ROOT / "third_party/LIBERO/libero/libero/bddl_files/libero_10").glob("*.bddl"):
            language = re.search(r"\(:language\s+([^)]*)\)", path.read_text())
            if language is None:
                raise RuntimeError(f"无法解析官方BDDL语言：{path}")
            tasks[re.sub(r"\s+", " ", language[1]).strip()] = path.name
        if len(tasks) != 10:
            raise RuntimeError("LIBERO10官方任务数量不是10")
        episodes = []
        for path in sorted((data_root / "meta/episodes").rglob("*.parquet")):
            episodes.extend(parquet.read_table(path, columns=["episode_index", "tasks", "data/chunk_index",
                                                              "data/file_index", "dataset_from_index", "dataset_to_index",
                                                              "length"]).to_pylist())
        episodes = [row for row in episodes if len(row["tasks"]) == 1 and row["tasks"][0] in tasks]
        from p0_dataset_index import index_modelscope_files, selected_files

        data_files = [item for item in blobs if item["Path"].startswith("data/") and item["Path"].endswith(".parquet")]
        actual_index = index_modelscope_files(api, repo, requested_revision, data_files,
                                             ROOT / ".cache/parquet_footers", run_dir / "parquet_index.json")
        needed = set(selected_files(actual_index, {row["episode_index"] for row in episodes}))
        old_files = {f"data/chunk-{row['data/chunk_index']:03d}/file-{row['data/file_index']:03d}.parquet" for row in episodes}
        selected = [item for item in blobs if item["Path"] in needed or item["Path"].startswith("meta/") or item["Path"] == "README.md"]
        if len(needed) != len([item for item in selected if item["Path"] in needed]):
            raise RuntimeError("镜像缺少被选中回合的数据分片")
        if {row["tasks"][0] for row in episodes} != set(tasks):
            raise RuntimeError("示范没有覆盖全部10个任务")
        selection = {"suite": "libero_10", "tasks": tasks, "episodes": episodes,
                     "expected_frames": sum(row["length"] for row in episodes), "data_files": sorted(needed),
                     "file_selection": "实际Parquet页脚episode_index统计，非过期的meta data/file_index",
                     "parquet_index": str(run_dir / "parquet_index.json"),
                     "metadata_mapping_mismatch": old_files != needed,
                     "completion_status": "分片下载校验；仍需逐回合行数、时序、相机和动作语义核查"}
    elif kind == "demo":
        selected = [item for item in blobs if item["Path"].startswith("meta/") or
                    item["Path"] in ["README.md", "data/chunk-000/file-000.parquet"]]
        if not any(item["Path"] == "data/chunk-000/file-000.parquet" for item in selected):
            raise RuntimeError("指定的首个数据分片不存在，禁止退化为下载全库")
    else:
        selected = [item for item in blobs if item["Path"] in ["README.md", "assets.zip"]]
    for item in selected:
        path = pathlib.PurePosixPath(item["Path"])
        if path.is_absolute() or ".." in path.parts:
            raise RuntimeError("文件列表包含越界路径")
    total_bytes = sum(item["Size"] for item in selected)
    snapshot_id = hashlib.sha256(json.dumps(
        sorted((item["Path"], item["Size"], item["Sha256"]) for item in selected), separators=(",", ":")
    ).encode()).hexdigest()
    if shutil.disk_usage(ROOT).free < total_bytes + 20 * 1024 ** 3:
        raise RuntimeError("下载后将不足20GiB工作空间")
    destination = ROOT / "data" / ("libero_v3_p0" if is_demo else "libero_plus_resource")
    (run_dir / "selected_files.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"repo": repo, "revision": revision, "selected_bytes": total_bytes,
                      "file_count": len(selected), "destination": str(destination)}, ensure_ascii=False), flush=True)
    dataset_snapshot_download(dataset_id=repo, revision=requested_revision, local_dir=str(destination),
                              cache_dir=str(ROOT / ".cache" / "modelscope"),
                              allow_patterns=[item["Path"] for item in selected], max_workers=4)
    checked = []
    for item in selected:
        path = destination / item["Path"]
        with path.open("rb") as source:
            checksum = hashlib.file_digest(source, "sha256").hexdigest()
        if checksum != item["Sha256"] or path.stat().st_size != item["Size"]:
            raise RuntimeError(f"下载校验失败：{item['Path']}")
        checked.append({"file": item["Path"], "bytes": item["Size"], "sha256": checksum})
    result = {"kind": kind, "backend": "modelscope", "repo": repo, "revision": revision,
              "requested_revision": requested_revision, "content_snapshot_sha256": snapshot_id,
              "revision_status": "注册表返回固定版本" if revision else "注册表未提供revision；以清单逐文件大小和SHA256锁定内容",
              "source_is_community_mirror": True, "path": str(destination), "files": checked}
    if is_demo:
        info = json.loads((destination / "meta" / "info.json").read_text())
        if info["codebase_version"] != "v3.0" or info["features"]["action"]["shape"] != [7] or info["features"]["observation.state"]["shape"] != [8]:
            raise RuntimeError("示范数据格式与本次核查的v3、7维动作、8维状态不一致")
        result["dataset_info"] = info
        result["scope"] = "LIBERO10所涉及的分片及元数据；尚未完成逐回合与控制语义核查" if selection else "一个分片及元数据，不代表已下载完整D0；相机语义需读取样本确认"
        if selection:
            (run_dir / "selection.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2) + "\n")
            result["selection"] = str(run_dir / "selection.json")
    else:
        archive = destination / "assets.zip"
        extracted = ROOT / "data" / f"libero_plus_assets_{revision}"
        if not extracted.exists():
            staging = extracted.with_suffix(".extracting")
            staging.mkdir(exist_ok=False)
            with zipfile.ZipFile(archive) as bundle:
                entries = bundle.infolist()
                needed = sum(item.file_size for item in entries)
                if shutil.disk_usage(ROOT).free < needed + 20 * 1024 ** 3:
                    raise RuntimeError("资源解压后将不足20GiB工作空间")
                for item in entries:
                    path = pathlib.PurePosixPath(item.filename)
                    if path.is_absolute() or ".." in path.parts or stat.S_ISLNK(item.external_attr >> 16):
                        raise RuntimeError("资源归档包含越界路径或符号链接")
                bundle.extractall(staging)
            staging.rename(extracted)
        # 镜像zip保留了发布者服务器目录前缀，不能假定assets位于解压目录的第一层。
        candidates = [path for path in [extracted, *extracted.rglob("assets")]
                      if path.is_dir() and (path / "scenes").is_dir() and (path / "textures").is_dir()]
        if len(candidates) != 1:
            raise RuntimeError(f"资源目录应唯一且包含scenes/textures，实际找到{len(candidates)}个")
        assets = candidates[0]
        result["assets_root"] = str(assets)
    (run_dir / "dataset_manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("kind", choices=["demo", "demo-libero10", "plus-assets"])
    args = parser.parse_args()
    return run_stage("dataset_download", lambda run_dir: fetch(run_dir, args.kind), vars(args))


if __name__ == "__main__":
    raise SystemExit(main())
