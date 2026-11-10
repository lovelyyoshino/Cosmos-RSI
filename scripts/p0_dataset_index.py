#!/usr/bin/env python3
"""读取实际Parquet页脚，避免使用镜像中已经过期的回合到分片映射。

@author pony
@date 2026-11-06
@version v1.0.1
@last_modified 2026-11-06
@changelog
    - v1.0.1 (2026-11-06): CDN握手超时后延长连接预算，已完成页脚按文件哈希复用。
    - v1.0.0 (2026-11-06): 严格限定HTTP Range读取量，并用实际回合统计选择分片。
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
import pathlib
import struct


def read_range(url: str, start: int, stop: int, size: int) -> bytes:
    import requests

    expected = stop - start + 1
    # Range被服务器忽略时立即关闭连接，不能意外下载整库的视频图像。
    with requests.get(url, headers={"Range": f"bytes={start}-{stop}"}, stream=True, timeout=(30, 45)) as response:
        if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {start}-{stop}/{size}":
            raise RuntimeError(f"服务器未遵守Parquet Range读取范围：HTTP {response.status_code}")
        value = response.raw.read(expected + 1)
        if len(value) != expected:
            raise RuntimeError("Parquet Range响应长度不符")
        return value


def footer_record(payload: bytes, item: dict) -> dict:
    import pyarrow.parquet as parquet

    if len(payload) < 8 or payload[-4:] != b"PAR1":
        raise RuntimeError("Parquet页脚缺少PAR1标记")
    length = struct.unpack("<I", payload[-8:-4])[0]
    if length > len(payload) - 8:
        raise RuntimeError("Parquet页脚内容不完整")
    footer = payload[-length - 8:]
    metadata = parquet.read_metadata(io.BytesIO(b"PAR1" + footer))
    episode_column = metadata.schema.names.index("episode_index")
    index_column = metadata.schema.names.index("index")
    groups = []
    for index in range(metadata.num_row_groups):
        group = metadata.row_group(index)
        episode_stats = group.column(episode_column).statistics
        index_stats = group.column(index_column).statistics
        if episode_stats is None or index_stats is None or not episode_stats.has_min_max or not index_stats.has_min_max:
            raise RuntimeError("分片缺少回合/全局帧号统计，禁止根据旧元数据猜测")
        groups.append({"rows": group.num_rows, "episode_min": int(episode_stats.min), "episode_max": int(episode_stats.max),
                       "index_min": int(index_stats.min), "index_max": int(index_stats.max)})
    return {"file": item["Path"], "bytes": item["Size"], "published_sha256": item["Sha256"],
            "footer_sha256": hashlib.sha256(footer).hexdigest(), "rows": metadata.num_rows, "row_groups": groups}


def selected_files(index: list[dict], episode_ids: set[int]) -> list[str]:
    selected = []
    covered = set()
    for item in index:
        matches = {episode for episode in episode_ids if any(
            group["episode_min"] <= episode <= group["episode_max"] for group in item["row_groups"])}
        if matches:
            selected.append(item["file"])
            covered.update(matches)
    if covered != episode_ids:
        raise RuntimeError(f"实际Parquet索引缺少所选回合：{sorted(episode_ids - covered)}")
    return sorted(selected)


def index_modelscope_files(api, repo: str, revision: str, files: list[dict], cache: pathlib.Path,
                          artifact: pathlib.Path) -> list[dict]:
    namespace, dataset = repo.split("/", 1)
    cache.mkdir(parents=True, exist_ok=True)

    def one(item):
        key = hashlib.sha256(f"{repo}:{item['Path']}:{item['Size']}:{item['Sha256']}".encode()).hexdigest()
        path = cache / (key + ".parquet-footer")
        if path.exists():
            payload = path.read_bytes()
        else:
            size = item["Size"]
            url = api.get_dataset_file_url(item["Path"], dataset, namespace, revision=revision)
            payload = read_range(url, max(0, size - 65536), size - 1, size)
            if payload[-4:] != b"PAR1":
                raise RuntimeError("文件不是带页脚的Parquet")
            length = struct.unpack("<I", payload[-8:-4])[0]
            if length > 8 * 1024 ** 2 or length > size - 8:
                raise RuntimeError("Parquet页脚长度异常，禁止扩大下载")
            if length > len(payload) - 8:
                payload = read_range(url, size - length - 8, size - 1, size)
            payload = payload[-length - 8:]
            footer_record(payload, item)
            temporary = path.with_suffix(".part")
            temporary.write_bytes(payload)
            temporary.replace(path)
        return footer_record(payload, item)

    records = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        pending = [executor.submit(one, item) for item in files]
        for future in as_completed(pending):
            records.append(future.result())
            if len(records) % 25 == 0 or len(records) == len(files):
                print(json.dumps({"indexed_parquet_files": len(records), "total_files": len(files)}), flush=True)
    records.sort(key=lambda item: item["file"])
    artifact.write_text(json.dumps({"repo": repo, "requested_revision": revision,
                                   "method": "HTTP Range页脚；仅候选分片后续做完整文件SHA256校验", "files": records},
                                  ensure_ascii=False, indent=2) + "\n")
    return records
