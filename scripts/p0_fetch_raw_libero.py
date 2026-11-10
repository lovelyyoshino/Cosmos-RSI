#!/usr/bin/env python3
"""按固定镜像revision与LFS摘要下载官方LIBERO原始HDF5，先验证单任务。

@author pony
@date 2026-11-06
@version v1.0.0
@last_modified 2026-11-06
@changelog
    - v1.0.0 (2026-11-06): 显式记录镜像来源，完整性通过后再发布数据文件。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil
import subprocess
import urllib.parse
import urllib.request

from p0_common import ROOT, run_stage
from p0_policy_runtime import sha256

REPOSITORY = "yifengzhu-hf/LIBERO-datasets"
ENDPOINT = "https://hf-mirror.com"


def metadata(url: str):
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


def fetch(run_dir: pathlib.Path, task: str) -> dict:
    info = metadata(f"{ENDPOINT}/api/datasets/{REPOSITORY}/revision/main")
    revision = info.get("sha", "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise RuntimeError("镜像没有提供有效revision，禁止使用可变main下载")
    entries = metadata(f"{ENDPOINT}/api/datasets/{REPOSITORY}/tree/{revision}/libero_10?limit=100")
    candidates = [item for item in entries if item.get("path", "").endswith("_demo.hdf5")
                  and (task == "all" or task in pathlib.PurePosixPath(item["path"]).name)]
    if not candidates or (task != "all" and len(candidates) != 1):
        raise RuntimeError("任务必须唯一匹配一个官方HDF5；全量下载需显式选择all")
    if task == "all" and len(candidates) != 10:
        raise RuntimeError("libero_10的原始任务文件数异常")
    destination = ROOT / "data" / ("libero_raw_" + revision[:12])
    destination.mkdir(exist_ok=True)
    (run_dir / "repository_metadata.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n")
    (run_dir / "file_metadata.json").write_text(json.dumps(candidates, ensure_ascii=False, indent=2) + "\n")
    if shutil.disk_usage(ROOT).free < sum(item["size"] for item in candidates) + 20 * 1024 ** 3:
        raise RuntimeError("原始轨迹下载后将不足20GiB工作空间")
    records = []
    for item in sorted(candidates, key=lambda value: value["size"]):
        relative = pathlib.PurePosixPath(item["path"])
        expected = item.get("lfs", {}).get("oid", "")
        if relative.parts[0] != "libero_10" or len(relative.parts) != 2 or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise RuntimeError("原始轨迹路径或LFS摘要不合法")
        output = destination / relative.name
        if output.exists():
            if output.stat().st_size != item["size"] or sha256(output) != expected:
                raise RuntimeError("已有原始轨迹与摘要不一致，保留文件并停止")
        else:
            partial = output.with_suffix(output.suffix + ".partial")
            url = f"{ENDPOINT}/datasets/{REPOSITORY}/resolve/{revision}/" + urllib.parse.quote(str(relative))
            print(json.dumps({"downloading": str(relative), "bytes": item["size"], "revision": revision}), flush=True)
            # 仅HTTPS下载并禁止HTTP重定向；无自动重试，不给镜像传递HF凭据。
            subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error", "--proto", "=https",
                            "--proto-redir", "=https", "--connect-timeout", "20", "--max-time", "3600",
                            "--output", str(partial), url], check=True)
            if partial.stat().st_size != item["size"] or sha256(partial) != expected:
                raise RuntimeError("原始轨迹字节数或LFS SHA256不符，未发布为可用文件")
            partial.rename(output)
        records.append({"file": str(output), "relative_file": str(relative), "bytes": item["size"], "sha256": expected})
        (run_dir / "completed_files.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"verified": str(output), "bytes": item["size"]}), flush=True)
    manifest = {"repository": REPOSITORY, "upstream_reference": "LIBERO@8f1084e3/libero/libero/utils/download_utils.py",
                "download_endpoint": ENDPOINT, "revision": revision, "source": "官方仓库的第三方HF镜像",
                "official_revision_cross_verified": False, "files": records, "task_selection": task,
                "eligible_for_training": False, "status": "下载完整性通过，HDF5字段和时序待验证",
                "precision": "CPU网络传输和SHA256，无模型计算", "batch": None, "seed": None,
                "steps": 0, "sample_count": len(records)}
    (run_dir / "raw_dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", default="STUDY_SCENE1_pick_up_the_book")
    args = parser.parse_args()
    raise SystemExit(run_stage("raw_libero_download", lambda directory: fetch(directory, args.task),
                              {"task": args.task, "download_only": True, "gpu_used": False}))
