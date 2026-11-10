#!/usr/bin/env python3
"""下载固定 revision 的模型及生成辅助组件；认证仅从进程环境读取。

@author pony
@date 2026-11-06
@version v1.5.0
@last_modified 2026-11-07
@changelog
    - v1.5.0 (2026-11-07): 增加base Transfer专用SigLIP2图像上下文组件，与安全检查的SigLIP分开校验。
    - v1.4.0 (2026-11-06): 增加SAM2.1 tiny的显式下载入口，供生成窗口独立分割几何检查。
    - v1.3.0 (2026-11-06): 支持生成辅助模型和显式镜像，统一两种后端的文件筛选。
    - v1.2.0 (2026-11-06): Transfer仅下载distilled edge权重，避免拉取全部控制模型。
    - v1.1.0 (2026-11-06): 增加已授权的ModelScope镜像并核验逐文件SHA256。
    - v1.0.0 (2026-11-06): 保存模型版本和校验和，不保存 token。
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import pathlib
import shutil
import time
import urllib.parse
import urllib.request

from p0_common import ROOT, run_stage


REPOS = ["nvidia/Cosmos-Reason2-2B", "nvidia/Cosmos-Transfer2.5-2B", "nvidia/Cosmos-Predict2.5-2B",
         "nvidia/Cosmos-Reason1-7B", "nvidia/Cosmos-Guardrail1", "Qwen/Qwen3Guard-Gen-0.6B",
         "google/siglip-so400m-patch14-384", "google/siglip2-so400m-patch16-naflex",
         "nvidia/GR00T-N1.7-LIBERO", "facebook/sam2.1-hiera-tiny",
         "nvidia/Cosmos-Policy-LIBERO-Predict2-2B"]


def select_file(name: str, args: argparse.Namespace) -> bool:
    path = pathlib.PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise RuntimeError("模型文件列表包含越界路径")
    if args.include:
        return any(fnmatch.fnmatchcase(name, pattern) for pattern in args.include)
    if args.repo.endswith("Cosmos-Transfer2.5-2B"):
        return name.startswith("distilled/general/edge/") or name in ["README.md", "configuration.json"]
    if args.repo.endswith("Cosmos-Predict2.5-2B"):
        return name in ["tokenizer.pth", "README.md", "LICENSE", "configuration.json"]
    if args.repo.endswith("Cosmos-Guardrail1"):
        return name.startswith(("blocklist/", "video_content_safety_filter/", "face_blur_filter/")) or name in ["README.md", "LICENSE"]
    if args.repo.endswith("GR00T-N1.7-LIBERO") and not name.startswith("libero_10/"):
        return name in ["README.md", "LICENSE"]
    return name.endswith((".safetensors", ".json", ".txt", ".model", ".jinja", ".md", ".yaml")) or name == "LICENSE"


def download_huggingface(run_dir: pathlib.Path, args: argparse.Namespace) -> dict:
    from huggingface_hub import HfApi, snapshot_download

    token = os.environ.get("HF_TOKEN")
    info = HfApi(token=token).model_info(args.repo, files_metadata=True)
    selected = [item for item in info.siblings if select_file(item.rfilename, args)]
    if not selected:
        raise RuntimeError("没有匹配的模型文件")
    total_bytes = sum(item.size or 0 for item in selected)
    if shutil.disk_usage(ROOT).free < total_bytes + 20 * 1024 ** 3:
        raise RuntimeError("磁盘余量不足以下载模型并保留20GiB工作空间")
    print(json.dumps({"repo": args.repo, "revision": info.sha, "gated": info.gated,
                      "download_bytes": total_bytes, "file_count": len(selected)}, ensure_ascii=False), flush=True)
    destination = ROOT / "checkpoints" / "base" / args.repo.split("/")[-1]
    destination.mkdir(parents=True, exist_ok=True)
    before = time.monotonic()
    snapshot_download(repo_id=args.repo, revision=info.sha, token=token, local_dir=destination,
                      allow_patterns=[item.rfilename for item in selected], max_workers=4)
    files = []
    for item in selected:
        path = destination / item.rfilename
        with path.open("rb") as source:
            checksum = hashlib.file_digest(source, "sha256").hexdigest()
        expected = item.lfs.sha256 if item.lfs is not None else None
        if expected is not None and checksum != expected:
            raise RuntimeError(f"LFS SHA256校验失败：{item.rfilename}")
        files.append({"file": item.rfilename, "bytes": path.stat().st_size,
                      "sha256": checksum, "upstream_lfs_sha256": expected})
    result = {"repo": args.repo, "revision": info.sha, "gated": info.gated,
              "path": str(destination), "files": files, "download_wall_s": time.monotonic() - before,
              "token_source": "process_environment" if token else "anonymous",
              "inference_performed": False}
    (run_dir / "model_manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def download_modelscope(run_dir: pathlib.Path, args: argparse.Namespace) -> dict:
    from modelscope.hub.snapshot_download import snapshot_download

    mirror = args.mirror or ("nv-community/" + args.repo.split("/")[-1] if args.repo.startswith("nvidia/") else args.repo)
    url = f"https://modelscope.cn/api/v1/models/{mirror}/repo/files?Revision=master&Recursive=true"
    with urllib.request.urlopen(url, timeout=30) as response:
        listing = json.load(response)
    if not listing.get("Success"):
        raise RuntimeError("ModelScope模型文件列表读取失败")
    candidates = [item for item in listing["Data"]["Files"] if item["Type"] == "blob"]
    revision = max(candidates, key=lambda item: item["CommittedDate"])["Revision"]
    fixed_url = url.replace("Revision=master", "Revision=" + urllib.parse.quote(revision, safe=""))
    with urllib.request.urlopen(fixed_url, timeout=30) as response:
        listing = json.load(response)
    if not listing.get("Success"):
        raise RuntimeError("固定版本的ModelScope模型文件列表读取失败")
    candidates = [item for item in listing["Data"]["Files"] if item["Type"] == "blob"]
    selected = [item for item in candidates if select_file(item["Path"], args)]
    if not selected:
        raise RuntimeError("没有匹配的模型文件")
    if args.repo.endswith("Cosmos-Transfer2.5-2B") and not args.include:
        if len([item for item in selected if item["Path"].endswith(".pt")]) != 1:
            raise RuntimeError("蒸馏edge权重数量与预期不符，禁止下载其他大模型代替")
    total_bytes = sum(item["Size"] for item in selected)
    if shutil.disk_usage(ROOT).free < total_bytes + 20 * 1024 ** 3:
        raise RuntimeError("磁盘余量不足以下载模型并保留20GiB工作空间")
    (run_dir / "mirror_listing.json").write_text(json.dumps(listing, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"backend": "modelscope", "mirror": mirror, "revision": revision,
                      "download_bytes": total_bytes, "file_count": len(selected)}, ensure_ascii=False), flush=True)
    destination = ROOT / "checkpoints" / "base" / args.repo.split("/")[-1]
    before = time.monotonic()
    snapshot_download(model_id=mirror, revision=revision, local_dir=str(destination),
                      cache_dir=str(ROOT / ".cache" / "modelscope"),
                      allow_patterns=[item["Path"] for item in selected], max_workers=4)
    files = []
    for item in selected:
        path = destination / item["Path"]
        with path.open("rb") as source:
            checksum = hashlib.file_digest(source, "sha256").hexdigest()
        if checksum != item["Sha256"] or path.stat().st_size != item["Size"]:
            raise RuntimeError(f"ModelScope文件校验失败：{item['Path']}")
        files.append({"file": item["Path"], "bytes": path.stat().st_size,
                      "sha256": checksum, "mirror_published_sha256": item["Sha256"]})
    result = {"canonical_repo": args.repo, "backend": "modelscope", "mirror_repo": mirror,
              "source_is_community_mirror": mirror.startswith(("nv-community/", "AI-ModelScope/")),
              "canonical_hf_hash_crosscheck": "unavailable_network",
              "revision": revision, "path": str(destination), "files": files,
              "download_wall_s": time.monotonic() - before, "inference_performed": False}
    (run_dir / "model_manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", choices=REPOS, default="nvidia/Cosmos-Reason2-2B")
    parser.add_argument("--backend", choices=["huggingface", "modelscope"], default="huggingface")
    parser.add_argument("--mirror", help="ModelScope实际仓库名；写入manifest，不伪装为HF来源")
    parser.add_argument("--include", action="append", default=[], help="显式文件glob，可重复；默认为所选组件的最小集合")
    args = parser.parse_args()
    download = download_huggingface if args.backend == "huggingface" else download_modelscope
    return run_stage("model_download", lambda run_dir: download(run_dir, args),
                     {"repo": args.repo, "precision": None, "batch": None, "steps": None, "seed": None, "sample_count": None})


if __name__ == "__main__":
    raise SystemExit(main())
