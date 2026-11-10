#!/usr/bin/env python3
"""经官方资产API预取cu130 ARM扩展，双重校验后安装到生成专用环境。

@author pony
@date 2026-11-06
@version v1.1.0
@last_modified 2026-11-06
@changelog
    - v1.1.0 (2026-11-06): 纳入cp39-abi3的xformers ARM扩展，复用已校验的四个wheel。
    - v1.0.0 (2026-11-06): 使用curl与官方资产API避开uv下载超时，保留锁文件和TLS校验。
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys
import tomllib
import urllib.parse
import urllib.request

from p0_common import ROOT, run_stage

RELEASE_API = "https://api.github.com/repos/nvidia-cosmos/cosmos-dependencies/releases/tags/v1.5.0"
ASSET_API = "https://api.github.com/repos/nvidia-cosmos/cosmos-dependencies/releases/assets/"
PACKAGES = {"decord", "flash-attn", "natten", "transformer-engine", "xformers"}


def checksum(path: pathlib.Path) -> str:
    with path.open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def prefetch(run_dir: pathlib.Path) -> dict:
    repo = ROOT / "third_party/cosmos-transfer2.5"
    lock_bytes = subprocess.check_output(["git", "-C", str(repo), "show", "HEAD:uv.lock"])
    lock = tomllib.loads(lock_bytes.decode())
    with urllib.request.urlopen(RELEASE_API, timeout=30) as response:
        release = json.load(response)
    assets = {asset["name"]: asset for asset in release["assets"]}
    selected = []
    for package in lock["package"]:
        if package["name"] not in PACKAGES:
            continue
        for wheel in package.get("wheels", []):
            name = urllib.parse.unquote(pathlib.PurePosixPath(urllib.parse.urlsplit(wheel["url"]).path).name)
            if not (name.endswith("linux_aarch64.whl") and "cu130.torch29-" in name
                    and any(tag in name for tag in ["cp313-cp313-", "cp39-abi3-"])):
                continue
            asset = assets[name]
            if asset.get("digest") != wheel["hash"] or asset["browser_download_url"] != wheel["url"]:
                raise RuntimeError(f"官方资产与上游锁文件的SHA256或地址不一致：{name}")
            selected.append((package, wheel, asset))
    if {package["name"] for package, _, _ in selected} != PACKAGES or len(selected) != len(PACKAGES):
        raise RuntimeError("官方锁文件未给出恰好五个匹配Python3.13/cu130的ARM扩展")
    destination = ROOT / "tools/wheels/cosmos-v1.5.0-cu130-cp313-aarch64"
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for package, wheel, asset in selected:
        target = destination / asset["name"]
        if not (target.is_file() and target.stat().st_size == asset["size"] and checksum(target) == wheel["hash"]):
            partial = target.with_suffix(".whl.part")
            print(json.dumps({"package": package["name"], "bytes": asset["size"], "status": "downloading"}), flush=True)
            # 原始API交付官方CDN；curl端到端验证TLS，不转发HF凭据、不改DNS或锁文件。
            subprocess.run(["curl", "-4", "--fail", "--location", "--silent", "--show-error",
                            "--connect-timeout", "15", "--max-time", "1200", "--retry", "0",
                            "--proto", "=https", "--proto-redir", "=https",
                            "--header", "Accept: application/octet-stream",
                            ASSET_API + str(asset["id"]) + "?download=1", "--output", str(partial)], check=True)
            if partial.stat().st_size != asset["size"] or checksum(partial) != wheel["hash"]:
                raise RuntimeError(f"预取文件与官方双重校验值不符：{asset['name']}")
            partial.replace(target)
        records.append({"package": package["name"], "version": package["version"], "path": str(target),
                        "bytes": asset["size"], "sha256": wheel["hash"], "official_asset_id": asset["id"],
                        "official_url": wheel["url"], "lock_and_official_release_agree": True})
        print(json.dumps({"package": package["name"], "status": "verified"}), flush=True)
    (run_dir / "wheel_manifest.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
    # 只预装官方锁定版本；随后仍须通过完整的uv frozen依赖收敛和实际GPU算子检查。
    subprocess.run([sys.executable, "-m", "uv", "pip", "install", "--python",
                    str(ROOT / "envs/transfer/bin/python"), "--no-deps", *[item["path"] for item in records]], check=True)
    return {"official_lock_sha256": hashlib.sha256(lock_bytes).hexdigest(), "wheels": records,
            "environment": str(ROOT / "envs/transfer"), "system_configuration_changed": False,
            "dependency_sync_complete": False, "gpu_test_performed": False}


if __name__ == "__main__":
    raise SystemExit(run_stage("cuda_wheel_prefetch", prefetch,
                              {"scope": "官方ARM扩展预取和本地安装，非完整生成环境验收"}))
