#!/usr/bin/env python3
"""只替换官方锁文件的PyPI传输地址，严格保留版本、包集合和内容哈希。

@author pony
@date 2026-11-06
@version v1.0.0
@last_modified 2026-11-06
@changelog
    - v1.0.0 (2026-11-06): 生成并应用可逆的下载镜像补丁，不更改官方cu130包选择。
"""
from __future__ import annotations

import difflib
import hashlib
import json
import pathlib
import re
import subprocess
import tomllib

from p0_common import ROOT, run_stage


def mirrored(value: str) -> str:
    if value == "https://pypi.org/simple":
        return "https://mirrors.aliyun.com/pypi/simple"
    if value.startswith("https://files.pythonhosted.org/packages/"):
        return value.replace("https://files.pythonhosted.org/packages/", "https://mirrors.aliyun.com/pypi/packages/", 1)
    return value


def translated(value):
    if isinstance(value, str):
        return mirrored(value)
    if isinstance(value, list):
        return [translated(item) for item in value]
    if isinstance(value, dict):
        return {key: translated(item) for key, item in value.items()}
    return value


def apply(run_dir: pathlib.Path) -> dict:
    repo = ROOT / "third_party/cosmos-transfer2.5"
    original = subprocess.check_output(["git", "-C", str(repo), "show", "HEAD:uv.lock"], text=True)
    desired = re.sub(r'"(https://[^"\s]+)"', lambda match: '"' + mirrored(match[1]) + '"', original)
    # 用TOML语义比对，任何包版本、marker或hash变动都必须拒绝。
    before = tomllib.loads(original)
    after = tomllib.loads(desired)
    if after != translated(before):
        raise RuntimeError("镜像转换修改了URL之外的锁文件语义")
    current = (repo / "uv.lock").read_text()
    if current not in [original, desired]:
        raise RuntimeError("上游锁文件已有其他修改，禁止覆盖")
    patch = "".join(difflib.unified_diff(original.splitlines(keepends=True), desired.splitlines(keepends=True),
                                       fromfile="a/uv.lock", tofile="b/uv.lock"))
    patch_path = ROOT / "patches/cosmos-transfer2.5/0004-pypi-transport-mirror.patch"
    patch_path.write_text(patch)
    if current != desired:
        subprocess.run(["git", "-C", str(repo), "apply", "--check", str(patch_path)], check=True)
        subprocess.run(["git", "-C", str(repo), "apply", str(patch_path)], check=True)
    subprocess.run(["git", "-C", str(repo), "apply", "--reverse", "--check", str(patch_path)], check=True)
    result = {"official_lock_sha256": hashlib.sha256(original.encode()).hexdigest(),
              "transport_lock_sha256": hashlib.sha256(desired.encode()).hexdigest(),
              "patch": str(patch_path), "patch_sha256": hashlib.sha256(patch.encode()).hexdigest(),
              "semantic_check": "仅PyPI registry和wheel/sdist传输URL改变；包集合、版本、marker、内容哈希不变",
              "pytorch_and_cosmos_cuda_urls_unchanged": True, "packages_in_lock": len(after["package"])}
    (run_dir / "mirror.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


if __name__ == "__main__":
    raise SystemExit(run_stage("transfer_mirror", apply))
