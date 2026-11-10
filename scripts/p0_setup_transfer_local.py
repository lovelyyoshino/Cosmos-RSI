#!/usr/bin/env python3
"""把已校验的官方wheel交给本机安装器，冻结依赖与哈希保持不变。

@author pony
@date 2026-11-06
@version v1.1.0
@last_modified 2026-11-06
@changelog
    - v1.1.0 (2026-11-06): 包含abi3 xformers，允许从已核验的部分传输补丁升级。
    - v1.0.0 (2026-11-06): 仅在loopback临时提供已校验扩展，安装完成后关闭服务。
"""
from __future__ import annotations

import difflib
import hashlib
import http.server
import json
import pathlib
import re
import socketserver
import subprocess
import threading
import tomllib
import urllib.parse

from p0_common import ROOT, run_stage
from p0_prefetch_cuda_wheels import PACKAGES, checksum
from p0_transfer_mirror import mirrored

PORT = 8955


def setup(run_dir: pathlib.Path) -> dict:
    repo = ROOT / "third_party/cosmos-transfer2.5"
    manifests = sorted(ROOT.glob("runs/p0/cuda_wheel_prefetch_*/wheel_manifest.json"))
    if not manifests:
        raise RuntimeError("缺少官方wheel预取清单")
    manifest = manifests[-1]
    probe = json.loads((manifest.parent / "probe.json").read_text())
    if probe["status"] != "passed":
        raise RuntimeError("官方扩展预取尚未通过")
    records = json.loads(manifest.read_text())
    directory = ROOT / "tools/wheels/cosmos-v1.5.0-cu130-cp313-aarch64"
    replacements = {}
    names = set()
    for item in records:
        path = pathlib.Path(item["path"]).resolve(strict=True)
        if path.parent != directory or path.stat().st_size != item["bytes"] or checksum(path) != item["sha256"]:
            raise RuntimeError("本地wheel已偏离官方双重校验清单")
        replacements[item["official_url"]] = f"http://127.0.0.1:{PORT}/" + urllib.parse.quote(path.name, safe="")
        names.add(path.name)
    if len(replacements) != 5 or {item["package"] for item in records} != PACKAGES:
        raise RuntimeError("本次只允许替换五个已校验的官方ARM扩展")
    original = subprocess.check_output(["git", "-C", str(repo), "show", "HEAD:uv.lock"], text=True)
    base = re.sub(r'"(https://[^"\s]+)"', lambda match: '"' + mirrored(match[1]) + '"', original)
    desired = base
    for before, after in replacements.items():
        if desired.count('"' + before + '"') != 1:
            raise RuntimeError("上游锁文件中的wheel地址不是唯一匹配")
        desired = desired.replace('"' + before + '"', '"' + after + '"')

    def translate(value):
        if isinstance(value, str):
            return replacements.get(value, value)
        if isinstance(value, list):
            return [translate(item) for item in value]
        if isinstance(value, dict):
            return {key: translate(item) for key, item in value.items()}
        return value

    if tomllib.loads(desired) != translate(tomllib.loads(base)):
        raise RuntimeError("锁文件转换修改了五个传输地址之外的内容")
    current = (repo / "uv.lock").read_text()
    reverse = {value: key for key, value in replacements.items()}
    def restore(value):
        if isinstance(value, str):
            return reverse.get(value, value)
        if isinstance(value, list):
            return [restore(item) for item in value]
        if isinstance(value, dict):
            return {key: restore(item) for key, item in value.items()}
        return value
    if restore(tomllib.loads(current)) != tomllib.loads(base):
        raise RuntimeError("上游锁文件有不属于本项目的修改，禁止覆盖")
    patch = "".join(difflib.unified_diff(base.splitlines(keepends=True), desired.splitlines(keepends=True),
                                       fromfile="a/uv.lock", tofile="b/uv.lock"))
    patch_path = ROOT / "patches/cosmos-transfer2.5/0005-local-wheel-transport.patch"

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(directory), **kwargs)

        def send_head(self):
            path = urllib.parse.urlsplit(self.path)
            if path.query or urllib.parse.unquote(path.path) not in {"/" + name for name in names}:
                self.send_error(404)
                return None
            return super().send_head()

        def log_message(self, format, *args):
            print("local-wheel: " + format % args, flush=True)

    class Server(socketserver.ThreadingTCPServer):
        daemon_threads = True
        allow_reuse_address = True

    # 只监听Thor loopback，只交付白名单文件；文件已通过官方TLS下载和SHA256校验。
    with Server(("127.0.0.1", PORT), Handler) as server:
        patch_path.write_text(patch)
        if current != desired:
            incremental = "".join(difflib.unified_diff(current.splitlines(keepends=True), desired.splitlines(keepends=True),
                                                     fromfile="a/uv.lock", tofile="b/uv.lock"))
            subprocess.run(["git", "-C", str(repo), "apply", "--check", "-"], input=incremental, text=True, check=True)
            subprocess.run(["git", "-C", str(repo), "apply", "-"], input=incremental, text=True, check=True)
        subprocess.run(["git", "-C", str(repo), "apply", "--reverse", "--check", str(patch_path)], check=True)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            completed = subprocess.run(["bash", str(ROOT / "scripts/p0_setup_transfer.sh")])
        finally:
            server.shutdown()
            worker.join(timeout=5)
        if completed.returncode:
            raise RuntimeError(f"生成环境依赖同步失败，退出码{completed.returncode}；不自动重试")
    return {"wheel_manifest": str(manifest), "wheel_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "lockfile_change": "仅五个已校验文件的交付地址；其余版本、依赖和所有哈希不变",
            "patch": str(patch_path), "patch_sha256": hashlib.sha256(patch.encode()).hexdigest(),
            "listener": f"127.0.0.1:{PORT}，仅本次安装，已关闭", "system_network_changed": False,
            "gpu_probe_performed": False, "generation_performed": False}


if __name__ == "__main__":
    raise SystemExit(run_stage("transfer_local_setup", setup,
                              {"scope": "完整官方cu130依赖安装，本机交付五个已校验wheel"}))
