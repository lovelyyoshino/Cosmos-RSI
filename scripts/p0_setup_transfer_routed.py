#!/usr/bin/env python3
"""仅为本次uv安装选择可达的官方CDN地址；TLS与上游锁文件均不变。

@author pony
@date 2026-11-06
@version v1.0.0
@last_modified 2026-11-06
@changelog
    - v1.0.0 (2026-11-06): loopback CONNECT隧道只转发TLS字节，不改系统DNS或解密流量。
"""
from __future__ import annotations

import json
import os
import pathlib
import select
import socket
import socketserver
import subprocess
import threading

from p0_common import ROOT, run_stage

CDN_HOST = "release-assets.githubusercontent.com"
CDN_ADDRESS = "185.199.111.133"
ALLOWED_HOSTS = {"github.com", CDN_HOST, "objects.githubusercontent.com", "nvidia-cosmos.github.io",
                 "mirrors.aliyun.com", "download.pytorch.org", "pypi.org", "files.pythonhosted.org"}


def setup(run_dir: pathlib.Path) -> dict:
    addresses = {entry[4][0] for entry in socket.getaddrinfo(CDN_HOST, 443, socket.AF_INET, socket.SOCK_STREAM)}
    if CDN_ADDRESS not in addresses:
        raise RuntimeError("预检可达地址已不在官方CDN的DNS结果中，拒绝继续")
    connections = []

    class Tunnel(socketserver.StreamRequestHandler):
        def handle(self):
            self.connection.settimeout(30)
            first = self.rfile.readline(4096).decode("ascii").strip().split()
            if len(first) != 3 or first[0] != "CONNECT":
                self.wfile.write(b"HTTP/1.1 405 Method Not Allowed\r\n\r\n")
                return
            host, port = first[1].rsplit(":", 1)
            if host not in ALLOWED_HOSTS or port != "443":
                self.wfile.write(b"HTTP/1.1 403 Forbidden\r\n\r\n")
                return
            size = 0
            while True:
                line = self.rfile.readline(4096)
                size += len(line)
                if size > 16384:
                    raise RuntimeError("CONNECT头部过大")
                if line in [b"\r\n", b"\n", b""]:
                    break
            address = CDN_ADDRESS if host == CDN_HOST else host
            item = {"host": host, "pinned_to_official_address": host == CDN_HOST, "bytes_up": 0, "bytes_down": 0}
            connections.append(item)
            with socket.create_connection((address, 443), timeout=30) as upstream:
                upstream.settimeout(300)
                self.connection.settimeout(300)
                self.connection.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                # 只转发加密TLS记录；证书校验与文件哈希由原uv客户端继续执行。
                while True:
                    readable, _, _ = select.select([self.connection, upstream], [], [], 300)
                    if not readable:
                        return
                    for source in readable:
                        data = source.recv(65536)
                        if not data:
                            return
                        destination = upstream if source is self.connection else self.connection
                        destination.sendall(data)
                        item["bytes_up" if source is self.connection else "bytes_down"] += len(data)

    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = False
        daemon_threads = True

    with Server(("127.0.0.1", 0), Tunnel) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        proxy = f"http://127.0.0.1:{server.server_address[1]}"
        env = {**os.environ, "HTTPS_PROXY": proxy, "HTTP_PROXY": proxy, "https_proxy": proxy, "http_proxy": proxy,
               "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost"}
        print(json.dumps({"transport": "临时本机CONNECT，TLS未解密", "cdn": CDN_HOST,
                          "address": CDN_ADDRESS, "official_dns_addresses": sorted(addresses)}, ensure_ascii=False), flush=True)
        try:
            completed = subprocess.run(["bash", str(ROOT / "scripts/p0_setup_transfer.sh")], env=env)
        finally:
            server.shutdown()
            worker.join(timeout=5)
            (run_dir / "connections.json").write_text(json.dumps(connections, indent=2) + "\n")
        if completed.returncode:
            raise RuntimeError(f"官方cu130安装退出码{completed.returncode}，详见安装日志；不自动重复运行")
    return {"official_cdn_host": CDN_HOST, "selected_address": CDN_ADDRESS,
            "tls_verification": "原uv客户端端到端执行，未关闭证书验证", "lockfile_changed": False,
            "system_dns_or_hosts_changed": False, "proxy_scope": "只供本次安装子进程，完成后退出",
            "connections": connections, "generation_performed": False}


if __name__ == "__main__":
    raise SystemExit(run_stage("transfer_routed_setup", setup, {"scope": "官方依赖安装，非生成或模型测速"}))
