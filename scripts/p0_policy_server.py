#!/usr/bin/env python3
"""仅监听Thor本机的P0策略服务，用未晋升检查点验证观测到动作接口。

@author pony
@date 2026-11-06
@version v1.0.0
@last_modified 2026-11-06
@changelog
    - v1.0.0 (2026-11-06): loopback HTTP入口，限制请求大小，动作只反归一化一次并记录延迟。
"""
from __future__ import annotations

import argparse
import base64
from http.server import BaseHTTPRequestHandler, HTTPServer
import io
import json
import os
import pathlib
import signal
import time

from p0_common import ROOT, ensure_gpu_idle, run_stage
from p0_policy_runtime import build_model, normalize, restore_checkpoint, sha256


def serve(run_dir: pathlib.Path, args: argparse.Namespace) -> dict:
    ensure_gpu_idle()
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from PIL import Image

    checkpoint = args.checkpoint.resolve(strict=True)
    if not checkpoint.is_relative_to(ROOT):
        raise RuntimeError("检查点必须位于Thor项目内")
    metadata = json.loads((checkpoint / "checkpoint.json").read_text())
    base = pathlib.Path(metadata["base_model"])
    if not base.resolve().is_relative_to(ROOT / "checkpoints/base"):
        raise RuntimeError("基座模型不在项目固定目录")
    manifest = pathlib.Path(metadata["base_manifest"])
    if sha256(manifest) != metadata["base_manifest_sha256"]:
        raise RuntimeError("检查点的基座清单已变化")
    for item in json.loads(manifest.read_text())["files"]:
        if sha256(base / item["file"]) != item["sha256"]:
            raise RuntimeError("基座权重内容与清单不符")
    config = OmegaConf.create(metadata["model_config"])
    if pathlib.Path(config.framework.qwenvl.base_vlm) != base:
        raise RuntimeError("检查点模型配置与基座路径不一致")
    model = build_model(base, metadata["recipe"]["name"], metadata["recipe"]["seed"], config=config)
    metadata, saved = restore_checkpoint(checkpoint, model)
    del saved
    model.eval()
    request_times = []
    errors = []
    state = {"running": True}

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, code, value):
            payload = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path != "/health":
                self.send_json(404, {"error": "unknown route"})
                return
            self.send_json(200, {"status": "ready", "pid": os.getpid(), "checkpoint": str(checkpoint), "state_sha256": metadata["state_sha256"],
                                 "eligible_for_promotion": False, "scope": "P0接口验证，不代表可用任务策略"})

        def do_POST(self):
            if self.path != "/predict":
                self.send_json(404, {"error": "unknown route"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 2 * 1024 ** 2:
                    raise ValueError("请求大小必须在2MiB以内")
                request = json.loads(self.rfile.read(length))
                started = time.perf_counter()
                instruction = request["instruction"]
                if not isinstance(instruction, str) or not 0 < len(instruction) <= 4000:
                    raise ValueError("instruction格式不符")
                observation = np.asarray(request["state"], dtype=np.float32)
                if observation.shape != (8,) or not np.isfinite(observation).all():
                    raise ValueError("state必须是有限的8维向量")
                encoded_images = request["images_png_base64"]
                if not isinstance(encoded_images, list) or len(encoded_images) != 2:
                    raise ValueError("必须提供外部和腕部双视角")
                images = []
                for value in encoded_images:
                    with Image.open(io.BytesIO(base64.b64decode(value, validate=True))) as original:
                        if original.size != (256, 256):
                            raise ValueError("图像必须是256x256")
                        images.append(original.convert("RGB"))
                seed = request.get("seed", 42)
                if type(seed) is not int or not 0 <= seed < 2 ** 32:
                    raise ValueError("seed格式不符")
                example = {"image": images, "lang": instruction,
                           "state": normalize(observation[None], metadata["normalizer"]["state"])}
                with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    torch.manual_seed(seed)
                    actions_normalized = model.predict_action([example])["normalized_actions"][0]
                if actions_normalized.shape != (8, 7) or not np.isfinite(actions_normalized).all():
                    raise RuntimeError("模型输出动作非法")
                actions = normalize(actions_normalized, metadata["normalizer"]["action"], inverse=True)
                torch.cuda.synchronize()
                wall_ms = (time.perf_counter() - started) * 1000
                request_times.append(wall_ms)
                self.send_json(200, {"actions": actions.tolist(), "normalized_actions": actions_normalized.tolist(),
                                     "state_sha256": metadata["state_sha256"], "latency_ms": wall_ms,
                                     "action_semantics": "已经反归一化的raw LIBERO OSC_POSE动作；每20Hz拍执行一行"})
                if args.max_inferences and len(request_times) >= args.max_inferences:
                    state["running"] = False
            except (ValueError, KeyError, TypeError) as error:
                self.send_json(400, {"error": str(error)})
            except Exception as error:
                errors.append(str(error))
                self.send_json(500, {"error": str(error)})

    def stop(_number, _frame):
        state["running"] = False

    for number in [signal.SIGTERM, signal.SIGINT]:
        signal.signal(number, stop)
    with HTTPServer(("127.0.0.1", args.port), Handler) as server:
        server.timeout = 1
        print(json.dumps({"listen": f"http://127.0.0.1:{args.port}", "ready": True, "checkpoint": str(checkpoint)}), flush=True)
        while state["running"]:
            server.handle_request()
    result = {"host": "127.0.0.1", "port": args.port, "checkpoint": str(checkpoint),
              "state_sha256": metadata["state_sha256"], "precision": metadata["precision"], "batch": 1,
              "inferences": len(request_times), "latencies_ms": request_times,
              "latency_includes": "PNG解码、归一化、双视角预处理、VLM、动作头、D2H、反归一化；不含网络和JSON编码",
              "errors": errors, "eligible_for_promotion": False, "environment_actions_sent": False}
    (run_dir / "server_run.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    if errors:
        raise RuntimeError("策略服务遇到推理错误，详情见server_run.json")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--max-inferences", type=int, default=0)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535 or args.max_inferences < 0:
        parser.error("port必须为非特权端口，max-inferences不能为负")
    return run_stage("policy_server", lambda directory: serve(directory, args),
                     {key: str(value) if isinstance(value, pathlib.Path) else value for key, value in vars(args).items()})


if __name__ == "__main__":
    raise SystemExit(main())
