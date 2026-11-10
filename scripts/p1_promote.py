#!/usr/bin/env python3
"""在具体门禁获得用户明确批准后保存、上线并验证策略；失败恢复原生产链接。

@author pony
@date 2026-11-09
@version v1.0.0
@last_modified 2026-11-09
@changelog
    - v1.0.0 (2026-11-09): 绑定具体通过门禁和审批引用，复制已晋升权重、生产冒烟与失败回滚。
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import fcntl
import io
import json
import os
import pathlib
import shutil

from p0_common import ROOT, ensure_gpu_idle, run_stage
from p0_deploy_smoke import atomic_slot, server_phase
from p0_lerobot_handoff import dump
from p0_policy_runtime import DemoCache
from p1_eval_gate import checked_update, sealed, sha256


def verify_approval(gate, actual_digest, approved_digest, reference):
    if actual_digest != approved_digest or len(approved_digest) != 64 or not reference.strip() \
            or gate.get("statistical_gate_passed") is not True or gate.get("eligible_for_promotion") is not True \
            or gate.get("decision") != "accept_pending_approval" or gate.get("human_approval_required") is not True:
        raise ValueError("没有针对这一份已通过门禁的明确用户审批")


def promote(directory, args):
    import numpy as np
    from PIL import Image

    ensure_gpu_idle()
    gate, _ = sealed(args.gate, "p1_evaluation_gate")
    verify_approval(gate, sha256(args.gate), args.approve_gate_sha256, args.approval_reference)
    # 具体门禁的所有输入仍须保留原字节，不能晋升审批后替换的候选或评测。
    for binding in gate["inputs"].values():
        path = pathlib.Path(binding["path"]).resolve(strict=True)
        if not path.is_relative_to(ROOT) or sha256(path) != binding["sha256"]:
            raise ValueError("门禁输入自审批后已经变化")
    training_path = pathlib.Path(gate["inputs"]["candidate_training"]["path"])
    training, metadata = checked_update(training_path, "candidate")
    candidate = pathlib.Path(training["output_ckpt"]).resolve(strict=True)
    parent = pathlib.Path(training["parent_checkpoint"]).resolve(strict=True)
    if gate["candidate_ckpt_sha256"] != training["output_ckpt_sha256"] \
            or gate["parent_ckpt_sha256"] != sha256(parent / "state.pt"):
        raise ValueError("具体门禁与待晋升父子权重不同")
    cache = DemoCache(pathlib.Path(metadata["cache_manifest"]).parent)
    if sha256(cache.manifest_path) != metadata["cache_manifest_sha256"]:
        raise ValueError("生产冒烟的固定示范版本不符")
    location = metadata["prediction_probe"]["locations"][0]
    episode, = [row for row in cache.manifest["episodes"] if row["episode_id"] == location["episode_id"]]
    if episode["split"] != "demo_validation" or not 0 <= location["frame"] < episode["length"]:
        raise ValueError("生产冒烟只接受原有离线验证观测")
    offset = episode["offset"] + location["frame"]
    images = []
    for view in range(2):
        buffer = io.BytesIO()
        Image.fromarray(np.array(cache.images[offset, view])).save(buffer, format="PNG")
        images.append(base64.b64encode(buffer.getvalue()).decode())
    payload = {"instruction": episode["instruction"], "state": cache.states[offset].tolist(),
               "images_png_base64": images, "seed": 44}
    slots = ROOT / "checkpoints"
    slots.mkdir(exist_ok=True)
    promoted_root = slots / "promoted"
    promoted_root.mkdir(exist_ok=True)
    destination = promoted_root / f"iter{gate['iteration']:03d}_{training['output_ckpt_sha256'][:12]}"
    with (slots / ".promotion.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        original = {}
        for name in ["current", "previous"]:
            path = slots / name
            if path.is_symlink():
                original[name] = path.resolve(strict=True)
            elif path.exists():
                raise ValueError("拒绝覆盖普通文件生产槽位")
            else:
                original[name] = None
        if original["current"] not in {None, parent} or (gate["iteration"] > 1 and original["current"] != parent):
            raise ValueError("生产当前权重已经变化，禁止覆盖另一圈晋升")
        if destination.exists():
            raise ValueError("已晋升保存目录已经存在，禁止覆盖；先核查此前晋升记录")
        shutil.copytree(candidate, destination)
        if sha256(destination / "state.pt") != training["output_ckpt_sha256"]:
            raise ValueError("已晋升保存权重与通过门禁的候选不一致")
        try:
            atomic_slot(slots, "previous", parent, ROOT)
            atomic_slot(slots, "current", destination, ROOT)
            phase = server_phase(directory, "production", slots / "current", payload, 18794, 5, 50)
            if phase["response"]["state_sha256"] != training["output_ckpt_sha256"]:
                raise ValueError("生产链接服务未使用新权重")
        except BaseException:
            for name, target in original.items():
                if target is None:
                    (slots / name).unlink(missing_ok=True)
                else:
                    atomic_slot(slots, name, target, ROOT)
            dump(directory / "rollback.json", {"restored_original_slots": True,
                 "saved_candidate_retained": str(destination), "production_deployed": False})
            raise
    result = {"schema": "thor-rsi-promotion-v1", "iteration": gate["iteration"],
              "gate": str(args.gate), "gate_sha256": sha256(args.gate),
              "explicit_user_approval": True, "approval_reference": args.approval_reference,
              "policy_ckpt_sha256": training["output_ckpt_sha256"], "parent_ckpt_sha256": training["parent_ckpt_sha256"],
              "promoted_checkpoint": str(destination), "promoted_metadata_sha256": sha256(destination / "checkpoint.json"),
              "production_link": str(slots / "current"), "rollback_to": str(parent),
              "production_deployed": True, "production_link_smoke_passed": True,
              "service": {key: value for key, value in phase.items() if key not in {"response", "rows"}},
              "policy_server_left_running": False, "serial_stage_policy": "部署冒烟正常退出，下一圈采集按current重新加载",
              "rollout_started_at": gate["rollout_started_at"],
              "rollout_to_deployment_wall_s": (dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(gate["rollout_started_at"])).total_seconds(),
              "real_robot_commands_sent": False, "complete_rsi": False,
              "backup_required": True, "backup_performed": False}
    dump(directory / "promotion.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gate", type=pathlib.Path, required=True)
    parser.add_argument("--approve-gate-sha256", required=True)
    parser.add_argument("--approval-reference", required=True, help="具体用户审批的可核验会话引用；代理不得自行批准")
    args = parser.parse_args()
    args.gate = args.gate.resolve(strict=True)
    raise SystemExit(run_stage("p1_promotion", lambda directory: promote(directory, args),
                               {"compute": "Thor生产链接服务冒烟，无环境动作", "batch": 1,
                                "seed": 44, "warmup": 5, "requests": 50}))
