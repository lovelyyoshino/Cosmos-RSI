"""恢复成功示范的再生环境历史，并保存MjSimState之外的重置快照。

@author pony
@date 2026-11-07
@version v1.0.0
@last_modified 2026-11-07
@changelog
    - v1.0.0 (2026-11-07): 按原始数字顺序重放前序示范，供源RGB/深度/分割共享同一重置状态。
"""
from __future__ import annotations

import pathlib

from p0_common import ROOT
from p0_policy_runtime import sha256
from p0_sim_data import (capture_fixture_state, capture_integration_state, kept_action_indices,
                         restore_fixture_state, restore_integration_state)


PROTOCOL = "raw-demo-history+fixture+mjSTATE_INTEGRATION-v1"


def preceding_demo_ids(keys, selected: str) -> list[str]:
    names = list(keys)
    if any(not name.startswith("demo_") or not name[5:].isdigit() for name in names):
        raise ValueError("原始示范必须采用demo_数字命名")
    ordered = sorted(names, key=lambda name: int(name[5:]))
    if selected not in ordered:
        raise ValueError("选中示范不在原始来源中")
    return ordered[:ordered.index(selected)]


def initialize_demo(env, row: dict, physical, reset_snapshot=None):
    """先恢复原采集历史或已核验快照，再按原流程设初态和稳定十拍。"""
    import numpy as np

    if row.get("init_seed") != 0 or row.get("settle_steps") != 10:
        raise RuntimeError("只支持既有D0的seed0和十步稳定协议")
    noop = np.asarray([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
    history = []
    if reset_snapshot is None:
        import h5py

        source = pathlib.Path(row["source_file"]).resolve(strict=True)
        if not source.is_relative_to(ROOT / "data") or sha256(source) != row["source_file_sha256"]:
            raise RuntimeError("再生历史的原始HDF5改变或越界")
        with h5py.File(source, "r") as handle:
            demos = handle["data"]
            history = preceding_demo_ids(demos.keys(), row["source_demo_id"])
            if not np.array_equal(demos[row["source_demo_id"]]["states"][0], physical["raw_initial_state"]):
                raise RuntimeError("缓存原始初态与已核验HDF5不一致")
            # 原再生脚本每个任务复用环境；只重新seed不会清除模型对象和采样器的历史。
            for demo_id in history:
                env.seed(0)
                env.reset()
                env.set_init_state(demos[demo_id]["states"][0])
                for _ in range(10):
                    env.step(noop)
                values = demos[demo_id]["actions"][:].astype(np.float32)
                for index in kept_action_indices(values):
                    env.step(values[index])
    env.seed(0)
    env.reset()
    if reset_snapshot is not None:
        restore_fixture_state(env, reset_snapshot)
        restore_integration_state(env, reset_snapshot)
    snapshot = {**capture_fixture_state(env), **capture_integration_state(env)}
    obs = env.set_init_state(physical["raw_initial_state"])
    for _ in range(10):
        obs, _, _, _ = env.step(noop)
    return obs, snapshot, {"protocol": PROTOCOL, "source_demo_id": row["source_demo_id"],
                            "history_demo_ids": history, "reused_reset_snapshot": reset_snapshot is not None,
                            "init_seed": 0, "settle_steps": 10}
