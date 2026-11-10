"""LIBERO本机再生与评测共用的动作过滤、状态语义和初态指纹。

@author pony
@date 2026-11-06
@version v1.3.0
@last_modified 2026-11-06
@changelog
    - v1.3.0 (2026-11-06): 捕获MuJoCo积分状态，覆盖求解器warmstart和控制输入等MjSimState遗漏项。
    - v1.2.0 (2026-11-06): 单独捕获和恢复MjSimState不包含的fixture模型位姿。
    - v1.1.0 (2026-11-06): 使用namespace package的__path__核验来源，支持同一worker切换任务。
    - v1.0.0 (2026-11-06): 明确动作执行前记录，初态比较排除时钟与机器人关节。
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sys

from p0_common import ROOT


def configure_libero(repo: str, directory: pathlib.Path) -> pathlib.Path:
    if repo not in {"LIBERO", "LIBERO-plus"}:
        raise ValueError("未知LIBERO实现")
    source = ROOT / "third_party" / repo
    benchmark = source / "libero/libero"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.yaml").write_text(json.dumps({
        "benchmark_root": str(benchmark), "bddl_files": str(benchmark / "bddl_files"),
        "init_states": str(benchmark / "init_files"), "datasets": str(ROOT / "data" / repo),
        "assets": str(benchmark / "assets")}) + "\n")
    # 两套实现的Python包同名；每个子进程只能导入其中一种。
    imported = sys.modules.get("libero")
    if imported:
        origins = [imported.__file__] if getattr(imported, "__file__", None) else list(imported.__path__)
        if not origins or any(not pathlib.Path(path).resolve().is_relative_to(source.resolve()) for path in origins):
            raise RuntimeError("不能在同一进程混用LIBERO与LIBERO-plus")
    os.environ["LIBERO_CONFIG_PATH"] = str(directory)
    sys.path.insert(0, str(source))
    return benchmark


def kept_action_indices(actions):
    import numpy as np

    values = np.asarray(actions)
    if values.ndim != 2 or values.shape[1] != 7 or not np.isfinite(values).all():
        raise ValueError("示范动作必须为有限的T×7数组")
    kept = []
    for index, action in enumerate(values):
        if np.linalg.norm(action[:-1]) < 1e-4:
            if not kept or action[-1] == values[kept[-1], -1]:
                continue
        kept.append(index)
    return np.asarray(kept, dtype=np.int64)


def capture_fixture_state(env) -> dict:
    import numpy as np

    names = sorted(obj.root_body for obj in env.env.fixtures_dict.values())
    ids = [env.sim.model.body_name2id(name) for name in names]
    # 固定物体的位姿写在MjModel中，不能依靠get_sim_state()保存。
    return {"fixture_body_names": np.asarray(names, dtype="U"),
            "fixture_body_pos": np.array(env.sim.model.body_pos[ids], dtype=np.float64),
            "fixture_body_quat": np.array(env.sim.model.body_quat[ids], dtype=np.float64)}


def restore_fixture_state(env, snapshot) -> None:
    import numpy as np

    expected = sorted(obj.root_body for obj in env.env.fixtures_dict.values())
    names = np.asarray(snapshot["fixture_body_names"])
    positions = np.asarray(snapshot["fixture_body_pos"])
    quaternions = np.asarray(snapshot["fixture_body_quat"])
    if names.ndim != 1 or names.tolist() != expected or len(set(expected)) != len(expected) \
            or positions.shape != (len(expected), 3) or quaternions.shape != (len(expected), 4) \
            or not np.isfinite(positions).all() or not np.isfinite(quaternions).all():
        raise RuntimeError("fixture快照名称、形状或有限性不符合当前场景")
    ids = [env.sim.model.body_name2id(name) for name in expected]
    env.sim.model.body_pos[ids] = positions
    env.sim.model.body_quat[ids] = quaternions


def capture_integration_state(env) -> dict:
    import mujoco
    import numpy as np

    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(env.sim.model._model, spec), dtype=np.float64)
    mujoco.mj_getState(env.sim.model._model, env.sim.data._data, state, spec)
    return {"integration_state": state, "integration_spec": np.asarray(int(spec), dtype=np.int64)}


def restore_integration_state(env, snapshot) -> None:
    import mujoco
    import numpy as np

    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.asarray(snapshot["integration_state"], dtype=np.float64)
    if int(snapshot["integration_spec"]) != int(spec) \
            or state.shape != (mujoco.mj_stateSize(env.sim.model._model, spec),) or not np.isfinite(state).all():
        raise RuntimeError("MuJoCo积分状态与当前场景不兼容")
    mujoco.mj_setState(env.sim.model._model, env.sim.data._data, state, spec)


def observation_state(obs):
    import numpy as np
    from robosuite.utils.transform_utils import quat2axisangle

    result = np.concatenate([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"].copy()),
                             obs["robot0_gripper_qpos"]]).astype(np.float32)
    if result.shape != (8,) or not np.isfinite(result).all():
        raise RuntimeError("示范观测必须为有限的8维EEF/夹爪状态")
    return result


def scene_joints(env) -> dict:
    import numpy as np

    result = {}
    for name in env.sim.model.joint_names:
        if name.startswith("robot"):
            continue
        value = np.atleast_1d(env.sim.data.get_joint_qpos(name)).astype(np.float64)
        if not np.isfinite(value).all():
            raise RuntimeError("场景初态包含非有限关节")
        # 自由关节的q与-q表示同一姿态，指纹统一四元数符号。
        if len(value) == 7:
            quaternion = value[3:]
            nonzero = np.flatnonzero(np.abs(quaternion) > 1e-12)
            if len(nonzero) and quaternion[nonzero[0]] < 0:
                value[3:] *= -1
        result[name] = value.tolist()
    if not result:
        raise RuntimeError("没有找到可用于去重的场景关节")
    return result


def scene_digest(scene: dict) -> str:
    import numpy as np

    canonical = {name: [float(x) for x in np.round(value, 8)] for name, value in sorted(scene.items())}
    return hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def scene_distance(first: dict, second: dict) -> float:
    import numpy as np

    # layout可能添加物体；比较两者共同的全部原任务关节，避免新物体掩盖旧初态重复。
    common = set(first) & set(second)
    if not common or any(np.shape(first[key]) != np.shape(second[key]) for key in common):
        raise ValueError("场景没有兼容的共同物体关节")
    return max(float(np.max(np.abs(np.asarray(first[key]) - second[key]))) for key in common)
