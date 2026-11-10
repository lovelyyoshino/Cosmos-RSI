"""LIBERO Predict的显式输入约定；不推测缺失动作、不沿用VLA的D0归一化。

@author pony
@date 2026-11-09
@version v1.0.0
@last_modified 2026-11-09
@changelog
    - v1.0.0 (2026-11-09): 官方min/max动作尺度、真实九维状态及相机方向转换。
"""
from __future__ import annotations


def predictor_observation(packet):
    import numpy as np
    for key in ["agentview", "wrist"]:
        values = packet[key]
        if values.dtype != np.uint8 or values.shape != (17, 256, 256, 3):
            raise ValueError("Predict验证需要完整十七帧无损双视角，不允许补帧")
    proprio = packet["proprio"]
    if proprio.shape != (17, 9) or not np.isfinite(proprio).all() \
            or not np.allclose(np.linalg.norm(proprio[:, 5:9], axis=-1), 1., atol=1e-5):
        raise ValueError("Predict需要实际采集的九维四元数状态")
    # 现有归档为双轴翻转；再水平翻转，得到官方评测使用的单竖直翻转方向。
    return {"primary_image": np.ascontiguousarray(packet["agentview"][0, :, ::-1]),
            "wrist_image": np.ascontiguousarray(packet["wrist"][0, :, ::-1]),
            "proprio": proprio[0].copy()}


def normalized_actions(actions, stats):
    import numpy as np
    values = np.asarray(actions, dtype=np.float32)
    low, high = [np.asarray(stats[key], dtype=np.float32) for key in ["actions_min", "actions_max"]]
    if values.shape != (16, 7) or low.shape != (7,) or high.shape != (7,) \
            or not np.isfinite(values).all() or not np.isfinite(low).all() or not np.isfinite(high).all() \
            or not np.all(high > low):
        raise ValueError("Predict动作需要真实连续16×7及官方七维min/max统计")
    return 2 * (values - low) / (high - low) - 1
