#!/usr/bin/env python3
"""为同一成功源准备仅桌面深棕材质参考；诊断控制不授予训练资格。

@author pony
@date 2026-11-08
@version v1.0.0
@last_modified 2026-11-08
@changelog
    - v1.0.0 (2026-11-08): 固定RGB增益与可改写区内渐变，绑定原RGB/标签和原生blur控制。
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

from p0_common import ROOT, ensure_gpu_idle, run_stage
from p0_policy_runtime import sha256


CONFIG_PATH = ROOT / "configs/table_reference_preview.yaml"
CONFIG = {"version": "thor-table-reference-preview-v1", "purpose": "design_only_not_training",
          "design_episode_suffix": "caddy-demo_16", "target_condition": {"background": "matte_dark_wood"},
          "model": "base_edge_vis", "frames": 93, "fps": 20, "resolution": "480", "num_steps": 35,
          "seed": 42, "batch": 1, "edge_weight": .2, "vis_weight": .8,
          "reference_rgb_gain": [.55, .42, .35], "reference_feather_inside_editable_pixels": 8,
          "protected_radius_pixels": 8, "blur_preset": "high", "eligible_for_training": False}
UPSTREAM_SHA = "2ff49d0717af02057ae79bc75c00fbff9da1b4e7"


def checked_config():
    import yaml

    if yaml.safe_load(CONFIG_PATH.read_text()) != CONFIG:
        raise ValueError("材质参考试验配置偏离运行前声明")
    return CONFIG


def table_reference(source, editable):
    """只在可改写桌体内部渐变变暗，杯子/书本/机器人及保护邻域保持源RGB。"""
    import cv2
    import numpy as np

    source, editable = np.asarray(source), np.asarray(editable)
    if source.ndim != 4 or source.shape[-1] != 3 or source.dtype != np.uint8 \
            or editable.dtype != np.bool_ or editable.shape != source.shape[:-1] \
            or not editable.any() or editable.all():
        raise ValueError("材质参考需要uint8源及同帧非平凡bool桌体区域")
    alpha = []
    for mask in editable:
        if not mask.any():
            raise ValueError("每帧必须存在可改写桌体")
        distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        ratio = np.clip(distance / CONFIG["reference_feather_inside_editable_pixels"], 0, 1)
        alpha.append(ratio * ratio * (3 - 2 * ratio))
    alpha = np.asarray(alpha, dtype=np.float32)
    dark = source.astype(np.float32) * np.asarray(CONFIG["reference_rgb_gain"], dtype=np.float32)
    result = np.rint(source * (1 - alpha[..., None]) + dark * alpha[..., None]).clip(0, 255).astype(np.uint8)
    if not np.array_equal(result[~editable], source[~editable]):
        raise ValueError("材质参考改动了不可改写区域")
    return result, alpha


def prepare(directory, args):
    ensure_gpu_idle()
    checked_config()
    import numpy as np
    from PIL import Image
    from p0_prepare_table_background import write_video
    from p0_lossless_background import read_exact_video
    from p0_table_background import checked_table_source

    evidence, source, masks, editable, _, _ = checked_table_source(args.source_input)
    if not evidence["source_episode_id"].endswith(CONFIG["design_episode_suffix"]):
        raise ValueError("新材质配方先只使用预声明的已见设计成功训练源")
    reference, alpha = table_reference(source, editable)
    write_video(directory / "reference_rgb.mp4", reference)
    actual, _ = read_exact_video(directory / "reference_rgb.mp4")
    if not np.array_equal(actual, reference):
        raise ValueError("参考RGB未按无损字节编码")
    repo = ROOT / "third_party/cosmos-transfer2.5"
    import subprocess

    if subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip() != UPSTREAM_SHA:
        raise ValueError("blur变换必须使用已审查固定上游")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, str(repo))
    from cosmos_transfer2._src.transfer2.datasets.augmentors.control_input import AddControlInputBlur

    transform = AddControlInputBlur(input_keys=["video"], use_random=False, downup_preset="high", resize_cuda=False)
    blurred = transform({"video": reference.transpose(3, 0, 1, 2)})["control_input_vis"]
    blurred = blurred.permute(1, 2, 3, 0).cpu().numpy().astype(np.uint8)
    write_video(directory / "vis.mp4", blurred)
    actual_blur, _ = read_exact_video(directory / "vis.mp4")
    if not np.array_equal(actual_blur, blurred):
        raise ValueError("材质blur控制未按无损字节编码")
    np.save(directory / "reference_alpha.npy", alpha, allow_pickle=False)
    original = json.loads((args.source_input.parent / "spec.json").read_text())
    spec = {"name": "p0_table_reference_preview_v1", "video_path": str(args.source_input.parent / "source.mp4"),
            "prompt": original["prompt"], "max_frames": 93, "num_video_frames_per_chunk": 93,
            "resolution": "480", "num_steps": 35, "seed": 42, "guidance": 3,
            "edge": {"control_path": str(args.source_input.parent / "edge.mp4"), "control_weight": .2,
                     "mask_path": str(args.source_input.parent / "scene_mask.mp4")},
            "vis": {"control_path": str(directory / "vis.mp4"), "control_weight": .8}}
    (directory / "spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n")
    record = {"method": CONFIG["version"], "config_sha256": sha256(CONFIG_PATH),
              "source_input": str(args.source_input), "source_input_sha256": sha256(args.source_input),
              "source_episode_id": evidence["source_episode_id"], "source_labels_sha256": evidence["labels_sha256"],
              "source_rgb_sha256": evidence["files_sha256"]["source.mp4"],
              "source_mask_sha256": evidence["files_sha256"]["scene_mask.mp4"],
              "source_edge_sha256": evidence["files_sha256"]["edge.mp4"],
              "reference_unchanged_protected_pixels": True, "source_instruction": evidence["instruction"],
              "target_condition": CONFIG["target_condition"], "spec": str(directory / "spec.json"),
              "spec_sha256": sha256(directory / "spec.json"), "upstream_sha": UPSTREAM_SHA,
              "files_sha256": {name: sha256(directory / name) for name in
                  ["reference_rgb.mp4", "vis.mp4", "reference_alpha.npy"]},
              "comparison_claim": "新base/双控制配方工程试验；非与distilled的单因素比较，非策略收益",
              "eligible_for_training": False, "diagnostic_only": True}
    (directory / "reference_input.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    for frame in [0, 46, 92]:
        sheet = np.concatenate([source[frame], reference[frame], blurred[frame]], axis=1)
        Image.fromarray(sheet).save(directory / f"reference_{frame:03d}.png")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-input", type=pathlib.Path, required=True)
    arguments = parser.parse_args()
    arguments.source_input = arguments.source_input.resolve(strict=True)
    raise SystemExit(run_stage("table_reference_input", lambda directory: prepare(directory, arguments),
                     {"method": CONFIG["version"], "compute": "CPU uint8/float32，不运行模型", "source_windows": 1,
                      "frames": 93, "fps": 20, "diagnostic_only": True}))
