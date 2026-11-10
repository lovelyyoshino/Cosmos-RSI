#!/usr/bin/env python3
"""执行生成前声明的Cosmos背景合成，保存无损RGB与可复核的原生来源。

@author pony
@date 2026-11-07
@version v1.0.0
@last_modified 2026-11-07
@changelog
    - v1.0.0 (2026-11-07): 禁止沿用无目标历史样本，解码验证全部前景与背景像素，默认不可入训。
"""
from __future__ import annotations

import argparse
import json
import pathlib

from p0_common import ROOT, ensure_gpu_idle, run_stage
from p0_policy_runtime import sha256
from p0_prepare_background_transfer import COMPOSITE_RECIPE, TARGET
from p0_protected_background_preview import protected_background


def check_declaration(evidence: dict) -> None:
    if evidence.get("target_condition") != TARGET or evidence.get("composite_recipe") != COMPOSITE_RECIPE \
            or evidence.get("source_success") is not True or not evidence.get("replay_validation", {}).get("passed"):
        raise ValueError("只接受生成前明确声明背景目标和前景保护方法的成功源")


def decode_frames(path: pathlib.Path):
    import cv2
    import imageio.v2 as imageio
    import numpy as np

    with imageio.get_reader(path) as reader:
        frames = np.asarray([cv2.resize(frame, (256, 256), interpolation=cv2.INTER_AREA) for frame in reader])
    if frames.shape != (93, 256, 256, 3) or frames.dtype != np.uint8:
        raise ValueError("窗口必须实际解码为93帧uint8 RGB，不截断或补帧")
    return frames


def verify_composite_pixels(source, native, masks, output, radius: int) -> None:
    import numpy as np

    if source.shape != native.shape or source.shape != output.shape or source.shape != (93, 256, 256, 3) \
            or masks.ndim != 4 or masks.shape[0] != 93 or masks.shape[2:] != (256, 256):
        raise ValueError("派生数据必须完整逐帧对齐同源RGB与ROI")
    for original, generated, rois, actual in zip(source, native, masks, output):
        expected, _ = protected_background(original, generated, rois, radius)
        if actual.dtype != np.uint8 or not np.array_equal(expected, actual):
            raise ValueError("实际派生视频偏离预声明合成方法或改变前景像素")


def composite(directory: pathlib.Path, args: argparse.Namespace) -> dict:
    ensure_gpu_idle()
    import imageio.v2 as imageio
    import numpy as np
    from p0_generated_window_gate import verified_generation_paths

    native = json.loads(args.probe.read_text())
    if native.get("stage") != "transfer_smoke":
        raise ValueError("只从原生Cosmos生成建立单级派生数据")
    source_path, native_path = verified_generation_paths(native, args.probe)
    evidence = json.loads(source_path.read_text())
    check_declaration(evidence)
    preparation = json.loads((source_path.parent / "probe.json").read_text())
    if preparation.get("stage") != "background_transfer_input" or preparation.get("status") != "passed" \
            or preparation.get("artifacts_sha256", {}).get(source_path.name) != sha256(source_path):
        raise ValueError("背景目标未绑定生成前的新源阶段")
    for name, checksum in {**evidence["files_sha256"], "source_labels.npz": evidence["labels_sha256"]}.items():
        if pathlib.PurePosixPath(name).name != name or sha256(source_path.parent / name) != checksum:
            raise ValueError("源RGB、ROI或动作标签改变")
    source, generated = decode_frames(source_path.parent / "source.mp4"), decode_frames(native_path)
    masks = np.load(source_path.parent / "roi_masks.npy", allow_pickle=False)
    output = np.asarray([protected_background(a, b, m, COMPOSITE_RECIPE["radius"])[0]
                         for a, b, m in zip(source, generated, masks)])
    video = directory / "protected_background_lossless.mp4"
    with imageio.get_writer(video, fps=20, codec="libx264rgb", pixelformat="rgb24", macro_block_size=1,
                           ffmpeg_params=["-crf", "0", "-preset", "veryfast", "-threads", "2", "-movflags", "+faststart"]) as writer:
        for frame in output:
            writer.append_data(frame)
    verify_composite_pixels(source, generated, masks, decode_frames(video), COMPOSITE_RECIPE["radius"])
    result = {"derived_kind": COMPOSITE_RECIPE["kind"], "composite_recipe": COMPOSITE_RECIPE,
              "native_generation_probe": str(args.probe), "native_generation_probe_sha256": sha256(args.probe),
              "native_video": str(native_path), "native_video_sha256": sha256(native_path),
              "source_success_evidence": native["result"]["source_success_evidence"],
              "outputs": [str(video)], "output_video_sha256": sha256(video),
              "guardrails_enabled": native["result"]["guardrails_enabled"],
              "guardrails_scope": "原生Cosmos输出已安全检查；派生仅回填已核验训练源的保护区域",
              "target_condition": TARGET, "source_labels_sha256": evidence["labels_sha256"],
              "lossless_pixels_verified": True, "foreground_max_abs_diff_all_frames": 0,
              "frames": 93, "fps": 20, "sample_count": 1, "batch": None, "seed": None,
              "precision": "CPU uint8 RGB；libx264rgb CRF0无损；全部解码逐像素复核",
              "generation_performed": False, "eligible_for_training": False,
              "limitation": "明确派生合成；完整原质量门、任务语义及背景目标条件仍须独立验收"}
    (directory / "composite.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--probe", type=pathlib.Path, required=True)
    args = parser.parse_args()
    args.probe = args.probe.resolve(strict=True)
    if not args.probe.is_relative_to(ROOT / "runs/p0"):
        parser.error("生成probe必须位于Thor项目P0目录")
    raise SystemExit(run_stage("background_composite", lambda directory: composite(directory, args),
                               {"compute": "CPU_only", "composite_recipe": COMPOSITE_RECIPE}))
