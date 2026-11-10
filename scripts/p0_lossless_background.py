#!/usr/bin/env python3
"""无损背景派生的像素、视频时间和成功训练源核验；不自行赋予训练资格。

@author pony
@date 2026-11-08
@version v1.1.0
@last_modified 2026-11-08
@changelog
    - v1.1.0 (2026-11-08): 同一入口拒绝实际文件错绑，核对指令及冻结来源，绑定完整设计/独立控制。
    - v1.0.0 (2026-11-08): 严格解码不缩放输出，逐帧核对源标签、ROI与实际合成像素。
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import pathlib
import subprocess

from p0_common import ROOT
from p0_policy_runtime import sha256
from p0_protected_background_preview import protected_background


EXPECTED_CONFIG = {
    "version": "thor-lossless-background-v1", "geometry_evidence_type": "lossless_foreground_pixels_v1",
    "required_frames": 93, "fps": 20, "width": 256, "height": 256, "protected_radius_pixels": 8,
    "roi_names": ["robot", "black_book_1", "desk_caddy_1"], "native_resize": "opencv_inter_area",
    "source_mask": "sim_seg", "design_episode_suffix": "caddy-demo_16", "heldout_episode_suffix": "caddy-demo_19",
    "temporal_residual_p95_max": .08, "semantic_review_required": True,
    "target_condition_review_required": True, "background_temporal_review_required": True,
}

METHOD_DEPENDENCIES = [
    "scripts/p0_lossless_background.py", "scripts/p0_lossless_background_controls.py",
    "scripts/p0_protected_background_preview.py", "scripts/p0_context_normalized_segmentation.py",
    "scripts/p0_segmentation_controls.py", "scripts/p0_background_preview_materials.py",
    "scripts/p0_policy_runtime.py", "configs/lossless_background_gate.yaml",
]
NEGATIVE_CONTROLS = {"identity", "lossless_roundtrip", "analytic_background_dim", "analytic_background_gray"}
ERROR_CONTROLS = {
    "protected_pixel_changed", "book_shift12", "robot_shift12", "global_shift24", "robot_box_occluded",
    "swap_first_last_frames", "wrong_fps", "missing_frame", "wrong_size", "wrong_action", "wrong_state",
    "wrong_frame_index", "wrong_timestamp", "wrong_roi_sha", "wrong_source_binding", "wrong_native_binding",
}


def checked_bound_file(path: pathlib.Path, checksum: str, allowed_root: pathlib.Path,
                       expected_path: pathlib.Path | None = None) -> pathlib.Path:
    """路径与字节同时绑定；相同内容的另一路径也不能替换预声明输入。"""
    try:
        resolved = path.resolve(strict=True)
        if not resolved.is_file() or not resolved.is_relative_to(allowed_root.resolve(strict=True)) \
                or (expected_path is not None and resolved != expected_path.resolve(strict=True)) \
                or not isinstance(checksum, str) or len(checksum) != 64 or sha256(resolved) != checksum:
            raise ValueError("实际文件的路径或SHA偏离预声明绑定")
    except (OSError, RuntimeError) as exc:
        raise ValueError("绑定文件不存在或不能解析") from exc
    return resolved


def validate_source_files(evidence: dict, directory: pathlib.Path) -> None:
    checksums = evidence.get("files_sha256", {})
    if not {"source.mp4", "roi_masks.npy", "source_reset_snapshot.npz"} <= checksums.keys():
        raise ValueError("成功源必须包含视频、仿真ROI及完整重放快照")
    for name, checksum in {**checksums, "source_labels.npz": evidence.get("labels_sha256")}.items():
        if pathlib.PurePosixPath(name).name != name:
            raise ValueError("源文件名不能越过当前目录")
        checked_bound_file(directory / name, checksum, directory)


def validate_controls_summary(report: dict, role: str) -> None:
    if report.get("method") != EXPECTED_CONFIG["version"] or report.get("role") != role \
            or report.get("frames") != list(range(93)) or report.get("validation_passed") is not True \
            or report.get("eligible_for_training") is not False \
            or set(report.get("negative_controls", {})) != NEGATIVE_CONTROLS \
            or set(report.get("positive_controls", {})) != ERROR_CONTROLS:
        raise ValueError("缺少固定版本的完整设计/独立正负控，旧简化检查不能替代")
    if any(row.get("passed") is not True or row.get("decoded_video_checked") is not True
           for row in report["negative_controls"].values()) \
            or any(row.get("detected") is not True for row in report["positive_controls"].values()):
        raise ValueError("完整视频负控未通过或错误控制存在漏检")


def checked_method_validation(design_path: pathlib.Path, heldout_path: pathlib.Path,
                              config_path: pathlib.Path) -> dict:
    from p0_background_preview_materials import checked_artifact
    from p0_segmentation_controls import check_independent_source

    checked_config(config_path)
    reports = []
    for path, role in [(design_path, "design"), (heldout_path, "heldout")]:
        path = path.resolve(strict=True)
        if not path.is_relative_to(ROOT / "runs/p0"):
            raise ValueError("方法验证报告越过P0目录")
        probe = checked_artifact(path, "lossless_background_controls")
        report = json.loads(path.read_text())
        validate_controls_summary(report, role)
        if report.get("method_config_sha256") != sha256(config_path) or any(
                probe.get("source_files_sha256", {}).get(name) != sha256(ROOT / name)
                for name in METHOD_DEPENDENCIES):
            raise ValueError("正负控的方法配置或代码依赖已改变，需重新设计并验证")
        source_path = checked_bound_file(pathlib.Path(report["source_input"]), report["source_input_sha256"], ROOT / "runs/p0")
        evidence, _, _, _ = checked_training_source(source_path)
        if not evidence["source_episode_id"].endswith(EXPECTED_CONFIG[role + "_episode_suffix"]):
            raise ValueError("设计与独立验证源不符合事先声明")
        reports.append((report, evidence))
    heldout_binding = reports[1][0].get("design_binding") or {}
    if heldout_binding.get("design_report") != str(design_path.resolve(strict=True)) \
            or heldout_binding.get("design_report_sha256") != sha256(design_path) \
            or heldout_binding.get("same_method_and_dependencies") is not True:
        raise ValueError("独立验证没有绑定同一完整设计结果")
    independence = check_independent_source(reports[1][1], reports[0][1])
    return {"passed": True, "design": {"path": str(design_path), "sha256": sha256(design_path)},
            "heldout": {"path": str(heldout_path), "sha256": sha256(heldout_path)},
            "dependencies_sha256": {name: sha256(ROOT / name) for name in METHOD_DEPENDENCIES},
            "source_independence": independence,
            "limitation": "两个成功训练示范的方法控制，不是独立策略评测；不把窗口帧当作独立样本"}


def checked_config(path: pathlib.Path) -> dict:
    import yaml

    config = yaml.safe_load(path.read_text())
    if config != EXPECTED_CONFIG:
        raise ValueError("无损派生验收必须匹配预声明版本，禁止同次试验更改配置")
    return config


def validate_video_metadata(metadata: dict, size: tuple[int, int] | None = (256, 256)) -> dict:
    """尺寸、帧数、双帧率及每帧时间均检查；不能先缩放输出再声称像素无损。"""
    import math

    streams = metadata.get("streams", [])
    frames = metadata.get("frames", [])
    if len(streams) != 1 or len(frames) != 93:
        raise ValueError("视频必须实际解码完整93帧，不能补帧或截断")
    stream = streams[0]
    actual_size = (stream.get("width"), stream.get("height"))
    if any(type(value) is not int or value <= 0 for value in actual_size) or (size is not None and actual_size != size):
        raise ValueError("视频实际尺寸不符合声明，禁止缩放后验收")
    try:
        if any(Fraction(stream.get(key, "0")) != 20 for key in ["r_frame_rate", "avg_frame_rate"]):
            raise ValueError("视频实际帧率必须为20fps")
        timestamps = [float(frame.get("best_effort_timestamp_time", "nan")) for frame in frames]
    except (ValueError, ZeroDivisionError, TypeError) as exc:
        raise ValueError("视频帧率或时间戳无效") from exc
    if any(not math.isfinite(value) or abs(value - index / 20) > 1e-6 for index, value in enumerate(timestamps)):
        raise ValueError("帧顺序或时间戳与20Hz标签不一致")
    return {"width": actual_size[0], "height": actual_size[1], "frames": 93, "fps": 20,
            "ordered_timestamps_verified": True, "decoder_resize_performed": False}


def read_exact_video(path: pathlib.Path, size: tuple[int, int] | None = (256, 256)):
    import imageio.v2 as imageio
    import numpy as np

    process = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
                              "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate:frame=best_effort_timestamp_time",
                              "-of", "json", str(path)], text=True, capture_output=True, timeout=120, check=True)
    metadata = validate_video_metadata(json.loads(process.stdout), size)
    with imageio.get_reader(path) as reader:
        frames = np.asarray([frame for frame in reader])
    if frames.dtype != np.uint8 or frames.shape != (93, metadata["height"], metadata["width"], 3):
        raise ValueError("实际RGB解码形状与ffprobe或预声明不符")
    return frames, metadata


def pixel_certificate(source, native, masks, output, roi_names: list[str]) -> dict:
    """只证明固定合成关系；不推断背景目标、全帧语义或策略收益。"""
    import numpy as np

    expected_shape = (93, 256, 256, 3)
    if any(np.asarray(value).dtype != np.uint8 or np.asarray(value).shape != expected_shape
           for value in [source, native, output]) or roi_names != EXPECTED_CONFIG["roi_names"] \
            or masks.shape != (93, len(roi_names), 256, 256) or not np.isin(masks, [0, 1]).all() \
            or not all(masks[:, index].any() for index in range(len(roi_names))):
        raise ValueError("像素证明要求完整uint8视频、固定任务ROI及二值仿真真值")
    rows = []
    for frame, (original, generated, rois, actual) in enumerate(zip(source, native, masks, output)):
        expected, support = protected_background(original, generated, rois, 8)
        if not support.any() or support.all():
            raise ValueError("每帧必须同时包含任务保护区和可改写背景")
        foreground = np.abs(actual.astype(np.int16) - original.astype(np.int16))
        background = np.abs(actual.astype(np.int16) - generated.astype(np.int16))
        rows.append({"frame": frame, "protected_max_abs_diff": int(foreground[support].max()),
                     "background_max_abs_diff": int(background[~support].max()),
                     "expected_composite_equal": bool(np.array_equal(actual, expected)),
                     "protected_pixels": int(support.sum()),
                     "changed_background_pixels": int(np.count_nonzero(np.any(actual != original, axis=-1) & ~support)),
                     "roi_max_abs_diff": {name: int(foreground[mask.astype(bool)].max()) if mask.any() else 0
                                           for name, mask in zip(roi_names, rois)},
                     "source_rgb_sha256": hashlib.sha256(original.tobytes()).hexdigest(),
                     "output_rgb_sha256": hashlib.sha256(actual.tobytes()).hexdigest()})
    return {"geometry_evidence_type": EXPECTED_CONFIG["geometry_evidence_type"], "frames": list(range(93)),
            "geometry_passed": all(row["expected_composite_equal"] and row["protected_max_abs_diff"] == 0
                                   and row["background_max_abs_diff"] == 0 for row in rows),
            "per_frame": rows, "sample_count": 1,
            "limitation": "只证明任务保护区与背景的来源像素关系；不证明背景语义、目标条件或策略收益"}


def checked_training_source(path: pathlib.Path):
    """绑定成功训练回合并与原缓存逐帧比较，禁止只相信复制的来源标志。"""
    import numpy as np

    path = path.resolve(strict=True)
    if not path.is_relative_to(ROOT / "runs/p0"):
        raise ValueError("成功源不在本项目P0目录")
    evidence = json.loads(path.read_text())
    probe = json.loads((path.parent / "probe.json").read_text())
    if probe.get("stage") not in {"verified_transfer_input", "edge_anchor_input", "background_transfer_input"} \
            or probe.get("status") != "passed" or probe.get("artifacts_sha256", {}).get(path.name) != sha256(path) \
            or evidence.get("source_success") is not True or evidence.get("source_kind") != "demo" \
            or evidence.get("mask_source") != "sim_seg" or not evidence.get("replay_validation", {}).get("passed") \
            or evidence.get("frames") != 93 or evidence.get("fps") != 20 \
            or evidence.get("temporal_resampling") != "none" or evidence.get("roi_names") != EXPECTED_CONFIG["roi_names"]:
        raise ValueError("来源缺少已核验的成功训练示范、仿真ROI或固定时间协议")
    cache = pathlib.Path(evidence["cache_manifest"]).resolve(strict=True)
    if not cache.is_relative_to(ROOT / "data") or sha256(cache) != evidence["cache_manifest_sha256"]:
        raise ValueError("训练缓存来源改变或越界")
    entries = [row for row in json.loads(cache.read_text())["episodes"] if row["episode_id"] == evidence["source_episode_id"]]
    if len(entries) != 1 or entries[0].get("split") != "demo_train" or entries[0].get("env_success") is not True \
            or entries[0].get("is_eval", False) is not False:
        raise ValueError("新路线禁止使用验证或冻结评测来源")
    if evidence.get("instruction") != entries[0].get("instruction") or not evidence.get("instruction", "").strip():
        raise ValueError("源语言指令与同一训练示范不一致")
    validate_source_files(evidence, path.parent)
    frozen_path = ROOT / "eval/frozen_eval_set.json"
    frozen = json.loads(frozen_path.read_text())
    frozen_content = {key: value for key, value in frozen.items() if key != "content_sha256"}
    content_sha = hashlib.sha256(json.dumps(frozen_content, ensure_ascii=False, sort_keys=True,
                                          separators=(",", ":")).encode()).hexdigest()
    if frozen_path.with_suffix(".sha256").read_text().strip() != sha256(frozen_path) \
            or (ROOT / "configs/eval_frozen.hash").read_text().splitlines()[0] != sha256(frozen_path) \
            or frozen.get("content_sha256") != content_sha or frozen.get("freeze_status") != "frozen" \
            or frozen.get("all_initial_states_replayed") is not True \
            or frozen.get("source_cache_manifest_sha256") != sha256(cache) \
            or frozen.get("eval_overlap_check", {}).get("passed") is not True \
            or any(row.get("initial_scene_sha256") in {
                entries[0].get("initial_scene_sha256"), entries[0].get("raw_initial_scene_sha256")}
                   or row.get("eval_id") == evidence["source_episode_id"] for row in frozen["episodes"]):
        raise ValueError("冻结评测未与同一训练缓存去重或来源发生重叠")
    frames, metadata = read_exact_video(path.parent / "source.mp4")
    masks = np.load(path.parent / "roi_masks.npy", allow_pickle=False)
    first, stop = (evidence["window"][key] for key in ["start_frame", "end_frame"])
    row = entries[0]
    if type(first) is not int or type(stop) is not int or first < 0 or stop - first != 93 or stop > row["length"]:
        raise ValueError("窗口越出成功回合或长度改变")
    images = np.load(cache.parent / "images.npy", mmap_mode="r", allow_pickle=False)
    actions = np.load(cache.parent / "actions.npy", mmap_mode="r", allow_pickle=False)
    states = np.load(cache.parent / "states.npy", mmap_mode="r", allow_pickle=False)
    start, end = row["offset"] + first, row["offset"] + stop
    if len(evidence["source_rgb_sha256"]) != 93 or not np.array_equal(frames, images[start:end, 0]) \
            or any(hashlib.sha256(frame.tobytes()).hexdigest() != checksum
                   for frame, checksum in zip(frames, evidence["source_rgb_sha256"])):
        raise ValueError("解码源图与逐帧来源或训练缓存不一致")
    with np.load(path.parent / "source_labels.npz", allow_pickle=False) as labels:
        expected = {"action": actions[start:end], "state": states[start:end],
                    "frame_index": np.arange(first, stop), "timestamp": np.arange(first, stop) / 20}
        validate_labels(dict(labels), expected)
    proof = {"source_input": str(path), "source_input_sha256": sha256(path),
             "source_episode_id": evidence["source_episode_id"], "cache_manifest_sha256": sha256(cache),
             "labels_sha256": evidence["labels_sha256"], "labels_match_training_cache": True,
             "instruction": evidence["instruction"], "instruction_matches_training_cache": True,
             "eval_overlap_check": {"passed": True, "frozen_eval_set_sha256": sha256(frozen_path),
                                    "scope": "沿用冻结时对完整D0原始/稳定场景的关节距离去重并核对同一缓存摘要"},
             "wrist_rgb_sha256": [hashlib.sha256(image.tobytes()).hexdigest() for image in images[start:end, 1]],
             "source_video": metadata, "training_split": "demo_train", "source_frames": 93}
    return evidence, frames, masks, proof


def validate_labels(actual: dict, expected: dict) -> None:
    import numpy as np

    if set(actual) != {"action", "state", "frame_index", "timestamp"} or set(actual) != set(expected) \
            or any(np.asarray(actual[key]).dtype != np.asarray(value).dtype
                   or not np.array_equal(actual[key], value) for key, value in expected.items()):
        raise ValueError("实际动作、状态、帧序或时间标签与成功源不一致")
