#!/usr/bin/env python3
"""诊断固定Cosmos桌面纹理及边界渐变；显式派生始终禁止直接入训。

@author pony
@date 2026-11-08
@version v1.0.0
@last_modified 2026-11-08
@changelog
    - v1.0.0 (2026-11-08): 预声明静态纹理提取、完整保护像素和实际视频复算。
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib

from p0_common import ROOT, ensure_gpu_idle, run_stage
from p0_policy_runtime import sha256

CONFIG_PATH = ROOT / "configs/table_texture_design.yaml"
CONFIG = {
    "version": "thor-table-texture-atlas-design-v1", "purpose": "design_only_not_training",
    "design_episode_suffix": "caddy-demo_16", "future_method_check_episode_suffix": "caddy-demo_23",
    "target_condition": {"background": "matte_dark_wood"}, "frames": 93, "fps": 20,
    "width": 256, "height": 256, "native_texture_frame": 0,
    "texture_crop": "largest_rectangle_in_all_frame_editable_intersection",
    "protected_radius_pixels": 8, "image_border": "outside_image_is_not_an_obstacle",
    "feather_inside_editable_pixels": 8, "minimum_crop_width_height": 8,
    "minimum_editable_pixels_per_frame": 1024, "texture_highpass_sigma_pixels": 4.0,
    "texture_rms_floor": 1.0, "texture_contrast": .08, "texture_standardized_clip": 2.0,
    "matte_albedo_rgb": [88., 50., 30.],
    "source_shading_reference": "median_first_frame_visible_table_luma",
    "source_shading_clip": [.5, 1.5], "eligible_for_training": False,
}


def checked_config():
    import yaml

    if yaml.safe_load(CONFIG_PATH.read_text()) != CONFIG:
        raise ValueError("固定纹理诊断偏离预声明配置；须单独版本及验证")
    return CONFIG


def largest_rectangle(mask):
    """坐标按半开区间(y0,x0,y1,x1)返回，同面积时固定选字典序最小矩形。"""
    import numpy as np

    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.dtype != np.bool_ or not mask.any():
        raise ValueError("纹理矩形需要非空二维bool区域")
    heights = np.zeros(mask.shape[1], dtype=np.int32)
    best, area = None, 0
    for y, row in enumerate(mask):
        heights = np.where(row, heights + 1, 0)
        stack = []
        for x in range(len(row) + 1):
            current = int(heights[x]) if x < len(row) else 0
            left = x
            while stack and stack[-1][1] > current:
                start, height = stack.pop()
                candidate = (y + 1 - height, start, y + 1, x)
                size = height * (x - start)
                if size > area or (size == area and (best is None or candidate < best)):
                    best, area = candidate, size
                left = start
            if current and (not stack or stack[-1][1] < current):
                stack.append((left, current))
    return best


def editable_region(raw_table):
    import cv2
    import numpy as np

    raw_table = np.asarray(raw_table)
    if raw_table.ndim != 3 or raw_table.dtype != np.bool_ or any(not f.any() or f.all() for f in raw_table):
        raise ValueError("每帧桌体分割必须是非平凡bool区域")
    radius = CONFIG["protected_radius_pixels"]
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
    # 画面外无实际遮挡物；边界延拓只避免虚构相框，所有画面内非桌体仍完整保护。
    editable = np.asarray([cv2.erode(f.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT,
                                    borderValue=1).astype(bool) for f in raw_table])
    if any(not frame.any() for frame in editable) or np.any(editable & ~raw_table):
        raise ValueError("八像素保护后桌体为空或超出同帧桌体真值")
    alpha = []
    for frame in editable:
        extended = np.pad(frame.astype(np.uint8), 1, constant_values=1)
        distance = cv2.distanceTransform(extended, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
        ratio = np.clip(distance / CONFIG["feather_inside_editable_pixels"], 0, 1)
        alpha.append(ratio * ratio * (3 - 2 * ratio))
    return editable, np.asarray(alpha, dtype=np.float32)


def static_texture(native_first, common_table):
    import cv2
    import numpy as np

    native_first, common_table = np.asarray(native_first), np.asarray(common_table)
    if native_first.ndim != 3 or native_first.shape[-1] != 3 or native_first.dtype != np.uint8 \
            or common_table.dtype != np.bool_ or common_table.shape != native_first.shape[:2]:
        raise ValueError("纹理必须取真实首帧uint8图和对齐的共同桌体")
    crop = largest_rectangle(common_table)
    y0, x0, y1, x1 = crop
    if min(y1 - y0, x1 - x0) < CONFIG["minimum_crop_width_height"]:
        raise ValueError("自动安全矩形不足；禁止人工改选或缩小保护区")
    pixels = native_first[y0:y1, x0:x1]
    luminance = pixels.astype(np.float32) @ np.array([.2126, .7152, .0722], dtype=np.float32)
    low = cv2.GaussianBlur(luminance, (0, 0), CONFIG["texture_highpass_sigma_pixels"],
                           borderType=cv2.BORDER_REFLECT_101)
    high = luminance - low
    # 消除卷积舍入的直流偏置，纯色输入应得到严格中性的纹理系数。
    high -= high.mean(dtype=np.float64)
    rms = float(np.sqrt(np.mean(high * high)))
    standardized = np.clip(high / max(rms, CONFIG["texture_rms_floor"]),
                           -CONFIG["texture_standardized_clip"], CONFIG["texture_standardized_clip"])
    factor = 1 + CONFIG["texture_contrast"] * cv2.resize(
        standardized, (native_first.shape[1], native_first.shape[0]), interpolation=cv2.INTER_LINEAR)
    return factor, {"crop_yxyx": list(crop), "native_texture_frame": 0,
                    "highpass_rms": rms, "factor_min": float(factor.min()), "factor_max": float(factor.max())}


def compose_texture(source, native_first, raw_table):
    import numpy as np

    source, raw_table = np.asarray(source), np.asarray(raw_table)
    if source.ndim != 4 or source.shape[-1] != 3 or source.dtype != np.uint8 \
            or raw_table.shape != source.shape[:-1] or source.shape[0] < 2:
        raise ValueError("固定纹理需要完整uint8源及同帧桌体分割")
    editable, alpha = editable_region(raw_table)
    factor, extraction = static_texture(native_first, editable.all(axis=0))
    luminance = source.astype(np.float32) @ np.array([.2126, .7152, .0722], dtype=np.float32)
    reference = float(np.median(luminance[0][raw_table[0]]))
    if not np.isfinite(reference) or reference <= 1:
        raise ValueError("首帧源桌体亮度不足以构造有效光照参考")
    shading = np.clip(luminance / reference, *CONFIG["source_shading_clip"])
    matte = shading[..., None] * factor[None, ..., None] * np.asarray(CONFIG["matte_albedo_rgb"], np.float32)
    result = np.rint(source * (1 - alpha[..., None]) + matte * alpha[..., None]).clip(0, 255).astype(np.uint8)
    if not np.array_equal(result[~editable], source[~editable]):
        raise ValueError("固定纹理派生改变了完整非桌体保护像素")
    return result, editable, alpha, factor, {**extraction, "source_luma_reference": reference}


def pixel_report(source, expected, actual, editable, masks):
    import numpy as np

    if actual.shape != expected.shape or actual.dtype != np.uint8 or masks.shape != (len(source), 3, *source.shape[1:3]) \
            or not np.isin(masks, [0, 1]).all() or np.any(editable & masks.any(axis=1)):
        raise ValueError("实际视频或三任务ROI不完整，或改写区进入任务前景")
    rows = []
    for index in range(len(source)):
        difference = np.abs(actual[index].astype(np.int16) - source[index].astype(np.int16))
        rows.append({"frame": index, "expected_recipe_max_abs_diff": int(np.abs(
            actual[index].astype(np.int16) - expected[index].astype(np.int16)).max()),
            "protected_max_abs_diff": int(difference[~editable[index]].max()),
            "editable_pixels": int(editable[index].sum()),
            "changed_table_pixels": int((np.any(actual[index] != source[index], axis=-1) & editable[index]).sum()),
            **{name + "_max_abs_diff": int(difference[mask.astype(bool)].max()) if mask.any() else 0
               for name, mask in zip(["robot", "book", "caddy"], masks[index])}})
    return {"formula_and_protection_passed": all(row["expected_recipe_max_abs_diff"] == 0 and
            row["protected_max_abs_diff"] == 0 for row in rows), "per_frame": rows,
            "training_eligibility_proved": False}


def diagnose(directory, args):
    ensure_gpu_idle()
    checked_config()
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from p0_lossless_background import read_exact_video
    from p0_prepare_table_background import write_video
    from p0_single_control_materials import static_table_changes
    from p0_table_background import checked_table_source, select_table_geometries
    from p0_table_reference_materials import native_video
    from p0_table_reference_sample import checked_input

    generation, native_path = native_video(args.generation_probe, "table_reference_generation")
    result = generation["result"]
    reference_path = pathlib.Path(result["reference_input"])
    if sha256(reference_path) != result["reference_input_sha256"]:
        raise ValueError("原生生成没有保持实际材质参考来源")
    reference, _ = checked_input(reference_path)
    source_path = pathlib.Path(reference["source_input"])
    evidence, source, masks, _, elements, _ = checked_table_source(source_path)
    if not evidence["source_episode_id"].endswith(CONFIG["design_episode_suffix"]):
        raise ValueError("本设计只允许已见训练源demo_16，不读未来独立源")
    if result.get("diagnostic_only") is not True or result.get("eligible_for_training") is not False \
            or result.get("guardrails_enabled") is not True:
        raise ValueError("只接受来源完整且安全检查开启的原生设计窗口")
    native, native_metadata = read_exact_video(native_path, size=None)
    if source.shape != (93, 256, 256, 3) or min(native.shape[1:3]) < 256:
        raise ValueError("需要93帧源及无需上采样的实际原生视频")
    resized = np.asarray([cv2.resize(f, (256, 256), interpolation=cv2.INTER_AREA) for f in native])
    table_ids = select_table_geometries(evidence["table_geometry"]["geometries"])
    raw_table = np.isin(elements, table_ids)
    output, editable, alpha, factor, extraction = compose_texture(source, resized[0], raw_table)
    if min(int(f.sum()) for f in editable) < CONFIG["minimum_editable_pixels_per_frame"]:
        raise ValueError("新方法的每帧可见桌体不足，禁止缩小保护范围")
    video = directory / "static_texture_design_lossless.mp4"
    write_video(video, output)
    actual, actual_metadata = read_exact_video(video)
    certificate = pixel_report(source, output, actual, editable, masks)
    if not certificate["formula_and_protection_passed"]:
        raise ValueError("实际编码视频没有完整复现显式配方与保护像素")
    np.save(directory / "editable_table.npy", editable, allow_pickle=False)
    np.save(directory / "texture_factor.npy", factor, allow_pickle=False)
    np.save(directory / "alpha.npy", alpha, allow_pickle=False)
    Image.fromarray(np.rint(factor[..., None] * np.asarray(CONFIG["matte_albedo_rgb"])).clip(0, 255).astype(np.uint8)).save(
        directory / "normalized_cosmos_texture.png")
    y0, x0, y1, x1 = extraction["crop_yxyx"]
    Image.fromarray(resized[0, y0:y1, x0:x1]).save(directory / "native_texture_crop.png")
    common = editable.all(axis=0)
    rows, temporal = certificate["per_frame"], []
    for kind, frames in [("source", source), ("native", resized), ("static_derived", actual)]:
        temporal += [{"kind": kind, **row} for row in static_table_changes(frames, common)]
    for name, values in [("pixel_certificate.csv", rows), ("static_table_temporal.csv", temporal)]:
        with (directory / name).open("x", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0])); writer.writeheader(); writer.writerows(values)
    font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 17, index=2)
    texture = np.rint(factor[..., None] * np.asarray(CONFIG["matte_albedo_rgb"])).clip(0, 255).astype(np.uint8)
    columns = [source, resized, np.repeat(texture[None], 93, axis=0), actual]
    for frame in [0, 46, 92]:
        sheet = Image.new("RGB", (1024, 324), "#f6f4ee"); draw = ImageDraw.Draw(sheet)
        for column, (label, frames) in enumerate(zip(["成功源", "Cosmos原生", "显式归一化纹理", "固定纹理派生诊断"], columns)):
            draw.text((column * 256 + 5, 5), label, font=font, fill="#23343e")
            sheet.paste(Image.fromarray(frames[frame]), (column * 256, 32))
        draw.text((6, 296), f"帧{frame}；归一化Cosmos纹理＋源光照；无独立方法验收，禁止入训；无RSI收益。", font=font, fill="#7a4237")
        sheet.save(directory / f"static_actual_{frame:03d}.png")
    for page in range(8):
        indices = list(range(page * 12, min(93, (page + 1) * 12)))
        sheet = Image.new("RGB", (1024, 32 + 276 * len(indices)), "#f6f4ee"); draw = ImageDraw.Draw(sheet)
        draw.text((5, 5), "实际全部帧：成功源 / 原生 / 解析归一化纹理 / 设计派生；不可入训", font=font, fill="#23343e")
        for row, frame in enumerate(indices):
            y = 32 + row * 276
            draw.text((5, y), f"帧{frame}", font=font, fill="#23343e")
            for column, frames in enumerate(columns):
                sheet.paste(Image.fromarray(frames[frame]), (column * 256, y + 20))
        sheet.save(directory / f"review_all_frames_{page:02d}.png")
    fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
    p95 = {}
    for kind in ["source", "native", "static_derived"]:
        values = [r["mean_abs_rgb_change_0_1"] for r in temporal if r["kind"] == kind]
        p95[kind] = float(np.percentile(values, 95))
        ax.plot(range(1, 93), values, label=kind)
    ax.set(xlabel="Actual frame / 20 fps", ylabel="Adjacent RGB change / 255",
           title="Same design window; normalized texture derivation; no policy evaluation")
    ax.legend(); ax.grid(alpha=.2)
    for extension in ["png", "svg"]:
        fig.savefig(directory / ("static_texture_temporal." + extension), dpi=180)
    plt.close(fig)
    report = {"method": CONFIG["version"], "config_sha256": sha256(CONFIG_PATH),
              "generation_probe": str(args.generation_probe), "generation_probe_sha256": sha256(args.generation_probe),
              "source_input": str(source_path), "source_input_sha256": sha256(source_path),
              "source_episode_id": evidence["source_episode_id"], "source_labels_sha256": evidence["labels_sha256"],
              "native_video": str(native_path), "native_video_sha256": sha256(native_path),
              "native_metadata": native_metadata, "derived_video": str(video), "derived_video_sha256": sha256(video),
              "derived_metadata": actual_metadata, "extraction": extraction,
              "pixel_certificate": certificate, "common_visible_table_pixels": int(common.sum()),
              "adjacent_mean_rgb_change_p95": p95, "diagnostic_only": True, "eligible_for_training": False,
              "policy_improvement_claimed": False, "source_windows": 1,
              "derivation": "固定首帧Cosmos高频纹理＋解析深棕基色＋源光照＋桌体内部渐变",
              "limitations": ["无新独立成功训练源与实际错误控制", "无完整身份/目标/时序/边界复核",
                              "固定相机屏幕坐标设计，不是移动相机/移动桌体方法", "没有训练或策略评测"]}
    (directory / "texture_design.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return {key: report[key] for key in ["method", "extraction", "adjacent_mean_rgb_change_p95",
                                        "eligible_for_training", "policy_improvement_claimed"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--generation-probe", type=pathlib.Path, required=True)
    arguments = parser.parse_args()
    arguments.generation_probe = arguments.generation_probe.resolve(strict=True)
    if not arguments.generation_probe.is_relative_to(ROOT / "runs/p0"):
        parser.error("只接收Thor本项目P0阶段的实际生成证据")
    raise SystemExit(run_stage("table_texture_design", lambda directory: diagnose(directory, arguments),
        {"method": CONFIG["version"], "compute": "Thor CPU uint8/float32显式派生；不运行模型",
         "source_windows": 1, "frames": 93, "fps": 20, "precision": "uint8解码/float32计算/CRF0编码",
         "batch": None, "seed": None, "training_steps": 0, "diagnostic_only": True}))
