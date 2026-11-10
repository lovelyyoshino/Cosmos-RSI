#!/usr/bin/env python3
"""按固定密度将既有Cosmos纹理平铺到原UV图集，重放已见源；仅作设计诊断。

@author pony
@date 2026-11-08
@version v1.0.0
@last_modified 2026-11-08
@changelog
    - v1.0.0 (2026-11-08): UV小面片的固定镜像平铺、纯色复现及全帧物理和像素证明。
"""
from __future__ import annotations

import csv
import json
import pathlib
import shutil

from p0_common import ROOT, ensure_gpu_idle, run_stage
from p0_policy_runtime import sha256

CONFIG_PATH = ROOT / "configs/table_uv_atlas_design.yaml"
CONFIG = {
    "version": "thor-table-uv-atlas-design-v1", "purpose": "design_only_not_training",
    "design_episode_suffix": "caddy-demo_16",
    "reference_run": "table_renderer_replay_20261108T162348_856568+0800",
    "texture_sha256": "1da3f6cf62bf0447dceaf7ef694252d53b1bb452eb89840becff985dc209db97",
    "atlas_side": 1024, "repeats_per_axis": 8, "tiling": "alternating_mirror_edge_continuous",
    "resize": "opencv_inter_area", "target_condition": {"background": "matte_dark_wood"},
    "mapping": "original_mesh_uv_and_material_repeat", "frames": 93, "fps": 20,
    "eligible_for_training": False, "diagnostic_only": True,
}


def checked_config():
    import yaml
    if yaml.safe_load(CONFIG_PATH.read_text()) != CONFIG:
        raise ValueError("UV图集设计偏离运行前固定配方")
    return CONFIG


def mirrored_atlas(texture, side=1024, repeats=8):
    """密度来自桌板小UV面片；交替镜像使每条平铺边界的RGB连续。"""
    import cv2
    import numpy as np

    texture = np.asarray(texture)
    if texture.shape != (256, 256, 3) or texture.dtype != np.uint8 \
            or type(side) is not int or type(repeats) is not int \
            or side != 1024 or repeats != 8:
        raise ValueError("只接受已登记RGB纹理与固定1024图集、每轴8次平铺")
    tile = cv2.resize(texture, (side // repeats, side // repeats), interpolation=cv2.INTER_AREA)
    pair = np.concatenate([tile, tile[:, ::-1]], axis=1)
    block = np.concatenate([pair, pair[::-1]], axis=0)
    return np.tile(block, (repeats // 2, repeats // 2, 1))


def diagnose(directory):
    ensure_gpu_idle()
    checked_config()
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from p0_demo_replay import initialize_demo
    from p0_lossless_background import read_exact_video
    from p0_prepare_table_background import write_video
    from p0_sim_data import configure_libero, observation_state
    from p0_single_control_materials import static_table_changes
    from p0_table_background import pixel_certificate
    from p0_table_renderer_design import install_renderer, protected_composite
    from p0_table_renderer_validation import checked_render, texture_source

    reference_path = ROOT / "runs/p0" / CONFIG["reference_run"] / "renderer_replay.json"
    reference, evidence, source, masks, proof = checked_render(reference_path, "design")
    if not evidence["source_episode_id"].endswith(CONFIG["design_episode_suffix"]):
        raise ValueError("只允许已见设计源，禁止按独立源调节图集")
    texture_path, texture_binding = texture_source()
    if sha256(texture_path) != CONFIG["texture_sha256"]:
        raise ValueError("原Cosmos纹理已经改变")
    atlas = mirrored_atlas(np.array(Image.open(texture_path).convert("RGB")),
                           CONFIG["atlas_side"], CONFIG["repeats_per_axis"])
    Image.fromarray(atlas).save(directory / "cosmos_uv_atlas.png")
    declaration = {"method": CONFIG, "config_sha256": sha256(CONFIG_PATH),
                   "reference_report": str(reference_path), "reference_sha256": sha256(reference_path),
                   "texture_binding": texture_binding, "atlas_sha256": sha256(directory / "cosmos_uv_atlas.png"),
                   "source_input": reference["source_input"], "source_input_sha256": reference["source_input_sha256"],
                   "eligible_for_training": False}
    (directory / "declaration.json").write_text(json.dumps(declaration, ensure_ascii=False, indent=2) + "\n")
    source_path = pathlib.Path(reference["source_input"])
    cache_path = pathlib.Path(evidence["cache_manifest"])
    row = next(r for r in json.loads(cache_path.read_text())["episodes"]
               if r["episode_id"] == evidence["source_episode_id"])
    arrays = {name: np.load(cache_path.parent / (name + ".npy"), mmap_mode="r", allow_pickle=False)
              for name in ["images", "actions", "states"]}
    elements = np.load(reference_path.parent / "elements.npy", allow_pickle=False)
    tables = elements == reference["table_id"]
    benchmark = configure_libero("LIBERO", directory / "libero_config")
    from libero.libero.envs import OffScreenRenderEnv
    env = OffScreenRenderEnv(bddl_file_name=str(benchmark / "bddl_files/libero_10" / row["bddl_file"]),
                             camera_heights=256, camera_widths=256, camera_segmentations="element")
    original_reset, attachment = env.reset, {}

    def observed_reset(*args, **kwargs):
        obs = original_reset(*args, **kwargs)
        state, original_render = install_renderer(env, reference["table_id"], atlas)
        if state["plan"]["width"] != 1024 or state["plan"]["height"] != 1024:
            raise ValueError("原桌体纹理尺寸不同，禁止隐式重采样固定UV图集")
        attachment.update(state=state, original_render=original_render, sim=env.sim)
        return obs

    env.reset = observed_reset
    frames = {name: [] for name in ["analytic", "cosmos", "raw_cosmos"]}
    errors = {"physics": 0., "state": 0., "rgb": 0, "wrist": 0, "elements": 0}
    first, stop = evidence["window"]["start_frame"], evidence["window"]["end_frame"]
    try:
        with np.load(pathlib.Path(row["directory"]) / "physical_states.npz", allow_pickle=False) as physical, \
                np.load(source_path.parent / "source_reset_snapshot.npz", allow_pickle=False) as snapshot:
            obs, _, _ = initialize_demo(env, row, physical, snapshot)
            if env.sim is not attachment["sim"]:
                raise ValueError("hard_reset替换了已观察sim")
            for index in range(stop):
                flat = row["offset"] + index
                errors["physics"] = max(errors["physics"], float(np.abs(env.get_sim_state() - physical["states"][index]).max()))
                errors["state"] = max(errors["state"], float(np.abs(observation_state(obs) - arrays["states"][flat]).max()))
                wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                errors["wrist"] = max(errors["wrist"], int(np.abs(wrist.astype(np.int16) - arrays["images"][flat, 1].astype(np.int16)).max()))
                if index >= first:
                    at = index - first
                    rgb = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    last = attachment["state"]["last"]
                    if last is None or not np.array_equal(last["source"][::-1, ::-1], rgb):
                        raise ValueError("UV图集渲染不来自同一次传感器观测")
                    errors["rgb"] = max(errors["rgb"], int(np.abs(rgb.astype(np.int16) - source[at].astype(np.int16)).max()))
                    seg = np.ascontiguousarray(obs["agentview_segmentation_element"][::-1, ::-1, 0])
                    errors["elements"] += int(np.count_nonzero(seg != elements[at]))
                    for name, kind in [("analytic", "analytic"), ("cosmos", "cosmos")]:
                        rendered = np.ascontiguousarray(last[kind][::-1, ::-1])
                        frames[name].append(protected_composite(rgb, rendered, tables[at], masks[at]))
                        if name == "cosmos":
                            frames["raw_cosmos"].append(rendered)
                obs, _, _, _ = env.step(np.array(arrays["actions"][flat]))
    finally:
        env.reset = original_reset
        if attachment:
            attachment["sim"].render = attachment["original_render"]
        env.close()
    if max(errors.values()) or not attachment["state"]["texture_has_effect"]:
        raise ValueError(f"源同帧、完整物理或纹理实效未通过：{errors}")
    decoded = {}
    for name, values in frames.items():
        write_video(directory / (name + ".mp4"), values)
        decoded[name], _ = read_exact_video(directory / (name + ".mp4"))
        if not np.array_equal(values, decoded[name]):
            raise ValueError("实际无损视频改变了同帧渲染")
    previous, _ = read_exact_video(reference_path.parent / "cosmos.mp4")
    analytic, _ = read_exact_video(reference_path.parent / "analytic.mp4")
    if not np.array_equal(analytic, decoded["analytic"]):
        raise ValueError("只改变图集密度后，纯色对照没有复现")
    certificate = pixel_certificate(source, decoded["raw_cosmos"], tables, decoded["cosmos"], masks)
    if not certificate["geometry_passed"]:
        raise ValueError("全部非桌体像素与同帧渲染配方未通过")
    shutil.copyfile(source_path.parent / "source_labels.npz", directory / "source_labels.npz")
    if sha256(directory / "source_labels.npz") != evidence["labels_sha256"]:
        raise ValueError("动作、状态或时间标签字节改变")
    videos = [("source", source), ("analytic", analytic), ("previous", previous), ("atlas", decoded["cosmos"])]
    temporal = [{"kind": name, **value} for name, video in videos
                for value in static_table_changes(video, tables.all(axis=0))]
    for filename, values in [("pixel_certificate.csv", certificate["per_frame"]), ("temporal.csv", temporal)]:
        with (directory / filename).open("x", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader(); writer.writerows(values)
    font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 17, index=2)
    labels = ["成功源", "解析纯色", "整图直接上传旧方法", "固定UV图集平铺"]
    for page in range(8):
        indices = list(range(page * 12, min(93, (page + 1) * 12)))
        sheet = Image.new("RGB", (1024, 32 + len(indices) * 276), "#f6f4ee")
        draw = ImageDraw.Draw(sheet)
        draw.text((4, 4), " / ".join(labels) + "；设计诊断不可入训", font=font, fill="#23343e")
        for slot, index in enumerate(indices):
            y = 32 + slot * 276
            draw.text((4, y), f"实际帧{index}", font=font, fill="#23343e")
            for col, (_, video) in enumerate(videos):
                sheet.paste(Image.fromarray(video[index]), (col * 256, y + 20))
        sheet.save(directory / f"review_{page:02d}.png")
    for index in [0, 46, 92]:
        sheet = Image.new("RGB", (1024, 328), "#f6f4ee"); draw = ImageDraw.Draw(sheet)
        for col, ((_, video), label) in enumerate(zip(videos, labels)):
            draw.text((col * 256 + 4, 4), label, font=font, fill="#23343e")
            sheet.paste(Image.fromarray(video[index]), (col * 256, 32))
        draw.text((4, 296), "Thor / 既有MAXN / demo_16 / 同帧重渲染 / 无模型、无训练 / 设计不可入训", font=font, fill="#7a4237")
        sheet.save(directory / f"actual_{index:03d}.png")
    report = {"method": CONFIG, "declaration_sha256": sha256(directory / "declaration.json"),
              "source_episode_id": evidence["source_episode_id"], "source_proof": proof,
              "source_replay_errors": errors, "render_state": {key: value for key, value in attachment["state"].items() if key != "last"},
              "pixel_certificate": certificate, "common_table_pixels": int(tables.all(axis=0).sum()),
              "adjacent_mean_rgb_change_p95": {name: float(np.percentile([r["mean_abs_rgb_change_0_1"] for r in temporal if r["kind"] == name], 95))
                                               for name, _ in videos},
              "diagnostic_only": True, "eligible_for_training": False, "independent_sources_read": 0,
              "output_kind": "background_composite", "derived_kind": "cosmos_mirrored_uv_atlas_sim_rerender_design"}
    (directory / "atlas_design.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return {key: report[key] for key in ["source_replay_errors", "adjacent_mean_rgb_change_p95", "eligible_for_training"]}


if __name__ == "__main__":
    raise SystemExit(run_stage("table_uv_atlas_design", diagnose,
        {"method": CONFIG["version"], "compute": "Thor CPU图集派生、物理及EGL重渲染；无模型或训练",
         "precision": "RGB uint8/物理fp64/监督状态fp32", "batch": 1, "seed": 0,
         "source_windows": 1, "frames": 93, "fps": 20, "width": 256, "height": 256,
         "training_steps": 0, "diagnostic_only": True}))
