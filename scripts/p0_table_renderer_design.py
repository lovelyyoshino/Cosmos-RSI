#!/usr/bin/env python3
"""在成功源同一观测时刻替换桌体视觉纹理，检验遮挡与边界；不授予训练资格。

@author pony
@date 2026-11-08
@version v1.0.1
@last_modified 2026-11-08
@changelog
    - v1.0.1 (2026-11-08): 在原hard_reset完成后绑定新sim，保持原成功源的重置流程。
    - v1.0.0 (2026-11-08): 同传感器重渲染、视觉状态恢复、纯色对照及完整窗口像素验证。
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import shutil

from p0_common import ROOT, ensure_gpu_idle, run_stage
from p0_policy_runtime import sha256

CONFIG_PATH = ROOT / "configs/table_renderer_design.yaml"
CONFIG = {
    "version": "thor-table-renderer-design-v1", "purpose": "design_only_not_training",
    "design_episode_suffix": "caddy-demo_16", "future_method_check_episode_suffix": "caddy-demo_23",
    "texture_design_run": "table_texture_design_20261108T132043_578426+0800",
    "texture_file": "normalized_cosmos_texture.png", "target_condition": {"background": "matte_dark_wood"},
    "frames": 93, "fps": 20, "width": 256, "height": 256, "table_body": "study_table",
    "table_mesh": "desk_vis", "table_visual_group": 1, "camera": "agentview",
    "texture_resize": "opencv_inter_linear", "analytic_albedo_rgb": [88, 50, 30],
    "matte_reflectance": 0., "matte_shininess": 0., "matte_specular": 0.,
    "composite_scope": "exact_same_observation_visible_table",
    "protected_pixels": "all_non_table_source_rgb", "eligible_for_training": False,
}


def checked_config():
    import yaml

    if yaml.safe_load(CONFIG_PATH.read_text()) != CONFIG:
        raise ValueError("桌体重渲染必须使用运行前固定的新方法；不能复用旧合成资格")
    return CONFIG


def protected_composite(source, rendered, table, task_masks):
    import numpy as np

    source, rendered, table, task_masks = map(np.asarray, [source, rendered, table, task_masks])
    if source.ndim != 3 or source.shape[-1] != 3 or source.dtype != np.uint8 \
            or rendered.shape != source.shape or rendered.dtype != np.uint8 \
            or table.shape != source.shape[:2] or table.dtype != np.bool_ \
            or not table.any() or table.all() or task_masks.shape != (3, *table.shape) \
            or not np.isin(task_masks, [0, 1]).all() or np.any(table & task_masks.any(axis=0)):
        raise ValueError("同帧桌体必须完整且不包含任何任务前景；禁止用空掩码取得保护通过")
    return np.where(table[..., None], rendered, source)


def texture_plan(model, table_id):
    """只允许独占材质和2D纹理，避免换桌体纹理时影响其他场景物体。"""
    import numpy as np

    material = int(model.geom_matid[table_id])
    if material < 0 or np.flatnonzero(model.geom_matid == material).tolist() != [table_id]:
        raise ValueError("桌体没有独占材质，禁止修改共享视觉资源")
    # MuJoCo当前版本以十个材质纹理角色保存索引；旧版则是单个索引。
    entry = np.asarray(model.mat_texid[material])
    textures = np.unique(entry[entry >= 0])
    if len(textures) != 1:
        raise ValueError("桌体必须只有一个明确的纹理来源")
    texture = int(textures[0])
    if int(model.tex_type[texture]) != 0:
        raise ValueError("当前设计只接受桌体已有2D UV纹理，不改网格或立方体映射")
    users = np.flatnonzero(np.any(model.mat_texid == texture, axis=1)) if model.mat_texid.ndim == 2 \
        else np.flatnonzero(model.mat_texid == texture)
    if users.tolist() != [material]:
        raise ValueError("桌体纹理被其他材质引用，禁止修改共享纹理")
    width, height = int(model.tex_width[texture]), int(model.tex_height[texture])
    channels = int(model.tex_nchannel[texture]) if hasattr(model, "tex_nchannel") else 3
    if min(width, height) < 8 or channels != 3:
        raise ValueError("当前设计需要实际RGB纹理，不自动转换通道或映射")
    address = int(model.tex_adr[texture])
    return {"geom_id": table_id, "material_id": material, "texture_id": texture,
            "width": width, "height": height, "channels": channels, "address": address,
            "bytes": width * height * channels}


def install_renderer(env, table_id, texture):
    """在RGB传感器读图处重渲染；不按env.step后的较晚物理时刻重画。"""
    import cv2
    import mujoco
    import numpy as np
    from p0_sim_data import capture_integration_state

    sim, original_render = env.sim, env.sim.render
    model = sim.model._model
    plan = texture_plan(model, table_id)
    material, texture_id = plan["material_id"], plan["texture_id"]
    # 只改当前环境内存，所有上游XML、纹理文件及物理参数保持原字节。
    pixels = model.tex_data if hasattr(model, "tex_data") else model.tex_rgb
    storage = pixels.reshape(-1)[plan["address"]:plan["address"] + plan["bytes"]]
    old_pixels = storage.copy()
    names = ["reflectance", "shininess", "specular"]
    old_material = {name: float(getattr(model, "mat_" + name)[material]) for name in names}
    mapped = cv2.resize(texture, (plan["width"], plan["height"]), interpolation=cv2.INTER_LINEAR).reshape(-1)
    analytic = np.broadcast_to(np.asarray(CONFIG["analytic_albedo_rgb"], np.uint8),
                              (plan["height"], plan["width"], 3)).reshape(-1)
    state = {"last": None, "calls": 0, "restore_max_abs_diff": 0, "integration_equal": True,
             "texture_has_effect": False, "plan": plan}

    def upload():
        context = sim._render_context_offscreen
        context.gl_ctx.make_current()
        mujoco.mjr_uploadTexture(model, context.con, texture_id)

    def restore():
        storage[:] = old_pixels
        for name, value in old_material.items():
            getattr(model, "mat_" + name)[material] = value
        upload()

    def render(*args, **kwargs):
        original = original_render(*args, **kwargs)
        if kwargs.get("camera_name") != CONFIG["camera"] or kwargs.get("segmentation") or kwargs.get("depth"):
            return original
        before = capture_integration_state(env)["integration_state"]
        variants = {}
        try:
            for name, values in [("analytic", analytic), ("cosmos", mapped)]:
                storage[:] = values
                for field in names:
                    getattr(model, "mat_" + field)[material] = CONFIG["matte_" + field]
                upload()
                variants[name] = original_render(*args, **kwargs)
        finally:
            restore()
        restored = original_render(*args, **kwargs)
        difference = int(np.abs(original.astype(np.int16) - restored.astype(np.int16)).max())
        state["restore_max_abs_diff"] = max(state["restore_max_abs_diff"], difference)
        state["integration_equal"] &= np.array_equal(before, capture_integration_state(env)["integration_state"])
        if difference or not state["integration_equal"]:
            raise ValueError("视觉材质恢复没有保持原始RGB或完整MuJoCo积分状态")
        state["texture_has_effect"] |= bool(np.any(variants["cosmos"] != variants["analytic"]))
        state["calls"] += 1
        state["last"] = {"source": original.copy(), **variants}
        return original

    sim.render = render
    return state, original_render


def diagnose(directory, args):
    ensure_gpu_idle()
    checked_config()
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from p0_background_preview_materials import checked_artifact
    from p0_demo_replay import initialize_demo
    from p0_lossless_background import read_exact_video
    from p0_prepare_table_background import write_video
    from p0_sim_data import configure_libero, observation_state
    from p0_single_control_materials import static_table_changes
    from p0_table_background import checked_table_source, select_table_geometries

    texture_dir = ROOT / "runs/p0" / CONFIG["texture_design_run"]
    texture_report_path = texture_dir / "texture_design.json"
    checked_artifact(texture_report_path, "table_texture_design")
    texture_report = json.loads(texture_report_path.read_text())
    texture_path = texture_dir / CONFIG["texture_file"]
    checked_artifact(texture_path, "table_texture_design")
    texture = np.asarray(Image.open(texture_path).convert("RGB"))
    source_path = pathlib.Path(texture_report["source_input"])
    if sha256(source_path) != texture_report["source_input_sha256"]:
        raise ValueError("设计纹理的成功训练源已经改变")
    evidence, source, masks, _, elements, proof = checked_table_source(source_path)
    if not evidence["source_episode_id"].endswith(CONFIG["design_episode_suffix"]) \
            or texture_report.get("eligible_for_training") is not False \
            or texture_report.get("diagnostic_only") is not True:
        raise ValueError("本试验只读已见设计源，不读取未来独立源或继承旧资格")
    table_ids = select_table_geometries(evidence["table_geometry"]["geometries"])
    table_masks = np.isin(elements, table_ids)
    cache_path = pathlib.Path(evidence["cache_manifest"])
    cache = json.loads(cache_path.read_text())
    row = next(r for r in cache["episodes"] if r["episode_id"] == evidence["source_episode_id"])
    images = np.load(cache_path.parent / "images.npy", mmap_mode="r", allow_pickle=False)
    actions = np.load(cache_path.parent / "actions.npy", mmap_mode="r", allow_pickle=False)
    states = np.load(cache_path.parent / "states.npy", mmap_mode="r", allow_pickle=False)
    benchmark = configure_libero("LIBERO", directory / "libero_config")
    from libero.libero.envs import OffScreenRenderEnv

    env = OffScreenRenderEnv(bddl_file_name=str(benchmark / "bddl_files/libero_10" / row["bddl_file"]),
                             camera_heights=256, camera_widths=256, camera_segmentations="element")
    attachment = {}
    original_reset = env.reset

    def observed_reset(*reset_args, **reset_kwargs):
        observation = original_reset(*reset_args, **reset_kwargs)
        state, original_render = install_renderer(env, table_ids[0], texture)
        attachment.update({"state": state, "original_render": original_render, "sim": env.sim})
        return observation

    # 原始成功源使用hard_reset；等其完成后观察新sim，不改变复位语义来迁就观察器。
    env.reset = observed_reset
    videos = {name: [] for name in ["analytic", "cosmos", "cosmos_raw_render"]}
    errors = {"physics": 0., "state": 0., "rgb": 0, "wrist": 0, "elements": 0}
    first, stop = evidence["window"]["start_frame"], evidence["window"]["end_frame"]
    try:
        with np.load(pathlib.Path(row["directory"]) / "physical_states.npz", allow_pickle=False) as physical, \
             np.load(source_path.parent / "source_reset_snapshot.npz", allow_pickle=False) as snapshot:
            obs, _, _ = initialize_demo(env, row, physical, snapshot)
            state = attachment["state"]
            if env.sim is not attachment["sim"]:
                raise ValueError("重置替换了已观察的sim，禁止使用不同时间的补渲染")
            for index in range(stop):
                flat = row["offset"] + index
                errors["physics"] = max(errors["physics"], float(np.abs(env.get_sim_state() - physical["states"][index]).max()))
                errors["state"] = max(errors["state"], float(np.abs(observation_state(obs) - states[flat]).max()))
                wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                errors["wrist"] = max(errors["wrist"], int(np.abs(wrist.astype(np.int16) - images[flat, 1].astype(np.int16)).max()))
                if index >= first:
                    at = index - first
                    rgb = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    last = state["last"]
                    if last is None or not np.array_equal(last["source"][::-1, ::-1], rgb):
                        raise ValueError("重渲染没有来自源RGB传感器的同一次观测")
                    errors["rgb"] = max(errors["rgb"], int(np.abs(rgb.astype(np.int16) - source[at].astype(np.int16)).max()))
                    segmentation = np.ascontiguousarray(obs["agentview_segmentation_element"][::-1, ::-1, 0])
                    errors["elements"] += int(np.count_nonzero(segmentation != elements[at]))
                    for name in ["analytic", "cosmos"]:
                        pixels = np.ascontiguousarray(last[name][::-1, ::-1])
                        videos[name].append(protected_composite(rgb, pixels, table_masks[at], masks[at]))
                        if name == "cosmos":
                            videos["cosmos_raw_render"].append(pixels)
                obs, _, _, _ = env.step(np.array(actions[flat]))
    finally:
        env.reset = original_reset
        if attachment:
            attachment["sim"].render = attachment["original_render"]
        env.close()
    if max(errors.values()) != 0 or not state["texture_has_effect"]:
        raise ValueError(f"同帧源、完整物理、标签或Cosmos纹理实效未通过：{errors}")
    decoded = {}
    for name, frames in videos.items():
        frames = np.asarray(frames)
        path = directory / (name + "_lossless.mp4")
        write_video(path, frames)
        decoded[name], _ = read_exact_video(path)
        if not np.array_equal(decoded[name], frames):
            raise ValueError("实际无损视频没有逐值保持同帧渲染结果")
    actual = decoded["cosmos"]
    rows = []
    for index in range(93):
        protected = ~table_masks[index]
        diff = np.abs(actual[index].astype(np.int16) - source[index].astype(np.int16))
        rows.append({"frame": index, "protected_max_abs_diff": int(diff[protected].max()),
                     "render_recipe_max_abs_diff": int(np.abs(actual[index].astype(np.int16) - protected_composite(
                         source[index], decoded["cosmos_raw_render"][index], table_masks[index], masks[index]).astype(np.int16)).max()),
                     "table_pixels": int(table_masks[index].sum()),
                     "cosmos_vs_analytic_mean_abs_diff": float(np.abs(actual[index].astype(np.int16) - decoded["analytic"][index].astype(np.int16))[table_masks[index]].mean())})
    if any(r["protected_max_abs_diff"] or r["render_recipe_max_abs_diff"] for r in rows):
        raise ValueError("实际视频没有保持所有非桌体像素及同帧渲染配方")
    shutil.copyfile(source_path.parent / "source_labels.npz", directory / "source_labels.npz")
    if sha256(directory / "source_labels.npz") != evidence["labels_sha256"]:
        raise ValueError("动作、状态或时间标签没有保持源字节")
    temporal, common = [], table_masks.all(axis=0)
    for name, frames in [("source", source), ("analytic", decoded["analytic"]), ("cosmos", actual)]:
        temporal.extend({"kind": name, **r} for r in static_table_changes(frames, common))
    for filename, values in [("pixel_certificate.csv", rows), ("static_table_temporal.csv", temporal)]:
        with (directory / filename).open("x", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0])); writer.writeheader(); writer.writerows(values)
    font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 17, index=2)
    old_design, _ = read_exact_video(pathlib.Path(texture_report["derived_video"]))
    columns = [source, old_design, decoded["analytic"], actual]
    for page in range(8):
        indices = list(range(page * 12, min(93, (page + 1) * 12)))
        sheet = Image.new("RGB", (1024, 32 + len(indices) * 276), "#f6f4ee"); draw = ImageDraw.Draw(sheet)
        draw.text((5, 5), "成功源 / 屏幕固定纹理旧失败 / 解析纯色重渲染 / Cosmos纹理重渲染；设计诊断不可入训", font=font, fill="#23343e")
        for row_index, frame in enumerate(indices):
            y = 32 + row_index * 276
            draw.text((5, y), f"实际帧{frame}", font=font, fill="#23343e")
            for column, frames in enumerate(columns):
                sheet.paste(Image.fromarray(frames[frame]), (column * 256, y + 20))
        sheet.save(directory / f"review_all_frames_{page:02d}.png")
    for frame in [0, 46, 92]:
        sheet = Image.new("RGB", (1024, 324), "#f6f4ee"); draw = ImageDraw.Draw(sheet)
        for column, (label, frames) in enumerate(zip(["成功源", "旧固定纹理派生", "纯色重渲染对照", "Cosmos纹理重渲染"], columns)):
            draw.text((column * 256 + 5, 5), label, font=font, fill="#23343e")
            sheet.paste(Image.fromarray(frames[frame]), (column * 256, 32))
        draw.text((5, 296), f"帧{frame}；仅视觉材质派生，非原生Cosmos视频；尚无独立方法与全质量门，禁止入训。", font=font, fill="#7a4237")
        sheet.save(directory / f"renderer_actual_{frame:03d}.png")
    report = {"method": CONFIG["version"], "config_sha256": sha256(CONFIG_PATH),
              "source_input": str(source_path), "source_input_sha256": sha256(source_path),
              "source_episode_id": evidence["source_episode_id"], "source_labels_sha256": evidence["labels_sha256"],
              "source_proof": proof, "source_replay_errors": errors,
              "texture_report": str(texture_report_path), "texture_report_sha256": sha256(texture_report_path),
              "texture_path": str(texture_path), "texture_sha256": sha256(texture_path),
              "native_generation_probe": texture_report["generation_probe"],
              "native_generation_probe_sha256": texture_report["generation_probe_sha256"],
              "render_state": {k: v for k, v in state.items() if k != "last"},
              "videos": {name: {"path": str(directory / (name + "_lossless.mp4")),
                                "sha256": sha256(directory / (name + "_lossless.mp4"))} for name in decoded},
              "pixel_certificate": rows, "common_table_pixels": int(common.sum()),
              "adjacent_mean_rgb_change_p95": {name: float(np.percentile([r["mean_abs_rgb_change_0_1"]
                  for r in temporal if r["kind"] == name], 95)) for name in ["source", "analytic", "cosmos"]},
              "geometry_evidence_type": "diagnostic_same_sensor_render_and_non_table_pixels",
              "output_kind": "background_composite", "derived_kind": "cosmos_texture_sim_rerender_design",
              "diagnostic_only": True, "eligible_for_training": False, "policy_improvement_claimed": False,
              "limitations": ["只验证已见设计成功源，未读取预留demo_23", "未完成独立源实际正负控及全帧视觉复核",
                              "原生Cosmos只提供纹理；基色归一化和MuJoCo重渲染均为显式派生", "没有训练或策略效果证据"]}
    (directory / "renderer_design.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return {key: report[key] for key in ["method", "source_replay_errors", "render_state",
                                       "adjacent_mean_rgb_change_p95", "eligible_for_training"]}


if __name__ == "__main__":
    arguments = argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    raise SystemExit(run_stage("table_renderer_design", lambda directory: diagnose(directory, arguments),
        {"compute": "Thor MuJoCo CPU物理与EGL视觉重渲染，无模型或训练", "method": CONFIG["version"],
         "source_windows": 1, "frames": 93, "fps": 20, "width": 256, "height": 256,
         "precision": "RGB uint8/物理fp64/动作状态fp32", "batch": 1, "seed": 0,
         "training_steps": 0, "diagnostic_only": True}))
