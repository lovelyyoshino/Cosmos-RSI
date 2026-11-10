#!/usr/bin/env python3
"""归档独立成功源的原生生成视频与全帧质量数据，保留既有素材及来源。

@author pony
@date 2026-11-07
@version v1.1.1
@last_modified 2026-11-08
@changelog
    - v1.1.1 (2026-11-08): 低vis单因素素材按新具名配方核验全部四分支实测权重。
    - v1.1.0 (2026-11-07): 素材按实际具名配方校验；ROI控制须绑定原生/参照数组及源mask摘要。
    - v1.0.0 (2026-11-07): 单窗口源/控制/输出同帧展示、ROI曲线和背景描述统计，不宣称跨场景消融收益。
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
import json
import pathlib
import shutil

from p0_common import ROOT, run_stage
from p0_multicontrol_materials import copy_training_progress_evidence
from p0_policy_runtime import sha256
from p0_transfer_smoke import MULTICONTROL_RECIPES, validate_multicontrol_trace
from p0_video_materials import render


def validate_sample_records(generation: dict, gate: dict, evidence: dict, generation_sha: str, source_sha: str,
                            *, generation_directory: pathlib.Path | None = None) -> None:
    if generation.get("stage") != "transfer_smoke" or generation.get("status") != "passed":
        raise ValueError("只归档实际完成的原生生成")
    result = generation["result"]
    if result.get("source_success_evidence", {}).get("sha256") != source_sha \
            or gate.get("source_input_sha256") != source_sha or gate.get("generation_probe_sha256") != generation_sha:
        raise ValueError("生成、成功源和质量报告的摘要绑定不一致")
    if not evidence.get("source_success") or not evidence.get("replay_validation", {}).get("passed") \
            or not evidence.get("control_replay_validation", {}).get("passed"):
        raise ValueError("独立源及控制必须先通过真实重放")
    if result.get("variant") != "base/edge+vis+depth+seg" or not result.get("guardrails_enabled") \
            or result.get("num_steps") != 35 or result.get("seed") != 42 or result.get("batch") != 1 \
            or result.get("sample_count") != 1 or str(result.get("resolution_requested")) != "480":
        raise ValueError("展示条件与实际生成配方不一致")
    # 旧四控制记录没有recipe字段；空间控制不能沿用标量校验，否则会误拒绝真实网络输入。
    recipe = result.get("recipe", "multicontrol-vis-edge")
    if recipe not in {"multicontrol-vis-edge", "multicontrol-balanced", "multicontrol-roi-background",
                      "multicontrol-low-vis-background"} \
            or evidence.get("control_recipe", recipe) != recipe:
        raise ValueError("生成配方与成功源预声明不一致或未受支持")
    weights = MULTICONTROL_RECIPES[recipe]
    spatial_evidence = result.get("native_control_weight_evidence")
    if recipe == "multicontrol-roi-background":
        import numpy as np

        if generation_directory is None or not isinstance(spatial_evidence, dict) \
                or result.get("control_scope") != "robot_and_targets_roi_all_four_controls" \
                or spatial_evidence.get("source_roi_position_verified") is not True \
                or spatial_evidence.get("requested_raw_weights") != weights \
                or not spatial_evidence.get("mask_sha256") \
                or spatial_evidence["mask_sha256"] != evidence.get("files_sha256", {}).get("mask.mp4"):
            raise ValueError("ROI素材缺少实际空间控制或同源mask证据")
        arrays = []
        for field, digest_key, filename in [("file", "sha256", "native_control_weights.npy"),
                                             ("expected_file", "expected_sha256", "expected_control_weights.npy")]:
            path = (generation_directory / filename).resolve(strict=True)
            if spatial_evidence.get(field) != filename or path.parent != generation_directory.resolve(strict=True) \
                    or sha256(path) != spatial_evidence.get(digest_key) \
                    or generation.get("artifacts_sha256", {}).get(filename) != spatial_evidence[digest_key]:
                raise ValueError("ROI控制数组未绑定实际生成阶段的摘要或已越界")
            arrays.append(np.load(path, allow_pickle=False))
        if arrays[0].shape[:3] != (4, 1, 24):
            raise ValueError("ROI控制时序与93帧窗口不一致")
        validate_multicontrol_trace(result["multicontrol_trace"], weights, *arrays)
    else:
        if spatial_evidence is not None:
            raise ValueError("全帧配方不能混入ROI控制数组")
        validate_multicontrol_trace(result["multicontrol_trace"], weights)
    if gate.get("segmented_frames") != list(range(93)) or [row["frame"] for row in gate.get("per_frame", [])] != list(range(93)):
        raise ValueError("必须保留有序全部93帧的原质量检查")
    if set(evidence["roi_names"]) != set().union(*(row["rois"] for row in gate["per_frame"])):
        raise ValueError("质量报告缺少源ROI")


def collect(directory: pathlib.Path, args: argparse.Namespace) -> dict:
    import cv2
    import imageio.v2 as imageio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    generation = json.loads(args.generation_probe.read_text())
    gate = json.loads(args.quality_gate.read_text())
    result = generation["result"]
    source_path = pathlib.Path(result["source_success_evidence"]["path"]).resolve(strict=True)
    if not source_path.is_relative_to(ROOT / "runs/p0"):
        raise RuntimeError("成功源越界")
    evidence = json.loads(source_path.read_text())
    validate_sample_records(generation, gate, evidence, sha256(args.generation_probe), sha256(source_path),
                            generation_directory=args.generation_probe.parent)
    for name, expected in {**evidence["files_sha256"], "source_labels.npz": evidence["labels_sha256"]}.items():
        if pathlib.PurePosixPath(name).name != name or sha256(source_path.parent / name) != expected:
            raise RuntimeError("源图、控制或原始标签偏离来源摘要")
    gate_probe_path = args.quality_gate.parent / "probe.json"
    gate_probe = json.loads(gate_probe_path.read_text())
    if gate_probe.get("status") != "passed" or gate_probe.get("stage") != "generated_window_gate" \
            or sha256(args.quality_gate) != gate_probe["artifacts_sha256"].get("quality_gate.json") \
            or gate["gate_config_sha256"] != sha256(ROOT / "configs/gen_gate.yaml"):
        raise RuntimeError("质量报告、原门配置或实际阶段记录改变")
    output_path = pathlib.Path(result["outputs"][0]).resolve(strict=True)
    if not output_path.is_relative_to(args.generation_probe.parent) \
            or sha256(output_path) != generation["artifacts_sha256"].get(str(output_path.relative_to(args.generation_probe.parent))):
        raise RuntimeError("原生生成输出偏离实际阶段摘要")
    old = json.loads(args.previous_manifest.read_text())
    clips = list(old["clips"])
    if old["clip_count"] != len(clips):
        raise RuntimeError("既有视频清单计数错误")
    for clip in clips:
        if pathlib.PurePosixPath(clip["file"]).name != clip["file"] or pathlib.PurePosixPath(clip["poster"]).name != clip["poster"]:
            raise RuntimeError("既有素材文件名越界")
        for name in [clip["file"], clip["poster"]]:
            original = args.previous_manifest.parent / name
            if name == clip["file"] and sha256(original) != clip["sha256"]:
                raise RuntimeError("已有视频摘要不一致")
            shutil.copyfile(original, directory / name)
    training = copy_training_progress_evidence(directory, args.previous_manifest, clips)
    filename = f"{len(clips) + 1:02d}_cosmos_independent_source.mp4"
    if any(clip["file"] == filename for clip in clips):
        raise RuntimeError("新素材名称与历史视频冲突")
    sources = [source_path.parent / "source.mp4", source_path.parent / "seg.mp4",
               source_path.parent / "vis.mp4", output_path]
    geometry = "通过" if gate["geometry_passed"] else "未通过"
    clips.append(render(directory, filename, sources, ["成功训练源RGB", "同帧真值分割控制", "原生blur外观控制", "Cosmos原生生成"],
        "Cosmos独立场景验证：成功源、控制与实际生成", "Thor / 既有MAXN / DiT bf16 / batch1 / 35步 / seed42 / 93帧 / 480×480",
        [f"任务：{evidence['instruction']}",
         f"原几何门{geometry}；语义及完整入训资格见质量报告。源动作/状态保持不变，无像素回填。",
         "一个独立训练源、一个种子；与取书场景不同，不作同源消融或策略/RSI收益结论。"], 93,
        {"kind": "independent_source_generation", "is_eval": False, "eligible_for_training": False,
         "source_episode_id": evidence["source_episode_id"], "source_input": str(source_path),
         "source_input_sha256": sha256(source_path), "generation_probe": str(args.generation_probe),
         "generation_probe_sha256": sha256(args.generation_probe), "quality_gate": str(args.quality_gate),
         "quality_gate_sha256": sha256(args.quality_gate), "geometry_passed": gate["geometry_passed"],
         "policy_improvement_claimed": False, "hardware": generation["hardware"],
         "conditions": {key: result[key] for key in ["precision", "batch", "num_steps", "seed", "sample_count"]}}))
    roi_rows = [{"frame": frame["frame"], "roi": name, **metrics} for frame in gate["per_frame"]
                for name, metrics in frame["rois"].items()]
    with (directory / "independent_source_roi.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(roi_rows[0]))
        writer.writeheader(); writer.writerows(roi_rows)
    figure, axes = plt.subplots(len(evidence["roi_names"]), 1, figsize=(12, 2.8 * len(evidence["roi_names"])),
                                sharex=True, squeeze=False, constrained_layout=True)
    for axis, name in zip(axes[:, 0], evidence["roi_names"]):
        selected = [row for row in roi_rows if row["roi"] == name]
        axis.plot([row["frame"] for row in selected], [row["source_calibration_iou"] for row in selected], label="Source SAM2 vs GT")
        axis.plot([row["frame"] for row in selected], [row["generated_iou_vs_sim_gt"] for row in selected], label="Generated SAM2 vs GT")
        axis.axhline(0.80 if name == "robot" else 0.75, color="#a04836", linestyle="--", label="Original generated threshold")
        axis.set_ylim(0, 1.02); axis.set_ylabel(f"{name}\nIoU"); axis.legend(loc="lower right", ncol=3)
    axes[-1, 0].set_xlabel("Frame / 20 fps")
    chinese = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    figure.suptitle("独立成功源的全帧原质量门：单窗口，不代表策略收益", fontproperties=chinese)
    figure.savefig(directory / "independent_source_roi_quality.png", dpi=180)
    plt.close(figure)
    font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 18, index=2)
    masks = np.load(source_path.parent / "roi_masks.npy", allow_pickle=False).astype(bool)
    appearance = []
    with ExitStack() as stack:
        readers = [stack.enter_context(imageio.get_reader(path)) for path in sources]
        sheet = Image.new("RGB", (1056, 292 * len(readers)), "#f5f4ef")
        draw = ImageDraw.Draw(sheet)
        for i, reader in enumerate(readers):
            for column, frame in enumerate([0, 30, 61, 92]):
                draw.text((column * 264 + 4, i * 292 + 3), f"{['源RGB', '分割控制', 'blur控制', '生成'][i]}：帧{frame}", font=font, fill="#20313c")
                sheet.paste(Image.fromarray(reader.get_data(frame)).resize((256, 256)), (column * 264, i * 292 + 31))
        sheet.save(directory / "independent_source_contact_sheet.png")
        for frame in range(93):
            background = ~masks[frame].any(axis=0)
            row = {"frame": frame, "background_pixels": int(background.sum())}
            for label, reader in [("source", readers[0]), ("generated", readers[-1])]:
                image = cv2.resize(reader.get_data(frame), (256, 256), interpolation=cv2.INTER_AREA)
                rgb = image[background].astype(np.float64) / 255
                row[label + "_luma_mean"] = float((rgb @ np.asarray([0.2126, 0.7152, 0.0722])).mean())
                row[label + "_red_minus_blue"] = float((rgb[:, 0] - rgb[:, 2]).mean())
            appearance.append(row)
    with (directory / "independent_source_appearance.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(appearance[0])); writer.writeheader(); writer.writerows(appearance)
    report = {"clips": clips, "clip_count": len(clips), "previous_manifest": {"path": str(args.previous_manifest),
              "sha256": sha256(args.previous_manifest)}, "training_progress_evidence": training,
              "source_input": str(source_path), "source_input_sha256": sha256(source_path),
              "generation_probe": str(args.generation_probe), "generation_probe_sha256": sha256(args.generation_probe),
              "quality_gate": str(args.quality_gate), "quality_gate_sha256": sha256(args.quality_gate),
              "hardware": generation["hardware"], "generation_wall_s": result["generate_wall_s"],
              "peak_mem_gb": generation["peak_mem_gb"], "quality_gate_reject_reasons": gate["reject_reasons"],
              "eligible_for_training": False, "policy_improvement_claimed": False, "presentation_created": False,
              "appearance_note": "背景亮度/红蓝差只是逐帧描述统计，未标记目标外观条件通过；帧数不是独立窗口数",
              "compute": "Thor CPU解码、图表与H264编码，不运行模型"}
    (directory / "video_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return {"materials": str(directory), "clip_count": len(clips), "eligible_for_training": False,
            "geometry_passed": gate["geometry_passed"], "policy_improvement_claimed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ["generation-probe", "quality-gate", "previous-manifest"]:
        parser.add_argument("--" + name, type=pathlib.Path, required=True)
    arguments = parser.parse_args()
    for key, value in vars(arguments).items():
        path = value.resolve(strict=True)
        if not path.is_relative_to(ROOT / "runs/p0"):
            parser.error("素材来源必须位于本项目P0目录")
        setattr(arguments, key, path)
    raise SystemExit(run_stage("video_materials", lambda directory: collect(directory, arguments), {"compute": "CPU_only"}))
