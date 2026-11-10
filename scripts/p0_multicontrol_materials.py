#!/usr/bin/env python3
"""保存深度/分割控制和原生多控制对照，保留真实质量门及原视频。

@author pony
@date 2026-11-07
@version v1.3.1
@last_modified 2026-11-08
@changelog
    - v1.3.1 (2026-11-08): 兼容实际新清单的父路径字符串加独立SHA字段，仍逐级校验并拒绝缺失。
    - v1.3.0 (2026-11-07): 沿已校验历史清单恢复初训配对CSV、逐帧截图和完整测量来源，缺失或改变即停。
    - v1.2.0 (2026-11-07): 保留全部二十段素材，追加只增加edge的原生四控制对照。
    - v1.1.0 (2026-11-07): 保留原十八段视频，追加原生vis外观约束的同源单因素对照及输入展示。
    - v1.0.1 (2026-11-07): 检查实际生成条件与原门禁摘要，导出全帧ROI曲线及配对CSV。
    - v1.0.0 (2026-11-07): 支持先保存控制信号，再追加全部93帧生成与质量门对照。
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
import json
import pathlib
import shutil

from p0_common import ROOT, run_stage
from p0_policy_runtime import sha256
from p0_transfer_smoke import validate_multicontrol_trace, MULTICONTROL_RECIPES
from p0_video_materials import render


def copy_training_progress_evidence(directory: pathlib.Path, previous_manifest: pathlib.Path,
                                    clips: list[dict]) -> dict | None:
    training = [clip for clip in clips if clip.get("kind") == "initial_training_progress"]
    if not training:
        return None
    path = previous_manifest.resolve(strict=True)
    visited = set()
    # 历史导出曾只复制MP4和海报；从有摘要的父清单恢复附件，禁止按同名文件猜来源。
    while True:
        if not path.is_relative_to(ROOT / "runs/p0") or path in visited:
            raise RuntimeError("初训来源链越界或成环")
        visited.add(path)
        manifest = json.loads(path.read_text())
        if manifest.get("paired_episodes") and manifest.get("subset_summary") \
                and manifest.get("training_conditions") and manifest.get("evaluation_conditions"):
            break
        parent = manifest.get("previous_manifest", {})
        # 两种历史清单保存同一来源关系；只兼容格式，不放宽逐级路径和摘要绑定。
        if isinstance(parent, str):
            parent = {"path": parent, "sha256": manifest.get("previous_manifest_sha256")}
        if not isinstance(parent, dict) or not parent.get("path") or not parent.get("sha256"):
            raise RuntimeError("缺少初训配对附件的原始来源")
        path = pathlib.Path(parent["path"]).resolve(strict=True)
        if not path.is_relative_to(ROOT / "runs/p0") or sha256(path) != parent["sha256"]:
            raise RuntimeError("初训来源链摘要不一致或越界")
    probe = json.loads((path.parent / "probe.json").read_text())
    if probe.get("stage") != "video_materials" or probe.get("status") != "passed" \
            or probe.get("artifacts_sha256", {}).get(path.name) != sha256(path):
        raise RuntimeError("初训原始清单偏离已完成阶段记录")
    originals = {clip["file"]: clip for clip in manifest["clips"]}
    files = []

    def copy_checked(name, expected):
        if not expected or pathlib.PurePosixPath(name).name != name:
            raise RuntimeError("初训附件缺摘要或文件名越界")
        source = (path.parent / name).resolve(strict=True)
        if not source.is_relative_to(path.parent) or sha256(source) != expected:
            raise RuntimeError("初训附件已改变或越界")
        shutil.copyfile(source, directory / name)
        if sha256(directory / name) != expected:
            raise RuntimeError("初训附件复制后摘要不同")
        files.append({"file": name, "sha256": expected, "source": str(source)})

    for clip in training:
        original = originals.get(clip["file"])
        if not original or original["sha256"] != clip["sha256"] \
                or original.get("screenshots") != clip.get("screenshots"):
            raise RuntimeError("初训视频或截图来源与原始配对记录不同")
        for screenshot in clip.get("screenshots", []):
            copy_checked(screenshot["file"], screenshot["sha256"])
    copy_checked("training_progress_pairs.csv", probe["artifacts_sha256"].get("training_progress_pairs.csv"))
    target = directory / "training_progress_original_manifest.json"
    shutil.copyfile(path, target)
    return {"source_manifest": str(path), "source_manifest_sha256": sha256(path),
            "archived_manifest": target.name, "archived_manifest_sha256": sha256(target), "files": files,
            "claim": manifest.get("claim"), "training_conditions": manifest["training_conditions"],
            "evaluation_conditions": manifest["evaluation_conditions"]}


def collect(directory: pathlib.Path, args: argparse.Namespace) -> dict:
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    evidence = json.loads(args.source_input.read_text())
    input_probe = json.loads((args.source_input.parent / "probe.json").read_text())
    four = evidence.get("generation_variant") == "base/edge+vis+depth+seg"
    visual = evidence.get("generation_variant") == "base/vis+depth+seg"
    if input_probe["status"] != "passed" or input_probe["stage"] != ("edge_anchor_input" if four else "visual_anchor_input" if visual else "multicontrol_input") \
            or not evidence.get("control_replay_validation", {}).get("passed"):
        raise RuntimeError("控制视频必须先通过真实逐帧重放核验")
    for filename, digest in evidence["files_sha256"].items():
        if sha256(args.source_input.parent / filename) != digest:
            raise RuntimeError("控制信号或原始成功源发生改变")
    old = json.loads(args.previous_manifest.read_text())
    clips = []
    for clip in old["clips"]:
        for name in [clip["file"], clip["poster"]]:
            path = args.previous_manifest.parent / name
            if name == clip["file"] and sha256(path) != clip["sha256"]:
                raise RuntimeError("既有展示视频偏离清单")
            shutil.copyfile(path, directory / name)
        clips.append(clip)
    training_evidence = copy_training_progress_evidence(directory, args.previous_manifest, clips)
    source = args.source_input.parent / "source.mp4"
    depth = args.source_input.parent / "depth.mp4"
    seg = args.source_input.parent / "seg.mp4"
    source_record = {"path": str(args.source_input), "sha256": sha256(args.source_input)}
    if not any(clip.get("kind") == "sim_depth_seg_controls" for clip in clips):
        clips.append(render(directory, "17_sim_source_depth_seg_controls.mp4", [source, depth, seg],
            ["同一成功示范RGB", "全窗口统一尺度的逆深度", "稳定身份的真值ROI颜色分割"],
            "原生多控制输入：同次观测的RGB、深度、分割", "Thor / 既有MAXN / MuJoCo CPU / 单成功示范 / 93帧 / 20Hz / seed0",
            ["物理状态、RGB、动作状态和ROI逐帧复现通过；没有采集真机数据。",
             "深度越近越亮；分割颜色只是类别编码，不是物体的真实颜色。",
             "此视频是控制准备，不是生成结果，不代表策略收益；原始标签和来源SHA256相邻保存。"], 93,
            {"kind": "sim_depth_seg_controls", "source_input": source_record,
             "policy_improvement_claimed": False, "eligible_for_training": False, "is_eval": False}))
    metadata = {"source_input": source_record, "previous_manifest": {
        "path": str(args.previous_manifest), "sha256": sha256(args.previous_manifest)},
        "control_generation_performed": False, "compute": "Thor CPU解码、排版和H264编码，不运行模型"}
    comparison_videos, labels = [source, depth, seg], ["成功源", "逆深度", "ROI颜色分割"]
    if args.generation_probe:
        probes = [json.loads(path.read_text()) for path in [args.before_probe, args.generation_probe]]
        if any(probe["stage"] != "transfer_smoke" or probe["status"] != "passed" for probe in probes):
            raise RuntimeError("生成对照必须实际完成")
        results = [probe["result"] for probe in probes]
        if results[1]["variant"] != evidence["generation_variant"] or not results[1].get("guardrails_enabled"):
            raise RuntimeError("新结果必须为启用安全检查且与输入匹配的原生多控制生成")
        if any(result.get("num_steps") != 35 or result.get("seed") != 42 or result.get("batch") != 1
               or result.get("sample_count") != 1 or str(result.get("resolution_requested")) != "480"
               for result in results):
            raise RuntimeError("生成图注必须与实际35步/seed42/batch1/480档的单窗口条件一致")
        validate_multicontrol_trace(results[1]["multicontrol_trace"],
                                   MULTICONTROL_RECIPES["multicontrol-vis-edge"] if four else MULTICONTROL_RECIPES["multicontrol-vis"] if visual else None)
        parent_path = pathlib.Path(results[0]["source_success_evidence"]["path"])
        if evidence["edge_parent_input_sha256" if four else "vis_parent_input_sha256" if visual else "parent_verified_input_sha256"] != sha256(parent_path) \
                or pathlib.Path(results[1]["source_success_evidence"]["path"]) != args.source_input:
            raise RuntimeError("生成不来自同一成功源窗口")
        parent = json.loads(parent_path.read_text())
        if parent["source_rgb_sha256"] != evidence["source_rgb_sha256"] or parent["labels_sha256"] != evidence["labels_sha256"]:
            raise RuntimeError("源RGB或动作标签不同，不允许作同源对照")
        gates = [json.loads(path.read_text()) for path in [args.before_gate, args.generation_gate]]
        if gates[0]["gate_config_sha256"] != gates[1]["gate_config_sha256"] \
                or gates[0]["sam_model_manifest_sha256"] != gates[1]["sam_model_manifest_sha256"]:
            raise RuntimeError("对照不得改变质量门阈值或SAM2检查模型")
        for gate, path in zip(gates, [args.before_probe, args.generation_probe]):
            if gate["generation_probe_sha256"] != sha256(path) or gate["segmented_frames"] != list(range(93)):
                raise RuntimeError("质量门未绑定生成摘要或未覆盖全93帧")
        videos = [pathlib.Path(result["outputs"][0]) for result in results]
        filename = "21_cosmos_visual_vs_four_controls.mp4" if four else "19_cosmos_depth_seg_vs_visual_anchor.mp4" if visual else "18_cosmos_edge_vs_depth_seg.mp4"
        if any(clip["file"] == filename for clip in clips):
            raise RuntimeError("新素材文件名已存在，禁止覆盖历史视频")
        labels_new = ["同一成功示范RGB", "原vis+depth+seg", "新edge+vis+depth+seg"] if four else ["同一成功示范RGB", "原base：depth+seg", "新base：vis+depth+seg"] if visual else ["同一成功示范RGB", "原base：edge控制", "新base：depth+seg多控制"]
        clips.append(render(directory, filename, [source, *videos], labels_new,
            "Cosmos轮廓控制对照：只增加edge=0.2" if four else "Cosmos外观控制对照：只增加vis/blur" if visual else "Cosmos配方对照：edge与depth/seg",
            "Thor / 既有MAXN / DiT bf16 / batch1 / 35步 / seed42 / 93帧 / 20fps",
            [f"原/新几何门：{'通过' if gates[0]['geometry_passed'] else '未通过'} / {'通过' if gates[1]['geometry_passed'] else '未通过'}；完整入训资格以质量报告为准。",
             "原生模型输出，无像素回填；" + ("只新增edge=0.2，其余控制、提示词及源标签不变。" if four else "只新增vis控制，depth/seg、提示词及源标签不变。" if visual else "模型分支及控制信号同时改变，这是配方对照。"),
             "单一成功窗口、单种子；不代表策略成功率、统计增益或RSI闭环收益。"], 93,
            {"kind": "native_four_control_comparison" if four else "native_visual_anchor_comparison" if visual else "native_depth_seg_generation_comparison", "is_eval": False,
             "generation_probes": [{"path": str(path), "sha256": sha256(path)}
                                    for path in [args.before_probe, args.generation_probe]],
             "quality_gates": [{"path": str(path), "sha256": sha256(path)}
                               for path in [args.before_gate, args.generation_gate]],
             "policy_improvement_claimed": False, "eligible_for_training": False}))
        if visual:
            clips.append(render(directory, "20_cosmos_source_blur_generated.mp4", [source, args.source_input.parent / "vis.mp4", videos[1]],
                ["同一成功源RGB", "原生高强度blur控制", "Cosmos原生输出"], "Cosmos外观约束：源、控制与生成",
                "Thor / 既有MAXN / DiT bf16 / batch1 / 35步 / seed42 / 93帧 / 20fps",
                ["blur是源图派生的控制输入，右侧才是生成；不做像素回填。",
                 "depth/seg各1，vis为0.5；权重按实际trace记录，未声称上游自动归一化。",
                 "生成数据资格以全部93帧原质量门和语义复核为准；不是策略优化收益。"], 93,
                {"kind": "native_visual_anchor_controls", "source_input": source_record, "is_eval": False,
                 "generation_probe": {"path": str(args.generation_probe), "sha256": sha256(args.generation_probe)},
                 "policy_improvement_claimed": False, "eligible_for_training": False}))
        recipe_labels = ["vis+depth+seg", "edge+vis+depth+seg"] if four else ["depth+seg", "vis+depth+seg"] if visual else ["edge", "depth+seg"]
        rows = []
        for label, gate in zip(recipe_labels, gates):
            for frame in gate["per_frame"]:
                for name, roi in frame["rois"].items():
                    rows.append({"recipe": label, "frame": frame["frame"], "roi": name, **roi})
        if rows:
            columns = list(dict.fromkeys(key for row in rows for key in row))
            with (directory / "multicontrol_quality.csv").open("w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader(); writer.writerows(rows)
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.font_manager import FontProperties

            chinese = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
            figure, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
            for axis, name in zip(axes, evidence["roi_names"]):
                for label in recipe_labels:
                    selected = [row for row in rows if row["recipe"] == label and row["roi"] == name]
                    axis.plot([row["frame"] for row in selected], [row["generated_iou_vs_sim_gt"] for row in selected],
                              label=label, linewidth=1.2)
                threshold = 0.80 if name == "robot" else 0.75
                axis.axhline(threshold, color="#954d43", linestyle="--", label=f"threshold {threshold:.2f}")
                axis.set_ylim(0, 1.02); axis.set_ylabel(f"{name}\nIoU"); axis.legend(loc="lower right", ncol=3)
            axes[-1].set_xlabel("Frame / 20 fps")
            figure.suptitle("同源单窗口原生配方对照：全部93帧，原阈值；不是策略收益", fontproperties=chinese)
            figure.savefig(directory / "multicontrol_roi_quality.png", dpi=180)
            plt.close(figure)
        metadata.update({"generation_probes": [{"path": str(path), "sha256": sha256(path)}
                                                for path in [args.before_probe, args.generation_probe]],
                         "quality_gates": [{"path": str(path), "sha256": sha256(path),
                                            "eligible_for_training": gate["eligible_for_training"],
                                            "geometry_passed": gate["geometry_passed"], "reject_reasons": gate["reject_reasons"]}
                                           for path, gate in zip([args.before_gate, args.generation_gate], gates)]})
        comparison_videos += videos
        labels += ["原vis/depth/seg生成", "新edge/vis/depth/seg生成"] if four else ["原depth/seg生成", "新vis/depth/seg生成"] if visual else ["原edge生成", "新depth/seg生成"]
    font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 19, index=2)
    with ExitStack() as stack:
        readers = [stack.enter_context(imageio.get_reader(path)) for path in comparison_videos]
        sheet = Image.new("RGB", (1056, 292 * len(readers)), "#f5f4ef")
        draw = ImageDraw.Draw(sheet)
        for row, (reader, label) in enumerate(zip(readers, labels)):
            draw.text((8, row * 292 + 4), label, font=font, fill="#20313c")
            for column, frame in enumerate([0, 30, 61, 92]):
                sheet.paste(Image.fromarray(reader.get_data(frame)).resize((256, 256)), (column * 264, row * 292 + 31))
        sheet.save(directory / "multicontrol_contact_sheet.png")
    report = {**metadata, "clips": clips, "clip_count": len(clips), "eligible_for_training": False,
              "training_progress_evidence": training_evidence,
              "policy_improvement_claimed": False, "presentation_created": False}
    (directory / "video_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (directory / "README.md").write_text("# 真实视频素材\n\n保留原视频及500/5000步初训对照，追加同帧深度分割控制。\n"
        "实际生成对照仅在模型与全帧质量门完成后导出，入训资格见独立质量报告。没有PPT。\n")
    return {"materials": str(directory), "clip_count": len(clips), "generation_compared": bool(args.generation_probe),
            "eligible_for_training": False, "policy_improvement_claimed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ["source-input", "previous-manifest"]:
        parser.add_argument("--" + name, type=pathlib.Path, required=True)
    for name in ["generation-probe", "generation-gate", "before-probe", "before-gate"]:
        parser.add_argument("--" + name, type=pathlib.Path)
    arguments = parser.parse_args()
    supplied = [bool(getattr(arguments, key)) for key in ["generation_probe", "generation_gate", "before_probe", "before_gate"]]
    if any(supplied) and not all(supplied):
        parser.error("生成对照必须同时提供新旧实际生成和全帧质量报告")
    for key, value in vars(arguments).items():
        if value:
            path = value.resolve(strict=True)
            if not path.is_relative_to(ROOT / "runs/p0"):
                parser.error("素材输入必须位于本项目P0目录")
            setattr(arguments, key, path)
    raise SystemExit(run_stage("video_materials", lambda directory: collect(directory, arguments), {"compute": "CPU_only"}))
