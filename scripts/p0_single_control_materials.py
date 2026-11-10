#!/usr/bin/env python3
"""导出真实单分支前向证据与桌面时序诊断，不生成训练资格或策略收益。

@author pony
@date 2026-11-08
@version v1.0.0
@last_modified 2026-11-08
@changelog
    - v1.0.0 (2026-11-08): 保存源/权重总览、全部逐层/逐帧CSV及同配方复跑像素核对。
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib

from p0_background_preview_materials import checked_artifact
from p0_common import ROOT, ensure_gpu_idle, run_stage
from p0_policy_runtime import sha256
from p0_single_control_trace import validate_trace, weight_statistics


def static_table_changes(frames, visible):
    """只量测整段共同可见桌体，不把相机/遮挡移动算成材质闪变。"""
    import numpy as np

    frames, visible = np.asarray(frames), np.asarray(visible)
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[0] < 2 or frames.shape[-1] != 3 \
            or visible.dtype != np.bool_ or visible.shape != frames.shape[1:3] or not visible.any():
        raise ValueError("桌面时序诊断需要完整uint8视频及共同可见bool区域")
    delta = np.abs(np.diff(frames.astype(np.int16), axis=0))[:, visible]
    return [{"from_frame": i, "to_frame": i + 1, "common_pixels": int(visible.sum()),
             "mean_abs_rgb_change_0_1": float(value.mean() / 255),
             "p95_abs_rgb_change_0_1": float(np.percentile(value, 95) / 255)}
            for i, value in enumerate(delta)]


def export(directory, args):
    ensure_gpu_idle()
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from p0_lossless_background import read_exact_video

    probe = json.loads(args.trace_probe.read_text())
    trace_path = args.trace_probe.parent / "single_control_trace.json"
    checked_artifact(trace_path, "single_control_trace")
    trace = json.loads(trace_path.read_text())
    validate_trace(trace["calls"], trace["controlled_blocks"])
    declaration_path = args.trace_probe.parent / "declaration.json"
    checked_artifact(declaration_path, "single_control_trace")
    declaration = json.loads(declaration_path.read_text())
    baseline_path = pathlib.Path(declaration["baseline_probe"])
    if sha256(baseline_path) != declaration["baseline_probe_sha256"]:
        raise ValueError("复跑比较基线记录已改变")
    baseline = json.loads(baseline_path.read_text())
    spec = pathlib.Path(probe["result"]["source_spec"])
    if sha256(spec) != declaration["source_spec_sha256"]:
        raise ValueError("复跑源配置已改变")
    from p0_table_background import checked_table_source
    source_path = spec.parent / "verified_input.json"
    if sha256(source_path) != declaration["source_input_sha256"]:
        raise ValueError("复跑的成功源证据已改变")
    checked_table_source(source_path)
    for run in [probe, baseline]:
        binding = run["result"].get("source_success_evidence", {})
        if binding.get("path") != str(source_path) or binding.get("sha256") != sha256(source_path):
            raise ValueError("原生生成与复跑没有绑定完全相同的成功源证据")

    def read_bound_video(record, path):
        video = pathlib.Path(record["result"]["outputs"][0]).resolve(strict=True)
        if not video.is_relative_to(path.parent) or sha256(video) != record["artifacts_sha256"].get(
                str(video.relative_to(path.parent))):
            raise ValueError("真实生成视频未绑定阶段产物摘要")
        return read_exact_video(video, size=None)[0]

    native = read_bound_video(probe, args.trace_probe)
    before = read_bound_video(baseline, baseline_path)
    if native.shape != before.shape:
        raise ValueError("同配方复跑输出尺寸或帧数改变")
    source, _ = read_exact_video(spec.parent / "source.mp4")
    editable = np.load(spec.parent / "editable_table.npy", allow_pickle=False)
    visible = editable.all(axis=0)
    resized = np.asarray([cv2.resize(frame, (256, 256), interpolation=cv2.INTER_AREA) for frame in native])
    source_rows, native_rows = static_table_changes(source, visible), static_table_changes(resized, visible)
    temporal_rows = [{"kind": kind, **row} for kind, rows in [("source", source_rows), ("native", native_rows)] for row in rows]
    for filename in ["actual_network_weights.npy", "expected_cpu_weights.npy"]:
        checked_artifact(args.trace_probe.parent / filename, "single_control_trace")
    actual = np.load(args.trace_probe.parent / "actual_network_weights.npy", allow_pickle=False)
    expected = np.load(args.trace_probe.parent / "expected_cpu_weights.npy", allow_pickle=False)
    if actual.shape != expected.shape or actual.shape[0] != 4:
        raise ValueError("实际/期望权重数组不覆盖相同的完整四步")
    for index in range(4):
        weight_statistics(actual[index], expected[index])
    layer_rows = [{"step": row["step"], **block} for row in trace["calls"] for block in row["blocks"]]
    for filename, rows in [("all_injection_layers.csv", layer_rows), ("static_table_temporal.csv", temporal_rows)]:
        with (directory / filename).open("x", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    difference = np.abs(native.astype(np.int16) - before.astype(np.int16))
    summary = {"trace_probe": str(args.trace_probe), "trace_probe_sha256": sha256(args.trace_probe),
               "baseline_probe": str(baseline_path), "baseline_probe_sha256": sha256(baseline_path),
               "source_spec_sha256": sha256(spec), "actual_steps": len(trace["calls"]),
               "injection_layers_per_step": len(trace["controlled_blocks"]),
               "all_layer_weights_match": True, "baseline_decoded_pixels_identical": bool((difference == 0).all()),
               "baseline_max_abs_rgb_difference": int(difference.max()),
               "baseline_mean_abs_rgb_difference": float(difference.mean()),
               "common_visible_table_pixels": int(visible.sum()),
               "source_adjacent_mean_rgb_change_p95": float(np.percentile([r["mean_abs_rgb_change_0_1"] for r in source_rows], 95)),
               "native_adjacent_mean_rgb_change_p95": float(np.percentile([r["mean_abs_rgb_change_0_1"] for r in native_rows], 95)),
               "eligible_for_training": False, "diagnostic_only": True,
               "caveat": "共同可见桌面的时序量测只作诊断；不替代材质、边界、身份复核或策略冻结评测"}
    (directory / "diagnostic_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    fig = plt.figure(figsize=(15, 8), constrained_layout=True)
    grid = fig.add_gridspec(2, 5)
    panels = [(source[46], "Success source / frame 46"), (resized[46], "Observed Cosmos output / frame 46"),
              (~editable[46], "Requested protection / frame 46"),
              (expected[0, 0, 0, 12, :, :, 0], "CPU reference / latent t=12"),
              (actual[0, 0, 0, 12, :, :, 0], "Actual network / latent t=12")]
    for index, (value, title) in enumerate(panels):
        ax = fig.add_subplot(grid[0, index])
        ax.imshow(value, cmap="gray", vmin=0, vmax=1) if value.ndim == 2 else ax.imshow(value)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    ax = fig.add_subplot(grid[1, :3])
    for label, rows in [("source", source_rows), ("Cosmos output", native_rows)]:
        ax.plot([r["to_frame"] for r in rows], [r["mean_abs_rgb_change_0_1"] for r in rows], label=label)
    ax.set(xlabel="Actual video frame / 20fps", ylabel="Adjacent mean absolute RGB change / 255",
           title=f"Same visible table throughout ({visible.sum():,} pixels)")
    ax.legend()
    ax.grid(alpha=.2)
    ax = fig.add_subplot(grid[1, 3:])
    heat = np.asarray([[r["max_abs_error_vs_network"] for r in row["blocks"]] for row in trace["calls"]])
    ax.imshow(heat, cmap="Greens_r", vmin=0, vmax=1, aspect="auto")
    ax.set(xlabel="Actual control injection layer", ylabel="Denoising forward call", title="Block vs network weights: max absolute error")
    ax.set_xticks(range(len(trace["controlled_blocks"])), trace["controlled_blocks"])
    ax.set_yticks(range(4), range(4))
    for y, row in enumerate(heat):
        for x, value in enumerate(row):
            ax.text(x, y, f"{value:g}", ha="center", va="center", fontsize=8)
    fig.suptitle("Thor / existing MAXN / edge-distilled / DiT bf16 / batch 1 / 4 steps / seed 42 / 1 x 93 frames", fontsize=11)
    fig.supxlabel("Observation rerun only. White constrains edges; black releases them. Mask efficacy does not prove material or temporal quality.", fontsize=10)
    for extension in ["png", "svg"]:
        fig.savefig(directory / ("single_control_diagnostic." + extension), dpi=180)
    plt.close(fig)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trace-probe", type=pathlib.Path, required=True)
    arguments = parser.parse_args()
    arguments.trace_probe = arguments.trace_probe.resolve(strict=True)
    raise SystemExit(run_stage("single_control_materials", lambda directory: export(directory, arguments),
                     {"compute": "CPU uint8视频解码/float32权重对照，不运行模型", "source_windows": 1,
                      "frames": 93, "fps": 20, "diagnostic_only": True}))
