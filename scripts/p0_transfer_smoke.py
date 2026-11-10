#!/usr/bin/env python3
"""使用已校验组件执行官方edge或depth/seg生成，区分蒸馏与base质量诊断。

@author pony
@date 2026-11-06
@version v1.18.0
@last_modified 2026-11-08
@changelog
    - v1.18.0 (2026-11-08): 区分桌面v2的全部非桌体mask与旧任务ROI控制，绑定实际新源掩码。
    - v1.17.0 (2026-11-08): 增加仅降低全帧vis控制的背景配方，核验预声明来源与实际分支权重。
    - v1.16.0 (2026-11-07): 前景区域四控制核验实际空间权重与上游逐位置归一化，禁止静默全帧控制。
    - v1.15.0 (2026-11-07): 新增四控制同比缩放的具名配方，保留旧权重并在结果中记录实际recipe。
    - v1.14.0 (2026-11-07): 支持只增加edge=0.2的四控制对照，保留所有既有配方并核验实测权重。
    - v1.13.0 (2026-11-07): 增加同源预计算vis外观控制，逐分支核对实际权重与请求一致。
    - v1.12.0 (2026-11-07): 增加显式本地四分支的depth/seg生成，核验实际latent输入和控制权重。
    - v1.11.0 (2026-11-07): 记录派生时空引导的来源及原生latent权重，不修改上游采样结果。
    - v1.10.1 (2026-11-07): 条件分支要求非零，CFG无条件分支允许零编码；校验前先保存实际计时。
    - v1.10.0 (2026-11-07): 记录实际图像参考输入及SigLIP2非零embedding，禁止仅凭配置声称条件生效。
    - v1.9.0 (2026-11-07): 加载后核验每个实际注意力分支，避免base图像分支静默退回TE。
    - v1.8.0 (2026-11-07): base edge额外校验本地SigLIP2上下文编码器，缺少依赖时在加载GPU模型前停止。
    - v1.7.0 (2026-11-07): base edge支持原生空间引导；拒绝蒸馏入口中实际无效的参数，按权重文件选择校验清单。
    - v1.6.0 (2026-11-06): 显式记录实际控制范围，区分局部mask与全帧edge，避免将检查ROI误称控制mask。
    - v1.5.0 (2026-11-06): 关联本机D0成功回合与逐帧重放验证，核验原生ROI掩码输入摘要。
    - v1.4.0 (2026-11-06): 关联本机成功示范回放的证据哈希，不把成功来源等同于生成质量通过。
    - v1.3.0 (2026-11-06): Thor缺少FlashAttention预编译内核时显式使用原生SDPA后端。
    - v1.2.0 (2026-11-06): 离线时仅注册Transfer蒸馏配置，避免无关模型的导入期下载。
    - v1.1.0 (2026-11-06): 在切换上游工作目录前解析输入路径，避免相对路径失效。
    - v1.0.0 (2026-11-06): 用显式本地覆盖串起官方生成API，保留原始帧和性能证据。
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time

from p0_common import ROOT, ensure_gpu_idle, run_stage


TRANSFER_WEIGHTS = {
    "edge/distilled": "distilled/general/edge/41f07f13-f2e4-4e34-ba4c-86f595acbc20_ema_bf16.pt",
    "edge": "general/edge/61f5694b-0ad5-4ecd-8ad7-c8545627d125_ema_bf16.pt",
    "vis": "general/blur/ba2f44f2-c726-4fe7-949f-597069d9b91c_ema_bf16.pt",
    "depth": "general/depth/626e6618-bfcd-4d9a-a077-1409e2ce353f_ema_bf16.pt",
    "seg": "general/seg/5136ef49-6d8d-42e8-8abf-7dac722a304a_ema_bf16.pt",
}
MULTICONTROL_ORDER = ["edge", "vis", "depth", "seg"]
# bf16空间插值及逐位置归一化可有约千分之几的舍入；不把数值误差当成控制失效。
SPATIAL_WEIGHT_RTOL = 8e-3
SPATIAL_WEIGHT_ATOL = 2e-4
MULTICONTROL_RECIPES = {
    "multicontrol": {"depth": 1.0, "seg": 1.0},
    "multicontrol-vis": {"vis": 0.5, "depth": 1.0, "seg": 1.0},
    "multicontrol-vis-edge": {"edge": 0.2, "vis": 0.5, "depth": 1.0, "seg": 1.0},
    "multicontrol-balanced": {"edge": 2 / 27, "vis": 5 / 27, "depth": 10 / 27, "seg": 10 / 27},
    "multicontrol-roi-background": {"edge": 0.2, "vis": 0.5, "depth": 1.0, "seg": 1.0},
    "multicontrol-low-vis-background": {"edge": 0.2, "vis": 0.1, "depth": 1.0, "seg": 1.0},
}


def checked_control_checkpoint(model: str) -> dict:
    filename = TRANSFER_WEIGHTS[model]
    selected = None
    for manifest in sorted(ROOT.glob("runs/p0/model_download_*/model_manifest.json")):
        record = json.loads(manifest.read_text())
        if pathlib.Path(record["path"]).name != "Cosmos-Transfer2.5-2B":
            continue
        for row in record["files"]:
            if row["file"] == filename:
                selected = (manifest, record, row)
    if selected is None:
        raise RuntimeError(f"缺少{model}分支的已校验下载清单")
    manifest, record, row = selected
    root = pathlib.Path(record["path"]).resolve(strict=True)
    path = (root / filename).resolve(strict=True)
    if root != ROOT / "checkpoints/base/Cosmos-Transfer2.5-2B" or not path.is_relative_to(root) \
            or not path.is_file() or path.stat().st_size != row["bytes"] or sha256_file(path) != row["sha256"]:
        raise RuntimeError(f"{model}分支不在项目目录或SHA256/大小不匹配")
    return {"path": str(path), "sha256": row["sha256"], "bytes": row["bytes"],
            "manifest": str(manifest), "manifest_sha256": sha256_file(manifest),
            "revision": record["revision"], "backend": record.get("backend", "huggingface")}


def expected_spatial_control_weights(mask, shape: tuple[int, ...], expected_weights: dict[str, float]):
    """按固定上游首帧独立、后续帧三线性缩放的规则，从实际解码ROI建立CPU参照。"""
    import numpy as np
    import torch
    import torch.nn.functional as functional

    binary = np.asarray(mask)
    if binary.ndim != 3 or binary.shape[0] != 93 or binary.dtype != np.bool_ \
            or len(shape) != 6 or shape[:2] != (4, 1) or shape[-1] != 1 \
            or shape[2] != 24 or min(shape[3:5]) <= 1:
        raise RuntimeError("源ROI或实际时空控制权重尺寸错误")
    tensor = torch.from_numpy(binary.copy()).to(dtype=torch.float32)[None, None]
    first = functional.interpolate(tensor[:, :, :1], size=(1, shape[3], shape[4]),
                                   mode="trilinear", align_corners=False)
    rest = functional.interpolate(tensor[:, :, 1:], size=(shape[2] - 1, shape[3], shape[4]),
                                  mode="trilinear", align_corners=False)
    common = torch.cat([first, rest], dim=2).permute(0, 2, 3, 4, 1).numpy()
    values = np.stack([common * expected_weights[key] for key in MULTICONTROL_ORDER])
    return values / np.maximum(1.0, values.sum(axis=0, keepdims=True))


def spatial_control_statistics(weights, expected_weights: dict[str, float], expected_spatial=None) -> dict:
    """核验送入网络的实际空间权重，不能用spec中的mask字段代替生效证据。"""
    import numpy as np

    values = np.asarray(weights)
    requested = np.asarray([expected_weights.get(key, 0.0) for key in MULTICONTROL_ORDER], dtype=np.float64)
    if values.ndim != 6 or values.shape[:2] != (4, 1) or values.shape[-1] != 1 \
            or min(values.shape[2:5]) <= 1 or not np.isfinite(values).all() or (values < 0).any() \
            or not np.isfinite(requested).all() or (requested <= 0).any():
        raise RuntimeError("空间四控制缺少有限、非负且逐位置可核验的实际权重")
    if not (values.sum(axis=0) <= 1.0 + SPATIAL_WEIGHT_RTOL).all():
        raise RuntimeError("上游空间四控制权重之和超过1")
    if expected_spatial is None or np.asarray(expected_spatial).shape != values.shape \
            or not np.isfinite(expected_spatial).all() or not np.allclose(
                values, expected_spatial, rtol=SPATIAL_WEIGHT_RTOL, atol=SPATIAL_WEIGHT_ATOL):
        raise RuntimeError("实际空间控制与源ROI位置、方向或逐帧权重不一致")
    # 固定SHA在每个位置把四分支之和截到1；全白区域的实际值不是请求的标量。
    normalized = requested / max(1.0, float(requested.sum()))
    common = values[0] / requested[0]
    if any(not np.allclose(values[index] / requested[index], common,
                           rtol=SPATIAL_WEIGHT_RTOL, atol=SPATIAL_WEIGHT_ATOL)
           for index in range(1, 4)):
        raise RuntimeError("四分支没有使用同一前景区域或实际比例改变")
    result = {}
    for key, value, maximum in zip(MULTICONTROL_ORDER, values, normalized):
        zero_fraction = float((value == 0).mean())
        if not 0 < zero_fraction < 1 or not np.isclose(value.max(), maximum,
                rtol=SPATIAL_WEIGHT_RTOL, atol=SPATIAL_WEIGHT_ATOL):
            raise RuntimeError("实际控制未释放背景，或全白前景权重偏离上游归一化")
        result[key] = {"weight_min": float(value.min()), "weight_max": float(value.max()),
                       "weight_mean": float(value.mean()), "weight_zero_fraction": zero_fraction,
                       "weight_shape": list(value.shape)}
    return result


def validate_multicontrol_trace(trace: list[dict], expected_weights: dict[str, float] | None = None,
                               spatial_weights=None, expected_spatial=None) -> None:
    if len(trace) != 1 or trace[0].get("branch_order") != MULTICONTROL_ORDER:
        raise RuntimeError("多控制没有记录唯一完整的实际四分支输入")
    branches = trace[0].get("branches", {})
    if spatial_weights is not None and expected_weights is None:
        raise RuntimeError("空间控制必须同时提供预声明标量配方")
    spatial = spatial_control_statistics(spatial_weights, expected_weights, expected_spatial) if spatial_weights is not None \
        and expected_weights is not None else None
    for name in MULTICONTROL_ORDER:
        branch = branches.get(name, {})
        active = expected_weights.get(name, 0) > 0 if expected_weights is not None else name in {"depth", "seg"}
        if not branch.get("finite") or branch.get("latent_nonzero", -1) < 0 \
                or (active and (branch["latent_nonzero"] == 0 or branch.get("weight_max", 0) <= 0)) \
                or (not active and (branch["latent_nonzero"] != 0 or branch.get("weight_max") != 0)):
            raise RuntimeError(f"实际{name}控制分支与depth/seg配方不一致")
        if expected_weights is not None:
            import math

            if spatial is not None:
                expected = spatial[name]
                if branch.get("weight_shape") != expected["weight_shape"] or any(
                        not math.isclose(branch.get(key, float("nan")), expected[key], rel_tol=1e-6, abs_tol=1e-7)
                        for key in ["weight_min", "weight_max", "weight_mean", "weight_zero_fraction"]):
                    raise RuntimeError(f"实际{name}空间权重与保存的网络输入不一致")
                continue
            expected = expected_weights.get(name, 0)
            if any(not math.isclose(branch.get(key, float("nan")), expected, rel_tol=1e-6, abs_tol=1e-7)
                   for key in ["weight_min", "weight_max", "weight_mean"]):
                raise RuntimeError(f"实际{name}控制权重与请求不一致")


def validate_image_context_trace(calls: list[dict]) -> None:
    active = [call for call in calls if call.get("input_present") is True]
    if not active or any(not call.get("embedding_finite") for call in calls) \
            or any(call.get("embedding_nonzero", 0) <= 0 for call in active) \
            or any(call.get("embedding_nonzero") != 0 for call in calls if not call.get("input_present")):
        raise RuntimeError("图像参考条件分支缺少非零有限embedding，或CFG无条件分支不符合零编码约定")


def validate_generation_spec(spec: dict, model: str, steps: int) -> None:
    if model not in {"edge", "edge/distilled", *MULTICONTROL_RECIPES} or type(steps) is not int or steps < 1:
        raise ValueError("只支持已固定的edge或depth/seg配方及正整数步数")
    if spec.get("max_frames") != 93 or spec.get("num_video_frames_per_chunk") != 93:
        raise ValueError("生成必须使用单个93帧窗口，禁止隐式补帧或截断")
    controls = [key for key in ["edge", "vis", "depth", "seg"] if spec.get(key) is not None]
    expected_controls = list(MULTICONTROL_RECIPES[model]) if model in MULTICONTROL_RECIPES else ["edge"]
    if controls != expected_controls:
        raise ValueError("控制参数必须与实际edge或depth/seg模型一致，禁止静默丢弃")
    if model.startswith("multicontrol") and (spec.get("image_context_path") is not None or any(
            not spec[key].get("control_path") or spec[key].get("control_weight", 1.0) != MULTICONTROL_RECIPES[model][key]
            for key in expected_controls)):
        raise ValueError("多控制必须逐分支匹配所选具名配方，使用预采样控制且无图像参考")
    if model.startswith("multicontrol"):
        masked = model == "multicontrol-roi-background"
        if any((not spec[key].get("mask_path") if masked else spec[key].get("mask_path") is not None)
               or spec[key].get("mask_prompt") is not None for key in expected_controls):
            raise ValueError("区域四控制必须显式提供各分支mask_path；其他具名配方禁止隐式更换控制区域")
    # 固定SHA的两个采样器都忽略传入的sigma_max；参数校验通过不等于推理使用了它。
    if spec.get("sigma_max") is not None:
        raise ValueError("当前固定版本edge采样器未使用sigma_max，禁止用无效参数声称低噪声生成")
    if spec.get("context_frame_index") is not None:
        raise ValueError("本入口用已核验首帧PNG，禁止context_frame_index覆盖实际图像参考路径")
    if model == "edge/distilled":
        if spec.get("image_context_path") is not None:
            raise ValueError("源首帧图像参考只在本入口的base edge模型中核验")
        if steps != 4:
            raise ValueError("蒸馏edge必须使用经过验证的四步配方")
        if spec.get("guided_generation_mask") is not None:
            raise ValueError("当前蒸馏采样器忽略guided_generation_mask；请使用base edge单独验证")
        if spec.get("guidance", 3) != 3:
            raise ValueError("蒸馏模型的CFG已固化，修改guidance不生效")
    if spec.get("guided_generation_mask") is not None:
        threshold = spec.get("guided_generation_step_threshold", 25)
        if type(threshold) is not int or not 0 <= threshold < steps:
            raise ValueError("空间引导步阈值必须落在本次采样的零基步号范围内")


def validate_bound_source_video(spec: dict, evidence: dict, directory: pathlib.Path) -> None:
    video = pathlib.Path(spec.get("video_path", "")).resolve(strict=True)
    expected = (directory / "source.mp4").resolve(strict=True)
    if video != expected or not video.is_relative_to(directory) \
            or sha256_file(video) != evidence.get("files_sha256", {}).get("source.mp4"):
        raise RuntimeError("实际RGB输入必须绑定同一成功源视频与监督标签")
    preparation_path = directory / "probe.json"
    if preparation_path.exists():
        preparation = json.loads(preparation_path.read_text())
        if preparation.get("status") != "passed" or any(
                preparation.get("artifacts_sha256", {}).get(name) != sha256_file(directory / name)
                for name in ["spec.json", "verified_input.json"]):
            raise RuntimeError("生成输入偏离准备阶段已记录的配方或成功源摘要")
    else:
        raise RuntimeError("生成源缺少准备阶段的完整运行记录")
    if sha256_file(directory / "source_labels.npz") != evidence.get("labels_sha256"):
        raise RuntimeError("生成源的动作与状态监督标签已经改变")


def checked_components(model: str = "edge/distilled") -> dict:
    required = ["Cosmos-Transfer2.5-2B", "Cosmos-Predict2.5-2B", "Cosmos-Reason1-7B",
                "Cosmos-Guardrail1", "Qwen3Guard-Gen-0.6B", "siglip-so400m-patch14-384"]
    if model == "edge":
        required.append("siglip2-so400m-patch16-naflex")
    manifests = {}
    for path in sorted(ROOT.glob("runs/p0/model_download_*/model_manifest.json")):
        record = json.loads(path.read_text())
        name = pathlib.Path(record["path"]).name
        # 同目录允许分次下载不同variant；不能用最后一次的base清单替代原蒸馏权重清单。
        if name == "Cosmos-Transfer2.5-2B" and TRANSFER_WEIGHTS[model] not in {row["file"] for row in record["files"]}:
            continue
        manifests[name] = (path, record)
    result = {}
    for name in required:
        if name not in manifests:
            raise RuntimeError(f"辅助模型尚未下载校验完成：{name}")
        path, record = manifests[name]
        local_root = pathlib.Path(record["path"]).resolve()
        if local_root != ROOT / "checkpoints/base" / name:
            raise RuntimeError("模型清单目录不在项目指定位置")
        for item in record["files"]:
            file = (local_root / item["file"]).resolve()
            if not file.is_relative_to(local_root) or not file.is_file() or file.stat().st_size != item["bytes"]:
                raise RuntimeError(f"本地模型文件不符合下载清单：{file}")
            with file.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != item["sha256"]:
                raise RuntimeError(f"本地模型SHA256不符：{file}")
        result[name] = {"manifest": str(path), "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "path": str(local_root), "repo": record.get("mirror_repo", record.get("repo")),
                        "revision": record["revision"], "backend": record.get("backend", "huggingface")}
    return result


def generate(run_dir: pathlib.Path, args: argparse.Namespace) -> dict:
    ensure_gpu_idle()
    spec = json.loads(args.spec.read_text())
    steps = args.num_steps if args.num_steps is not None else (4 if args.model == "edge/distilled" else 35)
    spec["resolution"] = args.resolution
    spec["num_steps"] = steps
    spec["seed"] = args.seed
    validate_generation_spec(spec, args.model, steps)
    multicontrol = args.model.startswith("multicontrol")
    active_controls = list(MULTICONTROL_RECIPES[args.model]) if multicontrol else ["edge"]
    vis_control = "vis" in active_controls
    variant = "base/" + "+".join(active_controls) if multicontrol else "distilled/edge" if args.model.endswith("/distilled") else "base/edge"
    components = checked_components("edge" if multicontrol else args.model)
    control_checkpoints = {key: checked_control_checkpoint(key) for key in MULTICONTROL_ORDER} if multicontrol else {}
    if multicontrol:
        components["multicontrol_checkpoints"] = control_checkpoints
    base = ROOT / "checkpoints/base"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["COSMOS_EXPERIMENTAL_CHECKPOINTS"] = "1"
    os.environ["RSI_TRANSFER_ONLY"] = "1"
    os.environ["RSI_QWEN3_GUARD_PATH"] = str(base / "Qwen3Guard-Gen-0.6B")
    os.environ["RSI_SIGLIP_PATH"] = str(base / "siglip-so400m-patch14-384")
    if args.model in {"edge", "multicontrol"}:
        os.environ["RSI_SIGLIP2_PATH"] = str(base / "siglip2-so400m-patch16-naflex")
    repo = ROOT / "third_party/cosmos-transfer2.5"
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    import torch
    from cosmos_oss.init import cleanup_environment, init_environment
    from cosmos_transfer2._src.imaginaire.auxiliary.guardrail.common.core import GUARDRAIL1_CHECKPOINT
    from cosmos_transfer2.config import InferenceArguments, SetupArguments
    from cosmos_transfer2.inference import Control2WorldInference

    GUARDRAIL1_CHECKPOINT._path = str(base / "Cosmos-Guardrail1")
    source_evidence = None
    reference_evidence = None
    alignment_path = args.spec.parent / "alignment.json"
    if alignment_path.exists():
        evidence = json.loads(alignment_path.read_text())
        replay = evidence.get("replay", {})
        if replay.get("generation_spec") == str(args.spec) and replay.get("env_success") is True:
            source_evidence = {"path": str(alignment_path), "sha256": hashlib.sha256(alignment_path.read_bytes()).hexdigest(),
                               "success": True, "success_source": "本机MuJoCo完整示范回放的环境真值"}
    verified_input = args.spec.parent / "verified_input.json"
    if verified_input.exists():
        evidence = json.loads(verified_input.read_text())
        validate_bound_source_video(spec, evidence, args.spec.parent)
        cache_manifest = pathlib.Path(evidence["cache_manifest"])
        if hashlib.sha256(cache_manifest.read_bytes()).hexdigest() != evidence["cache_manifest_sha256"]:
            raise RuntimeError("生成源D0清单已改变")
        source = json.loads(cache_manifest.read_text())
        episode = next(row for row in source["episodes"] if row["episode_id"] == evidence["source_episode_id"])
        if not episode["env_success"] or episode["split"] != "demo_train" or not evidence["replay_validation"]["passed"] or evidence["spec"] != str(args.spec):
            raise RuntimeError("生成源不是已核验成功且属于训练集的示范")
        for name, expected in evidence["files_sha256"].items():
            with (args.spec.parent / name).open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                    raise RuntimeError("生成输入的视频或掩码已改变")
        source_evidence = {"path": str(verified_input), "sha256": hashlib.sha256(verified_input.read_bytes()).hexdigest(),
                           "success": True, "success_source": "本机D0完整回放成功真值及窗口逐帧重放验证",
                           "mask_source": "sim_seg", "roi_names": evidence["roi_names"],
                           "guidance_mask_transform": evidence.get("mask_transform")}
        if evidence.get("control_recipe") == "table-only-edge-distilled-v2":
            from p0_table_background import checked_table_source
            checked_table_source(verified_input)
        if spec.get("guided_generation_mask") is not None:
            guided_mask = pathlib.Path(spec["guided_generation_mask"]).resolve(strict=True)
            if guided_mask != (args.spec.parent / "mask.mp4").resolve() or "mask.mp4" not in evidence["files_sha256"]:
                raise RuntimeError("空间引导必须使用已验证成功源绑定且逐文件核验的mask.mp4")
        if spec.get("image_context_path") is not None:
            reference = pathlib.Path(spec["image_context_path"]).resolve(strict=True)
            reference_evidence = evidence.get("image_reference")
            if not reference_evidence or reference != (args.spec.parent / "source_first_frame.png").resolve() \
                    or reference_evidence.get("source_frame") != 0 \
                    or sha256_file(reference) != reference_evidence.get("png_sha256") \
                    or reference.name not in evidence["files_sha256"]:
                raise RuntimeError("参考图必须绑定同一已核验成功源的解码首帧")
    elif spec.get("guided_generation_mask") is not None:
        raise RuntimeError("空间引导缺少逐帧成功源及掩码校验证据")
    if multicontrol and (source_evidence is None or not evidence.get("control_replay_validation", {}).get("passed")
            or evidence.get("generation_variant") != variant
            or not {"metric_depth.npy", *(f"{key}.mp4" for key in active_controls)} <= evidence["files_sha256"].keys()
            or any(pathlib.Path(spec[key]["control_path"]).resolve() != args.spec.parent / f"{key}.mp4"
                   for key in active_controls)):
        raise RuntimeError("depth/seg必须绑定同源RGB/标签及同次传感器观测的逐帧复现证据")
    if vis_control and (not evidence.get("vis_control", {}).get("decoded_pixels_match")
            or evidence["vis_control"].get("source_sha256") != evidence["files_sha256"]["source.mp4"]
            or evidence["vis_control"].get("sample_count") != 93
            or evidence["vis_control"].get("upstream_sha") != "2ff49d0717af02057ae79bc75c00fbff9da1b4e7"):
        raise RuntimeError("vis控制缺少同源、全帧及固定上游变换证据")
    if args.model == "multicontrol-balanced" and (evidence.get("control_recipe") != args.model
            or evidence.get("target_condition") != {"light": "warm_dim_indoor"}):
        raise RuntimeError("同比缩放试验须使用生成前声明目标及配方的新源证据")
    if args.model == "multicontrol-low-vis-background" and (source_evidence is None
            or evidence.get("control_recipe") != args.model
            or evidence.get("target_condition") != {"background": "matte_dark_wood"}
            or evidence.get("intervention") != {"kind": "vis_control_weight_reduction", "branch": "vis",
                "before": .5, "after": .1, "prompt_and_geometry_controls_unchanged": True}
            or evidence.get("vis_control", {}).get("control_weight") != .1
            or evidence.get("comparison_baseline", {}).get("intervention_kind") != "vis_control_weight_reduction"
            or spec.get("num_steps") != 35 or spec.get("seed") != 42 or str(spec.get("resolution")) != "480"):
        raise RuntimeError("低vis背景试验必须绑定同源比较基线和生成前声明的唯一权重干预")
    spatial_controls = args.model == "multicontrol-roi-background"
    if spatial_controls and (source_evidence is None or evidence.get("control_recipe") != args.model
            or evidence.get("target_condition") != {"background": "matte_dark_wood"}
            or evidence.get("intervention", {}).get("kind") != "native_control_roi_mask"
            or spec.get("num_steps") != 35 or spec.get("seed") != 42 or str(spec.get("resolution")) != "480"
            or "mask.mp4" not in evidence["files_sha256"] or any(
                pathlib.Path(spec[key]["mask_path"]).resolve(strict=True) != args.spec.parent / "mask.mp4"
                for key in active_controls)):
        raise RuntimeError("区域控制须绑定生成前预声明的新背景源及已核验同帧ROI")
    if spec.get("image_context_path") is not None and (args.model != "edge" or reference_evidence is None):
        raise RuntimeError("图像参考仅在本入口已核验成功源的base edge配方中使用")
    sample = InferenceArguments.model_validate(spec)
    dit = base / "Cosmos-Transfer2.5-2B" / TRANSFER_WEIGHTS["edge" if multicontrol else args.model]
    setup = SetupArguments(output_dir=run_dir / "output", model="edge" if multicontrol else args.model, checkpoint_path=str(dit),
                           disable_guardrails=False, offload_guardrail_models=False, keep_going=False)
    if not setup.enable_guardrails or sample.num_steps != steps or sample.num_video_frames_per_chunk != 93:
        raise RuntimeError("生成参数必须启用安全检查且保持约定的窗口与步数")
    overrides = [f"++model.config.tokenizer.vae_pth={base / 'Cosmos-Predict2.5-2B/tokenizer.pth'}",
                 f"model.config.text_encoder_config.ckpt_path={base / 'Cosmos-Reason1-7B'}",
                 f"++model.config.text_encoder_config.model_config.tokenizer.cache_dir={base / 'Cosmos-Reason1-7B'}",
                 "++model.config.text_encoder_config.model_config.model_config.attn_implementation=sdpa",
                 "model.config.net.atten_backend=torch"]
    (run_dir / "setup.json").write_text(setup.model_dump_json(indent=2) + "\n")
    (run_dir / "sample.json").write_text(sample.model_dump_json(indent=2) + "\n")
    (run_dir / "components.json").write_text(json.dumps(components, ensure_ascii=False, indent=2) + "\n")
    init_environment()
    try:
        before = time.perf_counter()
        inference = Control2WorldInference(setup, batch_hint_keys=active_controls,
                                          experiment_opts=overrides,
                                          **({"checkpoint_paths": [control_checkpoints[key]["path"]
                                               for key in MULTICONTROL_ORDER]} if multicontrol else {}))
        control_trace, control_hooks, control_weight_arrays, expected_control_arrays = [], [], [], []
        if spatial_controls:
            import cv2
            from cosmos_transfer2._src.transfer2.inference.utils import read_and_resize_input

            requested_mask, mask_fps, _, _ = read_and_resize_input(
                str(args.spec.parent / "mask.mp4"), resolution=args.resolution, interpolation=cv2.INTER_LINEAR)
            if requested_mask.shape[1] != 93 or mask_fps != 20:
                raise RuntimeError("原生ROI读取改变帧数或采样时序")
            expected_roi = (requested_mask[0].cpu().numpy() > 127.5)
        if multicontrol:
            def observe_controls(module, inputs, kwargs):
                if control_trace:
                    return
                import inspect

                values = inspect.signature(module.forward).bind_partial(*inputs, **kwargs).arguments
                latent = values["latent_control_input"]
                scale = values["control_context_scale"]
                if module.num_control_branches != 4:
                    raise RuntimeError("原生多控制必须按edge/vis/depth/seg加载四分支")
                chunks = latent.chunk(4, dim=1)
                branches = {}
                for index, (key, chunk) in enumerate(zip(MULTICONTROL_ORDER, chunks)):
                    weight = torch.as_tensor(scale[index], device=chunk.device).float()
                    branches[key] = {"latent_shape": list(chunk.shape),
                                     "latent_nonzero": int(torch.count_nonzero(chunk).item()),
                                     "finite": bool(torch.isfinite(chunk).all() and torch.isfinite(weight).all()),
                                     "weight_min": float(weight.min()), "weight_max": float(weight.max()),
                                     "weight_mean": float(weight.mean())}
                if spatial_controls:
                    import numpy as np

                    actual = torch.stack([torch.as_tensor(scale[index]).detach().float().cpu()
                                          for index in range(4)]).numpy()
                    expected = expected_spatial_control_weights(expected_roi, actual.shape, MULTICONTROL_RECIPES[args.model])
                    statistics = spatial_control_statistics(actual, MULTICONTROL_RECIPES[args.model], expected)
                    for key in MULTICONTROL_ORDER:
                        branches[key].update(statistics[key])
                    np.save(run_dir / "native_control_weights.npy", actual, allow_pickle=False)
                    np.save(run_dir / "expected_control_weights.npy", expected, allow_pickle=False)
                    control_weight_arrays.append(actual)
                    expected_control_arrays.append(expected)
                control_trace.append({"branch_order": MULTICONTROL_ORDER, "branches": branches})

            control_hooks.append(inference.inference_pipeline.model.net.register_forward_pre_hook(
                observe_controls, with_kwargs=True))
        actual_backends = Counter()
        for name, module in inference.inference_pipeline.model.net.named_modules():
            if hasattr(module, "backend") and hasattr(module, "attn_op"):
                actual_backends[module.backend] += 1
                if module.backend != "torch":
                    raise RuntimeError(f"实际注意力分支未遵守torch配置：{name}={module.backend}")
        if not actual_backends:
            raise RuntimeError("没有发现可核验的实际注意力分支")
        guidance_weights = []
        if spec.get("guided_generation_mask") is not None:
            pipeline = inference.inference_pipeline
            original_weight_map = pipeline.construct_latent_weight_map

            def observe_weight_map(*positional, **keywords):
                weights = original_weight_map(*positional, **keywords)
                # 只复制一个相同的通道用于诊断，返回原Tensor，不干预推理或像素。
                import numpy as np

                path = run_dir / f"native_guidance_weights_{len(guidance_weights):02d}.npy"
                values = weights[:, 0].detach().float().cpu().numpy()
                np.save(path, values, allow_pickle=False)
                guidance_weights.append({"shape": list(weights.shape), "saved_channel": 0,
                                         "file": path.name, "sha256": sha256_file(path),
                                         "min": float(values.min()), "max": float(values.max()),
                                         "mean": float(values.mean()),
                                         "fraction_ge_0_99": float((values >= 0.99).mean())})
                return weights

            pipeline.construct_latent_weight_map = observe_weight_map
        context_calls, context_hooks = [], []
        if reference_evidence is not None:
            def observe_context(module, inputs, kwargs, output):
                context = kwargs.get("image_context", inputs[1] if len(inputs) > 1 else None)
                embedding = output["img_context_emb"].detach()
                context_calls.append({"input_present": context is not None,
                                      "input_shape": list(context.shape) if context is not None else None,
                                      "embedding_shape": list(embedding.shape),
                                      "embedding_nonzero": int(torch.count_nonzero(embedding).item()),
                                      "embedding_finite": bool(torch.isfinite(embedding).all().item())})

            for module in inference.inference_pipeline.model.conditioner.modules():
                if module.__class__.__name__ == "SigLip2EmbImgContext":
                    context_hooks.append(module.register_forward_hook(observe_context, with_kwargs=True))
            if not context_hooks:
                raise RuntimeError("实际base模型没有可核验的SigLIP2图像参考分支")
        torch.cuda.synchronize()
        load_s = time.perf_counter() - before
        dtype_counts = Counter()
        for parameter in inference.inference_pipeline.model.net.parameters():
            dtype_counts[str(parameter.dtype)] += parameter.numel()
        before = time.perf_counter()
        outputs = inference.generate([sample], output_dir=setup.output_dir)
        torch.cuda.synchronize()
        generate_s = time.perf_counter() - before
        (run_dir / "generation_timing.json").write_text(json.dumps(
            {"load_wall_s": load_s, "generate_wall_s": generate_s}, indent=2) + "\n")
        for hook in context_hooks:
            hook.remove()
        for hook in control_hooks:
            hook.remove()
        (run_dir / "image_context_trace.json").write_text(json.dumps(context_calls, indent=2) + "\n")
        (run_dir / "guidance_weight_trace.json").write_text(json.dumps(guidance_weights, indent=2) + "\n")
        (run_dir / "multicontrol_trace.json").write_text(json.dumps(control_trace, indent=2) + "\n")
        if multicontrol:
            validate_multicontrol_trace(control_trace, {key: spec[key].get("control_weight", 1.0) for key in active_controls},
                                       control_weight_arrays[0] if spatial_controls and control_weight_arrays else None,
                                       expected_control_arrays[0] if spatial_controls and expected_control_arrays else None)
            if spatial_controls and len(control_weight_arrays) != 1:
                raise RuntimeError("区域控制缺少唯一实际网络权重数组")
        if reference_evidence is not None:
            validate_image_context_trace(context_calls)
        if len(outputs) != 1 or not pathlib.Path(outputs[0]).is_file():
            raise RuntimeError("没有产生通过安全检查的单个生成视频")
        video_info = json.loads(subprocess.check_output(
            ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
             "stream=width,height,nb_read_frames,r_frame_rate", "-of", "json", outputs[0]], text=True))
        if int(video_info["streams"][0]["nb_read_frames"]) != 93:
            raise RuntimeError("生成视频不是93帧，不能视为逐帧对齐通过")
        return {"model": "Cosmos-Transfer2.5-2B", "variant": variant, "recipe": args.model,
                "num_steps": steps,
                "resolution_requested": args.resolution, "video_info": video_info, "outputs": outputs,
                "batch": 1, "sample_count": 1, "seed": args.seed, "precision": "上游默认bf16，DiT参数dtype另列",
                "dit_parameter_dtypes": dict(dtype_counts), "load_wall_s": load_s, "generate_wall_s": generate_s,
                "generation_timing_includes": "文本安全、文本编码、VAE、扩散、视频安全和写文件，不含模型加载",
                "guardrails_enabled": True, "offline": True, "components": components,
                "attention_backends": {"reason1": "sdpa", "dit": "torch", "dit_actual_module_counts": dict(actual_backends)},
                "experiment_overrides": overrides, "source_spec": str(args.spec),
                "control_scope": "robot_and_targets_roi_all_four_controls" if spatial_controls else
                "full_frame_edge+blur+depth+sim_roi_color_seg" if "edge" in active_controls and vis_control else
                "full_frame_blur+depth+sim_roi_color_seg" if vis_control else "full_frame_depth+sim_roi_color_seg" if multicontrol else
                "all_non_table_pixels_with_eroded8_table_interior" if source_evidence is not None and evidence.get("control_recipe") == "table-only-edge-distilled-v2" else
                "robot_and_targets_mask" if spec["edge"].get("mask_path") else "full_frame_edge",
                "multicontrol_trace": control_trace,
                "native_control_weight_evidence": {"file": "native_control_weights.npy",
                    "sha256": sha256_file(run_dir / "native_control_weights.npy"),
                    "expected_file": "expected_control_weights.npy",
                    "expected_sha256": sha256_file(run_dir / "expected_control_weights.npy"),
                    "source_roi_position_verified": True,
                    "numerical_tolerance": {"rtol": SPATIAL_WEIGHT_RTOL, "atol": SPATIAL_WEIGHT_ATOL,
                                            "reason": "仅容纳bf16控制插值与归一化舍入；不改变生成质量门"},
                    "requested_raw_weights": MULTICONTROL_RECIPES[args.model],
                    "normalization": "固定上游按时空位置使四分支之和不超过1；前景白区除以2.7，背景为0",
                    "mask_sha256": evidence["files_sha256"]["mask.mp4"]} if spatial_controls else None,
                "vis_control": evidence.get("vis_control") if vis_control else None,
                "spatial_guidance": {"enabled": spec.get("guided_generation_mask") is not None,
                                     "mask": spec.get("guided_generation_mask"),
                                     "step_threshold": spec.get("guided_generation_step_threshold"),
                                     "mask_transform": source_evidence.get("guidance_mask_transform") if source_evidence else None,
                                     "native_weight_trace": guidance_weights},
                "source_success_evidence": source_evidence,
                "image_reference": reference_evidence, "image_context_trace": context_calls,
                "eligible_for_training": False, "quality_gate": "尚未做机器人/物体ROI几何与时序质量门",
                "torch_peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9}
    finally:
        cleanup_environment()


def sha256_file(path: pathlib.Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spec", type=pathlib.Path, required=True)
    parser.add_argument("--resolution", choices=["480", "720"], default="480")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", choices=["edge", "edge/distilled", *MULTICONTROL_RECIPES], default="edge/distilled")
    parser.add_argument("--num-steps", type=int, help="默认蒸馏4步，base35步；后者仅为质量诊断，单独报告耗时")
    args = parser.parse_args()
    args.spec = args.spec.resolve(strict=True)
    return run_stage("transfer_smoke", lambda directory: generate(directory, args),
                     {k: str(v) if isinstance(v, pathlib.Path) else v for k, v in vars(args).items()})


if __name__ == "__main__":
    raise SystemExit(main())
