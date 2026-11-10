#!/usr/bin/env python3
"""在禁止下载的条件下验证Transfer蒸馏配置，不加载模型权重。

@author pony
@date 2026-11-06
@version v1.1.0
@last_modified 2026-11-06
@changelog
    - v1.1.0 (2026-11-06): 实例化attrs模型配置以检查下划线字段的构造器别名。
    - v1.0.0 (2026-11-06): 验证选定配置无隐式下载且本地权重覆盖完整。
"""
from __future__ import annotations

import os
import pathlib
import sys
from unittest.mock import patch

from p0_common import ROOT, ensure_gpu_idle, run_stage


def probe(run_dir: pathlib.Path) -> dict:
    ensure_gpu_idle()
    repo = ROOT / "third_party/cosmos-transfer2.5"
    base = ROOT / "checkpoints/base"
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["COSMOS_EXPERIMENTAL_CHECKPOINTS"] = "1"
    os.environ["RSI_TRANSFER_ONLY"] = "1"
    from cosmos_transfer2._src.imaginaire.utils.config_helper import override
    from cosmos_transfer2._src.imaginaire.lazy_config import instantiate
    from cosmos_transfer2._src.interactive.configs.registry_transfer2p5 import make_config
    from cosmos_transfer2.config import SetupArguments

    setup = SetupArguments(
        output_dir=run_dir / "unused", model="edge/distilled",
        checkpoint_path=str(base / "Cosmos-Transfer2.5-2B/distilled/general/edge/41f07f13-f2e4-4e34-ba4c-86f595acbc20_ema_bf16.pt"),
        disable_guardrails=False,
    )
    options = ["--", f"experiment={setup.experiment}",
               f"++model.config.tokenizer.vae_pth={base / 'Cosmos-Predict2.5-2B/tokenizer.pth'}",
               f"model.config.text_encoder_config.ckpt_path={base / 'Cosmos-Reason1-7B'}",
               f"++model.config.text_encoder_config.model_config.tokenizer.cache_dir={base / 'Cosmos-Reason1-7B'}",
               "++model.config.text_encoder_config.model_config.model_config.attn_implementation=sdpa",
               "model.config.net.atten_backend=torch"]
    # 配置阶段若仍触发任何下载，必须失败；不能以补下载无关模型掩盖导入副作用。
    with patch("cosmos_transfer2._src.imaginaire.utils.checkpoint_db._hf_download",
               side_effect=RuntimeError("配置阶段禁止下载")) as download:
        config = override(make_config(), options)
        config.validate()
        text_config = instantiate(config.model.config.text_encoder_config.model_config.model_config)
        if text_config._attn_implementation != "sdpa" or config.model.config.net.atten_backend != "torch":
            raise RuntimeError("注意力后端没有按Thor配置生效")
        download.assert_not_called()
    for path in [setup.checkpoint_path, config.model.config.tokenizer.vae_pth,
                 config.model.config.text_encoder_config.ckpt_path,
                 config.model.config.text_encoder_config.model_config.tokenizer.cache_dir]:
        if not pathlib.Path(str(path)).exists():
            raise RuntimeError(f"配置中的本地权重不存在：{path}")
    if not setup.enable_guardrails:
        raise RuntimeError("本地配置必须保持安全检查开启")
    return {"experiment": setup.experiment, "config_file": setup.config_file,
            "config_validated": True, "implicit_download_count": 0,
            "guardrails_enabled": setup.enable_guardrails,
            "attention_backends": {"reason1": text_config._attn_implementation,
                                   "dit": config.model.config.net.atten_backend},
            "teacher_loading": "实际loader在推理时设置load_teacher_weights=False；本检查不初始化模型",
            "precision": "CPU配置检查，无张量计算", "batch": None, "seed": None,
            "steps": 0, "sample_count": 1, "model_weights_loaded": False}


if __name__ == "__main__":
    raise SystemExit(run_stage("transfer_config", probe))
