# Transfer2.5 本地适配

基于 `2ff49d0717af02057ae79bc75c00fbff9da1b4e7`。只在项目副本应用，原验证工程不变。

- `0001-local-resource-paths.patch`：保留传入的processor本地路径；通过进程环境选择Qwen3Guard和SigLIP的已校验本地权重，默认值与上游相同。不伪造HF缓存或revision。
- `0002-guardrail-fail-closed.patch`：上游文本判定遇到异常会返回安全；本地改成拒绝，并保留错误日志，避免依赖错误被当成通过。
- `0003-experiment-overrides.patch`：外层API透传可选配置覆盖，供项目传入本地VAE与Reason1路径；未传入时沿用原有行为。
- `0006-transfer-only-registry.patch`：设置项目扩展 `RSI_TRANSFER_ONLY=1` 时，只注册Transfer蒸馏实验，避免无关action-conditioned实验在导入时下载teacher；默认保留上游全量注册。实际推理模型、权重和安全检查不变。
- `0007-siglip2-local-path.patch`：base edge的图像上下文需要独立的SigLIP2；通过`RSI_SIGLIP2_PATH`传入已校验的本地资源目录，默认保持原模型，不替换或关闭该编码器。
- `0008-i2v-attention-backend.patch`：base模型的独立图像query分支遗漏了`backend`透传，导致显式选择torch后仍调用TE融合注意力。补齐同一参数，使图像交叉注意力遵守模型已有配置；保留全部参数与图像条件计算。
- `0009-local-multicontrol-checkpoints.patch`：给多控制入口增加显式本地权重列表；保持edge/vis/depth/seg四分支和上游计算顺序，拒绝缺失文件与蒸馏混用。未传入列表时沿用上游行为，安全检查不变。

在 Thor 的上游目录依次运行 `git apply --check <patch>`、`git apply <patch>`，已应用时用 `git apply --reverse --check` 确认。每次实验记录补丁SHA256与上游工作区diff。安全检查保持启用。
