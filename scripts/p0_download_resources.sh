#!/usr/bin/env bash
# p0_download_resources.sh — 只下载生成必需组件，逐个保存固定版本与校验记录。
#
# @author pony
# @date 2026-11-06
# @version v1.0.0
# @last_modified 2026-11-06
# @changelog
#   - v1.0.0 (2026-11-06): 串行下载VAE、完整安全组件和Reason1文本编码器。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/scripts/thor_env.sh"
export PATH="${ROOT}/envs/policy/bin:${PATH}"
for repo in nvidia/Cosmos-Predict2.5-2B nvidia/Cosmos-Guardrail1 Qwen/Qwen3Guard-Gen-0.6B google/siglip-so400m-patch14-384 nvidia/Cosmos-Reason1-7B; do
  "${ROOT}/envs/policy/bin/python" "${ROOT}/scripts/p0_download_model.py" --backend modelscope --repo "$repo"
done
