#!/usr/bin/env bash
# p0_setup_sim.sh — 在独立环境安装 LIBERO 最小仿真依赖。
#
# @author pony
# @date 2026-11-06
# @version v1.0.0
# @last_modified 2026-11-06
# @changelog
#   - v1.0.0 (2026-11-06): 分离仿真依赖，不运行上游整表安装脚本。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export UV_CACHE_DIR="${ROOT}/.cache/uv"
export GIT_LFS_SKIP_SMUDGE=1

RSI_ENV_NAME=sim bash "${ROOT}/scripts/p0_bootstrap.sh"
if [[ ! -d "${ROOT}/third_party/LIBERO/.git" ]]; then
  git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git "${ROOT}/third_party/LIBERO"
fi
if [[ ! -d "${ROOT}/third_party/LIBERO-plus/.git" ]]; then
  git clone --depth 1 https://github.com/sylvestf/LIBERO-plus.git "${ROOT}/third_party/LIBERO-plus"
fi

"${ROOT}/envs/policy/bin/python" -m uv pip install \
  --python "${ROOT}/envs/sim/bin/python" \
  -r "${ROOT}/configs/p0_sim_requirements.txt"
# LIBERO 与 plus 同包名；不把两份代码同时安装到 site-packages。
# 冒烟入口会明确指定 PYTHONPATH 和配置目录，避免写入用户家目录。
RSI_ENV_NAME=sim bash "${ROOT}/scripts/p0_bootstrap.sh"
echo '最小仿真依赖准备完成；尚未声称环境回合已跑通。'
