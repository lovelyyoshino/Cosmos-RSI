#!/usr/bin/env bash
# thor_env.sh — 项目缓存与 CUDA 路径，仅影响当前进程。
#
# @author pony
# @date 2026-11-06
# @version v1.4.0
# @last_modified 2026-11-06
# @changelog
#   - v1.4.0 (2026-11-06): 拒绝在zsh直接source，避免BASH_SOURCE为空导致缓存路径越界。
#   - v1.3.0 (2026-11-06): 进程内加载项目解包的ImageMagick，不改系统库或配置。
#   - v1.2.0 (2026-11-06): 后续uv解释器安装限定在项目目录。
#   - v1.1.0 (2026-11-06): 为ModelScope镜像下载设置项目专用缓存。
#   - v1.0.0 (2026-11-06): 将下载、渲染和编译缓存限制在项目目录。

if [[ -z "${BASH_VERSION:-}" ]]; then
  echo 'thor_env.sh 需要在 bash 中 source；请用 bash -c 执行项目命令。' >&2
  return 2
fi
RSI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RSI_ROOT
export PATH="${RSI_ROOT}/tools/bin:/usr/local/cuda-13.0/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH:-}"
if [[ -d "${RSI_ROOT}/tools/imagemagick/usr/lib/aarch64-linux-gnu" ]]; then
  export MAGICK_HOME="${RSI_ROOT}/tools/imagemagick/usr"
  export LD_LIBRARY_PATH="${MAGICK_HOME}/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH}"
  export MAGICK_CONFIGURE_PATH="${RSI_ROOT}/tools/imagemagick/etc/ImageMagick-6"
fi
export XDG_CACHE_HOME="${RSI_ROOT}/.cache"
export XDG_CONFIG_HOME="${RSI_ROOT}/.config"
export TMPDIR="${RSI_ROOT}/.cache/tmp"
mkdir -p "$TMPDIR"
export HF_HOME="${RSI_ROOT}/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export UV_CACHE_DIR="${RSI_ROOT}/.cache/uv"
export UV_PYTHON_INSTALL_DIR="${RSI_ROOT}/tools/python"
export MODELSCOPE_CACHE="${RSI_ROOT}/.cache/modelscope"
export MPLCONFIGDIR="${RSI_ROOT}/.cache/matplotlib"
export NUMBA_CACHE_DIR="${RSI_ROOT}/.cache/numba"
export TORCH_HOME="${RSI_ROOT}/.cache/torch"
export TORCHINDUCTOR_CACHE_DIR="${RSI_ROOT}/.cache/torchinductor"
export IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
