#!/usr/bin/env bash
# p0_setup_transfer.sh — 在独立副本中安装官方CUDA13生成环境。
#
# @author pony
# @date 2026-11-06
# @version v1.3.0
# @last_modified 2026-11-06
# @changelog
#   - v1.3.0 (2026-11-06): 对本机wheel传输补丁做完整TOML语义核验，避免补丁上下文冲突。
#   - v1.2.0 (2026-11-06): 大型ARM CUDA轮子增加下载超时，限制并发以减少连接饥饿。
#   - v1.1.0 (2026-11-06): 已有传输镜像补丁时使用对应registry，保留官方冻结锁定。
#   - v1.0.0 (2026-11-06): 保留policy环境，按上游cu130锁文件安装独立生成栈。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/scripts/thor_env.sh"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"
export UV_HTTP_CONCURRENT_DOWNLOADS="${UV_HTTP_CONCURRENT_DOWNLOADS:-4}"
export UV_HTTP_RETRIES="${UV_HTTP_RETRIES:-1}"
RSI_ENV_NAME=transfer bash "${ROOT}/scripts/p0_bootstrap.sh"
export VIRTUAL_ENV="${ROOT}/envs/transfer"
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
INDEX_ARGS=()
if [[ -f "${ROOT}/patches/cosmos-transfer2.5/0004-pypi-transport-mirror.patch" ]]; then
  # 后续wheel补丁会改变前一补丁的上下文；核验最终锁文件语义，不能把上下文冲突当作依赖变更。
  "${ROOT}/envs/policy/bin/python" - "${ROOT}" <<'PY'
import json, pathlib, subprocess, sys, tomllib, urllib.parse
root = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(root / 'scripts'))
from p0_transfer_mirror import translated
repo = root / 'third_party/cosmos-transfer2.5'
original = tomllib.loads(subprocess.check_output(['git', '-C', str(repo), 'show', 'HEAD:uv.lock'], text=True))
expected = translated(original)
current = tomllib.loads((repo / 'uv.lock').read_text())
manifests = sorted(root.glob('runs/p0/cuda_wheel_prefetch_*/wheel_manifest.json'))
mapping = {}
if manifests:
    for item in json.loads(manifests[-1].read_text()):
        mapping['http://127.0.0.1:8955/' + urllib.parse.quote(pathlib.Path(item['path']).name, safe='')] = item['official_url']
def restore(value):
    if isinstance(value, str): return mapping.get(value, value)
    if isinstance(value, list): return [restore(v) for v in value]
    if isinstance(value, dict): return {k: restore(v) for k, v in value.items()}
    return value
if restore(current) != expected:
    raise RuntimeError('锁文件与官方依赖不一致；只允许已记录的PyPI镜像和本机wheel传输地址')
print('官方锁文件语义验证通过：包版本、marker、依赖及文件哈希保持一致')
PY
  INDEX_ARGS=(--default-index https://mirrors.aliyun.com/pypi/simple)
fi
# cu130包含匹配Torch2.9的编译扩展，不能与policy的Torch2.11混装。
"${ROOT}/envs/policy/bin/python" -m uv sync \
  --project "${ROOT}/third_party/cosmos-transfer2.5" \
  --active --inexact --frozen --no-default-groups --extra cu130 "${INDEX_ARGS[@]}"
"${VIRTUAL_ENV}/bin/python" -c 'import importlib.metadata as m; print({n:m.version(n) for n in ["torch","torchvision","transformers","cosmos-oss"]})'
