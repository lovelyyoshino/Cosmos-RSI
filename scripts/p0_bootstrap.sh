#!/usr/bin/env bash
# p0_bootstrap.sh — 在 Thor 项目目录建立独立环境并锁定上游版本。
#
# @author pony
# @date 2026-11-06
# @version v1.2.0
# @last_modified 2026-11-06
# @changelog
#   - v1.2.0 (2026-11-06): 根据逆向检查记录已应用补丁，不把修改副本标为原样。
#   - v1.1.0 (2026-11-06): 支持独立 sim 环境并记录全部已克隆仓库。
#   - v1.0.0 (2026-11-06): 复制验证环境，不覆盖原环境或安装整表依赖。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_ENV="${RSI_BASE_ENV:-/home/thor/cosmos_validation/venv-cosmos3}"
ENV_NAME="${RSI_ENV_NAME:-policy}"
if [[ ! "$ENV_NAME" =~ ^[a-z][a-z0-9_-]*$ ]]; then
  echo '环境名必须是小写字母、数字、下划线或连字符。' >&2
  exit 2
fi
TARGET_ENV="${ROOT}/envs/${ENV_NAME}"
export GIT_LFS_SKIP_SMUDGE=1

if [[ "$(uname -m)" != "aarch64" || "$ROOT" != /home/thor/rsi ]]; then
  echo '本脚本只允许在 Thor 的 /home/thor/rsi 执行。' >&2
  exit 2
fi
if [[ ! -x "${BASE_ENV}/bin/python" ]]; then
  echo "原环境不可用：${BASE_ENV}" >&2
  exit 2
fi

mkdir -p "${ROOT}/envs" "${ROOT}/third_party" "${ROOT}/data" "${ROOT}/checkpoints"
if [[ ! -d "$TARGET_ENV" ]]; then
  # 禁止硬链接：后续 pip 写入不能回写已有验证环境。
  cp -a --reflink=auto "$BASE_ENV" "${TARGET_ENV}.copying"
  python3 - "$BASE_ENV" "${TARGET_ENV}.copying" "$TARGET_ENV" <<'PY'
import pathlib
import sys

source, copying, target = map(pathlib.Path, sys.argv[1:])
for path in [*copying.joinpath('bin').iterdir(), copying / 'pyvenv.cfg']:
    if path.is_symlink() or not path.is_file():
        continue
    data = path.read_bytes()
    if b'\0' not in data and str(source).encode() in data:
        path.write_bytes(data.replace(str(source).encode(), str(target).encode()))
copying.rename(target)
PY
fi
"${TARGET_ENV}/bin/python" -c 'import sys; print("独立环境：", sys.prefix, flush=True)'

clone_repo() {
  local name="$1" branch="$2" url="$3"
  if [[ ! -d "${ROOT}/third_party/${name}/.git" ]]; then
    git clone --depth 1 --branch "$branch" "$url" "${ROOT}/third_party/${name}"
  fi
  git -C "${ROOT}/third_party/${name}" rev-parse HEAD
}

clone_repo starVLA starVLA https://github.com/starVLA/starVLA.git
clone_repo cosmos-transfer2.5 main https://github.com/nvidia-cosmos/cosmos-transfer2.5.git
clone_repo Isaac-GR00T main https://github.com/NVIDIA/Isaac-GR00T.git

python3 - "$ROOT" <<'PY'
import datetime
import pathlib
import subprocess
import sys

root = pathlib.Path(sys.argv[1])
rows = []
for repo in sorted((root / 'third_party').iterdir()):
    if not (repo / '.git').is_dir():
        continue
    def git(*args):
        return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()
    applied = []
    for patch in sorted((root / 'patches' / repo.name).glob('*.patch')):
        check = subprocess.run(['git', '-C', str(repo), 'apply', '--reverse', '--check', str(patch)], capture_output=True)
        if check.returncode == 0:
            applied.append(f'`patches/{repo.name}/{patch.name}`')
    dirty = bool(git('diff', '--name-only', 'HEAD'))
    patch_status = '、'.join(applied) if applied else ('存在未归档修改，需核查' if dirty else '无')
    rows.append((git('remote', 'get-url', 'origin'), git('branch', '--show-current'), git('rev-parse', 'HEAD'), patch_status))
date = datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()
lines = ['# 上游版本锁定', '', f'锁定时间：{date}；机器：Thor。已应用补丁见表，LFS 大文件按需下载。', '',
         '| 仓库 | 分支 | SHA | 克隆日期 | 本地 patch |', '| --- | --- | --- | --- | --- |']
lines.extend(f'| {url} | {branch} | `{sha}` | {date} | {patch_status} |' for url, branch, sha, patch_status in rows)
(root / 'third_party' / 'PINS.md').write_text('\n'.join(lines) + '\n')
PY
echo 'P0 基础准备完成；尚未安装上游 requirements 或运行模型。'
