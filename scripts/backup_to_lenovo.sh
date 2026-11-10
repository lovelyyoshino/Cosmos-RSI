#!/usr/bin/env bash
# backup_to_lenovo.sh — 同步代码、运行元数据、展示图片/视频与已晋升权重。
#
# @author pony
# @date 2026-11-06
# @version v1.9.0
# @last_modified 2026-11-10
# @changelog
#   - v1.9.0 (2026-11-10): 备份机独有的公开仓库Cosmos-RSI与PPT工程目录不参与同步，推送时不被删除。
#   - v1.8.0 (2026-11-09): 备份完整候选退化的固定回合展示视频，继续排除嵌套原始视频。
#   - v1.7.0 (2026-11-09): 备份正式整圈导出的固定回合对比视频，继续排除原始评测视频。
#   - v1.6.1 (2026-11-08): 网络同步压缩重复文本快照与补丁，保持原过滤、硬链接及非破坏拉取。
#   - v1.6.0 (2026-11-07): 可按--p0-run只同步新增阶段，继续保留原过滤与非破坏拉取。
#   - v1.5.0 (2026-11-07): 保留完全相同媒体的硬链接，避免备份重新产生已清理的独立副本。
#   - v1.4.0 (2026-11-07): 备份显式导出的video_materials展示视频，继续排除原始回合视频。
#   - v1.3.0 (2026-11-06): 排除再生图像数组等原始数据，保留冻结初态；拉取不覆盖更新的本地编辑。
#   - v1.2.0 (2026-11-06): 排除runs内未晋升训练权重，保留检查点JSON元数据。
#   - v1.1.0 (2026-11-06): 支持从备份机主动拉取，排除环境、缓存和Git内部状态。
# 把本仓库同步备份到 lenovo 服务器。
# 用法：SSHPASS='<密码>' bash scripts/backup_to_lenovo.sh [--dry-run] [--with-ckpt] [--pull-from-thor]
#   默认：代码、文档、配置、图片、video_materials展示视频及runs元数据，不含原始视频和LeRobot数据。
#   --with-ckpt：同时同步 checkpoints/promoted/（只放已晋升的权重）。
# 密码只通过环境变量 SSHPASS 传入，不要写进文件。
set -euo pipefail

LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SSH_OPTS="-o StrictHostKeyChecking=accept-new"

DRY=""; WITH_CKPT=0; PULL=0
P0_RUNS=()
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY="--dry-run --itemize-changes" ;;
    --with-ckpt) WITH_CKPT=1 ;;
    --pull-from-thor) PULL=1 ;;
    --p0-run)
      [[ "$#" -ge 2 && "$2" =~ ^[a-zA-Z0-9_+.-]+$ && "$2" != . && "$2" != .. ]] || {
        echo '--p0-run只接受单个P0目录名。' >&2; exit 2;
      }
      P0_RUNS+=("$2")
      shift ;;
    *) echo "未知参数：$1" >&2; exit 2 ;;
  esac
  shift
done

if [[ "$PULL" -eq 1 ]]; then
  SRC="${RSI_THOR_SOURCE:?设置 Thor 源目录，例如 thor@<thor-host>:/home/thor/rsi}"
  DEST="${RSI_BACKUP_DEST:?设置备份目标，例如 <user>@<backup-host>:/path/to/cosmos-thor-rsi}"
  # 主动拉取可能与备份机上的编辑并存，禁止按远端状态删除本地文件。
  DELETE=(--update)
else
  SRC="$LOCAL_ROOT"
  DEST="${RSI_BACKUP_DEST:?设置备份目标，例如 <user>@<backup-host>:/path/to/cosmos-thor-rsi}"
  DELETE=(--delete)
fi

if [[ -n "${SSHPASS:-}" ]]; then
  RSH="sshpass -e ssh ${SSH_OPTS}"
else
  RSH="ssh ${SSH_OPTS}"
fi

FILTERS=()
if [[ "${#P0_RUNS[@]}" -gt 0 ]]; then
  FILTERS+=(--include '/runs/' --include '/runs/p0/')
  for run in "${P0_RUNS[@]}"; do FILTERS+=(--include "/runs/p0/$run/"); done
  # 只限制运行目录；代码和文档照常同步，原有原始数据过滤继续生效。
  FILTERS+=(--exclude '/runs/p0/*/' --exclude '/runs/*/')
fi
FILTERS+=(
  # 备份机上独立维护的目录：公开代码仓库与PPT工程，Thor上没有，推送的--delete不能删掉它们。
  --exclude '/Cosmos-RSI/'
  --exclude '/ppt_materials/deck/'
  --exclude '.git/'
  --exclude '.env'
  --exclude '.env.*'
  --exclude '.omc/'
  --exclude '__pycache__/'
  --exclude '.venv/'
  --exclude 'envs/'
  --exclude 'tools/'
  --exclude '.cache/'
  --exclude '.config/'
  --exclude 'data/'
  --include 'third_party/'
  --include 'third_party/PINS.md'
  --exclude 'third_party/*'
  --include 'runs/p0/video_materials_*/*.mp4'
  --include 'runs/p0/feedback_materials_*/*.mp4'
  --include 'runs/p0/p1_iteration_materials_*/*.mp4'
  --include 'runs/p0/p1_update_diagnostics_*/*.mp4'
  --exclude 'runs/**/*.mp4'
  --include 'runs/p0/freeze_eval_*/state_snapshots/**/*.npz'
  --exclude 'runs/**/*.npz'
  --exclude 'runs/**/*.npy'
  --exclude 'runs/**/*.hdf5'
  --exclude 'runs/**/*.parquet'
  --exclude 'runs/**/*.pt'
  --exclude 'runs/**/*.ckpt'
  --exclude 'runs/**/*.safetensors'
  --exclude 'runs/**/lerobot/'
  --exclude 'runs/**/videos/'
)
if [[ "$WITH_CKPT" -eq 1 ]]; then
  FILTERS+=(--include 'checkpoints/' --include 'checkpoints/promoted/***' --exclude 'checkpoints/*')
else
  FILTERS+=(--exclude 'checkpoints/')
fi

# 推送时只删除未被过滤的多余备份；拉取时不删除，且始终保留本地 Git 状态。
# 快照和上游diff含大量重复文本；压缩降低跨机流量，接收端恢复后仍按原字节校验。
rsync -azH "${DELETE[@]}" ${DRY} -e "${RSH}" "${FILTERS[@]}" "${SRC}/" "${DEST}/"
echo "备份完成：${SRC} -> ${DEST}${DRY:+（dry-run）}"
