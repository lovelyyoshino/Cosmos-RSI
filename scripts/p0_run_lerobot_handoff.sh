#!/usr/bin/env bash
# p0_run_lerobot_handoff.sh — 先单窗口验证原生格式，再扩至两个已准入窗口。
# @author pony
# @date 2026-11-08
# @version v1.0.0
# @last_modified 2026-11-08
# @changelog
#   - v1.0.0 (2026-11-08): 顺序运行、独立日志和明确退出状态，不启动策略训练。
set -euo pipefail
cd /home/thor/rsi
source scripts/thor_env.sh
RSI_EXEC_DIR="$(realpath -e "${1:?需要新的执行目录}")"
[[ "$(hostname)" == thor && "$RSI_EXEC_DIR" == /home/thor/rsi/runs/p0/lerobot_handoff_exec_* && ! -e "$RSI_EXEC_DIR/phase.txt" ]] || exit 2
trap 'printf "%s\n" "$?" > "$RSI_EXEC_DIR/exit_code"' EXIT
mkdir -p "$RSI_EXEC_DIR/logs"
RSI_ARGS=(--parent runs/p0/policy_train_20261107T003100_604340+0800/checkpoints/step_005000
  --gate runs/p0/uv_window_gate_20261108T184629_540070+0800/quality_gate.json)
for RSI_PHASE in single expanded; do
  printf '%s\n' "$RSI_PHASE" > "$RSI_EXEC_DIR/phase.txt"
  if [[ "$RSI_PHASE" == expanded ]]; then
    RSI_ARGS+=(--gate runs/p0/uv_window_gate_20261108T184634_129816+0800/quality_gate.json)
  fi
  CUDA_VISIBLE_DEVICES= envs/policy/bin/python -u scripts/p0_lerobot_handoff.py "${RSI_ARGS[@]}" \
    > "$RSI_EXEC_DIR/logs/$RSI_PHASE.log" 2>&1
  envs/policy/bin/python - "$RSI_EXEC_DIR" "$RSI_PHASE" <<'PY'
import json,pathlib,sys
directory=pathlib.Path(sys.argv[1]);phase=sys.argv[2]
paths=[line.split("：",1)[1].strip() for line in (directory/"logs"/(phase+".log")).read_text().splitlines() if line.startswith("记录目录：")]
if len(paths)!=1:raise ValueError("运行目录不唯一")
run=pathlib.Path(paths[0]);probe=json.loads((run/"probe.json").read_text())
if probe["status"]!="passed" or probe["stage"]!="lerobot_handoff":raise ValueError("实际交接未验收通过")
(directory/(phase+"_run.txt")).write_text(str(run)+"\n")
PY
done
printf 'completed_data_handoff_no_policy_gain\n' > "$RSI_EXEC_DIR/phase.txt"
