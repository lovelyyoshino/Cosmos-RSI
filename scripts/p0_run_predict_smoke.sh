#!/usr/bin/env bash
# 从已核验的真实转移恢复Predict验证，语言模型和预测模型按阶段串行。
# @author pony
# @date 2026-11-09
# @version v1.0.0
# @last_modified 2026-11-09
# @changelog
#   - v1.0.0 (2026-11-09): 复用转移证据；逐阶段保留日志和退出状态。
set -euo pipefail
cd /home/thor/rsi
source scripts/thor_env.sh
RSI_EXEC_DIR="$(realpath -e "${1:?需要新的执行目录}")"
RSI_TRANSITION="$(realpath -e "${2:?需要真实转移报告}")"
RSI_COLLECTION="$(realpath -e "${3:?需要源采集报告}")"
[[ "$(hostname)" == thor && "$RSI_EXEC_DIR" == /home/thor/rsi/runs/p0/predict_exec_* && ! -e "$RSI_EXEC_DIR/phase.txt" ]] || exit 2
trap 'printf "%s\n" "$?" > "$RSI_EXEC_DIR/exit_code"' EXIT
mkdir -p "$RSI_EXEC_DIR/logs"
printf 'language_encoding\n' > "$RSI_EXEC_DIR/phase.txt"
envs/policy/bin/python -u scripts/p0_predict_embeddings.py --collection "$RSI_COLLECTION" \
  > "$RSI_EXEC_DIR/logs/embeddings.log" 2>&1
RSI_EMBEDDINGS="$(envs/policy/bin/python - "$RSI_EXEC_DIR/logs/embeddings.log" <<'PY'
import hashlib,json,pathlib,sys
paths=[line.split("：",1)[1].strip() for line in pathlib.Path(sys.argv[1]).read_text().splitlines() if line.startswith("记录目录：")]
if len(paths)!=1:raise ValueError("语言编码目录不唯一")
run=pathlib.Path(paths[0]);probe=json.loads((run/"probe.json").read_text());artifact=run/'embeddings.json'
if probe['status']!='passed' or probe['stage']!='predict_embeddings' or probe['artifacts_sha256'].get(artifact.name)!=hashlib.sha256(artifact.read_bytes()).hexdigest():
 raise ValueError("语言编码未通过实际校验")
print(artifact)
PY
)"
printf '%s\n' "$RSI_EMBEDDINGS" > "$RSI_EXEC_DIR/embeddings_path.txt"
printf 'external_action_prediction\n' > "$RSI_EXEC_DIR/phase.txt"
envs/predict/bin/python -u scripts/p0_predict_smoke.py --transition "$RSI_TRANSITION" --embeddings "$RSI_EMBEDDINGS" \
  > "$RSI_EXEC_DIR/logs/predict.log" 2>&1
printf 'completed_predict_interface_smoke_only\n' > "$RSI_EXEC_DIR/phase.txt"
