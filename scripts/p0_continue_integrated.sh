#!/usr/bin/env bash
# 继续已在运行的采集，串行执行新版盲判、失败交接、素材与Predict动作条件冒烟。
# @author pony
# @date 2026-11-09
# @version v1.0.0
# @last_modified 2026-11-09
# @changelog
#   - v1.0.0 (2026-11-09): 复用本次已运行采集，逐阶段校验真实probe，工程失败立即停止。
set -euo pipefail
cd /home/thor/rsi
source scripts/thor_env.sh
RSI_EXEC_DIR="$(realpath -e "${1:?需要新的执行目录}")"
RSI_COLLECTION="$(realpath -m "${2:?需要等待的采集报告路径}")"
RSI_WAIT_PID="${3:?需要已有采集进程号}"
RSI_PREDICT_COLLECTION="$(realpath -e "${4:?需要Predict实际转移的采集报告}")"
[[ "$(hostname)" == thor && "$RSI_EXEC_DIR" == /home/thor/rsi/runs/p0/integrated_exec_* && ! -e "$RSI_EXEC_DIR/phase.txt" ]] || exit 2
trap 'printf "%s\n" "$?" > "$RSI_EXEC_DIR/exit_code"' EXIT
mkdir -p "$RSI_EXEC_DIR/logs"
printf 'waiting_for_existing_collection\n' > "$RSI_EXEC_DIR/phase.txt"
while kill -0 "$RSI_WAIT_PID" 2>/dev/null; do sleep 5; done

checked_stage() {
  envs/policy/bin/python - "$RSI_EXEC_DIR/logs/$1.log" "$2" "$3" <<'PY'
import hashlib,json,pathlib,sys
paths=[line.split("：",1)[1].strip() for line in pathlib.Path(sys.argv[1]).read_text().splitlines() if line.startswith("记录目录：")]
if len(paths)!=1:raise ValueError("阶段运行目录不唯一")
run=pathlib.Path(paths[0]);probe=json.loads((run/"probe.json").read_text());artifact=run/sys.argv[3]
if probe['status']!='passed' or probe['stage']!=sys.argv[2] or probe['artifacts_sha256'].get(sys.argv[3])!=hashlib.sha256(artifact.read_bytes()).hexdigest():
 raise ValueError("实际阶段未通过，禁止继续")
print(artifact)
PY
}

envs/policy/bin/python - "$RSI_COLLECTION" <<'PY'
import hashlib,json,pathlib,sys
p=pathlib.Path(sys.argv[1]);probe=json.loads((p.parent/'probe.json').read_text())
if probe['stage']!='feedback_rollout' or probe['status']!='passed' or probe['artifacts_sha256'].get(p.name)!=hashlib.sha256(p.read_bytes()).hexdigest():
 raise ValueError('待接续的采集没有实际完成')
PY
printf 'reasoning_judge\n' > "$RSI_EXEC_DIR/phase.txt"
envs/policy/bin/python -u scripts/p0_feedback_judge.py --collection "$RSI_COLLECTION" \
  --config configs/judge_calibration.yaml > "$RSI_EXEC_DIR/logs/judge.log" 2>&1
RSI_JUDGE="$(checked_stage judge feedback_judge judge_report.json)"
printf '%s\n' "$RSI_JUDGE" > "$RSI_EXEC_DIR/judge_path.txt"
printf 'failure_mining\n' > "$RSI_EXEC_DIR/phase.txt"
envs/policy/bin/python -u scripts/p0_feedback_mine.py --collection "$RSI_COLLECTION" --judge "$RSI_JUDGE" \
  > "$RSI_EXEC_DIR/logs/mining.log" 2>&1
checked_stage mining feedback_mine mining_report.json > "$RSI_EXEC_DIR/mining_path.txt"
printf 'feedback_materials\n' > "$RSI_EXEC_DIR/phase.txt"
envs/policy/bin/python -u scripts/p0_feedback_materials.py --collection "$RSI_COLLECTION" --judge "$RSI_JUDGE" \
  > "$RSI_EXEC_DIR/logs/materials.log" 2>&1
checked_stage materials feedback_materials materials_manifest.json > "$RSI_EXEC_DIR/materials_path.txt"
printf 'predict_true_transitions\n' > "$RSI_EXEC_DIR/phase.txt"
envs/sim/bin/python -u scripts/p0_predict_transition.py --collection "$RSI_PREDICT_COLLECTION" \
  > "$RSI_EXEC_DIR/logs/transition.log" 2>&1
RSI_TRANSITION="$(checked_stage transition predict_transition transition.json)"
printf '%s\n' "$RSI_TRANSITION" > "$RSI_EXEC_DIR/transition_path.txt"
printf 'predict_language_encoding\n' > "$RSI_EXEC_DIR/phase.txt"
envs/policy/bin/python -u scripts/p0_predict_embeddings.py --collection "$RSI_PREDICT_COLLECTION" \
  > "$RSI_EXEC_DIR/logs/embeddings.log" 2>&1
RSI_EMBEDDINGS="$(checked_stage embeddings predict_embeddings embeddings.json)"
printf '%s\n' "$RSI_EMBEDDINGS" > "$RSI_EXEC_DIR/embeddings_path.txt"
printf 'predict_external_action\n' > "$RSI_EXEC_DIR/phase.txt"
envs/predict/bin/python -u scripts/p0_predict_smoke.py --transition "$RSI_TRANSITION" --embeddings "$RSI_EMBEDDINGS" \
  > "$RSI_EXEC_DIR/logs/predict.log" 2>&1
checked_stage predict predict_external_action predict_report.json > "$RSI_EXEC_DIR/predict_path.txt"
printf 'completed_feedback_and_predict_smoke_no_policy_gain\n' > "$RSI_EXEC_DIR/phase.txt"
