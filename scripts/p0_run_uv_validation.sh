#!/usr/bin/env bash
# p0_run_uv_validation.sh — 设计通过后才读取新示范，逐阶段记录真实退出状态。
# @author pony
# @date 2026-11-08
# @version v1.0.0
# @last_modified 2026-11-08
# @changelog
#   - v1.0.0 (2026-11-08): 固定图集设计复现、完整控制及demo_26串行验证，不训练或晋升。
set -euo pipefail
RSI_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ "$RSI_PROJECT_ROOT" == /home/thor/rsi && "$(hostname)" == thor ]] || exit 2
cd "$RSI_PROJECT_ROOT"
source scripts/thor_env.sh
RSI_EXEC_DIR="$(realpath -e "${1:?需要新执行目录}")"
RSI_SOURCE="$(realpath -e "${2:?需要设计成功源}")"
RSI_CACHE="$(realpath -e "${3:?需要D0缓存目录}")"
[[ "$RSI_EXEC_DIR" == "$RSI_PROJECT_ROOT"/runs/p0/uv_validation_exec_* \
    && "$RSI_SOURCE" == "$RSI_PROJECT_ROOT"/runs/p0/*/verified_input.json \
    && "$RSI_CACHE" == "$RSI_PROJECT_ROOT"/data/* && -f "$RSI_CACHE/manifest.json" \
    && ! -e "$RSI_EXEC_DIR/phase.txt" && ! -e "$RSI_EXEC_DIR/exit_code" ]] || exit 2
mkdir -p "$RSI_EXEC_DIR/logs"
trap 'printf "%s\n" "$?" > "$RSI_EXEC_DIR/exit_code"' EXIT
envs/sim/bin/python -m unittest discover -s tests -p test_table_uv_validation.py -v > "$RSI_EXEC_DIR/logs/tests.log" 2>&1
envs/sim/bin/python - "$RSI_CACHE/manifest.json" > "$RSI_EXEC_DIR/logs/source_preflight.log" <<'PY'
import json,pathlib,sys
sys.path.insert(0,"scripts")
from p0_prepare_verified_transfer import select_training_source
from p0_table_uv_validation import CONFIG,checked_config
checked_config()
data=json.loads(pathlib.Path(sys.argv[1]).read_text())
episode="thor-regen-v1-STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy-"+CONFIG["heldout_episode_suffix"].split("caddy-",1)[1]
row=select_training_source(data,"STUDY_SCENE1",0,episode)
print(json.dumps({"episode_id":row["episode_id"],"split":row["split"],"env_success":row["env_success"],"images_read":False},ensure_ascii=False))
PY

run_recorded() {
  local phase="$1" script="$2"
  shift 2
  [[ ! -e "$RSI_EXEC_DIR/logs/$phase.log" ]] || return 2
  printf '%s\n' "$phase" > "$RSI_EXEC_DIR/phase.txt"
  envs/sim/bin/python -c 'import sys; sys.path.insert(0,"scripts"); from p0_common import ensure_gpu_idle; ensure_gpu_idle()'
  envs/sim/bin/python -u "scripts/$script" "$@" > "$RSI_EXEC_DIR/logs/$phase.log" 2>&1 || return "$?"
  envs/sim/bin/python - "$RSI_EXEC_DIR/logs/$phase.log" <<'PY'
import json,pathlib,sys
sys.path.insert(0,"scripts")
paths=[line.split("：",1)[1].strip() for line in pathlib.Path(sys.argv[1]).read_text().splitlines() if line.startswith("记录目录：")]
if len(paths)!=1:
    raise ValueError("阶段目录不唯一")
directory=pathlib.Path(paths[0]).resolve(strict=True)
probe=json.loads((directory/"probe.json").read_text())
if not directory.is_relative_to(pathlib.Path("/home/thor/rsi/runs/p0")) or probe["status"]!="passed":
    raise ValueError("阶段失败或路径越界")
if probe["stage"]=="table_uv_controls":
    from p0_table_uv_validation import checked_controls
    data=json.loads((directory/"renderer_controls.json").read_text())
    checked_controls(directory/"renderer_controls.json",data["role"])
print(directory)
PY
}

RSI_DESIGN="$(run_recorded design_render p0_table_uv_validation.py render --role design --source-input "$RSI_SOURCE")"
printf '%s\n' "$RSI_DESIGN" > "$RSI_EXEC_DIR/design_render_run.txt"
RSI_CONTROLS="$(run_recorded design_controls p0_table_uv_validation.py controls --role design --render-report "$RSI_DESIGN/renderer_replay.json")"
printf '%s\n' "$RSI_CONTROLS" > "$RSI_EXEC_DIR/design_controls_run.txt"
# demo_26只在固定设计与完整实际控制通过后读取；不能用它重新挑纹理或边界。
RSI_HELDOUT_SOURCE="$(run_recorded heldout_source p0_prepare_verified_transfer.py --cache "$RSI_CACHE" --task STUDY_SCENE1 \
  --episode-id thor-regen-v1-STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy-demo_26 --full-frame-edge)"
printf '%s\n' "$RSI_HELDOUT_SOURCE" > "$RSI_EXEC_DIR/heldout_source_run.txt"
RSI_HELDOUT="$(run_recorded heldout_render p0_table_uv_validation.py render --role heldout --source-input "$RSI_HELDOUT_SOURCE/verified_input.json" \
  --design-report "$RSI_CONTROLS/renderer_controls.json")"
printf '%s\n' "$RSI_HELDOUT" > "$RSI_EXEC_DIR/heldout_render_run.txt"
RSI_HELDOUT_CONTROLS="$(run_recorded heldout_controls p0_table_uv_validation.py controls --role heldout --render-report "$RSI_HELDOUT/renderer_replay.json")"
printf '%s\n' "$RSI_HELDOUT_CONTROLS" > "$RSI_EXEC_DIR/heldout_controls_run.txt"
printf '%s\n' 'method_controls_completed_visual_review_and_training_admission_pending' > "$RSI_EXEC_DIR/phase.txt"
