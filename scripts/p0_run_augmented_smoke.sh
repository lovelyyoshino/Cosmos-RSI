#!/usr/bin/env bash
# p0_run_augmented_smoke.sh — 同起点、同配对计划的短程候选/对照/消融串行运行。
# @author pony
# @date 2026-11-08
# @version v1.0.0
# @last_modified 2026-11-08
# @changelog
#   - v1.0.0 (2026-11-08): 后台入口、独立日志、真实退出状态及配对预算汇总。
set -euo pipefail
cd /home/thor/rsi
[[ "$(hostname)" == thor ]] || exit 2
source scripts/thor_env.sh
RSI_EXEC_DIR="$(realpath -e "${1:?需要新的执行目录}")"
[[ "$RSI_EXEC_DIR" == /home/thor/rsi/runs/p0/augmented_smoke_exec_* && ! -e "$RSI_EXEC_DIR/phase.txt" ]] || exit 2
trap 'printf "%s\n" "$?" > "$RSI_EXEC_DIR/exit_code"' EXIT
mkdir -p "$RSI_EXEC_DIR/logs"
for RSI_ROLE in candidate control analytic_ablation; do
  printf '%s\n' "$RSI_ROLE" > "$RSI_EXEC_DIR/phase.txt"
  envs/policy/bin/python -u scripts/p0_augmented_smoke.py --role "$RSI_ROLE" \
    --parent runs/p0/policy_train_20261107T003100_604340+0800/checkpoints/step_005000 \
    --gate runs/p0/uv_window_gate_20261108T184629_540070+0800/quality_gate.json \
    --gate runs/p0/uv_window_gate_20261108T184634_129816+0800/quality_gate.json \
    > "$RSI_EXEC_DIR/logs/$RSI_ROLE.log" 2>&1
  envs/policy/bin/python - "$RSI_EXEC_DIR" "$RSI_ROLE" <<'PY'
import json,pathlib,sys
directory=pathlib.Path(sys.argv[1]); role=sys.argv[2]
paths=[line.split("：",1)[1].strip() for line in (directory/"logs"/(role+".log")).read_text().splitlines() if line.startswith("记录目录：")]
if len(paths)!=1:
    raise ValueError("运行目录不唯一")
run=pathlib.Path(paths[0]); probe=json.loads((run/"probe.json").read_text())
if probe["status"]!="passed" or probe["stage"]!="augmented_training_smoke" or probe["result"]["role"]!=role:
    raise ValueError("本角色未完成真实短程训练及重载")
(directory/(role+"_run.txt")).write_text(str(run)+"\n")
PY
done
envs/policy/bin/python - "$RSI_EXEC_DIR" <<'PY'
import json,pathlib,sys
sys.path.insert(0,"scripts")
from p0_common import ensure_gpu_idle
directory=pathlib.Path(sys.argv[1]); rows=[]
for role in ["candidate","control","analytic_ablation"]:
    run=pathlib.Path((directory/(role+"_run.txt")).read_text().strip())
    data=json.loads((run/"training_smoke.json").read_text()); probe=json.loads((run/"probe.json").read_text())
    rows.append({key:data[key] for key in ["role","parent_ckpt_sha256","paired_plan_sha256","data_mix_slots","output_ckpt_sha256","training_wall_s","reload_max_abs_diff"]})
    rows[-1].update(directory=str(run),wall_s=probe["wall_s"],peak_mem_gb=probe["peak_mem_gb"])
if len({r["parent_ckpt_sha256"] for r in rows})!=1 or len({r["paired_plan_sha256"] for r in rows})!=1:
    raise ValueError("三角色的热启动或配对计划不相同")
ensure_gpu_idle()
(directory/"summary.json").write_text(json.dumps({"passed":True,"runs":rows,"source_windows":2,"steps_per_role":5,"batch":2,"seed":42,"policy_gain_evaluated":False,"promotion_performed":False},ensure_ascii=False,indent=2)+"\n")
PY
printf 'completed_smoke_only_no_policy_gain\n' > "$RSI_EXEC_DIR/phase.txt"
