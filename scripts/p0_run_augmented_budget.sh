#!/usr/bin/env bash
# p0_run_augmented_budget.sh — 相同热启动和配对预算，串行测三种真实训练角色。
# @author pony
# @date 2026-11-08
# @version v1.0.0
# @last_modified 2026-11-08
# @changelog
#   - v1.0.0 (2026-11-08): 已验收LeRobot交接、逐角色退出检查及汇总，无晋升。
set -euo pipefail
cd /home/thor/rsi
source scripts/thor_env.sh
RSI_EXEC_DIR="$(realpath -e "${1:?需要新的执行目录}")"
RSI_HANDOFF="$(realpath -e "${2:?需要已验收handoff.json}")"
[[ "$(hostname)" == thor && "$RSI_EXEC_DIR" == /home/thor/rsi/runs/p0/augmented_budget_exec_* && ! -e "$RSI_EXEC_DIR/phase.txt" ]] || exit 2
[[ "$RSI_HANDOFF" == /home/thor/rsi/runs/p0/lerobot_handoff_*/handoff.json ]] || exit 2
trap 'printf "%s\n" "$?" > "$RSI_EXEC_DIR/exit_code"' EXIT
mkdir -p "$RSI_EXEC_DIR/logs"
for RSI_ROLE in candidate control analytic_ablation; do
  printf '%s\n' "$RSI_ROLE" > "$RSI_EXEC_DIR/phase.txt"
  envs/policy/bin/python -u scripts/p0_augmented_budget.py --role "$RSI_ROLE" --handoff "$RSI_HANDOFF" \
    --parent runs/p0/policy_train_20261107T003100_604340+0800/checkpoints/step_005000 \
    --steps 40 --batch 8 --warmup 5 --seed 42 > "$RSI_EXEC_DIR/logs/$RSI_ROLE.log" 2>&1
  envs/policy/bin/python - "$RSI_EXEC_DIR" "$RSI_ROLE" <<'PY'
import json,pathlib,sys
directory=pathlib.Path(sys.argv[1]);role=sys.argv[2]
paths=[line.split("：",1)[1].strip() for line in (directory/"logs"/(role+".log")).read_text().splitlines() if line.startswith("记录目录：")]
if len(paths)!=1:raise ValueError("运行目录不唯一")
run=pathlib.Path(paths[0]);probe=json.loads((run/"probe.json").read_text())
if probe["status"]!="passed" or probe["stage"]!="augmented_budget" or probe["result"]["role"]!=role:
    raise ValueError("本角色没有完成实际训练及重载")
(directory/(role+"_run.txt")).write_text(str(run)+"\n")
PY
done
envs/policy/bin/python - "$RSI_EXEC_DIR" <<'PY'
import json,pathlib,sys
sys.path.insert(0,"scripts")
from p0_common import ensure_gpu_idle
directory=pathlib.Path(sys.argv[1]);rows=[]
for role in ["candidate","control","analytic_ablation"]:
    run=pathlib.Path((directory/(role+"_run.txt")).read_text().strip())
    data=json.loads((run/"train_run.json").read_text());probe=json.loads((run/"probe.json").read_text())
    row={k:data[k] for k in ["role","parent_ckpt_sha256","paired_plan_sha256","handoff_sha256","training_wall_s","measured_samples_per_s","reload_max_abs_diff","output_ckpt_sha256","data_mix"]}
    row.update(directory=str(run),wall_s=probe["wall_s"],peak_mem_gb=probe["peak_mem_gb"])
    rows.append(row)
for key in ["parent_ckpt_sha256","paired_plan_sha256","handoff_sha256"]:
    if len({row[key] for row in rows})!=1:raise ValueError("三角色预算或输入不一致")
ensure_gpu_idle()
(directory/"summary.json").write_text(json.dumps({"passed":True,"runs":rows,"source_windows":2,"steps_per_role":40,"batch":8,"seed":42,"warmup":5,"policy_gain_evaluated":False,"promotion_performed":False},ensure_ascii=False,indent=2)+"\n")
PY
printf 'completed_budget_only_no_policy_gain\n' > "$RSI_EXEC_DIR/phase.txt"
