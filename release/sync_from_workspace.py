"""把 Thor 工作区（或其备份）整理同步成可公开的 Cosmos-RSI 仓库。

@author pony
@date 2026-11-10
@version v1.1.0
@last_modified 2026-11-10
@changelog
    - v1.1.0 (2026-11-10): 只同步运行所需文本；文档由发布仓库维护，排除图片、演示和过程记录。
    - v1.0.0 (2026-11-10): 按流水线入口及真实 import 依赖挑选脚本；只带走对应测试、配置、
      配置与接口规范；去掉主机地址；把指向未公开运行目录的链接改成纯文本。

用法（在本仓库根目录）：
    python3 release/sync_from_workspace.py --workspace <工作区根目录> [--dry-run]

工作区是 Thor 上的 ~/rsi 或它的备份。运行产物（runs/、data/、checkpoints/、envs/）、上游克隆、
PPT、图片、过程记录与历史归档都不同步。README、PLAN、AGENTS、使用文档和 release/
由本仓库自己维护，同步时不覆盖也不删除；第三方克隆与运行数据不在清理范围内。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# 流水线入口：编排、五段、环境、数据与初训、评测、Predict、运维。其余脚本只在被这些脚本 import 时带上。
SCRIPT_SEEDS = [
    'loop.py',
    'p0_common.py',
    'thor_env.sh',
    'p1_workflow_status.py',
    'p1_stage_transition.py',
    'p1_promote.py',
    'backup_to_lenovo.sh',
    'p0_pipeline_status.py',
    'p0_bootstrap.sh',
    'p0_setup_sim.sh',
    'p0_setup_transfer.sh',
    'p0_setup_transfer_local.py',
    'p0_setup_transfer_routed.py',
    'p0_transfer_mirror.py',
    'p0_prefetch_cuda_wheels.py',
    'p0_setup_magick.py',
    'p0_download_model.py',
    'p0_download_resources.sh',
    'p0_probe.py',
    'p0_sim_smoke.py',
    'p0_policy_smoke.py',
    'p0_judge_smoke.py',
    'p0_transfer_probe.py',
    'p0_transfer_config_probe.py',
    'p0_fetch_dataset.py',
    'p0_dataset_index.py',
    'p0_fetch_raw_libero.py',
    'p0_map_raw_libero.py',
    'p0_raw_alignment.py',
    'p0_regenerate_d0.py',
    'p0_prepare_d0.py',
    'p0_sim_data.py',
    'p0_demo_replay.py',
    'p0_train_policy.py',
    'p0_policy_runtime.py',
    'p0_run_baseline.py',
    'p0_verify_resume.py',
    'p0_eval_catalog.py',
    'p0_freeze_eval.py',
    'p0_evaluate_policy.py',
    'p0_analyze_evaluation.py',
    'p0_verify_eval_workers.py',
    'p0_policy_server.py',
    'p0_feedback_chain.py',
    'p0_feedback_rollout.py',
    'p0_feedback_judge.py',
    'p0_feedback_mine.py',
    'p0_feedback_materials.py',
    'p0_verify_feedback_workers.py',
    'p1_reason_followup.py',
    'p0_transfer_smoke.py',
    'p0_prepare_verified_transfer.py',
    'p1_feedback_generation.py',
    'p0_table_texture_design.py',
    'p0_table_uv_atlas_design.py',
    'p0_table_uv_validation.py',
    'p0_run_uv_validation.sh',
    'p0_uv_window_gate.py',
    'p0_lerobot_handoff.py',
    'p0_run_lerobot_handoff.sh',
    'p1_policy_update.py',
    'p0_augmented_budget.py',
    'p0_augmented_smoke.py',
    'p0_paired_sampling.py',
    'p0_run_augmented_budget.sh',
    'p0_run_augmented_smoke.sh',
    'p1_conservative_pilot.py',
    'p1_policy_regression_probe.py',
    'p1_update_diagnostics.py',
    'p1_eval_gate.py',
    'p0_deploy_smoke.py',
    'p0_augmented_deploy_smoke.py',
    'p1_predict_validation.py',
    'p0_predict_smoke.py',
    'p0_predict_embeddings.py',
    'p0_predict_transition.py',
    'p0_predict_adapter.py',
    'p0_run_predict_smoke.sh',
    'p1_iteration_materials.py',
    'p1_pilot_materials.py',
    'p1_training_materials.py',
    'p1_regression_materials.py',
    'p1_reason_diagnostic_materials.py',
]

TOP_FILES = ["pytest.ini", "third_party/PINS.md"]
ALL_FILES_DIRS = ["eval", "patches"]
# 代码里没有按名字引用、但复现必需的配置：正式收口用的预算，以及 policy 环境的依赖清单
CONFIG_ALWAYS = ["configs/iteration_budget_conservative_formal.yaml", "configs/p0_policy_requirements.txt",
                 "configs/p0_policy_constraints.txt"]
# 本仓库自己维护的文件：同步不覆盖、不删除
PRESERVE = ["README.md", "AGENTS.md", "PLAN.md", "docs/workflow.md", "docs/runbook.md",
            "docs/installation.md", "docs/references.md", "docs/contracts.md", "docs/eval_protocol.md",
            "tests/test_release_sync.py",
            "configs/eval_frozen.hash",
            ".gitignore", "release/", "LICENSE", ".github/"]
MANAGED = ["rsi/", "scripts/", "configs/", "eval/", "patches/", "tests/", "docs/",
           "third_party/PINS.md", "pytest.ini"]
TEXT = {".py", ".sh", ".md", ".yaml", ".yml", ".json", ".txt", ".ini", ".hash", ".sha256", ".patch", ".diff",
        ".csv", ".toml", ".cfg"}
EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", "notes", "figures", "ppt_materials", "archive"}

# 公开仓库不带主机地址与备份机路径；备份脚本的源/目标改成必须显式设置。
# 规则全部是通用模式，本文件里不写任何真实地址或口令；其他私有字符串用环境变量 RSI_SYNC_FORBIDDEN 传入（逗号分隔）。
PRIVATE_IP = r"(?<![\d.])(?:100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])|192\.168|172\.(?:1[6-9]|2\d|3[01]))(?:\.\d{1,3}){2}(?![\d.])"
SANITIZE = [
    (r'"\$\{RSI_THOR_SOURCE:-[^}]*\}"', '"${RSI_THOR_SOURCE:?设置 Thor 源目录，例如 thor@<thor-host>:/home/thor/rsi}"'),
    (r'"\$\{RSI_BACKUP_DEST:-[^}]*\}"',
     '"${RSI_BACKUP_DEST:?设置备份目标，例如 <user>@<backup-host>:/path/to/cosmos-thor-rsi}"'),
    (r"\bthor@" + PRIVATE_IP, "thor@<thor-host>"),
    (r"\b[\w-]+@" + PRIVATE_IP, "<user>@<host>"),
    (PRIVATE_IP, "<host>"),
    (r"/media/[\w-]+/cosmos-thor-rsi", "<backup-root>"),
]
FORBIDDEN = [re.compile(p) for p in (PRIVATE_IP, r"/media/[\w-]+/", r"/home/(?!thor/)[\w-]+/", r"/Users/[\w-]+/", r"hf_[A-Za-z0-9]{20,}",
                                      r"sk-[A-Za-z0-9]{20,}", r"ghp_[A-Za-z0-9]{20,}", r"-----BEGIN [A-Z ]*PRIVATE KEY",
                                      r"SSHPASS=['\"]?(?!<)[^\s'\"<$]{4,}")]

REF = re.compile(r"[\w.-]+\.(?:py|sh)\b")
IMPORT = re.compile(r"^\s*(?:from\s+([\w.]+)\s+import|import\s+([^\n#]+))", re.MULTILINE)
SPEC = re.compile(r"spec_from_file_location\([^)]*?([\w-]+\.py)")
PATHREF = re.compile(r"(?:scripts|rsi)/([\w.-]+\.(?:py|sh))\b")  # 方法依赖清单、入口登记里写明的脚本路径
LINK = re.compile(r"(!?)\[([^\]]*)\]\(([^)\s]+)\)")


def read(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")


def publishable(path: Path, workspace: Path) -> bool:
    """上传范围仅接受文本源文件，避免目录递归带回素材和中间产物。"""
    return (path.is_file() and not path.is_symlink() and path.suffix.lower() in TEXT
            and not (set(path.relative_to(workspace).parts) & EXCLUDED_PARTS))


def preserved(rel: str) -> bool:
    return rel in PRESERVE or any(rel == p.rstrip("/") or rel.startswith(p)
                                  for p in PRESERVE if p.endswith("/"))


def script_refs(text: str, names: dict[str, Path], modules: dict[str, str], hard: bool) -> set[str]:
    out = set()
    for a, b in IMPORT.findall(text):
        for name in ([a] if a else [x.strip().split()[0] for x in b.split(",") if x.strip()]):
            if name in modules:
                out.add(modules[name])
    out |= {m for m in SPEC.findall(text) if m in names}
    out |= {m for m in PATHREF.findall(text) if m in names}  # 运行时会按路径读取或核验摘要
    if not hard:  # 测试与配置也可能按文件名引用脚本
        out |= {m for m in REF.findall(text) if m in names}
    return out


def select(ws: Path) -> tuple[dict[str, Path], dict]:
    names = {p.name: p for p in sorted((ws / "scripts").glob("*"))
             if publishable(p, ws) and p.suffix in {".py", ".sh"}}
    names["loop.py"] = ws / "rsi/loop.py"
    modules = {n[:-3]: n for n in names if n.endswith(".py")}
    missing = [s for s in SCRIPT_SEEDS if s not in names]
    if missing:
        sys.exit(f"工作区缺少入口脚本：{missing}")
    graph = {n: script_refs(read(p), names, modules, True) - {n} for n, p in names.items()}
    # 入口登记配置（rsi_pipeline.yaml、p1_execution.yaml）里写明路径的脚本也是入口
    seeds = list(SCRIPT_SEEDS)
    for cfg in ("configs/rsi_pipeline.yaml", "configs/p1_execution.yaml"):
        if (ws / cfg).is_file():
            seeds += sorted(script_refs(read(ws / cfg), names, modules, True))
    keep, stack = set(), seeds
    while stack:  # 只跟随真实 import：被入口代码引用到的模块一律带上，正文提到的不算
        n = stack.pop()
        if n not in keep:
            keep.add(n)
            stack += graph[n]
    files = {}
    for n in keep:
        rel = "rsi/loop.py" if n == "loop.py" else f"scripts/{n}"
        files[rel] = ws / rel
    tests_kept: list[str] = []
    tests_dropped: list[str] = []
    for p in sorted((ws / "tests").glob("test_*.py")):
        refs = script_refs(read(p), names, modules, False)
        (tests_kept if refs <= keep else tests_dropped).append(p.name)
        if refs <= keep:
            files[f"tests/{p.name}"] = p
    for p in sorted((ws / "tests").rglob("*")):  # 测试夹具
        if publishable(p, ws) and not p.name.startswith("test_"):
            files[str(p.relative_to(ws))] = p
    for rel in TOP_FILES:
        if (ws / rel).is_file():
            files[rel] = ws / rel
    for d in ALL_FILES_DIRS:
        for p in sorted((ws / d).rglob("*")):
            if publishable(p, ws):
                files[str(p.relative_to(ws))] = p
    # 配置：被同步的代码、测试或其他配置按名字引用的才带上（迭代到不动点）；文档里提到的历史配置不算
    configs = {p.name: p for p in sorted((ws / "configs").rglob("*")) if publishable(p, ws)}
    chosen, changed = set(), True
    for rel in CONFIG_ALWAYS:
        if (ws / rel).is_file():
            chosen.add(rel)
            files[rel] = ws / rel
    while changed:
        changed = False
        blob = "\n".join(read(p) for rel, p in files.items()
                         if p.suffix in TEXT and not rel.startswith("docs/") and rel not in TOP_FILES)
        for name, p in configs.items():
            rel = str(p.relative_to(ws))
            if rel not in chosen and (name in blob):
                chosen.add(rel)
                files[rel] = p
                changed = True
    report = {"scripts_kept": len(keep), "scripts_dropped": sorted(set(names) - keep),
              "tests_kept": len(tests_kept), "tests_dropped": tests_dropped,
              "configs_kept": len(chosen), "configs_dropped": sorted(str(p.relative_to(ws)) for p in configs.values()
                                                                    if str(p.relative_to(ws)) not in chosen),
              "notes_kept": 0}
    return {rel: path for rel, path in files.items() if not preserved(rel)}, report


def sanitize(text: str) -> str:
    for pattern, replacement in SANITIZE:
        text = re.sub(pattern, replacement, text)
    return text


def private_patterns() -> list[re.Pattern]:
    extra = [s.strip() for s in os.environ.get("RSI_SYNC_FORBIDDEN", "").split(",") if s.strip()]
    return FORBIDDEN + [re.compile(re.escape(s)) for s in extra]


def fix_links(rel: str, text: str, exported: set[str]) -> tuple[str, int]:
    """指向未同步文件（运行目录、PPT、历史归档、未保留笔记）的链接改成纯文本，路径留作出处。"""
    base = Path(rel).parent
    fixed = 0

    def sub(m):
        nonlocal fixed
        bang, label, target = m.groups()
        if re.match(r"[a-z]+:", target) or target.startswith("#"):
            return m.group(0)
        path = target.split("#")[0]
        norm = Path(*[p for p in (base / path).parts])
        parts = []
        for p in norm.parts:
            if p == "..":
                if parts:
                    parts.pop()
            elif p != ".":
                parts.append(p)
        dest = "/".join(parts)
        if dest in exported or any(dest == x.rstrip("/") or dest.startswith(x) for x in PRESERVE if x.endswith("/")) \
                or dest in PRESERVE:
            return m.group(0)
        fixed += 1
        if bang:
            return f"（图：{label or Path(path).name}，`{dest}`，未随仓库发布）"
        return f"{label}（`{dest}`）" if "runs/" in dest else label
    return LINK.sub(sub, text), fixed


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workspace", type=Path, required=True, help="Thor 上的 ~/rsi 或它的备份根目录")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    ws = args.workspace.resolve()
    if ws == REPO or REPO in ws.parents:
        sys.exit("工作区与发布仓库不能是同一目录，源工作区也不能位于发布仓库内部")
    if not (ws / "rsi/loop.py").is_file():
        sys.exit(f"{ws} 不是 RSI 工作区（缺 rsi/loop.py）")
    files, report = select(ws)
    exported = set(files)
    out, problems, links_fixed = {}, [], 0
    for rel, src in sorted(files.items()):
        data = src.read_bytes()
        if src.suffix in TEXT or src.name in ("AGENTS.md", "PLAN.md"):
            text = sanitize(data.decode("utf-8", errors="strict"))
            if src.suffix == ".md":
                text, n = fix_links(rel, text, exported)
                links_fixed += n
            for pat in private_patterns():
                for m in pat.finditer(text):
                    problems.append(f"{rel}: 第 {text.count(chr(10), 0, m.start()) + 1} 行")
            data = text.encode("utf-8")
        out[rel] = data
    # 仅检查随发布保留的文件；不扫描本地运行数据、凭据环境或上游克隆。
    preserved_paths = []
    for rel in PRESERVE:
        path = REPO / rel
        preserved_paths.extend(path.rglob("*") if rel.endswith("/") and path.is_dir() else [path])
    for p in sorted(preserved_paths):
        rel = str(p.relative_to(REPO))
        if p.is_file() and rel not in out and p.suffix in TEXT | {""}:
            text = read(p)
            for pat in private_patterns():
                for m in pat.finditer(text):
                    problems.append(f"{rel}: 第 {text.count(chr(10), 0, m.start()) + 1} 行")
    if problems:  # 只报位置，不回显命中的内容
        sys.exit("仍含主机地址、私有路径或凭据：\n" + "\n".join(problems[:20]))

    def managed(rel: str) -> bool:
        if preserved(rel):
            return False
        return any(rel == m or rel.startswith(m) for m in MANAGED)

    candidates = []
    for rel in MANAGED:
        path = REPO / rel
        candidates.extend(path.rglob("*") if rel.endswith("/") and path.is_dir() else [path])
    stale = sorted(str(p.relative_to(REPO)) for p in candidates
                   if p.is_file() and managed(str(p.relative_to(REPO))) and str(p.relative_to(REPO)) not in out)
    written = 0
    if not args.dry_run:
        for rel in stale:
            (REPO / rel).unlink()
        for rel, data in out.items():
            dst = REPO / rel
            if not dst.is_file() or dst.read_bytes() != data:
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(data)
                written += 1
            if (ws / rel).stat().st_mode & 0o111:
                dst.chmod(dst.stat().st_mode | 0o111)
        for rel in MANAGED:
            if not rel.endswith("/") or not (REPO / rel).is_dir():
                continue
            for d in sorted((p for p in (REPO / rel).rglob("*") if p.is_dir()), reverse=True):
                if not any(d.iterdir()):
                    d.rmdir()
        manifest = {rel: hashlib.sha256(data).hexdigest() for rel, data in sorted(out.items())}
        (REPO / "release/sync_manifest.json").write_text(
            json.dumps({"files": manifest, **{k: v for k, v in report.items()}}, ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8")
    print(json.dumps({"exported": len(out), "written": written, "removed_stale": len(stale), "links_to_text": links_fixed,
                      **{k: (v if isinstance(v, int) else len(v)) for k, v in report.items()}}, ensure_ascii=False))
    if args.dry_run:
        print("dropped scripts:", " ".join(report["scripts_dropped"]))
        print("dropped tests:", " ".join(report["tests_dropped"]))
        print("dropped configs:", " ".join(report["configs_dropped"]))


if __name__ == "__main__":
    main()
