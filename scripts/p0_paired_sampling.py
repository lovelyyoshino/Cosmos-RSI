#!/usr/bin/env python3
"""为已由上层验收的数据构建固定候选/对照抽样计划，不执行训练。

@author pony
@date 2026-11-08
@version v1.0.0
@last_modified 2026-11-08
@changelog
    - v1.0.0 (2026-11-08): 固定配额、部分episode边界、配对补齐和规范JSON摘要。

描述字段为kind、sample_id、source_episode_id、source_length、source_split、
is_eval和window(start_frame/end_frame，左闭右开)。kind为demo、generated或
self_distill；新增描述必须显式eligible_for_training=True。source_split仅允许
demo_train或rollout_train，demo只能来自前者，self_distill只能来自后者。

build_paired_plan的quotas是整份steps×batch计划中的generated/self_distill次数，
其余为demo。每类按有效动作起点有放回均匀抽样；step从1开始，slot从0开始。
source_frame为源episode绝对帧，window_frame为部分episode相对帧。

本模块不读取图像、标签、质量报告或冻结评测集；eligible标志只是上层调用契约，
不能单凭此标志授予真实训练资格。rollout_train也必须先由上层证明来源与隔离。
"""
from __future__ import annotations

from bisect import bisect_right
from collections import Counter
import copy
import hashlib
import json
import random


KINDS = ("demo", "generated", "self_distill")
EXTRA_KINDS = KINDS[1:]
REQUIRED_FIELDS = {"kind", "sample_id", "source_episode_id", "source_length", "source_split", "is_eval", "window"}


def _integer(value, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name}必须是大于等于{minimum}的整数")
    return value


def _descriptor(value: dict, horizon: int) -> dict:
    if not isinstance(value, dict) or not REQUIRED_FIELDS <= value.keys() \
            or set(value) - REQUIRED_FIELDS - {"eligible_for_training"}:
        raise ValueError("窗口描述缺少必填字段或包含未定义字段")
    kind = value["kind"]
    if not isinstance(kind, str) or kind not in KINDS:
        raise ValueError("窗口kind必须为demo、generated或self_distill")
    for key in ["sample_id", "source_episode_id"]:
        if not isinstance(value[key], str) or not value[key].strip() or value[key] != value[key].strip():
            raise ValueError(f"{key}必须为非空且无首尾空白的字符串")
    split = value["source_split"]
    if value["is_eval"] is not False or not isinstance(split, str) or split not in {"demo_train", "rollout_train"} \
            or (kind == "demo" and split != "demo_train") \
            or (kind == "self_distill" and split != "rollout_train"):
        raise ValueError("只接受明确非评测的对应训练来源，禁止验证或评测回合")
    if (kind in EXTRA_KINDS or "eligible_for_training" in value) and value.get("eligible_for_training") is not True:
        raise ValueError("新增窗口必须由上层显式声明eligible_for_training=True；拒绝未准入描述")
    length = _integer(value["source_length"], "source_length", 1)
    window = value["window"]
    if not isinstance(window, dict) or set(window) != {"start_frame", "end_frame"}:
        raise ValueError("window必须只含start_frame和end_frame")
    first = _integer(window["start_frame"], "start_frame")
    stop = _integer(window["end_frame"], "end_frame", 1)
    if first >= stop or stop > length or stop - first < horizon:
        raise ValueError("窗口越出源episode或短于动作块，禁止跨边界取样")
    return copy.deepcopy(value)


def valid_source_starts(description: dict, *, horizon: int) -> range:
    """返回经过边界检查的全部源绝对动作起点，不修改调用者的描述。"""
    _integer(horizon, "horizon", 1)
    item = _descriptor(description, horizon)
    window = item["window"]
    return range(window["start_frame"], window["end_frame"] - horizon + 1)


def paired_plan_sha256(plan: dict) -> str:
    """对移除顶层sha256自身字段后的完整规范JSON计算摘要。"""
    payload = {key: value for key, value in plan.items() if key != "sha256"}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_paired_plan(demo_windows: list[dict], extra_windows: list[dict], *, steps: int,
                      batch: int, horizon: int, seed: int, quotas: dict[str, int]) -> dict:
    """生成可持久化的配对计划；配额是抽样次数，不是独立episode数量。"""
    for name, value in [("steps", steps), ("batch", batch), ("horizon", horizon)]:
        _integer(value, name, 1)
    _integer(seed, "seed")
    if seed >= 2 ** 32:
        raise ValueError("seed必须在uint32范围内")
    if not isinstance(demo_windows, list) or not demo_windows or not isinstance(extra_windows, list):
        raise ValueError("demo_windows须为非空列表，extra_windows须为列表")
    if not isinstance(quotas, dict) or set(quotas) != set(EXTRA_KINDS):
        raise ValueError("quotas必须显式提供generated和self_distill的整份计划次数")
    for kind in EXTRA_KINDS:
        _integer(quotas[kind], f"quotas.{kind}")
    total = steps * batch
    if sum(quotas.values()) > total:
        raise ValueError("新增配额超过steps×batch的总槽位数")
    demos = [_descriptor(value, horizon) for value in demo_windows]
    extras = [_descriptor(value, horizon) for value in extra_windows]
    if any(row["kind"] != "demo" for row in demos) or any(row["kind"] not in EXTRA_KINDS for row in extras):
        raise ValueError("D0和新增描述放入了错误的输入列表")
    descriptions = sorted(demos + extras, key=lambda row: row["sample_id"])
    if len({row["sample_id"] for row in descriptions}) != len(descriptions):
        raise ValueError("sample_id必须在所有描述中唯一")
    sources = {}
    for row in descriptions:
        identity = (row["source_length"], row["source_split"])
        previous = sources.setdefault(row["source_episode_id"], identity)
        if previous != identity:
            raise ValueError("同一源episode的长度或训练分区存在矛盾")

    # 固定类别预算，避免候选和对照因数据集长度或RNG消耗不同而改变抽样次数。
    counts = {"demo": total - sum(quotas.values()), **quotas}
    pools = {}
    for kind in KINDS:
        members = [row for row in descriptions if row["kind"] == kind]
        cumulative, count = [], 0
        for row in members:
            count += row["window"]["end_frame"] - row["window"]["start_frame"] - horizon + 1
            cumulative.append(count)
        if not members and counts[kind]:
            raise ValueError(f"{kind}配额非零但没有已验收的窗口描述")
        pools[kind] = (members, cumulative)
    rng = random.Random(seed)
    slots = [kind for kind in KINDS for _ in range(counts[kind])]
    rng.shuffle(slots)

    def draw(kind: str, step: int, slot: int) -> dict:
        members, cumulative = pools[kind]
        index = rng.randrange(cumulative[-1])
        member = bisect_right(cumulative, index)
        row = members[member]
        relative = index - (cumulative[member - 1] if member else 0)
        return {"kind": kind, "sample_id": row["sample_id"], "source_episode_id": row["source_episode_id"],
                "source_length": row["source_length"], "source_split": row["source_split"], "is_eval": False,
                "source_frame": row["window"]["start_frame"] + relative, "window_frame": relative,
                "window": copy.deepcopy(row["window"]), "horizon": horizon, "step": step, "slot": slot, "seed": seed}

    candidate, control = [], []
    for index, kind in enumerate(slots):
        step, slot = divmod(index, batch)
        example = draw(kind, step + 1, slot)
        if kind == "demo":
            matched = copy.deepcopy(example)
        else:
            # 补齐始终重采样D0；保留候选身份，审计时可逐槽还原新增数据的预算。
            matched = {**draw("demo", step + 1, slot), "kind": "duplicate_fill",
                       "matched_candidate_sample_id": example["sample_id"], "matched_candidate_kind": kind}
        candidate.append(example)
        control.append(matched)
    plan = {"schema_version": "p0-paired-sampling-v1", "steps": steps, "batch": batch,
            "horizon": horizon, "seed": seed, "quotas": counts, "samples_per_role": total,
            "candidate_counts": dict(Counter(row["kind"] for row in candidate)),
            "control_counts": dict(Counter(row["kind"] for row in control)),
            "sampler": "valid_action_start_uniform_with_replacement/python_random-v1",
            "descriptions": descriptions, "candidate": candidate, "control": control,
            "quality_reports_verified": False, "training_performed": False,
            "boundary": "仅固定上层已验收描述的抽样；不验证真实质量、标签或冻结隔离，不授予训练资格"}
    plan["sha256"] = paired_plan_sha256(plan)
    return plan
