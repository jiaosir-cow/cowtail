"""Render an evidence-bounded Chinese six-cow report from local scored results.

No network, training, checkpoint selection, or result deletion is performed.
The complete 6 x 2 x 3 x 2 matrix and verified local-transfer receipt are
required before any report output is created. Run --help for input details.
"""
from __future__ import annotations

import argparse
import base64
import csv
from datetime import datetime, timezone
import hashlib
import html
import io
import itertools
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
COWS = tuple("ABCDEF")
IDS = dict(zip(COWS, ("COWID_04", "COWID_05", "COWID_01", "COWID_06", "COWID_07", "COWID_08")))
METHODS = ("scratch", "act", "ufro")
METHOD_NAMES = {"scratch": "Scratch", "act": "ACT", "ufro": "UFRO"}
STEPS = (1000, 3000)
BUDGETS = (10, 100)
WEIGHTINGS = ("window", "interval_balanced_window")
WEIGHT_NAMES = {"window": "窗口等权（主评价）", "interval_balanced_window": "区间平衡窗口权重（敏感性）"}
STATES = ("walking", "standing", "lying", "feeding")
STATE_NAMES = {"walking": "行走", "standing": "站立", "lying": "卧躺", "feeding": "采食"}
METRICS = {
    "macro_f1_fixed4": "固定四类 Macro-F1",
    "macro_f1_supported": "支持类 Macro-F1",
    "accuracy": "准确率",
    "mcc4_diagnostic": "四输出类别MCC（诊断）",
    "mcc4_complete_only": "完整四类MCC",
    "walking_ap": "Walking AP",
    "predictions_absent_class_fraction": "预测为缺真值类别的比例",
}
SCOPES = (
    ("all6_fixed4", "全六牛／固定四类诊断", COWS, "macro_f1_fixed4"),
    ("ad_fixed4", "A–D／完整四类子集", tuple("ABCD"), "macro_f1_fixed4"),
    ("all6_supported", "全六牛／各牛支持类", COWS, "macro_f1_supported"),
)
PAIRS = (("act", "scratch"), ("ufro", "scratch"), ("act", "ufro"))
EPSILON = 1e-12


class NotReady(RuntimeError):
    """Actual, complete, verified input evidence is not available."""


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise NotReady(f"缺少输入文件：{path}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        raise NotReady(f"缺少评分文件：{path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        result = list(csv.DictReader(handle))
    if not result:
        raise NotReady(f"评分文件为空：{path}")
    return result


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def first(row: dict, *names: str, default=None):
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return default


def cow_letter(value) -> str:
    text = str(value).strip()
    if text in IDS.values():
        return next(cow for cow, identity in IDS.items() if identity == text)
    text = text.replace("Cow ", "").replace("cow_", "").strip().upper()
    if text not in COWS:
        raise ValueError(f"未知牛别名：{value!r}")
    return text


def budget_percent(value) -> int:
    text = str(value).strip().removesuffix("%")
    number = float(text)
    if 0 < number <= 1:
        number *= 100
    if not math.isclose(number, round(number), abs_tol=1e-9) or round(number) not in BUDGETS:
        raise ValueError(f"非预设预算：{value!r}")
    return round(number)


def method_name(value) -> str:
    text = str(value).strip().lower().replace("-", "").replace("_", "")
    names = {"scratch": "scratch", "act": "act", "actbecalf": "act", "ufro": "ufro"}
    if text not in names:
        raise ValueError(f"未知方法：{value!r}")
    return names[text]


def checked_score(value, label) -> float:
    if value in (None, ""):
        if label[1] in {"mcc4_complete_only", "walking_ap"}:
            return None
        raise ValueError(f"{label}不允许缺失值。")
    result = float(value)
    lower = -1 if label[1].startswith("mcc") else 0
    if not math.isfinite(result) or not lower <= result <= 1:
        raise ValueError(f"{label} 必须为[{lower},1]有限实数，实际为 {value!r}")
    return result


def assert_local(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(ROOT.resolve()):
        raise ValueError(f"报告路径须位于本次工作目录 {ROOT} 内：{resolved}")
    return resolved


def load_inputs(analysis: Path, receipt_name: str, metrics_name: str, support_name: str):
    """Validate inputs before writing reports; schema aliases do not fill missing data."""
    transfer_path = ROOT / "LOCAL_TRANSFER_VERIFIED.json"
    transfer = read_json(transfer_path)
    if (transfer.get("state") != "ALL_RESULTS_TRANSFERRED_AND_HASH_VERIFIED"
            or transfer.get("formal_units") != 36 or transfer.get("endpoints") != 72
            or transfer.get("downloaded_after_all_units_complete") is not True):
        raise NotReady("本机完整下载验证回执尚未确认36个正式单元与72个固定终点。")
    receipt_path = analysis / receipt_name
    receipt = read_json(receipt_path)
    state = first(receipt, "state", "status")
    if state not in {"PASS", "COMPLETE", "SCORING_COMPLETE", "COMPLETE_VERIFIED_LOCAL", "ANALYSIS_COMPLETE"}:
        raise NotReady(f"评分回执未通过：{state!r}")
    if (receipt.get("formal_units") != 36 or receipt.get("endpoint_count") != 72
            or receipt.get("metric_row_count") != 144 or receipt.get("per_class_row_count") != 576
            or receipt.get("identities_independently_verified") is not False
            or receipt.get("computational_repetitions") != 1 or receipt.get("replicate_index") != 0
            or receipt.get("primary_endpoint") != "step1000/window"
            or receipt.get("identity_basis") != "user_attested_field_records_2026-10-18"
            or receipt.get("legacy_recording_groups_overlap_train_test") is not True):
        raise NotReady("评分回执未确认36单元/72终点/144指标行/576分类行/单次seed0及身份边界。")
    if (not transfer.get("archive_sha256") or transfer["archive_sha256"] != receipt.get("final_archive_sha256")
            or Path(transfer.get("bundle", "")).resolve() != Path(receipt.get("bundle", "")).resolve()):
        raise ValueError("本机传输回执与评分审计没有绑定到同一归档和解压目录。")
    source_paths = [transfer_path, receipt_path, analysis / metrics_name, analysis / support_name, analysis / "per_class.csv"]
    rows = read_csv(source_paths[2])
    metrics: dict[tuple, dict[str, float]] = {}
    metric_aliases = {
        "f1_macro_fixed4": "macro_f1_fixed4", "macro_f1_4class": "macro_f1_fixed4",
        "f1_macro_supported": "macro_f1_supported", "macro_f1_present": "macro_f1_supported",
        **{metric: metric for metric in METRICS},
    }
    for row in rows:
        cow = cow_letter(first(row, "cow_alias", "cow", "held_out_cow", "heldout_alias"))
        budget = budget_percent(first(row, "budget_percent", "budget", "budget_fraction"))
        step = int(first(row, "step", "steps", "endpoint_step"))
        if step not in STEPS:
            raise ValueError(f"非预设终点：{step}")
        method = method_name(first(row, "method", "initialization"))
        weighting = first(row, "weighting")
        if weighting not in WEIGHTINGS:
            raise ValueError(f"未知评分权重：{weighting!r}")
        expected_endpoint = "PRIMARY" if step == 1000 else "SENSITIVITY"
        if row.get("endpoint") != expected_endpoint:
            raise ValueError(f"评分行终点属性不符：{row.get('endpoint')}")
        seed = first(row, "seed", "replicate", default="0")
        if int(seed) != 0:
            raise ValueError(f"仅允许冻结的seed=0，实际为 {seed}")
        identity = first(row, "corrected_id", "cow_id", "held_out_id", "test_cow")
        if identity is not None and identity != IDS[cow]:
            raise ValueError(f"{cow}的身份字段与冻结映射不一致：{identity}")
        key = (cow, budget, step, method, weighting)
        record = metrics.setdefault(key, {})
        if "metric" in row:
            metric = metric_aliases.get(row["metric"])
            if metric is None:
                continue
            if metric in record:
                raise ValueError(f"重复评分行：{key}, {metric}")
            record[metric] = checked_score(first(row, "value", "score"), (key, metric))
        else:
            for source, metric in metric_aliases.items():
                if source in row:
                    if metric in record:
                        raise ValueError(f"重复指标：{key}, {metric}")
                    record[metric] = checked_score(row[source], (key, metric))
    expected = set(itertools.product(COWS, BUDGETS, STEPS, METHODS, WEIGHTINGS))
    if set(metrics) != expected:
        raise NotReady(f"评分矩阵不完整；应有144行双权重配置，实际{len(metrics)}；缺少{sorted(expected-set(metrics))}")
    required_metrics = set(METRICS)
    for key, record in metrics.items():
        if not required_metrics <= record.keys():
            raise NotReady(f"{key}缺少指标：{sorted(required_metrics-record.keys())}")
    common_metrics = set.intersection(*(set(record) for record in metrics.values()))
    if any(set(record) != common_metrics for record in metrics.values()):
        raise ValueError("部分配置缺少指标，不能仅展示可用配置。")
    support: dict[tuple[str, str], int] = {}
    identities = {cow: {"legacy_id": "见原始来源清单", "identity_basis": "user_attested_field_records_2026-10-18"} for cow in COWS}
    for row in read_csv(source_paths[3]):
        cow = cow_letter(first(row, "cow_alias", "cow", "held_out_cow", "heldout_alias"))
        if first(row, "legacy_id", "legacy_ids", "legacy_cow_id") is not None:
            identities[cow]["legacy_id"] = str(first(row, "legacy_id", "legacy_ids", "legacy_cow_id"))
        if row.get("cow_id") != IDS[cow]:
            raise ValueError(f"类别支持量表中的身份映射与冻结值不符：Cow {cow}")
        for field in ("interval_count", "session_count", "window_count"):
            identities[cow][field] = int(row[field])
        if row.get("identity_basis"):
            identities[cow]["identity_basis"] = row["identity_basis"]
        if "state" in row or "class" in row:
            state_name = str(first(row, "state", "class")).lower()
            if state_name not in STATES:
                raise ValueError(f"未知状态：{state_name}")
            counts = [(state_name, first(row, "n", "support", "count", "n_windows"))]
        else:
            counts = [(state_name, first(row, state_name, f"n_{state_name}", f"support_{state_name}", f"{state_name.upper()}_windows")) for state_name in STATES]
        for state_name, count in counts:
            value = float(count)
            if not math.isfinite(value) or value < 0 or value != int(value):
                raise ValueError(f"无效类别支持量：{row}")
            key = (cow, state_name)
            if key in support and support[key] != int(value):
                raise ValueError(f"{key}出现不一致的类别支持量。")
            support[key] = int(value)
    if set(support) != set(itertools.product(COWS, STATES)):
        raise NotReady("缺少六牛完整四类支持量（包含零支持类别）。")
    if any(support[(cow, state)] <= 0 for cow in "ABCD" for state in STATES):
        raise ValueError("A–D完整四类子集与实际支持量不符。")
    if support[("E", "feeding")] != 0 or support[("F", "lying")] != 0:
        raise ValueError("E缺feeding、F缺lying的冻结支持量边界与实际数据不符。")
    for cow in COWS:
        if sum(support[(cow, state)] for state in STATES) <= 0:
            raise ValueError(f"Cow {cow}没有评分真值窗口。")
        if sum(support[(cow, state)] for state in STATES) != identities[cow]["window_count"]:
            raise ValueError(f"Cow {cow}的四类支持量与总窗口数不一致。")
    for row in rows:
        cow = cow_letter(row["cow_alias"])
        observed = sum(support[(cow, state)] > 0 for state in STATES)
        if (int(row["test_window_count"]) != sum(support[(cow, state)] for state in STATES)
                or int(row["test_event_count"]) != identities[cow]["interval_count"]
                or int(row["observed_classes"]) != observed):
            raise ValueError(f"Cow {cow}评分行的真值支持量与覆盖表不一致。")
        key = (cow, budget_percent(row["budget_percent"]), int(row["step"]), method_name(row["method"]), row["weighting"])
        complete = metrics[key]["mcc4_complete_only"]
        if (observed == 4) != (complete is not None):
            raise ValueError(f"Cow {cow}的完整四类MCC缺失约定不符。")
    class_rows = read_csv(source_paths[4])
    expected_classes = set(itertools.product(COWS, BUDGETS, STEPS, METHODS, WEIGHTINGS, STATES))
    seen_classes = set()
    for row in class_rows:
        cow = cow_letter(row["cow_alias"])
        state_name = row["class_name"].lower()
        key = (cow, budget_percent(row["budget_percent"]), int(row["step"]), method_name(row["method"]), row["weighting"], state_name)
        if key not in expected_classes or key in seen_classes:
            raise ValueError(f"未知或重复分类评分行：{key}")
        seen_classes.add(key)
        present = support[(cow, state_name)] > 0
        if int(row["raw_window_support"]) != support[(cow, state_name)]:
            raise ValueError(f"分类支持量不符：{key}")
        if str(row["truth_present"]).lower() != str(present).lower():
            raise ValueError(f"分类真值存在标记不符：{key}")
        if not present and (row["recall"] not in (None, "") or float(row["f1_fixed_convention"]) != 0):
            raise ValueError(f"缺类recall/F1约定不符：{key}")
        for field in ("precision", "f1_fixed_convention"):
            checked_score(row[field], (key, field))
        if present:
            checked_score(row["recall"], (key, "recall"))
    if seen_classes != expected_classes:
        raise NotReady(f"分类明细不完整，应有576行，实际{len(seen_classes)}。")
    for field in ("artifact_sha256", "output_sha256", "output_hashes"):
        hashes = receipt.get(field)
        if not isinstance(hashes, dict):
            continue
        for relative, expected_digest in hashes.items():
            path = (analysis / relative).resolve()
            if not path.is_relative_to(analysis.resolve()) or not path.is_file():
                raise ValueError(f"评分回执引用了无效输出：{relative}")
            if digest(path) != expected_digest:
                raise ValueError(f"评分输出哈希不符：{relative}")
    outputs = receipt.get("outputs")
    if not isinstance(outputs, dict) or not outputs:
        raise ValueError("AUDIT.json缺少评分输出文件哈希。")
    audited_paths = {(analysis / relative).resolve() for relative in outputs}
    if not set(path.resolve() for path in source_paths[2:5]) <= audited_paths:
        raise ValueError("实际消费的评分、覆盖量和分类CSV必须全部列入AUDIT.json输出哈希。")
    for relative, description in outputs.items():
        path = (analysis / relative).resolve()
        if not path.is_relative_to(analysis) or not path.is_file() or digest(path) != description["sha256"]:
            raise ValueError(f"AUDIT.json评分输出哈希不符：{relative}")
        if path not in source_paths:
            source_paths.append(path)
    return metrics, support, identities, transfer, receipt, source_paths, common_metrics, class_rows


def pct(value: float) -> str:
    return f"{100 * value:.2f}"


def pp(value: float) -> str:
    return f"{100 * value:+.2f}"


def metric_text(value, metric) -> str:
    if value is None:
        return "NA（无完整真值支持）"
    return f"{value:.4f}" if metric.startswith("mcc") else pct(value)


def sign_counts(values) -> tuple[int, int, int]:
    values = list(values)
    return (sum(value > EPSILON for value in values),
            sum(abs(value) <= EPSILON for value in values),
            sum(value < -EPSILON for value in values))


def paired(metrics, cows, budget, step, metric, left, right):
    return [metrics[(cow, budget, step, left)][metric] - metrics[(cow, budget, step, right)][metric] for cow in cows]


def table(headers: list, rows: list, caption: str = "") -> dict:
    return {"type": "table", "headers": headers, "rows": rows, "caption": caption}


def paragraph(text: str) -> dict:
    return {"type": "paragraph", "text": text}


def headline(text: str, level: int = 2) -> dict:
    return {"type": "heading", "text": text, "level": level}


def build_blocks(metrics, support, identities, transfer, receipt, common_metrics) -> list[dict]:
    blocks = [
        paragraph("本报告只在全部36个正式训练单元、72个固定终点评分及本机传输验证通过后生成。所有配置均来自本次本机评分；1000步为预设主终点，3000步为敏感性终点。未按测试集表现重新选择终点、预算、牛只或类别权重。"),
        headline("本轮能够支持的结论"),
        paragraph("以下差值是同一留出牛、同一预算、同一固定终点下的配对结果。均值按牛等权平均，正／零／负表示每头牛的差值方向。单次seed=0不支持随机重复层面的推断，本报告不计算显著性或置信区间。"),
    ]
    for budget in BUDGETS:
        for left, right in PAIRS:
            values = paired(metrics, tuple("ABCD"), budget, 1000, "macro_f1_fixed4", left, right)
            positive, zero, negative = sign_counts(values)
            blocks.append(paragraph(
                f"主终点1000步、{budget}%标签预算、A–D完整四类子集：{METHOD_NAMES[left]} − {METHOD_NAMES[right]} 的固定四类Macro-F1均值差为 {pp(statistics.mean(values))} 个百分点；4头牛中提高{positive}、持平{zero}、下降{negative}。"))
    blocks.extend([
        paragraph("这些结果检验整体初始化在已设预算下的适应表现。正均值与负均值、个体提高与下降均保留；不能据此声称普遍提高、真实所需标签减少或人工标注时间减少。"),
        headline("身份、来源与类别覆盖"),
        paragraph("六组数据按用户现场记录声明为六头牛。原始资料曾以两个历史ID组织，后按会话／分组拆分并纠正身份；legacy ID、原始行索引与会话来源须随原始数据清单保留。身份依据为 user_attested_field_records_2026-10-18，未被独立核准。数据完整性通过与身份独立核准是不同事项；本轮不能表述为六牛独立外部验证。"),
    ])
    identity_rows = []
    all_windows = sum(support.values())
    for cow in COWS:
        counts = [support[(cow, state)] for state in STATES]
        missing = "、".join(STATE_NAMES[state] for state in STATES if not support[(cow, state)]) or "无"
        identity_rows.append([f"Cow {cow}", IDS[cow], identities[cow]["legacy_id"], identities[cow]["session_count"], identities[cow]["interval_count"], *counts,
                              sum(counts), f"{100*sum(counts)/all_windows:.2f}%", missing])
    blocks.append(table(["牛别名", "用户声明身份", "legacy来源", "会话数", "原区间数", *[STATE_NAMES[state] for state in STATES], "窗口合计", "窗口占比", "缺失真值类别"], identity_rows,
                        "类别支持量为留出测试真值窗口数。窗口占比描述数据不均衡；结果汇总按牛等权，不按窗口数加权。窗口之间不能视为独立动物重复。"))
    blocks.extend([
        paragraph("Cow E没有采食真值，Cow F没有卧躺真值。全六牛固定四类Macro-F1是预先统一类别集合的诊断口径：无真值类别按评分脚本约定计入；它不等价于E/F的完整四类验证。A–D子集四类均有真值。支持类口径按每牛真值中出现的类别计算，E/F只涵盖三个类别，不同牛之间的类别集合并不完全相同。"),
        paragraph("本轮按修订后的用户声明身份留出；历史legacy记录组在训练和测试之间存在重叠。因此不能把本轮改称按历史记录组完全独立的外部验证，也不能据此断言视频、会话或采集条件均独立。"),
        headline("实验范围与统一评分口径"),
        table(["项目", "本轮设置"], [
            ["正式矩阵", "6个留出牛 × 2个标签预算（10%、100%） × 3种方法 × 1次seed=0 = 36个训练单元"],
            ["方法", "Scratch、ACT（ActBeCalf来源初始化）、UFRO来源初始化"],
            ["固定终点", "1000步：预设主终点；3000步：敏感性终点；共72个已完成终点"],
            ["汇总", "每牛先独立评分，再按牛等权平均；所有配对均保持留出牛、预算、终点相同"],
            ["两种评分权重", "window：每窗口等权，作为主评价；interval_balanced_window：每窗口权重为该原始区间保留窗口数的倒数，每原区间总权重为1，作为敏感性口径；这不是区间分类"],
            ["评分范围", "全六牛固定四类诊断、A–D完整四类子集、全六牛各牛支持类；三种口径并列"],
            ["结果单位", "分数以百分数展示；配对差值以百分点展示；CSV保留[0,1]原始数值"],
        ]),
        headline("全部配置的均值与配对方向"),
    ])
    for step in STEPS:
        blocks.append(headline(f"{step}步：{'预设主终点' if step == 1000 else '敏感性终点'}", 3))
        rows = []
        for _, title, cows, metric in SCOPES:
            for budget in BUDGETS:
                means = [statistics.mean(metrics[(cow, budget, step, method)][metric] for cow in cows) for method in METHODS]
                cells = []
                for left, right in PAIRS:
                    values = paired(metrics, cows, budget, step, metric, left, right)
                    positive, zero, negative = sign_counts(values)
                    cells.append(f"{pp(statistics.mean(values))}（{positive}/{zero}/{negative}）")
                rows.append([title, f"{budget}%", len(cows), *map(pct, means), *cells])
        blocks.append(table(["范围／指标", "预算", "牛数", "Scratch %", "ACT %", "UFRO %",
                             "ACT−Scratch pp（+/0/−）", "UFRO−Scratch pp（+/0/−）", "ACT−UFRO pp（+/0/−）"], rows,
                            "每格差值后的括号为按牛统计的提高／持平／下降数；方向以未舍入原值判定（数值容差1e−12）。所有方法使用相同的预算与测试真值。"))
    blocks.extend([
        headline("MCC、准确率与Walking AP"),
        paragraph("MCC保留四个输出类别；全六牛支持类F1的口径不会重新定义或删除MCC输出类别。完整四类MCC只在A–D显示；E/F不赋予完整四类MCC。Walking AP是本轮适应后输出的类别指标，不是C1所需的source-only直接预测对照。MCC使用原始[−1,1]标度，其余分数用百分数。"),
    ])
    rows = []
    for _, title, cows, _ in SCOPES:
        for step, budget, method in itertools.product(STEPS, BUDGETS, METHODS):
            mcc_metric = "mcc4_complete_only" if cows == tuple("ABCD") else "mcc4_diagnostic"
            values = [statistics.mean(metrics[(cow, budget, step, method)][metric] for cow in cows)
                      for metric in (mcc_metric, "accuracy", "walking_ap")]
            rows.append([title, step, f"{budget}%", METHOD_NAMES[method], f"{values[0]:.4f}", pct(values[1]), pct(values[2])])
    blocks.append(table(["范围", "步数", "预算", "方法", "MCC", "准确率 %", "Walking AP %"], rows))
    blocks.extend([
        headline("每牛的全部配对差值"),
        paragraph("下表逐一展示两种预算、两个固定终点和六头牛。cowD*缺采食、COW_OTHER*缺卧躺；两牛固定四类行属于缺类诊断。负号即该配置下左方法低于右方法，不作删除或更换终点。"),
    ])
    for metric in ("macro_f1_fixed4", "macro_f1_supported"):
        rows = []
        for step, budget, cow in itertools.product(STEPS, BUDGETS, COWS):
            scores = [metrics[(cow, budget, step, method)][metric] for method in METHODS]
            deltas = [metrics[(cow, budget, step, left)][metric] - metrics[(cow, budget, step, right)][metric] for left, right in PAIRS]
            rows.append([step, f"{budget}%", f"Cow {cow}{'*' if cow in 'EF' else ''}", *map(pct, scores), *map(pp, deltas)])
        blocks.append(headline(METRICS[metric], 3))
        blocks.append(table(["步数", "预算", "留出牛", "Scratch %", "ACT %", "UFRO %",
                             "ACT−Scratch pp", "UFRO−Scratch pp", "ACT−UFRO pp"], rows))
    blocks.extend([
        headline("1000→3000步的变化"),
        paragraph("此表只描述延长训练后的变化。3000步结果不会替代预设1000步主终点，也不能按每牛或每方法挑选更高终点后重新汇总。"),
    ])
    rows = []
    for _, title, cows, metric in SCOPES:
        for budget, method in itertools.product(BUDGETS, METHODS):
            values = [metrics[(cow, budget, 3000, method)][metric] - metrics[(cow, budget, 1000, method)][metric] for cow in cows]
            positive, zero, negative = sign_counts(values)
            rows.append([title, f"{budget}%", METHOD_NAMES[method], pp(statistics.mean(values)), f"{positive}/{zero}/{negative}"])
    blocks.append(table(["范围／指标", "预算", "方法", "3000−1000步均值差 pp", "提高/持平/下降牛数"], rows))
    blocks.extend([
        headline("与论文三项贡献的证据边界"),
        table(["贡献", "本轮可以评价", "不能据此声称"], [
            ["C1：公开颈圈模型在不使用尾圈训练标签时提供walking提示", "本轮是有适应训练的初始化比较，不直接检验此项", "本轮重新验证了零尾圈训练标签的walking直接预测；六牛外部验证"],
            ["C2：复制编码器与walking源状态，fresh posture/feeding heads用于label refinement", "本轮比较ACT／UFRO整体初始化与Scratch在10%、100%预算的实际适应表现，逐牛正负结果见上表", "隔离证明复制walking状态或fresh heads的因果机制；实际标注需求或人工时间减少；所有牛或所有类别均改善"],
            ["C3：直接预测价值与初始化价值可能不同", "本轮可比较ACT、UFRO在相同预算与终点下的初始化差异", "新获得source-only直接预测对照；仅凭初始化结果独立重证直接预测与初始化价值的差异"],
        ]),
        paragraph("解释结果时应同时保留正向证据与限制性结果。10%预算只表示本协议规定的标签子集比例，并非真实标签需求曲线或节省人工时间的测量。六牛身份声明、会话来源、类别缺失和明显数据不均衡限制结论的外推。"),
        headline("全部逐牛评分明细"),
    ])
    ordered_metrics = [metric for metric in METRICS if metric in common_metrics]
    rows = []
    for step, budget, cow, method in itertools.product(STEPS, BUDGETS, COWS, METHODS):
        record = metrics[(cow, budget, step, method)]
        rows.append([step, f"{budget}%", f"Cow {cow}", METHOD_NAMES[method], *[metric_text(record[metric], metric) for metric in ordered_metrics]])
    blocks.append(table(["步数", "预算", "留出牛", "方法", *[METRICS[metric] + ("" if metric.startswith("mcc") else " %") for metric in ordered_metrics]], rows,
                        "这72行覆盖全部预设终点配置。四类Macro-F1使用统一四类；支持类指标使用该牛真值中有支持的类别。"))
    blocks.extend([
        headline("可追溯凭据"),
        table(["凭据", "值"], [
            ["本机下载状态", transfer["state"]],
            ["全部完成后下载", str(transfer["downloaded_after_all_units_complete"])],
            ["正式训练单元／终点", f"{transfer['formal_units']}／{transfer['endpoints']}"],
            ["原始归档SHA-256", transfer.get("archive_sha256", "未在传输回执提供")],
            ["本机下载验证时间UTC", transfer.get("verified_utc", "未提供")],
            ["评分回执状态", str(first(receipt, "state", "status"))],
            ["评分目录", str(receipt.get("analysis_directory", "见report_receipt.json来源文件列表"))],
            ["身份独立核准", "false；依据用户现场记录声明"],
            ["文件核对", "report_receipt.json记录输入与生成文件SHA-256；报告不会替代原评分CSV或原始回执"],
        ]),
    ])
    return blocks


def split_sections(blocks: list[dict]) -> tuple[list[dict], dict[str, list[dict]]]:
    intro, sections = [], {}
    current = intro
    for block in blocks:
        if block["type"] == "heading" and block["level"] == 2:
            current = sections.setdefault(block["text"], [])
        else:
            current.append(block)
    return intro, sections


def build_report(metrics, support, identities, transfer, receipt, common_metrics, class_rows):
    """Keep common evidence once and present both frozen weighting views together."""
    views = {weighting: {key[:4]: record for key, record in metrics.items() if key[4] == weighting}
             for weighting in WEIGHTINGS}
    sections = {}
    intro = []
    for weighting in WEIGHTINGS:
        this_intro, these_sections = split_sections(build_blocks(views[weighting], support, identities, transfer, receipt, common_metrics))
        intro = intro or this_intro
        sections[weighting] = these_sections
    shared = {"身份、来源与类别覆盖", "实验范围与统一评分口径", "与论文三项贡献的证据边界", "可追溯凭据"}
    blocks = intro + [paragraph("窗口等权（window）是主评价，区间平衡窗口权重（interval_balanced_window）作为敏感性口径并列展示。两种权重不互换，也不按哪种更正向决定结论；全部144条逐牛评分行与576条分类明细均保留。")]
    for title in sections[WEIGHTINGS[0]]:
        blocks.append(headline(title))
        if title in shared:
            blocks.extend(sections[WEIGHTINGS[0]][title])
            continue
        for weighting in WEIGHTINGS:
            blocks.append(headline(WEIGHT_NAMES[weighting], 3))
            for block in sections[weighting][title]:
                block = dict(block)
                if block["type"] == "heading":
                    block["level"] += 1
                blocks.append(block)
    blocks.extend([
        headline("全部分类明细"),
        paragraph("以下576行完整保留每种方法、预算、固定终点、权重和牛只的四类输出。缺真值类的Recall为空，固定类F1=0只是统一评分约定。预测为该牛缺真值类的窗口仍算错误；支持类口径没有删输出类别或重新归一化。分类明细中的支持量不是独立动物数。"),
    ])
    for weighting in WEIGHTINGS:
        rows = []
        ordered = sorted((row for row in class_rows if row["weighting"] == weighting),
                         key=lambda row: (int(row["step"]), budget_percent(row["budget_percent"]), cow_letter(row["cow_alias"]), METHODS.index(method_name(row["method"])), int(row["class_id"])))
        for row in ordered:
            present = str(row["truth_present"]).lower() == "true"
            rows.append([row["step"], f"{row['budget_percent']}%", row["cow_alias"], METHOD_NAMES[method_name(row["method"])],
                         STATE_NAMES[row["class_name"].lower()], row["raw_window_support"], f"{float(row['weighted_support']):.3f}",
                         pct(float(row["precision"])), pct(float(row["recall"])) if present else "NA（无真值）", pct(float(row["f1_fixed_convention"]))])
        blocks.append(headline(WEIGHT_NAMES[weighting], 3))
        blocks.append(table(["步数", "预算", "牛", "方法", "类别", "原始窗口支持量", "加权支持量", "Precision %", "Recall %", "固定约定F1 %"], rows))
    return blocks, views


def render_markdown(blocks: list[dict], figures: dict[str, bytes]) -> str:
    parts = ["# 六牛跨部位初始化实验：完整结果与证据边界\n"]
    for filename in figures:
        weighting = next(weighting for weighting in WEIGHTINGS if filename == f"paired_differences_{weighting}.svg")
        parts.append(f"![{WEIGHT_NAMES[weighting]}的全部固定终点每牛配对差值]({filename})\n")
        parts.append(f"图（{WEIGHT_NAMES[weighting]}）：正值表示左方法高于Scratch；四幅图共享零中心色标，单位为百分点。1000步为主终点，3000步为敏感性终点。E*、F*分别缺采食与卧躺真值。\n")
    for block in blocks:
        if block["type"] == "heading":
            parts.append("#" * block["level"] + " " + block["text"] + "\n")
        elif block["type"] == "paragraph":
            parts.append(block["text"] + "\n")
        elif block["type"] == "table":
            escape = lambda value: str(value).replace("|", "\\|").replace("\n", "<br>")
            parts.append("| " + " | ".join(map(escape, block["headers"])) + " |")
            parts.append("| " + " | ".join("---" for _ in block["headers"]) + " |")
            parts.extend("| " + " | ".join(map(escape, row)) + " |" for row in block["rows"])
            if block["caption"]:
                parts.append("\n" + block["caption"])
            parts.append("")
    return "\n".join(parts)


def render_html(blocks: list[dict], figures: dict[str, bytes]) -> str:
    escape = lambda value: html.escape(str(value), quote=True)
    parts = []
    for filename, figure in figures.items():
        weighting = next(weighting for weighting in WEIGHTINGS if filename == f"paired_differences_{weighting}.svg")
        encoded = base64.b64encode(figure).decode("ascii")
        parts.append(f'<figure><h2>{escape(WEIGHT_NAMES[weighting])}：逐牛差值图</h2><img alt="全部固定终点的每牛配对差值" src="data:image/svg+xml;base64,{encoded}"><figcaption>每格为ACT或UFRO减Scratch的Macro-F1差值（百分点）；四图共享零中心色标。1000步为主终点；3000步为敏感性终点。E*缺采食、F*缺卧躺真值。</figcaption></figure>')
    navigation = []
    for index, block in enumerate(blocks):
        if block["type"] == "heading":
            level = block["level"]
            parts.append(f'<h{level} id="section-{index}">{escape(block["text"])}</h{level}>')
            if level == 2:
                navigation.append(f'<a href="#section-{index}">{escape(block["text"])}</a>')
        elif block["type"] == "paragraph":
            parts.append(f'<p>{escape(block["text"])}</p>')
        elif block["type"] == "table":
            header = "".join(f"<th>{escape(value)}</th>" for value in block["headers"])
            rows = []
            for row in block["rows"]:
                cells = []
                for value in row:
                    text = str(value)
                    css = ""
                    if text.startswith(("+", "-")):
                        try:
                            number = float(text.split("（")[0])
                        except ValueError:
                            pass
                        else:
                            css = ' class="positive"' if number > 0 else (' class="negative"' if number < 0 else "")
                    cells.append(f"<td{css}>{escape(text)}</td>")
                rows.append("<tr>" + "".join(cells) + "</tr>")
            parts.append('<div class="table-wrap"><table><thead><tr>' + header + '</tr></thead><tbody>' + "".join(rows) + '</tbody></table></div>')
            if block["caption"]:
                parts.append(f'<p class="caption">{escape(block["caption"])}</p>')
    css = """
    :root{color-scheme:light;--ink:#172b3d;--muted:#536777;--line:#d9e3eb;--blue:#0f4d92}
    *{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;color:var(--ink);background:#f2f6f9;font:16px/1.8 'Microsoft YaHei','PingFang SC',sans-serif}
    header{background:#102f4e;color:white;padding:45px max(4vw,calc((100vw - 1280px)/2)) 38px}header .tag{font-size:12px;letter-spacing:2px;color:#b9d1e5}h1{font-size:clamp(25px,3vw,38px);line-height:1.35;margin:10px 0}header p{max-width:1000px;margin:12px 0 0;color:#d7e4ee}
    main{max-width:1360px;margin:24px auto;padding:0 32px 50px}nav{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 24px}nav a{background:white;border:1px solid var(--line);border-radius:5px;padding:5px 12px;text-decoration:none;color:var(--blue);font-size:13px}
    article{background:white;border:1px solid var(--line);border-radius:8px;padding:30px 34px}h2{font-size:23px;border-top:1px solid var(--line);padding-top:27px;margin:38px 0 15px}h3{font-size:18px;margin:24px 0 10px}p{margin:12px 0}.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:5px;margin:15px 0 9px}table{border-collapse:collapse;width:100%;font-size:13px;line-height:1.55;font-variant-numeric:tabular-nums}th,td{padding:10px 12px;text-align:left;border-bottom:1px solid #e5edf3;vertical-align:top}th{white-space:nowrap;background:#edf3f8;color:#254158;font-weight:600}tr:nth-child(even){background:#f8fafc}tr:last-child td{border-bottom:0}td.positive{color:#0e6657;font-weight:600}td.negative{color:#a3383c;font-weight:600}.caption,figcaption{color:var(--muted);font-size:13px}figure{margin:0 0 24px}figure img{display:block;width:100%;height:auto}figcaption{margin-top:10px}footer{color:var(--muted);font-size:12px;margin-top:20px}
    @media(max-width:650px){main{padding:0 10px}article{padding:14px}header{padding:28px 20px}th,td{padding:8px}nav{gap:5px}}
    @media print{body{background:white;font-size:10pt}header{padding:0;color:#172b3d;background:white}header p{color:#536777}main{margin:0;max-width:none;padding:0}nav{display:none}article{border:0;padding:0}h2,h3{break-after:avoid}.table-wrap{overflow:visible}table{font-size:8pt}tr{break-inside:avoid}figure{break-inside:avoid}}
    """
    return ('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<meta name="color-scheme" content="light"><title>六牛初始化实验完整结果</title><style>' + css + '</style></head><body>'
            '<header><div class="tag">LOCAL VERIFIED RESULTS · FIXED ENDPOINTS</div><h1>六牛跨部位初始化实验</h1><p>完整结果、每牛配对差值与论文证据边界 · 1000步主终点 / 3000步敏感性 · 36个训练单元 / 72个终点</p></header><main><nav>'
            + "".join(navigation) + '</nav><article>' + "".join(parts)
            + '</article><footer>报告与图均由实际评分CSV生成。输入和生成文件的SHA-256见同目录report_receipt.json。全部144条逐牛双权重评分、576条分类明细与历史来源随评分包保留。</footer></main></body></html>')


def make_figure(metrics, weighting, color_limit) -> tuple[bytes | None, bytes | None, str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
        import numpy as np
    except ImportError as exc:
        return None, None, f"可选绘图库不可用，保留全部表格：{exc}"
    arrays = []
    columns = ((10, "act"), (100, "act"), (10, "ufro"), (100, "ufro"))
    for step, metric in itertools.product(STEPS, ("macro_f1_fixed4", "macro_f1_supported")):
        arrays.append(np.array([[100*(metrics[(cow, budget, step, method)][metric] - metrics[(cow, budget, step, "scratch")][metric])
                                 for budget, method in columns] for cow in COWS]))
    maximum = color_limit
    cmap = LinearSegmentedColormap.from_list("blue_white_red", ["#b64342", "#ffffff", "#0f4d92"])
    style = {"font.family": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 11,
             "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 1.2,
             "svg.fonttype": "none", "legend.frameon": False}
    with plt.rc_context(style):
        figure, axes = plt.subplots(2, 2, figsize=(13.5, 10.0), layout="constrained")
        for index, (step, metric) in enumerate(itertools.product(STEPS, ("macro_f1_fixed4", "macro_f1_supported"))):
            axis = axes.flat[index]
            values = arrays[index]
            image = axis.imshow(values, cmap=cmap, norm=TwoSlopeNorm(0, -maximum, maximum), aspect="auto")
            axis.set_xticks(range(4), ["ACT 10%", "ACT 100%", "UFRO 10%", "UFRO 100%"])
            axis.set_yticks(range(6), [f"Cow {cow}{'*' if cow in 'EF' else ''}" for cow in COWS])
            endpoint = "Primary" if step == 1000 else "Sensitivity"
            label = "Fixed four classes" if metric == "macro_f1_fixed4" else "Supported classes"
            axis.set_title(f"{endpoint}: step {step} | {label}", loc="left", fontsize=12, pad=12)
            axis.tick_params(length=0, pad=7)
            axis.spines[:].set_visible(False)
            axis.set_xticks(np.arange(-.5, 4, 1), minor=True)
            axis.set_yticks(np.arange(-.5, 6, 1), minor=True)
            axis.grid(which="minor", color="white", linewidth=2)
            axis.tick_params(which="minor", bottom=False, left=False)
            for y, x in itertools.product(range(6), range(4)):
                value = values[y, x]
                color = "white" if abs(value) > .6 * maximum else "#172b3d"
                axis.text(x, y, f"{value:+.2f}", ha="center", va="center", color=color, fontsize=11)
        figure.colorbar(image, ax=axes, shrink=.88, pad=.025, label="Macro-F1 difference vs Scratch (percentage points)")
        figure.suptitle(f"All cows, budgets and fixed endpoints | seed 0 | {weighting}", fontsize=15)
        svg, png = io.BytesIO(), io.BytesIO()
        figure.savefig(svg, format="svg", facecolor="white", dpi=300)
        figure.savefig(png, format="png", facecolor="white", dpi=300)
        plt.close(figure)
    return svg.getvalue(), png.getvalue(), "matplotlib SVG/PNG；全部四幅图共享零中心色标"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", required=True, type=Path, help="评分输出目录（本次E盘工作目录内）")
    parser.add_argument("--output", type=Path, help="报告目录；默认<analysis>/report")
    parser.add_argument("--receipt", default="AUDIT.json", help="评分回执文件名")
    parser.add_argument("--metrics", default="metrics.csv", help="逐牛双权重评分CSV（完整144行）")
    parser.add_argument("--supports", default="identity_and_coverage.csv", help="身份来源及四类真值支持量CSV")
    parser.add_argument("--skip-figures", action="store_true", help="仅生成完整表格报告，不调用可选matplotlib")
    args = parser.parse_args(argv)
    analysis = assert_local(args.analysis)
    output = assert_local(args.output or analysis / "report")
    try:
        metrics, support, identities, transfer, receipt, source_paths, common_metrics, class_rows = load_inputs(
            analysis, args.receipt, args.metrics, args.supports)
    except NotReady as exc:
        print(json.dumps({"state": "NOT_READY", "reason": str(exc), "report_written": False}, ensure_ascii=False))
        return 2
    blocks, views = build_report(metrics, support, identities, transfer, receipt, common_metrics, class_rows)
    figures, figure_files, figure_status = {}, {}, {}
    color_limit = max(1.0, max(abs(100*(view[(cow, budget, step, method)][metric] - view[(cow, budget, step, "scratch")][metric]))
                               for view in views.values() for cow, budget, step, method, metric in
                               itertools.product(COWS, BUDGETS, STEPS, ("act", "ufro"), ("macro_f1_fixed4", "macro_f1_supported"))))
    for weighting in WEIGHTINGS:
        svg, png, status = (None, None, "按参数跳过；全部数据保留在报告表格中") if args.skip_figures else make_figure(views[weighting], weighting, color_limit)
        figure_status[weighting] = status
        if svg:
            filename = f"paired_differences_{weighting}.svg"
            figures[filename] = svg
            figure_files[filename] = svg
            figure_files[f"paired_differences_{weighting}.png"] = png
    markdown = render_markdown(blocks, figures)
    page = render_html(blocks, figures)
    output.mkdir(parents=True, exist_ok=True)
    files = {"六牛实验完整结果.md": markdown.encode("utf-8-sig"), "六牛实验完整结果.html": page.encode("utf-8"), **figure_files}
    for name, contents in files.items():
        (output / name).write_bytes(contents)
    report_receipt = {
        "state": "REPORT_COMPLETE_FROM_VERIFIED_SCORES",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_directory": str(analysis), "output_directory": str(output),
        "formal_units": 36, "endpoints": 72, "primary_step": 1000, "sensitivity_step": 3000,
        "seed": 0, "test_selected_endpoint": False, "complete_matrix_required": True,
        "metric_rows": 144, "per_class_rows": 576, "weightings": list(WEIGHTINGS),
        "aggregation": "equal_cow_weight", "sign_tolerance": EPSILON,
        "scopes": [scope[0] for scope in SCOPES], "metrics": sorted(common_metrics),
        "identity_basis": "user_attested_field_records_2026-10-18", "identities_independently_verified": False,
        "claim_boundaries": {"C1": "not_tested_by_adaptation", "C2": "whole_initialization_not_mechanism_or_label_savings", "C3": "initialization_comparison_without_new_source_only_contrast"},
        "figure_status": figure_status,
        "source_sha256": {str(path): digest(path) for path in source_paths},
        "output_sha256": {name: digest(output / name) for name in files},
        "renderer_sha256": digest(Path(__file__)),
    }
    (output / "report_receipt.json").write_text(json.dumps(report_receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"state": report_receipt["state"], "output": str(output), "files": list(files) + ["report_receipt.json"], "figure_status": figure_status}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    try:
        raise SystemExit(main())
    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(json.dumps({"state": "INVALID_INPUT", "reason": str(exc), "report_written": False}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1)
