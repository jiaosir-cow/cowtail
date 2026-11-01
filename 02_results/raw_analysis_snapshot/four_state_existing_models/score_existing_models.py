"""Rescore archived, held-out, 1000-step four-state predictions. NO TRAINING.

Only this output directory is written. Source data, labels, receipts, and predictions
are read-only inputs. Legacy fold f01 is used for A/D/E, f02 for B1/C1/F1/G1.
The combined cohort is an out-of-fold collection, not a single checkpoint.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, matthews_corrcoef, precision_recall_fscore_support


OUT = Path(__file__).resolve().parent
BASE = OUT.parents[3]
SOURCE = BASE / "revision_label_budget_20260909"
DATASET = BASE / "datasets/DATASET_X"
STATES = ("standing", "lying", "walking", "feeding")
CN = {"standing": "站立", "lying": "卧躺", "walking": "行走", "feeding": "采食"}
NEW = ("cowA", "cowC", "cowD", "COW_OTHER", "cowB", "COW_OTHER", "COW_OTHER")
FOUR = ("COW_OTHER", "cowB", "COW_OTHER", "COW_OTHER")
OLD = ("cowA", "COW_OTHER", "COW_OTHER", "cowC", "cowD", "COW_OTHER")
BUDGETS = (5, 10, 25, 50, 100)
METHODS = ("scratch", "act", "ufro")
METRICS = ("accuracy", "macro_f1_four", "macro_f1_supported", "mcc")


def require(ok, message):
    if not bool(ok):
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def dump_json(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def fmt(value):
    return f"{100 * float(value):.2f}"


def pm(row, metric):
    return f"{fmt(row[metric + '_mean'])} ± {fmt(row[metric + '_std'])}"


def markdown_table(headers, rows):
    return "\n".join(["| " + " | ".join(map(str, headers)) + " |", "| " + " | ".join("---" for _ in headers) + " |"] + ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def score(y, pred, weights=None):
    supported = sorted(np.unique(y).tolist())
    precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=list(range(4)), zero_division=0, sample_weight=weights)
    measures = {
        "accuracy": float(accuracy_score(y, pred, sample_weight=weights)),
        "macro_f1_four": float(f1_score(y, pred, labels=list(range(4)), average="macro", zero_division=0, sample_weight=weights)),
        "macro_f1_supported": float(f1_score(y, pred, labels=supported, average="macro", zero_division=0, sample_weight=weights)),
        "mcc": float(matthews_corrcoef(y, pred, sample_weight=weights)),
        "supported_class_count": len(supported),
        "missing_truth_classes": ";".join(STATES[i] for i in range(4) if i not in supported),
    }
    return measures, precision, recall, f1, support, confusion_matrix(y, pred, labels=list(range(4)), sample_weight=weights)


def main():
    require(SOURCE.is_dir(), f"Missing source: {SOURCE}")
    inputs = {"original_windows": SOURCE / "inputs/windows.csv", "truth": SOURCE / "analysis/test_truth.csv",
              "new_mapping": DATASET / "windows.csv", "units": SOURCE / "configs/units.json"}
    input_hashes = {name: {"path": str(path), "sha256": sha(path)} for name, path in inputs.items()}
    original = pd.read_csv(inputs["original_windows"])
    truth = pd.read_csv(inputs["truth"])
    mapping = pd.read_csv(inputs["new_mapping"])
    units = load_json(inputs["units"])["units"]
    grid = set(itertools.product(("f01", "f02"), range(3), BUDGETS, METHODS))
    require(len(units) == 90 and {(v["fold_id"], v["replicate"], v["percent"], v["method"]) for v in units.values()} == grid, "Incomplete 90-unit grid")
    require(len(original) == len(truth) == 29790 and len(mapping) == 28911, "Unexpected window count")
    for frame in (original, truth, mapping):
        require(frame.row_index.is_unique and set(frame.row_index) == set(range(len(frame))), "Noncanonical row index")
        frame.sort_values("row_index", inplace=True)
        frame.reset_index(drop=True, inplace=True)
    require(original.window_id.is_unique and mapping.window_id.is_unique, "Duplicate window ID")
    source_indices = original.set_index("window_id").loc[mapping.window_id, "row_index"].to_numpy(dtype=np.int64)
    source_subset = original.iloc[source_indices].reset_index(drop=True)
    source_truth = truth.iloc[source_indices].reset_index(drop=True)
    source_to_mapping = np.full(len(original), -1, dtype=np.int64)
    source_to_mapping[source_indices] = np.arange(len(mapping))
    for field in ("window_id", "continuous_row_index", "event_id", "device_mac", "session_id"):
        require(source_subset[field].astype(str).equals(mapping[field].astype(str)), f"Window mapping mismatch: {field}")
    require(original.event_id.equals(truth.event_id), "Truth event mapping mismatch")
    require(original.cow_id.astype(str).equals(truth.cow_id.astype(str)), "Truth legacy mapping mismatch")
    require(np.array_equal(source_truth.target4.to_numpy(), mapping.target4.to_numpy()), "Relabeling detected")
    require(source_subset.cow_id.astype(str).equals(mapping.legacy_cow_id.astype(str)), "Legacy identity changed")
    require(set(mapping.analysis_group) == set(NEW) and set(mapping.cow_alias) == set(OLD), "Unexpected group labels")
    require(mapping.groupby(["device_mac", "session_id"]).analysis_group.nunique().max() == 1, "Session split across new groups")
    require(mapping.groupby("event_id").analysis_group.nunique().max() == 1, "Event split across groups")
    for old_name in ("cowA", "cowC", "cowD"):
        require(np.array_equal(mapping.cow_alias.eq(old_name), mapping.analysis_group.eq(old_name)), f"Preserved group changed: {old_name}")
    require(np.array_equal(mapping.cow_alias.isin(["COW_OTHER", "COW_OTHER", "COW_OTHER"]), mapping.analysis_group.isin(FOUR)), "ALL_POOL pool changed")
    y_all = source_truth.target4.to_numpy(dtype=np.int64)
    require(set(np.unique(y_all)) == {0, 1, 2, 3}, "Unexpected target states")

    manifests, predictions, training_support = [], {}, []
    for unit_id, unit in sorted(units.items()):
        fold, rep, budget, method = unit["fold_id"], int(unit["replicate"]), int(unit["percent"]), unit["method"]
        require(unit_id == f"{fold}_s{rep}_b{budget:03d}_{method}" and unit["unit_id"] == unit_id, "Unit name mismatch")
        directory = SOURCE / "download/bundle/results" / unit_id
        receipt = load_json(directory / "COMPLETE.json")
        require(receipt["status"] == "complete" and receipt["unit_id"] == unit_id, "Incorrect historical completion receipt")
        require(receipt["binding"]["unit_config_sha256"] == canonical_sha(unit), "Unit configuration binding mismatch")
        require(receipt["binding"]["artifact_sha256"]["configs/units.json"] == input_hashes["units"]["sha256"], "Unit grid binding mismatch")
        names = ["attempt_001/step1000/test_indices.npy", "attempt_001/step1000/test_state_probabilities.npy",
                 "attempt_001/step1000/test_head_probabilities.npy", "attempt_001/step1000/prediction_metadata.json",
                 "attempt_001/train_support.json", "attempt_001/initialization_audit.json"]
        for name in names:
            path = directory / name
            actual_hash = sha(path)
            require(actual_hash == receipt["artifact_sha256"][name], f"Prediction hash mismatch: {unit_id}/{name}")
            manifests.append({"unit_id": unit_id, "fold_id": fold, "method": method, "percent": budget, "replicate": rep,
                              "relative_path": name, "path": str(path), "sha256": actual_hash})
        endpoint = directory / "attempt_001/step1000"
        metadata = load_json(endpoint / "prediction_metadata.json")
        require(metadata["step"] == 1000 and metadata["endpoint"] == "PRIMARY" and tuple(metadata["state_order"]) == STATES, "Prediction definition changed")
        require(metadata["held_out_label_access"] is False and metadata["held_out_scoring"] is False, "Held-out label access declared")
        idx = np.load(endpoint / "test_indices.npy", allow_pickle=False)
        prob = np.load(endpoint / "test_state_probabilities.npy", allow_pickle=False)
        heads = np.load(endpoint / "test_head_probabilities.npy", allow_pickle=False)
        expected_cow = "COWID_02" if fold == "f01" else "COWID_03"
        expected_idx = np.flatnonzero(original.cow_id.eq(expected_cow).to_numpy())
        require(unit["test_cow"] == expected_cow, "Wrong legacy test animal")
        require(idx.dtype.kind in "iu" and idx.ndim == 1 and np.array_equal(np.sort(idx), expected_idx), "Held-out index coverage mismatch")
        require(prob.shape == (len(idx), 4) and heads.shape == (len(idx), 3, 2), "Prediction shape mismatch")
        require(np.isfinite(prob).all() and (prob >= -1e-7).all() and (prob <= 1 + 1e-7).all(), "Invalid probabilities")
        require(np.allclose(prob.sum(axis=1), 1, atol=2e-6, rtol=0), "State probability sum mismatch")
        require(metadata["test_window_count"] == len(idx) and metadata["test_indices_sha256"] == sha(endpoint / "test_indices.npy"), "Endpoint index metadata mismatch")
        train_idx_path = SOURCE / unit["budget_dir"] / "train_indices.npy"
        require(sha(train_idx_path) == receipt["binding"]["artifact_sha256"][f"{unit['budget_dir']}/train_indices.npy"], "Train index binding mismatch")
        train_idx = np.load(train_idx_path, allow_pickle=False)
        require(not np.intersect1d(idx, train_idx).size, "Training/test window leakage")
        require(not set(original.iloc[idx].event_id) & set(original.iloc[train_idx].event_id), "Training/test event leakage")
        require(not set(original.iloc[idx].cow_id) & set(original.iloc[train_idx].cow_id), "Legacy test animal seen in training")
        support = load_json(directory / "attempt_001/train_support.json")
        support_row = {"unit_id": unit_id, "fold_id": fold, "percent": budget, "method": method, "replicate": rep,
                       "selected_events": support["budget"]["selected_event_count"], "selected_windows": support["budget"]["selected_window_count"],
                       "available_events": support["budget"]["available_event_count"]}
        support_row.update({"train_" + name: support["window_support"][name] for name in STATES})
        training_support.append(support_row)
        predictions[(fold, rep, budget, method)] = (idx, prob.argmax(axis=1))

    group_counts = []
    for version, column, groups in (("new", "analysis_group", NEW), ("old", "cow_alias", OLD)):
        for group in groups:
            frame = mapping.loc[mapping[column].eq(group)]
            row = {"version": version, "group": group, "windows": len(frame), "events": frame.event_id.nunique(),
                   "sessions": len(frame[["device_mac", "session_id"]].drop_duplicates()),
                   "supported_classes": frame.target4.nunique(), "legacy_cows": ";".join(sorted(frame.legacy_cow_id.unique()))}
            for i, state in enumerate(STATES):
                row[state + "_windows"] = int(frame.target4.eq(i).sum())
                row[state + "_events"] = frame.loc[frame.target4.eq(i), "event_id"].nunique()
            group_counts.append(row)
    counts = pd.DataFrame(group_counts)
    metric_rows, class_rows, confusion_rows, invariance = [], [], [], []
    for rep, budget, method in itertools.product(range(3), BUDGETS, METHODS):
        pooled_pred = np.full(len(mapping), -1, dtype=np.int64)
        covered = np.zeros(len(mapping), dtype=np.int64)
        for fold in ("f01", "f02"):
            idx, pred = predictions[(fold, rep, budget, method)]
            mapped_idx = source_to_mapping[idx]
            require(np.all(mapped_idx >= 0), "Held-out prediction outside new dataset")
            pooled_pred[mapped_idx] = pred
            covered[mapped_idx] += 1
        require(np.all(covered == 1), "Pooled OOF predictions incomplete or duplicated")
        old_members = mapping.cow_alias.isin(["COW_OTHER", "COW_OTHER", "COW_OTHER"]).to_numpy()
        new_members = mapping.analysis_group.isin(FOUR).to_numpy()
        require(np.array_equal(old_members, new_members), "Old/new test union differs")
        for weighting in ("window", "event_balanced_window"):
            for version, column, groups in (("new", "analysis_group", NEW), ("old", "cow_alias", OLD),
                                             ("cohort", None, ("all", "all_retained_OOF"))):
                for group in groups:
                    mask = np.ones(len(mapping), dtype=bool) if group == "all_retained_OOF" else old_members if group == "all" else mapping[column].eq(group).to_numpy()
                    idx = np.flatnonzero(mask)
                    y, pred = y_all[idx], pooled_pred[idx]
                    event_counts = mapping.iloc[idx].event_id.value_counts()
                    weights = None if weighting == "window" else 1.0 / mapping.iloc[idx].event_id.map(event_counts).to_numpy()
                    values, precision, recall, f1, support, cm = score(y, pred, weights)
                    base = {"version": version, "group": group, "percent": budget, "method": method, "replicate": rep,
                            "step": 1000, "weighting": weighting, "windows": len(idx), "events": mapping.iloc[idx].event_id.nunique(),
                            "folds": "f01+f02_OOF" if group == "all_retained_OOF" else "f02" if group in FOUR or group in ("COW_OTHER", "COW_OTHER", "COW_OTHER", "all") else "f01"}
                    metric_rows.append({**base, **values})
                    for i, state in enumerate(STATES):
                        class_rows.append({**base, "state": state, "state_cn": CN[state], "precision": float(precision[i]),
                                           "recall": float(recall[i]), "f1": float(f1[i]), "weighted_support": float(support[i]),
                                           "window_support": int(np.sum(y == i)), "predicted_windows": int(np.sum(pred == i))})
                    for a, b in itertools.product(range(4), repeat=2):
                        confusion_rows.append({**base, "true_state": STATES[a], "predicted_state": STATES[b], "count_or_weight": float(cm[a, b])})
            old_metrics = score(y_all[old_members], pooled_pred[old_members], None if weighting == "window" else 1 / mapping.loc[old_members, "event_id"].map(mapping.loc[old_members, "event_id"].value_counts()).to_numpy())[0]
            new_metrics = score(y_all[new_members], pooled_pred[new_members], None if weighting == "window" else 1 / mapping.loc[new_members, "event_id"].map(mapping.loc[new_members, "event_id"].value_counts()).to_numpy())[0]
            differences = {name + "_delta": new_metrics[name] - old_metrics[name] for name in METRICS}
            require(all(value == 0 for value in differences.values()), "Relabeling changed pooled score")
            invariance.append({"percent": budget, "method": method, "replicate": rep, "weighting": weighting,
                               "old_windows": int(old_members.sum()), "new_windows": int(new_members.sum()),
                               "identical_window_membership": True, "identical_predictions": True, **differences})

    metrics, classes, confusions = pd.DataFrame(metric_rows), pd.DataFrame(class_rows), pd.DataFrame(confusion_rows)
    keys = ["version", "group", "percent", "method", "step", "weighting", "windows", "events", "folds", "supported_class_count", "missing_truth_classes"]
    summary = metrics.groupby(keys, dropna=False).agg({name: ["mean", "std", "min", "max"] for name in METRICS})
    summary.columns = ["_".join(name) for name in summary.columns]
    summary = summary.reset_index()
    n = metrics.groupby(keys, dropna=False).replicate.nunique().reset_index(name="replicate_count")
    summary = summary.merge(n, on=keys, validate="one_to_one")
    require(summary.replicate_count.eq(3).all(), "Unpaired replicate summary")
    class_keys = ["version", "group", "percent", "method", "step", "weighting", "state", "state_cn", "window_support"]
    class_summary = classes.groupby(class_keys).agg({name: ["mean", "std"] for name in ("precision", "recall", "f1")})
    class_summary.columns = ["_".join(name) for name in class_summary.columns]
    class_summary = class_summary.reset_index()

    paired_keys = ["version", "group", "percent", "replicate", "step", "weighting"]
    scratch = metrics.loc[metrics.method.eq("scratch"), paired_keys + list(METRICS)].rename(columns={name: name + "_scratch" for name in METRICS})
    delta = metrics.loc[metrics.method.ne("scratch")].merge(scratch, on=paired_keys, validate="many_to_one")
    for metric in METRICS:
        delta[metric + "_delta_vs_scratch"] = delta[metric] - delta[metric + "_scratch"]
    delta_cols = [name + "_delta_vs_scratch" for name in METRICS]
    delta_summary = delta.groupby(["version", "group", "percent", "method", "step", "weighting"])[delta_cols].agg(["mean", "std"])
    delta_summary.columns = ["_".join(name) for name in delta_summary.columns]
    delta_summary = delta_summary.reset_index()
    group_evidence = []
    for method in METHODS:
        eligible = summary.loc[(summary.version == "new") & summary.group.isin(FOUR) & (summary.weighting == "window") & (summary.method == method)]
        for group in FOUR:
            frame = eligible.loc[eligible.group.eq(group)]
            single = metrics.loc[(metrics.version == "new") & metrics.group.eq(group) & metrics.weighting.eq("window") & metrics.method.eq(method)]
            d = delta.loc[delta.version.eq("new") & delta.group.eq(group) & delta.weighting.eq("window") & delta.method.eq(method)]
            wins = 0
            for budget in BUDGETS:
                at_budget = eligible.loc[eligible.percent.eq(budget)]
                wins += int(at_budget.loc[at_budget.macro_f1_four_mean.idxmax(), "group"] == group)
            group_evidence.append({"group": group, "method": method,
                                   "mean_of_all_5_budget_f1_means": float(frame.macro_f1_four_mean.mean()),
                                   "min_budget_f1_mean": float(frame.macro_f1_four_mean.min()), "max_budget_f1_mean": float(frame.macro_f1_four_mean.max()),
                                   "mean_of_all_5_budget_accuracy_means": float(frame.accuracy_mean.mean()),
                                   "mean_within_budget_f1_std": float(frame.macro_f1_four_std.mean()),
                                   "worst_single_replicate_f1": float(single.macro_f1_four.min()),
                                   "best_group_budget_count": wins,
                                   "positive_paired_replicates_vs_scratch": int(d.macro_f1_four_delta_vs_scratch.gt(0).sum()) if method != "scratch" else 0,
                                   "positive_mean_budgets_vs_scratch": int(d.groupby("percent").macro_f1_four_delta_vs_scratch.mean().gt(0).sum()) if method != "scratch" else 0,
                                   "note": "descriptive across heterogeneous budgets, not an unbiased selected-model estimate"})
    evidence = pd.DataFrame(group_evidence)
    outputs = {"group_counts.csv": counts, "metrics_by_replicate.csv": metrics, "metrics_mean_std.csv": summary,
               "class_metrics_by_replicate.csv": classes, "class_metrics_mean_std.csv": class_summary,
               "confusion_by_replicate.csv": confusions, "paired_deltas_vs_scratch.csv": delta,
               "paired_deltas_mean_std.csv": delta_summary, "group_stability_summary.csv": evidence,
               "new_old_pool_invariance.csv": pd.DataFrame(invariance), "verified_prediction_manifest.csv": pd.DataFrame(manifests),
               "training_support.csv": pd.DataFrame(training_support)}
    for name, frame in outputs.items():
        frame.to_csv(OUT / name, index=False, encoding="utf-8-sig")

    def row(group, method, budget, version="new"):
        found = summary.loc[summary.group.eq(group) & summary.method.eq(method) & summary.percent.eq(budget) & summary.version.eq(version) & summary.weighting.eq("window")]
        require(len(found) == 1, "Ambiguous report row")
        return found.iloc[0]

    report = ["# 现成四分类模型：新分组直接评价", "",
              "这次只读取已有1000步四分类预测，未训练、未调用模型推理、未使用GPU。共有90份预测：2个原始留出折 × 5个标签预算 × 3种模型起点 × 3次重复。", "",
              "**评价口径：**四分类为站立、卧躺、行走、采食；最高概率为最终类别。下表均为3次重复的均值，标准差见详细表。准确率统计答对的窗口比例；宏平均F1让每一类同等重要。它们都不是此前的行走AP。", "",
              "**测试独立性：**A、D、E使用原始f01折的模型；B1、C1、F1、G1使用原始f02折的模型。每组对应模型的训练索引与该组测试窗口、标注事件、原始留出牛号均无重叠。这些是原有各折模型，不能表述成同一个模型从未见过全部7组。全队列汇总是两个留出折的OOF预测汇总。", "",
              "**当前建议：维持原版作为主版本，新分组保留为分析附件。**C1在五个标签预算下都是新四组中四类宏F1最高的，可作为相对较好案例；F1中等，B1和G1整体偏弱。四组在10%标签时ACT均优于Scratch，但50%标签时均低于Scratch，因此不能把新版整体写成四分类高性能或迁移始终有效。", "",
              "**分组边界：**B1/C1/F1/G1是当前ALL_POOL池按完整会话形成的4个分析分组，不新增独立动物；拆分属于看到历史结果后的探索分析。保留或回滚只改变展示分组，不能把筛选后的组均值当成完整队列成绩。", "",
              "## 数据规模与类别覆盖", "",
              markdown_table(["组", "区间", "窗口", "站立", "卧躺", "行走", "采食", "会话"],
                             [[g, int((r := counts.loc[counts.version.eq('new') & counts.group.eq(g)].iloc[0]).events), int(r.windows)] + [int(r[s + '_windows']) for s in STATES] + [int(r.sessions)] for g in NEW]), "",
              "E没有采食真值。它的固定四类宏F1把缺失的采食类计0，另提供有真值三类宏F1；不能把三类数值当成完整四分类结果。", "",
              "## 新四组：ACT在全部标签预算下的四分类宏F1（%）", "",
              markdown_table(["组"] + [f"{b}%标签" for b in BUDGETS], [[g] + [pm(row(g, 'act', b), 'macro_f1_four') for b in BUDGETS] for g in FOUR]), "",
              "## 新四组：ACT准确率（%）", "",
              markdown_table(["组"] + [f"{b}%标签" for b in BUDGETS], [[g] + [pm(row(g, 'act', b), 'accuracy') for b in BUDGETS] for g in FOUR]), "",
              "## 迁移是否比从零训练更好：ACT减Scratch的四分类宏F1（百分点）", "",
              markdown_table(["组"] + [f"{b}%标签" for b in BUDGETS], [[g] + [f"{100 * (row(g, 'act', b).macro_f1_four_mean - row(g, 'scratch', b).macro_f1_four_mean):+.2f}" for b in BUDGETS] for g in FOUR]), "",
              "这里全部条件都展示，正负差值均保留。标签预算是原始实验训练事件比例，并非这次对测试组使用了相应比例的标签。", "",
              "## 全部保留组：10%标签条件（各自原始留出折）", "",
              markdown_table(["组", "Scratch四类F1", "ACT四类F1", "UFRO四类F1", "ACT准确率", "ACT支持类F1"], [[g] + [pm(row(g, m, 10), 'macro_f1_four') for m in METHODS] + [pm(row(g, 'act', 10), 'accuracy'), pm(row(g, 'act', 10), 'macro_f1_supported')] for g in NEW]), "",
              "## 全部保留组：100%标签条件（各自原始留出折）", "",
              markdown_table(["组", "Scratch四类F1", "ACT四类F1", "UFRO四类F1", "ACT准确率", "ACT支持类F1"], [[g] + [pm(row(g, m, 100), 'macro_f1_four') for m in METHODS] + [pm(row(g, 'act', 100), 'accuracy'), pm(row(g, 'act', 100), 'macro_f1_supported')] for g in NEW]), "",
              "## 原B/C/F在同一批模型下的对照：ACT四分类宏F1（%）", "",
              markdown_table(["原组"] + [f"{b}%标签" for b in BUDGETS], [[g] + [pm(row(g, 'act', b, 'old'), 'macro_f1_four') for b in BUDGETS] for g in ("COW_OTHER", "COW_OTHER", "COW_OTHER")]), "",
              "原F没有卧躺真值；它的固定四类F1与覆盖四类的新组不能只看一个数字直接判断。这里用的仍是原f02折模型，不能把这些数值与后来六牛独立重训练实验混为同一对照。", "",
              "## ACT各类识别表现：10%和100%标签", ""]
    for budget in (10, 100):
        rows = []
        for group in NEW:
            vals = []
            for state in STATES:
                r = class_summary.loc[class_summary.version.eq("new") & class_summary.group.eq(group) & class_summary.percent.eq(budget) & class_summary.method.eq("act") & class_summary.weighting.eq("window") & class_summary.state.eq(state)].iloc[0]
                vals.append("无真值" if r.window_support == 0 else f"{fmt(r.f1_mean)} ± {fmt(r.f1_std)}")
            rows.append([group] + vals)
        report += [f"### {budget}%标签：各类F1（%）", "", markdown_table(["组", "站立", "卧躺", "行走", "采食"], rows), ""]
    report += ["## 新旧分组是否真的提高了总成绩", "",
               "逐窗口核查：旧B+C+F与新B1+C1+F1+G1均为21,687个窗口、544个区间。45个方法/预算/重复条件及两种权重下，窗口、真值、预测完全相同；准确率、宏F1、MCC差值全部为0。分组只揭示不同片段的差异，不会提高合并成绩。", "",
               "## 全条件稳定性（仅描述，不选最好的预算）", "",
               markdown_table(["组", "ACT五预算F1均值", "预算均值范围", "五预算中最高组次数", "ACT胜Scratch预算数", "ACT胜Scratch重复数"],
                              [[g, fmt((r := evidence.loc[evidence.group.eq(g) & evidence.method.eq('act')].iloc[0]).mean_of_all_5_budget_f1_means), f"{fmt(r.min_budget_f1_mean)}–{fmt(r.max_budget_f1_mean)}", f"{int(r.best_group_budget_count)}/5", f"{int(r.positive_mean_budgets_vs_scratch)}/5", f"{int(r.positive_paired_replicates_vs_scratch)}/15"] for g in FOUR]), "",
               "这张表跨越不同标签预算，只辅助判断稳定性，不是一个新模型的性能估计。重复间标准差只体现3个原实验重复的变动，不是牛群总体的置信区间。", "",
               "还需看到具体失败：10%标签下，G1第3个重复的采食召回率为0；C1和F1同一重复的采食召回率也分别只有2.04%和4.94%。因此C1虽是相对最好组，低标签条件下仍会出现类别识别不稳定，不能仅凭均值称为四类可靠识别。", "",
               "## 文件与复现", "",
               "- `metrics_mean_std.csv`：全部新旧组、五预算、三方法、两种权重的均值与样本标准差（ddof=1）。", 
               "- `class_metrics_mean_std.csv`：每类精确率、召回率、F1与支持量；`confusion_by_replicate.csv`为逐重复混淆矩阵。",
               "- `paired_deltas_mean_std.csv`：同折、同预算、同重复下迁移相对Scratch的配对差值。",
               "- `new_old_pool_invariance.csv`：全部90个加权条件下的新旧合并分数一致性。",
               "- `AUDIT.json`与`verified_prediction_manifest.csv`：本次输入与保留预测的哈希、映射和无泄漏核验。",
               "- 执行 `python score_existing_models.py` 即可从原预测重建此目录报告与表格；不导入torch。", "",
               "**历史完整性限制：**原始3000步端点曾按要求清理，历史整套完成审计已经失效。本报告仅验证当前仍保留的1000步预测及所用元数据/索引，不声称原训练包的全部历史文件仍完整。", ""]
    report_path = OUT / "new_grouping_four_class_results.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    (OUT / "results_notes.md").write_text("完整报告见 [新分组四分类效果](new_grouping_four_class_results.md)。\n", encoding="utf-8")
    for name, entry in input_hashes.items():
        require(sha(inputs[name]) == entry["sha256"], f"Read-only input changed during scoring: {name}")
    for entry in manifests:
        require(sha(entry["path"]) == entry["sha256"], f"Source prediction changed during scoring: {entry['path']}")
    audit = {"status": "PASS", "created_utc": datetime.now(timezone.utc).isoformat(),
             "scope": "existing 1000-step predictions only; no training, no inference, no GPU", "units": len(units),
             "held_out_prediction_windows_per_condition": 28911, "new_groups": list(NEW), "old_groups": list(OLD),
             "unit_grid": {"folds": ["f01", "f02"], "budgets": list(BUDGETS), "methods": list(METHODS), "replicates": [0, 1, 2]},
             "verified_artifact_count": len(manifests), "inputs": input_hashes,
             "all_prediction_hashes_match_preserved_receipts": True, "all_probabilities_finite_and_normalized": True,
             "source_inputs_and_used_prediction_artifacts_before_after_sha256_identical": True,
             "window_and_label_mapping_unchanged": True, "all_units_no_test_train_window_event_legacy_identity_overlap": True,
             "all_sessions_intact": True, "new_old_union_all_scores_exactly_equal": True,
             "new_old_invariance_conditions": len(invariance), "four_class_macro_f1": "all four fixed labels; undefined class F1=0",
             "supported_macro_f1": "only classes with truth support; cowD has three",
             "replicate_std": "sample standard deviation, ddof=1; three paired training repeats, not biological replication",
             "whole_cohort": "pooled predictions from original f01/f02 held-out models, not a single checkpoint",
             "historical_full_archive_audit": "NOT CLAIMED; 3000-step historical cleanup invalidated original all-artifact receipts",
             "outputs": {name: {"rows": len(frame), "sha256": sha(OUT / name)} for name, frame in outputs.items()},
             "report_path": str(report_path), "report_sha256": sha(report_path), "script_sha256": sha(Path(__file__))}
    dump_json("AUDIT.json", audit)
    print(json.dumps({"status": "PASS", "units": len(units), "metrics": len(metrics), "summary": len(summary), "output": str(OUT)}, ensure_ascii=False))
    print(evidence.loc[evidence.method.eq("act")].to_string(index=False))


if __name__ == "__main__":
    main()
