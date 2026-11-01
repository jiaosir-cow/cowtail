"""Re-score archived, unadapted source models on every retained test group.

No training, threshold selection, label edits, or source-data mutation occurs.
The seven names are analysis partitions, not seven independently verified cows.
Run with the local research Python; needs numpy, pandas, scikit-learn.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

BASE = Path(r"PATH_X")
DEFAULT_ARCHIVE = BASE / "实验记录与结果/零标签设备开发验证_20261025/download/bundle"
DEFAULT_TRUTH = BASE / "实验记录与结果/零标签设备开发验证_20261025/analysis/evaluation_truth_LOCAL_ONLY.csv"
DEFAULT_DATA = BASE / "datasets/DATASET_X"
EXPECTED_PIN = "1fcc02e9910095097f3a1b1b14a4bb5a11a3c558e08efdbeef90f56d4a95ea4c"
NEW_GROUPS = ["cowA", "cowC", "cowD", "COW_OTHER", "cowB", "COW_OTHER", "COW_OTHER"]
OLD_POOL = ["COW_OTHER", "COW_OTHER", "COW_OTHER"]
SOURCES = ["jbb", "act", "ufro", "multi"]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def checked_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Manifest path escapes its root: {relative}")
    return path


def check_hashes(root: Path, hashes: dict, kind: str, receipts: list):
    for relative, expected in hashes.items():
        path = checked_path(root, relative)
        actual = sha(path)
        if actual != expected:
            raise AssertionError(f"{kind} hash mismatch: {relative}")
        receipts.append({"kind": kind, "path": str(path), "sha256": actual})


def binary_metrics(y: np.ndarray, score: np.ndarray) -> dict:
    predicted = score >= 0.0
    positive = y == 1
    tp = int(np.count_nonzero(predicted & positive))
    fp = int(np.count_nonzero(predicted & ~positive))
    tn = int(np.count_nonzero(~predicted & ~positive))
    fn = int(np.count_nonzero(~predicted & positive))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    return {
        "walking_ap": float(average_precision_score(y, score)) if positive.any() else np.nan,
        "roc_auc": float(roc_auc_score(y, score)) if np.unique(y).size == 2 else np.nan,
        "precision": precision, "recall": recall,
        "walking_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "binary_accuracy": (tp + tn) / len(y),
        "balanced_accuracy": (recall + specificity) / 2,
        "specificity": specificity, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--truth", type=Path, default=DEFAULT_TRUTH)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    receipts: list[dict] = []
    frozen_path = args.archive / "configs/FROZEN_PROTOCOL.json"
    pin = sha(frozen_path)
    assert pin == EXPECTED_PIN, (pin, EXPECTED_PIN)
    frozen = read_json(frozen_path)
    check_hashes(args.archive, frozen["artifact_sha256"], "archive_frozen_input", receipts)
    dataset_manifest = read_json(args.dataset / "MANIFEST.json")
    assert dataset_manifest["status"] == "PASS"
    check_hashes(args.dataset, dataset_manifest["artifact_sha256"], "new_dataset", receipts)

    windows = pd.read_csv(args.dataset / "windows.csv")
    truth = pd.read_csv(args.truth)
    assert len(windows) == len(truth) == 28911
    assert windows.row_index.tolist() == list(range(28911))
    assert windows.continuous_row_index.is_unique and truth.continuous_row_index.is_unique
    assert windows.window_id.is_unique and truth.window_id.is_unique
    assert truth.pool_row_index.is_unique
    assert set(windows.analysis_group) == set(NEW_GROUPS)
    joined = windows.merge(
        truth[["pool_row_index", "continuous_row_index", "window_id", "target4", "cow_alias"]],
        on="continuous_row_index", how="outer", validate="one_to_one", suffixes=("", "_truth"),
        indicator=True,
    ).sort_values("row_index").reset_index(drop=True)
    assert joined._merge.eq("both").all()
    assert joined.window_id.eq(joined.window_id_truth).all()
    assert joined.target4.eq(joined.target4_truth).all()
    assert joined.cow_alias.eq(joined.cow_alias_truth).all()
    assert joined.loc[joined.analysis_group.isin(NEW_GROUPS[3:]), "cow_alias"].isin(OLD_POOL).all()
    assert joined.loc[~joined.analysis_group.isin(NEW_GROUPS[3:]), "analysis_group"].eq(
        joined.loc[~joined.analysis_group.isin(NEW_GROUPS[3:]), "cow_alias"]
    ).all()
    y_all = joined.target4.eq(2).to_numpy(dtype=np.int64)
    masks = {group: joined.analysis_group.eq(group).to_numpy() for group in NEW_GROUPS}
    masks["All retained"] = np.ones(len(joined), dtype=bool)
    for group in OLD_POOL:
        masks["Previous " + group] = joined.cow_alias.eq(group).to_numpy()
    masks["Pooled ALL_POOL (old and new identical)"] = joined.cow_alias.isin(OLD_POOL).to_numpy()
    new_bcf_mask = joined.analysis_group.isin(NEW_GROUPS[3:]).to_numpy()
    assert np.array_equal(masks["Pooled ALL_POOL (old and new identical)"], new_bcf_mask)
    assert sum(masks[group].astype(np.int64) for group in NEW_GROUPS).tolist() == [1] * len(joined)
    supports = []
    for group, mask in masks.items():
        subset = joined.loc[mask]
        supports.append({
            "group": group, "windows": len(subset),
            "walking_windows": int(subset.target4.eq(2).sum()),
            "events": subset.event_id.nunique(),
            "walking_events": subset.loc[subset.target4.eq(2), "event_id"].nunique(),
            "sessions": len(subset[["session_id", "device_mac"]].drop_duplicates()),
            **{f"class_{label}_windows": int(subset.target4.eq(k).sum())
               for k, label in enumerate(["standing", "lying", "walking", "feeding"])},
        })
    support_by_group = {row["group"]: row for row in supports}
    results, invariants = [], []
    rms_reference = None
    for source in SOURCES:
        for seed in range(3):
            by_pool = {}
            for fold in ["l01", "l02"]:
                uid = f"{fold}_s{seed}_{source}"
                unit_root = args.archive / "results" / uid
                complete_path = unit_root / "COMPLETE.json"
                complete = read_json(complete_path)
                assert complete["status"] == "COMPLETE" and complete["mode"] == "formal"
                assert complete["unit_id"] == uid and complete["frozen_sha256"] == pin
                assert complete["unit"]["source"] == source
                assert complete["unit"]["replicate"] == seed
                assert complete["frozen_state_unchanged"] is True
                checkpoint = complete["source_checkpoint"]
                assert sha(args.archive / checkpoint["path"]) == checkpoint["sha256"]
                check_hashes(unit_root, complete["output_sha256"], "archive_unit_output", receipts)
                receipts.append({"kind": "archive_unit_complete", "path": str(complete_path), "sha256": sha(complete_path)})
                path = unit_root / "predictions/frozen_step0000/clean.npz"
                with np.load(path, allow_pickle=False) as pred:
                    rows = pred["row_indices"]
                    margins = pred["margin"]
                    rms = pred["rms_score"]
                    assert len(np.unique(rows)) == len(rows)
                    assert np.isfinite(margins).all() and np.isfinite(rms).all()
                    assert np.array_equal(rows, np.load(args.archive / f"inputs/folds/{fold}/test_indices.npy", allow_pickle=False))
                    for pool_index, margin, rms_score in zip(rows, margins, rms):
                        assert int(pool_index) not in by_pool
                        by_pool[int(pool_index)] = (float(margin), float(rms_score))
            assert len(by_pool) == 40920
            values = np.array([by_pool[int(index)] for index in joined.pool_row_index])
            margin_all, rms_all = values[:, 0], values[:, 1]
            if rms_reference is None:
                rms_reference = rms_all
            else:
                assert np.array_equal(rms_reference, rms_all)
            for group, mask in masks.items():
                row = {
                    "group": group, "source": source.upper(), "seed": seed,
                    "stage": "source_only_frozen_step0000_clean", "threshold_margin": 0.0,
                    **{k: v for k, v in support_by_group[group].items() if k != "group"},
                    **binary_metrics(y_all[mask], margin_all[mask]),
                    "rms_walking_ap": float(average_precision_score(y_all[mask], rms_all[mask])),
                }
                results.append(row)
            old_metric = binary_metrics(y_all[masks["Pooled ALL_POOL (old and new identical)"]], margin_all[masks["Pooled ALL_POOL (old and new identical)"]])
            new_metric = binary_metrics(y_all[new_bcf_mask], margin_all[new_bcf_mask])
            assert old_metric == new_metric
            invariants.append({"source": source.upper(), "seed": seed, "pooled_old_new_metrics_identical": True, **old_metric})
    detail = pd.DataFrame(results)
    detail.to_csv(args.output / "source_only_per_seed.csv", index=False, encoding="utf-8-sig")
    metric_names = ["walking_ap", "roc_auc", "precision", "recall", "walking_f1", "binary_accuracy", "balanced_accuracy", "specificity", "rms_walking_ap"]
    summary = detail.groupby(["group", "source"], sort=False)[metric_names].agg(["mean", "std", "min", "max"])
    summary.columns = ["_".join(col) for col in summary.columns]
    summary = summary.reset_index().merge(pd.DataFrame(supports), on="group", validate="many_to_one")
    summary.to_csv(args.output / "source_only_summary.csv", index=False, encoding="utf-8-sig")
    # Equal-group means change when a group is subdivided. Retain this diagnostic
    # rather than confusing those averages with unchanged pooled predictions.
    aggregation_rows = []
    previous_six = ["cowA", "cowC", "cowD", *["Previous " + group for group in OLD_POOL]]
    for source in [name.upper() for name in SOURCES]:
        for seed in range(3):
            subset = detail.loc[detail.source.eq(source) & detail.seed.eq(seed)]
            for name, groups in [("previous_six_equal_group_diagnostic", previous_six),
                                 ("new_seven_equal_group_diagnostic", NEW_GROUPS)]:
                rows = subset.loc[subset.group.isin(groups)]
                assert len(rows) == len(groups)
                aggregation_rows.append({"source": source, "seed": seed, "aggregation": name,
                                         "group_count": len(groups),
                                         **{metric: float(rows[metric].mean()) for metric in metric_names},
                                         "comparable_before_after_performance_gain": False})
    pd.DataFrame(aggregation_rows).to_csv(args.output / "equal_group_aggregation_diagnostic.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(supports).to_csv(args.output / "group_support.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(invariants).to_csv(args.output / "partition_invariance.csv", index=False, encoding="utf-8-sig")
    baseline = []
    for group, mask in masks.items():
        baseline.append({"group": group, "windows": int(mask.sum()), "walking_windows": int(y_all[mask].sum()),
                         "rms_walking_ap": float(average_precision_score(y_all[mask], rms_reference[mask])),
                         "rms_roc_auc": float(roc_auc_score(y_all[mask], rms_reference[mask]))})
    pd.DataFrame(baseline).to_csv(args.output / "rms_baseline.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(receipts).to_csv(args.output / "verified_source_hashes.csv", index=False, encoding="utf-8-sig")
    outputs = {path.name: sha(path) for path in args.output.glob("*.csv")}
    act = summary.loc[summary.source.eq("ACT")].set_index("group")
    lines = [
        "# 全部保留组：public_model_walking_direct的复核", "",
        "所有 28,911 个已标注窗口都只用于本地测试。模型为已有公开数据训练得到的冻结模型；本次没有新训练、尾圈标签学习或阈值调参。",
        "这里是行走/非行走二分类，不能当作站立、卧躺、行走、采食四分类成绩。", "",
        "B1、C1、F1、G1 是原 B/C/F 数据按完整会话重分的分析组。其总数据和预测未改变，所以合并成绩与重分前完全相同。分组不能创造总体准确率提升。",
        "旧实验已查看过这批测试标签；本次是对既有测试结果的回顾性重分分析，不是全新的独立测试集。", "",
        "AP 衡量行走片段的排序质量，不是准确率。精确率、召回率和 F1 使用原模型固定 margin ≥ 0 规则。下表为三个公开模型随机种子的均值。", "",
        "此前 ACT 的 93.03% 是原六组各自 AP 的等权平均；下表全体 89.79% 是把所有窗口合在一起计算 AP。两者统计口径不同，不是模型退化。重新分组会改变等权平均的权重，所以不能据此声称性能改善。", "",
        "| 分组 | 窗口数 | 行走窗口 | ACT 行走AP | ACT精确率 | ACT召回率 | ACT行走F1 | 简单运动强度AP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in [*NEW_GROUPS, "All retained", "Pooled ALL_POOL (old and new identical)"]:
        row = act.loc[group]
        lines.append(f"| {group} | {int(row.windows)} | {int(row.walking_windows)} | {row.walking_ap_mean:.2%} | {row.precision_mean:.2%} | {row.recall_mean:.2%} | {row.walking_f1_mean:.2%} | {row.rms_walking_ap_mean:.2%} |")
    lines += ["", "其余 JBB、UFRO、MULTI 来源的全部种子、旧 B/C/F 对照及四类真值数量均保存在 CSV。未按分组效果删除样本、来源或负向结果。", "", "四个新组都有四类真值；这只说明可评估四分类，不代表本项二分类模型已经具备四分类能力。"]
    (args.output / "results_notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    outputs["results_notes.md"] = sha(args.output / "results_notes.md")
    verification = {
        "status": "PASS", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha(Path(__file__)), "source_archive_frozen_sha256": pin,
        "truth_path": str(args.truth), "truth_sha256": sha(args.truth),
        "new_dataset_manifest_path": str(args.dataset / "MANIFEST.json"),
        "new_dataset_manifest_sha256": sha(args.dataset / "MANIFEST.json"),
        "all_retained_windows": 28911, "mapping_one_to_one": True,
        "groups": NEW_GROUPS, "model_sources": SOURCES, "seeds": [0, 1, 2],
        "completed_archived_units_verified": 24, "source_hash_records": len(receipts),
        "output_rows": len(detail), "summary_rows": len(summary),
        "partition_changes_pooled_predictions_or_metrics": False,
        "uses_new_training": False, "uses_target_labels_for_fit_or_threshold": False,
        "evaluation_kind": "retrospective regrouping of previously examined test records",
        "task": "binary walking versus nonwalking, not four-state",
        "output_sha256": outputs,
    }
    (args.output / "VERIFICATION.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "detail_rows": len(detail), "verified_hash_records": len(receipts), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
