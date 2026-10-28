"""Audit a complete six-cow download, then score both frozen endpoints locally.

No training, network access, checkpoint execution, or modification of the bundle.
Requires numpy, pandas, and scikit-learn. Output must be a new local directory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import sys
import warnings
import zipfile

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_recall_fscore_support)

STATES = ["standing", "lying", "walking", "feeding"]
HEADS = ["posture", "walking", "feeding"]
STEPS = [1000, 3000]
WEIGHTINGS = ["window", "interval_balanced_window"]
IDS = {"cowA": "COWID_04", "COW_OTHER": "COWID_05", "COW_OTHER": "COWID_01",
       "cowC": "COWID_06", "cowD": "COWID_07", "COW_OTHER": "COWID_08"}
EXPECTED_PROTOCOL_SHA256 = "d59ddd996bc99df9a018cc4bdbac519f3959d4cca2cfbe2213743f72113ebc20"
EXPECTED_UNITS_SHA256 = "0f69317d8f17168475f692b05993cd68c4bd57ccdbecab027be1f1a07615a824"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def safe_path(root, relative):
    require(isinstance(relative, str) and relative and not Path(relative).is_absolute(),
            f"Expected relative artifact path: {relative}")
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()) and path != root.resolve(),
            f"Artifact outside bundle: {relative}")
    return path


def expected_bindings(config):
    names = ["configs/units.json", "configs/folds.json", "source/budget_runner.py",
             "source/hierarchy_model.py", "source/model_paper_kernel8.py",
             "inputs/acceleration.npy", "inputs/windows.csv"]
    names += [config["fold_dir"] + "/" + name for name in ["train_labels.npy", "test_indices.npy"]]
    names += [config["budget_dir"] + "/" + name for name in
              ["train_indices.npy", "batches.npy", "selected_event_ids.json", "summary.json"]]
    if config["method"] != "scratch":
        names.append(config["checkpoint"])
    return names


def verify_gate(bundle):
    """All outputs and frozen files pass before any labels are loaded as arrays."""
    protocol_path = bundle / "configs/FROZEN_PROTOCOL.json"
    protocol = read_json(protocol_path)
    protocol_sha = sha(protocol_path)
    require(protocol_sha == EXPECTED_PROTOCOL_SHA256, "Pinned pre-run protocol SHA256 differs")
    require(sha(bundle / "configs/units.json") == EXPECTED_UNITS_SHA256, "Pinned pre-run unit-grid SHA256 differs")
    require(protocol.get("units") == 36 and protocol.get("folds") == 6
            and protocol.get("fixed_endpoints") == STEPS, "Not the frozen 36-unit six-cow protocol")
    declared = dict(protocol["artifact_sha256"])
    for subset in ["source_sha256", "input_sha256"]:
        require(all(declared.get(k) == v for k, v in protocol.get(subset, {}).items()),
                f"Inconsistent frozen {subset}")
    require(sha(bundle / "PROTOCOL.md") == protocol["protocol_text_sha256"], "Protocol text differs")
    for name, digest in declared.items():
        require(not name.startswith("analysis/") and "test_truth" not in name, "Full truth in server manifest")
        require(sha(safe_path(bundle, name)) == digest, f"Frozen artifact differs: {name}")
    units = read_json(bundle / "configs/units.json")
    units = units.get("units", units)
    folds = read_json(bundle / "configs/folds.json")
    require(len(units) == 36 and len(folds) == 6, "Unexpected unit/fold count")
    require({f["test_cow"] for f in folds.values()} == set(IDS.values()), "Cow ID membership differs")
    require({f["cow_alias"]: f["test_cow"] for f in folds.values()} == IDS, "Alias/ID mapping differs")
    grid = {(u["test_cow"], int(u["percent"]), u["method"], int(u["replicate"])) for u in units.values()}
    require(grid == set(itertools.product(IDS.values(), [10, 100], ["scratch", "act", "ufro"], [0])),
            "Incomplete/extra formal grid")
    status = read_json(bundle / "supervisor/STATUS.json")
    require(status.get("state") == "COMPLETE" and not status.get("active"), "Final queue is not complete/inactive")
    require(set(status.get("completed", [])) == set(units) | {"synthetic_gpu_smoke"}
            and len(status["completed"]) == 37, "Final queue does not contain all 36 units plus smoke")
    transfer_path = bundle.parent / "FINAL_ARCHIVE_RECEIPT.json"
    transfer = read_json(transfer_path)
    require(transfer.get("state") == "COMPLETE_VERIFIED_REMOTE" and transfer.get("training_units") == 36
            and transfer.get("endpoints_per_unit") == 2 and transfer.get("scoring_performed_remotely") is False,
            "Final archive receipt is incomplete")
    require(transfer.get("protocol_sha256") == EXPECTED_PROTOCOL_SHA256, "Final archive protocol differs")
    archive_path = safe_path(bundle.parent, transfer["archive"])
    require(archive_path.stat().st_size == transfer["bytes"] and sha(archive_path) == transfer["sha256"],
            "Downloaded final archive size/hash differs")
    archived_members = {}
    with zipfile.ZipFile(archive_path) as archive:
        for item in archive.infolist():
            if item.is_dir():
                continue
            require(item.filename not in archived_members, "Duplicate member in final archive")
            local = safe_path(bundle, item.filename)
            digest = hashlib.sha256()
            with archive.open(item) as stream:
                for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                    digest.update(block)
            archived_members[item.filename] = digest.hexdigest()
            require(sha(local) == digest.hexdigest(), f"Extracted final archive member differs: {item.filename}")
    require(len(archived_members) == transfer["files"], "Final archive file count differs")
    receipts, receipt_hashes = {}, {}
    for unit_id, config in sorted(units.items()):
        require(config["unit_id"] == unit_id, "Unit ID differs")
        require(config["test_cow"] == folds[config["fold_id"]]["test_cow"], "Fold/test identity differs")
        run = safe_path(bundle, "results/" + unit_id)
        receipt = read_json(run / "COMPLETE.json")
        require(receipt.get("status") == "complete" and receipt.get("unit_id") == unit_id,
                f"Missing complete formal unit: {unit_id}")
        require(receipt.get("endpoint_steps") == STEPS and receipt.get("parameter_count") == 828,
                f"Endpoint/model contract differs: {unit_id}")
        require(receipt.get("guard", {}).get("membership_verified") is True, "Worker guard not verified")
        binding = {"protocol_sha256": protocol_sha,
                   "artifact_sha256": {name: declared[name] for name in expected_bindings(config)},
                   "unit_config_sha256": canonical_sha(config)}
        require(receipt.get("binding") == binding, f"Frozen worker binding differs: {unit_id}")
        attempt = receipt.get("attempt", "")
        require(attempt.startswith("attempt_") and "/" not in attempt and "\\" not in attempt,
                f"Unsafe attempt name: {unit_id}")
        outputs = receipt.get("artifact_sha256", {})
        require(bool(outputs), "Empty output receipt")
        for name, digest in outputs.items():
            require(name.startswith(attempt + "/"), "Receipt refers to another attempt")
            require(sha(safe_path(run, name)) == digest, f"Output hash differs: {unit_id}/{name}")
        for step in STEPS:
            for name in ["checkpoint.pt", "test_indices.npy", "test_state_probabilities.npy",
                         "test_head_probabilities.npy", "prediction_metadata.json"]:
                require(f"{attempt}/step{step}/{name}" in outputs, f"Endpoint artifact missing: {unit_id}/{step}/{name}")
        for name in ["completion_metadata.json", "input_binding.json", "initialization_audit.json",
                     "train_support.json", "environment.json", "training_history.json", "progress.json"]:
            require(attempt + "/" + name in outputs, f"Completion artifact missing: {unit_id}/{name}")
        complete = read_json(run / attempt / "completion_metadata.json")
        require(complete.get("unit") == config and complete.get("binding") == binding,
                "Completion metadata binding differs")
        require(complete.get("held_out_scoring") is False and complete.get("source_checkpoint_unchanged") is True
                and complete.get("endpoint_steps") == STEPS, "Completion boundary differs")
        require(read_json(run / attempt / "input_binding.json") == binding, "Input binding file differs")
        environment = read_json(run / attempt / "environment.json")
        require(str(environment.get("device", "")).startswith("cuda"), "Formal result is not CUDA")
        receipts[unit_id] = receipt
        receipt_hashes[unit_id] = sha(run / "COMPLETE.json")
    require(transfer.get("unit_receipt_sha256") == receipt_hashes, "Final archive receipt membership/hash differs")
    extras = {p.parent.name for p in (bundle / "results").glob("*/COMPLETE.json")} - set(units)
    require(not extras, f"Unexpected formal completion folders: {sorted(extras)}")
    return {"protocol": protocol, "units": units, "folds": folds, "receipts": receipts,
            "protocol_sha256": protocol_sha, "pinned_protocol_sha256_supplied": True,
            "final_archive_sha256": transfer["sha256"], "final_archive_receipt_sha256": sha(transfer_path),
            "archive_members_verified": len(archived_members), "final_status_sha256": sha(bundle / "supervisor/STATUS.json"),
            "checked_before_truth_reconstruction": True, "unit_receipt_sha256": receipt_hashes,
            "frozen_artifact_count": len(declared), "formal_units": len(receipts), "endpoint_count": 2 * len(receipts)}


def reconstruct_truth(bundle, folds, original_truth=None):
    windows = pd.read_csv(bundle / "inputs/windows.csv", dtype={"cow_id": str, "event_id": str})
    require(len(windows) == 28911 and np.array_equal(windows.row_index, np.arange(28911)), "Window row identity differs")
    require(windows.event_id.nunique() == 781 and set(windows.cow_id) == set(IDS.values()), "Window event/ID membership differs")
    require("target4" not in windows and "original_state" not in windows, "Unmasked truth in server windows")
    require((windows.cow_alias.map(IDS) == windows.cow_id).all(), "Window alias differs")
    require(windows.groupby("event_id").cow_id.nunique().max() == 1, "Original interval crosses identities")
    labels, seen = np.full(len(windows), -1, dtype=np.int64), np.zeros(len(windows), dtype=np.int64)
    held_count = np.zeros(len(windows), dtype=np.int64)
    for fid, fold in sorted(folds.items()):
        y = np.load(bundle / f"inputs/folds/{fid}/train_labels.npy", allow_pickle=False)
        test = np.load(bundle / f"inputs/folds/{fid}/test_indices.npy", allow_pickle=False)
        expected = np.flatnonzero(windows.cow_id.eq(fold["test_cow"]))
        require(y.shape == (len(windows),) and y.dtype == np.int64, "Fold label shape/dtype differs")
        require(np.array_equal(test, expected) and np.issubdtype(test.dtype, np.integer), "Fold test indices differ")
        mask = windows.cow_id.eq(fold["test_cow"]).to_numpy()
        require((y[mask] == -1).all() and np.isin(y[~mask], range(4)).all(), "Held-out mask or train labels differ")
        require((labels[(seen > 0) & ~mask] == y[(seen > 0) & ~mask]).all(), "Frozen fold labels disagree")
        labels[~mask] = y[~mask]
        seen[~mask] += 1
        held_count[mask] += 1
        require(len(test) == fold["test_window_count"], "Fold support count differs")
    require((seen == 5).all() and (held_count == 1).all() and np.isin(labels, range(4)).all(), "Truth reconstruction incomplete")
    truth = windows.copy()
    truth["target4"] = labels
    truth["original_state"] = np.array(STATES)[labels]
    require(truth.groupby("event_id").target4.nunique().max() == 1, "Frozen labels disagree within an original interval")
    if original_truth:
        original = pd.read_csv(original_truth, dtype={"cow_id": str, "event_id": str})
        for key in ["row_index", "cow_id", "event_id", "target4"]:
            require(np.array_equal(truth[key], original[key]), f"Original local truth differs: {key}")
    require(set(truth.loc[truth.cow_alias.eq("cowD"), "target4"]) == {0, 1, 2}, "cowD truth class coverage differs")
    require(set(truth.loc[truth.cow_alias.eq("COW_OTHER"), "target4"]) == {0, 2, 3}, "COW_OTHER truth class coverage differs")
    for cow in ["cowA", "COW_OTHER", "COW_OTHER", "cowC"]:
        require(set(truth.loc[truth.cow_alias.eq(cow), "target4"]) == set(range(4)), "A-D four-class truth coverage differs")
    return truth


def validate_probabilities(states, heads):
    require(states.ndim == 2 and states.shape[1] == 4 and heads.shape == (len(states), 3, 2), "Probability shape differs")
    require(np.isfinite(states).all() and np.isfinite(heads).all(), "Nonfinite probability")
    require((states >= 0).all() and (states <= 1).all() and (heads >= 0).all() and (heads <= 1).all(), "Probability outside [0,1]")
    require(np.allclose(states.sum(1), 1, atol=2e-6, rtol=0)
            and np.allclose(heads.sum(2), 1, atol=2e-6, rtol=0), "Probability normalization differs")
    pl, pw, pf = [heads[:, i, 1].astype(float) for i in range(3)]
    composed = np.stack([(1-pw)*(1-pl)*(1-pf), (1-pw)*pl, pw, (1-pw)*(1-pl)*pf], axis=1)
    require(np.allclose(states, composed, atol=2e-6, rtol=0), "Conditional head composition differs")


def compute_metrics(y, probabilities, event_ids, weighting):
    y, event_ids = np.asarray(y, dtype=np.int64), np.asarray(event_ids, dtype=str)
    require(weighting in WEIGHTINGS and len(y) == len(event_ids) == len(probabilities), "Metric dimensions differ")
    _, inverse, counts = np.unique(event_ids, return_inverse=True, return_counts=True)
    weights = np.ones(len(y)) if weighting == "window" else 1.0 / counts[inverse]
    prediction = probabilities.argmax(1)
    supported = np.unique(y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        precision, recall, f1, support = precision_recall_fscore_support(y, prediction, labels=range(4), sample_weight=weights, zero_division=0)
        mcc = float(matthews_corrcoef(y, prediction, sample_weight=weights))
        values = {"macro_f1_fixed4": float(f1_score(y, prediction, labels=range(4), average="macro", sample_weight=weights, zero_division=0)),
                  "macro_f1_supported": float(f1_score(y, prediction, labels=supported, average="macro", sample_weight=weights, zero_division=0)),
                  "mcc4_diagnostic": mcc, "mcc4_complete_only": mcc if len(supported) == 4 else None,
                  "accuracy": float(accuracy_score(y, prediction, sample_weight=weights)),
                  "walking_ap": float(average_precision_score(y == 2, probabilities[:, 2], sample_weight=weights)) if len(np.unique(y == 2)) == 2 else None,
                  "predictions_absent_class_fraction": float(np.average(~np.isin(prediction, supported), weights=weights)),
                  "observed_classes": len(supported), "complete_four_class_truth": len(supported) == 4,
                  "test_window_count": len(y), "test_event_count": len(counts), "weight_sum": float(weights.sum())}
    classes = [{"class_id": c, "class_name": name, "raw_window_support": int((y == c).sum()),
                "weighted_support": float(support[c]), "truth_present": bool((y == c).any()),
                "precision": float(precision[c]), "recall": float(recall[c]) if (y == c).any() else None,
                "f1_fixed_convention": float(f1[c])} for c, name in enumerate(STATES)]
    return values, classes, confusion_matrix(y, prediction, labels=range(4), sample_weight=weights)


def aggregate(metrics):
    keys = ["unit_id", "step", "weighting"]
    require(len(metrics) == 144 and not metrics.duplicated(keys).any(), "Metric cell membership differs")
    summaries = []
    for scope, subset, f1, mcc in [
        ("all_six_fixed4_diagnostic", metrics, "macro_f1_fixed4", "mcc4_diagnostic"),
        ("complete_four_class_A_to_D", metrics[metrics.complete_four_class_truth], "macro_f1_fixed4", "mcc4_complete_only"),
        ("all_six_supported_classes", metrics, "macro_f1_supported", "mcc4_diagnostic")]:
        cols = ["method", "budget_percent", "step", "endpoint", "weighting"]
        summary = subset.groupby(cols, as_index=False).agg(
            macro_f1=(f1, "mean"), mcc=(mcc, "mean"), accuracy=("accuracy", "mean"),
            walking_ap=("walking_ap", "mean"), test_cow_count=("test_cow", "nunique"))
        summary.insert(0, "scope", scope)
        require((summary.test_cow_count == (4 if scope == "complete_four_class_A_to_D" else 6)).all(), "Scope cow count differs")
        base = summary[summary.method.eq("scratch")].drop(columns="method")
        match = ["scope", "budget_percent", "step", "endpoint", "weighting"]
        base = base[match + ["macro_f1", "mcc"]].rename(columns={"macro_f1": "scratch_f1", "mcc": "scratch_mcc"})
        summary = summary.merge(base, on=match, validate="many_to_one")
        summary["delta_f1_vs_scratch_pp"] = 100 * (summary.macro_f1 - summary.scratch_f1)
        summary["delta_mcc_vs_scratch"] = summary.mcc - summary.scratch_mcc
        summaries.append(summary)
    pairs = []
    match = ["fold_id", "cow_alias", "test_cow", "replicate", "budget_percent", "step", "endpoint", "weighting"]
    columns = ["macro_f1_fixed4", "macro_f1_supported", "mcc4_diagnostic", "accuracy", "walking_ap"]
    for left, right in [("act", "scratch"), ("ufro", "scratch"), ("act", "ufro")]:
        a = metrics[metrics.method.eq(left)][match + columns]
        b = metrics[metrics.method.eq(right)][match + columns]
        paired = a.merge(b, on=match, suffixes=("_left", "_right"), validate="one_to_one")
        paired["comparison"] = left + "-" + right
        for col in columns:
            paired[col + "_difference"] = paired[col + "_left"] - paired[col + "_right"]
        pairs.append(paired)
    return pd.concat(summaries, ignore_index=True), pd.concat(pairs, ignore_index=True)


def identity_coverage(truth):
    rows = []
    for alias, cow in IDS.items():
        t = truth[truth.cow_id.eq(cow)]
        row = {"cow_alias": alias, "cow_id": cow, "legacy_cow_id": "|".join(sorted(t.legacy_cow_id.astype(str).unique())),
               "interval_count": t.event_id.nunique(), "window_count": len(t), "session_count": t.session_id.nunique(),
               "identity_basis": "|".join(sorted(t.identity_basis.unique()))}
        for c, state in enumerate(STATES):
            s = t[t.target4.eq(c)]
            row[state.upper() + "_intervals"] = s.event_id.nunique()
            row[state.upper() + "_windows"] = len(s)
        rows.append(row)
    return pd.DataFrame(rows)


def score(bundle, output, original_truth=None):
    bundle, output = Path(bundle).resolve(), Path(output).resolve()
    require(not output.exists(), "Output directory exists; refusing overwrite")
    require(not output.is_relative_to(bundle), "Output must be outside the downloaded bundle")
    gate = verify_gate(bundle)
    # Receipt gate succeeds before labels are reconstructed or an optional truth file is read.
    truth = reconstruct_truth(bundle, gate["folds"], original_truth)
    rows, class_rows, confusion_rows, budgets = [], [], [], {}
    for unit_id, config in sorted(gate["units"].items()):
        receipt = gate["receipts"][unit_id]
        attempt = bundle / "results" / unit_id / receipt["attempt"]
        test = np.load(bundle / config["fold_dir"] / "test_indices.npy", allow_pickle=False)
        target = truth.iloc[test]
        budget = read_json(bundle / config["budget_dir"] / "summary.json")
        budgets[(config["fold_id"], config["percent"])] = budget
        train = np.load(bundle / config["budget_dir"] / "train_indices.npy", allow_pickle=False)
        train_rows = truth.iloc[train]
        require(not set(train) & set(test), "Train/test windows overlap")
        for key in ["cow_id", "event_id", "session_id"]:
            require(not set(train_rows[key]) & set(target[key]), f"Train/test {key} overlaps")
        batches = np.load(bundle / config["budget_dir"] / "batches.npy", allow_pickle=False)
        require(batches.shape == (3000, 32) and np.isin(batches, train).all(), "Frozen batch support differs")
        for step in STEPS:
            folder = attempt / f"step{step}"
            metadata = read_json(folder / "prediction_metadata.json")
            endpoint = "PRIMARY" if step == 1000 else "SENSITIVITY"
            require(metadata["step"] == step and metadata["endpoint"] == endpoint, "Prediction endpoint differs")
            require(metadata["state_order"] == STATES and metadata["head_order"] == HEADS, "Prediction class/head order differs")
            require(metadata["head_class_order"] == [["upright", "lying"], ["non_walking", "walking"], ["standing", "feeding"]], "Conditional class order differs")
            require(metadata["held_out_label_access"] is False and metadata["held_out_scoring"] is False, "Held-out scoring occurred on worker")
            require(metadata["checkpoint_sha256"] == sha(folder / "checkpoint.pt")
                    and metadata["test_indices_sha256"] == sha(folder / "test_indices.npy"), "Metadata artifact binding differs")
            indices = np.load(folder / "test_indices.npy", allow_pickle=False)
            require(np.array_equal(indices, test) and metadata["test_window_count"] == len(test), "Prediction index identity differs")
            states = np.load(folder / "test_state_probabilities.npy", allow_pickle=False)
            heads = np.load(folder / "test_head_probabilities.npy", allow_pickle=False)
            validate_probabilities(states, heads)
            require(len(states) == len(test), "Prediction/truth length differs")
            for weighting in WEIGHTINGS:
                identity = {"unit_id": unit_id, "fold_id": config["fold_id"], "cow_alias": target.cow_alias.iloc[0],
                            "test_cow": config["test_cow"], "replicate": config["replicate"], "method": config["method"],
                            "budget_percent": config["percent"], "step": step, "endpoint": endpoint, "weighting": weighting,
                            "identity_basis": gate["protocol"]["identity_basis"]}
                values, classes, matrix = compute_metrics(target.target4, states, target.event_id, weighting)
                rows.append({**identity, **values})
                class_rows.extend({**identity, **c} for c in classes)
                for c, pred in itertools.product(range(4), repeat=2):
                    confusion_rows.append({**identity, "true_class": STATES[c], "predicted_class": STATES[pred],
                                           "weighted_count": float(matrix[c, pred])})
    metrics = pd.DataFrame(rows)
    summary, pairs = aggregate(metrics)
    output.mkdir(parents=True, exist_ok=False)
    tables = {"metrics.csv": metrics, "per_cow_metrics.csv": metrics,
              "per_class.csv": pd.DataFrame(class_rows), "confusion_matrices.csv": pd.DataFrame(confusion_rows),
              "equal_cow_summary.csv": summary, "paired_differences.csv": pairs,
              "identity_and_coverage.csv": identity_coverage(truth), "label_budgets.csv": pd.DataFrame(budgets.values()),
              "reconstructed_test_truth.csv": truth}
    # Every downloaded artifact is catalogued, including failed attempts and negative outcomes.
    tables["bundle_inventory.csv"] = pd.DataFrame([
        {"path": p.relative_to(bundle).as_posix(), "bytes": p.stat().st_size, "sha256": sha(p)}
        for p in sorted(bundle.rglob("*")) if p.is_file()])
    for name, frame in tables.items():
        frame.to_csv(output / name, index=False, encoding="utf-8-sig", mode="x")
    audit = {"status": "PASS", "formal_units": 36, "endpoint_count": 72, "metric_row_count": len(metrics),
             "per_class_row_count": len(class_rows), "bundle": str(bundle), "scorer_sha256": sha(Path(__file__)),
             "protocol_sha256": gate["protocol_sha256"], "pinned_protocol_sha256_supplied": gate["pinned_protocol_sha256_supplied"],
             "final_archive_sha256": gate["final_archive_sha256"], "final_archive_receipt_sha256": gate["final_archive_receipt_sha256"],
             "archive_members_verified": gate["archive_members_verified"], "final_status_sha256": gate["final_status_sha256"],
             "frozen_artifact_count": gate["frozen_artifact_count"], "unit_receipt_sha256": gate["unit_receipt_sha256"],
             "receipt_gate_before_truth_reconstruction": True, "truth_basis": "reconstruction_from_six_frozen_fold_labels",
             "truth_cross_check": "Each row has five identical known labels and one held-out -1 matching its frozen cow ID",
             "original_truth_supplied": bool(original_truth), "original_truth_sha256": sha(original_truth) if original_truth else None,
             "primary_endpoint": "step1000/window", "sensitivity_endpoint": "step3000; no test-based endpoint selection",
             "class_order": STATES, "computational_repetitions": 1, "replicate_index": 0,
             "identity_basis": gate["protocol"]["identity_basis"], "identities_independently_verified": False,
             "legacy_recording_groups_overlap_train_test": True,
             "scope_definitions": {
                 "all_six_fixed4_diagnostic": "Six revised-ID folds, equal cow weights; fixed four labels with absent-class F1=0; diagnostic, not complete four-class validation",
                 "complete_four_class_A_to_D": "Four folds with all four truth classes; E/F excluded only from this declared complete-class scope",
                 "all_six_supported_classes": "Six folds; per-cow F1 mean over truth-supported classes, retaining all four outputs; E/F class sets differ; descriptive",
                 "mcc": "Multiclass MCC retains four output categories; complete-only MCC is null on E/F; supported-class F1 does not alter MCC",
                 "interval_balanced_window": "Each window weight is 1/(retained windows in its original interval); each interval total weight=1; not interval classification",
                 "missing_truth_classes": "E feeding and F lying recall are null; their fixed-class F1=0 is a convention, not observed recall",
                 "unsupported_predictions": "Predictions of absent truth classes remain errors; no output removal or renormalization"},
             "limits": ["User-attested field identity correction; not independently verified six-animal generalization",
                        "One computational seed; no population significance, confidence interval, or human-time-saving claim",
                        "Checkpoint bytes hash-verified but not executed or re-inferred by this scorer",
                        "All formal positive and negative results retained; all original bundle files remain unchanged"],
             "outputs": {name: {"rows": len(frame), "sha256": sha(output / name)} for name, frame in tables.items()},
             "completed_utc": datetime.now(timezone.utc).isoformat()}
    (output / "AUDIT.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True, help="Complete final download root with configs/source/inputs/results")
    parser.add_argument("--output", type=Path, required=True, help="New local analysis directory outside the bundle")
    parser.add_argument("--original-truth", type=Path, help="Optional original local truth for independent row-by-row comparison")
    args = parser.parse_args()
    try:
        audit = score(args.bundle, args.output, args.original_truth)
    except Exception as exc:
        print(json.dumps({"status": "FAIL_NOT_SCORED", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps({"status": audit["status"], "formal_units": audit["formal_units"],
                      "endpoint_count": audit["endpoint_count"], "metric_rows": audit["metric_row_count"],
                      "output": str(args.output.resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
