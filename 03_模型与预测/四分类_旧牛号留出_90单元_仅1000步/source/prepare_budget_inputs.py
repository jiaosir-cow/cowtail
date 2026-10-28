"""Prepare frozen event-budget inputs; never train or score a model.

The original annotation CSV and prior audited replay artifacts are read-only.
Actual target labels are fold-masked; complete truth stays under local analysis/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

STATES = ("STANDING", "LYING", "WALKING", "FEEDING")
STATE_TO_ID = {state: index for index, state in enumerate(STATES)}
PERCENTS = (5, 10, 25, 50, 100)
ANCHORS = ("COWID_02", "COWID_03")
BASE_SEED = 20260909
EXPECTED_AUDIT = "058a34756ab6ceb290382eb14a8c82582f7c9141648f7243143db260facb8961"
EXPECTED_ANNOTATIONS = "c2e68d919da2887a71292d8f040e093f2af3eddffb53ded0f5dfc7b3ef9e0972"
DEFAULT_ANNOTATIONS = Path(r"PATH_X")


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def save_array(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.save(stream, value, allow_pickle=False)


def save_table(path, table):
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, encoding="utf-8", mode="x", lineterminator="\n")


def leakage_group(cow_id: str) -> str:
    return "23489_uncertain_pair" if cow_id in {"COWID_09", "COWID_10"} else cow_id


def split_fold(windows: pd.DataFrame, labels: np.ndarray, test_cow: str):
    require(len(labels) == len(windows), "Label/window length mismatch")
    test_mask = windows.cow_id.eq(test_cow).to_numpy()
    require(test_mask.any() and (~test_mask).any(), "Empty train/test fold")
    training_groups = set(windows.loc[~test_mask, "leakage_group"])
    test_groups = set(windows.loc[test_mask, "leakage_group"])
    require(not training_groups.intersection(test_groups), "Cow leakage group crosses train/test")
    training_events = set(windows.loc[~test_mask, "event_id"])
    test_events = set(windows.loc[test_mask, "event_id"])
    require(not training_events.intersection(test_events), "Event crosses train/test")
    masked = np.asarray(labels, dtype=np.int64).copy()
    masked[test_mask] = -1
    return masked, np.flatnonzero(test_mask).astype(np.int64), np.flatnonzero(~test_mask).astype(np.int64)


def event_order(event_ids, fold_number: int, replicate: int):
    """Class-blind selection: this function has no label/support argument."""
    ordered = np.asarray(sorted(set(str(value) for value in event_ids)), dtype=str)
    require(len(ordered) > 0, "No eligible training events")
    rng = np.random.default_rng(np.random.SeedSequence([BASE_SEED, fold_number, replicate, 101]))
    return rng.permutation(ordered).tolist()


def select_prefix(order, percent: int):
    require(0 < percent <= 100, "Invalid percentage")
    count = (len(order) * percent + 99) // 100
    return list(order[:count])


def draw_batches(selected_event_ids, rows_by_event, fold_number, replicate, percent,
                 updates=3000, batch_size=32):
    """Uniform original event first, uniform retained window second."""
    require(len(selected_event_ids) > 0, "Empty selected events")
    groups = [np.asarray(rows_by_event[event], dtype=np.int64) for event in selected_event_ids]
    require(all(len(group) > 0 for group in groups), "Event has no retained windows")
    sizes = np.asarray([len(group) for group in groups], dtype=np.int64)
    offsets = np.r_[0, np.cumsum(sizes)[:-1]]
    flat_rows = np.concatenate(groups)
    require(flat_rows.min() >= 0 and flat_rows.max() <= np.iinfo(np.int32).max, "Invalid row index")
    rng = np.random.default_rng(np.random.SeedSequence([BASE_SEED, fold_number, replicate, percent, 202]))
    events = rng.integers(len(groups), size=updates * batch_size)
    local_rows = rng.integers(0, sizes[events])
    return flat_rows[offsets[events] + local_rows].reshape(updates, batch_size).astype(np.int32)


def class_support(values):
    values = np.asarray(values)
    return {state: int(np.sum(values == index)) for state, index in STATE_TO_ID.items()}


def prepare(root: Path, source: Path, annotation_path: Path):
    root, source, annotation_path = root.resolve(), source.resolve(), annotation_path.resolve()
    inputs, analysis = root / "inputs", root / "analysis"
    owned = [inputs / item for item in ("acceleration.npy", "windows.csv", "events.csv", "folds", "budgets", "PREPARATION.json")]
    owned.append(analysis / "test_truth.csv")
    require(not any(path.exists() for path in owned), "Owned output exists; preserve it and use a new root")
    require((root / "PROTOCOL.md").is_file(), "Missing frozen protocol")
    sources = {
        "annotation_csv": annotation_path,
        "continuous_array": source / "inputs/continuous_acceleration.npy",
        "continuous_index": source / "inputs/continuous_windows.csv",
        "truth_alignment": source / "analysis/results/window_truth_alignment.csv",
        "cold_audit": source / "analysis/results/AUDIT.json",
        "cold_independent_audit": source / "analysis/INDEPENDENT_AUDIT.json",
        "cold_preparation": source / "inputs/COMPLETE_PREPARATION.json",
        "cold_input_manifest": source / "inputs/continuous_inputs_manifest.json",
        "protocol": root / "PROTOCOL.md",
        "preparer": Path(__file__).resolve(),
    }
    for name in ("act", "ufro"):
        for replicate in range(3):
            sources[f"checkpoint_{name}_seed{replicate}"] = source / f"inputs/checkpoints/{name}_seed{replicate}.pt"
    before = {key: sha256(path) for key, path in sources.items()}
    require(before["annotation_csv"] == EXPECTED_ANNOTATIONS, "V2 annotation hash mismatch")
    require(before["cold_audit"] == EXPECTED_AUDIT, "Archived audit hash mismatch")
    audit = read_json(sources["cold_audit"])
    independent = read_json(sources["cold_independent_audit"])
    require(audit["status"] == "PASS" and independent["status"] == "PASS", "Prior audit did not pass")
    require(independent["formal_result_sha256"]["AUDIT.json"] == before["cold_audit"], "Independent audit binding failed")
    require(audit["annotation_sha256"] == before["annotation_csv"], "Annotation binding failed")
    require(audit["artifact_sha256"]["window_truth_alignment.csv"] == before["truth_alignment"], "Alignment hash mismatch")
    expected_inputs = audit["prediction_gate"]["input_sha256"]
    for key in ("continuous_array", "continuous_index", "cold_preparation", "cold_input_manifest"):
        relative = sources[key].relative_to(source).as_posix()
        require(expected_inputs[relative] == before[key], f"Archived source hash mismatch: {relative}")
    for key in [key for key in sources if key.startswith("checkpoint_")]:
        require(expected_inputs[sources[key].relative_to(source).as_posix()] == before[key], f"Checkpoint hash mismatch: {key}")
    print("Verified original annotation, archived array/index/alignment and six public checkpoints.", flush=True)

    windows = pd.read_csv(sources["continuous_index"], dtype={"cow_id": str, "event_id": str})
    aligned = pd.read_csv(sources["truth_alignment"], keep_default_na=False)
    raw = pd.read_csv(annotation_path, keep_default_na=False)
    require(len(windows) == 135958 and len(aligned) == len(windows), "Unexpected continuous source size")
    require(np.array_equal(windows.row_index.to_numpy(), np.arange(len(windows))), "Source rows not ordered")
    require(np.array_equal(windows.row_index.to_numpy(), aligned.row_index.to_numpy()), "Alignment row mismatch")
    require(np.array_equal(windows.window_id.to_numpy(), aligned.window_id.to_numpy()), "Alignment window identity mismatch")
    require(not windows.window_id.duplicated().any(), "Duplicated source window ID")
    keep = aligned.truth_status.eq("FULLY_LABELED").to_numpy()
    continuous_rows = np.flatnonzero(keep)
    retained = windows.loc[keep].reset_index(drop=True).copy()
    truth = aligned.loc[keep].reset_index(drop=True)
    require(len(retained) == 29790, "Unexpected full-window count")
    require(truth.original_state.isin(STATES).all(), "Unexpected four-state label")
    require(truth.containing_annotation_ids.str.len().gt(0).all(), "Missing event ID")
    require(not truth.containing_annotation_ids.str.contains(";", regex=False).any(), "Multiple containing event IDs")
    retained = retained.rename(columns={"row_index": "continuous_row_index"})
    retained.insert(0, "row_index", np.arange(len(retained), dtype=np.int64))
    retained["event_id"] = truth.containing_annotation_ids.astype(str)
    retained["leakage_group"] = retained.cow_id.map(leakage_group)
    require(retained.event_id.nunique() == 882, "Unexpected eligible event count")
    require(not raw.event_id.duplicated().any(), "Duplicate raw annotation event ID")
    mother = raw.set_index("event_id", drop=False)
    selected_mother = mother.loc[retained.event_id].reset_index(drop=True)
    for column in ("cow_id", "device_mac", "session_id"):
        require((retained[column].astype(str).to_numpy() == selected_mother[column].astype(str).to_numpy()).all(), f"Raw identity mismatch: {column}")
    require(selected_mother.layer.eq("body_state").all(), "Nonbody event enters pool")
    require((selected_mother.code.to_numpy() == truth.original_state.to_numpy()).all(), "Original state mismatch")
    require((selected_mother.t_start_rel_ms.to_numpy(float) <= retained.t_start_rel_ms.to_numpy(float)).all(), "Window before original interval")
    require((selected_mother.t_end_rel_ms.to_numpy(float) >= retained.t_end_rel_ms.to_numpy(float)).all(), "Window after original interval")
    require((retained.stop_index - retained.start_index).eq(256).all(), "Window frame count mismatch")
    require(np.allclose(retained.t_end_rel_ms - retained.t_start_rel_ms, 5120.0, rtol=0, atol=1e-6), "Window duration mismatch")
    labels = truth.original_state.map(STATE_TO_ID).to_numpy(np.int64)
    event_ids = sorted(retained.event_id.unique())
    events = mother.loc[event_ids, ["event_id", "cow_id", "device_mac", "session_id", "t_start_rel_ms", "t_end_rel_ms"]].copy().reset_index(drop=True)
    events["original_duration_ms"] = events.t_end_rel_ms.to_numpy(float) - events.t_start_rel_ms.to_numpy(float)
    events["window_count"] = events.event_id.map(retained.event_id.value_counts()).astype(np.int64)
    events["leakage_group"] = events.cow_id.map(leakage_group)
    require(events.original_duration_ms.gt(0).all(), "Invalid original event duration")
    event_classes = dict(zip(events.event_id, mother.loc[events.event_id, "code"].map(STATE_TO_ID).astype(int)))
    rows_by_event = {str(key): group.row_index.to_numpy(np.int64) for key, group in retained.groupby("event_id", sort=False)}
    durations = dict(zip(events.event_id, events.original_duration_ms.astype(float)))

    source_array = np.load(sources["continuous_array"], mmap_mode="r", allow_pickle=False)
    require(source_array.shape == (135958, 3, 256) and source_array.dtype == np.float32, "Source array contract mismatch")
    inputs.mkdir(parents=True, exist_ok=True)
    target = np.lib.format.open_memmap(inputs / "acceleration.npy", mode="w+", dtype=np.float32, shape=(len(retained), 3, 256))
    for start in range(0, len(retained), 1024):
        stop = min(len(retained), start + 1024)
        block = np.asarray(source_array[continuous_rows[start:stop]])
        require(np.isfinite(block).all(), "Nonfinite retained acceleration")
        target[start:stop] = block
    target.flush()
    del target, source_array
    outputs = [inputs / "acceleration.npy"]
    save_table(inputs / "windows.csv", retained)
    save_table(inputs / "events.csv", events)
    outputs.extend([inputs / "windows.csv", inputs / "events.csv"])
    local_truth = retained[["row_index", "event_id", "cow_id"]].copy()
    local_truth["target4"] = labels
    local_truth["original_state"] = truth.original_state.to_numpy()
    save_table(analysis / "test_truth.csv", local_truth)
    outputs.append(analysis / "test_truth.csv")
    require(not set(STATES).intersection(retained.columns), "Class column in common metadata")
    require(not {"target4", "original_state", "code", "label", "y_true"}.intersection(events.columns), "Class in event metadata")

    fold_records, budget_records = [], []
    for fold_number, test_cow in enumerate(ANCHORS, start=1):
        fold_id = f"f{fold_number:02d}"
        fold_dir = inputs / "folds" / fold_id
        masked, test_indices, train_indices = split_fold(retained, labels, test_cow)
        require((masked[test_indices] == -1).all(), "Test labels exposed")
        train_events = sorted(retained.iloc[train_indices].event_id.unique())
        save_array(fold_dir / "train_labels.npy", masked)
        save_array(fold_dir / "test_indices.npy", test_indices)
        outputs.extend([fold_dir / "train_labels.npy", fold_dir / "test_indices.npy"])
        fold_record = {"fold_id": fold_id, "fold_number": fold_number, "test_cow": test_cow,
                       "train_cows": sorted(retained.iloc[train_indices].cow_id.unique()),
                       "available_event_count": len(train_events), "available_window_count": len(train_indices),
                       "test_event_count": int(retained.iloc[test_indices].event_id.nunique()),
                       "test_window_count": len(test_indices), "validation_event_count": 0,
                       "fold_dir": fold_dir.relative_to(root).as_posix()}
        fold_records.append(fold_record)
        for replicate in range(3):
            order = event_order(train_events, fold_number, replicate)
            previous = set()
            for percent in PERCENTS:
                selected = select_prefix(order, percent)
                require(previous.issubset(selected), "Non-nested budgets")
                previous = set(selected)
                selected_rows = np.sort(np.concatenate([rows_by_event[event] for event in selected])).astype(np.int64)
                require(np.isin(selected_rows, train_indices).all(), "Selected event contains test rows")
                batches = draw_batches(selected, rows_by_event, fold_number, replicate, percent)
                require(np.isin(batches, selected_rows).all(), "Batch outside selected original events")
                budget_dir = inputs / "budgets" / fold_id / f"rep{replicate:02d}" / f"p{percent:03d}"
                write_json(budget_dir / "selected_event_ids.json", selected)
                save_array(budget_dir / "train_indices.npy", selected_rows)
                save_array(budget_dir / "batches.npy", batches)
                budget_files = [budget_dir / name for name in ("selected_event_ids.json", "train_indices.npy", "batches.npy")]
                summary = {"fold_id": fold_id, "fold_number": fold_number, "test_cow": test_cow,
                           "replicate": replicate, "nominal_percent": percent,
                           "available_event_count": len(train_events), "selected_event_count": len(selected),
                           "actual_event_fraction": len(selected) / len(train_events),
                           "selected_window_count": len(selected_rows), "test_window_count": len(test_indices),
                           "original_interval_duration_ms": math.fsum(durations[event] for event in selected),
                           "selected_class_support": class_support(labels[selected_rows]),
                           "selected_event_class_support": class_support([event_classes[event] for event in selected]),
                           "missing_selected_classes": [state for state, count in class_support([event_classes[event] for event in selected]).items() if count == 0],
                           "batches_shape": [3000, 32], "batch_dtype": "int32",
                           "event_order_seedsequence": [BASE_SEED, fold_number, replicate, 101],
                           "batch_seedsequence": [BASE_SEED, fold_number, replicate, percent, 202],
                           "sampler": "uniform selected original event, then uniform retained event window",
                           "uses_class_for_selection": False, "missing_class_redraw": False,
                           "budget_dir": budget_dir.relative_to(root).as_posix(),
                           "sha256": {path.name: sha256(path) for path in budget_files}}
                write_json(budget_dir / "summary.json", summary)
                budget_records.append(summary)
                outputs.extend(budget_files + [budget_dir / "summary.json"])
            print(f"Prepared {fold_id}, paired replicate {replicate}: {len(train_events)} training events.", flush=True)

    after = {key: sha256(path) for key, path in sources.items()}
    require(before == after, "Read-only source changed during preparation")
    result = {"status": "complete", "schema_version": 1,
              "completed_at_utc": datetime.now(timezone.utc).isoformat(),
              "array_shape": [29790, 3, 256], "array_dtype": "float32", "window_count": 29790,
              "eligible_event_count": 882, "original_body_event_count": 974,
              "excluded_original_body_event_count": 92, "class_order": list(STATES),
              "window_class_support": class_support(labels),
              "eligible_event_class_support": class_support([event_classes[event] for event in event_ids]),
              "folds": fold_records, "budgets": budget_records,
              "source_paths": {key: str(path) for key, path in sources.items()},
              "source_sha256_before": before, "source_sha256_after": after,
              "source_hashes_unchanged": True, "protocol_text_sha256": before["protocol"],
              "preparer_code_sha256": before["preparer"],
              "artifact_sha256": {path.relative_to(root).as_posix(): sha256(path) for path in outputs},
              "local_only_artifacts": ["analysis/test_truth.csv"],
              "retrospective_development_benchmark": True, "human_time_measured": False,
              "budget_unit": "original event_id with at least one fully contained sensor-defined window",
              "no_validation_checkpoint_selection": True,
              "other_fold_training_labels_exist_in_bundle": True,
              "identity_note": "Original cow_id preserved; 23489 pair grouped conservatively for leakage only, not declared one animal."}
    write_json(inputs / "PREPARATION.json", result)
    print(json.dumps({key: result[key] for key in ("status", "array_shape", "eligible_event_count", "window_class_support", "folds")}, ensure_ascii=False, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--source", type=Path)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    args = parser.parse_args()
    source = args.source or args.root.parent / "revision_cold_start_20260908"
    prepare(args.root, source, args.annotations)


if __name__ == "__main__":
    main()
