"""Copy source bytes for the four displayed cases; never alter source assets."""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
PAPER = ROOT.parent.parent
DATASET = PAPER / "datasets" / "DATASET_X"
CANONICAL = Path(r"PATH_X")
DISPLAY = {"cowA": "cowA", "cowB": "cowB", "cowC": "cowC", "cowD": "cowD"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relocated(value: str) -> Path:
    return Path(value.replace("PATH_X", "PATH_X").replace("\\DATASET_X数据集\\new\\", "\\DATASET_X数据集\\标签\\"))


def main() -> None:
    canonical_before = sha256(CANONICAL)
    windows_before = sha256(DATASET / "windows.csv")
    windows = pd.read_csv(DATASET / "windows.csv")
    sessions = windows[windows.analysis_group.isin(DISPLAY)][["analysis_group", "device_mac", "session_id"]].drop_duplicates()
    assert len(sessions) == 31
    labels = pd.read_csv(CANONICAL).merge(sessions, on=["device_mac", "session_id"], validate="many_to_one")
    assert len(labels) == 468
    labels["display_case"] = labels.analysis_group.map(DISPLAY)
    jobs = []
    matches = []
    missing = []
    for old_path, group in labels.groupby("source_csv_path", sort=False):
        path = relocated(old_path)
        if not path.is_file():
            missing.append({"kind": "source_csv", "original_path": old_path, "resolved_path": str(path)})
            continue
        source_rows = pd.read_csv(path)
        for _, row in group.iterrows():
            ordinal = int(row.source_row_ordinal)
            candidate = source_rows.iloc[ordinal - 1]
            for field in ["code", "layer", "session_id", "t_start_rel_ms", "t_end_rel_ms"]:
                equal = float(row[field]) == float(candidate[field]) if field.startswith("t_") else str(row[field]) == str(candidate[field])
                assert equal, (row.event_id, field, row[field], candidate[field])
            matches.append({"display_case": row.display_case, "analysis_group": row.analysis_group, "event_id": row.event_id, "source_csv_path": old_path, "resolved_csv_path": str(path), "source_row_ordinal": ordinal, "code_layer_session_start_end_match": True})
    for field, kind in [("source_raw_path", "sensor_json"), ("source_csv_path", "source_events_csv"), ("source_meta_path", "source_events_meta")]:
        for old_path, group in labels.groupby(field, sort=False):
            assert group.display_case.nunique() == 1
            case = group.display_case.iloc[0]
            device = group.device_mac.iloc[0]
            session = group.session_id.iloc[0]
            path = relocated(old_path)
            if not path.is_file():
                missing.append({"kind": kind, "original_path": old_path, "resolved_path": str(path)})
                continue
            if kind == "sensor_json":
                destination = ROOT / case / "sensor_sessions" / device / path.name
            else:
                relative_labels = path.relative_to(Path(r"PATH_X"))
                destination = ROOT / case / "source_labels" / relative_labels
            jobs.append((case, group.analysis_group.iloc[0], device, session, kind, old_path, path, destination))
    assert not missing, missing
    assert len(jobs) == 107
    index = []
    for case, group, device, session, kind, old_path, source, destination in jobs:
        before = sha256(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            assert sha256(destination) == before, str(destination)
        else:
            shutil.copy2(source, destination)
        copied = sha256(destination)
        after = sha256(source)
        assert before == copied == after
        index.append({"display_case": case, "analysis_group": group, "device_mac": device, "session_id": session, "asset_kind": kind, "canonical_original_path": old_path, "resolved_source_path": str(source), "destination_relative_to_raw_assets": str(destination.relative_to(ROOT)), "bytes": source.stat().st_size, "source_sha256_before": before, "destination_sha256": copied, "source_sha256_after": after, "all_sha256_equal": True, "path_resolution": "literal_drive_relocation_and_new_to_label_directory"})
    pd.DataFrame(index).to_csv(ROOT / "raw_asset_index.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(matches).to_csv(ROOT / "canonical_source_row_verification.csv", index=False, encoding="utf-8-sig")
    summary = pd.DataFrame(index).groupby(["display_case", "asset_kind"], sort=True).agg(file_count=("asset_kind", "size"), total_bytes=("bytes", "sum")).reset_index()
    summary.to_csv(ROOT / "raw_asset_counts.csv", index=False, encoding="utf-8-sig")
    assert canonical_before == sha256(CANONICAL)
    assert windows_before == sha256(DATASET / "windows.csv")
    verification = {
        "status": "PASS", "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_canonical_sha256_before_and_after": canonical_before,
        "source_windows_sha256_before_and_after": windows_before,
        "sensor_session_count": 31, "canonical_annotation_count": 468,
        "source_events_csv_count": 38, "source_events_meta_count": 38,
        "copied_asset_count": len(index), "canonical_source_row_match_count": len(matches),
        "source_and_destination_hash_mismatch_count": 0, "missing_required_assets": missing,
        "raw_json_schema_observed": {"top_level_keys": ["configs", "create_by", "create_time", "device", "imu", "motion", "temperature", "uid", "update_by", "update_time", "version"], "imu_encoding": "encoded string; axis decoding not independently revalidated in this packaging operation"},
        "display_rename": {"cowA": "cowA", "cowB": "cowB", "cowC": "cowC", "cowD": "cowD"},
        "notes": ["Original sensor JSON, original CSV, and original events_meta bytes are copied unchanged.", "Display aliases do not replace source animal identifiers or prove independent animal identity.", "Canonical labels include source row selection and adjudication; source CSVs may contain additional rows not selected in the canonical table.", "Derived per-event clips, videos, software environments and caches are not necessary for this full-session raw archive and are not copied."]
    }
    (ROOT / "RAW_ASSETS_VERIFIED.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (ROOT / "notes.md").write_text("""# 原始传感与标签资料

本目录保存 cowA、cowB、cowC、cowD 四个展示案例所对应的完整原始传感会话、原始标签 CSV 及其 events_meta 文件。cowB 对应原 cowB 的展示记录；源文件中的牛号、设备号、时间、标注内容保持原样。

- 完整传感会话：31 个，按展示案例和原设备号保存。
- 原始标签 CSV：38 个；对应元数据：38 个。
- 这些会话对应的规范标注：468 条，已逐行核对源 CSV 中的行序号、行为码、标注层、会话和起止时间，全部一致。规范标注表在资料包的数据部分保存；原 CSV 还可能包括规范表未选用的其他标注行。
- 107 个源文件复制前、复制后和目标文件的 SHA-256 全部一致。详见 `raw_asset_index.csv`、`canonical_source_row_verification.csv` 和 `RAW_ASSETS_VERIFIED.json`。
- 历史路径通过固定盘符迁移及 `new` 到 `标签` 的目录迁移恢复，无任意版本替换。
- 原始 JSON 中 IMU 为编码字符串；本次只做原字节归档，未另行声称完成轴向解码或硬件校准验证。
- 未复制视频、运行环境、缓存和派生事件切片；完整传感会话与对应原始标签均已保存。
""", encoding="utf-8")
    print(json.dumps({"status": "PASS", "asset_count": len(index), "total_bytes": int(sum(r["bytes"] for r in index)), "by_case_and_kind": summary.to_dict("records")}, ensure_ascii=True))


if __name__ == "__main__":
    main()
