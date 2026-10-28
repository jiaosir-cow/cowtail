"""Copy selected existing results and immutable model evidence; no training/inference."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil
import pandas as pd

BASE = Path(r"PATH_X")
DEST = BASE / "四牛案例资料_20261027"
OUT = DEST / "02_实验结果"
MODELS = DEST / "03_模型与预测"
REVIEW = BASE / "实验记录与结果/ALL_POOL重分四组实验_20261027/all_retained_test_review"
SELECTED = ["cowA", "cowB", "cowC", "cowD"]
RENAME = {"cowA": "cowA", "cowB": "cowB", "cowC": "cowC", "cowD": "cowD"}
COPIED = []
DERIVED = []
TOUCHED = {}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def touch(path):
    path = Path(path)
    h = sha(path)
    if str(path) in TOUCHED:
        assert TOUCHED[str(path)] == h
    TOUCHED[str(path)] = h
    return h


def copy(source, target):
    source, target = Path(source), Path(target)
    before = touch(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        assert sha(target) == before, f"Existing destination mismatch: {target}"
    else:
        shutil.copy2(source, target)
    assert sha(target) == before == sha(source)
    COPIED.append({"source": str(source), "destination": str(target.relative_to(DEST)), "bytes": source.stat().st_size, "sha256": before})


def copy_tree(source, target, allow=None):
    for path in sorted(Path(source).rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and (allow is None or allow(path)):
            copy(path, Path(target)/path.relative_to(source))


def subset(source, target, column="group", version=True):
    touch(source)
    data = pd.read_csv(source)
    if column in data:
        mask = data[column].isin(SELECTED)
        if version and "version" in data:
            mask &= data.version.eq("new")
        data = data.loc[mask].copy()
        data.insert(data.columns.get_loc(column)+1, "source_analysis_group", data[column])
        data[column] = data[column].replace(RENAME)
    target.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(target, index=False, encoding="utf-8-sig")
    DERIVED.append({"path": str(target.relative_to(DEST)), "rows": len(data), "columns": len(data.columns), "source": str(source), "source_sha256": TOUCHED[str(source)], "sha256": sha(target)})
    return data


def table(headers, rows):
    return "\n".join(["| "+" | ".join(headers)+" |", "| "+" | ".join(["---"]*len(headers))+" |"]+["| "+" | ".join(map(str,r))+" |" for r in rows])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    four = REVIEW / "four_state_existing_models"
    zero = REVIEW / "source_only"
    for source in [four, zero]:
        target = OUT/"原始分析快照"/source.name
        for path in sorted(source.iterdir()):
            if path.is_file():
                copy(path, target/path.name)

    for name in ["group_counts.csv", "metrics_by_replicate.csv", "metrics_mean_std.csv", "class_metrics_by_replicate.csv", "class_metrics_mean_std.csv", "confusion_by_replicate.csv", "paired_deltas_vs_scratch.csv", "paired_deltas_mean_std.csv", "group_stability_summary.csv"]:
        subset(four/name, OUT/"四分类_统一旧牛号留出协议"/name)
    for name in ["source_only_per_seed.csv", "source_only_summary.csv", "group_support.csv", "rms_baseline.csv"]:
        subset(zero/name, OUT/"公开模型直接识别行走"/name, version=False)

    legacy = BASE/"revision_label_budget_20260909"
    target = MODELS/"四分类_旧牛号留出_90单元_仅1000步"
    for path in (legacy/"configs").glob("*.json"):
        copy(path, target/"configs"/path.name)
    copy_tree(legacy/"source", target/"source", lambda p: p.suffix in (".py", ".json"))
    copy_tree(legacy/"inputs", target/"inputs")
    copy(legacy/"analysis/test_truth.csv", target/"analysis/test_truth.csv")
    copy(BASE/"datasets/DATASET_X/windows.csv", target/"analysis/source_group_mapping.csv")
    manifest = pd.read_csv(four/"verified_prediction_manifest.csv")
    assert manifest.unit_id.nunique() == 90 and len(manifest) == 540
    for row in manifest.itertuples():
        assert touch(Path(row.path)) == row.sha256
    units = sorted(manifest.unit_id.unique())
    for uid in units:
        source = legacy/"download/bundle/results"/uid
        copy_tree(source, target/"results"/uid)
    assert len(list((target/"results").glob("*/attempt_001/step1000/checkpoint.pt"))) == 90
    assert not list((target/"results").glob("*/attempt_001/step3000/*"))
    (target/"归档范围说明.md").write_text("此处保存90个已有训练单元的1000步checkpoint、概率、索引、元数据、初始化审计，以及对应源代码、配置、完整输入数组与真值。原COMPLETE.json按原字节保留；历史3000步产物已不在原目录，原回执中涉及3000步的完整性不作本包验证结论。本包验证的是实际复制文件与540项1000步预测产物，未重新训练或推理。A/D/E取f01；当前显示Cow C取f02内原分析组Cow C1。\n", encoding="utf-8")

    sixroot = BASE/"实验记录与结果/六牛实验_20261019_20261020"
    six = sixroot/"download/bundle"
    sixout = MODELS/"四分类_原六组留出_ADE_18单元_1000及3000步"
    for path in (six/"configs").glob("*.json"):
        copy(path, sixout/"configs"/path.name)
    copy(six/"PROTOCOL.md", sixout/"PROTOCOL.md")
    copy_tree(six/"source", sixout/"source", lambda p: p.parent == six/"source" and p.suffix in (".py", ".json"))
    for name in ["acceleration.npy", "events.csv", "windows.csv", "IDENTITY_BASIS.json", "BUNDLE_SHA256SUMS.json"]:
        copy(six/"inputs"/name, sixout/"inputs"/name)
    copy_tree(six/"inputs/checkpoints", sixout/"inputs/checkpoints")
    for fold in ["f01", "f04", "f05"]:
        copy_tree(six/"inputs/folds"/fold, sixout/"inputs/folds"/fold)
        copy_tree(six/"inputs/budgets"/fold, sixout/"inputs/budgets"/fold)
        for source in (six/"results").glob(fold+"_*"):
            copy_tree(source, sixout/"results"/source.name)
    for script in ["score_six_results.py", "test_score_six_results.py", "render_six_report.py"]:
        copy(sixroot/"operations"/script, sixout/"analysis"/script)
    selected_six_receipts = list((sixout/"results").glob("*/COMPLETE.json"))
    assert len(selected_six_receipts) == 18
    for receipt_path in selected_six_receipts:
        attempt = json.loads(receipt_path.read_text(encoding="utf-8-sig"))["attempt"]
        for step in [1000, 3000]:
            assert (receipt_path.parent/attempt/f"step{step}"/"checkpoint.pt").is_file()
    (sixout/"归档范围说明.md").write_text("A/D/E共18个原六组留出单元的1000和3000步产物齐全。每单元以原COMPLETE.json指定的attempt为准，其中f05_s0_b100_scratch采用attempt_002；其attempt_001是中断记录，已一并保留，不能把该中断记录当作最终结果。所有最终回执列出的产物及其对应输入绑定均逐SHA256核验。\n", encoding="utf-8")
    score = sixroot/"analysis/scored_results"
    for path in sorted(score.glob("*")):
        if path.is_file():
            copy(path, OUT/"原始分析快照/原六组留出协议"/path.name)
    for name in ["metrics.csv", "per_cow_metrics.csv", "per_class.csv", "confusion_matrices.csv", "paired_differences.csv", "identity_and_coverage.csv"]:
        subset(score/name, OUT/"原六组留出协议_ADE"/name, column="cow_alias", version=False)
    copy(score/"reconstructed_test_truth.csv", sixout/"analysis/reconstructed_test_truth.csv")

    device = BASE/"实验记录与结果/零标签设备开发验证_20261025"
    frozen = device/"download/bundle"
    frozenout = MODELS/"公开模型直接识别_l01_l02_24份原始预测"
    copy_tree(frozen/"configs", frozenout/"configs")
    copy_tree(frozen/"source", frozenout/"source", lambda p: p.parent == frozen/"source" and p.suffix in (".py", ".json"))
    copy_tree(frozen/"inputs/checkpoints", frozenout/"inputs/checkpoints")
    for name in ["acceleration.npy", "metadata.csv"]:
        copy(frozen/"inputs"/name, frozenout/"inputs"/name)
    for fold in ["l01", "l02"]:
        copy_tree(frozen/"inputs/folds"/fold, frozenout/"inputs/folds"/fold)
        for seed in range(3):
            for source in ["act", "jbb", "ufro", "multi"]:
                uid = f"{fold}_s{seed}_{source}"
                fromdir = frozen/"results"/uid
                copy(fromdir/"COMPLETE.json", frozenout/"results"/uid/"COMPLETE.json")
                copy(fromdir/"predictions/frozen_step0000/clean.npz", frozenout/"results"/uid/"predictions/frozen_step0000/clean.npz")
    copy(device/"analysis/evaluation_truth_LOCAL_ONLY.csv", frozenout/"analysis/evaluation_truth_LOCAL_ONLY.csv")
    assert len(list((frozenout/"results").glob("*/predictions/frozen_step0000/clean.npz"))) == 24
    (frozenout/"归档范围说明.md").write_text("此处保存原始公开模型12份(4来源×3seed)，l01/l02两个原始牛号留出折的24份frozen_step0000/clean.npz，以及对应输入、索引、真值与源代码。COMPLETE原始回执涉及无标签适配和扰动等更多产物，按原字节作为来源记录保存；本目录只复制直接识别所需的24份clean预测，不声称复刻96单元全部适配/扰动输出。\n", encoding="utf-8")

    verified_six_outputs = 0
    missing_six_outputs = []
    for receipt_path in (sixout/"results").glob("*/COMPLETE.json"):
        receipt = json.loads(receipt_path.read_text(encoding="utf-8-sig"))
        for relative, expected in receipt["artifact_sha256"].items():
            path = receipt_path.parent/relative
            if path.exists():
                assert sha(path) == expected, str(path)
                verified_six_outputs += 1
            else:
                missing_six_outputs.append(str(path.relative_to(sixout)))
        for relative, expected in receipt["binding"]["artifact_sha256"].items():
            assert sha(sixout/relative) == expected, relative
    assert len(missing_six_outputs) == 0
    for receipt_path in (frozenout/"results").glob("*/COMPLETE.json"):
        receipt = json.loads(receipt_path.read_text(encoding="utf-8-sig"))
        relative = "predictions/frozen_step0000/clean.npz"
        assert sha(receipt_path.parent/relative) == receipt["output_sha256"][relative]
    dump(OUT/"ARCHIVED_RECEIPT_CHECK.json", {"status":"PASS_FOR_PRESENT_FILES","legacy_1000_step_artifacts_verified":540,"six_group_output_artifacts_verified":verified_six_outputs,"six_group_missing_historical_outputs":missing_six_outputs,"six_group_selected_input_bindings_verified":True,"source_only_clean_artifacts_verified":24,"limits":"Missing historical endpoints disclosed; no full original COMPLETE receipt validity claim for partial archives"})

    metrics = pd.read_csv(OUT/"四分类_统一旧牛号留出协议/metrics_mean_std.csv")
    pairs = pd.read_csv(OUT/"四分类_统一旧牛号留出协议/paired_deltas_mean_std.csv")
    source = pd.read_csv(OUT/"公开模型直接识别行走/source_only_summary.csv")
    counts = pd.read_csv(OUT/"四分类_统一旧牛号留出协议/group_counts.csv")
    pct = lambda n: f"{float(n)*100:.2f}"
    rows = []
    for cow in ["cowA", "cowB", "cowC", "cowD"]:
        for budget in [5, 10, 25, 50, 100]:
            r = metrics[(metrics.group==cow)&(metrics.percent==budget)&(metrics.method=="act")&(metrics.weighting=="window")].iloc[0]
            p = pairs[(pairs.group==cow)&(pairs.percent==budget)&(pairs.method=="act")&(pairs.weighting=="window")].iloc[0]
            rows.append([cow, budget, pct(r.accuracy_mean), pct(r.macro_f1_four_mean)+" ± "+pct(r.macro_f1_four_std), pct(r.macro_f1_supported_mean), f"{100*p.macro_f1_four_delta_vs_scratch_mean:+.2f}"])
    directrows = []
    for cow in ["cowA", "cowB", "cowC", "cowD"]:
        r=source[(source.group==cow)&(source.source=="ACT")].iloc[0]
        directrows.append([cow,pct(r.walking_ap_mean),pct(r.precision_mean),pct(r.recall_mean),pct(r.walking_f1_mean),int(r.walking_windows)])
    supportrows = [[r.group,int(r.events),int(r.windows),int(r.sessions),int(r.standing_windows),int(r.lying_windows),int(r.walking_windows),int(r.feeding_windows)] for r in counts.itertuples()]
    report = ["# 四牛实验结果", "", "显示名称为cowA、cowB、cowC、cowD。cowB沿用此前Cow C1的同一批样本及同一份预测，只改显示名称。本资料包没有重新训练、推理、修改标签或重新挑选样本。", "", "## 数据与评价口径", "", table(["案例","标注区间","窗口","会话","站立窗口","卧躺窗口","行走窗口","采食窗口"],supportrows), "", "四分类类别固定为站立、卧躺、行走、采食。cowD无采食真值；固定四类宏F1把该类F1按0计，另列有真值三类宏F1，二者不能混称。每个窗口默认同权，同时保留区间等权结果。原始采集会话、原标签和分组对应关系在数据及源分析快照中保留。", "", "## 公开模型直接识别行走", "", "下表为ACT直接使用、无需尾圈训练标签的结果，三个source seed的均值。AP是排序指标，不是准确率。其余JBB、UFRO、混合来源和每次重复、运动强度参照也完整保留。", "", table(["案例","行走AP%","精确率%","召回率%","行走F1%","行走窗口"],directrows), "", "## 四分类：统一旧牛号留出协议", "", "下表为ACT在已有1000步检查点上的三个重复均值。标签比例是当初训练预算，不是本次训练。cowA、cowC、cowD使用原f01测试预测，cowB使用原f02测试预测；四个案例由不同留出模型评价，并非同一个模型一次同时留出全部四牛。训练和测试按旧数据牛号隔离；分组更名不会改变模型或合并样本的总体表现。", "", table(["案例","标签预算%","准确率%","固定四类宏F1% ± SD","有真值类宏F1%","四类F1较Scratch差值pp"],rows), "", "完整结果覆盖5/10/25/50/100%预算、Scratch/ACT/UFRO三种起点、三个重复、窗口等权与区间等权；每类精确率/召回率/F1、混淆矩阵和正负向配对差值均在同目录CSV中。SD为三次训练重复的样本标准差，不是独立动物间误差。", "", "## 原六组留出协议中的cowA、cowC、cowD结果", "", "此前引用的cowC 63.83%→68.09%来自原六组逐组留出、10%预算、1000步、replicate 0、窗口等权的Scratch→ACT四类宏F1；cowD 64.13%→79.50%是同条件的有真值三类宏F1。两项与上表的旧牛号隔离协议不同，训练池也不同，必须分开解释。原六组协议仍存在原始记录组在训练与测试间交叉的限制，不能以此宣称已证明新动物独立泛化。原结果cowA、cowC、cowD的全部2预算×3方法×2端点及其逐类和混淆结果都保留。", "", "## 资料与复核", "", "02_实验结果保存四牛显示表及完整原始分析快照；03_模型与预测保存90份1000步旧牛号留出模型、cowA、cowC、cowD原六组18份模型的两个端点，以及24份公开模型直接识别原始预测与12份公开来源检查点。原快照和COMPLETE回执不改写；本包清单单独界定实际复制和验证范围。", "", "牧场身份由用户现场记录提供；本次没有独立核实其动物身份。数据档案中保留源组和原始标识，显示名称不增加独立动物数。所有成绩和来源均如实保留，未按效果删除失败条件。", ""]
    report.insert(-1, "原六组18单元的最终结果按COMPLETE指定attempt绑定，1000步与3000步各18份齐全。其中f05_s0_b100_scratch的最终attempt为attempt_002，中断的attempt_001仅作为原始记录保留。")
    (OUT/"四牛实验结果.md").write_text("\n".join(report),encoding="utf-8")

    for sourcepath, expected in TOUCHED.items():
        assert sha(sourcepath) == expected, f"Source changed during packaging: {sourcepath}"
    files=[]
    for directory in [OUT, MODELS]:
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.name != "RESULTS_MANIFEST.json":
                files.append({"path":str(path.relative_to(DEST)),"bytes":path.stat().st_size,"sha256":sha(path)})
    dump(OUT/"RESULTS_MANIFEST.json", {"status":"PASS","created_utc":datetime.now(timezone.utc).isoformat(),"display_names":["cowA","cowB","cowC","cowD"],"display_rename":{"cowA":"cowA","cowB":"cowB","cowC":"cowC","cowD":"cowD"},"training_or_inference_executed":False,"source_files_unchanged":True,"copy_hashes_verified":True,"source_copy_count":len(COPIED),"copied_files":COPIED,"derived_tables":DERIVED,"files":files,"inventory":{"four_state_legacy_units":90,"four_state_legacy_endpoint_steps":[1000],"six_group_ADE_units":18,"six_group_ADE_checkpoints_by_step":{"1000":18,"3000":18},"source_only_raw_predictions":24,"public_source_checkpoints":12},"completeness_limits":["No historical 3000-step legacy2fold integrity claim","Source-only bundle contains clean frozen outputs only, not all96 adapted/corrupted units","Source snapshots preserved unchanged including original group identity/protocol caveats"]})
    print(json.dumps({"status":"PASS","copied_files":len(COPIED),"all_output_files":len(files)+1,"total_bytes":sum(v['bytes'] for v in files),"derived_tables":len(DERIVED)},ensure_ascii=True))


if __name__ == "__main__":
    main()
