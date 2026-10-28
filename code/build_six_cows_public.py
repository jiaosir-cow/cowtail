# -*- coding: utf-8 -*-
"""Source of 05_six_cows_full — kept for provenance, not runnable from the release.

Inputs (author-side only, not distributed):
  datasets/six_cows_field_record_20261018/   six-cow windows, events, labels
  datasets/BCF_repartition_B1_C1_F1_G1_20261027/  analysis-slice assignment
  四牛案例资料_20261027/01_数据/session_identity_mapping.csv  device/session numbering

It renumbers devices and sessions, renames animals to cowA-cowF, drops registry
identifiers and internal paths, and leaves every numeric column untouched.
"""
"""构建六牛全量脱敏数据集（训练池 + 测试池），输出到 四牛数据集_公开版/05_six_cows_full。

原则：
- 数值列一律不改（acceleration.npy 直接复制，CSV 数值逐列校验）。
- 设备/会话编号沿用四牛包已用的 DEVICE_01..、SESSION_01..，新增会话从 32 续编。
- 牛用 cowA..cowF，对应原始 Cow A..Cow F。
- 用 split 列表达 20201-3（train）/ 21074-1（test）两个记录桶；
  测试桶再附 analysis_slice（B1/C1/F1/G1）以保留论文的四切片结果。
"""
import json
import os
import shutil

import pandas as pd

SRC = r"datasets/six_cows_field_record_20261018"
FOUR = r"四牛案例资料_20261027"
BCF = r"datasets/BCF_repartition_B1_C1_F1_G1_20261027"
OUT = r"四牛数据集_公开版/05_six_cows_full"

COW = {"Cow A": "cowA", "Cow B": "cowB", "Cow C": "cowC",
       "Cow D": "cowD", "Cow E": "cowE", "Cow F": "cowF"}
POOL = {"20201-3": "train", "21074-1": "test"}

DROP = ["cache_key", "pseudo_group_id", "legacy_source_cow_id",
        "legacy_source_cow_alias", "identity_basis", "leakage_group",
        "legacy_leakage_group", "display_name"]


ID2COW = {}


def build_maps():
    # 会话编号：先沿用四牛包使用的 31 个（原始映射文件里按时间排序即当初的编号顺序）
    om = pd.read_csv(os.path.join(FOUR, "01_数据", "session_identity_mapping.csv"),
                     encoding="utf-8-sig")
    known = sorted(om.session_id.unique())
    sess = {s: f"SESSION_{i+1:02d}" for i, s in enumerate(known)}
    dev = {m: f"DEVICE_{i+1:02d}"
           for i, m in enumerate(sorted(om.device_mac.unique()))}

    ident = pd.read_csv(os.path.join(SRC, "cow_identity_and_counts.csv"), encoding="utf-8-sig")
    for _, r in ident.iterrows():
        ID2COW[str(r.cow_id)] = COW[str(r.cow_alias)]

    six = pd.read_csv(os.path.join(SRC, "windows.csv"), encoding="utf-8-sig", low_memory=False)
    extra = sorted(set(six.session_id) - set(known))
    for i, s in enumerate(extra):
        sess[s] = f"SESSION_{len(known)+i+1:02d}"
    for m in sorted(set(six.device_mac) - set(dev)):
        dev[m] = f"DEVICE_{len(dev)+1:02d}"
    return sess, dev


def add_group_col(df):
    """七组分析切片（A/B1/C1/D/E/F1/G1），来自论文的固定分组表。"""
    bw = pd.read_csv(os.path.join(BCF, "windows.csv"), encoding="utf-8-sig", low_memory=False)
    if "analysis_group" not in bw.columns:
        raise RuntimeError("BCF windows.csv 缺少 analysis_group")
    m = bw.set_index("window_id").analysis_group.str.replace("Cow ", "", regex=False)
    df["analysis_slice"] = df.window_id.map(m)
    return df


def anonymize(df, sess, dev, is_window=True):
    df = df.copy()
    if "device_mac" in df:
        df["device_id"] = df.device_mac.map(dev)
        df = df.drop(columns=["device_mac"])
    if "session_id" in df:
        df["session_id"] = df.session_id.map(sess)
    if "cow_id" in df:
        df["cow_id"] = df.cow_id.map(lambda x: ID2COW.get(str(x), x))
    if "cow_alias" in df:
        df["cow_alias"] = df.cow_alias.map(lambda x: COW.get(str(x), x))
    if "legacy_cow_id" in df:
        df["split"] = df.legacy_cow_id.map(POOL)
        df = df.drop(columns=["legacy_cow_id"])
    df = df.drop(columns=[c for c in DROP if c in df.columns])
    if is_window:
        df = add_group_col(df)
    else:
        # 事件表用 event_id 关联分组
        be = pd.read_csv(os.path.join(BCF, "events.csv"), encoding="utf-8-sig", low_memory=False)
        if "analysis_group" in be.columns:
            m = be.set_index("event_id").analysis_group.str.replace("Cow ", "", regex=False)
            df["analysis_slice"] = df.event_id.map(m)
        else:
            df["analysis_slice"] = pd.NA
    # 列顺序：标识在前，标签与数值在后
    front = [c for c in ["split", "analysis_slice", "cow_id", "device_id", "session_id"] if c in df.columns]
    rest = [c for c in df.columns if c not in front]
    return df[front + rest]


def main():
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)
    sess, dev = build_maps()

    w = pd.read_csv(os.path.join(SRC, "windows.csv"), encoding="utf-8-sig", low_memory=False)
    e = pd.read_csv(os.path.join(SRC, "events.csv"), encoding="utf-8-sig", low_memory=False)
    t = pd.read_csv(os.path.join(SRC, "test_truth.csv"), encoding="utf-8-sig", low_memory=False)

    wa = anonymize(w, sess, dev, True)
    ea = anonymize(e, sess, dev, False)
    ta = anonymize(t, sess, dev, False)

    wa.to_csv(os.path.join(OUT, "windows.csv"), index=False, encoding="utf-8-sig")
    ea.to_csv(os.path.join(OUT, "events.csv"), index=False, encoding="utf-8-sig")
    ta.to_csv(os.path.join(OUT, "test_truth.csv"), index=False, encoding="utf-8-sig")

    shutil.copy2(os.path.join(SRC, "acceleration.npy"), os.path.join(OUT, "acceleration.npy"))
    shutil.copy2(os.path.join(SRC, "source_window_indices.npy"),
                 os.path.join(OUT, "source_window_indices.npy"))

    # 每头牛的子目录
    for cow_src, cow_dst in COW.items():
        d = os.path.join(OUT, cow_dst)
        os.makedirs(d)
        lab = pd.read_csv(os.path.join(SRC, cow_src, "labels.csv"),
                          encoding="utf-8-sig", low_memory=False)
        anonymize(lab, sess, dev, False).to_csv(
            os.path.join(d, "labels.csv"), index=False, encoding="utf-8-sig")
        shutil.copy2(os.path.join(SRC, cow_src, "window_indices.npy"),
                     os.path.join(d, "window_indices.npy"))

    summary = (wa.groupby(["cow_id", "split", "original_state"]).size()
               .unstack(fill_value=0).reset_index())
    summary.to_csv(os.path.join(OUT, "class_support.csv"), index=False, encoding="utf-8-sig")

    print("窗口:", len(wa), " 事件:", len(ea), " 会话:", wa.session_id.nunique(),
          " 设备:", wa.device_id.nunique())
    print(summary.to_string(index=False))
    print("\nanalysis_slice 分布:")
    print(wa.analysis_slice.fillna("(train)").value_counts().to_string())


if __name__ == "__main__":
    main()
