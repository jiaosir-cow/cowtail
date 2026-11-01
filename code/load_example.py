#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Minimal end-to-end loader for the four-cow release.

Shows the three things a user needs on day one:
  1. read a labelled window from 01_data/acceleration.npy
  2. resolve which raw session it came from
  3. decode that session's 9-axis stream and slice the same 256 samples

Run from the repository root:
    python code/load_example.py            # picks the first WALKING window
    python code/load_example.py cowA 100   # cow, row_index
"""
import base64
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_session(path):
    """Decode one raw session JSON into (elapsed_ms, acc_g, gyro_dps, mag_gauss)."""
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)
    frames = np.frombuffer(base64.b64decode(obj["imu"]), dtype="<i2").reshape(-1, 11)
    u16 = frames.view("<u2")
    elapsed_ms = u16[:, 0].astype(np.int64) | (u16[:, 1].astype(np.int64) << 16)
    acc = frames[:, 2:5].astype(np.float32) / 4096.0
    gyro = frames[:, 5:8].astype(np.float32) / 32.0
    mag = frames[:, 8:11].astype(np.float32) / 1000.0
    return elapsed_ms, acc, gyro, mag


def main():
    cow = sys.argv[1] if len(sys.argv) > 1 else None
    row = int(sys.argv[2]) if len(sys.argv) > 2 else None

    pkg = pd.read_csv(os.path.join(ROOT, "01_data", "windows.csv"),
                      encoding="utf-8-sig", low_memory=False)
    if cow is None:
        cand = pkg[pkg.original_state == "WALKING"]
        rec = cand.iloc[0] if len(cand) else pkg.iloc[0]
        cow = rec.display_cow
        per = pd.read_csv(os.path.join(ROOT, "01_data", cow, "windows.csv"),
                          encoding="utf-8-sig", low_memory=False)
        sel = per[per.window_id == rec.window_id]      # 按 window_id 找回该牛表里的本地行
    else:
        per = pd.read_csv(os.path.join(ROOT, "01_data", cow, "windows.csv"),
                          encoding="utf-8-sig", low_memory=False)
        sel = per[per.row_index == row] if row is not None else per.head(1)
    if not len(sel):
        raise SystemExit(f"no window for cow={cow} row_index={row}")
    rec = sel.iloc[0]

    cow = rec.display_cow
    x = np.load(os.path.join(ROOT, "01_data", cow, "acceleration.npy"), mmap_mode="r")
    win = np.asarray(x[int(rec.row_index)])            # (3, 256), scale-normalised
    win_g = win * float(rec.scale_g)                    # back to approximate g

    raw_path = os.path.join(ROOT, "04_raw_sensors_labels", cow, "sensor_sessions",
                            rec.device_id, rec.session_id + ".json")
    elapsed_ms, acc, gyro, mag = load_session(raw_path)

    # 用时间定位原始帧；start_index 属于上游连续流，不能直接索引原始文件
    t0 = int(rec.t_start_rel_ms)
    f = int(np.searchsorted(elapsed_ms, t0))
    grid = np.linspace(t0, t0 + 5119, 256)
    j0, j1 = max(0, f - 2), min(len(elapsed_ms), f + 260)
    seg_res = np.stack([np.interp(grid, elapsed_ms[j0:j1], acc[j0:j1, c]) for c in range(3)],
                       axis=1)

    # 两条路径应对应同一段运动: 归一化后逐轴相关系数通常 0.9 左右
    a = win_g - win_g.mean(axis=1, keepdims=True)        # (3, 256)
    b = (seg_res - seg_res.mean(axis=0)).T               # (3, 256)
    corr = [float(np.corrcoef(a[c], b[c])[0, 1]) for c in range(3)]

    print(f"cow            : {cow}")
    print(f"session        : {rec.session_id}  device: {rec.device_id}")
    print(f"label          : {rec.original_state} (target4={rec.target4})")
    print(f"window_id      : {rec.window_id[:16]}…")
    print(f"time relative  : {t0} – {rec.t_end_rel_ms} ms "
          f"({(rec.t_end_rel_ms - t0) / 1000:.2f} s)")
    print(f"model array    : {win.shape} {win.dtype}  scale_g={rec.scale_g}")
    print(f"raw session    : {len(elapsed_ms)} frames, "
          f"{elapsed_ms[-1] / 60000:.1f} min, median dt "
          f"{int(np.median(np.diff(elapsed_ms)))} ms")
    print(f"raw slice      : frames {f}:{f + 256}")
    print(f"denormalised   : min={win_g.min():.3f} g  max={win_g.max():.3f} g")
    print(f"axis agreement : corr={np.round(corr, 3).tolist()}  (median≈0.94 expected)")
    print(f"acc/gyro/mag   : {acc.shape} {gyro.shape} {mag.shape}")
    print("OK — window and raw stream resolve to the same segment"
          if min(corr) > 0.8 else
          "WARNING — weak agreement, check the time-based join")


if __name__ == "__main__":
    main()
