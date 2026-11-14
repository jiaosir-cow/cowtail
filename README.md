<div align="center">
  <img src="assets/brand/cowmata-logo.svg" width="300" alt="COWMATA">

# CowTail-4

**Tail-root IMU behaviour dataset for the manuscript's ten usable cows.**

[English](README.md) · [简体中文](README.zh-CN.md)

<img alt="data licence" src="https://img.shields.io/badge/data-CC%20BY%204.0-green">
<img alt="code licence" src="https://img.shields.io/badge/code-MIT-blue">

</div>

The manuscript uses ten usable cows. cowA–cowD are the reported test cattle: 12,736 non-overlapping 5.12 s windows from 31 sessions, with raw nine-axis streams. train1–train6 are six further animals used only as additional training data: 101 intervals and 879 windows, in [`01_data/additional_training/`](01_data/additional_training/README.md). The roster is [`01_data/study_roles.csv`](01_data/study_roles.csv).

## Install

From the repository root. Packages are in [`requirements.txt`](requirements.txt) (`numpy`, `pandas`).

```bash
pip install -r requirements.txt
python code/load_example.py            # first walking window
python code/load_example.py cowA 100   # cow, row_index
```

```python
import numpy as np
import pandas as pd

X = np.load("01_data/acceleration.npy")   # (12736, 3, 256), reported test cattle, normalised per window
meta = pd.read_csv("01_data/windows.csv", encoding="utf-8-sig")
row = meta.iloc[0]
approx_g = X[int(row.row_index)] * float(row.scale_g)
```

`row_index` is the row in `acceleration.npy`. Multiply by that row's `scale_g` for approximate acceleration in *g*. Order in the reported-test array: cowA 0–6341, cowC 6342–6669, cowD 6670–7223, cowB 7224–12735. Per-cow arrays are `01_data/<cow>/acceleration.npy`. Raw nine-axis streams for these four animals are in `04_raw_sensors_labels/` and join on `session_id`.

## Manuscript roster

| Role | Animals |
|---|---|
| Reported test, M1 | cowA, cowC, cowD |
| Reported test, M2 | cowB |
| Additional training only | train1–train6 |

M1 tests cowA, cowC and cowD. M2 tests cowB. Each reported test animal is left out of the training data of the model that evaluates it. train1–train6 are release labels ordered by window count. The manuscript does not name those six animals. They are not the cowA–cowF labels in `05_six_cows_full`.

## Reported test cattle

| | |
|---|---|
| Animals | cowA–cowD |
| Devices | 2 tail-root loggers, reused across animals |
| Sessions | 31, one day, one farm |
| Channels | 9 axes: acceleration, angular rate, magnetic field |
| Sampling | 50 Hz |
| Window | 256 samples, 5.12 s, non-overlapping |
| Windows | 12,736 |
| Classes | standing, lying, walking, feeding |
| Labels | Three annotators, then a veterinary graduate review |
| Alignment | Video–sensor residual ≤ 0.1 s |

- **Standing** — upright on all four limbs, including standing rumination.
- **Lying** — sternal or lateral recumbency.
- **Walking** — forward locomotion (198 windows).
- **Feeding** — intake at the trough. Rumination, licking, and head movement alone do not count as feeding.

cowD has no feeding labels (`FEEDING` = 0 in `01_data/four_cow_label_and_window_counts.csv`). Where standing and feeding overlap, both intervals are kept. The four-class table uses one label per window. Video is withheld. Public ids are `DEVICE_NN`, `SESSION_NN`, and relative milliseconds. Column definitions: [`docs/data_dictionary.md`](docs/data_dictionary.md).

## Layout

| Path | Contents |
|---|---|
| `01_data/` | Reported test cattle, `additional_training/`, and `study_roles.csv` |
| `02_results/` | Archived metrics |
| `03_models_predictions/` | Checkpoints and per-window predictions |
| `04_raw_sensors_labels/` | Raw nine-axis streams for the reported test cattle |
| `05_six_cows_full/` | Older export. Not the ten-cow roster |
| `code/` | Loader example and manifest builder |
| `FILE_MANIFEST.json` | Size and SHA-256 of every file |

Four-cow cowA is six-cow cowA. Four-cow cowB is one slice of six-cow cowB. Four-cow cowC is six-cow cowD. Four-cow cowD is six-cow cowE. Those six display names are not train1–train6. Counts and the crosswalk table are in [`05_six_cows_full/README.md`](05_six_cows_full/README.md).

Check the snapshot with `python verify_and_recompute.py`.

## Citation

```bibtex
@dataset{cowtail4_2026,
  title  = {CowTail-4: A Tail-Mounted Nine-Axis IMU Behaviour Dataset for Dairy Cattle},
  author = {Yang, Zhanhai and Jiao, Tengyu and Zhang, Jianqi and Zhang, Xiangqing and
            Zhang, Yalong and Jiang, Qinglin and Zhang, Yihui},
  year   = {2026},
  url    = {https://github.com/jiaosir-cow/cowtail},
  note   = {Data licence: CC BY 4.0; code licence: MIT}
}
```

The same record is in [`CITATION.cff`](CITATION.cff). Also cite the companion manuscript when it is published.

## Licence

Data: [CC BY 4.0](LICENSE). Code: [MIT](LICENSE-CODE). The COWMATA name and the files in [`assets/brand/`](assets/brand/) stay company brand assets. See [`NOTICE`](NOTICE).

Approved by the Ethics Committee of the Experimental Farm of Yangzhou University. External tail-root mounting only. Ten usable cows on one farm; the reported test recordings are from one day. Labels name observable states. Absolute time, device MAC addresses, original cattle IDs, video filenames, and annotator names are removed.

## Contact

Xiangqing Zhang (corresponding author) — jiaosirty@stu.yau.edu.cn

Questions: open a GitHub issue.
