<div align="center">
  <img src="assets/brand/cowmata-logo.svg" width="300" alt="COWMATA">

# CowTail-4

**A fully labelled 9-axis IMU behaviour dataset recorded at the cattle tail root**

[English](README.md) · [简体中文](README.zh-CN.md)

<img alt="cows" src="https://img.shields.io/badge/cows-4-2f6f4e">
<img alt="sessions" src="https://img.shields.io/badge/sessions-31-2f6f4e">
<img alt="windows" src="https://img.shields.io/badge/windows-12%2C736-1f4e79">
<img alt="rate" src="https://img.shields.io/badge/50%20Hz-256%20samples%20(5.12%20s)-8a5a2b">
<img alt="classes" src="https://img.shields.io/badge/classes-4-444444">
<img alt="data licence" src="https://img.shields.io/badge/data-CC%20BY%204.0-green">
<img alt="code licence" src="https://img.shields.io/badge/code-MIT-blue">

</div>

---

## 📖 Overview

CowTail-4 is a small, fully labelled field dataset of dairy-cattle behaviour recorded with a
tail-mounted nine-axis inertial sensor. It contains the **raw sensor streams**, the **annotation
layer** used for reference labels, and the **model-input windows** used in our experiments, so that a
published result can be traced back to a specific recording, label and window.

The dataset was collected on a commercial dairy farm and is released in anonymised form
(`DEVICE_NN` / `SESSION_NN`, relative milliseconds only). Video is **not** released, because it
contains identifiable farm staff and the farm's animal identifiers.

This dataset accompanies the manuscript *Reusing public collar models for cattle behaviour
recognition from tail acceleration with limited labels*.

## 📊 Dataset at a glance

| Item | Value |
|---|---|
| Animals | 4 dairy cows (cowA–cowD in the manuscript) |
| Devices | 2 tail-root IMU loggers, reused across animals |
| Recording sessions | 31 |
| Sensor channels | 9 axes: tri-axial acceleration, angular rate, magnetic field |
| Sampling | 50 Hz, native |
| Window | 256 samples = 5.12 s, non-overlapping |
| Labelled windows | 12,736 |
| Behaviour classes | standing · lying · walking · feeding |
| Ground truth | three independent annotators + veterinary graduate review |
| Video–sensor alignment | pinned to ≤ 0.1 s |

**Class definitions** used for the reference labels:

- **Standing** — upright on all four limbs; *includes* standing rumination. Rumination is not a separate class.
- **Lying** — sternal or lateral recumbency.
- **Walking** — forward locomotion; the class is scarce (198 windows) and its per-class metrics are unstable.
- **Feeding** — actual feed intake at the trough. Rumination, licking and isolated head movement are **not** feeding.

> **cowD has no feeding ground truth.** Any feeding metric computed for cowD is therefore meaningless
> and is excluded from the manuscript's four-state evaluation.

## 🗂 Repository layout

```
cowtail/
├── 01_data/                  master window table, per-cow arrays, normalisation scale
├── 02_results/              archived metrics and confusion counts
├── 03_models_predictions/            checkpoints and per-window predictions
├── 04_raw_sensors_labels/        raw nine-axis streams and reference intervals
├── 05_six_cows_full/         extended six-cow field set (training + test)
├── code/                     loader example and manifest builder
├── docs/                     data dictionary and external-source notes
├── FILE_MANIFEST.json        every file with size and SHA-256
├── FINAL_VERIFICATION.json   verification record for the released snapshot
├── LICENSE                   data licence (CC BY 4.0)
└── LICENSE-CODE              code licence (MIT)
```

Folder names are kept in Chinese to match the annotation software's on-disk convention
(COWMATA Pro). See [`docs/data_dictionary.md`](docs/data_dictionary.md) for every field.

## 🚀 Quick start

```python
import numpy as np
import pandas as pd

X = np.load("01_data/acceleration.npy")          # (12736, 3, 256), window-normalised
meta = pd.read_csv("01_data/windows.csv", encoding="utf-8-sig")

print(X.shape)                                   # (12736, 3, 256)
print(meta[["display_cow", "session_id", "original_state"]].head())
```

`acceleration.npy` is **normalised per window**; multiply a window by its stored `scale_g` to
recover approximate acceleration in *g*. Per-cow arrays sit alongside it as
`01_data/<cow>/acceleration.npy`. Raw nine-axis streams live in `04_raw_sensors_labels/` and join on
`session_id`.

The end-to-end example follows one labelled window back to its raw session and checks that both
paths describe the same motion:

```bash
python code/load_example.py            # first WALKING window
python code/load_example.py cowA 100   # cow, row_index
```

## 📝 Annotation protocol

- Three annotators labelled every recording independently.
- A veterinary graduate reviewer checked every label and resolved disagreements.
- When standing and feeding occurred simultaneously, the interval was annotated **twice** —
  standing first, then feeding — and both labels were retained.
- Video and sensor clocks were aligned once per session and pinned; residual alignment error is
  ≤ 0.1 s.
- The four-class evaluation uses one unambiguous frozen label per window; overlapping dual labels are
  kept in the annotation layer but are not scored.

## ⚠️ Intended use and limitations

- **Scope.** Four cows from a single farm; the four-cow set was recorded on one day. Do not treat
  these animals as a population sample.
- **Balance.** Walking is heavily under-represented (198 windows); per-class F1 for walking is
  unstable and should not be quoted without its support count.
- **Missing truth.** cowD has no feeding ground truth.
- **Anonymisation.** Absolute time, device MAC addresses, original cattle IDs, video filenames and
  annotator names are removed. Intervals cannot be re-linked to footage except by the data owner.
- **Not for welfare diagnosis.** Behaviour labels describe observable states, not health outcomes.

## 🔐 Integrity

`FILE_MANIFEST.json` lists every file with its size and SHA-256. Re-verify a download with:

```bash
python code/make_manifest.py      # regenerate
python verify_and_recompute.py              # read-only verification
```

## 📚 Citation

If you use this dataset, please cite the manuscript and the dataset record:

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

## ⚖️ Licence and ethics

- **Data:** CC BY 4.0 — see [`LICENSE`](LICENSE).
- **Code:** MIT — see [`LICENSE-CODE`](LICENSE-CODE).
- **Ethics:** procedures were non-invasive (external tail-root mounting only, no blood sampling).
  Written approval was provided by the Experimental Farm of Yangzhou University; the full committee
  reference is given in the companion manuscript.

## ✉️ Contact

- Xiangqing Zhang (corresponding author) — zxq1991@mail.nwpu.edu.cn
- Issues and questions: please open a GitHub issue.
