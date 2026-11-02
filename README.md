<div align="center">
  <img src="assets/brand/cowmata-logo.svg" width="300" alt="COWMATA">

# CowTail-4

**A fully labelled 9-axis IMU behaviour dataset recorded at the cattle tail root.**

[English](README.md) · [简体中文](README.zh-CN.md)

<img alt="data licence" src="https://img.shields.io/badge/data-CC%20BY%204.0-green">
<img alt="code licence" src="https://img.shields.io/badge/code-MIT-blue">

</div>

CowTail-4 is the four-cow tail-root dataset released with the manuscript *Reusing public collar models for cattle behaviour recognition from tail acceleration with limited labels*. It contains the raw sensor streams, the annotation layer, and the model-input windows, so a published result can be traced to a recording, a label, and a window.

## Start in 60 seconds

From the repository root, with the packages in [`requirements.txt`](requirements.txt) (`numpy`, `pandas`):

```bash
pip install -r requirements.txt
python code/load_example.py            # first WALKING window
python code/load_example.py cowA 100   # cow, row_index
```

The same arrays can be read directly:

```python
import numpy as np
import pandas as pd

X = np.load("01_data/acceleration.npy")          # (12736, 3, 256), window-normalised
meta = pd.read_csv("01_data/windows.csv", encoding="utf-8-sig")

print(X.shape)                                   # (12736, 3, 256)
print(meta[["display_cow", "session_id", "original_state"]].head())
row = meta.iloc[0]
approx_g = X[int(row.row_index)] * float(row.scale_g)
```

`acceleration.npy` is normalised per window. `row_index` in `01_data/windows.csv` is the row of that array; multiply by the same row's `scale_g` to recover approximate acceleration in *g*. The package array is **not** stored in cowA–cowD order: cowA (rows 0–6341), cowC (6342–6669), cowD (6670–7223), cowB (7224–12735). Per-cow arrays are `01_data/<cow>/acceleration.npy`. Raw nine-axis streams are in `04_raw_sensors_labels/` and join on `session_id`.

## Dataset

| Item | Value |
|---|---|
| Animals | 4 dairy cows (cowA–cowD in the manuscript) |
| Devices | 2 tail-root IMU loggers, reused across animals |
| Recording sessions | 31, one day, one farm |
| Sensor channels | 9 axes: tri-axial acceleration, angular rate, magnetic field |
| Sampling | 50 Hz, native |
| Window | 256 samples = 5.12 s, non-overlapping |
| Labelled windows | 12,736 |
| Behaviour classes | standing · lying · walking · feeding |
| Ground truth | three independent annotators + veterinary graduate review |
| Video–sensor alignment | pinned to ≤ 0.1 s |

Class definitions used for the reference labels:

- **Standing** — upright on all four limbs; includes standing rumination. Rumination is not a separate class.
- **Lying** — sternal or lateral recumbency.
- **Walking** — forward locomotion. This class has 198 windows, and its per-class metrics are unstable.
- **Feeding** — actual feed intake at the trough. Rumination, licking, and isolated head movement are not feeding.

cowD has no feeding ground truth (`FEEDING` count is 0 in `01_data/four_cow_label_and_window_counts.csv`). A feeding metric for cowD is not interpretable and is excluded from the manuscript's four-state evaluation.

Video is not released. The public files use `DEVICE_NN` / `SESSION_NN` and relative milliseconds only, because the recordings show farm staff and the farm's animal identifiers.

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

The same record is in [`CITATION.cff`](CITATION.cff).

## Ethics and limitations

- **Licence.** Data: CC BY 4.0 ([`LICENSE`](LICENSE)). Code: MIT ([`LICENSE-CODE`](LICENSE-CODE)).
- **Ethics.** Approved by the Ethics Committee of the Experimental Farm of Yangzhou University. Procedures were non-invasive: external tail-root mounting only, no blood sampling.
- **Scope.** Four cows from a single farm, recorded on one day. These animals are not a population sample.
- **Balance.** Walking has 198 windows. Do not quote a walking F1 without that support count.
- **Missing truth.** cowD has no feeding ground truth.
- **Anonymisation.** Absolute time, device MAC addresses, original cattle IDs, video filenames, and annotator names are removed. Intervals cannot be re-linked to footage except by the data owner.
- **Not a welfare diagnosis.** Labels describe observable states, not health outcomes.

## Repository layout

```
cowtail/
├── 01_data/                  master window table, per-cow arrays, normalisation scale
├── 02_results/               archived metrics and confusion counts
├── 03_models_predictions/    checkpoints and per-window predictions
├── 04_raw_sensors_labels/    raw nine-axis streams and reference intervals
├── 05_six_cows_full/         separate field package (see its own README)
├── code/                     loader example and manifest builder
├── docs/                     data dictionary and external-source notes
├── FILE_MANIFEST.json        every file with size and SHA-256
├── FINAL_VERIFICATION.json   verification record for the released snapshot
├── LICENSE                   data licence (CC BY 4.0)
└── LICENSE-CODE              code licence (MIT)
```

`05_six_cows_full/README.md` states that package's own counts and split notes. Do not treat those figures as the four-cow table above. The same display names are not the same animals: four-cow cowB is one slice of six-cow cowB, four-cow cowC is six-cow cowD, and four-cow cowD is six-cow cowE. Use the crosswalk table in that README before joining the two packages. Field definitions are in [`docs/data_dictionary.md`](docs/data_dictionary.md).

## Annotation protocol

- Three annotators labelled every recording independently.
- A veterinary graduate reviewer checked every label and resolved disagreements.
- When standing and feeding occurred together, the interval was annotated twice — standing first, then feeding — and both labels were retained.
- Video and sensor clocks were aligned once per session and pinned; residual alignment error is ≤ 0.1 s.
- The four-class evaluation uses one unambiguous frozen label per window. Overlapping dual labels stay in the annotation layer and are not scored.

## Integrity

`FILE_MANIFEST.json` lists every file with its size and SHA-256.

```bash
python code/make_manifest.py      # regenerate
python verify_and_recompute.py    # read-only verification
```

## Contact

Xiangqing Zhang (corresponding author) — zxq1991@mail.nwpu.edu.cn

Questions: open a GitHub issue.
