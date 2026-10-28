# Four-Cow Tail-Mounted IMU Behaviour Dataset

A small, fully labelled field dataset of dairy-cow behaviour from tail-mounted
9-axis IMU collars, released with the raw sensor streams, the annotation layer,
and the model-ready window arrays used in our experiments.

- 4 cows, 2 collars, 31 sessions, one collection day
- 50 Hz, 9-axis (accelerometer + gyroscope + magnetometer)
- 12,736 labelled windows of 5.12 s (256 samples), 4 behaviour classes
- Raw streams, window tables, per-window labels, and reported metrics included

**Start here:** [`docs/data_dictionary.md`](docs/data_dictionary.md) defines every
file, column, unit and binary layout. [`code/load_example.py`](code/load_example.py)
loads a labelled window and the matching raw samples end to end.

---

## Two datasets in one repository

| | `01_数据`–`04_原始传感与标签` | `05_six_cows_full` |
|---|---|---|
| Scope | the held-out four-cow test subset | **the complete field data, training + test** |
| Animals | 4 | 6 |
| Windows | 12,736 | 28,911 |
| Sessions | 31 | 62 |
| Raw 9-axis | yes, all 31 sessions | only the 31 sessions shared with the subset |
| Labels | 4 classes | 4 classes + `split` + `analysis_slice` |

The four-cow folder is a strict subset of `05_six_cows_full`, but the names differ:
the four-cow `cowB` is only an 11-session slice of six-cow `cowB`, and four-cow
`cowC`/`cowD` are six-cow `cowD`/`cowE`. Always state which naming you use —
see [`05_six_cows_full/cow_crosswalk.csv`](05_six_cows_full/cow_crosswalk.csv).

## Six-cow dataset (`05_six_cows_full`)

| Cow | Split | Windows | Sessions | STANDING | LYING | WALKING | FEEDING |
|---|---|---:|---:|---:|---:|---:|---:|
| `cowA` | train | 6,342 | 17 | 2,041 | 2,034 | 97 | 2,170 |
| `cowB` | test | 19,841 | 38 | 6,412 | 9,006 | 246 | 4,177 |
| `cowC` | test | 1,563 | 3 | 222 | 538 | 10 | 793 |
| `cowD` | train | 328 | 2 | 39 | 211 | 20 | 58 |
| `cowE` | train | 554 | 1 | 194 | 353 | 7 | 0 |
| `cowF` | test | 283 | 1 | 22 | 0 | 2 | 259 |
| **Total** | | **28,911** | **62** | 8,930 | 12,142 | 382 | 7,457 |

`split` comes from the paper's fixed record buckets (`20201-3` for training,
`21074-1` for testing) and was not tuned on results. `analysis_slice` preserves the
seven reporting slices: `A`, `D`, `E` in training, and `B1`, `C1`, `F1`, `G1` inside
the test bucket. Those four test slices are analytical groupings of one record
bucket, **not four additional animals** — do not treat slice differences as
individual differences.

`cowE` has no `FEEDING` and `cowF` has no `LYING`; neither is a complete four-class
validation animal.

## Four-cow test subset (`01_数据`–`04`)

| Cow | Sessions | Windows | STANDING | LYING | WALKING | FEEDING |
|---|---:|---:|---:|---:|---:|---:|
| `cowA` | 17 | 6,342 | 2,041 | 2,034 | 97 | 2,170 |
| `cowB` | 11 | 5,512 | 1,368 | 2,060 | 74 | 2,010 |
| `cowC` | 2 | 328 | 39 | 211 | 20 | 58 |
| `cowD` | 1 | 554 | 194 | 353 | 7 | 0 |
| **Total** | **31** | **12,736** | 3,642 | 4,658 | 198 | 4,238 |

468 annotated intervals back these windows. `cowD` has no `FEEDING`; `WALKING`
covers only 198 windows and its per-class scores are unstable.

```
01_数据/                        four-cow test subset: tables, labels, arrays
02_实验结果/                    reported metrics, per class and per replicate
03_模型与预测/                  archived predictions and unit-level metrics
04_原始传感与标签/              raw 9-axis JSON + original label files
05_six_cows_full/               complete six-cow field data (train + test)
docs/                           data dictionary, label scheme, limitations
code/                           loader, manifest generator, verification
```

Three derived training-pool arrays under `03_模型与预测/…/inputs/acceleration.npy`
(85–120 MB each, 28,911–40,920 windows) are not included, to respect GitHub's
100 MB per-file limit. They are not needed for the four-cow data or for the
reported metrics; ask the authors if you need them for exact recomputation.

## Quick start

```bash
pip install numpy pandas
python code/load_example.py            # first WALKING window of cowA
python code/load_example.py cowB 100   # cow, row_index inside that cow's table
```

The loader reads a window from `acceleration.npy`, finds the matching raw session
by `session_id`, and slices the 9-axis stream by time. Expect per-axis correlation
around 0.94: the `.npy` array is a resampled, scale-normalised view of the raw file,
not a bit-exact copy.

## Labels

| id | name | meaning |
|---:|---|---|
| 0 | `STANDING` | standing |
| 1 | `LYING` | lying down |
| 2 | `WALKING` | walking |
| 3 | `FEEDING` | feeding / eating |

## Two representations, one dataset

- **Raw** (`04_原始传感与标签`): one JSON per session, 11 little-endian int16 per
  frame (uint32 `elapsed_ms` + acc/gyro/mag ×3), 50 Hz. Units: acc/4096 g,
  gyro/32 dps, mag/1000 gauss. This is the acquisition record.
- **Model-ready** (`01_数据/acceleration.npy`): `(N, 3, 256)` float32, acceleration
  only, divided by each window's `scale_g`. Multiply by `scale_g` to return to
  approximate g. Do not report these values as physical acceleration.

Absolute wall-clock time, collar MAC addresses, animal registry numbers, video file
names and annotator identities are **not** part of this release. Times are relative
to each session; devices appear as `DEVICE_01`/`DEVICE_02`, sessions as
`SESSION_01`–`SESSION_31`.

## Intended use and limits

Intended for behaviour-classification research on tail-mounted IMU signals,
annotation-efficiency studies, and transfer experiments against other cattle
datasets. Not intended as a benchmark of 9-axis model performance (the released
model arrays are 3-axis), nor as evidence of cross-farm generalisation: n = 4
animals, one farm, one day. See [`docs/data_dictionary.md`](docs/data_dictionary.md) §8.

Videos are not released. Intervals can be re-linked to original footage only by the
data owner.

## Integrity

`FILE_MANIFEST.json` lists every file with size and SHA-256. Verify with:

```bash
python code/make_manifest.py     # regenerate
python 核验与复算.py              # read-only verification of this folder
```

## Ethics

Data were collected on a commercial farm under the owner's consent. Animal handling
followed [[ETHICS_APPROVAL]] — *fill in the approval body and reference number
before publication*.

## License

Data: CC BY 4.0. Code: MIT. See `LICENSE` and `LICENSE-CODE` — *confirm before
publishing*.

## Citation

```bibtex
@dataset{[[CITATION_KEY]],
  title  = {Four-Cow Tail-Mounted IMU Behaviour Dataset},
  author = {[[AUTHORS]]},
  year   = {2026},
  note   = {[[AFFILIATION]]},
  url    = {[[REPOSITORY_URL]]}
}
```

Contact: [[CONTACT]].

---

## 中文说明

本目录是一份四头奶牛尾戴九轴惯性传感器行为数据集，包含原始传感流、标注层和模型
输入窗口。规模为 4 头牛、2 个项圈、31 个采集会话、12,736 个 5.12 秒窗口（256 点，
50 Hz），行为类别为站、卧、走、采食四类。

- 逐字段定义见 `docs/data_dictionary.md`；快速上手见 `code/load_example.py`。
- `01_数据/acceleration.npy` 是**按窗归一化**后的三轴加速度，乘回 `scale_g` 才是近似 g；
  原始九轴数据在 `04_原始传感与标签`，按 `session_id` 与窗口表对应。
- 发布版不含绝对时间、设备 MAC、原始牛号、视频文件名和标注人姓名；设备与会话
  统一改为 `DEVICE_NN` / `SESSION_NN`，时间只保留相对毫秒。
- 已知限制：仅 4 头牛、单牧场单日；`cowD` 无采食真值；行走类仅 198 窗，指标不稳定。

<!-- 发布前需补齐：ETHICS_APPROVAL、AUTHORS、AFFILIATION、CITATION_KEY、
     REPOSITORY_URL、CONTACT，并确认 CC BY 4.0 / MIT 许可选择。 -->
