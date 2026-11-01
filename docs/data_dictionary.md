# Data Dictionary — Four-Cow Behaviour Dataset (v1.0)

This document defines every file, column, unit and binary layout in this release.
If a value here disagrees with a file, the file is wrong — please open an issue.

- Animals: 4 cows (`cowA`–`cowD`), 2 tail-mounted 9-axis collars (`DEVICE_01`, `DEVICE_02`)
- Collection: 31 sessions on a single day, 50 Hz, 5.12 s non-overlapping windows
- Labelled windows: 12,736 across four behaviours
- Repository layout version: `schema = 1`

---

## 1. Directory layout

```
01_data/                          package-level tables and model input
  acceleration.npy                (12736, 3, 256) float32
  windows.csv                     12,736 rows, 1 row per window
  events.csv                      468 annotated intervals
  canonical_annotations.csv       468 annotation records (full annotation layer)
  session_identity_mapping.csv    31 sessions -> device + cow
  four_cow_label_and_window_counts.csv           per-cow summary counts
  DATA_MANIFEST.json              shapes, counts, class support
  cowA/  cowB/  cowC/  cowD/
    acceleration.npy              (N_cow, 3, 256) float32
    windows.csv                   N_cow rows
    events.csv                    annotated intervals for this cow
    all_annotations.csv           annotation records for this cow
    test_truth.csv                window-level truth used by the reported models
    body_annotations_outside_evaluation.csv
    package_row_indices.npy       int64 (N_cow,) row -> package acceleration.npy
    source_dataset_row_indices.npy int64 (N_cow,)

02_results/                      reported metrics (per class, per replicate, confusion)
03_models_predictions/                    archived predictions and per-unit metrics
04_raw_sensors_labels/                raw sensor streams and the original label files
  <cow>/sensor_sessions/<device>/SESSION_NN.json
  <cow>/source_labels/source_dataset/<device>/SESSION_NN.events.csv
  <cow>/source_labels/source_dataset/<device>/SESSION_NN.events_meta.json
```

Per cow: `cowA` 6,342 windows / 17 sessions, `cowB` 5,512 / 11,
`cowC` 328 / 2, `cowD` 554 / 1.

**Package array order is not alphabetical.** `01_data/acceleration.npy` stores the
cows in this order: `cowA` (rows 0–6341), `cowC` (6342–6669), `cowD` (6670–7223),
`cowB` (7224–12735). Always index it through `package_row_index`, never by assuming
`cowA, cowB, cowC, cowD`.

---

## 2. Label scheme

`target4` is the integer label and `original_state` its name. Order is fixed.

| target4 | original_state | meaning | windows |
|---:|---|---|---:|
| 0 | `STANDING` | standing, not walking, not feeding | 3,642 |
| 1 | `LYING` | lying down | 4,658 |
| 2 | `WALKING` | walking | 198 |
| 3 | `FEEDING` | feeding / eating | 4,238 |

`cowD` has no `FEEDING` windows; do not treat it as a four-class validation animal.
`WALKING` is strongly under-represented (1.6% of windows); report per-class scores,
not accuracy alone.

Windows inherit the label of the annotation interval that fully contains them
(`included_in_current_window_evaluation = True`). Intervals that are not fully
contained are kept in `body_annotations_outside_evaluation.csv` and are not scored.

---

## 3. `windows.csv`

One row per window, same order as the matching `acceleration.npy`.

| column | type | meaning |
|---|---|---|
| `row_index` | int | in a **per-cow** file: 0-based row in that cow's `acceleration.npy`; in the **package** file: row in `01_data/acceleration.npy` (equals `package_row_index`) |
| `package_row_index` | int | row in `01_data/acceleration.npy` — use this to look up package data |
| `source_row_index` | int | provenance index upstream; treat as opaque |
| `continuous_row_index` | int | provenance index in the continuous stream; opaque |
| `window_id` | str | SHA-256 identifier, stable across releases |
| `device_id` | str | `DEVICE_01` / `DEVICE_02` |
| `session_id` | str | `SESSION_01` … `SESSION_31`, matches `04` file names |
| `segment_id` | int | contiguous run of samples with no gap > 0.5 s (global counter) |
| `start_index` | int | internal index in the upstream continuous stream — **not** usable to slice the raw files (see §5.3) |
| `stop_index` | int | `start_index + 256` |
| `t_start_rel_ms` | int | window start, **ms since the start of that session**; use this to locate raw frames |
| `t_end_rel_ms` | int | window end; `t_end_rel_ms - t_start_rel_ms = 5120` |
| `quality_fraction` | float | fraction (0–1) of the 256 samples carrying a valid timing flag |
| `scale_g` | float | median acceleration magnitude of this window, in g |
| `motion_rms_g` | float | RMS of (acc − window axis-mean), in g; a motion-intensity proxy |
| `schema` | int | windowing contract version (= 1) |
| `event_id` | str | annotation interval this window belongs to |
| `target4` | int | label, see §2 |
| `original_state` | str | label name |
| `analysis_group` | str | equals `display_cow`; kept for compatibility |
| `display_cow` | str | `cowA` … `cowD` |

Notes:

- `t_start_rel_ms` is derived from the device clock and is the reliable key for
  locating raw samples. `start_index` comes from an upstream continuous stream that
  carries a small lead-in offset (observed ≈ 16–18 frames, i.e. 320–360 ms) and must
  not be used to index the raw arrays directly.
- Absolute wall-clock time is **not** released. Times are relative to each session.

---

## 4. `acceleration.npy` — read this before using the numbers

Shape `(N, 3, 256)`, `float32`, channel order `acc_x, acc_y, acc_z`, 50 Hz.

**Values are per-window scale-normalised, not calibrated g.**
Each window was divided by its own `scale_g` (median magnitude, stored in `windows.csv`):

```python
x_g = acceleration.npy[i] * windows.scale_g[i]     # back to approximate g
```

This normalisation was part of the original model input contract. Do not compare
raw amplitudes across windows without undoing it, and do not report `acceleration.npy`
values as physical acceleration.

The package file `01_data/acceleration.npy` is the concatenation of the four
per-cow arrays; use `package_row_indices.npy` to map back.

---

## 5. Raw 9-axis sensor files (`04_raw_sensors_labels`)

One JSON per session: `{"imu": ..., "motion": ..., "temperature": ...}`.
All three values are Base64 strings; the decoded bytes are **little-endian int16**.

### 5.1 `imu` — 11 int16 per frame, 50 Hz

| int16 offset | content |
|---:|---|
| 0–1 | `elapsed_ms` as uint32 (low word, high word) — ms since session start |
| 2–4 | accelerometer x, y, z (raw LSB) |
| 5–7 | gyroscope x, y, z (raw LSB) |
| 8–10 | magnetometer x, y, z (raw LSB) |

Unit conversion (LSB → physical):

```python
acc_g    = raw_acc   / 4096.0   # g
gyro_dps = raw_gyro  / 32.0     # deg/s
mag_gauss= raw_mag   / 1000.0   # gauss (native unit, not calibrated)
```

Reference decoder:

```python
import base64, json
import numpy as np

def load_session(path):
    obj = json.load(open(path, encoding="utf-8"))
    raw = np.frombuffer(base64.b64decode(obj["imu"]), dtype="<i2").reshape(-1, 11)
    u16 = raw.view("<u2")                       # same bytes, unsigned view
    elapsed_ms = u16[:, 0].astype(np.int64) | (u16[:, 1].astype(np.int64) << 16)
    acc  = raw[:, 2:5].astype(np.float32)  / 4096.0
    gyro = raw[:, 5:8].astype(np.float32)  / 32.0
    mag  = raw[:, 8:11].astype(np.float32) / 1000.0
    return elapsed_ms, acc, gyro, mag
```

### 5.2 `motion` and `temperature`

60 int16 each, one value per minute of the session.

- `temperature`: device-reported body temperature, unit **0.01 °C** (e.g. `3898` = 38.98 °C).
- `motion`: device-side activity counter, uncalibrated, 0–435 observed.
  Treat as an auxiliary signal, not a validated activity measure.

### 5.3 Linking windows to raw samples

Locate the raw frames by **time**, not by `start_index`:

```python
f = np.searchsorted(elapsed_ms, rec.t_start_rel_ms)   # first frame >= window start
seg = acc[f : f + 256]                                 # coarse slice
# closer agreement: resample that span onto 256 uniform points
grid = np.linspace(rec.t_start_rel_ms, rec.t_start_rel_ms + 5119, 256)
seg = np.stack([np.interp(grid, elapsed_ms, acc[:, c]) for c in range(3)], axis=1)
```

Accuracy of the join: the released model array is a **derived representation**
(resampled to a uniform 256-point grid, then scale-normalised), while the raw file
keeps the device's irregular frame timing (6–258 ms, median 20 ms). On a 186-window
sample, per-axis correlation between the time-resampled raw slice and the
denormalised model array has median 0.94 and 10th percentile 0.88. Expect close but
not bit-exact agreement; treat the raw file as ground truth and the `.npy` as the
model-ready view.

`session_id` in `windows.csv` equals the JSON file stem. This join is complete:
every session referenced by the tables has exactly one raw file.

---

## 6. `events.csv` and annotations

| column | meaning |
|---|---|
| `event_id` | annotation interval identifier (SHA-256) |
| `t_start_rel_ms`, `t_end_rel_ms` | interval bounds, ms since session start |
| `original_duration_ms` | `t_end_rel_ms - t_start_rel_ms` |
| `window_count` | number of evaluation windows fully inside the interval |
| `target4`, `original_state` | label, see §2 |
| `display_cow`, `analysis_group` | animal identifier |

`all_annotations.csv` / `canonical_annotations.csv` retain the annotation layer
(`layer`, `label`, `code`, `evidence`, `type`, `adjudication`, temperature statistics
per interval, clipping and gap diagnostics). Annotator names and file paths are removed
in this release.

---

## 7. Provenance and anonymisation

The following were replaced by stable anonymous identifiers, and the replacement is
consistent everywhere in the release:

| original | released as |
|---|---|
| collar MAC address | `DEVICE_01`, `DEVICE_02` |
| session timestamp | `SESSION_01` … `SESSION_31` (chronological) |
| registered animal number | removed; animals appear only as `cowA`–`cowD` |
| video file name | removed (videos are not part of this release) |
| annotator name, server address, absolute paths, internal dataset names | removed |

Timing is preserved only in relative form. `imu`, `motion` and `temperature`
payloads were verified byte-identical to the acquisition files.

`window_id` and `event_id` are content hashes carried over from the original
processing pipeline. They are stable identifiers, not reversible, but they were
computed from the pre-anonymisation record. If you need a release whose identifiers
cannot be linked to any earlier archive, recompute them with a salt (§8, item 6).

---

## 8. Known limitations

1. n = 4 animals, one collection day, one farm. Do not read the reported scores as
   cross-farm generalisation.
2. `acceleration.npy` is 3-axis and scale-normalised; the raw files are 9-axis.
   A claim about 9-axis models is not supported by these arrays.
3. `cowD` has no `FEEDING`; `cowC` has 328 windows. Per-cow variance is large.
4. `WALKING` covers 198 windows; its per-class metrics are unstable.
5. Sensor mounting was on the tail; behaviour semantics are not transferable to
   collar-mounted or ear-tag sensors without re-validation.
6. `window_id` / `event_id` predate anonymisation. Salted re-hashing is available on
   request and is recommended if the release must not be linkable to internal
   archives.

---

## 9. Six-cow dataset (`05_six_cows_full`)

The complete field collection, 28,911 windows over 6 cows and 62 sessions. Same
window contract as §4 (50 Hz, 256 samples, 5.12 s, per-window `scale_g`
normalisation). Extra columns compared with §3:

| column | meaning |
|---|---|
| `split` | `train` (record bucket 20201-3, 7,224 windows) or `test` (bucket 21074-1, 21,687 windows) |
| `analysis_slice` | reporting group: `A`, `D`, `E` in training; `B1`, `C1`, `F1`, `G1` inside the test bucket |
| `cow_id` | `cowA` … `cowF` |

The four test slices are session-contiguous groupings of **one record bucket**, not
four additional animals: `cowB` alone contributes 19,841 windows to that bucket, and
`B1`/`C1`/`F1`/`G1` partition it. Do not treat slice differences as individual
differences, and do not use slices as independent replicates.

Naming differs from the four-cow subset in §1: four-cow `cowB` is the 11-session
slice `C1` of six-cow `cowB`; four-cow `cowC` and `cowD` are six-cow `cowD` and
`cowE`. See `05_six_cows_full/cow_crosswalk.csv`.

Raw 9-axis coverage: only the 31 sessions shared with the four-cow subset (12,736 of
28,911 windows, 44%). The remaining 31 sessions exist only as 3-axis derived windows.
