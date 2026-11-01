# 05_six_cows_full — full six-cow dataset (training pool + test pool)

This is the **complete set** of field data used in the manuscript: 6 cows, 28,911 windows,
62 recording sessions and 2 tail-mounted nine-axis collars. `01_data`–`04_raw_sensors_labels` is a
test subset of it; the crosswalk is at the end of this file.

## Contents

```
acceleration.npy           (28911, 3, 256) float32  per-window normalised tri-axial acceleration
source_window_indices.npy  int64 (28911,)           upstream window indices
windows.csv                28,911 rows, one row per window
events.csv                 781 annotated intervals
test_truth.csv             window-level ground truth
class_support.csv          window count per cow x split x class
cow_crosswalk.csv          name crosswalk against the four-cow package in `01_data`
cowA/ … cowF/labels.csv + window_indices.npy
```

The window definition matches `01_data` exactly: 50 Hz, 256 samples, 5.12 s, normalised per window
by the median magnitude (multiply by `scale_g` to recover approximate g). Field meanings are in
`../docs/data_dictionary.md`.

## Training / test split

The `split` column comes from the manuscript's fixed record buckets, assigned by the old recording
identifier rather than adjusted to results:

| split | record bucket | cows | windows | sessions |
|---|---|---|---:|---:|
| `train` | 20201-3 | cowA 6,342 / cowD 328 / cowE 554 | 7,224 | 20 |
| `test` | 21074-1 | cowB 19,841 / cowC 1,563 / cowF 283 | 21,687 | 42 |

The `analysis_slice` column preserves the seven analysis slices reported in the manuscript: the
training pool is `A`, `D` and `E`; the test pool is cut by whole session into `B1` (6,114),
`C1` (5,512), `F1` (5,216) and `G1` (4,845). **These four slices are analysis groups inside one
record bucket, not four additional independent animals.** Do not treat them as repeated samples
from four cows for statistical inference, and do not interpret differences between slices as
individual differences.

## Per cow

| cow | split | windows | sessions | STANDING | LYING | WALKING | FEEDING |
|---|---|---:|---:|---:|---:|---:|---:|
| cowA | train | 6,342 | 17 | 2,041 | 2,034 | 97 | 2,170 |
| cowB | test | 19,841 | 38 | 6,412 | 9,006 | 246 | 4,177 |
| cowC | test | 1,563 | 3 | 222 | 538 | 10 | 793 |
| cowD | train | 328 | 2 | 39 | 211 | 20 | 58 |
| cowE | train | 554 | 1 | 194 | 353 | 7 | 0 |
| cowF | test | 283 | 1 | 22 | 0 | 2 | 259 |

`cowE` has no feeding ground truth and `cowF` has no lying ground truth; do not use them as
complete four-class validation sets.

## Relationship to the four-cow package (01–04)

The 12,736 windows in `01_data` are a strict subset of this set, but the names differ:

| four-cow package | six-cow package | windows in package | total windows for that cow | note |
|---|---|---:|---:|---|
| cowA | cowA | 6,342 | 6,342 | same animal, same windows |
| cowB | cowB | 5,512 | 19,841 | only the 11-session slice of cowB (the old C1 group), not the whole animal |
| cowC | cowD | 328 | 328 | display name changed only |
| cowD | cowE | 554 | 554 | display name changed only |

**State which naming scheme you used when citing**, otherwise `cowB`/`cowC`/`cowD` will point to
different animals.

## Raw nine-axis coverage

`04_raw_sensors_labels` contains only the 31 sessions of the four-cow package (12,736 windows, 44%
of the full set). The remaining 31 sessions have only the three-axis derived windows in this
directory; their raw nine-axis streams are not released with this repository.
