# Additional training cattle

These six animals are the manuscript's additional training data.
They are not the reported test cattle cowA–cowD, and they are not the cowA–cowF labels in `05_six_cows_full`.
The manuscript does not name them. This release calls them train1–train6, ordered by window count from largest to smallest.

Both reported evaluations can draw training examples from these animals.
M1 tests cowA, cowC and cowD. M2 tests cowB.
Each reported test animal is left out of the training data of the model that evaluates it.

| Item | Value |
|---|---|
| Animals | 6, training only |
| Intervals | 101 |
| Windows | 879 |
| Array | `acceleration.npy`, shape (879, 3, 256), float32 |
| Window | 256 samples, 5.12 s, 50 Hz, scaled by that window's median magnitude |
| Raw nine-axis | not released for these animals |
| Loggers | `DEVICE_01` and `DEVICE_T1`–`DEVICE_T5`. Only train6 uses `DEVICE_01`. The T labels are not numbered in train1–train5 order |

Array order is train1, train2, train3, train4, train5, train6.
`row_index` is the row of `acceleration.npy` in this directory.
Multiply a row by the same row's `scale_g` to recover approximate acceleration in g.
Session ids such as `train1_S01` belong to this directory only. They are not `SESSION_01`–`SESSION_31` of the reported test cattle.

| Animal | Windows | Intervals | Sessions | Standing | Lying | Walking | Feeding | Rows |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| train1 | 620 | 48 | 13 | 184 | 103 | 166 | 167 | 0–619 |
| train2 | 153 | 12 | 6 | 126 | 0 | 27 | 0 | 620–772 |
| train3 | 56 | 19 | 9 | 0 | 0 | 56 | 0 | 773–828 |
| train4 | 27 | 9 | 4 | 0 | 0 | 27 | 0 | 829–855 |
| train5 | 14 | 7 | 5 | 0 | 0 | 14 | 0 | 856–869 |
| train6 | 9 | 6 | 3 | 0 | 0 | 9 | 0 | 870–878 |

train2 has standing and walking only. train3, train4, train5 and train6 have walking windows only. train1 has all four classes.
Original cattle numbers, device MAC addresses, absolute clock time and video are not in these files.
