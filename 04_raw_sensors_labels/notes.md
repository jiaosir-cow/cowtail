# Raw sensor and label assets

This directory holds the complete raw sensor sessions, raw label CSVs and their `events_meta` files
for the four display cases cowA, cowB, cowC and cowD. The cow identifiers, device numbers, times and
annotation content inside the source files are unchanged.

- Complete sensor sessions: 31, stored by display case and original device number.
- Raw label CSVs: 38; corresponding metadata files: 38.
- Canonical annotations for these sessions: 468, each checked line by line against the row index,
  behaviour code, annotation layer, session and start/end times in the source CSVs, all consistent.
  The canonical annotation table is stored in the data part of the package; the original CSVs may
  also contain annotation rows that the canonical table did not select.
- SHA-256 values for the 107 source files match before copying, after copying and in the target.
  See `raw_asset_index.csv`, `canonical_source_row_verification.csv` and `RAW_ASSETS_VERIFIED.json`.
- Historical paths were recovered through a fixed-drive migration and a directory move from `new`
  to `labels`; no arbitrary version substitution was performed.
- The IMU payload in the raw JSON is an encoded string; this package archives the original bytes only
  and makes no separate claim of completed axis decoding or hardware calibration verification.
- Video, runtime environments, caches and derived event slices were not copied; the complete sensor
  sessions and their raw labels are all included.
