# cowD

This directory holds the sensor windows, labels, all corresponding parent annotations and the source
index used for this case.

Eligible intervals: 11; windows: 554; parent annotations: 15.

`row_index` indexes `acceleration.npy` in this directory; `package_row_index` indexes the merged
array one level up. The original label content and source cow identifiers are unchanged.

Identity statements and the source crosswalk are in `CASE.json`; results are in `02_results` at the
repository root.

This case has no feeding ground truth, so it cannot serve as a complete four-class validation set.
