# Four-cow results

Display names are cowA, cowB, cowC and cowD. cowB reuses the same samples and the same predictions
as the earlier Cow C1 set; only the display name changed. This package involved no retraining,
no new inference, no label edits and no reselection of samples.

## Data and scoring conventions

| case | annotated intervals | windows | sessions | standing windows | lying windows | walking windows | feeding windows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cowA | 209 | 6342 | 17 | 2041 | 2034 | 97 | 2170 |
| cowC | 17 | 328 | 2 | 39 | 211 | 20 | 58 |
| cowD | 11 | 554 | 1 | 194 | 353 | 7 | 0 |
| cowB | 127 | 5512 | 11 | 1368 | 2060 | 74 | 2010 |

The four classes are fixed as standing, lying, walking and feeding. cowD has no feeding ground
truth; the fixed four-class macro-F1 counts that class's F1 as zero, and a separate three-class
macro-F1 over classes with ground truth is reported as well. The two must not be conflated. Every
window is weighted equally by default, and interval-equal-weight results are retained alongside.
The original recording sessions, original labels and grouping crosswalk are preserved in the data
and in the source-analysis snapshot.

## public_model_walking_direct

The table below gives results for ACT used directly, without any tail-supervised training labels,
averaged over three source seeds. AP is a ranking metric, not accuracy. The remaining JBB, UFRO and
pooled-source results, every repetition and the motion-intensity reference are all retained.

| case | walking AP % | precision % | recall % | walking F1 % | walking windows |
| --- | --- | --- | --- | --- | --- |
| cowA | 97.95 | 82.54 | 96.56 | 88.58 | 97 |
| cowB | 94.35 | 74.28 | 91.44 | 80.91 | 74 |
| cowC | 99.69 | 96.97 | 100.00 | 98.41 | 20 |
| cowD | 95.63 | 64.81 | 100.00 | 77.83 | 7 |

## Four classes: unified old-cow-identifier holdout protocol

The table below gives ACT means over three repetitions at the existing 1,000-step checkpoints. The
label percentages are the original training budgets, not new training performed for this package.
cowA, cowC and cowD use the original f01 test predictions; cowB uses the original f02 test
predictions. The four cases are therefore evaluated by different holdout models: this is not one
model holding out all four cows at once. Training and test are separated by the old data cow
identifier; renaming the groups neither changes the models nor the pooled performance.

| case | label budget % | accuracy % | fixed four-class macro-F1 % ± SD | macro-F1 over classes with truth % | four-class F1 vs scratch, pp |
| --- | --- | --- | --- | --- | --- |
| cowA | 5 | 64.69 | 57.62 ± 9.91 | 57.62 | -0.01 |
| cowA | 10 | 57.59 | 52.32 ± 7.67 | 52.32 | +2.79 |
| cowA | 25 | 54.73 | 48.78 ± 14.97 | 48.78 | +1.00 |
| cowA | 50 | 64.14 | 61.97 ± 16.23 | 61.97 | +9.64 |
| cowA | 100 | 57.77 | 51.51 ± 8.81 | 51.51 | -2.90 |
| cowB | 5 | 58.74 | 47.60 ± 7.69 | 47.60 | +5.68 |
| cowB | 10 | 60.51 | 53.48 ± 13.86 | 53.48 | +17.74 |
| cowB | 25 | 74.13 | 62.80 ± 7.60 | 62.80 | +3.06 |
| cowB | 50 | 75.93 | 63.62 ± 6.69 | 63.62 | -4.01 |
| cowB | 100 | 75.60 | 64.02 ± 4.82 | 64.02 | +1.37 |
| cowC | 5 | 81.30 | 62.47 ± 6.23 | 62.47 | -4.80 |
| cowC | 10 | 76.12 | 56.15 ± 6.89 | 56.15 | +0.51 |
| cowC | 25 | 78.25 | 57.66 ± 10.45 | 57.66 | +1.97 |
| cowC | 50 | 82.52 | 67.88 ± 22.36 | 67.88 | +14.57 |
| cowC | 100 | 78.86 | 59.29 ± 7.60 | 59.29 | -0.11 |
| cowD | 5 | 84.96 | 61.23 ± 16.35 | 81.64 | +7.80 |
| cowD | 10 | 94.65 | 63.30 ± 6.64 | 84.40 | +1.83 |
| cowD | 25 | 95.43 | 62.62 ± 7.38 | 83.50 | +1.14 |
| cowD | 50 | 85.62 | 60.88 ± 9.42 | 81.17 | -3.61 |
| cowD | 100 | 97.65 | 63.91 ± 4.67 | 85.22 | -1.71 |

The complete results cover the 5/10/25/50/100% budgets, all three starting points
(scratch/ACT/UFRO), three repetitions, and both window-equal and interval-equal weighting; per-class
precision, recall and F1, the confusion matrices and the paired positive/negative differences are
all in the CSVs in the same directory. SD is the sample standard deviation across three training
repetitions, not an error term across independent animals.

## cowA, cowC and cowD under the six_group_holdout_protocol

The previously quoted cowC figure of 63.83% → 68.09% comes from the original six-group per-group
holdout at 10% budget, 1,000 steps, replicate 0, window-equal weighting, scratch → ACT four-class
macro-F1; cowD's 64.13% → 79.50% is the three-class macro-F1 over classes with ground truth under
the same conditions. Both differ from the old-cow-identifier holdout protocol in the table above and
use a different training pool, so they must be interpreted separately. The original six-group
protocol still has the limitation that original record groups cross between training and test, so
it cannot be used to claim demonstrated generalisation to new, independent animals. All 2 budgets ×
3 methods × 2 endpoints for the original cowA, cowC and cowD results are retained, together with
their per-class and confusion results.

## Materials and verification

`02_results` holds the four-cow display tables and the complete `raw_analysis_snapshot`;
`03_models_predictions` holds the 90 old-cow-holdout models at 1,000 steps, both endpoints of the
18 original six-group models for cowA, cowC and cowD, the 24 raw predictions from direct public-model
recognition, and the 12 public-source checkpoints. The original snapshot and the COMPLETE receipts
are not rewritten; this package's manifest defines separately what was actually copied and verified.

Farm identity was supplied by the user's field records; animal identity was not independently
verified here. Source groups and original identifiers are retained in the data archive; adding
display names does not increase the number of independent animals. All scores and sources are
retained as they are; no failure condition was deleted on the basis of its result.
The final results for the 18 six-group units are bound to the attempts designated by COMPLETE, with
18 files at 1,000 steps and 18 at 3,000 steps. The final attempt for `f05_s0_b100_scratch` is
`attempt_002`; the interrupted `attempt_001` is retained only as a raw record.
