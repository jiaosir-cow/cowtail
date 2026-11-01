# Existing four-class models: direct scoring on the new grouping

This round only reads existing 1,000-step four-class predictions; nothing was trained, no model
inference was invoked and no GPU was used. There are 90 predictions in total:
2 original holdout folds × 5 label budgets × 3 model starting points × 3 repetitions.

**Scoring convention:** the four classes are standing, lying, walking and feeding, and the highest
probability is the final class. Every table below is the mean over three repetitions; standard
deviations are in the detailed tables. Accuracy is the share of windows answered correctly; macro-F1
gives every class equal weight. Neither is the earlier walking AP.

**Test independence:** A, D and E use models from the original f01 fold; B1, C1, F1 and G1 use models
from the original f02 fold. For every group the training indices of its model have no overlap with
that group's test windows, annotation events or original holdout cow identifiers. These are the
original per-fold models, so they must not be described as a single model that never saw all seven
groups. Pooled queue figures aggregate the out-of-fold predictions of the two holdout folds.

**Current recommendation: keep the original version as the main version, and retain the new
grouping as an analysis annex.** C1 has the highest four-class macro-F1 among the four new groups at
all five label budgets and can serve as a relatively good case; F1 is middling, while B1 and G1 are
weaker overall. ACT beats scratch on all four groups at 10% labels but falls below scratch at 50%
labels, so the new version cannot be written up as generally high four-class performance or as
transfer always working.

**Group boundaries:** B1/C1/F1/G1 are four analysis groups formed from the current ALL_POOL pool along
whole sessions and add no independent animals; the split is an exploratory analysis carried out after
seeing historical results. Keeping or rolling back the grouping changes only the display grouping;
the mean of a selected group must not be presented as a full-queue score.

## Data scale and class coverage

| group | intervals | windows | standing | lying | walking | feeding | sessions |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cowA | 209 | 6342 | 2041 | 2034 | 97 | 2170 | 17 |
| cowC | 17 | 328 | 39 | 211 | 20 | 58 | 2 |
| cowD | 11 | 554 | 194 | 353 | 7 | 0 | 1 |
| COW_OTHER | 163 | 6114 | 2342 | 2989 | 69 | 714 | 11 |
| cowB | 127 | 5512 | 1368 | 2060 | 74 | 2010 | 11 |
| COW_OTHER | 124 | 5216 | 1726 | 2531 | 69 | 890 | 10 |
| COW_OTHER | 130 | 4845 | 1220 | 1964 | 46 | 1615 | 10 |

E has no feeding ground truth. Its fixed four-class macro-F1 counts the missing feeding class as
zero, and a three-class macro-F1 over classes with ground truth is reported as well; the three-class
value must not be presented as a complete four-class result.

## New four groups: ACT four-class macro-F1 across all label budgets (%)

| group | 5% labels | 10% labels | 25% labels | 50% labels | 100% labels |
| --- | --- | --- | --- | --- | --- |
| COW_OTHER | 35.31 ± 19.90 | 47.66 ± 7.23 | 51.58 ± 11.32 | 44.15 ± 5.96 | 44.87 ± 3.04 |
| cowB | 47.60 ± 7.69 | 53.48 ± 13.86 | 62.80 ± 7.60 | 63.62 ± 6.69 | 64.02 ± 4.82 |
| COW_OTHER | 42.70 ± 3.80 | 51.55 ± 7.61 | 50.36 ± 2.50 | 46.94 ± 2.92 | 48.55 ± 2.34 |
| COW_OTHER | 38.37 ± 10.35 | 45.16 ± 6.49 | 49.27 ± 7.86 | 44.79 ± 3.03 | 51.11 ± 3.08 |

## New four groups: ACT accuracy (%)

| group | 5% labels | 10% labels | 25% labels | 50% labels | 100% labels |
| --- | --- | --- | --- | --- | --- |
| COW_OTHER | 45.09 ± 30.86 | 60.23 ± 12.33 | 66.61 ± 13.87 | 57.90 ± 7.97 | 57.98 ± 4.61 |
| cowB | 58.74 ± 8.66 | 60.51 ± 7.23 | 74.13 ± 9.22 | 75.93 ± 6.75 | 75.60 ± 3.04 |
| COW_OTHER | 50.35 ± 17.68 | 64.30 ± 1.28 | 60.86 ± 1.35 | 54.70 ± 2.29 | 56.36 ± 1.95 |
| COW_OTHER | 48.60 ± 14.41 | 55.54 ± 0.84 | 63.16 ± 9.22 | 55.58 ± 3.30 | 62.67 ± 7.51 |

## Is transfer better than training from scratch: ACT minus scratch, four-class macro-F1 (pp)

| group | 5% labels | 10% labels | 25% labels | 50% labels | 100% labels |
| --- | --- | --- | --- | --- | --- |
| COW_OTHER | -1.19 | +12.66 | +6.05 | -0.60 | -0.04 |
| cowB | +5.68 | +17.74 | +3.06 | -4.01 | +1.37 |
| COW_OTHER | +4.26 | +13.11 | -1.15 | -2.19 | -1.55 |
| COW_OTHER | +1.68 | +12.04 | +3.74 | -4.36 | +1.38 |

Every condition is shown here, positive and negative differences alike. The label budget is the
share of training events in the original experiments; it is not a claim that this share of labels was
applied to the test groups here.

## All retained groups: 10% label condition (each group's own original holdout fold)

| group | scratch four-class F1 | ACT four-class F1 | UFRO four-class F1 | ACT accuracy | ACT F1 over supported classes |
| --- | --- | --- | --- | --- | --- |
| cowA | 49.53 ± 8.86 | 52.32 ± 7.67 | 39.29 ± 14.09 | 57.59 ± 9.69 | 52.32 ± 7.67 |
| cowC | 55.63 ± 7.29 | 56.15 ± 6.89 | 62.82 ± 17.31 | 76.12 ± 5.42 | 56.15 ± 6.89 |
| cowD | 61.47 ± 4.16 | 63.30 ± 6.64 | 39.28 ± 12.65 | 94.65 ± 4.70 | 84.40 ± 8.86 |
| COW_OTHER | 35.01 ± 11.06 | 47.66 ± 7.23 | 44.35 ± 5.08 | 60.23 ± 12.33 | 47.66 ± 7.23 |
| cowB | 35.74 ± 3.50 | 53.48 ± 13.86 | 43.47 ± 20.55 | 60.51 ± 7.23 | 53.48 ± 13.86 |
| COW_OTHER | 38.44 ± 7.94 | 51.55 ± 7.61 | 46.84 ± 9.76 | 64.30 ± 1.28 | 51.55 ± 7.61 |
| COW_OTHER | 33.12 ± 3.90 | 45.16 ± 6.49 | 42.22 ± 13.36 | 55.54 ± 0.84 | 45.16 ± 6.49 |

## All retained groups: 100% label condition (each group's own original holdout fold)

| group | scratch four-class F1 | ACT four-class F1 | UFRO four-class F1 | ACT accuracy | ACT F1 over supported classes |
| --- | --- | --- | --- | --- | --- |
| cowA | 54.42 ± 7.35 | 51.51 ± 8.81 | 57.64 ± 5.09 | 57.77 ± 10.73 | 51.51 ± 8.81 |
| cowC | 59.40 ± 7.69 | 59.29 ± 7.60 | 63.84 ± 0.49 | 78.86 ± 5.72 | 59.29 ± 7.60 |
| cowD | 65.62 ± 2.09 | 63.91 ± 4.67 | 68.02 ± 5.30 | 97.65 ± 1.54 | 85.22 ± 6.23 |
| COW_OTHER | 44.91 ± 6.25 | 44.87 ± 3.04 | 43.32 ± 6.02 | 57.98 ± 4.61 | 44.87 ± 3.04 |
| cowB | 62.65 ± 3.89 | 64.02 ± 4.82 | 65.90 ± 3.15 | 75.60 ± 3.04 | 64.02 ± 4.82 |
| COW_OTHER | 50.10 ± 1.60 | 48.55 ± 2.34 | 54.54 ± 4.66 | 56.36 ± 1.95 | 48.55 ± 2.34 |
| COW_OTHER | 49.73 ± 3.06 | 51.11 ± 3.08 | 52.32 ± 0.97 | 62.67 ± 7.51 | 51.11 ± 3.08 |

## Original B/C/F under the same models: ACT four-class macro-F1 (%)

| original group | 5% labels | 10% labels | 25% labels | 50% labels | 100% labels |
| --- | --- | --- | --- | --- | --- |
| COW_OTHER | 40.62 ± 8.33 | 49.61 ± 4.89 | 53.89 ± 1.80 | 50.78 ± 2.12 | 52.47 ± 2.69 |
| COW_OTHER | 40.16 ± 10.20 | 43.05 ± 8.70 | 45.68 ± 10.89 | 41.04 ± 5.74 | 49.09 ± 3.41 |
| COW_OTHER | 33.17 ± 14.59 | 13.84 ± 6.15 | 31.80 ± 21.11 | 28.32 ± 16.05 | 32.50 ± 14.32 |

Original F has no lying ground truth, so its fixed four-class F1 cannot be judged against
four-class-covering new groups from a single number. These still use the original f02-fold models and
must not be conflated with the later independent six-cow retraining experiments.

## ACT per-class performance: 10% and 100% labels

### 10% labels: per-class F1 (%)

| group | standing | lying | walking | feeding |
| --- | --- | --- | --- | --- |
| cowA | 55.05 ± 12.25 | 83.40 ± 17.32 | 63.14 ± 13.46 | 7.71 ± 13.36 |
| cowC | 27.35 ± 26.36 | 95.16 ± 4.19 | 96.79 ± 1.34 | 5.29 ± 9.16 |
| cowD | 91.34 ± 7.87 | 98.57 ± 1.52 | 63.30 ± 17.97 | no truth |
| COW_OTHER | 77.33 ± 14.75 | 69.43 ± 6.13 | 13.67 ± 7.04 | 30.23 ± 16.64 |
| cowB | 59.28 ± 2.10 | 84.89 ± 0.02 | 35.58 ± 29.49 | 34.16 ± 27.79 |
| COW_OTHER | 71.78 ± 6.14 | 78.33 ± 0.78 | 26.51 ± 20.53 | 29.59 ± 19.80 |
| COW_OTHER | 52.01 ± 5.68 | 85.44 ± 3.62 | 28.19 ± 12.70 | 15.02 ± 22.77 |

### 100% labels: per-class F1 (%)

| group | standing | lying | walking | feeding |
| --- | --- | --- | --- | --- |
| cowA | 54.39 ± 15.99 | 87.68 ± 17.65 | 63.99 ± 5.51 | 0.00 ± 0.00 |
| cowC | 42.09 ± 26.26 | 97.50 ± 4.12 | 97.56 ± 0.00 | 0.00 ± 0.00 |
| cowD | 96.59 ± 2.22 | 99.62 ± 0.53 | 59.43 ± 16.29 | no truth |
| COW_OTHER | 31.60 ± 19.58 | 90.54 ± 5.29 | 29.23 ± 8.88 | 28.13 ± 4.49 |
| cowB | 46.61 ± 15.95 | 94.79 ± 8.12 | 40.36 ± 17.28 | 74.33 ± 7.32 |
| COW_OTHER | 29.76 ± 18.72 | 76.60 ± 13.49 | 35.59 ± 13.55 | 52.27 ± 5.62 |
| COW_OTHER | 26.00 ± 15.70 | 88.00 ± 8.92 | 31.36 ± 1.12 | 59.10 ± 12.54 |

## Does the new grouping actually raise the overall score

Checked window by window: old B+C+F and new B1+C1+F1+G1 both contain 21,687 windows and 544
intervals. Across the 45 method/budget/repetition conditions and both weightings the windows, ground
truth and predictions are identical, and the accuracy, macro-F1 and MCC differences are all zero.
The grouping only exposes differences between segments; it does not raise the pooled score.

## Stability across all conditions (description only, no best-budget selection)

| group | ACT mean F1 over five budgets | range of budget means | times best among the five budgets | budgets where ACT beats scratch | repetitions where ACT beats scratch |
| --- | --- | --- | --- | --- | --- |
| COW_OTHER | 44.72 | 35.31–51.58 | 0/5 | 2/5 | 9/15 |
| cowB | 58.30 | 47.60–64.02 | 5/5 | 4/5 | 10/15 |
| COW_OTHER | 48.02 | 42.70–51.55 | 0/5 | 2/5 | 6/15 |
| COW_OTHER | 45.74 | 38.37–51.11 | 0/5 | 4/5 | 10/15 |

This table spans different label budgets and only helps judge stability; it is not a performance
estimate of a new model. The between-repetition standard deviation reflects variation over the three
original experimental repetitions, not a confidence interval over the cow population.

The concrete failures also need to be visible: at 10% labels the third repetition of G1 had a feeding
recall of 0, and the same repetition of C1 and F1 had feeding recalls of only 2.04% and 4.94%. So
even though C1 is the relatively best group, class-level recognition remains unstable under low-label
conditions, and the means alone do not justify calling this reliable four-class recognition.

## Files and reproduction

- `metrics_mean_std.csv`: means and sample standard deviations (ddof=1) for all old and new groups,
  five budgets, three methods and both weightings.
- `class_metrics_mean_std.csv`: per-class precision, recall, F1 and support;
  `confusion_by_replicate.csv` holds the per-repetition confusion matrices.
- `paired_deltas_mean_std.csv`: paired transfer-minus-scratch differences within the same fold,
  budget and repetition.
- `new_old_pool_invariance.csv`: consistency of the old and new pooled scores across all 90 weighted
  conditions.
- `AUDIT.json` and `verified_prediction_manifest.csv`: hashes, mappings and leakage checks for this
  round's inputs and retained predictions.
- Running `python score_existing_models.py` rebuilds this directory's report and tables from the
  original predictions; it does not import torch.

**Historical-integrity limitation:** the original 3,000-step endpoint was cleaned up as requested and
the historical whole-run completion audit is no longer valid. This report verifies only the 1,000-step
predictions still retained and the metadata/indices they use; it does not claim that every historical
file of the original training package is still intact.
