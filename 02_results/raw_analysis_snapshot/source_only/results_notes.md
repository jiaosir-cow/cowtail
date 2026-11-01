# All retained groups: a re-check of public_model_walking_direct

All 28,911 annotated windows are used only for local testing. The model is a frozen model trained on
existing public data; there is no new training, no tail-label learning and no threshold tuning here.
This is a walking / non-walking binary task and must not be read as a four-class
standing/lying/walking/feeding score.

B1, C1, F1 and G1 are analysis groups formed by re-cutting the original B/C/F data along complete
sessions. Their windows and predictions are unchanged, so the pooled scores are exactly the same as
before the re-cut. Regrouping cannot create an improvement in overall accuracy.
The old experiments had already looked at these test labels; this is a retrospective re-grouping of
existing test results, not a fresh independent test set.

AP measures the ranking quality of walking segments, not accuracy. Precision, recall and F1 use the
original model's fixed rule of margin ≥ 0. The table below averages three random seeds of the public
model.

The earlier ACT figure of 93.03% was an equal-weight mean of the per-group AP under the original six
groups; the 89.79% for all retained windows below computes AP over all windows pooled. The two are
different statistics, not a degradation of the model. Regrouping changes the weights of the
equal-weight mean, so it cannot be used to claim a performance improvement.

| group | windows | walking windows | ACT walking AP | ACT precision | ACT recall | ACT walking F1 | motion-RMS AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| cowA | 6342 | 97 | 97.95% | 82.54% | 96.56% | 88.58% | 96.35% |
| cowC | 328 | 20 | 99.69% | 96.97% | 100.00% | 98.41% | 100.00% |
| cowD | 554 | 7 | 95.63% | 64.81% | 100.00% | 77.83% | 67.39% |
| COW_OTHER | 6114 | 69 | 79.21% | 46.99% | 91.30% | 61.38% | 25.56% |
| cowB | 5512 | 74 | 94.35% | 74.28% | 91.44% | 80.91% | 93.05% |
| COW_OTHER | 5216 | 69 | 85.72% | 66.21% | 82.61% | 72.57% | 69.88% |
| COW_OTHER | 4845 | 46 | 84.65% | 35.44% | 88.41% | 49.75% | 70.41% |
| All retained | 28911 | 382 | 89.79% | 60.89% | 91.36% | 72.29% | 62.88% |
| Pooled ALL_POOL (old and new identical) | 21687 | 258 | 85.84% | 53.47% | 88.50% | 65.77% | 48.82% |

Every seed of the remaining JBB, UFRO and MULTI sources, the original B/C/F comparisons and the
four-class ground-truth counts are all kept in the CSVs. No sample, source or negative result was
deleted on the basis of how a group scored.

All four new groups have ground truth for all four classes; that only means four-class scoring is
possible, not that this binary model already has four-class capability.
