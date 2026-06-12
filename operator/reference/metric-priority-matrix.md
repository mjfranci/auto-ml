# reference/metric-priority-matrix.md

Quick-lookup matrix used by Rule 1.3 and Rule 3.1. The primary metric drives all sweep and promotion decisions; secondaries are reported but never decide.

## Classification

| Imbalance ratio | Primary | Secondary | Disqualified |
|---|---|---|---|
| ≤ 3:1 | Accuracy or F1 (honor researcher) | the other one | -- |
| 3:1 – 10:1 | F1 (macro) | Accuracy, ROC-AUC | -- |
| > 10:1 | PR-AUC | Recall @ 90% precision, F1 | Accuracy (headline) |
| Multiclass, any class < 5% support | F1 (macro) | Per-class recall table | Plain accuracy |

## Regression

| Target shape | Primary | Secondary | Disqualified |
|---|---|---|---|
| Roughly symmetric, single scale | RMSE | MAE | -- |
| Spans ≥ 2 orders of magnitude | MAE on log target | MAPE | Raw MSE/RMSE |
| Heavy outliers, can't log (zeros/negatives) | MAE | Huber loss | MSE |
| Forecasting (temporal) | Same as above, computed on the time-based holdout only | -- | Any metric on a shuffled split |

## Eval-set sizing rule (used by Rule 3.1)

Minimum eval samples for a trustworthy promotion, by the margin you need to detect:

| Margin to detect (relative) | Min eval samples (per class, classification) |
|---|---|
| ≥ 5% | 200 |
| 2–5% | 500 |
| 1–2% | 2,000 |
| < 1% | 5,000+ -- operator reports "margin below resolvable threshold" and treats candidates as tied |

If the available eval set is below the row matching the observed margin, the margin is **not real** for decision purposes → apply the tie rules, not the winner rules.

## Bootstrap CI procedure (small-eval caveat, Rule 3.2)

1. Resample the eval set with replacement, 1,000 times
2. Compute the primary metric on each resample
3. Report the 2.5th–97.5th percentile interval
4. If the interval of the winner overlaps the point estimate of the runner-up → downgrade to tie, run the tiebreak chain
