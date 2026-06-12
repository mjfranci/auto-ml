# reference/model-selection-rubric.md

Used by Rule 1.2 (candidate selection) and Rule 3.2 (tiebreak step 3, simplicity class).

## Simplicity classes (for tiebreaks)

Lower class number wins ties. Within a class, smaller parameter count wins.

| Class | Models |
|---|---|
| 1 | Trivial baselines (mean/majority/seasonal-naive), linear & logistic regression, ridge, lasso |
| 2 | Decision tree, random forest, gradient-boosted trees, k-NN, naive Bayes |
| 3 | Shallow NN (≤3 layers), fine-tuned classical embeddings + linear head |
| 4 | Deep NN, LSTM/transformer, fine-tuned pretrained models |

## Data-size gates

| Rows | Max permitted class |
|---|---|
| < 1,000 | Class 1–2 only |
| 1,000 – 10,000 | Class 1–3 |
| 10,000 – 100,000 | Class 1–4 (class 4 gets ≤ 30% of budget) |
| > 100,000 | All; budget split at operator discretion within Phase 1 plan |

If the researcher requests a model above the gate for their data size: include the rule-permitted alternative as primary, the requested model gets at most 20% of budget as a "researcher request" candidate, and the plan states the gate that was applied.

## Mandatory baseline definitions

- **Classification trivial baseline:** majority-class predictor. Imbalanced data: also report the "predict-all-minority" recall for context.
- **Regression trivial baseline:** mean (or median if MAE is primary).
- **Forecasting trivial baseline:** seasonal-naive (value from one season ago); if no seasonality, last-value carry-forward.
- **Simple-model baseline:** logistic/linear regression with standard scaling, no feature engineering beyond one-hot encoding.

## Feature-type adjustments

| Dominant feature type | Adjustment |
|---|---|
| High-cardinality categoricals (>100 levels) | Target/ordinal encoding for tree models; one-hot disqualified above 100 levels |
| Text | Add TF-IDF + linear as a class-1-adjacent baseline before any transformer |
| Heavy missingness (>30% in any used feature) | GBT with native missing handling preferred; imputation strategy must be fit on train folds only |
| Mixed image + tabular | Late-fusion only; never train the image tower from scratch under 50k images |
