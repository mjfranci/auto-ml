# reference/stopping-criteria.md

Operational detail behind Phase 2 rules. Checked per checkpoint, top to bottom, first match wins.

## Checkpoint cadence

- Define a "checkpoint" as: every epoch (NN), every 50 boosting rounds (GBT), or every 10% of budgeted iterations -- whichever the framework exposes.
- All Phase 2 rules count in checkpoints, not wall-clock time.

## Kill signals (per run)

| # | Signal | Threshold | Action |
|---|---|---|---|
| K1 | Val loss rising | 3 consecutive checkpoints | Kill, keep best checkpoint |
| K2 | Train–val gap | > 15% relative and widening for 2 checkpoints | Kill, tag "needs regularization" |
| K3 | Successive halving | Bottom 50% of live cohort at 50% of run budget | Kill |
| K4 | NaN/Inf loss | Any occurrence | Kill immediately |
| K5 | No improvement | < 0.1% improvement over 5 checkpoints, not in top 3 | Kill |

## Continue/extend signals (per run)

| # | Signal | Action |
|---|---|---|
| C1 | Top-3 AND improving ≥ 0.5%/checkpoint | Continue; eligible for one 50% budget extension |
| C2 | Budget exhausted, curve descending | Projection rule (below). One extension max. |

## Projection rule (C2 detail)

1. Take the last 5 checkpoint values of the primary metric
2. Fit a linear slope (simple least squares is sufficient; this is a heuristic, not a forecast)
3. Projected gain = slope × (number of checkpoints in a doubled budget)
4. Gain ≥ 1% relative → extend once. Gain < 1% → mark converged, stop.

## Sweep-level suspension triggers

| # | Signal | Action |
|---|---|---|
| S1 | Val metric > train metric beyond noise | Suspend. Audit the split (wrong direction of generalization = split bug or duplicate rows across folds). |
| S2 | Near-perfect train + mediocre val at checkpoint 1 | Suspend. Run leakage audit (below). |
| S3 | 3+ runs NaN across different seeds | Suspend. Architecture or preprocessing bug -- sweeping over a broken pipeline burns budget. |
| S4 | All runs below trivial baseline at 25% of total budget | Suspend. Return to Phase 1 redesign. |

## Leakage audit procedure (S2)

Run in order; stop at first confirmed cause:

1. **Split integrity:** any identical or near-duplicate rows across train/val? (hash rows; >0.5% overlap = split bug)
2. **Group bleed:** any entity ID (patient, user, SKU) present in both train and val? If yes → re-split with group k-fold.
3. **Target proxies:** for each feature, compute correlation/mutual information with the target. Any single feature explaining > 80% of target variance → inspect whether it's recorded *after* the outcome (post-outcome features are leaks). Drop and document.
4. **Temporal bleed:** if data is temporal, confirm every val timestamp > every train timestamp.
5. If steps 1–4 all pass, the high score may be real → resume sweep, note "leakage audit passed" in the log.

## Budget accounting

- Global budget set by researcher = hard ceiling (identity.md).
- Extensions (C1/C2) draw from the 10% reserve first, then from budget freed by kills.
- When global budget hits 100%: stop everything, proceed to Phase 3 with current best checkpoints. The operator never asks for more compute mid-sweep; it reports what more compute would buy in the verdict's next-action line.
