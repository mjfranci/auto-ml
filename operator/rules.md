# rules.md -- Decision Logic

This file is the operator's brain. Every decision in every phase must trace back to a rule here. **Rule 0 governs how the operator moves through the pipeline; it overrides any reading of a later rule that would have the operator pause for permission.** If a situation isn't covered, follow the **Default Escalation Rule** at the bottom -- never improvise silently.

---

## Rule 0: Autonomy mandate (non-negotiable)

Stage 01 intake is the only researcher-facing gate in the pipeline. Once the problem brief is written and the operator has the information it needs, it MUST immediately start Stage 02. It then runs Stage 02 → Stage 03 → Stage 04 → Stage 05 to a fully trained, promoted model without pausing for permission, approval, confirmation, or sign-off at any stage boundary. (When the Stage 04 verdict is KILLED or BLOCKED, no model is promoted, Stage 05 does not run, and the verdict is the terminal output.) Advancing to the next stage is not a decision the operator surfaces -- it is the default and only behavior.

The operator may interrupt the autonomous run for exactly three reasons, none of which is a request for permission to proceed:

1. A **HARD STOP / HARD BLOCK** fires (§2.0, §2.1, §3.1). The operator suspends, records the cause to the dashboard and sweep log, and follows the prescribed recovery. It does not ask whether to suspend.
2. An **environmental dependency fails** and cannot be restored autonomously (e.g., expired Colab login, §2.0). The operator names the specific action needed to restore execution, then resumes. It does not ask whether to continue.
3. A situation **matches no rule** in this file AND the two interpretations differ by >25% of budget (Default Escalation Rule). The operator presents its recommendation and the deciding evidence, not an open question.

Any other pause -- "ready for Stage 03?", "shall I proceed to promotion?", "does the plan look right?" -- is a rule violation. Presenting an artifact for visibility -- showing the experiment plan, streaming sweep checkpoints to the dashboard -- is not a pause: the operator displays it and proceeds in the same step. Researcher input is collected once, at Stage 01; the next time the operator addresses the researcher is the final trained model -- or the verdict, when it kills or blocks -- unless one of the three interrupts above fires.

---

## Phase 1: Experiment Design

**Input required:** problem brief (target variable, business goal), dataset profile (row count, feature count, feature types, class balance, temporal flag, missing-data %).

**If any required input is missing:** infer it from the data if possible (e.g., compute class balance yourself). Only escalate if the target variable itself is undefined.

### 1.1 Task framing

| Condition | Decision |
|---|---|
| Target is continuous | Frame as regression |
| Target is categorical, 2 classes | Binary classification |
| Target is categorical, 3+ classes | Multiclass; check per-class support (see 1.3) |
| Target has <2 unique values | **HARD STOP** -- degenerate target. Report to researcher with the value counts. |
| Researcher's stated framing conflicts with the data (e.g., asks for classification on a continuous target) | Override the researcher's framing, state the override and the reason in the experiment plan. Do not ask -- decide and flag. |

### 1.2 Model family selection

| Condition | Mandatory inclusions |
|---|---|
| <1,000 rows | Linear/logistic baseline + regularized linear (ridge/lasso). NO deep learning -- flag as "insufficient data for NN" if researcher requested it. |
| 1,000–100,000 rows, tabular | Gradient-boosted trees (primary), linear baseline, random forest |
| >100,000 rows, tabular | GBT primary; NN permitted as one candidate, never the only one |
| Text features dominant | Add: TF-IDF + linear as baseline; transformer fine-tune only if >10,000 labeled rows |
| Image data | CNN/ViT transfer learning; never train from scratch under 50,000 images |
| Researcher names a specific model | Include it, but never as the only candidate. Always add the rule-mandated baseline. |

**Baseline rule (non-negotiable):** every experiment plan includes (a) a majority-class / mean predictor and (b) one simple linear model. A sweep without baselines is invalid -- the operator cannot judge "good" without "trivial."

### 1.3 Metric selection

| Condition | Primary metric | Disqualified metrics |
|---|---|---|
| Class imbalance ≤ 3:1 | Accuracy or F1 (researcher's choice honored) | -- |
| Class imbalance 3:1 to 10:1 | F1 (macro) | Accuracy demoted to secondary |
| Class imbalance > 10:1 | PR-AUC, report recall at fixed precision | Accuracy **disqualified** -- do not report it as a headline number even if asked |
| Regression, target spans orders of magnitude | MAE on log-target or MAPE | Raw MSE (dominated by outliers) |
| Regression, normal-ish target | RMSE primary, MAE secondary | -- |
| Researcher's stated metric conflicts with the table above | Optimize the rule-mandated metric. Report the researcher's metric as secondary. State the override prominently in the plan header. |

**The override rule is the point:** an operator that silently optimizes the wrong metric because the researcher asked for it has failed. Decide correctly, report the disagreement.

### 1.4 Validation strategy

| Condition | Strategy |
|---|---|
| Temporal column present OR rows are time-ordered | Time-based split (train on past, validate on future). Random k-fold **disqualified**. |
| Grouped data (multiple rows per patient/user/site) | Group k-fold on the entity ID. Random split disqualified -- it leaks. |
| <1,000 rows | 5-fold CV minimum; single holdout disqualified |
| 1,000–50,000 rows | 5-fold CV or 80/20 holdout with fixed seed |
| >50,000 rows | Single holdout acceptable; CV optional |
| No entity ID column but duplicate-looking rows detected (>1% near-duplicates) | Flag possible grouping, deduplicate before split, note in plan |

### 1.5 Phase 1 output contract

The experiment plan must contain, in order: task framing, any overrides of researcher requests (with reasons), model candidates (with the mandated baselines), primary + secondary metrics, validation strategy, and compute budget allocation per candidate. Then proceed directly to Phase 2 per Rule 0 -- no approval, no pause -- **unless** a HARD STOP was triggered.

---

## Phase 2: Sweep Triage

Run per-checkpoint and per-run. Rules are evaluated top to bottom; **first match wins.**

### 2.0 Pre-flight integrity (evaluated before any triage rule)

| Signal | Decision |
|---|---|
| Loaded data contradicts the problem brief on any field that drives a Phase 1 gate (row-count band, imbalance band, temporal/group flag) | Trust the data, not the brief. Recompute the full profile from the loaded dataset, correct the brief in place with a logged correction, and re-run Phase 1 design on the corrected profile before writing any training code. |
| Target column missing from the loaded data, or unparseable | **HARD STOP.** The target is the one field the operator cannot infer. Report the loaded column list to the researcher. |
| Colab mode: output scraped from the notebook fails schema validation, lacks a run ID, or carries a run ID that does not match the cell just executed | Do not triage. Stale or partial telemetry is not evidence. Re-run the cell via `ColabRunner`, re-scrape, and log the rejected read in the sweep log. |
| Colab mode: browser session disconnected, runtime recycled, or Google login expired | Reconnect headlessly with the saved profile and re-run the interrupted cell. If the login itself has expired, pause the loop, ask the researcher to repeat the one-time headed login (`python library/colab_runner.py --login`), then resume. Log every interruption in the sweep log. |

### 2.1 Hard stops (override everything)

| Signal | Decision |
|---|---|
| Validation metric > training metric by a margin that exceeds noise (e.g., val accuracy 5+ points above train) | **SUSPEND SWEEP.** Likely leakage or split bug. Diagnose split before any further compute. |
| Train metric near-perfect (>99%) while val metric is mediocre, on the first epoch | **SUSPEND SWEEP.** Probable target leakage in features. Run leakage audit (see reference/stopping-criteria.md). |
| Loss = NaN or Inf | Kill run. If 3+ runs NaN with different seeds → flag architecture/preprocessing bug, suspend sweep. |
| All runs below the trivial baseline after 25% of budget | **SUSPEND SWEEP.** Return to Phase 1 redesign. Spending the remaining 75% is the wrong call by rule, not by judgment. |

**Recurrence rule:** every hard stop prescribes a recovery -- a pipeline reset or a Phase 1 redesign. Before recovering, check the previous sweep log (`stages/03-execution/output/sweep-log.md`) for a prior occurrence of the same signal on the same data source. If the same hard-stop signal fires a second consecutive time, do not recover again: suspend the pipeline, present both occurrences side by side on the dashboard, and require researcher action. If the repeated signal is "all runs below baseline," emit the Phase 3 verdict `KILLED -- signal not found` (Rule 3.1) instead of a third design. A recovery that reproduces its own failure is a loop, not a recovery.

### 2.2 Per-run triage (when no hard stop applies)

| Signal | Decision |
|---|---|
| Val loss increased for 3 consecutive checkpoints | Kill run (overfit). Keep best checkpoint. |
| Train–val gap > 15% (relative) and growing | Kill run, log "regularization insufficient," seed the narrowed sweep with stronger regularization |
| Run in bottom 50% of cohort at 50% of its budget | Kill (successive-halving rule) |
| Run in top 3 AND val metric still improving ≥ 0.5% per checkpoint | Continue; extend budget by up to 50% if global budget allows |
| Run in top 3 AND improvement < 0.5% per checkpoint for 3 checkpoints | Stop run, mark converged, keep checkpoint |
| Learning curve still clearly descending at budget limit | Apply projection: fit slope on last 5 checkpoints. If projected gain over a doubled budget < 1% → stop, mark converged. If ≥ 1% → extend once (one extension max per run). |

### 2.3 Sweep-level decisions

| Signal | Decision |
|---|---|
| Top-3 runs cluster in one hyperparameter region | Narrow the search space to that region ±1 step; relaunch remaining budget there |
| Top runs sit at the edge of the search range (e.g., best LR is the minimum tested) | Extend the range one step beyond the edge before narrowing -- the optimum may be outside the box |
| Compute budget 100% consumed | Stop sweep. Proceed to Phase 3 with whatever exists. Never silently overspend. |

---

## Phase 3: Promotion

Evaluated top to bottom; first match wins.

### 3.1 Blocks (cannot promote)

| Condition | Decision |
|---|---|
| Candidate trained on a split later flagged as leaky | **HARD BLOCK.** Requires re-train on clean split. No researcher override accepted on this one -- leaky numbers are not numbers. |
| Eval set < 200 samples | **No promotion.** Report best candidate with "statistically unreliable" status; prescribe minimum additional eval data needed (see reference/metric-priority-matrix.md for the sizing rule). |
| Best candidate beats trivial baseline by < 1% (relative) | Kill all candidates. Verdict: "signal not found at current data/feature quality." Prescribe next step (more data, better features) -- do not promote a model that learned nothing. |

### 3.2 Promotion logic

| Condition | Decision |
|---|---|
| Clear winner: beats #2 by > 1% (relative) on primary metric AND beats baseline by ≥ 2% | **PROMOTE.** Full verdict report. |
| Beats baseline by 1–2% | Promote with **"marginal gain"** flag: state the operational cost of the model vs. the simple baseline and recommend the baseline if serving cost matters. The operator still picks one -- it picks, then caveats. |
| Top-2 within 0.5% (relative) on primary metric | Tiebreak in strict order: (1) inference latency -- faster wins if ≥ 2x faster; (2) model size -- smaller wins if ≥ 5x smaller; (3) simplicity class (linear > trees > NN); (4) secondary metric. The chain always terminates; ties don't escape to the researcher. |
| Eval set 200–500 samples | Promote winner, attach confidence interval (bootstrap, 1,000 resamples) and "small-eval caveat." Flag for researcher review but the decision stands unless overturned. |
| Winner's per-class recall is < 50% on any class with ≥ 5% support | Promote with **"class blind-spot"** flag naming the class; recommend targeted data collection for it. |

### 3.3 Verdict report contract

Every Phase 3 output ends with exactly one of: `PROMOTED`, `PROMOTED-WITH-FLAGS`, `KILLED`, `BLOCKED -- <reason>`. Followed by: the one-line decision, the rule(s) that produced it (by section number), the evidence, and the single prescribed next action.

---

## Default Escalation Rule

Escalation is a decision, not a punt. The operator escalates only when:
1. A HARD STOP / HARD BLOCK rule fires, **or**
2. The situation matches no rule in this file **and** both available interpretations lead to materially different compute spend (>25% of budget).

**Advancing between stages is never one of these conditions.** A completed stage is not an uncovered situation -- it is the trigger to start the next one (Rule 0). The operator never escalates merely to confirm it may proceed.

When escalating, the operator must present: the situation, the two candidate decisions, its own recommendation, and what evidence would settle it. "What do you want to do?" alone is forbidden output.
