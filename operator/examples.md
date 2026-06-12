# examples.md -- Decisions in Action

Six full walkthroughs. Example 1 is a clean path. Examples 2 and 3 are edge cases in the training results. Examples 4 through 6 are edge cases outside the code results entirely: a brief the data contradicts, a recovery mechanism that loops, and a telemetry channel with a browser in it.

---

## Example 1 -- Clean path: churn prediction (tabular, moderate imbalance)

**Researcher input:**
> "Predict customer churn. 42,000 rows, 31 features (mixed numeric/categorical), churn rate 18% (≈4.5:1 imbalance). No timestamps. Budget: 40 GPU-hours. I'd like to optimize accuracy."

**Phase 1 decisions:**
- Task: binary classification (Rule 1.1)
- **Override:** imbalance is 4.5:1 → Rule 1.3 mandates macro-F1 as primary; accuracy demoted to secondary. *Stated in plan header: "Researcher requested accuracy; F1 (macro) selected per Rule 1.3 (imbalance 3:1–10:1). Accuracy reported as secondary."*
- Models: GBT (primary), logistic regression, random forest + mandated trivial baseline (Rule 1.2)
- Validation: 5-fold CV, fixed seed (Rule 1.4, 1k–50k rows)
- Budget: 60% GBT sweep, 20% RF, 10% logistic, 10% reserve

**Phase 2 (mid-sweep excerpt):**
- Run gbt-07: val loss rose 3 consecutive checkpoints → **killed**, best checkpoint kept (Rule 2.2)
- Runs gbt-03/04/09 cluster at depth 6–8, LR 0.05–0.1 → search space narrowed to that region (Rule 2.3)
- Run gbt-12 in top 3, improving 0.7%/checkpoint → budget extended 50% (Rule 2.2)

**Phase 3 verdict:**

```
PROMOTED: gbt-12 (depth=7, lr=0.07, n_estimators=480)
Rule: 3.2 clear winner -- F1 0.713 vs #2 (rf-04) 0.681 (+4.7% rel), baseline 0.305
Evidence: 5-fold CV, std 0.008 across folds; accuracy (secondary) 0.871
Next action: none required. Model card attached.
```

---

## Example 2 -- Edge case: the leakage trap (medical, grouped data)

**Researcher input:**
> "Classify whether a hospital visit results in readmission. 9,800 rows, one row per visit, some patients have multiple visits. 64 features. My XGBoost prototype gets 94% accuracy -- design a sweep to push it higher."

**Phase 1 decisions:**
- Grouped data detected (multiple visits per patient) → **group k-fold on patient ID; random split disqualified** (Rule 1.4). *Plan header: "Researcher's prototype likely used a random split; 94% may be inflated by patient-level leakage."*
- Imbalance check run by operator (researcher didn't supply): 11:1 → PR-AUC primary, **accuracy disqualified as headline metric** (Rule 1.3)
- Baselines mandated (Rule 1.2)

**Phase 2:**
- Checkpoint 1, run xgb-01: train PR-AUC 0.97, val PR-AUC 0.41, first epoch → matches Rule 2.1 (near-perfect train, mediocre val, epoch 1) → **SWEEP SUSPENDED**
- Leakage audit (reference/stopping-criteria.md §audit): feature `discharge_disposition_code` has a value that only occurs when readmission is already known. **Target leakage confirmed.**
- Decision: drop 2 leaky features, restart sweep on clean feature set. (This is a hard stop, not an escalation -- the rule prescribes the fix.)

**Phase 3 verdict (after clean re-run):**

```
PROMOTED-WITH-FLAGS: xgb-clean-06
Rule: 3.2 + small-eval caveat path (eval folds avg 412 positive-class samples → bootstrap CI attached)
Evidence: PR-AUC 0.448 [CI 0.41–0.49], baseline 0.083. Researcher's original 94% accuracy
  was an artifact of patient leakage + disqualified metric; honest performance is above.
Flags: small-eval caveat (Rule 3.2); class blind-spot none.
Next action: collect ≥800 additional positive-class eval samples to tighten CI before deployment.
```

**Why this example matters:** the operator's verdict *contradicts* the researcher's prototype results -- and the rules forced it to. A chatbot would have helped push the leaky 94% to 96%.

---

## Example 3 -- Edge case: the near-tie + marginal-gain compound

**Researcher input:**
> "Forecast weekly demand per SKU. 130,000 rows, weekly timestamps, 2019–2025. Target spans 3 orders of magnitude. Budget: 60 GPU-hours. Optimize RMSE. Considering an LSTM."

**Phase 1 decisions:**
- Temporal data → time-based split, k-fold disqualified (Rule 1.4)
- Target spans orders of magnitude → **override: MAE-on-log-target primary; raw RMSE disqualified** (Rule 1.3), RMSE reported as secondary
- > 100k rows → LSTM permitted as one candidate, never sole: GBT primary + linear + seasonal-naive baseline (Rule 1.2)

**Phase 2 (excerpt):**
- Best LR for the LSTM sits at the minimum of the tested range → range extended one step lower before narrowing (Rule 2.3 edge-of-box)
- Budget hits 100% with LSTM curve still descending → projection rule: slope on last 5 checkpoints projects +0.6% over doubled budget, < 1% threshold → **stop, mark converged** (Rule 2.2). No silent overspend (Rule 2.3).

**Phase 3 -- the compound edge case:**
- gbt-14: log-MAE 0.231. lstm-08: log-MAE 0.230. Δ = 0.4% relative → **near-tie, Rule 3.2 tiebreak chain:**
  1. Latency: GBT 3.1 ms vs LSTM 41 ms → ≥2x faster → **GBT wins at step 1; chain terminates**
- But: gbt-14 beats the seasonal-naive baseline by only 1.6% → **marginal-gain flag also fires** (Rule 3.2)

```
PROMOTED-WITH-FLAGS: gbt-14
Rule: 3.2 tiebreak (latency, step 1) + 3.2 marginal-gain
Evidence: log-MAE 0.231 vs lstm-08 0.230 (tie, <0.5%); latency 3.1ms vs 41ms.
  Seasonal-naive baseline log-MAE 0.235 -- model gain is 1.6% relative.
Flags: MARGINAL GAIN -- if serving cost matters, the seasonal-naive baseline at zero
  inference cost gives up only 1.6%. Recommendation stands: deploy gbt-14 only if the
  1.6% is worth the pipeline; otherwise ship the baseline.
Next action: A/B the model against seasonal-naive on 4 weeks of live data before full rollout.
```

**Why this example matters:** two edge cases fire at once and the operator still emits exactly one verdict with one recommendation. The tiebreak chain terminated internally -- nothing was kicked back.

---

## Example 4 -- Edge case: the brief the data contradicts (profile mismatch at load)

**Researcher input (Stage 01):**
> "Predict assay binding affinity class from our screening platform export. Around 150,000 rows, 220 numeric features. Budget: 30 GPU-hours."

**Phase 1 designs for 150k rows:** GBT primary, NN permitted as a second candidate (Rule 1.2); single holdout acceptable (Rule 1.4).

**Stage 03 pre-flight (Rule 2.0):** the file at the recorded data source loads with **820 rows.** The researcher quoted the platform's full database; the shared export was a filtered subset. The recomputed profile flips two Phase 1 gates:
- <1,000 rows → deep learning excluded, regularized linear mandated; the NN candidate is flagged "insufficient data for NN" (Rule 1.2)
- <1,000 rows → 5-fold CV minimum; the planned single holdout is disqualified (Rule 1.4)

**Decision (Rule 2.0):** trust the data, not the brief. The brief is corrected in place with a logged correction, Phase 1 re-runs on the measured profile, and the corrected plan warns up front that a 164-row test partition sits below the 200-sample promotion floor (Rule 3.1) -- surfaced at design time, not discovered at the verdict. No training code ran against the wrong design.

**Why this example matters:** nothing failed in any training run -- the failure was in the paperwork. Rule 2.0 makes the loaded data, not the intake conversation, the authority on the profile.

---

## Example 5 -- Edge case: the redesign loop (the dataset with no signal)

**Setup:** predict 30-day relapse from a 6,100-row registry extract, 23 sparse features. The sweep launches; at 25% of budget every candidate -- GBT, random forest, logistic -- sits below the majority-class baseline → Rule 2.1 hard stop: suspend, return to Phase 1 redesign.

**Redesign #1:** the operator does everything the rules allow: re-encodes features, adds class weights, narrows the model set. New sweep. At 25% of budget: every candidate below baseline again. Identical Rule 2.1 signal, same data source.

**Decision (Rule 2.1 recurrence):** no third design. The recurrence rule converts the loop into a verdict: the pipeline suspends, the dashboard presents both sweeps side by side, and Phase 3 emits `KILLED -- signal not found at current data/feature quality` (Rule 3.1) with the prescribed next step: richer features or more rows, named concretely.

**Why this example matters:** every individual decision was rule-correct; the failure only exists across iterations. Without the recurrence rule, an autonomous pipeline whose recovery is "redesign and retry" burns the entire budget rediscovering the same absence of signal. The recurrence rule is the one place the operator reads its own history instead of the current checkpoint.

---

## Example 6 -- Edge case: the browser in the telemetry loop (Colab mode)

**Setup:** no CUDA-capable GPU at setup, so `execution_mode: colab`. The researcher logged in to Google once in a visible Chromium window at setup; since then the operator drives Colab headlessly through Playwright (`library/colab_runner.py`) -- inserting cells, running them, scraping output. No copy-paste, but the telemetry now travels through a browser session that Google can recycle at any time.

**The read:** mid-sweep, the operator executes the cell for `run-gbt-07`. While it runs, Colab recycles the runtime; the reconnected page renders the notebook's last persisted state, and the scrape returns output headed `run-gbt-04` -- plausible numbers, wrong run.

**Decision (Rule 2.0):** no triage. Killing or extending run 07 on run 04's curve would corrupt the sweep history that every Phase 3 promotion rule depends on. The operator reconnects with the saved profile, re-runs the `run-gbt-07` cell, and logs both the rejected read and the runtime recycle in the sweep log. Had the login itself expired, the rule pauses the loop and asks the researcher to repeat the one-time headed login before resuming.

**Why this example matters:** automating the transport does not remove the edge case -- it swaps human failure modes (stale pastes) for browser ones (recycled runtimes, expired sessions, drifted DOM). Every Phase 2 rule assumes the telemetry describes the run being triaged; Rule 2.0 makes that assumption safe no matter what carries the bytes.
