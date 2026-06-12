# Stage 03 -- Experiment Execution

Autonomous training loop. Write code, execute, read checkpoint output, apply Phase 2 triage rules, adjust hyperparameters. Repeat until the compute budget is exhausted or all runs are resolved.

## Inputs

| Source | File/Location | Section/Scope | Why |
|--------|--------------|---------------|-----|
| Stage 02 output | `../02-experiment-design/output/experiment-plan.md` | Full file | Model candidates, hyperparameter ranges, budget, validation strategy |
| Operator rules | `../../operator/rules.md` | Phase 2 -- §2.0 through §2.3 | Pre-flight integrity checks plus per-checkpoint and sweep-level triage logic |
| Reference | `references/setup-guide.md` | Full file | Python and Node.js environment requirements |
| Reference | `../../operator/reference/stopping-criteria.md` | Full file | Hard-stop diagnosis and leakage audit procedures |
| Library | `../../library/[model]-template.py` | Full file | Starting code for the model family in the plan |
| Library | `../../library/hardware.py` | Full file | Device selection, batch scaling, OOM guard |
| Library | `../../library/data_cleaner.py` | Full file | Preprocessing, split integrity, leakage audit |
| Library | `../../library/tracker.py` | Full file | Checkpoint telemetry to dashboard |
| Library | `../../library/colab_runner.py` | Full file | Playwright control of Colab in colab mode -- headless cell execution and output scraping |
| Setup | `../../setup/environment.md` | `execution_mode`, `colab_notebook_url` | Determines whether code executes locally or in a Playwright-driven Colab notebook |

## Process

In colab mode, every code-execution step below runs inside the Colab notebook through `ColabRunner` (`library/colab_runner.py`): the operator inserts the code as a cell, executes it headlessly, and scrapes the printed output. The data source must be reachable from Colab (public URL or mounted Google Drive) -- escalate at pre-flight if it is not. Store the notebook URL as `colab_notebook_url` in `../../setup/environment.md` after the first connect and reuse it for every later cell. Kill and suspend decisions are enforced in colab mode with `ColabRunner.interrupt_cell()` -- a run the rules have killed must not keep consuming Colab compute; confirm the interrupt landed by scraping for `KeyboardInterrupt` before relaying the final status to the dashboard.

**Data preparation (once, before the loop):**

1. Load the dataset from the data source recorded in `../01-intake/output/problem-brief.md`; run the §2.0 pre-flight -- recompute the profile from the loaded data; if any Phase 1 gate flips, correct the brief, log the correction, and re-run Stage 02 design before continuing; HARD STOP if the target column is missing
2. Split into three non-overlapping partitions using the strategy from the experiment plan:
   - 0.64 train / 0.16 validation / 0.20 test
   - Use time-based split if temporal flag is set; use GroupKFold if entity/group flag is set
3. Run `DataCleaner.run_leakage_audit(train_df, val_df)` -- if any flag is raised, trigger HARD STOP (§2.1) before writing any training code
4. Apply `DataCleaner.impute_missing` and `DataCleaner.scale_features` fitting on train set only; transform all three partitions

**Training loop (repeat until budget exhausted or all runs resolved):**

5. Select the library template matching the current model candidate
6. Write training code for this run -- configure hyperparameters (including `patience` and `min_delta`), wire `MLTracker`; each epoch must: train on train set, then evaluate on validation set before advancing; if validation error has not improved by at least `min_delta` for `patience` consecutive epochs, stop training immediately and proceed to test evaluation; after all epochs complete or early stopping triggers, evaluate once on test set; in colab mode the code must print a run ID header with every checkpoint and a `RUN-COMPLETE [run-id]` sentinel at the end so scraped output can be validated
7. Check `execution_mode` in `../../setup/environment.md`: if `local`, execute the training code; if `colab`, execute it in the Colab notebook via `ColabRunner.run_cell` -- reconnect with the saved profile if the session dropped; never run training code on the local CPU
8. Read checkpoint output from the tracker (local mode) or scrape it from the Colab cell output via `ColabRunner` (colab mode); validate scraped output against the expected run ID and schema (§2.0) before applying any triage rule -- on mismatch or a recycled session, re-run the cell instead of triaging; relay scraped checkpoints to the local dashboard so live monitoring keeps working
9. Apply §2.1 hard-stop rules first (first match wins); if triggered: suspend immediately, notify researcher via dashboard, pipeline resets to Stage 01; if the same hard-stop signal fires a second consecutive time on the same data source, suspend permanently instead of resetting (§2.1 recurrence rule)
10. Apply §2.2 per-run rules (first match wins): kill, narrow, or continue
11. Apply §2.3 sweep-level rules: narrow search space or extend range as indicated
12. If continuing or narrowing: adjust hyperparameters and return to step 6
13. When all runs are resolved: write the sweep log

## Outputs

| Artifact | Location | Format |
|----------|----------|--------|
| Sweep log | `output/sweep-log.md` | Per-run record: code configuration, checkpoint history, triage decision, rule citation, final status |
