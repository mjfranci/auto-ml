# Stage 05 -- Final Model Training

Train the promoted model to completion. Take the winning configuration from the Stage 04 verdict, write its training code, run it to full convergence, save the deployable model artifact and its test-set performance, and produce a sample of its predictions for the researcher to view.

## Inputs

| Source | File/Location | Section/Scope | Why |
|--------|--------------|---------------|-----|
| Stage 04 output | `../04-promotion/output/verdict.md` | Full file | Verdict keyword and promoted run name |
| Stage 02 output | `../02-experiment-design/output/experiment-plan.md` | Promoted candidate's hyperparameter block + validation strategy | Winning config and split strategy |
| Stage 01 output | `../01-intake/output/problem-brief.md` | Data source field | Where to load the dataset |
| Operator rules | `../../operator/rules.md` | §2.0 pre-flight; §2.1 hard stops | Integrity gates still apply to the committed run |
| Reference | `references/final-training-guide.md` | Full file | Committed-run semantics: train to completion, no sweep triage |
| Library | `../../library/[model]-template.py` | Full file | Architecture for the promoted model family |
| Library | `../../library/data_cleaner.py` | Full file | Reproduce the Stage 03 split + preprocessing |
| Library | `../../library/hardware.py` | Full file | Device selection, batch scaling, OOM guard |
| Library | `../../library/tracker.py` | Full file | Stream final-run telemetry to the dashboard |
| Library | `../../library/colab_runner.py` | Full file | Headless Colab execution in colab mode |
| Setup | `../../setup/environment.md` | `execution_mode`, `device`, `colab_notebook_url` | Local vs Colab execution |

## Process

The final run is the committed training of the chosen model -- not a sweep. Sweep triage (§2.2/§2.3 kills, budget halving) does not apply; the run trains to completion. Integrity hard stops (§2.0 leakage/NaN, §2.1) still apply.

1. Read the verdict. If the keyword is `KILLED` or `BLOCKED`, no model was promoted -- stop; the verdict is the terminal output and Stage 05 does not run. Continue only on `PROMOTED` / `PROMOTED-WITH-FLAGS`
2. Extract the promoted run's model family and winning hyperparameters from the verdict and the matching candidate block in the experiment plan
3. Reload the dataset from the problem brief's data source; always split into the fixed 0.64 train / 0.16 validation / 0.20 test partitions (same seed and strategy as Stage 03 -- never merge train and validation or use any other ratio); run §2.0 pre-flight (HARD STOP on leakage or a missing target)
4. Select the matching `library/[model]-template.py`; write final training code with the winning hyperparameters, wired to `MLTracker`; in colab mode, print the run-ID header and `RUN-COMPLETE` sentinel with each checkpoint
5. Train to completion with mandatory `patience`-based early stopping: train on the train partition, validate each epoch on the validation partition, and stop when validation error has not improved by `min_delta` for `patience` consecutive epochs or the max epoch count is reached; do not apply sweep-triage kills. Execute locally or via `ColabRunner.run_cell` per `execution_mode`
6. Evaluate the converged model once on the held-out test partition
7. Save the trained weights to `output/final-model.pt` and write the model card
8. Produce an example output for the researcher to view: run the trained model on a small sample of held-out test examples (8-12), and write `output/example-output.md` showing, per sample, the input (a saved thumbnail under `output/examples/` for image data, or the feature row / record id otherwise), the model's prediction with confidence, and the true label -- so the researcher can see the model's behaviour on real inputs at a glance

## Audit

| Check | Pass Condition |
|-------|---------------|
| Split fixed | Data partitioned 0.64 / 0.16 / 0.20 (train / val / test) -- no train+val merge and no alternate ratio |
| Trained to completion | Run ended by `patience` early-stopping convergence or max epochs -- not a triage kill or budget halt |
| Test metric reported | Final test-set value for the verdict's primary metric is recorded in the model card |
| Artifact saved | `output/final-model.pt` exists and loads back into the template architecture |
| Example output shown | `output/example-output.md` exists, with held-out sample predictions each paired with the true label |

## Outputs

| Artifact | Location | Format |
|----------|----------|--------|
| Final model | `output/final-model.pt` | Trained PyTorch weights (state dict) for the promoted configuration |
| Model card | `output/final-model-card.md` | Promoted config, hyperparameters, epochs trained, final test metric, artifact path, execution mode |
| Example output | `output/example-output.md` (+ `output/examples/` thumbnails for image data) | Sample predictions on held-out test examples: input (thumbnail for images), predicted label + confidence, true label -- formatted for human viewing |
