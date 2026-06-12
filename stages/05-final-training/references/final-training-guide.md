# Final Training Guide -- Stage 05

How the committed final run differs from a sweep run. Stage 03 explores many configurations and kills most of them; Stage 05 trains the single winner to completion.

---

## What changes from the sweep

| Aspect | Sweep (Stage 03) | Final run (Stage 05) |
|--------|------------------|----------------------|
| Number of runs | Many candidates, many seeds | One: the promoted configuration |
| Triage kills (§2.2) | Active -- overfit/plateau/bottom-50% runs killed early | Not applied -- the run trains to completion |
| Budget halving / successive halving | Active | Not applied |
| Early stopping (`patience`/`min_delta`) | Active | Active -- this is how "complete" is defined |
| Integrity hard stops (§2.0, §2.1) | Active | Active -- leakage and NaN still abort |
| Hyperparameters | Swept across ranges | Fixed to the winning values from the verdict |

"Run until training is complete" means: train full epochs until early stopping converges (no `min_delta` improvement for `patience` epochs) or the configured max epoch count is reached -- whichever comes first. The run is never stopped for being a poor performer, because it already won.

## Data

Always split into the fixed 0.64 train / 0.16 validation / 0.20 test partitions (same seed and `DataCleaner` fit-on-train transforms as Stage 03). The final run never merges train and validation, and never uses any other ratio -- the partitioning is identical to the sweep so the final model sees the same training distribution the sweep selected on. Train on the train partition with mandatory `patience`-based early stopping measured on the validation partition, then evaluate once on the held-out test partition after convergence and report that number in the model card as the headline metric.

## Artifact

Save the trained weights with `torch.save(model.state_dict(), "output/final-model.pt")`. The model card records everything needed to reload and serve the model: model family, full hyperparameters, epochs actually trained, final test metric, device / execution mode, and the artifact path.

## Example output

After training and test evaluation, run the model on a small sample of held-out test examples (8-12) and write `output/example-output.md` so the researcher can see the model in action without loading anything. For each sample show the input, the prediction with its confidence, and the true label. For image tasks, save a thumbnail of each sample under `output/examples/` and reference it from the markdown so the picture sits beside its predicted and true class; for tabular tasks, show the record id or key feature values instead. This is a demonstration artifact, not an evaluation -- the headline number is still the full test-set metric in the model card.

## Execution mode

Local: execute the training code directly. Colab: run it through `ColabRunner.run_cell` exactly as in Stage 03, with the run-ID header and `RUN-COMPLETE` sentinel so the long final run can be validated and, if a runtime recycles, re-issued. Relay checkpoints to the dashboard either way so the researcher can watch the final run converge.
