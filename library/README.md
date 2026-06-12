# AutoML Experiment Pilot Library

Repeatable PyTorch starter templates and helper utilities for the ML Experiment Pilot operator. The operator selects and configures templates from this library during Stage 03 execution.

---

## Folder Structure

```
library/
├── README.md               # This file
├── tracker.py              # HTTP telemetry tracker -- streams checkpoints to dashboard
├── hardware.py             # Device inspector and auto-tuner (AMP, OOM guard, batch scaling)
├── data_cleaner.py         # Tabular/sequence preprocessing, split leak auditor
├── colab_runner.py         # Playwright driver for Google Colab (colab mode)
├── template_skeleton.py    # Boilerplate for adding new model templates
├── cnn_template.py         # 2D Convolutional Neural Network
├── rnn_template.py         # Vanilla Recurrent Neural Network
├── lstm_template.py        # Long Short-Term Memory
├── gru_template.py         # Gated Recurrent Unit
├── gcn_template.py         # Graph Convolutional Network (pure PyTorch)
├── tokenizer_template.py   # HuggingFace Tokenizer integration
└── diffusion_template.py   # Denoising Diffusion Probabilistic Model (DDPM)
```

---

## Core Helper Utilities

### tracker.py -- `MLTracker`
Interfaces with the Node.js dashboard.
- Registers sweeps, hyperparameters, and model classes
- Streams training and validation checkpoints (loss, primary metric, LR, custom metrics) in real time
- Receives Phase 2 early-stopping recommendations (KILL, SUSPEND) from the dashboard

### hardware.py -- `HardwareOptimizer`
Adapts execution to host compute resources.
- **Device selection:** maps tensors to `cuda`, `mps`, or `cpu` automatically
- **Batch size scaling:** adjusts dynamically based on CUDA VRAM
- **Multiprocessing:** sets `num_workers=0` on Windows; scales on Linux/macOS
- **AMP:** enables PyTorch float16 autocasting on supported GPUs
- **OOM guard:** catches CUDA out-of-memory errors, halves batch size, resumes

### data_cleaner.py -- `DataCleaner`
Builds clean data flows and audits splits.
- Imputation and scaling fitted strictly on training folds (no split leakage)
- Hashes split rows to detect exact/near-duplicate leakage across folds
- Checks entity (patient, user, site) overlap between train and val splits
- Flags proxy features with correlation > 0.85 to the target before training

### colab_runner.py -- `ColabRunner`
Runs training on Google Colab when no local CUDA GPU exists.
- One-time headed Chromium login to Google (`python library/colab_runner.py --login`); auth persists in a local browser profile
- All later control is headless: inserts cells, executes them, scrapes printed output
- `run_cell` polls for the run-ID sentinel and takes an `on_output` callback for relaying checkpoints to the dashboard
- `interrupt_cell()` stops the executing cell (Ctrl+M I, Runtime-menu fallback) -- how KILL and SUSPEND verdicts are enforced on Colab; verify with a `KeyboardInterrupt` scrape
- Best-effort `ensure_gpu()` switches the Colab runtime to a GPU; always verify with a `torch.cuda.is_available()` probe cell

---

## How to Add New Model Templates

1. Duplicate `template_skeleton.py` and name it `[model]-template.py`
2. Implement your PyTorch `nn.Module` architecture
3. Set `simplicity_class` in `MLTracker(...)` -- see `operator/reference/model-selection-rubric.md` Simplicity classes table
4. Log parameter count, precision type, and estimated inference latency in the `hyperparameters` dict
5. Call `triage = tracker.log_checkpoint(...)` at the end of each epoch
6. Inspect the returned `triage` recommendation -- halt if status is `KILL` or `SUSPEND`
