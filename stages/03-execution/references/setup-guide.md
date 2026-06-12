# Setup Guide -- Experiment Execution

Required tools for Stage 03. Install before running the first sweep.

---

## Python Environment

**Required:** Python 3.8 or later

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pandas scikit-learn numpy
```

**macOS note:** install from the default PyPI index instead (`pip install torch torchvision torchaudio pandas scikit-learn numpy`) -- the CUDA index has no macOS wheels, and MPS (Apple Silicon GPU) support ships in the standard build.

**Linux + AMD note:** with ROCm drivers installed (`rocm-smi` works), use the ROCm wheels: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2` (match the index to your ROCm version where one exists). ROCm presents as CUDA inside PyTorch -- `torch.cuda.is_available()` returns True and the library needs no changes. AMD on Windows and Intel GPUs are not supported for local training; setup routes those machines to Colab and tells the researcher why.

GPU support: the hardware auto-tuner (`library/hardware.py`) selects `cuda`, `mps`, or `cpu` automatically. No manual device configuration needed.

**Windows note:** `library/hardware.py` sets `num_workers=0` on Windows automatically to prevent multiprocessing deadlocks. No action required.

---

## Node.js Dashboard

**Required:** Node.js 18 or later

```bash
cd dashboard/
npm install
node server.js
```

Dashboard runs at `{{DASHBOARD_URL}}`. Keep it running during sweeps -- the tracker streams checkpoint data to it in real time.

---

## Colab Mode (no local CUDA GPU)

If `setup/environment.md` says `execution_mode: colab`, training runs on Google Colab through a Playwright-driven browser instead of locally.

```bash
pip install playwright
playwright install chromium
python library/colab_runner.py --login
```

The last command opens a visible Chromium window on Colab. Log in to your Google account there; the window closes itself once login is detected. Auth persists in a local browser profile (`~/.ml-experiment-pilot/chromium-profile`), and every later Colab interaction is headless -- the operator inserts cells, runs them, and reads output with no further action from you.

Notes:
- On Linux, run `playwright install --with-deps chromium` instead -- it also installs the system libraries Chromium needs (may require sudo). Windows and macOS need no extra step.
- The data source must be reachable from Colab: a public URL or a mounted Google Drive path. Local-only file paths cannot be used in colab mode.
- The local PyTorch install above is unnecessary in colab mode. The dashboard install still applies -- the operator relays scraped checkpoints to it.
- If the Google login expires mid-sweep, the operator pauses and asks you to re-run the `--login` command.

---

## Verifying the Setup

Before launching the first sweep, confirm:

1. `python -c "import torch; print(torch.cuda.is_available())"` -- returns True if GPU is available
2. Dashboard server starts without errors and the browser shows an empty sweep list
3. Run `library/tracker.py` standalone to confirm it can reach the dashboard: `python library/tracker.py`
