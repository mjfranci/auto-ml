# ml-experiment-pilot

An automated ML operator for biomedical technology researchers. Describe your research problem and dataset; the operator designs the experiment, writes and executes training code, manages the sweep, returns an auditable promotion verdict, and trains the promoted model to completion.

## Folder Map

```
ml-experiment-pilot/
├── CLAUDE.md              (you are here)
├── CONTEXT.md             (task routing)
├── setup/                 (onboarding questionnaire)
├── operator/              (decision rules, identity, examples, reference)
├── library/               (model templates and helper utilities)
├── dashboard/             (live sweep monitoring interface)
└── stages/
    ├── 01-intake/         (collect problem brief)
    ├── 02-experiment-design/  (design the experiment plan)
    ├── 03-execution/      (autonomous training loop)
    ├── 04-promotion/      (model promotion verdict)
    └── 05-final-training/ (train promoted model to completion)
```

## Triggers

| Trigger | Description / Prompt | Action to Take |
|---------|---------------------|----------------|
| `setup` | Install dependencies & run onboarding | 0. Verify prerequisites: run `python --version` and `node --version`. If either command is not found, stop and tell the researcher to install it per the Prerequisites section of README.md (Python 3.8+, Node.js 18+) before re-running `setup` -- do not continue to GPU detection or package installs.<br>1. Detect a usable GPU, in order: (a) Run `nvidia-smi` -- if it reports a device, write `setup/environment.md` with `execution_mode: local` and `device: cuda`. (b) If `nvidia-smi` fails (it only exists with NVIDIA drivers -- AMD GPUs, Intel GPUs, and Macs all fail here) and the OS is macOS with `uname -m` returning `arm64`, this is an Apple Silicon Mac: write `execution_mode: local` and `device: mps`; after step 3's install, verify with `python -c "import torch; print(torch.backends.mps.is_available())"` and fall back to branch (d) if it prints False. (c) If the OS is Linux and `rocm-smi` succeeds, this is an AMD GPU with ROCm drivers: write `execution_mode: local` and `device: rocm`; after step 3's install, verify with `python -c "import torch; print(torch.cuda.is_available())"` (ROCm builds report as CUDA) and fall back to branch (d) if it prints False. (d) Otherwise no locally usable GPU exists. First check whether an unsupported GPU is present (Windows: `Get-CimInstance win32_VideoController | Select-Object Name`; Linux: `lspci | grep -iE 'vga|3d'`): if an AMD or Intel GPU appears, tell the researcher explicitly that their GPU cannot run local PyTorch training (AMD requires Linux with ROCm drivers; Intel GPUs are not supported by this library) -- do not imply they have no GPU. Then: tell the researcher that training locally would exceed a typical compute budget and strongly recommend Google Colab; run `pip install playwright` then `playwright install chromium` (Linux: `playwright install --with-deps chromium`, which also pulls system libraries and may need sudo); tell the researcher to log in to their Google account in the Chromium window that is about to open, then run `python library/colab_runner.py --login` -- this opens a visible browser on Colab, waits for the login, and saves the auth to a local profile; after login the window closes and all further Colab control is headless via `library/colab_runner.py` -- you insert, execute, and read training code in Colab automatically, as if running locally; write `setup/environment.md` with `execution_mode: colab`.<br>2. Ask Qs in [setup/questionnaire.md](./setup/questionnaire.md).<br>3. Skip if `execution_mode` is `colab`. If `device` is `mps`: run `pip install pandas scikit-learn numpy torch torchvision torchaudio` (default PyPI -- the CUDA index has no macOS wheels, and MPS support ships in the standard build), then run the MPS verification from step 1b. If `device` is `rocm`: read the installed ROCm version (`cat /opt/rocm/.info/version`) and run `pip install pandas scikit-learn numpy torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm[X.Y]` using the closest available rocm index (default rocm6.2), then run the verification from step 1c. If `device` is `cuda`: run `nvcc --version` to check for the CUDA toolkit; if missing, run `pip install cuda-python` to install CUDA runtime bindings and warn the researcher that the full CUDA toolkit can be installed from https://developer.nvidia.com/cuda-downloads if custom kernel compilation is needed. Parse the CUDA version from `nvidia-smi` output (the "CUDA Version" field) and select the matching PyTorch index URL: cu118 for CUDA 11.x, cu121 for CUDA 12.1, cu124 for CUDA 12.4 or higher; default to cu118 if the version cannot be parsed. Run `pip install pandas scikit-learn numpy torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu[VERSION]`<br>4. Run `npm install --prefix dashboard` (shell-neutral -- `&&` chaining fails in Windows PowerShell) |
| `run dashboard` | Start Node.js dashboard server | Run `node dashboard/server.js` |
| `verify` | Check PyTorch GPU & Tracker | 1. Run `python -c "import torch; print('CUDA:', torch.cuda.is_available(), '/ MPS:', torch.backends.mps.is_available())"`<br>2. Run `python library/tracker.py` |
| `status` | Show pipeline stage status | Check the output directories for each stage (`stages/*/output/`) and report completion status. |
| `extend library` | Add a new model template | Copy `library/template_skeleton.py` to `library/[model]-template.py` and implement requested architecture. |

## Routing

| Task | Go To |
|------|-------|
| Run a new experiment | `stages/01-intake/CONTEXT.md` |
| Design experiment plan | `stages/02-experiment-design/CONTEXT.md` |
| Execute the sweep | `stages/03-execution/CONTEXT.md` |
| Get the promotion verdict | `stages/04-promotion/CONTEXT.md` |
| Train the promoted model | `stages/05-final-training/CONTEXT.md` |
| Add a model template | `library/README.md` -- "How to Add New Model Templates" |

## What to Load

| Task | Load These | Do NOT Load |
|------|-----------|-------------|
| Research Intake | `stages/01-intake/references/intake-questionnaire.md`, `library/README.md` "How to Add New Model Templates", `library/template_skeleton.py`, `operator/reference/model-selection-rubric.md` Simplicity classes | `operator/rules.md`, `operator/examples.md`, `library/tracker.py`, `library/hardware.py`, model templates, `dashboard/` |
| Experiment Design | `stages/01-intake/output/problem-brief.md`, `operator/rules.md` §1.1–1.5, `operator/reference/experiment-design-checklist.md` | operator/examples.md, library/, dashboard/, stages/03-execution/, stages/04-promotion/ |
| Experiment Execution | `stages/02-experiment-design/output/experiment-plan.md`, `operator/rules.md` §2.0–2.3, `operator/reference/stopping-criteria.md`, matching `library/[model]-template.py`, `library/hardware.py`, `library/data_cleaner.py`, `library/tracker.py`, `library/colab_runner.py` (colab mode) | operator/examples.md, operator/reference/model-selection-rubric.md, stages/01-intake/, stages/04-promotion/ |
| Promotion Decision | `stages/03-execution/output/sweep-log.md`, `operator/rules.md` §3.1–3.3, `operator/reference/metric-priority-matrix.md`, `operator/reference/model-selection-rubric.md` Simplicity classes | operator/examples.md, library/, dashboard/, stages/01-intake/, stages/02-experiment-design/ |
| Final Training | `stages/04-promotion/output/verdict.md`, `stages/01-intake/output/problem-brief.md` (data source), `stages/02-experiment-design/output/experiment-plan.md`, `operator/rules.md` §2.0–2.1, matching `library/[model]-template.py`, `library/hardware.py`, `library/data_cleaner.py`, `library/tracker.py`, `library/colab_runner.py` (colab mode) | operator/examples.md, operator/reference/, stages/03-execution/ |
| Library Extension | `library/template_skeleton.py`, `library/README.md` "How to Add New Model Templates", `operator/reference/model-selection-rubric.md` Simplicity classes | stages/, operator/rules.md, dashboard/ |

## Stage Handoffs

Each stage writes its output to its own `output/` folder. The next stage reads from there. If you edit an output file between stages, the next stage picks up your edits.

