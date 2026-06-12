# AutoML Experiment Pilot

An automated, rule-based machine learning operator for biomedical technology researchers. Describe your research problem and dataset; the operator designs the experiment plan, handles training sweeps, applies early-stopping and leakage-audit rules, and delivers an auditable promotion verdict—all monitored via a live telemetry dashboard.

This project implements the **interpretable context methodology** (folders as architecture, rules over vibes) to build a trustworthy, portable AI operator.

---

## 🗺️ Project Architecture

```
ml-experiment-pilot/
├── README.md               # This developer guide
├── CLAUDE.md               # Task routing and agent triggers
├── CONTEXT.md              # Shared resource map
├── setup/                  # Project initialization questionnaire
├── operator/               # Decision rules, identity, and examples
│   ├── CONTEXT.md          # Operator onboarding instructions
│   ├── identity.md         # Role, tone, and boundaries
│   ├── rules.md            # The rules engine (Phases 1, 2, & 3)
│   ├── examples.md         # Walkthrough cases of complex decisions
│   └── reference/          # Checklists, stopping criteria, rubrics
├── library/                # PyTorch templates & training utilities
└── dashboard/              # Express/Node.js web interface
```

---

## Prerequisites

Install these on your machine **before running `setup`**. The agent installs Python and Node *packages* for you, but it cannot install the runtimes themselves:

- **Python 3.8+** -- [python.org/downloads](https://www.python.org/downloads/). On Windows, tick "Add Python to PATH" during installation.
- **Node.js 18+** -- [nodejs.org](https://nodejs.org/). Required for the dashboard server.

Confirm both are visible from your shell before continuing: `python --version` and `node --version`.

---

## Permissions Required

This workspace requires Claude to create and edit files and run shell commands. You must grant these permissions before running `setup` or any other trigger.

### Option 1 — Allow each time (simplest)

When Claude asks to run a command or write a file, a prompt will appear. Select **Yes, always allow** to grant that permission for the rest of the session.

### Option 2 — Grant permissions up front (recommended)

Add a `.claude/settings.json` file to this folder with the following content:

```json
{
  "permissions": {
    "allow": ["Bash", "PowerShell", "Edit", "Write"]
  }
}
```

This grants Claude permission to run shell commands and create or edit files without prompting each time. The file applies only to this workspace. Two things matter here: the rules nest under `permissions.allow` (a top-level `allowedTools` key is not read by Claude Code), and shell commands are matched per shell -- on **Windows** they run through PowerShell, so the `PowerShell` rule is what stops the prompts there, while `Bash` covers macOS and Linux. List both so the workspace stays hands-off on every platform.

### Option 3 — Use the `/permissions` command

In the Claude Code chat, type:

```
/permissions
```

This opens an interactive dialog where you add `Bash`, `Edit`, and `Write` to the allow list and pick the scope (project, user, or local). The command opens the dialog -- it does not take tool names as arguments, so add each rule inside it. `/allowed-tools` is an alias for the same command.

---

## ⚡ Quick Start: Zero-Install Agent Setup

**Beyond the prerequisites above, you do not need to install anything else.** Drop this folder into **Claude Code** and instruct the agent to run the triggers -- the agent handles all package installation, dashboard launching, and environment verification for you.

### How to Run:

1.  **Drop this folder into Claude Code.**
2.  **Tell the agent to `setup`:** 
    The agent will walk you through the onboarding questionnaire (setting up your default budget and dashboard URL) and automatically run:
    *   `pip install pandas scikit-learn numpy torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
    *   `npm install` inside the `dashboard/` directory.
    *   **No NVIDIA GPU?** Apple Silicon Macs train locally on the built-in GPU (MPS), and AMD GPUs on Linux train locally through ROCm. Anything else -- AMD on Windows, Intel GPUs, or no GPU at all -- switches to Colab mode, and the agent tells you why: it installs Playwright, opens a Chromium window for you to log in to your Google account once, then drives Colab headlessly -- writing, running, and reading training code for you exactly as if it were local.
3.  **Tell the agent to `run dashboard`:**
    The agent will launch the Node.js telemetry dashboard server in the background. Once running, you can open `http://localhost:3000` to monitor training runs and sweeps in real-time.    - The top-right dashboard settings are:
      - **Baseline Metric:** the reference score used to compare each candidate model against a trivial or baseline performance level. The operator uses this value to assess whether a model is meaningfully better than the baseline.
      - **Eval Set Size:** the effective number of evaluation examples used when the dashboard computes the promotion verdict. It helps the dashboard estimate whether the reported improvement is statistically reliable for the chosen metric.4.  **Tell the agent to `verify`:**
    The agent will run verification audits automatically to check your PyTorch GPU support and test the tracker's connection to the dashboard server.

Once setup is complete, you can begin your machine learning experiments simply by describing your research problem and dataset profile directly to the agent.

---

## 🐾 Example: Classify Dogs, Cats, and Pandas

A complete first run, from a real dataset to a trained model.

**1. Download the dataset.** Get the *Animal Image Dataset (Dog, Cat, and Panda)* from Kaggle:

https://www.kaggle.com/datasets/ashishsaxena2209/animal-image-datasetdog-cat-and-panda

**2. Extract it into the project folder.** Unzip the download directly into this directory -- the same level as this `README.md`. You should end up with an `archive/` folder holding the 3,000 labelled images:

```
ml-experiment-pilot/
├── README.md          <- you are here
├── archive/           <- extracted dataset (3,000 labelled images)
├── CLAUDE.md
├── operator/
├── library/
└── dashboard/
```

**3. Tell the operator what you want.** Describe the task in plain language to start the workflow:

> I want to classify the images in the archive folder. There are 3000 images, already labelled.

That is all the operator needs. It assesses your skill level once, then runs the five stages autonomously -- reviewing the literature to choose an image-classification model, designing the experiment with mandatory baselines, sweeping candidates while streaming progress to the dashboard, emitting a promotion verdict, and training the winner to completion on the fixed 0.64 / 0.16 / 0.20 split. You re-engage at the final trained model -- together with a sample of its predictions on held-out images that you can view.

> **Final note -- transfer learning is your call.** You can tell the operator whether to use transfer learning. If you instruct it to use transfer learning, it trains on a pre-existing (pretrained) model, adapting it to your data. If you instruct it not to, it trains a new model from scratch. Add the directive to your instruction, for example *"...use transfer learning"* or *"...train from scratch"*.

---

## 🌐 Running on Google Colab

If you have no local GPU, the agent switches to Colab mode during `setup` (it tells you when it does). A few things to know if you are running this way:

- **Upload your dataset to Google Drive first, and note its location.** Colab cannot see files on your computer, so -- instead of extracting into the project folder as in the example above -- put your data in Drive (for example, `My Drive > {your folder name}`) and give the operator that location when you describe your task, so it can mount Drive and load the data.
- **Don't open the dashboard until Stage 03.** In Colab mode the operator only relays telemetry once training begins, so opening the dashboard earlier just shows an empty page. Wait until the sweep is running.
- **Expect Google login prompts on a second device or by email.** Signing in to Google through the agent's browser usually triggers a security check -- you will probably need to approve the sign-in from your phone or confirm it by email before the headless session can take over.

---

## 🔄 End-to-End Workflow Stages

The workflow executes sequentially through 5 stages. Each stage reads its input from the previous stage's `output/` directory, runs its processes, and writes a structured artifact:

```
[STAGED FLOW]
Stages: 01-Intake ──> 02-Design ──> 03-Execution ──> 04-Promotion ──> 05-Training
Files:  problem-brief   experiment-plan   sweep-log       verdict       final-model
```

### Stage 1: Research Intake
*   **Location:** [stages/01-intake/](./stages/01-intake/)
*   **Description:** Collects the scientific goal, dataset parameters, and budget ceiling. The operator assesses the researcher's skill level to calibrate explanation depth.
*   **Output:** Creates `output/problem-brief.md`.

### Stage 2: Experiment Design
*   **Location:** [stages/02-experiment-design/](./stages/02-experiment-design/)
*   **Description:** Evaluates Phase 1 rules (framing, model gates, baseline selection, metrics). Corrects user mistakes (e.g. overrides Accuracy with macro-F1 for imbalanced sets) and outputs a design plan.
*   **Output:** Creates `output/experiment-plan.md`.

### Stage 3: Sweep Execution
*   **Location:** [stages/03-execution/](./stages/03-execution/)
*   **Description:** Executes autonomous training loops using PyTorch templates in the [library/](./library/) folder. The `MLTracker` telemetry client streams checkpoints to the dashboard server. The server runs Phase 2 triage logic, issuing immediate kill (overfitting/plateauing) or suspend (data leakage/NaN loss) commands.
*   **Output:** Creates `output/sweep-log.md`.

### Stage 4: Promotion Verdict
*   **Location:** [stages/04-promotion/](./stages/04-promotion/)
*   **Description:** Applies Phase 3 rules to sweep results. Compares candidate scores to baselines, boots confidence intervals for small validation sets, and resolves near-ties deterministically using a hierarchical tiebreak chain.
*   **Output:** Creates `output/verdict.md`.

### Stage 5: Final Model Training
*   **Location:** [stages/05-final-training/](./stages/05-final-training/)
*   **Description:** Takes the promoted configuration from the verdict and trains it to completion -- a single committed run, not a sweep, so triage kills and budget halving do not apply (early stopping and integrity hard stops still do). Reproduces the Stage 03 split, trains until convergence, and evaluates once on the held-out test set. Runs only when a model was promoted; a KILLED or BLOCKED verdict ends the pipeline at Stage 4.
*   **Output:** Creates `output/final-model.pt`, `output/final-model-card.md`, and `output/example-output.md` (sample predictions you can view).

---

## 📐 AutoML Rules & Edge Case Coverage

The operator's ruleset ([operator/rules.md](./operator/rules.md)) is built to handle complex ML dilemmas:

1.  **Imbalanced Classes (Accuracy vs. F1/Recall):** Imbalance ratios $>10:1$ disqualify accuracy and prioritize PR-AUC. Ratios $3:1$ to $10:1$ mandate macro-F1 (§1.3).
2.  **Model Near-Ties:** If two top models perform within $0.5\%$ of each other, ties are resolved without user intervention in this sequence: Inference Latency $\rightarrow$ Model Size $\rightarrow$ Simplicity Class $\rightarrow$ Secondary Metric (§3.2).
3.  **Descending Curves at Budget Limit:** Fits a linear slope over the last 5 checkpoints to project future gains; extends the budget only if the projected gain over doubled compute is $\ge 1\%$ (§2.2).
4.  **Small Evaluation Datasets:** Blocks promotions if eval count is $<200$ samples. Applies bootstrap resampling with a confidence interval and a "small-eval caveat" if sample count is $200-500$ (§3.1, §3.2).
5.  **Mid-Sweep Data Leakage:** Instantly suspends training if validation score is abnormally high compared to train ($>5\%$) or if train is near-perfect with mediocre validation at epoch 1 (§2.1). Triggers a 5-step leakage audit ([stopping-criteria.md](./operator/reference/stopping-criteria.md#L43-L51)).
6.  **Underperforming Sweeps:** If all models underperform baseline metrics after consuming $25\%$ of the budget, the sweep is aborted to prevent wasting compute (§2.1).
7.  **Profile Drift Between Intake and Load:** Stage 03 pre-flight recomputes the dataset profile from the loaded file. If the measured profile flips any Phase 1 gate (row-count band, imbalance band, temporal/group flag), the brief is corrected from the data and the design is re-run before any training code executes (§2.0).
8.  **Hard-Stop Recurrence (Reset Loops):** If the identical hard-stop signal fires twice consecutively on the same data source, the pipeline refuses a second recovery -- it suspends, presents both occurrences, and requires researcher action rather than burning budget in a restart loop (§2.1).
9.  **Stale or Partial Colab Telemetry:** In Colab mode the operator drives the notebook through a Playwright-controlled browser (one-time Google login, then headless control). Output scraped from the notebook is validated against the expected run ID and schema before any triage rule fires; mismatched reads and recycled runtimes trigger a re-run, never a triage (§2.0).
