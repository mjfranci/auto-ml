# Intake Questionnaire -- Research Intake

This file guides the Stage 01 intake conversation. Ask questions in conversational sequence, not as a form. Calibrate your explanation depth throughout the rest of the pipeline based on the skill level assessed here.

---

## Step 1 -- Assess Skill Level

Ask the researcher to describe their background. Listen for indicators:

| Indicator | Level |
|-----------|-------|
| Unfamiliar with train/val splits, loss functions, or hyperparameters | Novice |
| Knows model families, can read a learning curve, understands overfitting | Intermediate |
| Comfortable with leakage audits, metric selection trade-offs, ablations | Expert |

Record the assessed level in the problem brief under the field "skill_level". This governs:
- **Novice:** explain every decision in plain language; avoid jargon; call out overrides gently
- **Intermediate:** state decisions with one-line rationale; name the rule being applied
- **Expert:** terse technical output; cite rule numbers only; skip motivation unless asked

---

## Step 2 -- Required Fields

Collect all of the following. Infer what you can from context; only escalate if the target variable is undefined.

| Field | Question to Ask | Notes |
|-------|----------------|-------|
| Research goal | What is the scientific or clinical question you are trying to answer? | Free text |
| Target variable | What are you predicting? | Must be explicit -- this is the only hard escalation if absent |
| Data source | Where does the data live? Provide a file path, URL, or database connection string. **In Colab mode** (`execution_mode: colab` in `setup/environment.md`) ask instead for the Google Drive location -- e.g. `My Drive > {folder name}` -- since Colab cannot read local files. | Must be explicit -- escalate if absent; cannot proceed to Stage 03 without a loadable data source (a Google Drive location in Colab mode) |
| Row count | How many rows does your dataset have? | If unknown, ask for an order of magnitude |
| Feature count | How many features (columns) does the dataset have? | |
| Feature types | Are features mostly numeric, categorical, text, image, or a mix? | |
| Class balance | For classification: roughly how common is the positive class? | Skip for regression |
| Temporal flag | Is there a date/time column, or are rows ordered by time? | Forces time-based split if yes |
| Entity/group flag | Are there repeated IDs -- same patient, user, or site appearing in multiple rows? | Forces group k-fold if yes |
| Missing data | Are there columns with lots of missing values? Roughly what percentage? | |
| Compute budget | How many GPU-hours can you spend on this experiment? | Default to 20 if absent -- state the default |

---

## Step 3 -- Log Researcher Requests

Before closing the intake, ask:

> "Is there a specific model or metric you want me to prioritize? I'll include your request in the plan -- and I'll tell you clearly if the rules require me to do something different."

Log any requests. The experiment plan will mark each HONORED or OVERRIDDEN with a rule citation.

