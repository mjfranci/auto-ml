# Stage 01 -- Research Intake

Collect the researcher's problem description, assess skill level, conduct a literature review to identify the best model architecture, write a library template if one does not exist, and produce the problem brief.

## Inputs

| Source | File/Location | Section/Scope | Why |
|--------|--------------|---------------|-----|
| Researcher | Conversation | Problem description, dataset profile, compute budget, background | Raw material for the brief |
| Reference | `references/intake-questionnaire.md` | Full file | Intake question sequence and skill-level assessment guide |
| Setup | `../../setup/environment.md` | `execution_mode` field | Decide whether to ask for a local path/URL or a Google Drive location for the data source |
| Library | `../../library/README.md` | "How to Add New Model Templates" | Integration checklist for writing a new template |
| Library | `../../library/template_skeleton.py` | Full file | Base to copy when a new template is needed |
| Reference | `../../operator/reference/model-selection-rubric.md` | "Simplicity classes" section | Assign simplicity class when writing a new template |
| Tool | Web search | Literature review | Find the best-performing model architecture for the stated problem |

## Process

1. Assess researcher skill level (novice / intermediate / expert) using the intake questionnaire -- calibrate explanation depth for all subsequent stages
2. Collect required fields: research goal, target variable, data source (in Colab mode, the Google Drive location -- e.g. `My Drive > {folder name}`), dataset profile (row count, feature types, class balance, temporal flag, missing-data %), compute budget ceiling
3. Infer any missing profile fields from the researcher's descriptions; only escalate if the target variable itself is undefined
4. Conduct a literature review -- search for the best-performing model architecture for the stated problem type and dataset characteristics; identify the top candidate from recent research
5. Check whether a matching template exists in `../../library/`; if not, write `../../library/[model]-template.py` following the integration checklist in `library/README.md` before continuing
6. Write `output/problem-brief.md` including the literature-selected model as the primary candidate
7. Once the problem brief contains the required information, immediately hand off to Stage 02; do not wait for researcher approval or ask whether to continue.

## Outputs

| Artifact | Location | Format |
|----------|----------|--------|
| Problem brief | `output/problem-brief.md` | Structured doc: research goal, target variable, data source, dataset profile, compute budget, skill level, literature-selected primary model |
| Model template | `../../library/[model]-template.py` | Written only if no matching template exists -- PyTorch nn.Module wired to MLTracker, simplicity class set |
