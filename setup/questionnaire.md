# Onboarding Questionnaire -- ml-experiment-pilot

<!-- Agent instructions: Read this file when the user types "setup". Ask ALL questions
     in a single conversational pass. The user should be able to answer everything in one
     message. Collect answers. Replace placeholders across the specified files. After all
     replacements, verify no {{PLACEHOLDER}} patterns remain in the workspace. -->

---

### Q1: What URL does your monitoring dashboard run at?

The dashboard server (`dashboard/server.js`) streams live sweep data to this address. If you are running it locally with the default settings, you can skip this.

- Placeholder: `{{DASHBOARD_URL}}`
- Files: `stages/03-execution/references/setup-guide.md`
- Type: free text
- Default: `http://localhost:3000`

### Q2: What compute budget should the operator use when a researcher does not specify one?

This becomes the fallback default stated in the experiment plan when a researcher leaves the budget open. Enter a number and unit (e.g. "20 GPU-hours", "50 GPU-hours", "100 A100-hours").

- Placeholder: `{{DEFAULT_BUDGET}}`
- Files: `operator/reference/experiment-design-checklist.md`
- Type: free text
- Default: `20 GPU-hours`

---

## Per-Run Variables (not setup questions)

The following variables are collected conversationally at the start of each experiment run by Stage 01 -- Research Intake. They are NOT configured during setup because they change every run.

| Variable | Collected By |
|----------|-------------|
| Researcher name | Stage 01 intake conversation |
| Research goal | Stage 01 intake conversation |
| Target variable | Stage 01 intake conversation |
| Dataset name and profile | Stage 01 intake conversation |
| Compute budget (per run) | Stage 01 intake conversation |
| Researcher skill level | Stage 01 skill assessment |

---

## After Onboarding

Tell the researcher:

> "Setup complete. Your dashboard URL is set to `{{DASHBOARD_URL}}` and the default compute budget is `{{DEFAULT_BUDGET}}`.
>
> To start an experiment, go to **Stage 01 -- Research Intake**. The operator will walk you through describing your problem and dataset, assess your skill level, and produce a validated problem brief before anything runs."

After all replacements, scan the entire workspace for remaining `{{` patterns. If any remain, ask for the missing information before proceeding.
